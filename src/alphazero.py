"""
AlphaZero-style training for Extinction Chess.

Full implementation:
 - ResNet with policy + value heads (20 blocks, 256 filters, ~24M params)
 - MCTS with network policy priors
 - Self-play producing (state, policy_target, value_target) training data
 - Training on both heads: cross-entropy(policy) + MSE(value)

The C++ engine (_ext_chess) handles all game logic and board encoding.
"""

import math
import random
import time
import os
import json
from typing import List, Tuple, Optional
from multiprocessing import Process, Queue

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from extinction_chess import ExtinctionChess, Color, PieceType, Position
from state_encoder import StateEncoder


# ═════════════════════════════════════════════════════════════════════════════
# Move encoding  (Move ↔ policy index)
#
# 76 planes × 64 squares = 4864 total policy logits
#   Planes  0–55: Queen-type moves (8 directions × 7 distances)
#   Planes 56–63: Knight moves (8 offsets)
#   Planes 64–72: Underpromotions (knight/bishop/rook × 3 file deltas)
#   Planes 73–75: King promotions (3 file deltas)
#   Queen promotions are encoded as queen-type moves (no special plane).
# ═════════════════════════════════════════════════════════════════════════════

NUM_POLICY_PLANES = 76
POLICY_SIZE = NUM_POLICY_PLANES * 64   # 4864

# Direction table for queen-type moves
#   index → (dr, df)
QUEEN_DIRS = [
    ( 1,  0),  # N
    ( 1,  1),  # NE
    ( 0,  1),  # E
    (-1,  1),  # SE
    (-1,  0),  # S
    (-1, -1),  # SW
    ( 0, -1),  # W
    ( 1, -1),  # NW
]

# Knight offsets
KNIGHT_OFFSETS = [
    ( 2,  1), ( 2, -1), (-2,  1), (-2, -1),
    ( 1,  2), ( 1, -2), (-1,  2), (-1, -2),
]

# Build reverse lookup: (dr, df) → direction index
_DIR_LOOKUP = {}
for i, (dr, df) in enumerate(QUEEN_DIRS):
    for dist in range(1, 8):
        _DIR_LOOKUP[(dr * dist, df * dist)] = (i, dist)

_KNIGHT_LOOKUP = {}
for i, (dr, df) in enumerate(KNIGHT_OFFSETS):
    _KNIGHT_LOOKUP[(dr, df)] = i


def move_to_index(move) -> int:
    """Convert a Move to a policy index in [0, 4864)."""
    fr = move.from_pos
    to = move.to_pos
    from_sq = fr.rank * 8 + fr.file
    dr = to.rank - fr.rank
    df = to.file - fr.file

    # Underpromotions / king promotions
    promo = move.promotion
    if promo is not None and promo != PieceType.QUEEN:
        # File delta → direction index (0, 1, 2 for df = -1, 0, +1)
        df_idx = df + 1   # -1→0, 0→1, +1→2

        if promo == PieceType.KNIGHT:
            plane = 64 + df_idx
        elif promo == PieceType.BISHOP:
            plane = 64 + 3 + df_idx
        elif promo == PieceType.ROOK:
            plane = 64 + 6 + df_idx
        elif promo == PieceType.KING:
            plane = 73 + df_idx
        else:
            plane = 0  # fallback
        return plane * 64 + from_sq

    # Knight moves
    if (dr, df) in _KNIGHT_LOOKUP:
        plane = 56 + _KNIGHT_LOOKUP[(dr, df)]
        return plane * 64 + from_sq

    # Queen-type moves (includes queen promotions and castling king moves)
    if (dr, df) in _DIR_LOOKUP:
        direction, distance = _DIR_LOOKUP[(dr, df)]
        plane = direction * 7 + (distance - 1)
        return plane * 64 + from_sq

    # Fallback (shouldn't happen for valid moves)
    return 0


def index_to_move_info(index: int) -> Tuple[int, int, int, Optional[int]]:
    """Convert policy index back to (from_sq, to_sq, plane, promotion).
    Returns raw squares — caller maps to actual Move objects."""
    plane = index // 64
    from_sq = index % 64
    from_r, from_f = from_sq // 8, from_sq % 8

    promo = None

    if plane < 56:
        # Queen-type move
        direction = plane // 7
        distance = (plane % 7) + 1
        dr, df = QUEEN_DIRS[direction]
        to_r = from_r + dr * distance
        to_f = from_f + df * distance
    elif plane < 64:
        # Knight move
        ki = plane - 56
        dr, df = KNIGHT_OFFSETS[ki]
        to_r = from_r + dr
        to_f = from_f + df
    elif plane < 73:
        # Underpromotion
        sub = plane - 64
        piece_idx = sub // 3     # 0=knight, 1=bishop, 2=rook
        df_idx = sub % 3         # 0=left, 1=straight, 2=right
        df = df_idx - 1
        dr = 1 if from_r < 4 else -1  # white goes up, black goes down
        to_r = from_r + dr
        to_f = from_f + df
        promo = [PieceType.KNIGHT, PieceType.BISHOP, PieceType.ROOK][piece_idx]
    else:
        # King promotion
        df_idx = plane - 73
        df = df_idx - 1
        dr = 1 if from_r < 4 else -1
        to_r = from_r + dr
        to_f = from_f + df
        promo = PieceType.KING

    to_sq = to_r * 8 + to_f
    return from_sq, to_sq, plane, promo


# ═════════════════════════════════════════════════════════════════════════════
# Network
# ═════════════════════════════════════════════════════════════════════════════

class ResBlock(nn.Module):
    """Pre-activation residual block: BN → ReLU → Conv → BN → ReLU → Conv + skip."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class AlphaZeroNet(nn.Module):
    """
    AlphaZero-scale network for Extinction Chess.

    Architecture (matches the original paper):
      Input:  14 × 8 × 8
      Body:   Conv 3×3 → 20 residual blocks (256 filters)
      Policy: Conv 1×1 (2 filters) → BN → ReLU → FC → 4864 logits
      Value:  Conv 1×1 (1 filter)  → BN → ReLU → FC → 256 → ReLU → FC → 1 → tanh
    """

    def __init__(self, in_channels: int = 14, num_filters: int = 256,
                 num_blocks: int = 20, policy_size: int = POLICY_SIZE):
        super().__init__()
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.num_blocks = num_blocks
        self.policy_size = policy_size

        # Input convolution
        self.input_conv = nn.Conv2d(in_channels, num_filters, 3, padding=1, bias=False)
        self.input_bn   = nn.BatchNorm2d(num_filters)

        # Residual tower
        self.res_blocks = nn.Sequential(
            *[ResBlock(num_filters) for _ in range(num_blocks)]
        )

        # Policy head (AlphaZero paper: conv 1×1 → 2 filters → BN → ReLU → FC)
        self.policy_conv = nn.Conv2d(num_filters, 2, 1, bias=False)
        self.policy_bn   = nn.BatchNorm2d(2)
        self.policy_fc   = nn.Linear(2 * 8 * 8, policy_size)

        # Value head (conv 1×1 → 1 filter → BN → ReLU → FC 256 → ReLU → FC 1 → tanh)
        self.value_conv = nn.Conv2d(num_filters, 1, 1, bias=False)
        self.value_bn   = nn.BatchNorm2d(1)
        self.value_fc1  = nn.Linear(8 * 8, 256)
        self.value_fc2  = nn.Linear(256, 1)

    def forward(self, x):
        """Returns (policy_logits, value) where policy is raw logits (batch, 4864)."""
        # Body
        out = F.relu(self.input_bn(self.input_conv(x)))
        out = self.res_blocks(out)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)).squeeze(-1)

        return p, v

    def save_checkpoint(self, path, **metadata):
        metadata["model_type"] = "alphazero"
        metadata["in_channels"] = self.in_channels
        metadata["num_filters"] = self.num_filters
        metadata["num_blocks"] = self.num_blocks
        metadata["policy_size"] = self.policy_size
        torch.save({"state_dict": self.state_dict(), "metadata": metadata}, path)

    @classmethod
    def load_checkpoint(cls, path) -> Tuple["AlphaZeroNet", dict]:
        data = torch.load(path, weights_only=False, map_location="cpu")
        meta = data.get("metadata", {})
        net = cls(
            in_channels=meta.get("in_channels", 14),
            num_filters=meta.get("num_filters", 256),
            num_blocks=meta.get("num_blocks", 20),
            policy_size=meta.get("policy_size", POLICY_SIZE),
        )
        net.load_state_dict(data["state_dict"])
        return net, meta


# ═════════════════════════════════════════════════════════════════════════════
# Evaluator wrapper (for compatibility with existing code)
# ═════════════════════════════════════════════════════════════════════════════

class AlphaZeroEvaluator:
    """Wraps AlphaZeroNet for inference.  Returns value only (for backward compat)."""

    def __init__(self, model: AlphaZeroNet, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.encoder = StateEncoder()
        self.model.eval()

    def evaluate(self, game: ExtinctionChess) -> float:
        if game.game_over:
            if game.winner == Color.WHITE:  return 1.0
            elif game.winner == Color.BLACK: return -1.0
            return 0.0
        board_tensor = self.encoder.encode_board(game)
        with torch.no_grad():
            t = torch.tensor(board_tensor, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
            _, v = self.model(t)
            return v.item()

    def evaluate_with_policy(self, game: ExtinctionChess):
        """Return (policy_logits, value) for a single position."""
        board_tensor = self.encoder.encode_board(game)
        with torch.no_grad():
            t = torch.tensor(board_tensor, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
            p, v = self.model(t)
            return p.squeeze(0).cpu().numpy(), v.item()

    def batch_evaluate(self, games: List[ExtinctionChess]) -> np.ndarray:
        boards = np.stack([self.encoder.encode_board(g) for g in games])
        with torch.no_grad():
            t = torch.tensor(boards, dtype=torch.float32, device=self.device)
            _, v = self.model(t)
            return v.cpu().numpy()

    def batch_evaluate_with_policy(self, games: List[ExtinctionChess]):
        """Return (policy_logits, values) for a batch of positions.
        policy_logits: numpy (N, 4864), values: numpy (N,)"""
        boards = np.stack([self.encoder.encode_board(g) for g in games])
        with torch.no_grad():
            t = torch.tensor(boards, dtype=torch.float32, device=self.device)
            p, v = self.model(t)
            return p.cpu().numpy(), v.cpu().numpy()


# ═════════════════════════════════════════════════════════════════════════════
# MCTS with policy priors
# ═════════════════════════════════════════════════════════════════════════════

class MCTSNode:
    __slots__ = ('game', 'parent', 'move', 'prior',
                 'children', 'visit_count', 'value_sum', 'is_expanded',
                 'virtual_loss')

    def __init__(self, game, parent=None, move=None, prior=1.0):
        self.game = game
        self.parent = parent
        self.move = move
        self.prior = prior
        self.children: List[MCTSNode] = []
        self.visit_count = 0
        self.value_sum = 0.0      # from WHITE's perspective
        self.is_expanded = False
        self.virtual_loss = 0

    def q_from_parent(self):
        vc = self.visit_count + self.virtual_loss
        if vc == 0:
            return 0.0
        wq = (self.value_sum - self.virtual_loss) / vc
        return wq if self.parent.game.current_player == Color.WHITE else -wq

    def ucb(self, c_puct):
        vc = self.visit_count + self.virtual_loss
        parent_vc = self.parent.visit_count + self.parent.virtual_loss
        return (self.q_from_parent()
                + c_puct * self.prior
                * math.sqrt(parent_vc) / (1 + vc))

    def best_child(self, c_puct):
        return max(self.children, key=lambda c: c.ucb(c_puct))


def _copy_game(game):
    """Make a copy of a Game for search."""
    gc = ExtinctionChess()
    gc.board = game.board.copy()
    gc.current_player = game.current_player
    gc.game_over = game.game_over
    return gc


def _eval_white(evaluator: AlphaZeroEvaluator, game):
    """Evaluate from White's perspective."""
    if game.game_over:
        if game.winner == Color.WHITE:  return 1.0
        elif game.winner == Color.BLACK: return -1.0
        return 0.0
    v = evaluator.evaluate(game)
    return v if game.current_player == Color.WHITE else -v


def _expand_node(node, policy_logits):
    """Expand a node using policy logits. Returns value from white's perspective."""
    child_legal = node.game.get_legal_moves()
    if not child_legal:
        node.is_expanded = True
        return _terminal_value(node.game)

    m_indices = [move_to_index(m) for m in child_legal]
    m_logits = np.array([policy_logits[i] for i in m_indices])
    m_logits -= m_logits.max()
    child_probs = np.exp(m_logits)
    child_probs /= child_probs.sum() + 1e-8

    for cm, cp in zip(child_legal, child_probs):
        gc = _copy_game(node.game)
        if gc.make_move(cm):
            node.children.append(MCTSNode(gc, parent=node, move=cm, prior=cp))
    node.is_expanded = True
    return None  # value comes from the evaluator


def _terminal_value(game):
    """Get value from white's perspective for a terminal game."""
    if game.game_over:
        if game.winner == Color.WHITE: return 1.0
        if game.winner == Color.BLACK: return -1.0
        return 0.0
    return 0.0


def _add_virtual_loss(node):
    """Add virtual loss along path from node to root."""
    n = node
    while n is not None:
        n.virtual_loss += 1
        n = n.parent


def _remove_virtual_loss(node):
    """Remove virtual loss along path from node to root."""
    n = node
    while n is not None:
        n.virtual_loss -= 1
        n = n.parent


def _backpropagate(node, white_value):
    """Backpropagate value from node to root."""
    while node is not None:
        node.visit_count += 1
        node.value_sum += white_value
        node = node.parent


BATCH_SIZE_MCTS = 8  # Number of leaves to collect before batched eval

# Try to import C++ MCTS and batched self-play
try:
    from _ext_chess import MCTS as CppMCTS, move_to_index as cpp_move_to_index
    HAS_CPP_MCTS = True
except ImportError:
    HAS_CPP_MCTS = False

try:
    from _ext_chess import SelfPlayManager as CppSelfPlayManager
    HAS_CPP_SELFPLAY = True
except ImportError:
    HAS_CPP_SELFPLAY = False


def mcts_search_cpp(game, evaluator: AlphaZeroEvaluator,
                    num_simulations: int = 800, c_puct: float = 2.5,
                    dirichlet_alpha: float = 0.3, noise_weight: float = 0.25,
                    tactical_shortcuts: bool = True,
                    batch_size: int = 8):
    """
    MCTS using C++ tree search with Python neural network evaluation.
    Returns list of (move, visit_count) pairs.
    """
    legal = game.get_legal_moves()
    if not legal:
        return []

    # Build index→move lookup for mapping results back
    index_to_move = {}
    for m in legal:
        idx = move_to_index(m)
        index_to_move[idx] = m

    # Create C++ MCTS object
    mcts = CppMCTS(game, num_simulations, c_puct,
                   dirichlet_alpha, noise_weight,
                   tactical_shortcuts, batch_size)

    # First: expand root with a single NN call
    policy_logits, _ = evaluator.evaluate_with_policy(game)
    mcts.expand_root(policy_logits)

    # Main search loop (skipped if tactical shortcut fired)
    while not mcts.is_done():
        # Phase 1: C++ selects leaves and encodes their boards
        boards = mcts.select_leaves(batch_size)

        if len(boards) == 0:
            continue

        # Phase 2: batch NN evaluation on GPU
        with torch.no_grad():
            t = torch.tensor(np.asarray(boards), dtype=torch.float32,
                             device=evaluator.device)
            policies, values = evaluator.model(t)
            policies = policies.cpu().numpy()
            values = values.cpu().numpy()

        # Phase 3: C++ expands nodes and backpropagates
        mcts.process_results(policies, values)

    # Map results back to Move objects using policy indices
    results = mcts.get_results()  # list of (policy_index, visit_count)
    out = []
    for idx, visits in results:
        if idx in index_to_move:
            out.append((index_to_move[idx], visits))
    return out


def mcts_search(game, evaluator: AlphaZeroEvaluator,
                num_simulations: int = 800, c_puct: float = 2.5,
                dirichlet_alpha: float = 0.3, noise_weight: float = 0.25,
                tactical_shortcuts: bool = True):
    """
    AlphaZero-style MCTS with batched neural network evaluation.
    Collects multiple leaves via virtual loss, evaluates in one GPU call.
    Returns (move_visits, root_value) where move_visits is list of (move, visit_count)
    and root_value is the value head's evaluation of the root position.
    """
    root = MCTSNode(game)

    # Expand root with policy priors (single eval)
    legal = game.get_legal_moves()
    if not legal:
        return [], 0.0

    # Tactical shortcut: force instant wins (but allow blunders)
    current = game.current_player
    if tactical_shortcuts:
        for m in legal:
            gc = _copy_game(game)
            if not gc.make_move(m):
                continue

            # 1-ply: this move immediately wins
            if gc.game_over and gc.winner == current:
                return [(m, num_simulations)] + [(mm, 0) for mm in legal if mm != m], 1.0

    policy_logits, root_value = evaluator.evaluate_with_policy(game)

    move_indices = [move_to_index(m) for m in legal]
    move_logits = np.array([policy_logits[i] for i in move_indices])
    move_logits -= move_logits.max()
    probs = np.exp(move_logits)
    probs /= probs.sum() + 1e-8

    if dirichlet_alpha > 0 and noise_weight > 0:
        noise = np.random.dirichlet([dirichlet_alpha] * len(legal))
        probs = (1 - noise_weight) * probs + noise_weight * noise

    for m, p in zip(legal, probs):
        gc = _copy_game(game)
        if gc.make_move(m):
            root.children.append(MCTSNode(gc, parent=root, move=m, prior=p))
    root.is_expanded = True

    if not root.children:
        return [], root_value

    # Run simulations in batches
    sims_done = 0
    while sims_done < num_simulations:
        batch_size = min(BATCH_SIZE_MCTS, num_simulations - sims_done)
        leaves = []       # nodes needing NN eval
        terminal = []     # (node, value) for terminal/already-expanded nodes

        # Select batch_size leaves using virtual loss for diversity
        for _ in range(batch_size):
            node = root
            while node.is_expanded and node.children:
                node = node.best_child(c_puct)

            if node.game.game_over:
                tv = _terminal_value(node.game)
                terminal.append((node, tv))
                _add_virtual_loss(node)
            elif node.is_expanded:
                tv = _terminal_value(node.game)
                terminal.append((node, tv))
                _add_virtual_loss(node)
            else:
                leaves.append(node)
                _add_virtual_loss(node)

        # Batch evaluate all non-terminal leaves
        if leaves:
            games_batch = [n.game for n in leaves]
            all_policies, all_values = evaluator.batch_evaluate_with_policy(games_batch)

            for i, node in enumerate(leaves):
                _expand_node(node, all_policies[i])
                v = all_values[i]
                white_value = v if node.game.current_player == Color.WHITE else -v
                _remove_virtual_loss(node)
                _backpropagate(node, white_value)
                sims_done += 1

        # Handle terminal nodes
        for node, white_value in terminal:
            _remove_virtual_loss(node)
            _backpropagate(node, white_value)
            sims_done += 1

    return [(ch.move, ch.visit_count) for ch in root.children], root_value


# ═════════════════════════════════════════════════════════════════════════════
# Self-play
# ═════════════════════════════════════════════════════════════════════════════

def self_play_game(evaluator: AlphaZeroEvaluator,
                   num_simulations: int = 800,
                   c_puct: float = 2.5,
                   dirichlet_alpha: float = 0.3,
                   noise_weight: float = 0.25,
                   temp_threshold: int = 30,
                   max_moves: int = 200):
    """
    Play one self-play game using MCTS.

    Returns:
        boards:   list of encoded board tensors (14×8×8 numpy arrays)
        policies: list of policy target vectors (length 4864)
        outcome:  1.0 (white wins), -1.0 (black wins), 0.0 (draw)
    """
    game = ExtinctionChess()
    encoder = StateEncoder()

    boards = []
    policies = []
    players = []    # track which side is to move for each position

    move_count = 0
    while not game.game_over and move_count < max_moves:
        # Encode board
        boards.append(encoder.encode_board(game))
        players.append(game.current_player)

        # MCTS search (prefer C++ if available)
        if HAS_CPP_MCTS:
            move_visits = mcts_search_cpp(game, evaluator, num_simulations,
                                          c_puct, dirichlet_alpha, noise_weight)
        else:
            move_visits, _ = mcts_search(game, evaluator, num_simulations,
                                          c_puct, dirichlet_alpha, noise_weight)
        if not move_visits:
            break

        moves, counts = zip(*move_visits)
        counts = np.array(counts, dtype=np.float64)

        # Build policy target: normalized visit counts mapped to policy indices
        policy_target = np.zeros(POLICY_SIZE, dtype=np.float32)
        total_visits = counts.sum()
        for m, c in zip(moves, counts):
            idx = move_to_index(m)
            policy_target[idx] = c / total_visits
        policies.append(policy_target)

        # Select move
        if move_count < temp_threshold:
            # Temperature 1: sample proportional to visit counts
            probs = counts / total_visits
            idx = np.random.choice(len(moves), p=probs)
        else:
            # Temperature → 0: pick most-visited
            idx = int(np.argmax(counts))

        game.make_move(moves[idx])
        move_count += 1

    # Game outcome
    if game.winner == Color.WHITE:
        outcome = 1.0
    elif game.winner == Color.BLACK:
        outcome = -1.0
    else:
        outcome = 0.0

    return boards, policies, players, outcome


# ═════════════════════════════════════════════════════════════════════════════
# Parallel self-play
# ═════════════════════════════════════════════════════════════════════════════

def _self_play_worker(model_path: str, device: str, num_games: int,
                      num_simulations: int, temp_threshold: int,
                      result_queue: Queue):
    """Worker process: loads model, plays num_games, puts results on queue."""
    model, _ = AlphaZeroNet.load_checkpoint(model_path)
    model = model.to(device)
    model.eval()
    evaluator = AlphaZeroEvaluator(model, device=device)

    results = []
    for _ in range(num_games):
        boards, policies, players, outcome = self_play_game(
            evaluator, num_simulations=num_simulations,
            temp_threshold=temp_threshold,
        )
        results.append((boards, policies, players, outcome))
    result_queue.put(results)


def parallel_self_play(model_path: str, games_per_iteration: int,
                       num_simulations: int, num_workers: int = 4,
                       temp_threshold: int = 30, device: str = "cuda"):
    """Run self-play across multiple processes, each with its own model copy.

    If multiple GPUs are available, workers are distributed across them.
    """
    num_gpus = torch.cuda.device_count() if device.startswith("cuda") else 0

    # Split games across workers
    base = games_per_iteration // num_workers
    remainder = games_per_iteration % num_workers
    games_per_worker = [base + (1 if i < remainder else 0) for i in range(num_workers)]

    result_queue = Queue()
    processes = []
    for i in range(num_workers):
        if games_per_worker[i] == 0:
            continue
        # Distribute workers across GPUs
        if num_gpus > 1:
            worker_device = f"cuda:{i % num_gpus}"
        else:
            worker_device = device
        p = Process(target=_self_play_worker,
                    args=(model_path, worker_device, games_per_worker[i],
                          num_simulations, temp_threshold, result_queue))
        p.start()
        processes.append(p)

    # Collect results
    all_results = []
    for _ in processes:
        all_results.extend(result_queue.get())

    for p in processes:
        p.join()

    return all_results


# ═════════════════════════════════════════════════════════════════════════════
# Batched self-play (C++ manages all games, Python only does GPU inference)
# ═════════════════════════════════════════════════════════════════════════════

def batched_self_play(model, device, games_per_iteration: int,
                      num_simulations: int = 800, c_puct: float = 2.5,
                      dirichlet_alpha: float = 0.3, noise_weight: float = 0.25,
                      temp_threshold: int = 30, max_moves: int = 200,
                      num_parallel: int = 50, max_batch: int = 512,
                      mcts_batch_size: int = 8):
    """
    Batched self-play using C++ SelfPlayManager.

    All game logic and MCTS run in C++. Python only handles batched GPU
    inference. This maximizes GPU utilization by collecting leaf positions
    from many simultaneous games into one large batch.

    Returns list of (boards, policies, players, outcome) tuples.
    """
    manager = CppSelfPlayManager(
        num_parallel_games=num_parallel,
        total_games=games_per_iteration,
        num_simulations=num_simulations,
        c_puct=c_puct,
        dirichlet_alpha=dirichlet_alpha,
        noise_weight=noise_weight,
        tactical_shortcuts=True,
        temp_threshold=temp_threshold,
        max_moves=max_moves,
        mcts_batch_size=mcts_batch_size,
    )

    total_evals = 0
    while not manager.is_done():
        # C++ collects positions from all active games
        boards = manager.collect_leaves(max_batch)

        if len(boards) == 0:
            continue

        # Batch GPU inference
        with torch.no_grad():
            t = torch.tensor(np.asarray(boards), dtype=torch.float32, device=device)
            policies, values = model(t)
            policies = policies.cpu().numpy()
            values = values.cpu().numpy()

        # Feed results back to C++
        manager.process_results(policies, values)
        total_evals += len(boards)

    print(f"         batched self-play: {total_evals} NN evals, "
          f"{manager.games_completed()} games", flush=True)

    # Convert C++ results to the same format as self_play_game
    raw_records = manager.get_results()
    results = []
    for rec in raw_records:
        boards = [b.reshape(14, 8, 8) for b in rec["boards"]]
        policies = list(rec["policies"])
        players = list(rec["players"])
        outcome = rec["outcome"]
        results.append((boards, policies, players, outcome))

    return results


# ═════════════════════════════════════════════════════════════════════════════
# Testing utility
# ═════════════════════════════════════════════════════════════════════════════

def test_vs_random(evaluator: AlphaZeroEvaluator, num_games: int = 100,
                   num_simulations: int = 100) -> float:
    """Win rate using MCTS against a random opponent."""
    wins, losses, draws = 0, 0, 0
    for i in range(num_games):
        game = ExtinctionChess()
        model_is_white = (i % 2 == 0)
        moves = 0
        while not game.game_over and moves < 200:
            legal = game.get_legal_moves()
            if not legal:
                break
            is_model = (game.current_player == Color.WHITE) == model_is_white
            if is_model:
                if HAS_CPP_MCTS:
                    mv = mcts_search_cpp(game, evaluator, num_simulations=num_simulations,
                                         dirichlet_alpha=0, noise_weight=0,
                                         tactical_shortcuts=False)
                else:
                    mv, _ = mcts_search(game, evaluator, num_simulations=num_simulations,
                                        dirichlet_alpha=0, noise_weight=0,
                                        tactical_shortcuts=False)
                if mv:
                    best = max(mv, key=lambda x: x[1])
                    move = best[0]
                else:
                    move = random.choice(legal)
            else:
                move = random.choice(legal)
            game.make_move(move)
            moves += 1
        if game.winner:
            if (game.winner == Color.WHITE) == model_is_white:
                wins += 1
            else:
                losses += 1
        else:
            draws += 1
    print(f"         W={wins} L={losses} D={draws}")
    return (wins + 0.5 * draws) / num_games


# ═════════════════════════════════════════════════════════════════════════════
# Instant-win position generator (supplementary training data)
# ═════════════════════════════════════════════════════════════════════════════

def _copy_game(game):
    """Copy a C++ game object (can't use deepcopy)."""
    gc = ExtinctionChess()
    gc.board = game.board.copy()
    gc.current_player = game.current_player
    gc.game_over = game.game_over
    gc.winner = game.winner
    return gc


def generate_instant_win_positions(num_positions: int, max_random_moves: int = 200):
    """Generate positions where the current player has an instant win.

    Makes random moves until a position with a winning capture appears.
    Returns (boards, policies, values) in the same format as self-play data.
    """
    boards = []
    policies = []
    values = []

    while len(boards) < num_positions:
        game = ExtinctionChess()

        for _ in range(max_random_moves):
            if game.game_over:
                break

            legal_moves = game.get_legal_moves()
            if not legal_moves:
                break

            # Check if current player has any instant wins
            winning_moves = []
            for m in legal_moves:
                gc = _copy_game(game)
                if gc.make_move(m) and gc.game_over and gc.winner == game.current_player:
                    winning_moves.append(m)

            if winning_moves:
                # Found a position with instant win(s) — record it
                board = np.asarray(game.encode_board(), dtype=np.float32)

                # Policy: uniform over all winning moves
                policy = np.zeros(POLICY_SIZE, dtype=np.float32)
                for wm in winning_moves:
                    policy[move_to_index(wm)] = 1.0
                policy /= policy.sum()

                # Value: +1 from WHITE's perspective
                value = 1.0 if game.current_player == Color.WHITE else -1.0

                boards.append(board)
                policies.append(policy)
                values.append(value)
                break  # Start a new game

            # No instant win — make a random move
            move = random.choice(legal_moves)
            game.make_move(move)

    return boards[:num_positions], policies[:num_positions], values[:num_positions]


# ═════════════════════════════════════════════════════════════════════════════
# Training loop
# ═════════════════════════════════════════════════════════════════════════════

def train(
    iterations: int = 100,
    games_per_iteration: int = 100,
    num_simulations: int = 800,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    num_filters: int = 256,
    num_blocks: int = 20,
    models_dir: str = "models",
    resume: bool = True,
    eval_simulations: int = 100,
    num_workers: int = 1,
    instant_win_positions: int = 0,
    max_wall_time: float = 0,
    num_epochs: int = 5,
):
    job_start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"AlphaZero config: {num_blocks} blocks, {num_filters} filters, "
          f"{num_simulations} MCTS sims/move")

    os.makedirs(models_dir, exist_ok=True)
    checkpoint_path = os.path.join(models_dir, "az_latest.pt")

    # Load or create model
    if resume and os.path.exists(checkpoint_path):
        model, meta = AlphaZeroNet.load_checkpoint(checkpoint_path)
        start_iter = meta.get("iteration", 0)
        print(f"Resumed from iteration {start_iter}")
    else:
        model = AlphaZeroNet(in_channels=14, num_filters=num_filters,
                             num_blocks=num_blocks)
        start_iter = 0
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Fresh AlphaZeroNet: {total_params:,} parameters")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
    evaluator = AlphaZeroEvaluator(model, device=str(device))

    best_win_rate = 0.0
    training_log = []

    for iteration in range(start_iter, start_iter + iterations):
        t0 = time.time()
        iter_num = iteration + 1

        # ── Wall-time check: exit cleanly before starting expensive self-play ──
        if max_wall_time > 0 and iter_num > start_iter + 1:
            elapsed = t0 - job_start_time
            remaining = max_wall_time - elapsed
            iters_done = iter_num - start_iter - 1
            avg_gen_time = (t0 - job_start_time) / iters_done
            needed = avg_gen_time + 1000
            if remaining < needed:
                print(f"         Wall time check: {remaining:.0f}s remaining, "
                      f"need ~{needed:.0f}s (avg {avg_gen_time:.0f}s + 1000s buffer), "
                      f"stopping before iter {iter_num}")
                break

        model.eval()

        # ── Self-play ───────────────────────────────────────────────────────
        all_boards = []
        all_policies = []
        all_values = []
        # Pre-terminal positions: last position of each decisive game
        terminal_boards = []
        terminal_policies = []
        terminal_values = []
        wins_w, wins_b, draws = 0, 0, 0

        if HAS_CPP_SELFPLAY:
            # Batched self-play: C++ manages all games, Python does GPU inference
            game_results = batched_self_play(
                model, device, games_per_iteration,
                num_simulations=num_simulations,
                temp_threshold=30,
                num_parallel=min(50, games_per_iteration),
                max_batch=512,
            )
            for boards, policies, players, outcome in game_results:
                for b, pi, player in zip(boards, policies, players):
                    value = outcome if player == 0 else -outcome
                    all_boards.append(b)
                    all_policies.append(pi)
                    all_values.append(value)
                if outcome != 0 and len(boards) > 0:
                    terminal_boards.append(boards[-1])
                    terminal_policies.append(policies[-1])
                    tv = outcome if players[-1] == 0 else -outcome
                    terminal_values.append(tv)
                if outcome > 0: wins_w += 1
                elif outcome < 0: wins_b += 1
                else: draws += 1
        elif num_workers > 1:
            # Save current model for workers to load
            tmp_path = os.path.join(models_dir, "az_tmp_selfplay.pt")
            model.save_checkpoint(tmp_path, iteration=iter_num)
            game_results = parallel_self_play(
                tmp_path, games_per_iteration,
                num_simulations=num_simulations,
                num_workers=num_workers,
                device=str(device),
            )
            for boards, policies, players, outcome in game_results:
                for b, pi, player in zip(boards, policies, players):
                    value = outcome if player == Color.WHITE else -outcome
                    all_boards.append(b)
                    all_policies.append(pi)
                    all_values.append(value)
                if outcome != 0 and len(boards) > 0:
                    terminal_boards.append(boards[-1])
                    terminal_policies.append(policies[-1])
                    tv = outcome if players[-1] == Color.WHITE else -outcome
                    terminal_values.append(tv)
                if outcome > 0: wins_w += 1
                elif outcome < 0: wins_b += 1
                else: draws += 1
        else:
            for g in range(games_per_iteration):
                boards, policies, players, outcome = self_play_game(
                    evaluator,
                    num_simulations=num_simulations,
                    temp_threshold=30,
                )
                for b, pi, player in zip(boards, policies, players):
                    value = outcome if player == Color.WHITE else -outcome
                    all_boards.append(b)
                    all_policies.append(pi)
                    all_values.append(value)
                if outcome != 0 and len(boards) > 0:
                    terminal_boards.append(boards[-1])
                    terminal_policies.append(policies[-1])
                    tv = outcome if players[-1] == Color.WHITE else -outcome
                    terminal_values.append(tv)

                if outcome > 0: wins_w += 1
                elif outcome < 0: wins_b += 1
                else: draws += 1

        gen_time = time.time() - t0
        elapsed = time.time() - job_start_time
        eh, em, es = int(elapsed // 3600), int(elapsed % 3600 // 60), int(elapsed % 60)
        print(f"[iter {iter_num}] W={wins_w} B={wins_b} D={draws} "
              f"| {len(all_boards)} positions | gen={gen_time:.1f}s "
              f"(total={eh}:{em:02d}:{es:02d})")

        # ── Supplementary instant-win positions (iters 270-280) ────────────
        if instant_win_positions > 0 and 270 <= iter_num <= 280:
            iw_boards, iw_policies, iw_values = generate_instant_win_positions(
                instant_win_positions
            )
            all_boards.extend(iw_boards)
            all_policies.extend(iw_policies)
            all_values.extend(iw_values)
            print(f"         +{len(iw_boards)} instant-win positions "
                  f"({len(all_boards)} total)")

        # ── Training ────────────────────────────────────────────────────────
        t1 = time.time()
        model.train()

        X = torch.tensor(np.array(all_boards), dtype=torch.float32, device=device)
        pi_target = torch.tensor(np.array(all_policies), dtype=torch.float32, device=device)
        v_target = torch.tensor(np.array(all_values), dtype=torch.float32, device=device)

        dataset = torch.utils.data.TensorDataset(X, pi_target, v_target)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        total_loss, total_ploss, total_vloss, n_batches = 0, 0, 0, 0
        for _epoch in range(num_epochs):
            for bx, bpi, bv in loader:
                optimizer.zero_grad()
                pred_p, pred_v = model(bx)

                # Policy loss: cross-entropy with MCTS visit distribution
                # -sum(pi * log(softmax(pred_p)))
                log_probs = F.log_softmax(pred_p, dim=1)
                policy_loss = -torch.sum(bpi * log_probs, dim=1).mean()

                # Value loss: MSE
                value_loss = F.mse_loss(pred_v, bv)

                loss = policy_loss + value_loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_ploss += policy_loss.item()
                total_vloss += value_loss.item()
                n_batches += 1

        train_time = time.time() - t1
        avg_loss = total_loss / max(n_batches, 1)
        avg_pl = total_ploss / max(n_batches, 1)
        avg_vl = total_vloss / max(n_batches, 1)
        print(f"         loss={avg_loss:.4f} (policy={avg_pl:.4f} value={avg_vl:.4f}) "
              f"| train={train_time:.1f}s")

        # ── Terminal position drilling ──────────────────────────────────
        if terminal_boards:
            t2 = time.time()
            tX = torch.tensor(np.array(terminal_boards),
                              dtype=torch.float32, device=device)
            tpi = torch.tensor(np.array(terminal_policies),
                               dtype=torch.float32, device=device)
            tv = torch.tensor(np.array(terminal_values),
                              dtype=torch.float32, device=device)

            t_dataset = torch.utils.data.TensorDataset(tX, tpi, tv)
            t_loader = torch.utils.data.DataLoader(
                t_dataset, batch_size=batch_size, shuffle=True)

            t_loss, t_ploss, t_vloss, t_nb = 0, 0, 0, 0
            for _epoch in range(num_epochs):
                for bx, bpi, bv in t_loader:
                    optimizer.zero_grad()
                    pred_p, pred_v = model(bx)
                    log_probs = F.log_softmax(pred_p, dim=1)
                    policy_loss = -torch.sum(bpi * log_probs, dim=1).mean()
                    value_loss = F.mse_loss(pred_v, bv)
                    loss = policy_loss + value_loss
                    loss.backward()
                    optimizer.step()
                    t_loss += loss.item()
                    t_ploss += policy_loss.item()
                    t_vloss += value_loss.item()
                    t_nb += 1

            t_time = time.time() - t2
            t_avg = t_loss / max(t_nb, 1)
            t_avgp = t_ploss / max(t_nb, 1)
            t_avgv = t_vloss / max(t_nb, 1)
            print(f"         terminal drilling: {len(terminal_boards)} positions, "
                  f"loss={t_avg:.4f} (p={t_avgp:.4f} v={t_avgv:.4f}) "
                  f"| {t_time:.1f}s")

        # Save checkpoint
        model.save_checkpoint(checkpoint_path, iteration=iter_num)

        training_log.append({
            "iteration": iter_num,
            "loss": avg_loss,
            "policy_loss": avg_pl,
            "value_loss": avg_vl,
            "wins_white": wins_w,
            "wins_black": wins_b,
            "draws": draws,
            "gen_time": gen_time,
            "train_time": train_time,
        })

        # ── Evaluate every 10 iterations ────────────────────────────────────
        if iter_num % 10 == 0:
            model.eval()
            eval_evaluator = AlphaZeroEvaluator(model, device=str(device))
            wr = test_vs_random(eval_evaluator, num_games=50,
                                num_simulations=eval_simulations)
            print(f"         win rate vs random: {wr:.1%}")

            versioned = os.path.join(models_dir,
                f"az_iter_{iter_num}_{int(wr*100)}pct.pt")
            model.save_checkpoint(versioned, iteration=iter_num, win_rate=wr)

            if wr > best_win_rate:
                best_win_rate = wr
                model.save_checkpoint(
                    os.path.join(models_dir, "az_best.pt"),
                    iteration=iter_num, win_rate=wr)
                print(f"         ★ new best: {wr:.1%}")

    # Final save
    model.eval()
    final_evaluator = AlphaZeroEvaluator(model, device=str(device))
    final_wr = test_vs_random(final_evaluator, num_games=100,
                              num_simulations=eval_simulations)
    print(f"\nFinal win rate vs random: {final_wr:.1%}")

    model.save_checkpoint(
        os.path.join(models_dir, "az_final.pt"),
        iteration=start_iter + iterations, win_rate=final_wr)

    log_path = os.path.join(models_dir, "az_training_log.json")
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)

    print(f"Done — {start_iter + iterations} iterations, best={best_win_rate:.1%}")
