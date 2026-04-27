"""
PyTorch neural network for Extinction Chess position evaluation.
Includes both the original MLP (ChessNet) and the CNN (ChessCNN).
"""

import math
import multiprocessing as mp
import torch
import torch.nn as nn
import numpy as np
import json
import random
from typing import List, Optional, Tuple

from extinction_chess import ExtinctionChess, Color, PieceType, Position
from state_encoder import StateEncoder


# ── Original MLP (39 features) ───────────────────────────────────────────────

class ChessNet(nn.Module):
    """PyTorch feedforward network for position evaluation."""

    def __init__(self, input_size: int = 39, hidden_sizes: Optional[List[int]] = None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64]

        layers: list[nn.Module] = []
        prev = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def save_checkpoint(self, path: str, **metadata):
        torch.save({"state_dict": self.state_dict(), "metadata": metadata}, path)

    @classmethod
    def load_checkpoint(
        cls, path: str, input_size: int = 39, hidden_sizes: Optional[List[int]] = None
    ) -> Tuple["ChessNet", dict]:
        data = torch.load(path, weights_only=False, map_location="cpu")
        net = cls(input_size, hidden_sizes)
        net.load_state_dict(data["state_dict"])
        return net, data.get("metadata", {})

    @classmethod
    def from_numpy_model(cls, path: str) -> "ChessNet":
        """Load weights saved by SimpleNeuralNetwork.save() (JSON format)."""
        with open(path) as f:
            data = json.load(f)
        cfg = data["config"]
        net = cls(cfg["input_size"], cfg["hidden_sizes"])

        sd = net.state_dict()
        layer_idx = 0
        for w, b in zip(data["weights"], data["biases"]):
            w_t = torch.tensor(np.array(w), dtype=torch.float32).T
            b_t = torch.tensor(np.array(b), dtype=torch.float32).squeeze()
            sd[f"net.{layer_idx}.weight"] = w_t
            sd[f"net.{layer_idx}.bias"] = b_t
            layer_idx += 2
        net.load_state_dict(sd)
        return net


# ── CNN with Residual Blocks (14×8×8 board tensor) ───────────────────────────

class ResidualBlock(nn.Module):
    """Residual block: conv → BN → ReLU → conv → BN + skip → ReLU"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = torch.relu(out + residual)
        return out


class ChessCNN(nn.Module):
    """
    CNN for position evaluation over 14×8×8 board tensor.
    Architecture: input conv → 6 residual blocks → value head
    ~500K parameters.
    """

    def __init__(self, in_channels: int = 14, num_filters: int = 128, num_res_blocks: int = 6):
        super().__init__()
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.num_res_blocks = num_res_blocks

        # Input convolution: 14 channels → num_filters
        self.input_conv = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(num_filters)

        # Residual tower
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_res_blocks)]
        )

        # Value head: conv → flatten → FC → FC → tanh
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 14, 8, 8)
        out = torch.relu(self.input_bn(self.input_conv(x)))
        out = self.res_blocks(out)

        # Value head
        v = torch.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)  # flatten to (batch, 64)
        v = torch.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        return v.squeeze(-1)

    def save_checkpoint(self, path: str, **metadata):
        metadata["model_type"] = "cnn"
        metadata["in_channels"] = self.in_channels
        metadata["num_filters"] = self.num_filters
        metadata["num_res_blocks"] = self.num_res_blocks
        torch.save({"state_dict": self.state_dict(), "metadata": metadata}, path)

    @classmethod
    def load_checkpoint(
        cls, path: str, in_channels: int = 14, num_filters: int = 128, num_res_blocks: int = 6
    ) -> Tuple["ChessCNN", dict]:
        data = torch.load(path, weights_only=False, map_location="cpu")
        meta = data.get("metadata", {})
        net = cls(
            in_channels=meta.get("in_channels", in_channels),
            num_filters=meta.get("num_filters", num_filters),
            num_res_blocks=meta.get("num_res_blocks", num_res_blocks),
        )
        net.load_state_dict(data["state_dict"])
        return net, meta


# ── Evaluators ────────────────────────────────────────────────────────────────

class TorchEvaluator:
    """Wraps ChessNet (MLP) for use with the existing game engine."""

    def __init__(self, model: ChessNet, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.encoder = StateEncoder()
        self.model.eval()

    def evaluate(self, game: ExtinctionChess) -> float:
        if game.game_over:
            if game.winner == Color.WHITE:
                return 1.0
            elif game.winner == Color.BLACK:
                return -1.0
            return 0.0

        features = self.encoder.get_simple_features(game)
        with torch.no_grad():
            t = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.model(t).item()

    def batch_evaluate(self, games: List[ExtinctionChess]) -> np.ndarray:
        features = np.stack([self.encoder.get_simple_features(g) for g in games])
        with torch.no_grad():
            t = torch.tensor(features, dtype=torch.float32, device=self.device)
            return self.model(t).cpu().numpy()


class CNNEvaluator:
    """Wraps ChessCNN for use with the existing game engine."""

    def __init__(self, model: ChessCNN, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.encoder = StateEncoder()
        self.model.eval()

    def evaluate(self, game: ExtinctionChess) -> float:
        if game.game_over:
            if game.winner == Color.WHITE:
                return 1.0
            elif game.winner == Color.BLACK:
                return -1.0
            return 0.0

        board_tensor = self.encoder.encode_board(game)
        with torch.no_grad():
            t = torch.tensor(board_tensor, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.model(t).item()

    def batch_evaluate(self, games: List[ExtinctionChess]) -> np.ndarray:
        boards = np.stack([self.encoder.encode_board(g) for g in games])
        with torch.no_grad():
            t = torch.tensor(boards, dtype=torch.float32, device=self.device)
            return self.model(t).cpu().numpy()


# ── Minimax search agent ─────────────────────────────────────────────────────

class SearchAgent:
    """Wraps any evaluator with negamax + alpha-beta search for stronger play."""

    def __init__(self, evaluator, depth: int = 3):
        self.evaluator = evaluator
        self.depth = depth

    def _negamax(self, game, depth, alpha, beta):
        """Negamax with alpha-beta.  Returns score from current player's POV."""
        if game.game_over:
            if game.winner is None:
                return 0.0
            # Engine quirk: current_player is NOT switched on game end,
            # so current_player == the side that just moved == the winner
            # (only the winner can cause extinction).
            # That means the *previous* mover won; it's now nominally
            # "their turn" still.  From the current_player's perspective
            # the game is over and they won → but negamax is called from
            # the opponent's frame after the move, so we need to check.
            if game.winner == game.current_player:
                # Current player (mover) won — good for them
                return 1.0
            else:
                # Current player lost
                return -1.0

        if depth == 0:
            return self.evaluator.evaluate(game)

        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return 0.0

        best = -float("inf")
        for m in legal_moves:
            gc = ExtinctionChess()
            gc.board = game.board.copy()
            gc.current_player = game.current_player
            gc.game_over = game.game_over
            if not gc.make_move(m):
                continue

            if gc.game_over:
                # Terminal — score from mover's perspective
                if gc.winner == game.current_player:
                    return 1.0  # instant win, prune
                elif gc.winner is not None:
                    score = -1.0
                else:
                    score = 0.0
            else:
                # Non-terminal — current_player switched, negate
                score = -self._negamax(gc, depth - 1, -beta, -alpha)

            best = max(best, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best

    def select_move(self, game, temperature=0):
        """Pick the best move using depth-limited negamax search."""
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return None

        best_score = -float("inf")
        best_move = legal_moves[0]

        for m in legal_moves:
            gc = ExtinctionChess()
            gc.board = game.board.copy()
            gc.current_player = game.current_player
            gc.game_over = game.game_over
            if not gc.make_move(m):
                continue

            if gc.game_over:
                if gc.winner == game.current_player:
                    return m  # instant win
                elif gc.winner is not None:
                    score = -1.0
                else:
                    score = 0.0
            else:
                score = -self._negamax(gc, self.depth - 1, -float("inf"), float("inf"))

            if score > best_score:
                best_score = score
                best_move = m

        return best_move


# ── Greedy-random opponent ────────────────────────────────────────────────────

def greedy_random_move(game: ExtinctionChess):
    """Pick a move that causes extinction if possible, otherwise random."""
    legal_moves = game.get_legal_moves()
    if not legal_moves:
        return None

    enemy_color = Color.BLACK if game.current_player == Color.WHITE else Color.WHITE
    enemy_counts = game.board.get_piece_count(enemy_color)

    # Check for any move that captures an endangered piece (instant win)
    for m in legal_moves:
        target = game.board.get_piece(m.to_pos)
        if target and target.color == enemy_color and enemy_counts[target.piece_type] == 1:
            return m

        # Also check en passant captures
        if m.is_en_passant:
            captured_pos = Position(m.from_pos.rank, m.to_pos.file)
            target = game.board.get_piece(captured_pos)
            if target and target.color == enemy_color and enemy_counts[target.piece_type] == 1:
                return m

    return random.choice(legal_moves)


# ── Play against a past checkpoint ────────────────────────────────────────────

def play_vs_checkpoint(
    evaluator,
    opponent_evaluator,
    max_moves: int = 200,
    dirichlet_alpha: float = 0.3,
    noise_weight: float = 0.25,
    use_cnn: bool = False,
    model_is_white: bool = True,
) -> Tuple[List[np.ndarray], float]:
    """Play a game against a past checkpoint. Model uses noise, opponent plays greedy."""
    game = ExtinctionChess()
    encoder = StateEncoder()
    positions: List[np.ndarray] = []

    move_count = 0
    while not game.game_over and move_count < max_moves:
        if use_cnn:
            positions.append(encoder.encode_board(game))
        else:
            positions.append(encoder.get_simple_features(game))

        is_model_turn = (game.current_player == Color.WHITE) == model_is_white

        if is_model_turn:
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                break
            scores = _evaluate_moves(evaluator, game, legal_moves)
            move = _select_move_with_noise(legal_moves, scores, dirichlet_alpha, noise_weight)
        else:
            # Opponent plays greedy (no noise) — like a fixed policy
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                break
            scores = _evaluate_moves(opponent_evaluator, game, legal_moves)
            move = legal_moves[int(np.argmax(scores))]

        game.make_move(move)
        move_count += 1

    if game.winner == Color.WHITE:
        outcome = 1.0
    elif game.winner == Color.BLACK:
        outcome = -1.0
    else:
        outcome = 0.0

    return positions, outcome


# ── Self-play (works with either evaluator) ───────────────────────────────────

def _evaluate_moves(evaluator, game, legal_moves):
    """Batch evaluate all legal moves. Returns scores from mover's perspective.

    Subtlety: the game engine does NOT switch current_player when a move
    ends the game (extinction / draw), so after a terminal move
    current_player == mover, while after a non-terminal move
    current_player == opponent.  We handle the two cases separately.
    """
    game_copies = []
    valid_indices = []
    for idx, m in enumerate(legal_moves):
        gc = ExtinctionChess()
        gc.board = game.board.copy()
        gc.current_player = game.current_player
        gc.game_over = game.game_over
        if gc.make_move(m):
            game_copies.append(gc)
            valid_indices.append(idx)

    full_scores = np.full(len(legal_moves), -2.0)
    if not game_copies:
        return full_scores

    # Split terminal vs non-terminal positions
    non_terminal_idx = []
    non_terminal_games = []
    for i, gc in enumerate(game_copies):
        if gc.game_over:
            # Terminal: assign exact score from mover's perspective
            if gc.winner == game.current_player:   # mover won
                full_scores[valid_indices[i]] = 1.0
            elif gc.winner is not None:             # mover lost
                full_scores[valid_indices[i]] = -1.0
            else:                                   # draw
                full_scores[valid_indices[i]] = 0.0
        else:
            non_terminal_idx.append(i)
            non_terminal_games.append(gc)

    # Non-terminal: current_player switched to opponent, negate
    if non_terminal_games:
        nn_scores = -evaluator.batch_evaluate(non_terminal_games)
        for j, i in enumerate(non_terminal_idx):
            full_scores[valid_indices[i]] = nn_scores[j]

    return full_scores


def _select_move_with_noise(legal_moves, scores, dirichlet_alpha=0.3, noise_weight=0.25):
    """Select a move using softmax over scores mixed with Dirichlet noise."""
    # Normalize scores to [0, 1] range for mixing with noise
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min > 1e-8:
        normalized = (scores - s_min) / (s_max - s_min)
    else:
        normalized = np.ones_like(scores) / len(scores)

    # Mix with Dirichlet noise
    noise = np.random.dirichlet([dirichlet_alpha] * len(legal_moves))
    mixed = (1 - noise_weight) * normalized + noise_weight * noise

    # Sample from the distribution
    probs = mixed / mixed.sum()
    return legal_moves[np.random.choice(len(legal_moves), p=probs)]


def play_one_game(
    evaluator,
    max_moves: int = 200,
    dirichlet_alpha: float = 0.3,
    noise_weight: float = 0.25,
    use_cnn: bool = False,
) -> Tuple[List[np.ndarray], float]:
    """Play a single self-play game with Dirichlet noise. Returns (positions, outcome)."""
    game = ExtinctionChess()
    encoder = StateEncoder()
    positions: List[np.ndarray] = []

    move_count = 0
    while not game.game_over and move_count < max_moves:
        if use_cnn:
            positions.append(encoder.encode_board(game))
        else:
            positions.append(encoder.get_simple_features(game))

        legal_moves = game.get_legal_moves()
        if not legal_moves:
            break

        scores = _evaluate_moves(evaluator, game, legal_moves)
        move = _select_move_with_noise(legal_moves, scores, dirichlet_alpha, noise_weight)

        game.make_move(move)
        move_count += 1

    if game.winner == Color.WHITE:
        outcome = 1.0
    elif game.winner == Color.BLACK:
        outcome = -1.0
    else:
        outcome = 0.0

    return positions, outcome


def play_vs_greedy(
    evaluator,
    max_moves: int = 200,
    dirichlet_alpha: float = 0.3,
    noise_weight: float = 0.25,
    use_cnn: bool = False,
    model_is_white: bool = True,
) -> Tuple[List[np.ndarray], float]:
    """Play a game against the greedy-random opponent with Dirichlet noise."""
    game = ExtinctionChess()
    encoder = StateEncoder()
    positions: List[np.ndarray] = []

    move_count = 0
    while not game.game_over and move_count < max_moves:
        if use_cnn:
            positions.append(encoder.encode_board(game))
        else:
            positions.append(encoder.get_simple_features(game))

        is_model_turn = (game.current_player == Color.WHITE) == model_is_white

        if is_model_turn:
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                break
            scores = _evaluate_moves(evaluator, game, legal_moves)
            move = _select_move_with_noise(legal_moves, scores, dirichlet_alpha, noise_weight)
        else:
            move = greedy_random_move(game)
            if not move:
                break

        game.make_move(move)
        move_count += 1

    if game.winner == Color.WHITE:
        outcome = 1.0
    elif game.winner == Color.BLACK:
        outcome = -1.0
    else:
        outcome = 0.0

    return positions, outcome


def test_vs_random(evaluator, num_games: int = 100) -> float:
    """Win-rate of the model against a uniform-random opponent.
    Returns score: wins + 0.5*draws, divided by num_games."""
    wins, losses, draws = 0, 0, 0
    for i in range(num_games):
        game = ExtinctionChess()
        model_is_white = i % 2 == 0
        moves = 0
        while not game.game_over and moves < 200:
            legal = game.get_legal_moves()
            if not legal:
                break
            is_model_turn = (game.current_player == Color.WHITE) == model_is_white
            if is_model_turn:
                scores = _evaluate_moves(evaluator, game, legal)
                best_idx = int(np.argmax(scores))
                move = legal[best_idx]
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


# ── MCTS (AlphaZero-style self-play) ────────────────────────────────────────

class MCTSNode:
    """MCTS tree node.  All values stored from White's perspective so
    we never have to worry about the engine's no-switch-on-terminal quirk."""

    __slots__ = ('game', 'parent', 'move', 'prior', 'children',
                 'visit_count', 'value_sum', 'is_expanded')

    def __init__(self, game, parent=None, move=None, prior=1.0):
        self.game = game
        self.parent = parent
        self.move = move
        self.prior = prior
        self.children: List["MCTSNode"] = []
        self.visit_count = 0
        self.value_sum = 0.0          # cumulative value from White's POV
        self.is_expanded = False

    def q_from_parent(self):
        """Q-value from the parent's current_player perspective."""
        if self.visit_count == 0:
            return 0.0
        wq = self.value_sum / self.visit_count
        return wq if self.parent.game.current_player == Color.WHITE else -wq

    def ucb(self, c_puct):
        return (self.q_from_parent()
                + c_puct * self.prior
                * math.sqrt(self.parent.visit_count) / (1 + self.visit_count))

    def best_child(self, c_puct):
        return max(self.children, key=lambda c: c.ucb(c_puct))

    def expand(self):
        """Create children with uniform priors."""
        if self.game.game_over:
            self.is_expanded = True
            return
        legal = self.game.get_legal_moves()
        if not legal:
            self.is_expanded = True
            return
        p = 1.0 / len(legal)
        for m in legal:
            gc = ExtinctionChess()
            gc.board = self.game.board.copy()
            gc.current_player = self.game.current_player
            gc.game_over = self.game.game_over
            if gc.make_move(m):
                self.children.append(MCTSNode(gc, parent=self, move=m, prior=p))
        self.is_expanded = True

    def backpropagate(self, white_value):
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += white_value
            node = node.parent


def _eval_white(evaluator, game):
    """Evaluate a position and return value from White's perspective."""
    if game.game_over:
        if game.winner == Color.WHITE:
            return 1.0
        elif game.winner == Color.BLACK:
            return -1.0
        return 0.0
    # Network returns value from current_player's perspective
    v = evaluator.evaluate(game)
    return v if game.current_player == Color.WHITE else -v


def _batch_eval_white(evaluator, games):
    """Batch evaluate positions and return values from White's perspective."""
    if not games:
        return []
    scores = evaluator.batch_evaluate(games)
    results = []
    for i, g in enumerate(games):
        if g.game_over:
            if g.winner == Color.WHITE:
                results.append(1.0)
            elif g.winner == Color.BLACK:
                results.append(-1.0)
            else:
                results.append(0.0)
        else:
            v = scores[i]
            results.append(v if g.current_player == Color.WHITE else -v)
    return results


VIRTUAL_LOSS = 1.0  # discourages other paths from selecting same node


def mcts_search(game, evaluator, num_simulations=100, c_puct=1.5,
                dirichlet_alpha=0.3, noise_weight=0.25, batch_size=8):
    """Batched MCTS: collects multiple leaves per step, evaluates in one
    GPU call.  Uses virtual losses to diversify selection paths."""
    root = MCTSNode(game)
    root.expand()
    if not root.children:
        return []

    # Dirichlet noise at root for exploration
    if dirichlet_alpha > 0 and noise_weight > 0:
        noise = np.random.dirichlet([dirichlet_alpha] * len(root.children))
        for i, ch in enumerate(root.children):
            ch.prior = (1 - noise_weight) * ch.prior + noise_weight * noise[i]

    sims_done = 0
    while sims_done < num_simulations:
        # Collect a batch of leaves
        current_batch = min(batch_size, num_simulations - sims_done)
        leaves = []
        for _ in range(current_batch):
            node = root
            # Selection with virtual loss
            while node.is_expanded and node.children:
                node = node.best_child(c_puct)
            # Apply virtual loss to discourage same path
            node.visit_count += 1
            node.value_sum -= VIRTUAL_LOSS
            # Expansion
            if not node.game.game_over:
                node.expand()
            leaves.append(node)

        # Batch evaluate all leaves
        games_to_eval = [leaf.game for leaf in leaves]
        values = _batch_eval_white(evaluator, games_to_eval)

        # Undo virtual losses and backpropagate real values
        for leaf, white_value in zip(leaves, values):
            # Undo the virtual loss we applied during selection
            leaf.visit_count -= 1
            leaf.value_sum += VIRTUAL_LOSS
            # Now do real backpropagation
            leaf.backpropagate(white_value)

        sims_done += current_batch

    return [(ch.move, ch.visit_count) for ch in root.children]


def play_one_game_mcts(
    evaluator,
    num_simulations: int = 100,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 0.3,
    noise_weight: float = 0.25,
    use_cnn: bool = False,
    max_moves: int = 200,
    temp_threshold: int = 30,
) -> Tuple[List[np.ndarray], float]:
    """Self-play game using MCTS for both sides."""
    game = ExtinctionChess()
    encoder = StateEncoder()
    positions: List[np.ndarray] = []

    move_count = 0
    while not game.game_over and move_count < max_moves:
        positions.append(
            encoder.encode_board(game) if use_cnn
            else encoder.get_simple_features(game)
        )
        move_visits = mcts_search(game, evaluator, num_simulations, c_puct,
                                  dirichlet_alpha, noise_weight)
        if not move_visits:
            break

        moves, counts = zip(*move_visits)
        counts = np.array(counts, dtype=np.float64)

        if move_count < temp_threshold:
            # Temperature 1: sample proportional to visit counts
            probs = counts / counts.sum()
            idx = np.random.choice(len(moves), p=probs)
        else:
            # Temperature → 0: pick most-visited
            idx = int(np.argmax(counts))

        game.make_move(moves[idx])
        move_count += 1

    if game.winner == Color.WHITE:
        outcome = 1.0
    elif game.winner == Color.BLACK:
        outcome = -1.0
    else:
        outcome = 0.0
    return positions, outcome


def play_vs_greedy_mcts(
    evaluator,
    num_simulations: int = 100,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 0.3,
    noise_weight: float = 0.25,
    use_cnn: bool = False,
    max_moves: int = 200,
    temp_threshold: int = 30,
    model_is_white: bool = True,
) -> Tuple[List[np.ndarray], float]:
    """Model uses MCTS; opponent uses greedy-random."""
    game = ExtinctionChess()
    encoder = StateEncoder()
    positions: List[np.ndarray] = []

    move_count = 0
    while not game.game_over and move_count < max_moves:
        positions.append(
            encoder.encode_board(game) if use_cnn
            else encoder.get_simple_features(game)
        )
        is_model_turn = (game.current_player == Color.WHITE) == model_is_white

        if is_model_turn:
            move_visits = mcts_search(game, evaluator, num_simulations, c_puct,
                                      dirichlet_alpha, noise_weight)
            if not move_visits:
                break
            moves, counts = zip(*move_visits)
            counts = np.array(counts, dtype=np.float64)
            if move_count < temp_threshold:
                probs = counts / counts.sum()
                idx = np.random.choice(len(moves), p=probs)
            else:
                idx = int(np.argmax(counts))
            move = moves[idx]
        else:
            move = greedy_random_move(game)
            if not move:
                break

        game.make_move(move)
        move_count += 1

    if game.winner == Color.WHITE:
        outcome = 1.0
    elif game.winner == Color.BLACK:
        outcome = -1.0
    else:
        outcome = 0.0
    return positions, outcome


# ── Parallel self-play helpers ───────────────────────────────────────────────
#
# PyTorch models and evaluators aren't picklable, so we use a
# worker-initializer pattern: each process receives the serialized
# state_dict and rebuilds the evaluator once on start-up.

_worker_evaluator = None
_worker_game_kwargs: dict = {}


def _init_torch_worker(state_dict_bytes: bytes, model_cls_name: str,
                       model_init_kwargs: dict, device: str,
                       game_fn_name: str, game_kwargs: dict):
    """Pool initializer — reconstruct the model & evaluator in each worker."""
    import io as _io
    global _worker_evaluator, _worker_game_kwargs
    _worker_game_kwargs = game_kwargs
    _worker_game_kwargs["_fn"] = game_fn_name

    state_dict = torch.load(
        _io.BytesIO(state_dict_bytes),
        weights_only=False, map_location="cpu",
    )

    cls = {"ChessNet": ChessNet, "ChessCNN": ChessCNN}[model_cls_name]
    model = cls(**model_init_kwargs)
    model.load_state_dict(state_dict)
    model.eval()

    evaluator_cls = CNNEvaluator if model_cls_name == "ChessCNN" else TorchEvaluator
    _worker_evaluator = evaluator_cls(model, device=device)


def _play_game_worker(seed: int) -> Tuple[List, float]:
    """Worker target — play one game and return (positions-as-lists, outcome)."""
    random.seed(seed)
    np.random.seed(seed % (2**32))

    fn_name = _worker_game_kwargs["_fn"]
    fn = {
        "play_one_game": play_one_game,
        "play_vs_greedy": play_vs_greedy,
        "play_one_game_mcts": play_one_game_mcts,
        "play_vs_greedy_mcts": play_vs_greedy_mcts,
    }[fn_name]

    kwargs = {k: v for k, v in _worker_game_kwargs.items() if k != "_fn"}
    positions, outcome = fn(_worker_evaluator, **kwargs)
    return [p.tolist() for p in positions], outcome


def _serialize_state_dict(model: nn.Module) -> bytes:
    """Serialize a model's state_dict to bytes for passing to workers."""
    import io as _io
    buf = _io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getvalue()


def parallel_play(
    model: nn.Module,
    num_games: int,
    game_fn: str = "play_one_game",
    num_workers: int = 0,
    device: str = "cpu",
    **game_kwargs,
) -> List[Tuple[List[np.ndarray], float]]:
    """Play multiple self-play games in parallel using a process pool.

    Args:
        model: A ChessNet or ChessCNN instance (weights will be serialized
               and sent to each worker).
        num_games: Number of games to generate.
        game_fn: Which game function to call per worker.  One of
                 ``"play_one_game"``, ``"play_vs_greedy"``,
                 ``"play_one_game_mcts"``, ``"play_vs_greedy_mcts"``.
        num_workers: Worker processes.  0 = ``cpu_count() - 1``.
        device: Torch device for inference inside workers (usually "cpu"
                since each worker is its own process).
        **game_kwargs: Forwarded to the chosen game function
                       (e.g. ``use_cnn=True``, ``max_moves=200``).

    Returns:
        List of ``(positions, outcome)`` tuples, one per game.
    """
    if num_workers == 0:
        num_workers = max(1, mp.cpu_count() - 1)

    if num_workers == 1:
        evaluator_cls = CNNEvaluator if isinstance(model, ChessCNN) else TorchEvaluator
        evaluator = evaluator_cls(model, device=device)
        fn = {
            "play_one_game": play_one_game,
            "play_vs_greedy": play_vs_greedy,
            "play_one_game_mcts": play_one_game_mcts,
            "play_vs_greedy_mcts": play_vs_greedy_mcts,
        }[game_fn]
        return [fn(evaluator, **game_kwargs) for _ in range(num_games)]

    # Determine model class info for reconstruction in workers
    if isinstance(model, ChessCNN):
        cls_name = "ChessCNN"
        init_kwargs = {
            "in_channels": model.in_channels,
            "num_filters": model.num_filters,
            "num_res_blocks": model.num_res_blocks,
        }
    else:
        cls_name = "ChessNet"
        linear_layers = [m for m in model.net if isinstance(m, nn.Linear)]
        init_kwargs = {
            "input_size": linear_layers[0].in_features,
            "hidden_sizes": [l.out_features for l in linear_layers[:-1]],
        }

    state_bytes = _serialize_state_dict(model)
    base_seed = random.randint(0, 2**31)
    seeds = [base_seed + i for i in range(num_games)]

    print(f"Generating {num_games} games ({num_workers} workers, fn={game_fn})...")

    with mp.Pool(
        processes=num_workers,
        initializer=_init_torch_worker,
        initargs=(state_bytes, cls_name, init_kwargs, device,
                  game_fn, game_kwargs),
    ) as pool:
        raw_results = pool.map(_play_game_worker, seeds)

    results = []
    for positions_lists, outcome in raw_results:
        positions = [np.array(p, dtype=np.float32) for p in positions_lists]
        results.append((positions, outcome))

    return results
