"""Position state wrapper for the positional evaluation tool.

Wraps ExtinctionChess to support two modes:
  - GAME_SETUP: play moves from the starting position. history planes are
    populated naturally by ExtinctionChess.make_move(). Supports step-back
    and step-forward via a move history list.
  - CONSTRUCTION: start with an empty board, place pieces manually, set
    whose turn, set castling rights, set en passant target. History planes
    will be empty (the model was trained on real histories, so evaluations
    of constructed positions have this systematic bias worth remembering).

Validation runs before evaluation. In extinction chess:
  - Both sides need >= 1 of each piece type (extinction rule)
  - Each side can have at most 16 pieces
  - No pawns on rank 1 or 8 (they would have promoted)
  - One piece per square
  - Multiple kings per side are legal (via promotion)
"""

from __future__ import annotations

import json
import os
import sys
from enum import Enum
from typing import List, Optional, Tuple

# Add src/ to sys.path so we can import project modules
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from extinction_chess import (  # noqa: E402
    ExtinctionChess, Color, PieceType, Piece, Position, Move,
)


class Mode(Enum):
    GAME_SETUP = "game_setup"
    CONSTRUCTION = "construction"


MAX_PIECES_PER_SIDE = 16


class PositionState:
    """Wraps an ExtinctionChess instance plus mode-specific state.

    Access the underlying game via `get_game()` for evaluation.
    """

    def __init__(self, mode: Mode = Mode.GAME_SETUP):
        self.mode = mode
        self.game = ExtinctionChess()
        self.move_history: List[Move] = []
        self.step_position: int = 0  # index into move_history

        if mode == Mode.CONSTRUCTION:
            self._clear_board()

    # ── Internal helpers ────────────────────────────────────────────────────

    def _clear_board(self) -> None:
        for r in range(8):
            for f in range(8):
                self.game.board.grid[r][f] = None
        self.game.board.en_passant_target = None
        self.game.board.halfmove_clock = 0
        self.game.board.fullmove_number = 1
        self.game.current_player = Color.WHITE
        self.game.game_over = False
        self.game.winner = None

    def _rebuild_from_history(self) -> None:
        """Recreate the game state by replaying the first `step_position` moves."""
        self.game = ExtinctionChess()
        for i in range(self.step_position):
            self.game.make_move(self.move_history[i])

    # ── Game setup mode ─────────────────────────────────────────────────────

    def make_move(self, from_pos: Position, to_pos: Position,
                  promotion: Optional[PieceType] = None) -> bool:
        """Play a move (GAME_SETUP only). Returns True on success.

        If we had previously stepped back, this discards the "future" history
        beyond the current step so we branch cleanly.
        """
        if self.mode != Mode.GAME_SETUP:
            return False
        if self.step_position < len(self.move_history):
            self.move_history = self.move_history[:self.step_position]

        for m in self.game.get_legal_moves():
            if (m.from_pos == from_pos and m.to_pos == to_pos
                    and m.promotion == promotion):
                if self.game.make_move(m):
                    self.move_history.append(m)
                    self.step_position += 1
                    return True
                return False
        return False

    def step_back(self) -> bool:
        if self.mode != Mode.GAME_SETUP or self.step_position == 0:
            return False
        self.step_position -= 1
        self._rebuild_from_history()
        return True

    def step_forward(self) -> bool:
        if (self.mode != Mode.GAME_SETUP
                or self.step_position >= len(self.move_history)):
            return False
        self.step_position += 1
        self._rebuild_from_history()
        return True

    # ── Construction mode ───────────────────────────────────────────────────

    def place_piece(self, pos: Position, piece_type: PieceType,
                    color: Color) -> bool:
        """Place a piece at `pos`. Overwrites whatever was there."""
        if self.mode != Mode.CONSTRUCTION:
            return False
        piece = Piece(piece_type, color, pos)
        # Default has_moved=True — castling only enabled explicitly via
        # set_castling() and only if kings/rooks are on starting squares.
        piece.has_moved = True
        self.game.board.grid[pos.rank][pos.file] = piece
        return True

    def remove_piece(self, pos: Position) -> bool:
        if self.mode != Mode.CONSTRUCTION:
            return False
        self.game.board.grid[pos.rank][pos.file] = None
        return True

    def set_current_player(self, color: Color) -> bool:
        if self.mode != Mode.CONSTRUCTION:
            return False
        self.game.current_player = color
        return True

    def set_castling(self, white_ks: bool, white_qs: bool,
                     black_ks: bool, black_qs: bool) -> bool:
        """Enable castling rights (CONSTRUCTION only).

        Only takes effect if the relevant king and rook are on their starting
        squares. Sets has_moved=False on those pieces.
        """
        if self.mode != Mode.CONSTRUCTION:
            return False

        def _at(rank: int, file: int) -> Optional[Piece]:
            return self.game.board.get_piece(Position(rank, file))

        # White king on e1 (rank 0, file 4)
        wk = _at(0, 4)
        if (white_ks or white_qs) and wk and wk.piece_type == PieceType.KING \
                and wk.color == Color.WHITE:
            wk.has_moved = False
        # Rooks
        wr_h = _at(0, 7)
        if white_ks and wr_h and wr_h.piece_type == PieceType.ROOK \
                and wr_h.color == Color.WHITE:
            wr_h.has_moved = False
        wr_a = _at(0, 0)
        if white_qs and wr_a and wr_a.piece_type == PieceType.ROOK \
                and wr_a.color == Color.WHITE:
            wr_a.has_moved = False

        bk = _at(7, 4)
        if (black_ks or black_qs) and bk and bk.piece_type == PieceType.KING \
                and bk.color == Color.BLACK:
            bk.has_moved = False
        br_h = _at(7, 7)
        if black_ks and br_h and br_h.piece_type == PieceType.ROOK \
                and br_h.color == Color.BLACK:
            br_h.has_moved = False
        br_a = _at(7, 0)
        if black_qs and br_a and br_a.piece_type == PieceType.ROOK \
                and br_a.color == Color.BLACK:
            br_a.has_moved = False

        return True

    def set_en_passant_target(self, pos: Optional[Position]) -> bool:
        if self.mode != Mode.CONSTRUCTION:
            return False
        self.game.board.en_passant_target = pos
        return True

    # ── Validation ──────────────────────────────────────────────────────────

    def validate(self) -> Tuple[bool, List[str]]:
        """Sanity-check the current board. Returns (is_valid, error_messages).

        Rules applied (per extinction chess):
          - each side must have >= 1 of every piece type
          - each side must have <= 16 total pieces
          - no pawns on rank 1 or 8
          - one piece per square is enforced structurally by the grid
        """
        errors: List[str] = []

        white_counts = {pt: 0 for pt in PieceType}
        black_counts = {pt: 0 for pt in PieceType}
        for r in range(8):
            for f in range(8):
                p = self.game.board.grid[r][f]
                if p is None:
                    continue
                if p.color == Color.WHITE:
                    white_counts[p.piece_type] += 1
                else:
                    black_counts[p.piece_type] += 1

        for pt in PieceType:
            if white_counts[pt] == 0:
                errors.append(
                    f"White has no {pt.value} (extinction rule requires >=1)")
            if black_counts[pt] == 0:
                errors.append(
                    f"Black has no {pt.value} (extinction rule requires >=1)")

        w_total = sum(white_counts.values())
        b_total = sum(black_counts.values())
        if w_total > MAX_PIECES_PER_SIDE:
            errors.append(
                f"White has {w_total} pieces (max {MAX_PIECES_PER_SIDE})")
        if b_total > MAX_PIECES_PER_SIDE:
            errors.append(
                f"Black has {b_total} pieces (max {MAX_PIECES_PER_SIDE})")

        # Pawns on rank 1 (index 0) or 8 (index 7)
        for f in range(8):
            for r in (0, 7):
                p = self.game.board.grid[r][f]
                if p and p.piece_type == PieceType.PAWN:
                    sq = Position(r, f).to_algebraic()
                    errors.append(
                        f"Pawn on {sq} (illegal: pawn would have promoted)")

        return (len(errors) == 0, errors)

    # ── Access ──────────────────────────────────────────────────────────────

    def get_game(self) -> ExtinctionChess:
        return self.game

    def get_piece_at(self, pos: Position) -> Optional[Piece]:
        return self.game.board.get_piece(pos)

    def get_legal_moves(self) -> List[Move]:
        return self.game.get_legal_moves()

    def get_current_player(self) -> Color:
        return self.game.current_player

    def is_game_over(self) -> bool:
        return self.game.game_over

    def get_winner(self) -> Optional[Color]:
        return self.game.winner

    def move_history_length(self) -> int:
        return len(self.move_history)

    def current_step(self) -> int:
        return self.step_position

    # ── Serialization ───────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        board = []
        for r in range(8):
            row = []
            for f in range(8):
                p = self.game.board.grid[r][f]
                if p is None:
                    row.append(None)
                else:
                    row.append({
                        "type": p.piece_type.value,
                        "color": "W" if p.color == Color.WHITE else "B",
                        "has_moved": p.has_moved,
                    })
            board.append(row)

        ep = self.game.board.en_passant_target
        return {
            "mode": self.mode.value,
            "board": board,
            "to_move": "W" if self.game.current_player == Color.WHITE else "B",
            "en_passant": [ep.rank, ep.file] if ep else None,
            "halfmove_clock": self.game.board.halfmove_clock,
            "fullmove_number": self.game.board.fullmove_number,
            "move_history": [
                {
                    "from": [m.from_pos.rank, m.from_pos.file],
                    "to": [m.to_pos.rank, m.to_pos.file],
                    "promotion": m.promotion.value if m.promotion else None,
                }
                for m in self.move_history
            ],
            "step_position": self.step_position,
        }

    def save_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "PositionState":
        mode = Mode(data["mode"])
        ps = cls(mode)

        if mode == Mode.GAME_SETUP and data.get("move_history"):
            # Rebuild by replaying moves — history planes populate naturally
            ps.move_history = [
                Move(
                    Position(m["from"][0], m["from"][1]),
                    Position(m["to"][0], m["to"][1]),
                    promotion=(
                        PieceType(m["promotion"]) if m["promotion"] else None),
                )
                for m in data["move_history"]
            ]
            ps.step_position = data.get("step_position", len(ps.move_history))
            ps._rebuild_from_history()
            return ps

        # Construction mode (or game_setup with no history): load board directly
        ps._clear_board()
        for r in range(8):
            for f in range(8):
                cell = data["board"][r][f]
                if cell is None:
                    continue
                piece = Piece(
                    PieceType(cell["type"]),
                    Color.WHITE if cell["color"] == "W" else Color.BLACK,
                    Position(r, f),
                )
                piece.has_moved = cell.get("has_moved", True)
                ps.game.board.grid[r][f] = piece

        ps.game.current_player = (
            Color.WHITE if data["to_move"] == "W" else Color.BLACK)
        ep = data.get("en_passant")
        if ep:
            ps.game.board.en_passant_target = Position(ep[0], ep[1])
        ps.game.board.halfmove_clock = data.get("halfmove_clock", 0)
        ps.game.board.fullmove_number = data.get("fullmove_number", 1)

        return ps

    @classmethod
    def load_json(cls, path: str) -> "PositionState":
        with open(path) as f:
            return cls.from_dict(json.load(f))
