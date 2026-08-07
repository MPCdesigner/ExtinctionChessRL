"""Board widget for the positional evaluation tool.

Renders an 8x8 chess board with pieces at a given screen position, and
routes clicks to whoever owns it.

- Rank 0 (white's back rank) is drawn at the BOTTOM (standard chess view).
- Squares alternate light/dark.
- Pieces rendered as Unicode glyphs so no external image assets needed.
- The widget itself is stateless w.r.t. game rules; it just renders
  whatever board it's given and reports clicks back as (rank, file).

Owner is responsible for:
  - Deciding what to do with clicks (select, place, remove, promote)
  - Passing highlight sets (selected square, legal-move targets, last move)
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Set, Tuple

import pygame

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from extinction_chess import Color, PieceType, Position  # noqa: E402


# Unicode glyphs — filled = white, hollow = black
_PIECE_GLYPHS = {
    (PieceType.KING,   Color.WHITE): "♔",
    (PieceType.QUEEN,  Color.WHITE): "♕",
    (PieceType.ROOK,   Color.WHITE): "♖",
    (PieceType.BISHOP, Color.WHITE): "♗",
    (PieceType.KNIGHT, Color.WHITE): "♘",
    (PieceType.PAWN,   Color.WHITE): "♙",
    (PieceType.KING,   Color.BLACK): "♚",
    (PieceType.QUEEN,  Color.BLACK): "♛",
    (PieceType.ROOK,   Color.BLACK): "♜",
    (PieceType.BISHOP, Color.BLACK): "♝",
    (PieceType.KNIGHT, Color.BLACK): "♞",
    (PieceType.PAWN,   Color.BLACK): "♟",
}


class BoardWidget:
    LIGHT_SQUARE = (240, 217, 181)
    DARK_SQUARE  = (181, 136, 99)
    SELECTED_TINT = (100, 180, 255, 120)
    LEGAL_TARGET_TINT = (100, 220, 100, 90)
    LAST_MOVE_TINT = (240, 230, 100, 90)
    ENDANGERED_TINT = (255, 90, 90, 100)

    def __init__(self, x: int, y: int, size: int = 480):
        """Board occupies a square region of `size` pixels starting at (x, y)."""
        self.x = x
        self.y = y
        self.size = size
        self.square_size = size // 8
        # Font sized for the square; leave a small margin
        self.font = pygame.font.SysFont(
            "Segoe UI Symbol,DejaVu Sans,Arial Unicode MS",
            int(self.square_size * 0.75),
        )

    # ── Coordinate conversion ──────────────────────────────────────────────

    def square_rect(self, rank: int, file: int) -> pygame.Rect:
        """Return the pygame.Rect for the given (rank, file) in screen coords."""
        # rank 0 at bottom -> screen y grows downward, so invert
        screen_row = 7 - rank
        return pygame.Rect(
            self.x + file * self.square_size,
            self.y + screen_row * self.square_size,
            self.square_size,
            self.square_size,
        )

    def pixel_to_square(self, mx: int, my: int) -> Optional[Position]:
        """Convert a mouse click to a Position, or None if outside the board."""
        if not (self.x <= mx < self.x + self.size
                and self.y <= my < self.y + self.size):
            return None
        file = (mx - self.x) // self.square_size
        screen_row = (my - self.y) // self.square_size
        rank = 7 - screen_row
        return Position(int(rank), int(file))

    # ── Rendering ──────────────────────────────────────────────────────────

    def draw(self, surface: pygame.Surface, position_state,
             selected: Optional[Position] = None,
             legal_targets: Optional[Set[Position]] = None,
             last_move_from: Optional[Position] = None,
             last_move_to: Optional[Position] = None,
             endangered_squares: Optional[Set[Position]] = None) -> None:
        """Draw the board plus overlays.

        Arguments after `position_state` are optional highlights.

        endangered_squares (extinction-chess-specific): squares whose piece
        is the last of its type for its color. Rendered with a red tint
        under other highlights so tactical stakes are visible at a glance.
        Set only when the corresponding settings option is on; passing None
        or empty set draws nothing.
        """
        game = position_state.get_game()

        # Squares
        for rank in range(8):
            for file in range(8):
                rect = self.square_rect(rank, file)
                is_light = (rank + file) % 2 == 1
                pygame.draw.rect(
                    surface,
                    self.LIGHT_SQUARE if is_light else self.DARK_SQUARE,
                    rect,
                )

        # Highlight overlays (drawn under pieces so pieces stay legible).
        # Endangered goes first so more transient highlights (selected,
        # legal targets, last move) render over it if they overlap.
        overlay = pygame.Surface((self.square_size, self.square_size),
                                 pygame.SRCALPHA)
        if endangered_squares:
            overlay.fill(self.ENDANGERED_TINT)
            for sq in endangered_squares:
                surface.blit(overlay, self.square_rect(sq.rank, sq.file).topleft)
        if last_move_from is not None:
            overlay.fill(self.LAST_MOVE_TINT)
            surface.blit(overlay, self.square_rect(
                last_move_from.rank, last_move_from.file).topleft)
        if last_move_to is not None:
            overlay.fill(self.LAST_MOVE_TINT)
            surface.blit(overlay, self.square_rect(
                last_move_to.rank, last_move_to.file).topleft)
        if legal_targets:
            overlay.fill(self.LEGAL_TARGET_TINT)
            for t in legal_targets:
                surface.blit(overlay, self.square_rect(t.rank, t.file).topleft)
        if selected is not None:
            overlay.fill(self.SELECTED_TINT)
            surface.blit(overlay, self.square_rect(
                selected.rank, selected.file).topleft)

        # Pieces
        for rank in range(8):
            for file in range(8):
                piece = position_state.get_piece_at(Position(rank, file))
                if piece is None:
                    continue
                glyph = _PIECE_GLYPHS.get((piece.piece_type, piece.color), "?")
                # Render in black for both colors; white/black is encoded in
                # the glyph itself (filled vs hollow).
                text = self.font.render(glyph, True, (0, 0, 0))
                rect = self.square_rect(rank, file)
                surface.blit(
                    text,
                    text.get_rect(center=rect.center),
                )

        # Border
        pygame.draw.rect(
            surface, (0, 0, 0),
            pygame.Rect(self.x, self.y, self.size, self.size),
            width=2,
        )

        # Rank / file labels (small, along left and bottom)
        label_font = pygame.font.SysFont("Arial", 12)
        for i in range(8):
            # Files a-h along bottom
            label = label_font.render(chr(97 + i), True, (60, 60, 60))
            surface.blit(label, (
                self.x + i * self.square_size + 2,
                self.y + self.size - 14,
            ))
            # Ranks 1-8 along left (rank 0 at bottom -> label '1')
            label = label_font.render(str(i + 1), True, (60, 60, 60))
            surface.blit(label, (
                self.x + 2,
                self.y + (7 - i) * self.square_size + 2,
            ))


def glyph_for(piece_type: PieceType, color: Color) -> str:
    """Public accessor for the piece glyph lookup — used by the palette widget."""
    return _PIECE_GLYPHS.get((piece_type, color), "?")
