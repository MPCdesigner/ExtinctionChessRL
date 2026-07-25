"""Piece palette widget for construction mode.

Shows 12 clickable slots (6 white pieces + 6 black pieces) plus an "Erase"
slot. Click a slot to select that piece type/color; the currently-selected
slot is highlighted. Clicking the board in construction mode then places
the selected piece.

Layout: 2 rows of 7 slots (top row white + erase, bottom row black + erase),
displayed as a horizontal strip.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import pygame

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from extinction_chess import Color, PieceType  # noqa: E402
from .board_widget import glyph_for  # noqa: E402


# What the palette exposes, in display order.
# Each slot is (label, piece_type_or_None, color_or_None)
# None on both means "erase"
_SLOTS_ROW_WHITE = [
    ("K", PieceType.KING,   Color.WHITE),
    ("Q", PieceType.QUEEN,  Color.WHITE),
    ("R", PieceType.ROOK,   Color.WHITE),
    ("B", PieceType.BISHOP, Color.WHITE),
    ("N", PieceType.KNIGHT, Color.WHITE),
    ("P", PieceType.PAWN,   Color.WHITE),
]
_SLOTS_ROW_BLACK = [
    ("k", PieceType.KING,   Color.BLACK),
    ("q", PieceType.QUEEN,  Color.BLACK),
    ("r", PieceType.ROOK,   Color.BLACK),
    ("b", PieceType.BISHOP, Color.BLACK),
    ("n", PieceType.KNIGHT, Color.BLACK),
    ("p", PieceType.PAWN,   Color.BLACK),
]
# Selection type: (PieceType, Color) for a piece, or (None, None) for erase.
Selection = Tuple[Optional[PieceType], Optional[Color]]


class PaletteWidget:
    SLOT_SIZE = 48
    SLOT_MARGIN = 4
    BORDER = (60, 60, 60)
    SELECTED_BORDER = (30, 120, 220)
    BG_LIGHT = (235, 235, 235)
    BG_HOVER = (215, 225, 240)
    ERASE_BG = (240, 200, 200)

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.selection: Selection = (PieceType.PAWN, Color.WHITE)
        self.font_glyph = pygame.font.SysFont(
            "Segoe UI Symbol,DejaVu Sans,Arial Unicode MS",
            int(self.SLOT_SIZE * 0.7),
        )
        self.font_label = pygame.font.SysFont("Arial", 12)

    # ── Geometry ────────────────────────────────────────────────────────────

    @property
    def width(self) -> int:
        # 6 pieces + 1 erase = 7 slots per row
        return 7 * (self.SLOT_SIZE + self.SLOT_MARGIN) + self.SLOT_MARGIN

    @property
    def height(self) -> int:
        return 2 * (self.SLOT_SIZE + self.SLOT_MARGIN) + self.SLOT_MARGIN + 20

    def _slot_rect(self, row: int, col: int) -> pygame.Rect:
        return pygame.Rect(
            self.x + self.SLOT_MARGIN + col * (self.SLOT_SIZE + self.SLOT_MARGIN),
            self.y + 20 + self.SLOT_MARGIN
            + row * (self.SLOT_SIZE + self.SLOT_MARGIN),
            self.SLOT_SIZE, self.SLOT_SIZE,
        )

    # ── Interaction ─────────────────────────────────────────────────────────

    def handle_click(self, mx: int, my: int) -> bool:
        """If a slot was clicked, update selection and return True."""
        for row_idx, row in enumerate([_SLOTS_ROW_WHITE, _SLOTS_ROW_BLACK]):
            for col_idx, (_, pt, col) in enumerate(row):
                if self._slot_rect(row_idx, col_idx).collidepoint(mx, my):
                    self.selection = (pt, col)
                    return True
            # Erase slot at end of each row
            if self._slot_rect(row_idx, 6).collidepoint(mx, my):
                self.selection = (None, None)
                return True
        return False

    def get_selection(self) -> Selection:
        return self.selection

    # ── Rendering ──────────────────────────────────────────────────────────

    def draw(self, surface: pygame.Surface) -> None:
        title = self.font_label.render(
            "Palette (click to select, then click board to place)",
            True, (30, 30, 30))
        surface.blit(title, (self.x + 4, self.y + 3))

        for row_idx, row in enumerate([_SLOTS_ROW_WHITE, _SLOTS_ROW_BLACK]):
            for col_idx, (label, pt, col) in enumerate(row):
                self._draw_piece_slot(surface, row_idx, col_idx, pt, col)
            self._draw_erase_slot(surface, row_idx, 6)

    def _draw_piece_slot(self, surface, row, col_idx, piece_type, color):
        rect = self._slot_rect(row, col_idx)
        selected = self.selection == (piece_type, color)
        pygame.draw.rect(surface, self.BG_LIGHT, rect)
        pygame.draw.rect(
            surface,
            self.SELECTED_BORDER if selected else self.BORDER,
            rect, width=3 if selected else 1,
        )
        glyph = glyph_for(piece_type, color)
        text = self.font_glyph.render(glyph, True, (0, 0, 0))
        surface.blit(text, text.get_rect(center=rect.center))

    def _draw_erase_slot(self, surface, row, col_idx):
        rect = self._slot_rect(row, col_idx)
        selected = self.selection == (None, None)
        pygame.draw.rect(surface, self.ERASE_BG, rect)
        pygame.draw.rect(
            surface,
            self.SELECTED_BORDER if selected else self.BORDER,
            rect, width=3 if selected else 1,
        )
        label = self.font_label.render("erase", True, (100, 30, 30))
        surface.blit(label, label.get_rect(center=rect.center))
