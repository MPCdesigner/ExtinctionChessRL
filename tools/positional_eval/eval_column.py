"""Eval column widget — one column per model in the results area.

Each column stacks sections vertically, one per sim count. Each section shows:
  - Header line:   "N sims" (or "raw NN")   value=+0.42
  - Top-K moves table: "e2-e4     37.2%"
                       "d2-d4     22.5%"
                       ...

Only moves with non-zero probability/visits are shown, up to a cap (20).

Owner is responsible for:
  - Providing the current position + latest evaluation results dict
  - Managing horizontal scroll offset if more columns than fit on screen
  - Managing per-column vertical scroll offset
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pygame

from .model_manager import EvalResult


MAX_MOVES_SHOWN = 20  # cap; stop earlier when we hit a zero-prob entry


class EvalColumn:
    WIDTH = 240
    HEADER_HEIGHT = 30
    SECTION_HEADER_HEIGHT = 22
    MOVE_ROW_HEIGHT = 18
    SECTION_GAP = 8
    PAD = 8

    BG = (250, 250, 252)
    BORDER = (200, 200, 210)
    SECTION_BG = (240, 240, 245)
    HEADER_FG = (10, 10, 40)

    def __init__(self):
        self.font_header = pygame.font.SysFont("Arial", 15, bold=True)
        self.font_sub = pygame.font.SysFont("Arial", 12, bold=True)
        self.font_row = pygame.font.SysFont("Consolas,Menlo,Courier", 12)

    # ── Layout math ─────────────────────────────────────────────────────────

    def section_height(self, n_moves: int) -> int:
        rows = min(n_moves, MAX_MOVES_SHOWN)
        return self.SECTION_HEADER_HEIGHT + rows * self.MOVE_ROW_HEIGHT

    def column_total_height(self, results: Dict[int, EvalResult],
                            sim_counts_in_order: List[int]) -> int:
        h = self.HEADER_HEIGHT + self.PAD
        for s in sim_counts_in_order:
            r = results.get(s)
            n_moves = len(r.moves) if r else 0
            h += self.section_height(n_moves) + self.SECTION_GAP
        return h

    # ── Drawing ────────────────────────────────────────────────────────────

    def draw(self, surface: pygame.Surface, x: int, y: int, height: int,
             label: str,
             results: Dict[int, EvalResult],
             sim_counts_in_order: List[int],
             vert_scroll: int = 0) -> None:
        """Draw this column at (x, y). `height` is the visible height.

        `vert_scroll` is the y offset applied to content inside the column;
        content clipped to the column's visible area.
        """
        col_rect = pygame.Rect(x, y, self.WIDTH, height)

        # Solid background + border
        pygame.draw.rect(surface, self.BG, col_rect)
        pygame.draw.rect(surface, self.BORDER, col_rect, width=1)

        # Set a clip so contents don't spill outside the column
        prev_clip = surface.get_clip()
        surface.set_clip(col_rect)

        cy = y - vert_scroll

        # Header
        header_rect = pygame.Rect(x, cy, self.WIDTH, self.HEADER_HEIGHT)
        pygame.draw.rect(surface, (230, 235, 245), header_rect)
        text = self.font_header.render(label, True, self.HEADER_FG)
        surface.blit(text, text.get_rect(center=header_rect.center))
        cy += self.HEADER_HEIGHT + self.PAD

        # One section per sim count
        for s in sim_counts_in_order:
            r = results.get(s)
            cy = self._draw_section(surface, x, cy, s, r)

        # Restore clip
        surface.set_clip(prev_clip)

    def _draw_section(self, surface: pygame.Surface, x: int, cy: int,
                      sim_count: int, result: Optional[EvalResult]) -> int:
        """Draw one sim-count section. Returns the y position after the section."""
        n_moves = len(result.moves) if result else 0
        section_h = self.section_height(n_moves)

        # Section background
        sec_rect = pygame.Rect(x + 2, cy, self.WIDTH - 4, section_h)
        pygame.draw.rect(surface, self.SECTION_BG, sec_rect)

        # Section header: "raw NN" or "N sims" + value
        if result is None:
            header_text = self._sim_label(sim_count) + "   (pending)"
        else:
            header_text = (f"{self._sim_label(sim_count)}   "
                           f"value = {result.value:+.3f}")
        text = self.font_sub.render(header_text, True, (20, 40, 90))
        surface.blit(text, (x + self.PAD, cy + 3))
        cy += self.SECTION_HEADER_HEIGHT

        # Move rows
        if result is not None:
            rows_to_show = min(len(result.moves), MAX_MOVES_SHOWN)
            for i in range(rows_to_show):
                move, prob, visits = result.moves[i]
                if prob <= 0.0:
                    # early stop: no more meaningful entries
                    break
                move_str = self._format_move(move)
                pct_str = f"{prob * 100:5.1f}%"
                # Show visit count when available (MCTS runs), blank for raw NN
                visits_str = f"{visits:>4}" if visits > 0 else "   -"
                line = f"{move_str:<10} {visits_str}  {pct_str}"
                text = self.font_row.render(line, True, (30, 30, 30))
                surface.blit(text, (x + self.PAD, cy + 2))
                cy += self.MOVE_ROW_HEIGHT

        cy += self.SECTION_GAP
        return cy

    # ── Formatting helpers ──────────────────────────────────────────────────

    @staticmethod
    def _sim_label(sim_count: int) -> str:
        return "raw NN" if sim_count == 1 else f"{sim_count} sims"

    @staticmethod
    def _format_move(move) -> str:
        """Compact algebraic-ish move: 'e2-e4', 'e7-e8=Q'."""
        s = f"{move.from_pos.to_algebraic()}-{move.to_pos.to_algebraic()}"
        if getattr(move, "promotion", None):
            s += f"={move.promotion.value}"
        return s
