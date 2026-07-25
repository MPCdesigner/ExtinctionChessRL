"""Settings panel — collapsible panel showing max-sim selector + model checkboxes.

Overlays the right side of the screen when open. Click "Settings" in the
controls row to toggle. Owns:

  - Max sim count (one of 1, 20, 50, 100, 200, 400, 800) — only sim counts
    <= this are actually evaluated. Higher = slower, but more coverage.
  - Per-model include-in-eval checkboxes.

The panel is stateless w.r.t. what the app does with the selections; it
just holds and displays them. The app pulls `selected_sim_counts()` and
`selected_model_indices()` before dispatching an evaluation.
"""

from __future__ import annotations

from typing import List, Tuple

import pygame

from .model_manager import SIM_OPTIONS


class SettingsPanel:
    WIDTH = 320
    PAD = 12
    ROW_HEIGHT = 24
    HEADER_HEIGHT = 26
    BG = (245, 245, 250)
    BORDER = (150, 150, 160)

    def __init__(self, x: int, y: int, height: int,
                 model_labels: List[str]):
        self.x = x
        self.y = y
        self.height = height

        self.open = False
        self.max_sims: int = SIM_OPTIONS[-1]     # default: allow up to 800
        self.model_labels = list(model_labels)
        self.model_enabled = [True] * len(model_labels)

        self.font_header = pygame.font.SysFont("Arial", 14, bold=True)
        self.font_row = pygame.font.SysFont("Arial", 13)

    # ── Toggle / accessors ─────────────────────────────────────────────────

    def toggle(self) -> None:
        self.open = not self.open

    def is_open(self) -> bool:
        return self.open

    def selected_sim_counts(self) -> List[int]:
        return [s for s in SIM_OPTIONS if s <= self.max_sims]

    def selected_model_indices(self) -> List[int]:
        return [i for i, on in enumerate(self.model_enabled) if on]

    # ── Layout helpers ─────────────────────────────────────────────────────

    def _panel_rect(self) -> pygame.Rect:
        return pygame.Rect(self.x, self.y, self.WIDTH, self.height)

    def _sim_row_rect(self, index: int) -> pygame.Rect:
        # Sim rows begin below the "Max sim count" header
        y = self.y + self.HEADER_HEIGHT * 2 + self.PAD + index * self.ROW_HEIGHT
        return pygame.Rect(
            self.x + self.PAD, y,
            self.WIDTH - self.PAD * 2, self.ROW_HEIGHT - 2,
        )

    def _models_start_y(self) -> int:
        sim_block_h = self.HEADER_HEIGHT * 2 + self.PAD + \
                      len(SIM_OPTIONS) * self.ROW_HEIGHT + self.PAD
        return self.y + sim_block_h

    def _model_row_rect(self, index: int) -> pygame.Rect:
        y = self._models_start_y() + self.HEADER_HEIGHT + index * self.ROW_HEIGHT
        return pygame.Rect(
            self.x + self.PAD, y,
            self.WIDTH - self.PAD * 2, self.ROW_HEIGHT - 2,
        )

    # ── Interaction ────────────────────────────────────────────────────────

    def handle_click(self, mx: int, my: int) -> bool:
        """Returns True if the click was consumed by this panel."""
        if not self.open:
            return False
        if not self._panel_rect().collidepoint(mx, my):
            return False

        # Sim row hit tests (radio behavior)
        for i, s in enumerate(SIM_OPTIONS):
            if self._sim_row_rect(i).collidepoint(mx, my):
                self.max_sims = s
                return True

        # Model row hit tests (checkbox toggle)
        for i in range(len(self.model_labels)):
            if self._model_row_rect(i).collidepoint(mx, my):
                self.model_enabled[i] = not self.model_enabled[i]
                return True

        return True  # consume the click even if it hit blank panel space

    # ── Drawing ────────────────────────────────────────────────────────────

    def draw(self, surface: pygame.Surface) -> None:
        if not self.open:
            return

        rect = self._panel_rect()
        pygame.draw.rect(surface, self.BG, rect)
        pygame.draw.rect(surface, self.BORDER, rect, width=1)

        # Header: max sims
        header = self.font_header.render("Max sim count", True, (30, 30, 30))
        surface.blit(header, (self.x + self.PAD,
                              self.y + self.PAD))
        sub = self.font_row.render(
            "(sim counts up to this will be evaluated)",
            True, (100, 100, 100))
        surface.blit(sub, (self.x + self.PAD,
                           self.y + self.PAD + self.HEADER_HEIGHT - 2))

        # Sim options (radio-style)
        for i, s in enumerate(SIM_OPTIONS):
            row = self._sim_row_rect(i)
            selected = (self.max_sims == s)
            # Radio circle
            cx = row.left + 8
            cy = row.centery
            pygame.draw.circle(surface, (80, 80, 80), (cx, cy), 6, width=1)
            if selected:
                pygame.draw.circle(surface, (30, 120, 220), (cx, cy), 3)
            label = self.font_row.render(
                "raw NN (1)" if s == 1 else f"{s} sims",
                True, (30, 30, 30))
            surface.blit(label, (row.left + 24, row.top + 3))

        # Header: models
        y = self._models_start_y()
        header = self.font_header.render(
            "Include models in evaluation", True, (30, 30, 30))
        surface.blit(header, (self.x + self.PAD, y + self.PAD - 8))

        # Model checkboxes
        for i, label in enumerate(self.model_labels):
            row = self._model_row_rect(i)
            checked = self.model_enabled[i]
            box = pygame.Rect(row.left + 4, row.centery - 6, 12, 12)
            pygame.draw.rect(surface, (250, 250, 250), box)
            pygame.draw.rect(surface, (80, 80, 80), box, width=1)
            if checked:
                pygame.draw.line(surface, (30, 120, 220),
                                 (box.left + 2, box.centery),
                                 (box.centerx - 1, box.bottom - 3), 2)
                pygame.draw.line(surface, (30, 120, 220),
                                 (box.centerx - 1, box.bottom - 3),
                                 (box.right - 1, box.top + 2), 2)
            text = self.font_row.render(label, True, (30, 30, 30))
            surface.blit(text, (row.left + 24, row.top + 3))
