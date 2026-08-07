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

from .model_manager import SIM_OPTIONS, SIM_UNLIMITED


def _sim_label(s: int) -> str:
    if s == 1:
        return "raw NN (1)"
    if s == SIM_UNLIMITED:
        return "unlimited"
    return f"{s} sims"


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

        # Display options — session-only (not persisted). Extensible: to add
        # a new toggle, append (attribute_name, label) here and everything
        # else (layout, click, draw) handles it automatically.
        self.highlight_endangered: bool = False
        self._display_options = [
            ("highlight_endangered", "Highlight endangered pieces"),
        ]

        # MCTS parameters — session-only. Defaults MATCH the tool's current
        # hardcoded behavior (c_puct=2.5 from mcts_search default; noise_weight=0
        # and dirichlet_alpha's default doesn't matter when noise_weight=0 —
        # we pick 0.3 which is what training uses so the value is meaningful
        # once noise is enabled). Click a param row to cycle its preset value.
        #
        # Presets are chosen to bracket the common range for each param:
        #   c_puct:          2.5 (default) → higher = more exploration
        #   noise_weight:    0.0 (default, no noise) → 0.25 = training default
        #   dirichlet_alpha: 0.3 (training default) → higher = more uniform
        self._c_puct_presets = [2.5, 3.0, 4.0, 5.0, 10.0]
        self._noise_weight_presets = [0.0, 0.15, 0.25, 0.35, 0.50]
        self._dirichlet_alpha_presets = [0.3, 0.6, 1.0, 2.0]
        self._c_puct_index = 0
        self._noise_weight_index = 0
        self._dirichlet_alpha_index = 0

        # Config-driven so adding a new param is a one-line change here.
        # Each tuple: (label, presets_attr_name, index_attr_name, format_str).
        self._mcts_params = [
            ("c_puct",          "_c_puct_presets",          "_c_puct_index",          "{:.2g}"),
            ("noise_weight",    "_noise_weight_presets",    "_noise_weight_index",    "{:.2f}"),
            ("dirichlet_alpha", "_dirichlet_alpha_presets", "_dirichlet_alpha_index", "{:.2g}"),
        ]

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

    def highlight_endangered_enabled(self) -> bool:
        return self.highlight_endangered

    # ── MCTS parameter accessors ──────────────────────────────────────────

    def get_c_puct(self) -> float:
        return self._c_puct_presets[self._c_puct_index]

    def get_noise_weight(self) -> float:
        return self._noise_weight_presets[self._noise_weight_index]

    def get_dirichlet_alpha(self) -> float:
        return self._dirichlet_alpha_presets[self._dirichlet_alpha_index]

    def _cycle_mcts_param(self, index_attr: str, presets_attr: str) -> None:
        """Advance the given param's index, wrapping around at the end."""
        cur = getattr(self, index_attr)
        n = len(getattr(self, presets_attr))
        setattr(self, index_attr, (cur + 1) % n)

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

    def _options_start_y(self) -> int:
        """Y position of the 'Display options' section (below models)."""
        models_end = self._models_start_y() + self.HEADER_HEIGHT + \
                     len(self.model_labels) * self.ROW_HEIGHT + self.PAD
        return models_end

    def _option_row_rect(self, index: int) -> pygame.Rect:
        y = self._options_start_y() + self.HEADER_HEIGHT + index * self.ROW_HEIGHT
        return pygame.Rect(
            self.x + self.PAD, y,
            self.WIDTH - self.PAD * 2, self.ROW_HEIGHT - 2,
        )

    def _mcts_start_y(self) -> int:
        """Y position of the 'MCTS parameters' section (below display options)."""
        opts_end = self._options_start_y() + self.HEADER_HEIGHT + \
                   len(self._display_options) * self.ROW_HEIGHT + self.PAD
        return opts_end

    def _mcts_row_rect(self, index: int) -> pygame.Rect:
        y = self._mcts_start_y() + self.HEADER_HEIGHT + index * self.ROW_HEIGHT
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

        # Display option row hit tests (checkbox toggle by attribute name)
        for i, (attr_name, _label) in enumerate(self._display_options):
            if self._option_row_rect(i).collidepoint(mx, my):
                setattr(self, attr_name, not getattr(self, attr_name))
                return True

        # MCTS parameter row hit tests (cycle to next preset value)
        for i, (_label, presets_attr, index_attr, _fmt) in enumerate(self._mcts_params):
            if self._mcts_row_rect(i).collidepoint(mx, my):
                self._cycle_mcts_param(index_attr, presets_attr)
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
            label = self.font_row.render(_sim_label(s), True, (30, 30, 30))
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
            self._draw_checkbox(surface, row, checked, label)

        # Header: display options
        y = self._options_start_y()
        header = self.font_header.render(
            "Display options", True, (30, 30, 30))
        surface.blit(header, (self.x + self.PAD, y + self.PAD - 8))

        # Display option checkboxes
        for i, (attr_name, label) in enumerate(self._display_options):
            row = self._option_row_rect(i)
            checked = bool(getattr(self, attr_name))
            self._draw_checkbox(surface, row, checked, label)

        # Header: MCTS parameters
        y = self._mcts_start_y()
        header = self.font_header.render(
            "MCTS parameters", True, (30, 30, 30))
        surface.blit(header, (self.x + self.PAD, y + self.PAD - 8))

        # MCTS param rows — click cycles the value. Non-default values are
        # highlighted so it's obvious at a glance when the tool is running
        # in exploration-diagnostic mode vs its normal deterministic defaults.
        for i, (label, presets_attr, index_attr, fmt) in enumerate(self._mcts_params):
            row = self._mcts_row_rect(i)
            value = getattr(self, presets_attr)[getattr(self, index_attr)]
            is_default = (getattr(self, index_attr) == 0)
            self._draw_mcts_row(surface, row, label, fmt.format(value),
                                is_default)

    def _draw_mcts_row(self, surface: pygame.Surface, row: pygame.Rect,
                       label: str, value_str: str, is_default: bool) -> None:
        """Draw a cyclable MCTS-param row: label on left, value pill on right."""
        # Label text
        label_surf = self.font_row.render(label, True, (30, 30, 30))
        surface.blit(label_surf, (row.left + 6, row.top + 3))

        # Value "pill" on the right — highlighted when non-default so the
        # user can see at a glance that they've changed something.
        pill_w = 68
        pill = pygame.Rect(row.right - pill_w - 4, row.top + 1,
                           pill_w, row.height - 2)
        pill_bg = (230, 230, 245) if is_default else (255, 235, 200)
        pill_border = (150, 150, 170) if is_default else (200, 150, 60)
        pygame.draw.rect(surface, pill_bg, pill)
        pygame.draw.rect(surface, pill_border, pill, width=1)
        val_surf = self.font_row.render(value_str, True, (30, 30, 30))
        surface.blit(val_surf, val_surf.get_rect(center=pill.center))

    def _draw_checkbox(self, surface: pygame.Surface, row: pygame.Rect,
                       checked: bool, label: str) -> None:
        """Shared draw helper for a labeled checkbox row (models + options)."""
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
