"""Controls widget — the row of buttons across the top of the UI.

Buttons exposed:
  - Evaluate           run selected models at chosen sim counts on current position
  - Mode toggle        switch between Game Setup and Construction modes
  - Turn: W / Turn: B  (visible only in Construction mode) — set whose turn
  - Step ←             go back one move in Game Setup history
  - Step →             go forward one move in Game Setup history
  - Save               write current position to JSON
  - Load               load position from JSON
  - Settings           show/hide the settings panel

The widget owns button geometry and click detection; it does not perform
the actions itself. The main loop reads `last_action` after handling
events to decide what to do.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple

import pygame

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from extinction_chess import Color  # noqa: E402


# Actions the widget can report back to the main loop.
ACT_EVALUATE = "evaluate"
ACT_TOGGLE_MODE = "toggle_mode"
ACT_SET_TURN_W = "set_turn_white"
ACT_SET_TURN_B = "set_turn_black"
ACT_STEP_BACK = "step_back"
ACT_STEP_FWD = "step_forward"
ACT_SAVE = "save"
ACT_LOAD = "load"
ACT_TOGGLE_SETTINGS = "toggle_settings"


class Button:
    """Minimal rectangular button. Immediate-mode style."""
    def __init__(self, label: str, action: str, width: int = 96, height: int = 30):
        self.label = label
        self.action = action
        self.width = width
        self.height = height
        self.rect = pygame.Rect(0, 0, width, height)
        self.enabled = True
        self.highlighted = False

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        if not self.enabled:
            bg = (200, 200, 200)
            fg = (140, 140, 140)
        elif self.highlighted:
            bg = (210, 230, 250)
            fg = (0, 0, 0)
        else:
            bg = (240, 240, 240)
            fg = (0, 0, 0)
        pygame.draw.rect(surface, bg, self.rect)
        pygame.draw.rect(surface, (60, 60, 60), self.rect, width=1)
        text = font.render(self.label, True, fg)
        surface.blit(text, text.get_rect(center=self.rect.center))

    def hit(self, mx: int, my: int) -> bool:
        return self.enabled and self.rect.collidepoint(mx, my)


class ControlsWidget:
    ROW_HEIGHT = 40
    PAD = 6

    def __init__(self, x: int, y: int, total_width: int):
        self.x = x
        self.y = y
        self.total_width = total_width
        self.font = pygame.font.SysFont("Arial", 13)

        self.btn_evaluate = Button("Evaluate", ACT_EVALUATE, width=100)
        self.btn_mode = Button("Mode: Game Setup", ACT_TOGGLE_MODE, width=170)
        self.btn_turn_w = Button("Turn: W", ACT_SET_TURN_W, width=80)
        self.btn_turn_b = Button("Turn: B", ACT_SET_TURN_B, width=80)
        self.btn_step_back = Button("Step ←", ACT_STEP_BACK, width=70)
        self.btn_step_fwd  = Button("Step →", ACT_STEP_FWD,  width=70)
        self.btn_save = Button("Save", ACT_SAVE, width=60)
        self.btn_load = Button("Load", ACT_LOAD, width=60)
        self.btn_settings = Button("Settings", ACT_TOGGLE_SETTINGS, width=90)

        self.last_action: Optional[str] = None

    # ── State refresh from the outside ──────────────────────────────────────

    def sync(self, mode_label: str, current_turn: Color,
             show_construction_controls: bool,
             can_step_back: bool, can_step_forward: bool,
             settings_open: bool) -> None:
        """Update button labels + enable/disable + highlight per app state."""
        self.btn_mode.label = f"Mode: {mode_label}"

        self.btn_turn_w.enabled = show_construction_controls
        self.btn_turn_b.enabled = show_construction_controls
        self.btn_turn_w.highlighted = (
            show_construction_controls and current_turn == Color.WHITE)
        self.btn_turn_b.highlighted = (
            show_construction_controls and current_turn == Color.BLACK)

        self.btn_step_back.enabled = can_step_back
        self.btn_step_fwd.enabled = can_step_forward

        self.btn_settings.highlighted = settings_open

    # ── Layout ──────────────────────────────────────────────────────────────

    def _layout(self) -> List[Button]:
        order = [
            self.btn_evaluate,
            self.btn_mode,
            self.btn_turn_w, self.btn_turn_b,
            self.btn_step_back, self.btn_step_fwd,
            self.btn_save, self.btn_load,
            self.btn_settings,
        ]
        # Left-align buttons in a row
        cx = self.x + self.PAD
        cy = self.y + (self.ROW_HEIGHT - 30) // 2
        for b in order:
            b.rect.topleft = (cx, cy)
            cx += b.width + self.PAD
        return order

    # ── Events ──────────────────────────────────────────────────────────────

    def handle_click(self, mx: int, my: int) -> Optional[str]:
        """If a button was clicked, set self.last_action and return it."""
        self.last_action = None
        for b in self._layout():
            if b.hit(mx, my):
                self.last_action = b.action
                return b.action
        return None

    # ── Drawing ─────────────────────────────────────────────────────────────

    def draw(self, surface: pygame.Surface) -> None:
        # Row background
        pygame.draw.rect(
            surface, (250, 250, 250),
            pygame.Rect(self.x, self.y, self.total_width, self.ROW_HEIGHT))
        pygame.draw.line(
            surface, (200, 200, 200),
            (self.x, self.y + self.ROW_HEIGHT),
            (self.x + self.total_width, self.y + self.ROW_HEIGHT))
        for b in self._layout():
            b.draw(surface, self.font)
