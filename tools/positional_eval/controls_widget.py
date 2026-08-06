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
# Also make own package dir importable so `from value_dataset import ...` works
# when the tool is launched as `python -m tools.positional_eval`.
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from extinction_chess import Color  # noqa: E402

from value_dataset import VALUE_TAGS  # noqa: E402


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

# Value-drilling generation actions.
ACT_TOGGLE_VALUE_GEN = "toggle_value_gen"
ACT_SAVE_VALUE_WHITE = "save_value_white"     # +1
ACT_SAVE_VALUE_DRAW = "save_value_draw"       # 0
ACT_SAVE_VALUE_BLACK = "save_value_black"     # -1
# Tag chips: action name derived at runtime as ACT_TAG_PREFIX + tag_name.
# Handled internally by ControlsWidget (mutates active_tags); __main__ does
# not need to dispatch these itself.
ACT_TAG_PREFIX = "toggle_tag_"


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

        # ── Value-drilling generation controls ──────────────────────────────
        # Toggle button is always visible; the save/tag buttons only appear
        # when value_gen_enabled is True (see _layout).
        self.btn_value_gen_toggle = Button(
            "Value Dataset: OFF", ACT_TOGGLE_VALUE_GEN, width=170)
        self.btn_save_value_white = Button(
            "White wins +1", ACT_SAVE_VALUE_WHITE, width=110)
        self.btn_save_value_draw = Button(
            "Draw 0", ACT_SAVE_VALUE_DRAW, width=70)
        self.btn_save_value_black = Button(
            "Black wins -1", ACT_SAVE_VALUE_BLACK, width=110)
        # Tag chips — one per name in VALUE_TAGS. Their "action" is a
        # synthetic string; the widget consumes it internally in
        # handle_click() (mutates active_tags) rather than reporting up.
        self.tag_chips: List[Button] = [
            Button(f"[{tag}]", ACT_TAG_PREFIX + tag, width=110)
            for tag in VALUE_TAGS
        ]

        # State — mutated by handle_click for tag chips, by set_value_gen_state
        # for the toggle. UI reflects both immediately on next draw.
        self.value_gen_enabled: bool = False
        self.active_tags: List[str] = []  # subset of VALUE_TAGS

        self.last_action: Optional[str] = None

    # ── State refresh from the outside ──────────────────────────────────────

    def sync(self, mode_label: str, current_turn: Color,
             show_construction_controls: bool,
             can_step_back: bool, can_step_forward: bool,
             settings_open: bool,
             dataset_total: int = 0,
             dataset_session_count: int = 0) -> None:
        """Update button labels + enable/disable + highlight per app state.

        dataset_total / dataset_session_count are only shown when
        value_gen_enabled is True — they let the toggle button surface a
        one-line dataset summary without adding a separate UI row.
        """
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

        # Value-gen toggle label reflects state + dataset stats when ON.
        if self.value_gen_enabled:
            self.btn_value_gen_toggle.label = (
                f"Value Dataset: ON | {dataset_total} total"
                f" | {dataset_session_count} this session")
            # Auto-fit width to label so the row layout doesn't clip.
            self.btn_value_gen_toggle.width = max(
                260, 20 + self.font.size(self.btn_value_gen_toggle.label)[0])
        else:
            self.btn_value_gen_toggle.label = "Value Dataset: OFF"
            self.btn_value_gen_toggle.width = 170
        self.btn_value_gen_toggle.rect.width = self.btn_value_gen_toggle.width
        self.btn_value_gen_toggle.highlighted = self.value_gen_enabled

        # Tag chips highlighted iff active.
        for chip in self.tag_chips:
            tag = chip.action[len(ACT_TAG_PREFIX):]
            chip.highlighted = tag in self.active_tags

    # ── Layout ──────────────────────────────────────────────────────────────

    def _layout(self) -> List[Button]:
        # Base row: existing controls, then the value-gen toggle. When the
        # toggle is ON, save buttons + tag chips are appended after it.
        order = [
            self.btn_evaluate,
            self.btn_mode,
            self.btn_turn_w, self.btn_turn_b,
            self.btn_step_back, self.btn_step_fwd,
            self.btn_save, self.btn_load,
            self.btn_settings,
            self.btn_value_gen_toggle,
        ]
        if self.value_gen_enabled:
            order.extend([
                self.btn_save_value_white,
                self.btn_save_value_draw,
                self.btn_save_value_black,
            ])
            order.extend(self.tag_chips)

        # Left-align buttons in a row
        cx = self.x + self.PAD
        cy = self.y + (self.ROW_HEIGHT - 30) // 2
        for b in order:
            b.rect.topleft = (cx, cy)
            cx += b.width + self.PAD
        return order

    # ── Events ──────────────────────────────────────────────────────────────

    def handle_click(self, mx: int, my: int) -> Optional[str]:
        """If a button was clicked, set self.last_action and return it.

        Special cases handled INTERNALLY (return None so __main__ ignores):
          - The toggle button flips value_gen_enabled here (still reports the
            action too, so __main__ can e.g. lazily initialize the dataset).
          - Tag chip clicks toggle their tag in active_tags; NOT reported up.
        """
        self.last_action = None
        for b in self._layout():
            if b.hit(mx, my):
                action = b.action
                # Tag chip: toggle internally, don't report.
                if action.startswith(ACT_TAG_PREFIX):
                    tag = action[len(ACT_TAG_PREFIX):]
                    if tag in self.active_tags:
                        self.active_tags.remove(tag)
                    else:
                        self.active_tags.append(tag)
                    return None
                # Value-gen toggle: flip local state, then let __main__ know
                # so it can start a dataset session on first-ON.
                if action == ACT_TOGGLE_VALUE_GEN:
                    self.value_gen_enabled = not self.value_gen_enabled
                self.last_action = action
                return action
        return None

    # ── External state setters ──────────────────────────────────────────────

    def set_value_gen_enabled(self, enabled: bool) -> None:
        """External override, e.g. if __main__ wants to force-disable on error."""
        self.value_gen_enabled = enabled

    def get_active_tags(self) -> List[str]:
        """Snapshot of currently-active tags. Returned by copy — safe to keep."""
        return list(self.active_tags)

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
