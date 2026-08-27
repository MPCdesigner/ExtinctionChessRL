"""Startup dialog — pick model, side, time control (with odds).

Uses tkinter (already a dependency via other tools). Returns a settings
dict on OK, or None if the user closed the window.

The dialog has three columns:
  - Model file picker (file dialog button + label showing choice)
  - "Play as" radio (White / Black)
  - Two time-control frames side by side: "Yours" and "Model"
    Each frame: base minutes + increment seconds (integer entries)
  - Presets row (5+3, 3+2, 10+5, 1+0) — click applies to BOTH sides
    (odds is set by hand-editing one side after)
  - Start button (disabled until model chosen)
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional

import tkinter as tk
from tkinter import filedialog, ttk


PRESETS = [
    ("1+0",  1, 0),
    ("3+2",  3, 2),
    ("5+3",  5, 3),
    ("10+5", 10, 5),
]


def show_startup_dialog(
    default_model_path: Optional[str] = None,
    default_side: str = "W",
    default_your_min: int = 5,
    default_your_inc: int = 3,
    default_model_min: int = 5,
    default_model_inc: int = 3,
) -> Optional[Dict]:
    """Show the modal startup dialog. Returns settings dict or None.

    Returned dict:
        {
          "model_path": str,
          "user_side": "W" | "B",
          "user_base_seconds": int,
          "user_increment_seconds": int,
          "model_base_seconds": int,
          "model_increment_seconds": int,
        }

    All defaults are used to PRE-FILL the dialog when it opens — this is
    what the "New Game" button uses to restart with the same settings.
    """
    result: Dict = {}

    root = tk.Tk()
    root.title("Extinction Chess — Play Timed Match")
    root.attributes("-topmost", True)
    root.resizable(False, False)

    # ── Model file picker ──────────────────────────────────────────────
    model_path_var = tk.StringVar(value=default_model_path or "")

    model_frame = ttk.LabelFrame(root, text="Opponent (model)", padding=10)
    model_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 5),
                     sticky="ew")

    def pick_model():
        p = filedialog.askopenfilename(
            parent=root,
            title="Choose model checkpoint",
            filetypes=[("PyTorch checkpoint", "*.pt")],
            initialdir=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", "models"),
        )
        if p:
            model_path_var.set(p)
            _refresh_start_state()

    ttk.Button(model_frame, text="Choose .pt file...",
               command=pick_model).grid(row=0, column=0, padx=(0, 8))
    model_label = ttk.Label(model_frame, textvariable=model_path_var,
                            width=48, anchor="w", wraplength=380)
    model_label.grid(row=0, column=1, sticky="w")

    # ── Play-as side ───────────────────────────────────────────────────
    side_var = tk.StringVar(value=default_side)
    side_frame = ttk.LabelFrame(root, text="Play as", padding=10)
    side_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
    ttk.Radiobutton(side_frame, text="White (move first)",
                    variable=side_var, value="W").grid(row=0, column=0, sticky="w")
    ttk.Radiobutton(side_frame, text="Black",
                    variable=side_var, value="B").grid(row=1, column=0, sticky="w")

    # ── Time controls ──────────────────────────────────────────────────
    tc_frame = ttk.LabelFrame(root, text="Time control (odds supported)",
                              padding=10)
    tc_frame.grid(row=1, column=1, padx=10, pady=5, sticky="nsew")

    # Your side
    ttk.Label(tc_frame, text="Yours:").grid(row=0, column=0, sticky="e")
    your_base_var = tk.IntVar(value=default_your_min)
    your_inc_var = tk.IntVar(value=default_your_inc)
    ttk.Spinbox(tc_frame, from_=0, to=180, width=5,
                textvariable=your_base_var).grid(row=0, column=1, padx=4)
    ttk.Label(tc_frame, text="min +").grid(row=0, column=2, sticky="w")
    ttk.Spinbox(tc_frame, from_=0, to=60, width=5,
                textvariable=your_inc_var).grid(row=0, column=3, padx=4)
    ttk.Label(tc_frame, text="sec").grid(row=0, column=4, sticky="w")

    # Model side
    ttk.Label(tc_frame, text="Model:").grid(row=1, column=0, sticky="e", pady=(6, 0))
    model_base_var = tk.IntVar(value=default_model_min)
    model_inc_var = tk.IntVar(value=default_model_inc)
    ttk.Spinbox(tc_frame, from_=0, to=180, width=5,
                textvariable=model_base_var).grid(row=1, column=1, padx=4, pady=(6, 0))
    ttk.Label(tc_frame, text="min +").grid(row=1, column=2, sticky="w", pady=(6, 0))
    ttk.Spinbox(tc_frame, from_=0, to=60, width=5,
                textvariable=model_inc_var).grid(row=1, column=3, padx=4, pady=(6, 0))
    ttk.Label(tc_frame, text="sec").grid(row=1, column=4, sticky="w", pady=(6, 0))

    # Presets — apply to both sides
    preset_frame = ttk.Frame(tc_frame)
    preset_frame.grid(row=2, column=0, columnspan=5, pady=(10, 0))

    def apply_preset(mins: int, secs: int):
        your_base_var.set(mins)
        your_inc_var.set(secs)
        model_base_var.set(mins)
        model_inc_var.set(secs)

    ttk.Label(preset_frame, text="Presets (both sides):").grid(row=0, column=0,
                                                                sticky="w")
    for i, (label, mins, secs) in enumerate(PRESETS):
        ttk.Button(preset_frame, text=label, width=6,
                   command=lambda m=mins, s=secs: apply_preset(m, s)).grid(
            row=0, column=i + 1, padx=2)

    # ── Start / cancel ─────────────────────────────────────────────────
    button_frame = ttk.Frame(root)
    button_frame.grid(row=2, column=0, columnspan=2, pady=(10, 10))

    def on_start():
        result.update({
            "model_path": model_path_var.get(),
            "user_side": side_var.get(),
            "user_base_seconds": int(your_base_var.get()) * 60,
            "user_increment_seconds": int(your_inc_var.get()),
            "model_base_seconds": int(model_base_var.get()) * 60,
            "model_increment_seconds": int(model_inc_var.get()),
        })
        root.destroy()

    def on_cancel():
        result.clear()
        root.destroy()

    start_btn = ttk.Button(button_frame, text="Start game", command=on_start)
    start_btn.grid(row=0, column=0, padx=4)
    ttk.Button(button_frame, text="Cancel", command=on_cancel).grid(row=0, column=1, padx=4)

    def _refresh_start_state():
        start_btn.configure(state=("normal" if model_path_var.get() else "disabled"))

    _refresh_start_state()

    root.mainloop()
    return result if result else None
