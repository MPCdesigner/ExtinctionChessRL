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
    default_your_delay: int = 0,
    default_model_min: int = 5,
    default_model_inc: int = 3,
    default_model_delay: int = 0,
    default_tactical_level: str = "basic",
    default_c_puct: float = 2.5,
    default_dirichlet_alpha: float = 0.0,
    default_noise_weight: float = 0.0,
    default_sim_ceiling: int = 800,
) -> Optional[Dict]:
    """Show the modal startup dialog. Returns settings dict or None.

    Returned dict:
        {
          "model_path": str,
          "user_side": "W" | "B",
          "user_base_seconds": int,
          "user_increment_seconds": int,
          "user_delay_seconds": int,
          "model_base_seconds": int,
          "model_increment_seconds": int,
          "model_delay_seconds": int,
          "tactical_level": "off" | "basic" | "advanced",
          "c_puct": float,
          "dirichlet_alpha": float,
          "noise_weight": float,
          "sim_ceiling": int,
        }

    tactical_level explanation:
      off       — no forced shortcuts. Model must find mate-in-1 through
                  its value head + MCTS visits like any other move.
      basic     — force mate-in-1 (extinction-in-1) if available. Same as
                  training. Distributes root sims across all winning moves.
      advanced  — basic + loss avoidance. If MCTS's top pick would give
                  the opponent a mate-in-1 reply AND a move exists that
                  doesn't, prefer the safe one (highest-visited such move).
                  Was the training-time shortcut pre-iter-101; removed to
                  force the value head to learn "don't step here". OK to
                  re-enable during inference — no gradient at play time.

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

    # ── Time controls (base + increment + delay per side) ─────────────
    tc_frame = ttk.LabelFrame(
        root, text="Time control (odds + Bronstein delay supported)",
        padding=10)
    tc_frame.grid(row=1, column=1, padx=10, pady=5, sticky="nsew")

    # Header row so the two extra spinboxes are self-explanatory
    ttk.Label(tc_frame, text="", width=6).grid(row=0, column=0)
    ttk.Label(tc_frame, text="base (min)").grid(row=0, column=1, columnspan=2)
    ttk.Label(tc_frame, text="+ inc (sec)").grid(row=0, column=3, columnspan=2)
    ttk.Label(tc_frame, text="+ delay (sec)").grid(row=0, column=5, columnspan=2)

    # Your side
    ttk.Label(tc_frame, text="Yours:").grid(row=1, column=0, sticky="e", pady=(6, 0))
    your_base_var = tk.IntVar(value=default_your_min)
    your_inc_var = tk.IntVar(value=default_your_inc)
    your_delay_var = tk.IntVar(value=default_your_delay)
    ttk.Spinbox(tc_frame, from_=0, to=180, width=5,
                textvariable=your_base_var).grid(row=1, column=1, padx=4, pady=(6, 0))
    ttk.Label(tc_frame, text="min").grid(row=1, column=2, sticky="w", pady=(6, 0))
    ttk.Spinbox(tc_frame, from_=0, to=60, width=5,
                textvariable=your_inc_var).grid(row=1, column=3, padx=4, pady=(6, 0))
    ttk.Label(tc_frame, text="sec").grid(row=1, column=4, sticky="w", pady=(6, 0))
    ttk.Spinbox(tc_frame, from_=0, to=60, width=5,
                textvariable=your_delay_var).grid(row=1, column=5, padx=4, pady=(6, 0))
    ttk.Label(tc_frame, text="sec").grid(row=1, column=6, sticky="w", pady=(6, 0))

    # Model side
    ttk.Label(tc_frame, text="Model:").grid(row=2, column=0, sticky="e", pady=(6, 0))
    model_base_var = tk.IntVar(value=default_model_min)
    model_inc_var = tk.IntVar(value=default_model_inc)
    model_delay_var = tk.IntVar(value=default_model_delay)
    ttk.Spinbox(tc_frame, from_=0, to=180, width=5,
                textvariable=model_base_var).grid(row=2, column=1, padx=4, pady=(6, 0))
    ttk.Label(tc_frame, text="min").grid(row=2, column=2, sticky="w", pady=(6, 0))
    ttk.Spinbox(tc_frame, from_=0, to=60, width=5,
                textvariable=model_inc_var).grid(row=2, column=3, padx=4, pady=(6, 0))
    ttk.Label(tc_frame, text="sec").grid(row=2, column=4, sticky="w", pady=(6, 0))
    ttk.Spinbox(tc_frame, from_=0, to=60, width=5,
                textvariable=model_delay_var).grid(row=2, column=5, padx=4, pady=(6, 0))
    ttk.Label(tc_frame, text="sec").grid(row=2, column=6, sticky="w", pady=(6, 0))

    # Presets — apply base+increment to both sides; zero out delay
    preset_frame = ttk.Frame(tc_frame)
    preset_frame.grid(row=3, column=0, columnspan=7, pady=(10, 0))

    def apply_preset(mins: int, secs: int):
        your_base_var.set(mins)
        your_inc_var.set(secs)
        your_delay_var.set(0)
        model_base_var.set(mins)
        model_inc_var.set(secs)
        model_delay_var.set(0)

    ttk.Label(preset_frame, text="Presets (both sides):").grid(row=0, column=0,
                                                                sticky="w")
    for i, (label, mins, secs) in enumerate(PRESETS):
        ttk.Button(preset_frame, text=label, width=6,
                   command=lambda m=mins, s=secs: apply_preset(m, s)).grid(
            row=0, column=i + 1, padx=2)

    # ── Tactical shortcuts ─────────────────────────────────────────────
    tactical_var = tk.StringVar(value=default_tactical_level)
    tactical_frame = ttk.LabelFrame(root, text="Tactical shortcuts", padding=10)
    tactical_frame.grid(row=2, column=0, padx=10, pady=5, sticky="nsew")
    ttk.Radiobutton(
        tactical_frame,
        text="Off — model must find mate-in-1 via MCTS",
        variable=tactical_var, value="off",
    ).grid(row=0, column=0, sticky="w")
    ttk.Radiobutton(
        tactical_frame,
        text="Basic — force mate-in-1 (same as training)",
        variable=tactical_var, value="basic",
    ).grid(row=1, column=0, sticky="w")
    ttk.Radiobutton(
        tactical_frame,
        text="Advanced — basic + avoid opponent mate-in-1 (depth 2)",
        variable=tactical_var, value="advanced",
    ).grid(row=2, column=0, sticky="w")

    # ── MCTS parameters (advanced knobs) ───────────────────────────────
    # Defaults match compare_extensive.py (deterministic benchmark MCTS):
    # c_puct=2.5, dirichlet_alpha=0, noise_weight=0. Sim ceiling is the
    # early-play threshold — if the ponder tree already has this many
    # sims when the model's turn starts, play immediately without burning
    # the rest of the time budget. Default 800 matches training's per-move
    # sim count so we play at "training-equivalent depth" and bank time.
    mcts_frame = ttk.LabelFrame(root, text="MCTS parameters", padding=10)
    mcts_frame.grid(row=2, column=1, padx=10, pady=5, sticky="nsew")

    c_puct_var = tk.DoubleVar(value=default_c_puct)
    dirichlet_var = tk.DoubleVar(value=default_dirichlet_alpha)
    noise_var = tk.DoubleVar(value=default_noise_weight)
    sim_ceiling_var = tk.IntVar(value=default_sim_ceiling)

    ttk.Label(mcts_frame, text="c_puct:").grid(row=0, column=0, sticky="e")
    ttk.Spinbox(mcts_frame, from_=0.5, to=100.0, increment=0.5, width=7,
                textvariable=c_puct_var).grid(row=0, column=1, padx=4)

    ttk.Label(mcts_frame, text="dirichlet_alpha:").grid(row=1, column=0, sticky="e", pady=(6, 0))
    ttk.Spinbox(mcts_frame, from_=0.0, to=10.0, increment=0.1, width=7,
                textvariable=dirichlet_var).grid(row=1, column=1, padx=4, pady=(6, 0))

    ttk.Label(mcts_frame, text="noise_weight:").grid(row=2, column=0, sticky="e", pady=(6, 0))
    ttk.Spinbox(mcts_frame, from_=0.0, to=1.0, increment=0.05, width=7,
                textvariable=noise_var).grid(row=2, column=1, padx=4, pady=(6, 0))

    ttk.Label(mcts_frame, text="sim ceiling:").grid(row=3, column=0, sticky="e", pady=(6, 0))
    ttk.Spinbox(mcts_frame, from_=50, to=100000, increment=100, width=7,
                textvariable=sim_ceiling_var).grid(row=3, column=1, padx=4, pady=(6, 0))

    # ── Start / cancel ─────────────────────────────────────────────────
    button_frame = ttk.Frame(root)
    button_frame.grid(row=3, column=0, columnspan=2, pady=(10, 10))

    def on_start():
        result.update({
            "model_path": model_path_var.get(),
            "user_side": side_var.get(),
            "user_base_seconds": int(your_base_var.get()) * 60,
            "user_increment_seconds": int(your_inc_var.get()),
            "user_delay_seconds": int(your_delay_var.get()),
            "model_base_seconds": int(model_base_var.get()) * 60,
            "model_increment_seconds": int(model_inc_var.get()),
            "model_delay_seconds": int(model_delay_var.get()),
            "tactical_level": tactical_var.get(),
            "c_puct": float(c_puct_var.get()),
            "dirichlet_alpha": float(dirichlet_var.get()),
            "noise_weight": float(noise_var.get()),
            "sim_ceiling": int(sim_ceiling_var.get()),
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
