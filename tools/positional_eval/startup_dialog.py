"""Startup file-picker dialog for the positional evaluation tool.

Uses tkinter (Python stdlib) so no extra dependencies. Opens a multi-select
dialog anchored to the project's models/ directory. Returns the list of
selected paths; empty list means user cancelled.
"""

from __future__ import annotations

import os
from typing import List


def _default_models_dir() -> str:
    """Return absolute path to the project's models/ directory if it exists."""
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(here, "..", ".."))
    models_dir = os.path.join(project_root, "models")
    return models_dir if os.path.isdir(models_dir) else project_root


def pick_models() -> List[str]:
    """Show a multi-select file dialog. Returns absolute paths (may be empty)."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()  # hide the root window; only show the dialog
    root.attributes("-topmost", True)

    paths = filedialog.askopenfilenames(
        parent=root,
        title="Select model checkpoints to load (.pt) — pick as many as you want",
        initialdir=_default_models_dir(),
        filetypes=[("PyTorch checkpoints", "*.pt"), ("All files", "*.*")],
    )
    root.destroy()

    if not paths:
        return []
    # tkinter returns a tuple on some platforms; normalize to list of str
    return [os.path.abspath(p) for p in paths]


if __name__ == "__main__":
    # Quick manual smoke test
    picked = pick_models()
    print(f"Picked {len(picked)} model(s):")
    for p in picked:
        print(f"  {p}")
