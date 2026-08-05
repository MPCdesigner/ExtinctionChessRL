"""Value-target dataset for hand-authored training positions.

Manages a persistent JSON file (default: dataset/value_targets.json at project
root) that accumulates positions the user has curated as value-only training
examples. Each entry carries:
  - the full position dict from PositionState.to_dict() (preserves move_history,
    castling, en passant, halfmove clock — everything needed to reconstruct
    the NN's 115-channel input with correct history planes)
  - value target in {-1, 0, +1} (White's perspective, standard chess convention)
  - tags (record-keeping only — do not affect training)
  - session_id (ISO timestamp of the tool run that added this entry — useful
    for auditing and rolling back a bad session without losing older work)
  - added_at (ISO timestamp of the specific save)

Design notes:
  - Duplicates ARE allowed. The same position may be saved multiple times
    (possibly with different values) — get_value_breakdown() lets the UI
    show a warning before adding a duplicate. See docstring on that method.
  - Auto-saves on every add. JSON is small; disk write cost is negligible.
  - Position identity: matched on the SEMANTIC position (board layout +
    to_move + castling flags + en_passant + halfmove_clock), NOT on
    move_history. Two different game paths to the same position count as
    matching. Uses _canonical_position_key().
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


# Hardcoded set of recognised tags. To add more, append here and the UI
# will pick them up automatically (see controls_widget.py VALUE_TAGS use).
VALUE_TAGS: List[str] = ["forced_win", "material_adv"]


def _iso_now() -> str:
    """UTC-naive ISO timestamp, seconds precision, safe for filenames."""
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _canonical_position_key(pos: Dict[str, Any]) -> Tuple:
    """Semantic identity for a position — ignores move_history/step_position.

    Two positions are "the same" if their piece layout, side-to-move, castling
    rights (via has_moved per piece), en-passant target, and halfmove clock
    match. Different games reaching the same board state produce the same key.
    """
    board_key = tuple(
        tuple(
            None if cell is None else
            (cell.get("type"), cell.get("color"), cell.get("has_moved", True))
            for cell in row
        )
        for row in pos.get("board", [])
    )
    ep = pos.get("en_passant")
    ep_key = tuple(ep) if ep else None
    return (
        board_key,
        pos.get("to_move"),
        ep_key,
        pos.get("halfmove_clock", 0),
    )


class ValueDataset:
    """Persistent value-target dataset with session tracking."""

    def __init__(self, path: str):
        self.path = path
        self.sessions: List[Dict[str, Any]] = []
        self.entries: List[Dict[str, Any]] = []
        self._current_session_id: Optional[str] = None
        self._load()

    # ── Session lifecycle ───────────────────────────────────────────────────

    def start_session(self) -> str:
        """Register a new session for this tool run. Idempotent within a run —
        subsequent calls in the same ValueDataset instance return the same ID."""
        if self._current_session_id is not None:
            return self._current_session_id
        sid = _iso_now()
        # If a session with this ID already exists (extremely rare — same-second
        # tool restart), disambiguate with a suffix.
        existing_ids = {s["session_id"] for s in self.sessions}
        base = sid
        suffix = 1
        while sid in existing_ids:
            sid = f"{base}_{suffix}"
            suffix += 1
        self.sessions.append({
            "session_id": sid,
            "started_at": sid,
            "entry_count": 0,
        })
        self._current_session_id = sid
        self._save()
        return sid

    def current_session_id(self) -> Optional[str]:
        return self._current_session_id

    # ── Entry operations ────────────────────────────────────────────────────

    def add_entry(self, position: Dict[str, Any], value: int,
                  tags: List[str]) -> Dict[str, Any]:
        """Append a new entry. Auto-saves. Returns the entry dict."""
        if value not in (-1, 0, 1):
            raise ValueError(f"value must be one of -1, 0, +1 (got {value!r})")
        for t in tags:
            if t not in VALUE_TAGS:
                # Not fatal — tags may drift over time — but log so drift is
                # visible. Persisted anyway.
                print(f"[value_dataset] warning: unknown tag {t!r} "
                      f"(known: {VALUE_TAGS})")
        if self._current_session_id is None:
            self.start_session()
        entry = {
            "value": int(value),
            "tags": list(tags),
            "session_id": self._current_session_id,
            "added_at": _iso_now(),
            "position": position,
        }
        self.entries.append(entry)
        # Update session count in-place
        for s in self.sessions:
            if s["session_id"] == self._current_session_id:
                s["entry_count"] = s.get("entry_count", 0) + 1
                break
        self._save()
        return entry

    # ── Queries ─────────────────────────────────────────────────────────────

    def get_matches(self, position: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return existing entries whose position semantically matches."""
        target = _canonical_position_key(position)
        return [e for e in self.entries
                if _canonical_position_key(e["position"]) == target]

    def get_value_breakdown(self, position: Dict[str, Any]) -> Dict[int, int]:
        """{-1: n, 0: m, +1: k} count of prior saves of this position.

        Used by the UI to render the duplicate-save warning:
          "This position was saved 3× with value +1, 0× with value 0, 1× with -1.
           Save again with +1?"
        Absent keys default to 0 in the caller's mental model, but the returned
        dict always contains all three keys for straightforward formatting.
        """
        counts = {-1: 0, 0: 0, 1: 0}
        for e in self.get_matches(position):
            v = int(e["value"])
            if v in counts:
                counts[v] += 1
        return counts

    def total_count(self) -> int:
        return len(self.entries)

    def session_summary(self) -> List[Dict[str, Any]]:
        """Return a stable-order list of session info dicts. Not a live view;
        make a fresh call after add_entry to see updated counts."""
        return [dict(s) for s in self.sessions]

    # ── Persistence ─────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            # Don't destroy an unreadable file — just start empty and warn.
            # User can inspect the file themselves.
            print(f"[value_dataset] warning: could not load {self.path}: "
                  f"{type(e).__name__}: {e}. Starting fresh (existing file "
                  f"will be OVERWRITTEN on next save).")
            return
        self.sessions = list(data.get("sessions", []))
        self.entries = list(data.get("entries", []))

    def _save(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                {"sessions": self.sessions, "entries": self.entries},
                f, indent=2,
            )
        # Atomic replace so a crash mid-write can't corrupt the dataset.
        os.replace(tmp, self.path)


# ── Convenience ────────────────────────────────────────────────────────────

def default_dataset_path() -> str:
    """Default location: dataset/value_targets.json at project root."""
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(here, "..", ".."))
    return os.path.join(project_root, "dataset", "value_targets.json")
