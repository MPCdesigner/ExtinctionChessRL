"""
Cluster-side cron entry for the Modal helper pipeline.

Runs one pass per cron tick (--once). Watches ~/extinction-chess/models/ for
new versioned checkpoints (az_iter_<N>_<XX>pct.pt) that match the
configured cadence. For each unprocessed iter it:
  1. Uploads the checkpoint to both Modal accounts' 'extinction-chess-ckpts'
     volumes.
  2. Invokes 'modal run modal_helper.py::run_helper' N times per account
     (N = config.default_num_helpers_per_account OR the override for that
     iter).
  3. Downloads each helper's output from Modal.
  4. Validates the .npz format (correct keys, shapes).
  5. Atomically moves each into ~/extinction-chess/replay_buffer/ as
     iter_<N>_modal<A|B><i>.npz.
  6. Marks the iter as processed in the state file.

Safety features:
  - Sentinel file (~/extinction-chess/MODAL_DISABLED): exit immediately.
  - Rate limit: max N processing events per rolling hour (default 3).
  - Every subprocess call has a timeout.
  - Format validation before commit to replay_buffer/.
  - Failed iters are NOT marked processed → next tick retries.

Cron entry (5-min tick):
  */5 * * * * /home/h74liang/.local/bin/python \\
      /home/h74liang/extinction-chess/modal_helper_cron.py --once \\
      >> /home/h74liang/modal_helper_cron.log 2>&1
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone


HOME              = os.path.expanduser("~")
EXT_DIR           = os.path.join(HOME, "extinction-chess")
MODELS_DIR        = os.path.join(EXT_DIR, "models")
REPLAY_DIR        = os.path.join(EXT_DIR, "replay_buffer")
CONFIG_FILE       = os.path.join(EXT_DIR, "modal_helper_config.json")
STATE_FILE        = os.path.join(EXT_DIR, "modal_helper_state.json")
SENTINEL_FILE     = os.path.join(EXT_DIR, "MODAL_DISABLED")
MODAL_HELPER_PY   = os.path.join(EXT_DIR, "modal_helper.py")
# Full path to modal CLI — cron doesn't inherit ~/.local/bin in PATH.
# Override with MODAL_BIN env var if pip installed elsewhere.
MODAL_BIN         = os.environ.get(
    "MODAL_BIN", os.path.join(HOME, ".local", "bin", "modal"))

LOG = "[modal-cron]"

# Modal profile names → filename tag ("A" or "B"). Order determines
# which account launches first for parallel invocation.
PROFILES = [("henry-account-a", "A"),
            ("henry-account-b", "B")]

# Default config, applied when the config file is absent or missing keys.
DEFAULT_CONFIG = {
    "cadence": 10,
    "default_num_helpers_per_account": 1,
    "default_num_games_per_helper": 200,
    "rate_limit_max_events_per_hour": 3,
    "iter_overrides": {},
}

CKPT_RE = re.compile(r"^az_iter_(\d+)_\d+pct\.pt$")
SUBPROC_TIMEOUT = 8000   # 2h20m — enough for one helper invocation


# ── Config / state IO ─────────────────────────────────────────────────────

def load_config():
    """Read config file, merge over defaults. Missing file → all defaults."""
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                user_cfg = json.load(f)
            cfg.update(user_cfg)
        except Exception as e:
            print(f"{LOG} WARNING: config file unreadable ({e}), using defaults",
                  flush=True)
    return cfg


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception as e:
            print(f"{LOG} WARNING: state file unreadable ({e}), starting fresh",
                  flush=True)
    return {"processed_iters": [], "recent_events": []}


def save_state(state):
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)


# ── Discovery ─────────────────────────────────────────────────────────────

def find_new_checkpoints(cfg, processed):
    """Return sorted list of (iter_num, filename) for un-processed iters
    matching the cadence."""
    cadence = int(cfg["cadence"])
    out = []
    for fname in os.listdir(MODELS_DIR):
        m = CKPT_RE.match(fname)
        if not m:
            continue
        n = int(m.group(1))
        if n % cadence == 0 and n not in processed:
            out.append((n, fname))
    return sorted(out)


# ── Rate limit ────────────────────────────────────────────────────────────

def within_rate_limit(cfg, recent_events):
    """Filter events to the last hour; return (allowed, pruned_list)."""
    cap = int(cfg["rate_limit_max_events_per_hour"])
    cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
    pruned = [x for x in recent_events
              if datetime.fromisoformat(x["ts"]) > cutoff]
    return len(pruned) < cap, pruned


# ── Modal subprocess wrappers ─────────────────────────────────────────────

def run_modal(profile, args, tag=""):
    """Fire a modal command via subprocess. Returns (rc, stdout, stderr)."""
    cmd = [MODAL_BIN, "--profile", profile] + args
    print(f"{LOG} {tag} $ {' '.join(cmd)}", flush=True)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=SUBPROC_TIMEOUT)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as e:
        return 124, "", f"TIMEOUT after {SUBPROC_TIMEOUT}s"


def upload_checkpoint(profile, ckpt_fname, tag):
    local_path = os.path.join(MODELS_DIR, ckpt_fname)
    return run_modal(profile, [
        "volume", "put", "extinction-chess-ckpts",
        local_path, f"/{ckpt_fname}", "--force",
    ], tag=f"[upload {tag}]")


def invoke_helper(profile, ckpt_fname, out_fname, num_games, tag):
    return run_modal(profile, [
        "run", MODAL_HELPER_PY + "::run_helper",
        "--checkpoint-filename", ckpt_fname,
        "--output-filename", out_fname,
        "--num-games", str(num_games),
    ], tag=f"[invoke {tag}]")


def download_output(profile, out_fname, local_dest, tag):
    return run_modal(profile, [
        "volume", "get", "extinction-chess-helper-outputs",
        f"/{out_fname}", local_dest, "--force",
    ], tag=f"[download {tag}]")


# ── Format validation ─────────────────────────────────────────────────────

def validate_npz(path):
    """Return (ok, reason) tuple. Loads the .npz and checks keys/shapes."""
    try:
        import numpy as np
        with np.load(path) as data:
            for k in ("boards", "policies", "values"):
                if k not in data:
                    return False, f"missing key '{k}'"
            b = data["boards"]
            p = data["policies"]
            v = data["values"]
            if b.ndim != 4 or b.shape[1:] != (115, 8, 8):
                return False, f"boards shape {b.shape}, expected (N,115,8,8)"
            if p.ndim != 2 or p.shape[1] != 4864:
                return False, f"policies shape {p.shape}, expected (N,4864)"
            if v.ndim != 1:
                return False, f"values shape {v.shape}, expected (N,)"
            if not (len(b) == len(p) == len(v)):
                return False, (f"length mismatch b={len(b)} p={len(p)} "
                               f"v={len(v)}")
            if len(b) == 0:
                return False, "empty arrays"
        return True, "ok"
    except Exception as e:
        return False, f"load error: {type(e).__name__}: {e}"


# ── Per-iter processing ───────────────────────────────────────────────────

def process_one_helper_launch(profile, tag_letter, launch_idx,
                              iter_num, ckpt_fname, num_games):
    """Execute one helper launch end-to-end. Returns True if the .npz was
    delivered to replay_buffer/, False otherwise."""
    tag         = f"{profile[:1].upper()}{launch_idx}"  # e.g. "A1"
    out_fname   = f"iter_{iter_num}_modal{tag_letter}{launch_idx}.npz"
    tmp_dest    = os.path.join(REPLAY_DIR, out_fname + ".part")
    final_dest  = os.path.join(REPLAY_DIR, out_fname)

    # 2. Invoke helper (blocks until Modal function returns)
    rc, _, se = invoke_helper(profile, ckpt_fname, out_fname, num_games, tag)
    if rc != 0:
        print(f"{LOG} [{tag}] helper failed rc={rc}: {se[-500:]}",
              flush=True)
        return False

    # 3. Download to .part
    rc, _, se = download_output(profile, out_fname, tmp_dest, tag)
    if rc != 0:
        print(f"{LOG} [{tag}] download failed rc={rc}: {se[-500:]}",
              flush=True)
        return False

    # 4. Format validation
    ok, reason = validate_npz(tmp_dest)
    if not ok:
        print(f"{LOG} [{tag}] FORMAT INVALID ({reason}), discarding "
              f"{tmp_dest}", flush=True)
        try:
            os.remove(tmp_dest)
        except OSError:
            pass
        return False

    # 5. Atomic rename into place
    os.replace(tmp_dest, final_dest)
    size_mb = os.path.getsize(final_dest) / (1024 * 1024)
    print(f"{LOG} [{tag}] delivered {out_fname} ({size_mb:.1f} MB)",
          flush=True)
    return True


def process_iter(iter_num, ckpt_fname, cfg, state):
    """Process one un-processed iter: upload + fire all helpers + collect
    results. Returns True if AT LEAST ONE helper landed a file."""
    overrides = cfg.get("iter_overrides", {}).get(str(iter_num), {})
    n_per_acct = int(overrides.get(
        "num_helpers_per_account",
        cfg["default_num_helpers_per_account"]))
    n_games = int(overrides.get(
        "num_games_per_helper",
        cfg["default_num_games_per_helper"]))
    if n_per_acct <= 0:
        print(f"{LOG} iter {iter_num}: num_helpers_per_account=0, skipping",
              flush=True)
        return True   # marks as processed so we don't try again

    total = n_per_acct * len(PROFILES)
    print(f"{LOG} iter {iter_num}: {total} helpers "
          f"({n_per_acct} per account × {len(PROFILES)} accounts) × "
          f"{n_games} games each = {total * n_games} bonus games",
          flush=True)

    # Rate limit check based on processing EVENTS (this iter counts as one)
    ok, recent = within_rate_limit(cfg, state["recent_events"])
    state["recent_events"] = recent
    if not ok:
        print(f"{LOG} rate-limited ({cfg['rate_limit_max_events_per_hour']}/h "
              f"cap), deferring iter {iter_num}", flush=True)
        return False
    state["recent_events"].append({
        "iter": iter_num,
        "ts": datetime.now(timezone.utc).isoformat(),
    })

    # 1. Upload checkpoint to both accounts (in parallel)
    def upload_task(idx):
        profile, tag_letter = PROFILES[idx]
        rc, _, se = upload_checkpoint(profile, ckpt_fname, tag_letter)
        if rc != 0:
            print(f"{LOG} [{tag_letter}] upload failed rc={rc}: {se[-500:]}",
                  flush=True)
            return False
        return True

    with ThreadPoolExecutor(max_workers=len(PROFILES)) as ex:
        upload_ok = list(ex.map(upload_task, range(len(PROFILES))))
    if not any(upload_ok):
        print(f"{LOG} iter {iter_num}: ALL uploads failed, aborting",
              flush=True)
        return False

    # 2. Fire helpers on both accounts in parallel. Each account may launch
    #    multiple helpers sequentially (Modal supports parallel per account
    #    but we serialize to keep resource cost predictable per account).
    def account_task(idx):
        if not upload_ok[idx]:
            return 0
        profile, tag_letter = PROFILES[idx]
        landed = 0
        for launch_idx in range(1, n_per_acct + 1):
            if process_one_helper_launch(
                    profile, tag_letter, launch_idx,
                    iter_num, ckpt_fname, n_games):
                landed += 1
        return landed

    with ThreadPoolExecutor(max_workers=len(PROFILES)) as ex:
        results = list(ex.map(account_task, range(len(PROFILES))))
    total_landed = sum(results)
    print(f"{LOG} iter {iter_num}: {total_landed}/{total} helpers delivered",
          flush=True)
    return total_landed > 0


# ── Main tick ─────────────────────────────────────────────────────────────

def tick():
    if os.path.exists(SENTINEL_FILE):
        print(f"{LOG} disabled via {SENTINEL_FILE}, exiting", flush=True)
        return

    os.makedirs(REPLAY_DIR, exist_ok=True)

    cfg = load_config()
    state = load_state()
    processed = set(state["processed_iters"])

    news = find_new_checkpoints(cfg, processed)
    if not news:
        # Quiet: nothing to do, don't spam the log every 5 min.
        return

    print(f"{LOG} tick: {len(news)} un-processed iter(s) found: "
          f"{[n for n, _ in news]}", flush=True)

    for iter_num, ckpt_fname in news:
        success = process_iter(iter_num, ckpt_fname, cfg, state)
        if success:
            processed.add(iter_num)
            state["processed_iters"] = sorted(processed)
        save_state(state)
        if not success:
            # Don't hammer a broken iter — let cron retry next tick.
            print(f"{LOG} iter {iter_num} not fully successful, will retry",
                  flush=True)
            break


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true",
                   help="Single tick then exit (for cron). Default: loop with "
                        "5-min sleep (for manual/tmux use).")
    args = p.parse_args()

    if args.once:
        tick()
    else:
        while True:
            try:
                tick()
            except Exception as e:
                print(f"{LOG} tick failed: {type(e).__name__}: {e}",
                      flush=True)
            time.sleep(300)


if __name__ == "__main__":
    main()
