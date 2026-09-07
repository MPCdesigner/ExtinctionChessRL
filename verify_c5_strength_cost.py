"""Measure the strength cost of the C5 Python-fallback encoder bug.

C5 = StateEncoder's Python fallback never populates 96 history planes
(12-107) or 4 castling planes (110-113). At inference on any target
without a built _ext_chess extension (local pygame, Modal container),
the net sees 100 of its 115 input planes as constant zero.

This script asks: does the net actually LEAN on those planes?

Method:
  1. Load real (C++-encoded, all 115 planes populated) positions from
     a training buffer.
  2. Load az_iter1040.
  3. For each position:
       (a) Forward pass with full tensor        -> (policy_full, value_full)
       (b) Forward pass with planes 12-107 and
           110-113 zeroed (Python-fallback)     -> (policy_z, value_z)
  4. Compare policy top-K + value across many positions. Report
     aggregate divergence.

Interpretation:
  - value |delta| mean < 0.05, top-1 agreement > 90%
      -> net barely uses those planes. C5 is cosmetic; fixing it
         won't measurably change engine strength.
  - value |delta| mean > 0.15 OR top-1 agreement < 70%
      -> net leans on those planes. Deployment strength is well
         below trained ceiling. Build C++ into Modal before shipping.
  - anything in between -> real but modest cost.

Run:
    python verify_c5_strength_cost.py
"""
import sys
import numpy as np

sys.path.insert(0, 'src')
import torch
from alphazero import AlphaZeroNet

BUFFER = 'replay_buffer/iter_696.npz'
CHECKPOINT = 'models/az_iter1040.pt'
N_POSITIONS = 200          # sampled across the buffer
SEED = 4242
TOP_K = 5                  # for policy overlap

# Planes zeroed by the Python fallback: 12-107 (history), 110-113 (castling)
ZERO_RANGES = [(12, 108), (110, 114)]


def zero_bug_planes(tensor):
    """Return a copy with the C5-zeroed plane ranges set to 0."""
    out = tensor.clone()
    for lo, hi in ZERO_RANGES:
        out[:, lo:hi, :, :] = 0.0
    return out


def kl_topk(p_full, p_zero, k):
    """Symmetric-ish top-k comparison: how many of the top-k logits
    from p_full also appear in top-k of p_zero, and vice versa."""
    top_full = set(np.argsort(-p_full)[:k].tolist())
    top_zero = set(np.argsort(-p_zero)[:k].tolist())
    return len(top_full & top_zero) / k


def main():
    rng = np.random.default_rng(SEED)
    print(f"Loading buffer {BUFFER}...")
    d = np.load(BUFFER)
    boards = d['boards']  # (N, 115, 8, 8) uint8
    n_total = boards.shape[0]
    idx = rng.choice(n_total, size=min(N_POSITIONS, n_total), replace=False)
    idx.sort()
    sample = boards[idx].astype(np.float32)
    print(f"Sampled {len(idx)} positions from {n_total} total")

    print(f"Loading {CHECKPOINT}...")
    net, meta = AlphaZeroNet.load_checkpoint(CHECKPOINT, migrate=True)
    net.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)
    print(f"Model iter {meta.get('iteration')} on {device}")

    x_full = torch.from_numpy(sample).to(device)
    x_zero = zero_bug_planes(x_full)

    print(f"\nRunning forward passes...")
    with torch.no_grad():
        p_full, v_full = net(x_full)
        p_zero, v_zero = net(x_zero)
        p_full = torch.softmax(p_full, dim=-1).cpu().numpy()
        p_zero = torch.softmax(p_zero, dim=-1).cpu().numpy()
        v_full = v_full.cpu().numpy().flatten()
        v_zero = v_zero.cpu().numpy().flatten()

    # Value comparison
    v_delta = v_full - v_zero
    v_abs_mean = np.abs(v_delta).mean()
    v_abs_med = float(np.median(np.abs(v_delta)))
    v_abs_p95 = float(np.percentile(np.abs(v_delta), 95))
    v_abs_max = float(np.abs(v_delta).max())

    # Policy top-1 agreement
    top1_full = p_full.argmax(axis=1)
    top1_zero = p_zero.argmax(axis=1)
    top1_agree = (top1_full == top1_zero).mean()

    # Policy top-K overlap (average across positions)
    overlaps = [kl_topk(p_full[i], p_zero[i], TOP_K) for i in range(len(idx))]
    overlap_mean = float(np.mean(overlaps))

    # KL divergence: full || zero (asymmetric; how much info the zeroed
    # policy loses relative to full). Add a tiny epsilon to avoid log(0).
    eps = 1e-12
    kl = (p_full * (np.log(p_full + eps) - np.log(p_zero + eps))).sum(axis=1)
    kl_mean = float(kl.mean())
    kl_med = float(np.median(kl))

    print(f"\n{'='*60}")
    print(f"C5 STRENGTH-COST MEASUREMENT — az_iter1040, {len(idx)} positions")
    print(f"{'='*60}")
    print(f"\nVALUE DIVERGENCE (net's win-probability estimate)")
    print(f"  |delta| mean:   {v_abs_mean:.4f}")
    print(f"  |delta| median: {v_abs_med:.4f}")
    print(f"  |delta| p95:    {v_abs_p95:.4f}")
    print(f"  |delta| max:    {v_abs_max:.4f}")

    print(f"\nPOLICY AGREEMENT")
    print(f"  Top-1 same:              {top1_agree*100:.1f}%")
    print(f"  Top-{TOP_K} overlap (mean): {overlap_mean*100:.1f}%")
    print(f"  KL(full || zero) mean:   {kl_mean:.4f}")
    print(f"  KL(full || zero) median: {kl_med:.4f}")

    # Interpret
    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")
    if v_abs_mean < 0.05 and top1_agree > 0.9:
        print("COSMETIC — net barely uses the missing planes. "
              "Fixing C5 will not measurably change engine strength. "
              "Web demo has been playing at ~full trained level.")
    elif v_abs_mean > 0.15 or top1_agree < 0.7:
        print("MATERIAL — net leans heavily on the missing planes. "
              "Deployment strength has been well below trained ceiling. "
              "Build _ext_chess into the Modal image (and locally) "
              "before shipping the web demo.")
    else:
        print("MODEST — real but not dramatic. Net uses the planes "
              "some but doesn't collapse without them. Worth fixing "
              "for the web demo, not urgent for local play.")

    print("\nRule-of-thumb thresholds:")
    print("  cosmetic:  |dv| mean < 0.05  AND  top-1 agree > 90%")
    print("  material:  |dv| mean > 0.15  OR   top-1 agree < 70%")


if __name__ == "__main__":
    main()
