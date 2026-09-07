"""
Distill the iter-1040 AlphaZero teacher into a small student network that can
run in a browser via onnxruntime-web.

Why: the teacher is 20 blocks x 256 filters (~24.5M params, ~3 GFLOPs per
position). That's ~10 sims/sec on a fast desktop CPU and a 98 MB download —
unusable client-side. The student below is ~50x cheaper per evaluation and
about 2 MB, which puts 400 MCTS sims within a couple of seconds in plain
WebAssembly (no WebGPU required, so it works on Safari, Firefox and mobile).

Run:
    # one-time: put the teacher checkpoint on the models volume
    modal volume put extinction-chess-models models/az_iter1040.pt /az_iter1040.pt

    # train + export
    modal run distill_web.py

    # fetch the artifacts
    modal volume get extinction-chess-models /web/student.onnx      ./web/
    modal volume get extinction-chess-models /web/student_meta.json ./web/

Cost: one L4 for well under an hour on the ~103k positions currently in
replay_buffer/. Add more iter_*.npz files there for a stronger student —
the script picks up whatever is present.
"""

import modal

app = modal.App("extinction-chess-distill")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "onnx", "onnxruntime")
    .add_local_dir(
        "src",
        remote_path="/root/src",
        ignore=lambda p: ".venv" in str(p) or "__pycache__" in str(p),
    )
    # ~20 MB of buffers — small enough to bake into the image
    .add_local_dir("replay_buffer", remote_path="/root/replay_buffer")
)

models_volume = modal.Volume.from_name("extinction-chess-models", create_if_missing=True)
MODELS_DIR = "/models"


# ═════════════════════════════════════════════════════════════════════════════
# Student network
#
# Kept deliberately close to the teacher so behaviour transfers, with two
# changes that matter:
#
#   1. Conv policy head instead of Linear(128 -> 4864). The teacher's FC head
#      is 622k params — larger than this entire student's residual tower — and
#      it flattens away the board's spatial structure. Since the policy index
#      is `plane * 64 + from_square` and planes are [channel][rank][file], a
#      1x1 conv to 76 channels flattens to exactly the same 4864 layout for
#      ~5k params. Structurally identical output, ~120x cheaper.
#
#   2. Narrower value head (64 -> 64 -> 1 rather than 64 -> 256 -> 1).
# ═════════════════════════════════════════════════════════════════════════════

STUDENT_SRC = '''
import torch, torch.nn as nn, torch.nn.functional as F

NUM_INPUT_CHANNELS = 115
NUM_POLICY_PLANES  = 76
POLICY_SIZE        = NUM_POLICY_PLANES * 64  # 4864


class ResBlock(nn.Module):
    """Mirrors src/alphazero.py ResBlock exactly."""
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(c)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class StudentNet(nn.Module):
    def __init__(self, in_channels=NUM_INPUT_CHANNELS, filters=64, blocks=6):
        super().__init__()
        self.in_channels = in_channels
        self.filters = filters
        self.blocks = blocks

        self.input_conv = nn.Conv2d(in_channels, filters, 3, padding=1, bias=False)
        self.input_bn   = nn.BatchNorm2d(filters)
        self.res_blocks = nn.Sequential(*[ResBlock(filters) for _ in range(blocks)])

        # 1x1 conv -> (B, 76, 8, 8) -> flatten == plane * 64 + from_square
        self.policy_conv = nn.Conv2d(filters, NUM_POLICY_PLANES, 1)

        self.value_conv = nn.Conv2d(filters, 1, 1, bias=False)
        self.value_bn   = nn.BatchNorm2d(1)
        self.value_fc1  = nn.Linear(64, 64)
        self.value_fc2  = nn.Linear(64, 1)

    def forward(self, x):
        out = F.relu(self.input_bn(self.input_conv(x)))
        out = self.res_blocks(out)

        p = self.policy_conv(out).flatten(1)          # (B, 4864) logits

        v = F.relu(self.value_bn(self.value_conv(out))).flatten(1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)).squeeze(-1)  # (B,) in [-1, 1]
        return p, v
'''


@app.function(
    image=image,
    gpu="L4",
    timeout=14400,  # 4 hours
    volumes={MODELS_DIR: models_volume},
)
def distill(
    teacher_ckpt: str = "az_iter1040.pt",
    filters: int = 64,
    blocks: int = 6,
    epochs: int = 30,
    batch_size: int = 512,
    lr: float = 2e-3,
    temperature: float = 1.0,
    value_weight: float = 1.0,
    hard_target_weight: float = 0.0,
    val_frac: float = 0.05,
    seed: int = 0,
):
    """
    Label stored positions with the teacher, then fit the student to those
    soft targets.

    hard_target_weight blends the *stored* MCTS visit distributions back in.
    Those came from whichever model generated that iteration (688-696 here),
    so they're weaker than iter-1040's policy — but they are search output
    rather than a raw forward pass. Default 0.0 (pure teacher). Try 0.2-0.3
    if the student underfits sharp tactical positions.
    """
    import sys, os, json, glob, time
    import numpy as np
    import torch
    import torch.nn.functional as F

    sys.path.insert(0, "/root/src")
    torch.manual_seed(seed)
    np.random.seed(seed)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}")

    # ── Student definition (shared verbatim with the export step) ────────────
    ns: dict = {}
    exec(STUDENT_SRC, ns)
    StudentNet = ns["StudentNet"]
    POLICY_SIZE = ns["POLICY_SIZE"]

    # ── Load positions ───────────────────────────────────────────────────────
    files = sorted(glob.glob("/root/replay_buffer/iter_*.npz"))
    if not files:
        raise SystemExit("no replay buffers found under replay_buffer/")

    boards_l, pol_l, val_l = [], [], []
    for f in files:
        d = np.load(f)
        boards_l.append(d["boards"])
        pol_l.append(d["policies"])
        val_l.append(d["values"])
        print(f"  {os.path.basename(f)}: {len(d['boards']):,} positions")

    boards = np.concatenate(boards_l).astype(np.float32)
    stored_pol = np.concatenate(pol_l)
    stored_val = np.concatenate(val_l)
    N = len(boards)
    print(f"total positions: {N:,}")

    # ⚠ Plane 114 (halfmove clock) is ALWAYS ZERO in every stored buffer.
    # src/state_encoder.py:100 writes `halfmove_clock / 100.0`, a float in
    # [0,1), but src/alphazero.py:1973 saves boards as uint8 — which truncates
    # every value below 1.0 to 0. So the teacher has never seen a nonzero
    # halfmove plane, while live inference feeds it real fractions.
    # We zero it here to match what the teacher actually learned, and the
    # browser encoder must do the same. See the note printed at the end.
    boards[:, 114] = 0.0

    # MCTS only expands legal moves, so the support of the stored visit
    # distribution is a reliable legal-move mask. We reuse it to mask the
    # teacher's logits, so student and teacher are compared over the same set.
    legal_mask = stored_pol > 0

    # A row with no legal moves would make masked_fill(-inf) + softmax emit
    # NaN and silently poison training. The current buffers have none (checked:
    # 0 of 103,200, min 1 legal move), but cluster buffers may include terminal
    # positions — so drop them rather than trust the invariant.
    keep = legal_mask.any(axis=1)
    if not keep.all():
        print(f"dropping {(~keep).sum():,} positions with no legal moves")
        boards, stored_pol, stored_val = boards[keep], stored_pol[keep], stored_val[keep]
        legal_mask = legal_mask[keep]
        N = len(boards)

    # ── Teacher ──────────────────────────────────────────────────────────────
    from alphazero import AlphaZeroNet

    tpath = os.path.join(MODELS_DIR, teacher_ckpt)
    if not os.path.exists(tpath):
        raise SystemExit(
            f"teacher checkpoint not found at {tpath}\n"
            f"upload it first:\n"
            f"  modal volume put extinction-chess-models "
            f"models/{teacher_ckpt} /{teacher_ckpt}"
        )

    teacher, tmeta = AlphaZeroNet.load_checkpoint(tpath, migrate=True)
    teacher = teacher.to(dev).eval()
    print(f"teacher: iter {tmeta.get('iteration', '?')}, "
          f"{sum(p.numel() for p in teacher.parameters()):,} params")

    # ── Label every position with the teacher ────────────────────────────────
    print("labelling with teacher...")
    t0 = time.time()
    t_pol = np.empty((N, POLICY_SIZE), dtype=np.float32)
    t_val = np.empty(N, dtype=np.float32)

    with torch.no_grad():
        for i in range(0, N, 1024):
            xb = torch.from_numpy(boards[i:i + 1024]).to(dev)
            logits, v = teacher(xb)
            # Mask to legal support before softmax so the student is never
            # asked to reproduce probability mass on illegal moves.
            m = torch.from_numpy(legal_mask[i:i + 1024]).to(dev)
            logits = logits.masked_fill(~m, float("-inf"))
            t_pol[i:i + 1024] = F.softmax(logits / temperature, dim=1).cpu().numpy()
            t_val[i:i + 1024] = v.cpu().numpy()
            if i % 20480 == 0:
                print(f"  {i:,}/{N:,}")
    print(f"labelled in {time.time() - t0:.0f}s")

    del teacher
    torch.cuda.empty_cache()

    # Optionally blend the stored search distributions back in
    if hard_target_weight > 0:
        w = hard_target_weight
        t_pol = (1 - w) * t_pol + w * stored_pol
        t_val = (1 - w) * t_val + w * stored_val
        print(f"blended stored targets at weight {w}")

    # ── Split ────────────────────────────────────────────────────────────────
    idx = np.random.permutation(N)
    n_val = int(N * val_frac)
    va_idx, tr_idx = idx[:n_val], idx[n_val:]
    print(f"train {len(tr_idx):,} / val {len(va_idx):,}")

    def batches(indices, bs, shuffle=True):
        order = np.random.permutation(indices) if shuffle else indices
        for i in range(0, len(order), bs):
            j = order[i:i + bs]
            yield (torch.from_numpy(boards[j]).to(dev),
                   torch.from_numpy(t_pol[j]).to(dev),
                   torch.from_numpy(t_val[j]).to(dev),
                   torch.from_numpy(legal_mask[j]).to(dev))

    # ── Student ──────────────────────────────────────────────────────────────
    student = StudentNet(filters=filters, blocks=blocks).to(dev)
    n_params = sum(p.numel() for p in student.parameters())
    print(f"student: {blocks} blocks x {filters} filters, {n_params:,} params "
          f"(~{n_params * 4 / 1e6:.1f} MB fp32)")

    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * max(1, len(tr_idx) // batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.15)

    def evaluate():
        student.eval()
        agree = tot = 0
        vmae = 0.0
        with torch.no_grad():
            for xb, pb, vb, mb in batches(va_idx, 1024, shuffle=False):
                logits, v = student(xb)
                logits = logits.masked_fill(~mb, float("-inf"))
                agree += (logits.argmax(1) == pb.argmax(1)).sum().item()
                vmae += (v - vb).abs().sum().item()
                tot += len(xb)
        student.train()
        return agree / tot, vmae / tot

    print("training...")
    step = 0
    for ep in range(epochs):
        pl_sum = vl_sum = nb = 0
        for xb, pb, vb, mb in batches(tr_idx, batch_size):
            logits, v = student(xb)
            logits = logits.masked_fill(~mb, float("-inf"))

            # Cross-entropy against the teacher's soft distribution.
            # log_softmax is -inf on masked entries and the teacher assigns
            # them exactly 0 probability, so the naive product is 0 * -inf =
            # NaN. Zero the masked log-probs first; the teacher contributes no
            # mass there anyway, and masked logits already get no gradient.
            logp = F.log_softmax(logits, dim=1).masked_fill(~mb, 0.0)
            p_loss = -(pb * logp).sum(dim=1).mean()
            v_loss = F.mse_loss(v, vb)
            loss = p_loss + value_weight * v_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            if step < steps - 1:
                sched.step()
            step += 1

            pl_sum += p_loss.item(); vl_sum += v_loss.item(); nb += 1

        top1, vmae = evaluate()
        print(f"  epoch {ep + 1:>3}/{epochs}  policy {pl_sum / nb:.4f}  "
              f"value {vl_sum / nb:.4f}  |  top1-agree {top1 * 100:.1f}%  "
              f"value-MAE {vmae:.4f}")

    top1, vmae = evaluate()

    # ── Export ───────────────────────────────────────────────────────────────
    outdir = os.path.join(MODELS_DIR, "web")
    os.makedirs(outdir, exist_ok=True)

    torch.save({"state_dict": student.state_dict(),
                "filters": filters, "blocks": blocks}, os.path.join(outdir, "student.pt"))

    student.eval().to("cpu")
    dummy = torch.zeros(1, 115, 8, 8)
    onnx_path = os.path.join(outdir, "student.onnx")
    torch.onnx.export(
        student, dummy, onnx_path,
        input_names=["board"], output_names=["policy", "value"],
        dynamic_axes={"board": {0: "batch"},
                      "policy": {0: "batch"},
                      "value": {0: "batch"}},
        opset_version=17,
    )

    meta = {
        "teacher": teacher_ckpt,
        "teacher_iteration": int(tmeta.get("iteration", -1)),
        "student": {"blocks": blocks, "filters": filters, "params": n_params},
        "input": {
            "shape": [1, 115, 8, 8],
            "layout": "channel, rank, file",
            "note": "plane 114 (halfmove) MUST be zero — teacher never saw it nonzero",
        },
        "policy": {"size": POLICY_SIZE, "index": "plane * 64 + from_rank * 8 + from_file"},
        "value": {"range": [-1, 1], "perspective": "side to move"},
        "training": {
            "positions": int(N), "epochs": epochs, "temperature": temperature,
            "hard_target_weight": hard_target_weight,
        },
        "eval": {"top1_agreement_with_teacher": round(top1, 4),
                 "value_mae": round(vmae, 4)},
    }
    with open(os.path.join(outdir, "student_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    models_volume.commit()

    size_mb = os.path.getsize(onnx_path) / 1e6
    print("\n" + "=" * 62)
    print(f"student.onnx     {size_mb:.2f} MB")
    print(f"top-1 agreement  {top1 * 100:.1f}%  (student picks teacher's move)")
    print(f"value MAE        {vmae:.4f}")
    print("=" * 62)
    print("\nfetch with:")
    print("  modal volume get extinction-chess-models /web/student.onnx      ./web/")
    print("  modal volume get extinction-chess-models /web/student_meta.json ./web/")
    print("\n⚠ pipeline bug found while reading the buffers:")
    print("  state_encoder.py:100 sets plane 114 = halfmove_clock / 100.0 (a")
    print("  float < 1.0), but alphazero.py:1973 saves boards as uint8, so it")
    print("  truncates to 0 in every stored position. Verified: 0 nonzero out")
    print("  of ~103k positions. The net trains on halfmove=0 but gets real")
    print("  fractions at inference — a train/serve mismatch on that plane.")

    return meta


@app.local_entrypoint()
def main(
    teacher: str = "az_iter1040.pt",
    filters: int = 64,
    blocks: int = 6,
    epochs: int = 30,
    hard_target_weight: float = 0.0,
):
    meta = distill.remote(
        teacher_ckpt=teacher,
        filters=filters,
        blocks=blocks,
        epochs=epochs,
        hard_target_weight=hard_target_weight,
    )
    print("\ndone:")
    import json
    print(json.dumps(meta, indent=2))
