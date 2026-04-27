"""
Merge incremental tournament results into tournament_results.txt.

Usage:
    python merge_tournament.py tournament_output.txt

Or pipe directly:
    python merge_tournament.py < tournament_output.txt

The tournament output should come from tournament.py with --new-models.
This script parses the H2H results and merges them into the existing
tournament_results.txt, updating all matrices and standings.
"""
import re
import sys
from collections import defaultdict


RESULTS_FILE = "tournament_results.txt"
SIM_SETTINGS = [20, 50, 100, 200, 400]


def parse_existing_results(filepath):
    """Parse tournament_results.txt into per-sim H2H dicts."""
    with open(filepath, "r") as f:
        lines = f.readlines()

    # h2h[sims][(model_a, model_b)] = score of a vs b
    h2h = {s: {} for s in SIM_SETTINGS}
    models = []

    # Parse header for model list
    for line in lines:
        if line.startswith("Models:"):
            # Format may be "iter 30, 40, 50, ..." or "iter 30, iter 40, ..."
            raw = line[len("Models:"):].strip()
            models = re.findall(r"iter \d+|\d+", raw)
            # Ensure all have "iter " prefix
            models = [m if m.startswith("iter") else f"iter {m}" for m in models]
            break

    # Parse each sim section's H2H matrix
    current_sim = None
    matrix_models = []
    reading_matrix = False

    for line in lines:
        # Detect sim section
        m = re.match(r"\s*Standings \((\d+) sims\):", line)
        if m:
            current_sim = int(m.group(1))
            matrix_models = []
            reading_matrix = False
            continue

        # Stop parsing per-sim data when we hit the OVERALL section
        if "OVERALL" in line:
            current_sim = None
            reading_matrix = False
            continue

        # Detect matrix header row (model names)
        if current_sim and not reading_matrix:
            stripped = line.strip()
            if stripped and "iter" in stripped and "Score" not in stripped and "Model" not in stripped:
                # This is the header row with model names
                matrix_models = re.findall(r"iter \d+", stripped)
                # Must have multiple models to be a header (not a standings row)
                if len(matrix_models) > 1:
                    reading_matrix = True
                    continue

        # Skip separator line
        if reading_matrix and re.match(r"\s*-+\s*$", line):
            continue

        # Parse matrix rows
        if reading_matrix and current_sim:
            stripped = line.strip()
            if not stripped or "=" in stripped:
                reading_matrix = False
                continue

            # Row format: "  iter 100         ---       1.0       2.0 ..."
            row_match = re.match(r"\s*(iter \d+)\s+(.*)", stripped)
            if row_match:
                row_model = row_match.group(1)
                values_str = row_match.group(2)
                # Split on whitespace, getting ---/— or numbers
                values = values_str.split()
                for col_idx, val in enumerate(values):
                    if col_idx >= len(matrix_models):
                        break
                    col_model = matrix_models[col_idx]
                    if val in ("—", "\u2014", "---"):
                        continue
                    h2h[current_sim][(row_model, col_model)] = float(val)

    return models, h2h


def parse_tournament_output(text):
    """Parse tournament.py output into per-sim H2H results for new matches."""
    # h2h[sims][(model_a, model_b)] = score of a vs b
    h2h = {s: {} for s in SIM_SETTINGS}
    models = set()

    current_sim = None
    model_a = None
    model_b = None

    for line in text.split("\n"):
        # Detect sim setting
        m = re.match(r"\s*SIM SETTING: (\d+)", line)
        if m:
            current_sim = int(m.group(1))
            continue

        # Match result lines like:
        # "[2/480] iter 170 vs iter 30 (20 sims)..."
        # followed by "1.0-1.0  (45.2s)"
        match_header = re.match(
            r"\s*\[\d+/\d+\]\s+(iter \d+)\s+vs\s+(iter \d+)\s+\((\d+) sims\)", line
        )
        if match_header:
            model_a = match_header.group(1)
            model_b = match_header.group(2)
            current_sim = int(match_header.group(3))
            models.add(model_a)
            models.add(model_b)
            continue

        # Match score line: "1.0-1.0  (45.2s)"
        score_match = re.match(r"\s*(\d+\.?\d*)-(\d+\.?\d*)\s+\(", line)
        if score_match and current_sim and model_a and model_b:
            a_score = float(score_match.group(1))
            b_score = float(score_match.group(2))
            h2h[current_sim][(model_a, model_b)] = a_score
            h2h[current_sim][(model_b, model_a)] = b_score
            # Reset so we don't accidentally re-consume
            model_a = None
            model_b = None

    return sorted(models, key=lambda m: int(m.split()[-1])), h2h


def merge_h2h(existing_h2h, new_h2h):
    """Merge new H2H results into existing ones."""
    merged = {}
    for sims in SIM_SETTINGS:
        merged[sims] = dict(existing_h2h.get(sims, {}))
        for pair, score in new_h2h.get(sims, {}).items():
            merged[sims][pair] = score
    return merged


def write_results(filepath, models, h2h):
    """Write the full tournament_results.txt."""
    n = len(models)

    # Compute per-sim scores and overall
    overall_scores = defaultdict(float)
    overall_h2h = defaultdict(float)

    with open(filepath, "w") as f:
        f.write("Extinction Chess AlphaZero — Tournament Results\n")
        f.write("=================================================\n")
        f.write(f"Date: 2026-04-27\n")
        f.write(f"Models: {', '.join(models)}\n")
        f.write("Format: Round-robin, 1 match per pair per sim setting (2 games: each side as white)\n")
        f.write("Games are deterministic (temperature=0), so one match per setting suffices.\n")
        total_pairs = n * (n - 1) // 2
        total_games = total_pairs * len(SIM_SETTINGS) * 2
        f.write(f"Total: {n} models, {total_pairs} pairs, {len(SIM_SETTINGS)} sim settings, {total_games} games\n")

        for sims in SIM_SETTINGS:
            sim_scores = defaultdict(float)
            sim_h2h = h2h.get(sims, {})

            for (a, b), score in sim_h2h.items():
                sim_scores[a] += score
                overall_scores[a] += score
                overall_h2h[(a, b)] += score

            # Sort by score descending
            ranking = sorted(models, key=lambda m: sim_scores[m], reverse=True)
            max_sim = (n - 1) * 2

            f.write(f"\n{'='*60}\n")
            f.write(f"  Standings ({sims} sims):\n")
            f.write(f"  {'Model':<12} {'Score':>6} / {max_sim}\n")
            f.write(f"  {'-'*25}\n")
            for m in ranking:
                f.write(f"  {m:<12} {sim_scores[m]:>6.1f}\n")

            # H2H matrix
            _write_h2h_matrix(f, models, ranking, sim_h2h)

        # Overall
        max_score = (n - 1) * 2 * len(SIM_SETTINGS)
        ranking = sorted(models, key=lambda m: overall_scores[m], reverse=True)

        f.write(f"\n{'='*60}\n")
        f.write(f"  OVERALL STANDINGS (all sim settings combined)\n")
        f.write(f"{'='*60}\n")
        f.write(f"  {'Model':<12} {'Score':>7} / {max_score}  {'Win%':>6}\n")
        f.write(f"  {'-'*35}\n")
        for m in ranking:
            pct = overall_scores[m] / max_score * 100 if max_score > 0 else 0
            f.write(f"  {m:<12} {overall_scores[m]:>7.1f}  {pct:>6.1f}%\n")

        f.write(f"\n  Head-to-head (all sims combined, each cell out of {len(SIM_SETTINGS) * 2}):\n")
        _write_h2h_matrix(f, models, ranking, dict(overall_h2h))


def _write_h2h_matrix(f, all_models, ranking, h2h):
    """Write a H2H matrix to file."""
    cw = max(len(m) for m in all_models) + 2

    # Header
    header = " " * (cw + 2)
    for m in ranking:
        header += f"{m:>{cw}}"
    f.write(f"\n{header}\n")
    f.write(f"  {'-' * (cw + cw * len(ranking))}\n")

    for row_m in ranking:
        row = f"  {row_m:<{cw}}"
        for col_m in ranking:
            if row_m == col_m:
                row += f"{'---':>{cw}}"
            elif (row_m, col_m) in h2h:
                score = h2h[(row_m, col_m)]
                row += f"{score:>{cw}.1f}"
            else:
                row += f"{'':>{cw}}"
        f.write(row + "\n")


def main():
    # Read tournament output
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            new_output = f.read()
    else:
        print("Paste tournament output below (Ctrl+D or Ctrl+Z when done):")
        new_output = sys.stdin.read()

    # Parse existing results
    print(f"Reading existing results from {RESULTS_FILE}...")
    existing_models, existing_h2h = parse_existing_results(RESULTS_FILE)
    print(f"  Existing models: {existing_models}")

    # Parse new tournament output
    print("Parsing new tournament output...")
    new_models, new_h2h = parse_tournament_output(new_output)
    print(f"  Models in new output: {new_models}")

    # Count new matches
    new_match_count = sum(len(v) for v in new_h2h.values()) // 2
    print(f"  New match results: {new_match_count}")

    # Merge
    all_models = list(existing_models)
    for m in new_models:
        if m not in all_models:
            all_models.append(m)
    all_models.sort(key=lambda m: int(m.split()[-1]))

    merged_h2h = merge_h2h(existing_h2h, new_h2h)

    # Verify symmetry
    for sims in SIM_SETTINGS:
        for (a, b), score in merged_h2h[sims].items():
            if (b, a) in merged_h2h[sims]:
                total = score + merged_h2h[sims][(b, a)]
                if abs(total - 2.0) > 0.01:
                    print(f"  WARNING: {a} vs {b} at {sims} sims: {score} + {merged_h2h[sims][(b,a)]} = {total} (expected 2.0)")

    # Write updated results
    print(f"\nWriting updated results to {RESULTS_FILE}...")
    print(f"  Total models: {len(all_models)}: {all_models}")
    write_results(RESULTS_FILE, all_models, merged_h2h)
    print("Done!")


if __name__ == "__main__":
    main()
