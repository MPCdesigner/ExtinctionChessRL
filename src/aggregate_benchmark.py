"""Aggregate a benchmark battery's per-test logs into a single summary file.

Sbatched by `launch_benchmark_battery` (in alphazero.py) with
`--dependency=afterany` on all test jobs. Runs after every test finishes
(success or fail), reads each log, extracts the summary block, and
concatenates them into a human-readable file at:

    ~/extinction-chess/benchmark_results/iter_<N>.txt

Design principle: no regex parsing of test results. Each test writes its
own summary section at the end of its log; we just extract that block
verbatim. This matches exactly what the user has been reading manually
when interpreting benchmark output, and stays robust to future format
changes in the underlying test scripts.

Extraction rules per test type:
  - H2H (compare_extensive.py):  tail 30 lines
  - Tactical random:              tail 30 lines
  - Win-taking:                   from last "Win-Taking Test Summary"
                                  marker to EOF (captures the model table
                                  AND the "Tough Positions" section);
                                  fallback to tail 120 lines if marker
                                  not found.
"""

import argparse
import glob
import os
import re
from datetime import datetime


def _tail_lines(log_path, n):
    with open(log_path, "r", errors="replace") as f:
        lines = f.readlines()
    return "".join(lines[-n:])


def extract_h2h_summary(log_path):
    return _tail_lines(log_path, 30)


def extract_tactical_summary(log_path):
    return _tail_lines(log_path, 30)


def extract_win_taking_summary(log_path):
    """From the last 'Win-Taking Test Summary' marker to EOF.

    Includes the divider line immediately above the marker if present, so
    the block looks identical to the raw test output.
    """
    with open(log_path, "r", errors="replace") as f:
        content = f.read()

    marker = "Win-Taking Test Summary"
    idx = content.rfind(marker)
    if idx < 0:
        # Marker missing (test likely errored before summary); fall back.
        return _tail_lines(log_path, 120)

    # Rewind to the start of the marker's line.
    line_start = content.rfind("\n", 0, idx) + 1

    # If the line immediately above is a divider (===), include it too so
    # the extracted block visually matches raw test output.
    prev_newline = line_start - 1
    if prev_newline > 0:
        prev_line_start = content.rfind("\n", 0, prev_newline) + 1
        prev_line = content[prev_line_start:prev_newline]
        if prev_line.strip().startswith("=="):
            line_start = prev_line_start

    return content[line_start:]


def _parse_jobid(log_path):
    """Extract JOBID from a log filename like `compare_760vs750_567785.log`."""
    m = re.search(r"_(\d+)\.log$", os.path.basename(log_path))
    return m.group(1) if m else "?"


def _build_test_plan(iter_num):
    """Return an ordered list of (category, subject, glob_pattern, extractor)."""
    tests = []
    for offset in (10, 20, 30, 40, 50):
        opp = iter_num - offset
        tests.append((
            "recent H2H",
            f"vs iter {opp}",
            f"compare_{iter_num}vs{opp}_*.log",
            extract_h2h_summary,
        ))
    for offset in (110, 120, 130, 140, 150):
        opp = iter_num - offset
        tests.append((
            "distant H2H",
            f"vs iter {opp}",
            f"compare_{iter_num}vs{opp}_*.log",
            extract_h2h_summary,
        ))
    tests.append((
        "win-taking",
        "multi-model",
        "bench_win_taking_*.log",
        extract_win_taking_summary,
    ))
    tests.append((
        "tactical random",
        "full-game vs advanced tactical",
        f"tactical_{iter_num}_*.log",
        extract_tactical_summary,
    ))
    tests.append((
        "baseline H2H",
        "vs iter 100",
        f"compare_{iter_num}vs100_*.log",
        extract_h2h_summary,
    ))
    return tests


def aggregate(iter_num, iter_dir, results_file):
    tests = _build_test_plan(iter_num)
    total = len(tests)

    parts = []
    parts.append("=" * 80)
    parts.append(f"BENCHMARK BATTERY: iter {iter_num}")
    parts.append(f"Aggregated: {datetime.now().isoformat(timespec='seconds')}")
    parts.append(f"Source directory: {iter_dir}")
    parts.append("=" * 80)
    parts.append("")

    for i, (category, subject, pattern, extractor) in enumerate(tests, 1):
        matches = sorted(glob.glob(os.path.join(iter_dir, pattern)))

        parts.append("=" * 80)
        if matches:
            jobid = _parse_jobid(matches[0])
            parts.append(f"  [{i}/{total}] {category}: {subject}  (job {jobid})")
        else:
            parts.append(f"  [{i}/{total}] {category}: {subject}")
        parts.append("=" * 80)
        parts.append("")

        if not matches:
            parts.append(
                "Status: SKIPPED (log file not found — sbatch may have failed "
                "or reference checkpoint was missing)"
            )
        else:
            log_path = matches[0]
            try:
                parts.append(extractor(log_path).rstrip())
            except Exception as exc:
                parts.append(f"Status: PARSE ERROR reading {log_path}")
                parts.append(f"Exception: {exc}")

        parts.append("")
        parts.append("")

    parts.append("=" * 80)
    parts.append(f"END OF BENCHMARK BATTERY: iter {iter_num}")
    parts.append("=" * 80)
    parts.append("")

    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "w") as f:
        f.write("\n".join(parts))

    print(f"[aggregator] wrote {results_file}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iter", type=int, required=True,
                        help="Iteration number the battery was launched for.")
    parser.add_argument("--iter-dir", required=True,
                        help="Directory containing this iter's per-test log files.")
    parser.add_argument("--results-file", required=True,
                        help="Absolute path to the summary file to write.")
    args = parser.parse_args()

    aggregate(args.iter, args.iter_dir, args.results_file)


if __name__ == "__main__":
    main()
