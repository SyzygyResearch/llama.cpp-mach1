#!/usr/bin/env python3
"""Per-width median scoreboard across gate-passing wall draws.

The pool degrades mach1 arms stochastically (one leg or another, q4
untouched - see the six-draw table in docs/l40s-throughput-beam.md), so
single draws cannot certify a ratio target. This tool pulls the saved
batched_*.txt files for a list of wall tags, applies the hardened host
gate per draw (control B1 >= 150 AND control B16 >= 850 when present),
and reports per-width MEDIANS of each arm's TG t/s and its ratio to the
same-draw q4km row.

Usage:
  python3 median_scoreboard.py --tags dp-wall-s2,mbase-wall-s3 \
      --arms p16ntlo,p16hb [--fetch]

--fetch pulls the files from the modal volume first (needs SSL_CERT_FILE
and the modal client); without it the tool reads ./scoreboard_cache/.
"""
import argparse
import os
import re
import statistics
import subprocess
import sys

WIDTHS = (1, 2, 4, 8, 16)
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scoreboard_cache")


def fetch(tag: str, arm: str) -> str | None:
    dst = os.path.join(CACHE, f"{tag}_{arm}.txt")
    if os.path.exists(dst):
        return dst
    os.makedirs(CACHE, exist_ok=True)
    r = subprocess.run(
        [sys.executable, "-m", "modal", "volume", "get", "mach1-build-cache",
         f"/bench/results/{tag}/batched_{arm}.txt", dst, "--force"],
        capture_output=True, text=True, timeout=120)
    if r.returncode != 0 or not os.path.exists(dst):
        return None
    return dst


def tg_rows(path: str) -> dict[int, float]:
    rows: dict[int, float] = {}
    for line in open(path):
        # llama-batched-bench table row: | PP | TG | B | ... | S_TG t/s | ...
        m = re.match(r"\|\s+128\s+\|\s+128\s+\|\s+(\d+)\s+\|", line)
        if not m:
            continue
        cells = [c.strip() for c in line.split("|")]
        try:
            b = int(cells[3])
            s_tg = float(cells[8])
        except (ValueError, IndexError):
            continue
        if b in WIDTHS:
            rows[b] = s_tg
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--b1-floor", type=float, default=150.0)
    ap.add_argument("--b16-floor", type=float, default=850.0)
    ap.add_argument("--target", type=float, default=1.10)
    args = ap.parse_args()

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    control = arms[0]

    # {arm: {width: [ratio-vs-q4 per usable draw]}} and raw t/s
    ratios: dict[str, dict[int, list[float]]] = {a: {w: [] for w in WIDTHS} for a in arms}
    raw: dict[str, dict[int, list[float]]] = {a: {w: [] for w in WIDTHS} for a in arms}
    used, skipped = [], []

    for tag in tags:
        get = fetch if args.fetch else (
            lambda t, a: (lambda p: p if os.path.exists(p) else None)(
                os.path.join(CACHE, f"{t}_{a}.txt")))
        qp = get(tag, "q4km")
        cp = get(tag, control)
        if qp is None or cp is None:
            skipped.append((tag, "missing files"))
            continue
        q = tg_rows(qp)
        c = tg_rows(cp)
        if not q or not c:
            skipped.append((tag, "no rows"))
            continue
        if c.get(1, 0.0) < args.b1_floor:
            skipped.append((tag, f"control B1 {c.get(1, 0.0):.1f} < {args.b1_floor}"))
            continue
        if 16 in c and c[16] < args.b16_floor:
            skipped.append((tag, f"control B16 {c[16]:.1f} < {args.b16_floor}"))
            continue
        used.append(tag)
        for arm in arms:
            path = get(tag, arm)
            if path is None:
                continue
            rows = tg_rows(path)
            for w in WIDTHS:
                if w in rows and w in q and q[w] > 0:
                    ratios[arm][w].append(rows[w] / q[w])
                    raw[arm][w].append(rows[w])

    print(f"gate-passing draws: {used or 'NONE'}")
    for tag, why in skipped:
        print(f"  skipped {tag}: {why}")
    if not used:
        return 1
    for arm in arms:
        print(f"== {arm} (median ratio vs same-draw q4km; n per width)")
        verdicts = []
        for w in WIDTHS:
            rs = ratios[arm][w]
            if not rs:
                print(f"  B{w:<3} no data")
                verdicts.append(False)
                continue
            med = statistics.median(rs)
            ok = med >= args.target
            verdicts.append(ok)
            print(f"  B{w:<3} {med:.3f}x  (n={len(rs)}, t/s med {statistics.median(raw[arm][w]):.1f})"
                  f"  {'PASS' if ok else 'short of ' + format(args.target, '.2f')}")
        print(f"  => {'ALL WIDTHS PASS' if all(verdicts) else 'target not met at every width'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
