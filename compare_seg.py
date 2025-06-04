#!/usr/bin/env python
"""
compare_seg.py
──────────────
Evaluate model TextGrids against ground-truth TextGrids.

Outputs
  • <out_dir>/sub_<ID>/run_<N>/<stem>.png   (ONLY for non-repeated stimuli)
  • <out_dir>/summary.csv                   per-file + overall metrics
Console summary prints global stats.
"""

from __future__ import annotations

# stdlib
import argparse, itertools, math, pathlib, re, warnings
# 3rd-party
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from tqdm import tqdm
from praatio import textgrid as ptg

# colours / hatch
TRUTH_CLR, PRED_CLR, ERR_CLR = "tab:blue", "tab:red", "darkgreen"
HATCH = r"\\"

# ───────────────────────── helper functions ──────────────────────────────
_tail_re = re.compile(r"_stim-\d+_(.+)$")  # capture everything after _stim-XX_


def _tokens_after_stim(stem: str) -> list[str]:
    """Return tokens (syllables) that appear after “_stim-XX_” in the filename."""
    m = _tail_re.search(stem)
    if not m:
        return []
    tail = m.group(1).replace("_", " ")  # normalise underscores→spaces
    return tail.split()


def _is_repeated(tokens: list[str], min_repeats: int = 3) -> bool:
    """
    A stimulus is 'repeated-syllable' if, after dropping any token
    that contains a digit, *all remaining tokens are identical* and
    there are at least `min_repeats` of them.
    """
    sylls = [t for t in tokens if not any(ch.isdigit() for ch in t)]
    return len(sylls) >= min_repeats and len(set(sylls)) == 1


def _sub_number(stem: str) -> str | None:
    for part in stem.split("_"):
        if part.startswith("sub-"):
            return part.split("-")[1]
    return None


def _run_number(stem: str) -> str | None:
    for part in stem.split("_"):
        if part.startswith("run-"):
            return part.split("-")[1]
    return None


def load_intervals(p: pathlib.Path, tier: str) -> list[tuple[float, float]]:
    tg = ptg.openTextgrid(p, includeEmptyIntervals=True, duplicateNamesMode="error")
    t = tg.getTier(tier)
    entries = getattr(t, "entries", None) or getattr(t, "entryList")
    return [(s, e) for s, e, _ in entries]


def onset_time(ints) -> float | None:
    return ints[1][0] if len(ints) >= 2 else None


def boundary_errors(truth, pred):
    err = []
    for (ts, te), (ps, pe) in zip(truth, pred):
        err += [abs(ts - ps), abs(te - pe)]
    return np.asarray(err)


# ───────────────────────── plotting ──────────────────────────────────────
def _make_fig():
    return plt.subplots(figsize=(10, 2.5))


def plot_timeline(truth, pred, tol, out_png):
    fig, ax = _make_fig()
    xmax = max(truth[-1][1] if truth else 0,
               pred[-1][1] if pred else 0)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 2)
    ax.set_yticks([])
    ax.set_xlabel("time (s)")
    # grey ±tol band
    for s, _ in truth:
        ax.axvspan(max(0, s - tol), s + tol, 0, .4, color="lightgrey", alpha=.3, zorder=0)
    # truth vs model bars
    for s, e in truth:
        ax.barh(1.25, e - s, left=s, height=.5, color=TRUTH_CLR, alpha=.6, edgecolor="k")
    for s, e in pred:
        ax.barh(.25, e - s, left=s, height=.5, color=PRED_CLR, alpha=.6, edgecolor="k")
    # error markers
    for (ts, te), (ps, pe) in zip(truth, pred):
        if abs(ts - ps) > tol:
            ax.axvline(ps, 0, .4, color=ERR_CLR, ls="--", lw=2, alpha=.25)
        if abs(te - pe) > tol:
            ax.axvline(pe, 0, .4, color=ERR_CLR, ls="--", lw=2, alpha=.25)
    # hatching for extra intervals
    if len(truth) != len(pred):
        extra, y = ((truth[len(pred):], 1.25)
                    if len(truth) > len(pred) else (pred[len(truth):], .25))
        for s, e in extra:
            ax.barh(y, e - s, left=s, height=.5, facecolor="none",
                    edgecolor="grey", hatch=HATCH, lw=1.2, alpha=.4)
    ax.legend(["±tol", "truth", "model"], loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ───────────────────────── main ──────────────────────────────────────────
def main(args):
    truth_dir, pred_dir, out_root = map(pathlib.Path,
                                        (args.truth_dir, args.pred_dir, args.out_dir))
    out_root.mkdir(parents=True, exist_ok=True)

    rows, onset_errs = [], []
    same_cnt = diff_cnt = rept_cnt = 0

    for tf in tqdm(sorted(truth_dir.glob("*.TextGrid")), desc="files"):
        stem = tf.stem
        pf = pred_dir / f"{stem}.TextGrid"
        if not pf.exists():
            warnings.warn(f"missing model grid for {stem}")
            continue

        tokens = _tokens_after_stim(stem)
        repeated = _is_repeated(tokens)

        # ---------- skip plotting early if repeated ----------
        if repeated:
            rept_cnt += 1
            truth = load_intervals(tf, args.tier)
            pred = load_intervals(pf, args.tier)
            o_tr, o_pr = onset_time(truth), onset_time(pred)
            onset_errs.append(abs(o_tr - o_pr) * 1000 if None not in (o_tr, o_pr) else math.nan)
            rows.append(dict(file=stem, onset_error_ms=onset_errs[-1],
                             mean_error_ms=np.nan, max_error_ms=np.nan,
                             within_tol=np.nan, note="repeated syllables"))
            continue  # -> next file (NO PNG)

        # ---------- non-repeated  ----------
        truth = load_intervals(tf, args.tier)
        pred = load_intervals(pf, args.tier)

        o_tr, o_pr = onset_time(truth), onset_time(pred)
        onset_err = abs(o_tr - o_pr) * 1000 if None not in (o_tr, o_pr) else math.nan
        onset_errs.append(onset_err)

        if len(truth) != len(pred):
            diff_cnt += 1
            rows.append(dict(file=stem, onset_error_ms=onset_err,
                             mean_error_ms=np.nan, max_error_ms=np.nan,
                             within_tol=np.nan,
                             note=f"count mismatch {len(truth)} vs {len(pred)}"))
        else:
            same_cnt += 1
            errs = boundary_errors(truth, pred)
            rows.append(dict(file=stem, onset_error_ms=onset_err,
                             mean_error_ms=errs.mean() * 1000,
                             max_error_ms=errs.max() * 1000,
                             within_tol=(errs <= args.tol).mean() * 100,
                             note=""))

        # plot
        sub = _sub_number(stem) or "unk"
        run = _run_number(stem) or "unk"
        out_dir = out_root / f"sub_{sub}" / f"run_{run}"
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_timeline(truth, pred, args.tol, out_dir / f"{stem}.png")

    # ---------- summary ----------
    df = pd.DataFrame(rows)
    if "mean_error_ms" in df.columns:
        ok = df.dropna(subset=["mean_error_ms"])
    else:
        ok = pd.DataFrame()
    overall = dict(file="_overall",
                   onset_error_ms=np.nanmean(onset_errs),
                   mean_error_ms=ok.mean_error_ms.mean() if not ok.empty else np.nan,
                   max_error_ms=ok.max_error_ms.max() if not ok.empty else np.nan,
                   within_tol=ok.within_tol.mean() if not ok.empty else np.nan,
                   note=f"{same_cnt} match / {diff_cnt} mismatch / {rept_cnt} repeated")
    df = pd.concat([df, pd.DataFrame([overall])])
    df.to_csv(out_root / "summary.csv", index=False)

    # console
    print("\n──────── evaluation summary ────────")
    print(f"Files compared            : {len(rows)}")
    print(f"Same # intervals          : {same_cnt}")
    print(f"Interval count mismatch   : {diff_cnt}")
    print(f"Repeated-syllable files   : {rept_cnt}")
    print(f"Mean onset error          : {overall['onset_error_ms']:.1f} ms")
    if same_cnt:
        print(f"Mean boundary error       : {overall['mean_error_ms']:.1f} ms")
        print(f"Max  boundary error       : {overall['max_error_ms']:.1f} ms")
        print(f"Within ±{args.tol * 1000:.0f} ms       : {overall['within_tol']:.1f} %")
    print(f"Artifacts saved under     : {out_root}")


# ───────────────────────── CLI ───────────────────────────────────────────
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth_dir", required=True)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--out_dir", required=True,
                    help="e.g. results/comparison_results")
    ap.add_argument("--tier", default="syll")
    ap.add_argument("--tol", type=float, default=0.02,
                    help="tolerance (seconds) for boundary agreement")
    main(ap.parse_args())
