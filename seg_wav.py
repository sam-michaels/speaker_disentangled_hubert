#!/usr/bin/env python
"""
Batch syllable segmentation with **S5-HuBERT + minCut**
────────────────────────────────────────────────────────
For every .wav in <in_path>
→ creates <out_dir>/sub_<ID>/run_<N>/<stem>.TextGrid

• Accepts any sample-rate, mono or stereo.
• Uses the same min-cut + optional “merge-adjacent” logic you already had.
• Needs the mincut package that ships with VG-HuBERT
  (run `bash scripts/setup.sh` once to pull it in).
"""

# ─────────────── Imports ────────────────────

# Standard Libraries
import argparse
import pathlib
import warnings

# Third-Party Libraries
import librosa
import numpy as np
import soundfile as sf
import torch

from src.s5hubert import S5HubertForSyllableDiscovery
from src.s5hubert.mincut import mincut_utils as mincut


# ───────────────── filename helpers ─────────────────
def _run_number(stem):  # returns int | None
    for part in stem.split("_"):
        if part.startswith("run-"):
            try:
                return int(part.split("-", 1)[1])
            except (ValueError, IndexError):
                return None
    return None


def _sub_number(stem):  # returns int | None
    for part in stem.split("_"):
        if part.startswith("sub-"):
            try:
                return int(part.split("-", 1)[1])
            except (ValueError, IndexError):
                return None
    return None


def _tokens_after_stim(stem):
    parts = stem.split("_", 4)
    return parts[4].split() if len(parts) == 5 else []


def _has_run_of(tokens, n=3):
    run = 1
    for i in range(1, len(tokens)):
        run = run + 1 if tokens[i] == tokens[i - 1] else 1
        if run >= n:
            return True
    return False


# ───────────────── min-cut helpers ────────────────
def _adjacent_sims(pairs, feat):
    vecs = [feat[l:r].mean(0) for l, r in pairs]
    return [np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            for a, b in zip(vecs[:-1], vecs[1:])]


def _mincut_pairs(audio_sec,
                  feat,  # numpy [T, D]
                  sec_per_syllable,
                  method,
                  sec_per_frame):  # seconds per frame
    """Return list[[l,r], …] of frame-index pairs."""
    n_syl = int(np.ceil(audio_sec / sec_per_syllable))

    # run official numpy implementation (fast on CPU)
    frame_pairs = mincut.mincut_numpy(
        feat,
        sec_per_frame=sec_per_frame,
        sec_per_syllable=sec_per_syllable,
        merge_threshold=None,  # we merge manually below
        min_duration=3,
        max_duration=35,
    )[2]  # shape (N, 2) — frame indices

    pairs = [[l, r] for l, r in frame_pairs if r - l > 2] or frame_pairs.tolist()

    # optional similarity-based merge (unchanged from your logic)
    if "merge" in method.lower() and len(pairs) >= 3:
        thr = float(method.split("-")[-1])
        while True:
            sims = _adjacent_sims(pairs, feat)
            i = int(np.argmax(sims))
            if sims[i] < thr or len(pairs) < 3:
                break
            pairs[i:i + 2] = [[pairs[i][0], pairs[i + 1][1]]]
    return pairs


# ───────────────── TextGrid helpers ───────────────
def _intervalize(pairs, spf, audio_len):
    ints, prev = [], 0.0
    for l, r in pairs:
        s, e = l * spf, r * spf
        if s > prev:  ints.append((prev, s))
        ints.append((s, e))
        prev = e
    if prev < audio_len:
        ints.append((prev, audio_len))
    return ints


def _write_textgrid(ints, xmax, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as tg:
        tg.write('File type = "ooTextFile"\nObject class = "TextGrid"\n\n')
        tg.write('xmin = 0\n');
        tg.write(f'xmax = {xmax}\n')
        tg.write('tiers? <exists>\nsize = 1\nitem []:\n    item [1]:\n')
        tg.write('        class = "IntervalTier"\n        name = "syll"\n')
        tg.write('        xmin = 0\n');
        tg.write(f'        xmax = {xmax}\n')
        tg.write(f'        intervals: size = {len(ints)}\n')
        for i, (xmin, xmax) in enumerate(ints, 1):
            tg.write(f'        intervals [{i}]:\n')
            tg.write(f'            xmin = {xmin}\n')
            tg.write(f'            xmax = {xmax}\n')
            tg.write('            text = ""\n')


# ───────────────────────── main ─────────────────────────
def main(args):
    wav_input = pathlib.Path(args.in_path).expanduser()
    wav_paths = ([wav_input] if wav_input.is_file()
                 else sorted(wav_input.rglob("*.wav")))
    if not wav_paths:
        raise SystemExit(f"No .wav files found under {wav_input}")

    out_root = pathlib.Path(args.out_dir).expanduser()
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = S5HubertForSyllableDiscovery.from_pretrained(args.checkpoint)
    model.to(device).eval()
    torch.set_grad_enabled(False)

    for wav in wav_paths:
        audio, sr = sf.read(wav, dtype="float32")
        if audio.ndim == 2:  # stereo → mono
            audio = audio.mean(-1)
        if sr != 16_000:  # resample
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16_000)
            sr = 16_000

        wav_t = torch.from_numpy(audio).unsqueeze(0).to(device)

        with torch.inference_mode():
            chunks = model(wav_t)  # list[dict]

        # take frame-level embeddings (key == "dense")
        if "dense" not in chunks[0]:
            raise RuntimeError(f"Model output keys: {chunks[0].keys()}  (expected 'dense')")
        hidden = torch.cat([c["dense"] for c in chunks], dim=1)  # [1,T,D]
        feat = hidden.squeeze(0).cpu().numpy()  # [T,D]
        spf = len(audio) / sr / feat.shape[0]  # sec per frame

        pairs = _mincut_pairs(len(audio) / sr, feat,
                              args.sec_per_syllable,
                              args.segment_method,
                              spf)

        sub = _sub_number(wav.stem) or "unk"
        run = _run_number(wav.stem) or "unk"
        stem = "_".join(wav.stem.split())  # collapse spaces
        tg_path = out_root / f"sub_{sub:02}" / f"run_{run}" / (stem + ".TextGrid")

        intervals = _intervalize(pairs, spf, len(audio) / sr)
        _write_textgrid(intervals, len(audio) / sr, tg_path)

        if not args.quiet and _has_run_of(_tokens_after_stim(wav.stem), 3):
            print(f"{wav.name}: {len(pairs)} syllables")


# ───────────────────────── CLI ─────────────────────────
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="ryota-komatsu/s5-hubert",
                    help="HF repo or local folder with S5-HuBERT weights")
    ap.add_argument("--in_path", required=True,
                    help="WAV file **or** directory (quote if path has spaces)")
    ap.add_argument("--out_dir", required=True,
                    help="Top-level folder for TextGrids")
    ap.add_argument("--segment_method", default="minCutMerge-0.3",
                    help="minCut or minCutMerge-<thr>")
    ap.add_argument("--sec_per_syllable", type=float, default=0.20)
    ap.add_argument("--quiet", action="store_true")
    main(ap.parse_args())
