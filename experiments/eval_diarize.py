"""Evaluate our speaker diarization on synthetic conversations of unseen LibriSpeech speakers.

python experiments/eval_diarize.py --stage tune    # dev-clean: pick the clustering threshold, store it in models/speaker/model.pt
python experiments/eval_diarize.py --stage test    # test-clean: DER table -> results/diarize/

Conversations: 2-5 speakers taking turns; each turn continues that speaker's own reading for an
exponentially distributed 1-15 s (mean 5 s), with 0.15-1.0 s pauses and a random level per speaker.
No overlapping speech (our diarizer is one-speaker-at-a-time). dev-clean and test-clean speakers
are disjoint from the 251 training speakers. Score: DER with a 0.25 s collar (metrics/diarization.py).
Systems: MFCC statistics (no training; the same clustering, its own tuned threshold), our trained
embedding, and our embedding told the true number of speakers (oracle K).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics.diarization import der  # noqa: E402
from subfix.paths import MODELS_DIR, RESULTS_DIR  # noqa: E402
from subfix.speaker import MFCCStats, OwnDiarizer, SpeakerEmbedder  # noqa: E402
from train_aligner import manifest  # noqa: E402

SR = 16000
OUT = RESULTS_DIR / "diarize"


def conversation(rows: list[dict], k: int, seed: int, target: float = 150.0):
    """(audio, reference turns) of one synthetic k-speaker conversation."""
    import soundfile as sf
    rng = random.Random(seed)
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(Path(r["path"]).name.split("-")[0], []).append(r)
    spk = rng.sample(sorted(by), k)
    queues = {s: rng.sample(by[s], len(by[s])) for s in spk}
    gain = {s: 10 ** rng.uniform(-0.15, 0.15) for s in spk}
    pos = {s: [0, None, 0] for s in spk}           # utterance index, loaded audio, sample offset

    def take(s, n):
        out = []
        while n > 0:
            p = pos[s]
            if p[1] is None:
                p[1], _ = sf.read(str(ROOT / queues[s][p[0] % len(queues[s])]["path"]), dtype="float32")
                p[2] = 0
            part = p[1][p[2]:p[2] + n]
            out.append(part)
            n -= len(part)
            p[2] += len(part)
            if p[2] >= len(p[1]):
                p[0], p[1] = p[0] + 1, None
        return np.concatenate(out) * gain[s]

    audio, ref, t, prev = [], [], 0.0, None
    while t < target:
        s = rng.choice([x for x in spk if x != prev])
        dur = min(max(rng.expovariate(1 / 5.0), 1.0), 15.0)
        seg = take(s, int(dur * SR))
        pause = np.zeros(int(rng.uniform(0.15, 1.0) * SR), np.float32)
        audio += [pause, seg]
        t += len(pause) / SR
        ref.append((t, t + len(seg) / SR, s))
        t += len(seg) / SR
        prev = s
    audio.append(np.zeros(int(0.5 * SR), np.float32))
    return np.concatenate(audio).astype(np.float32), ref


def dataset(split: str, per_k: int, seed: int):
    rows = manifest(split, max_dur=40.0)
    return [(k, *conversation(rows, k, seed * 1000 + 10 * k + i)) for k in (2, 3, 4, 5) for i in range(per_k)]


def score(diarizer, embedded, data, threshold=None, oracle=False):
    rows = []
    for e, (k, _, ref) in zip(embedded, data):
        hyp = diarizer.decode(e, num_speakers=k if oracle else 0, threshold=threshold)
        d = der(ref, [(t.start, t.end, t.speaker) for t in hyp.turns])
        d.update(k=k, found=len(hyp.speakers))
        rows.append(d)
    return rows


def total(rows) -> dict:
    tot = sum(r["ref_speech"] for r in rows)
    parts = {p: sum(r[p] for r in rows) / tot for p in ("miss", "fa", "conf")}
    return {"der": round(sum(parts.values()) * 100, 2), **{p: round(v * 100, 2) for p, v in parts.items()},
            "count_exact": round(float(np.mean([r["found"] == r["k"] for r in rows])), 3),
            "count_abs_err": round(float(np.mean([abs(r["found"] - r["k"]) for r in rows])), 2), "n": len(rows)}


def tune(diarizer, embedded, data, grid) -> tuple[float, list]:
    curve = [(round(th, 3), total(score(diarizer, embedded, data, th))) for th in grid]
    best = min(curve, key=lambda c: (c[1]["der"], c[1]["count_abs_err"]))
    return best[0], curve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["tune", "test"], required=True)
    ap.add_argument("--per-k", type=int, default=None, help="conversations per speaker count (dev 8, test 15)")
    ap.add_argument("--model", default=str(MODELS_DIR / "speaker" / "model.pt"))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    split, per_k, seed = ("dev-clean", args.per_k or 8, 1) if args.stage == "tune" else ("test-clean", args.per_k or 15, 2)
    data = dataset(split, per_k, seed)
    hours = sum(len(a) for _, a, _ in data) / SR / 3600
    print(f"[diar] {split}: {len(data)} conversations, {hours:.2f} h", flush=True)
    own = OwnDiarizer(SpeakerEmbedder(args.model, device))
    mfcc = OwnDiarizer(MFCCStats())
    emb_own = [own.embed(a, SR) for _, a, _ in data]
    emb_mfcc = [mfcc.embed(a, SR) for _, a, _ in data]
    tuned_path = OUT / "tuned.json"
    if args.stage == "tune":
        th_own, curve_own = tune(own, emb_own, data, np.arange(0.20, 1.21, 0.05))
        th_mfcc, curve_mfcc = tune(mfcc, emb_mfcc, data, np.arange(0.20, 1.21, 0.05))
        print(f"[diar] own threshold {th_own}: {dict(curve_own)[th_own]}", flush=True)
        print(f"[diar] mfcc threshold {th_mfcc}: {dict(curve_mfcc)[th_mfcc]}", flush=True)
        tuned_path.write_text(json.dumps({"own": th_own, "mfcc": th_mfcc, "curve_own": curve_own,
                                          "curve_mfcc": curve_mfcc}, indent=1), encoding="utf-8")
        ck = torch.load(args.model, map_location="cpu", weights_only=False)
        ck["threshold"] = th_own
        torch.save(ck, args.model)
        return
    tuned = json.loads(tuned_path.read_text(encoding="utf-8"))
    systems = {"MFCC statistics (untrained)": score(mfcc, emb_mfcc, data, tuned["mfcc"]),
               "Own speaker model": score(own, emb_own, data, tuned["own"]),
               "Own speaker model, oracle K": score(own, emb_own, data, oracle=True)}
    res = {name: {"all": total(rows), **{f"k{k}": total([r for r in rows if r["k"] == k]) for k in (2, 3, 4, 5)}}
           for name, rows in systems.items()}
    (OUT / "metrics.json").write_text(json.dumps({"split": split, "conversations": len(data), "hours": round(hours, 2),
                                                  "thresholds": {"own": tuned["own"], "mfcc": tuned["mfcc"]},
                                                  "results": res}, indent=1), encoding="utf-8")
    lines = [f"# Speaker diarization on synthetic {split} conversations", "",
             f"{len(data)} conversations ({hours:.2f} h), 2-5 unseen speakers each, no overlapping speech; "
             "DER with a 0.25 s collar. Thresholds tuned on dev-clean conversations.", "",
             "| System | DER % | Miss % | FA % | Confusion % | Speaker count exact | Count error |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, r in res.items():
        a = r["all"]
        lines.append(f"| {name} | {a['der']} | {a['miss']} | {a['fa']} | {a['conf']} | {a['count_exact']:.0%} | {a['count_abs_err']} |")
    lines += ["", "DER % by number of speakers:", "", "| System | 2 | 3 | 4 | 5 |", "|---|---:|---:|---:|---:|"]
    for name, r in res.items():
        lines.append(f"| {name} | " + " | ".join(str(r[f'k{k}']['der']) for k in (2, 3, 4, 5)) + " |")
    (OUT / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
