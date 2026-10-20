"""Collect every number the report shows from the repository itself (checkpoints, metrics files,
data caches, results tables, source files), so the PDF can be rebuilt after any retraining."""
from __future__ import annotations

import datetime as _dt
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

MODELS = ROOT / "models"
RESULTS = ROOT / "results"

# Facts measured while training that are not stored in the model folders (from the training logs
# of the session that trained the models, 2026-10-18). Wall-clock times on the RTX 4050 laptop.
TRAINING_LOG = {
    "lm": {"train_seconds": 114, "rare_types": 29_293, "bits_per_word": 16,
           "discounts": {1: [0.146, 1.746, 2.618], 2: [0.738, 1.073, 1.379], 3: [0.849, 1.152, 1.355],
                         4: [0.914, 1.288, 1.451]},
           "histories": {1: 1, 2: 55_756, 3: 1_644_956, 4: 4_613_916}},
    "pos": {"train_sentences": 54_473, "heldout_sentences": 2_867,
            "train_accuracy": [0.9508, 0.9632, 0.9712, 0.9756, 0.9788]},
    "punct": {"minutes_per_epoch": 4.5, "total_minutes": 41, "prep_minutes": 2.5},
    "aligner": {"planned_epochs": 15, "wall_clock": "2 h 56 min", "epoch_minutes": "8-21",
                "pilot": {"utterances": 11_353, "hours": 40.6, "speakers": 102, "epochs": 2}},
    "corrector": {"train_sentences": 513_355, "confusion_entries": 268, "batches_per_epoch": 2_297,
                  "total_minutes": 76, "bpe_train_seconds": 87},
}

# Pretrained models the project used before (public parameter counts).
OLD_MODELS = [
    {"role": "Fusion language model", "old": "distilgpt2", "old_params": 82e6, "new_key": "lm"},
    {"role": "Punctuation restoration", "old": "FullStop XLM-RoBERTa-large", "old_params": 560e6, "new_key": "punct"},
    {"role": "Grammar / ASR-error corrector", "old": "vennify T5-base (+ CoEdIT-large 770M)", "old_params": 220e6,
     "new_key": "corrector"},
    {"role": "Part-of-speech tags for line breaks", "old": "spaCy en_core_web_sm (12 MB CNN pipeline)",
     "old_params": None, "new_key": "pos"},
    {"role": "Forced alignment (word timing)", "old": "MMS-300M wav2vec2 aligner", "old_params": 300e6,
     "new_key": "aligner"},
]


def _json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _size_mb(path: Path) -> float:
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 2**20
    return path.stat().st_size / 2**20 if path.exists() else 0.0


def _layers(module) -> list[tuple[str, str, int]]:
    """(name, description, parameters) for the direct children of a torch module."""
    rows = []
    for name, child in module.named_children():
        n = sum(p.numel() for p in child.parameters())
        rows.append((name, type(child).__name__, n))
    return rows


# ------------------------------------------------------------------ neural models
def punct_facts() -> dict | None:
    path = MODELS / "punct" / "model.pt"
    if not path.exists():
        return None
    import numpy as np
    import torch
    from subfix.textfix.punct import CASES, CHARS, LABELS, MAX_CHARS, build_net
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = build_net(len(ck["vocab"]), **ck.get("net", {}))
    net.load_state_dict(ck["model"])
    metrics = _json(MODELS / "punct" / "metrics.json") or {}
    data = {}
    cache = ROOT / "cache" / "train" / "punct_data.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        data = {"train_words": int(len(z["tid"])), "dev_words": int(len(z["dtid"])), "types": int(len(z["chars"])),
                "train_labels": np.bincount(z["lab"], minlength=4).tolist(),
                "dev_labels": np.bincount(z["dlab"], minlength=4).tolist()}
    return {"params": sum(p.numel() for p in net.parameters()), "layers": _layers(net),
            "vocab": len(ck["vocab"]), "labels": LABELS, "cases": CASES, "chars": len(CHARS) + 2,
            "max_chars": MAX_CHARS, "metrics": metrics, "data": data, "size_mb": _size_mb(path),
            **TRAINING_LOG["punct"]}


def corrector_facts() -> dict | None:
    path = MODELS / "corrector" / "model.pt"
    if not path.exists():
        return None
    import torch
    from subfix.bpe import BPE
    from subfix.textfix.seq2seq import DEFAULT_NET, build_seq2seq
    bpe = BPE.load(MODELS / "corrector" / "bpe.json")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net_cfg = ck.get("net", DEFAULT_NET)
    model = build_seq2seq(len(bpe), **net_cfg)
    model.load_state_dict(ck["model"])
    sample = "The recieved subtitles were uncorrected."
    return {"params": sum(p.numel() for p in model.parameters()), "layers": _layers(model), "net": net_cfg,
            "enc_layer_params": sum(p.numel() for p in model.encoder.layers[0].parameters()),
            "dec_layer_params": sum(p.numel() for p in model.decoder.layers[0].parameters()),
            "bpe_vocab": len(bpe), "bpe_alphabet": len(bpe.alphabet), "bpe_merges": len(bpe.merges),
            "bpe_sample": (sample, [bpe.vocab[i] for i in bpe.encode(sample)]),
            "metrics": _json(MODELS / "corrector" / "metrics.json") or {}, "size_mb": _size_mb(path),
            **TRAINING_LOG["corrector"]}


def aligner_facts() -> dict:
    import torch
    from subfix.acoustic import FRAME_S, HOP, N_FFT, N_MELS, SR, TOKENS, build_acoustic
    path = MODELS / "aligner" / "model.pt"
    net_cfg: dict = {}
    trained = path.exists()
    shift = 0.0
    if trained:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        net_cfg = ck.get("net", {}) or {}
        shift = float(ck.get("onset_shift", 0.0))
    model = build_acoustic(**net_cfg)
    if trained:
        model.load_state_dict(ck["model"])
    manifest = _json(ROOT / "cache" / "train" / "aligner_train-clean-100.json") or {}
    rows = manifest.get("rows", [])
    rows20 = [r for r in rows if r["dur"] <= 20.0]
    speakers = {r["path"].replace("\\", "/").split("/")[-3] for r in rows}
    return {"trained": trained, "params": sum(p.numel() for p in model.parameters()), "layers": _layers(model),
            "tokens": TOKENS, "sr": SR, "n_fft": N_FFT, "hop": HOP, "n_mels": N_MELS, "frame_s": FRAME_S,
            "metrics": _json(MODELS / "aligner" / "metrics.json") if trained else None, "onset_shift": shift,
            **TRAINING_LOG["aligner"],
            "data": {"utterances": len(rows20), "hours": sum(r["dur"] for r in rows20) / 3600,
                     "speakers": len(speakers), "full_split_utterances": 28_539, "full_split_hours": 100.6,
                     "full_split_speakers": 251},
            "size_mb": _size_mb(path) if trained else 0.0}


# ------------------------------------------------------------------ count-based models
def lm_facts() -> dict | None:
    rep = _json(MODELS / "lm" / "kn4.json")
    if rep is None:
        return None
    return {**rep, **TRAINING_LOG["lm"], "size_mb": _size_mb(MODELS / "lm" / "kn4.npz"),
            "entries": sum(rep[f"ngrams_{k}"] for k in range(1, rep["order"] + 1))}


def pos_facts() -> dict | None:
    rep = _json(MODELS / "pos" / "perceptron.metrics.json")
    if rep is None:
        return None
    return {**rep, **TRAINING_LOG["pos"], "size_mb": _size_mb(MODELS / "pos" / "perceptron.json.gz")}


# ------------------------------------------------------------------ results
NUM_COLS = ["WER", "CER", "ERRANT F0.5", "Edit F0.5", "Over-corr", "Punct F1", "Onset ms", "<=250 ms",
            "CPS<=17", "CPS<=20", "Lines ok", "RTF"]


def parse_summary(path: Path, section: str = "ALL") -> dict | None:
    """{'A': {system: {col: value}}, 'B': {...}} from the '## ALL' tables of a SUMMARY.md."""
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    m = re.search(rf"^## {re.escape(section)}\s*$(.*?)(?=^## |\Z)", text, re.M | re.S)
    if not m:
        return None
    out: dict[str, dict] = {}
    mode = None
    for line in m.group(1).splitlines():
        if line.startswith("Mode A"):
            mode = "A"
        elif line.startswith("Mode B"):
            mode = "B"
        if not line.startswith("|") or line.startswith("|---") or "WER %" in line or mode is None:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != len(NUM_COLS) + 1:
            continue
        name = cells[0].replace("**", "")
        vals = {}
        for col, c in zip(NUM_COLS, cells[1:]):
            try:
                vals[col] = float(c)
            except ValueError:
                vals[col] = None
        out.setdefault(mode, {})[name] = vals
    return out or None


ABL_COLS = ["WER", "CER", "ERRANT F0.5", "Edit F0.5", "Over-corr", "Punct F1", "Case acc", "Onset ms",
            "<=100 ms", "<=250 ms", "Word ms", "CPS<=17", "CPS<=20", "Lines ok", "All fmt ok", "RTF"]


def parse_ablation(path: Path) -> dict | None:
    """{system: {col: value}} from the '## ALL' table of an ablation results.md."""
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    m = re.search(r"^## ALL\s*$(.*?)(?=^## |\Z)", text, re.M | re.S)
    if not m:
        return None
    out = {}
    for line in m.group(1).splitlines():
        if not line.startswith("|") or line.startswith("|---") or "WER" in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != len(ABL_COLS) + 1:
            continue
        vals = {}
        for col, c in zip(ABL_COLS, cells[1:]):
            try:
                vals[col] = float(c)
            except ValueError:
                vals[col] = None
        out[cells[0]] = vals
    return out or None


def results_facts() -> dict:
    old = parse_summary(RESULTS / "pretrained_v1" / "SUMMARY.md")
    new = parse_summary(RESULTS / "SUMMARY.md")
    # the new table is final only if the test run happened after the models were trained
    run_time = _mtime(RESULTS / "test" / "metrics.json")
    model_time = max(_mtime(MODELS / k / f) for k, f in (("corrector", "model.pt"), ("punct", "model.pt"),
                                                         ("aligner", "model.pt")))
    fresh = new is not None and run_time > model_time and (MODELS / "aligner" / "model.pt").exists()
    tune_new = _json(RESULTS / "tuning" / "fusion_grid.json") or {}
    tune_old = _json(RESULTS / "pretrained_v1" / "tuning" / "fusion_grid.json") or {}
    return {"old": old, "new": new if fresh else None, "fresh": fresh,
            "tuning": {"new": tune_new.get("best"), "old": tune_old.get("best")},
            "timing_new": parse_ablation(RESULTS / "ablations" / "timing_source" / "results.md") if fresh else None,
            "timing_old": parse_ablation(RESULTS / "pretrained_v1" / "ablations" / "timing_source" / "results.md"),
            "gec_new": parse_ablation(RESULTS / "ablations" / "gec_model" / "results.md") if fresh else None}


# ------------------------------------------------------------------ code + tests
CODE_FILES = {
    "Kneser-Ney LM": ["src/subfix/ngram.py", "src/subfix/lm.py", "experiments/train_lm.py"],
    "Punctuation BiLSTM": ["src/subfix/textfix/punct.py", "experiments/train_punct.py"],
    "Transformer corrector + BPE": ["src/subfix/textfix/seq2seq.py", "src/subfix/bpe.py",
                                    "src/subfix/textfix/gec.py", "experiments/train_corrector.py"],
    "Perceptron POS tagger": ["src/subfix/postag.py", "experiments/train_pos.py"],
    "CTC acoustic aligner": ["src/subfix/acoustic.py", "src/subfix/timing.py", "experiments/train_aligner.py"],
    "Shared training corpus": ["experiments/corpus.py"],
    "Web backend (FastAPI)": [str(p.relative_to(ROOT)) for p in sorted((ROOT / "app" / "backend").glob("*.py"))],
    "Web frontend (React)": [str(p.relative_to(ROOT)) for p in sorted((ROOT / "frontend" / "src").rglob("*"))
                             if p.suffix in (".ts", ".tsx", ".css") and ".test." not in p.name],
}


def _lines(rel: str) -> int:
    try:
        return sum(1 for _ in (ROOT / rel).open(encoding="utf-8"))
    except OSError:
        return 0


def code_facts() -> dict:
    files = {k: {"files": len(v), "lines": sum(_lines(f) for f in v)} for k, v in CODE_FILES.items()}
    py_tests = sum(len(re.findall(r"^def test_", (p.read_text(encoding="utf-8")), re.M))
                   for p in (ROOT / "tests").glob("test_*.py"))
    ts_tests = sum(len(re.findall(r"^\s*it\(", p.read_text(encoding="utf-8"), re.M))
                   for p in (ROOT / "frontend" / "src").rglob("*.test.ts*"))
    src_lines = sum(_lines(str(p.relative_to(ROOT))) for p in (ROOT / "src" / "subfix").rglob("*.py"))
    return {"modules": files, "py_tests": py_tests, "ts_tests": ts_tests, "src_lines": src_lines}


def handover_notes() -> str | None:
    p = Path.home() / "AppData" / "Roaming" / "Claude" / "side-session-notes" / \
        "local_cc04439a-6442-4547-81ec-57246dc70584" / "models-handover.md"
    try:
        return p.read_text(encoding="utf-8")
    except OSError:
        return None


def collect() -> dict:
    f = {
        "date": _dt.date.today().strftime("%d %B %Y"),
        "lm": lm_facts(),
        "pos": pos_facts(),
        "punct": punct_facts(),
        "corrector": corrector_facts(),
        "aligner": aligner_facts(),
        "results": results_facts(),
        "code": code_facts(),
        "old_models": OLD_MODELS,
        "handover": handover_notes(),
    }
    f["ready"] = all(f[k] for k in ("lm", "pos", "punct", "corrector")) and f["aligner"]["trained"] \
        and f["results"]["fresh"]
    return f


if __name__ == "__main__":
    import pprint
    facts = collect()
    for k in ("punct", "corrector", "aligner"):
        if facts[k]:
            facts[k] = {kk: vv for kk, vv in facts[k].items() if kk not in ("metrics",)}
    pprint.pprint(facts, width=140, compact=True)
