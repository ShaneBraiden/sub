"""Silver-gold reference subtitles — imp.md 5.2.

1. Utterance segmentation: LibriSpeech clips keep their original utterance boundaries; TED talks
   are cut into <= 15 s utterances at pauses using provisional times from aligning the reference
   text to the cached Whisper words.
2. Montreal Forced Aligner 3 (conda env `mfa`, english_us_arpa) aligns the reference text per
   utterance -> word times (JSON output), mapped back to the reference tokens.
3. Reference cues are built from the timed reference words with the DP segmenter under the
   strict rules (42 chars, 2 lines, 17 CPS) and the standard cue timing rules.
4. As a proxy for the silver reference's own error, MFA word boundaries are compared with an
   independent CTC aligner (our own acoustic model) on the same text.

Fallback when MFA is unavailable: our CTC aligner (flagged in refs/<id>.meta.json).
Evaluation-only tool: MFA (pretrained english_us_arpa) and spaCy POS for the reference cue breaks
are kept here so the existing references stay reproducible; neither is used by the system.
(References built before 2026-10-18 recorded `agreement_vs_mms`, against the MMS aligner.)

Usage: python tools/make_refs.py [--ids ...] [--aligner mfa|ctc] [--jobs 4]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subfix import text as T  # noqa: E402
from subfix.align import align  # noqa: E402
from subfix.config import Config  # noqa: E402
from subfix.paths import DATA_DIR  # noqa: E402
from subfix.segment import pos_tags, segment  # noqa: E402
from subfix.subio import Word, tokenize, write_srt  # noqa: E402
from subfix.timing import plan_chunks, provisional_times  # noqa: E402
from subfix.utils.cache import load_json, save_json  # noqa: E402

REFS = DATA_DIR / "refs"
MFA_WORK = DATA_DIR / "mfa"
CONDA = os.environ.get("CONDA_EXE") or str(Path.home() / "Miniconda3" / "condabin" / "conda.bat")


def ref_tokens(clip: dict) -> list[str]:
    return tokenize(clip["text"])


def strict_config() -> Config:
    cfg = Config.load(use_tuned=False)
    return cfg.override({"segment.method": "dp", "segment.max_chars": 42, "segment.max_lines": 2,
                         "segment.max_cps": 17.0, "segment.use_pos": True})


# ------------------------------------------------------------------ utterances
def libri_utterances(clip: dict, toks: list[str]) -> list[dict]:
    out, k = [], 0
    for u in clip["utterances"]:
        n = len(tokenize(u["text"]))
        out.append({"start": u["start"], "end": u["end"], "tok": (k, k + n)})
        k += n
    assert k == len(toks), (k, len(toks))
    return out


def asr_provisional(clip: dict, toks: list[str], asr_words: list[Word]) -> np.ndarray:
    ops = align(toks, [w.text for w in asr_words])
    words = [Word(t) for t in toks]
    for op, i, j in ops:
        if op in ("match", "sub") and i is not None and j is not None:
            if op == "match" or T.canon(toks[i])[:3] == T.canon(asr_words[j].text)[:3]:
                words[i].start, words[i].end = asr_words[j].start, asr_words[j].end
    return provisional_times(words, clip["duration"])


def ted_utterances(clip: dict, toks: list[str], asr_words: list[Word], max_s: float = 15.0) -> list[dict]:
    prov = asr_provisional(clip, toks, asr_words)
    ranges = plan_chunks(prov, clip["duration"], max_s)
    utts = []
    for a, b in ranges:
        utts.append({"start": float(prov[a, 0]), "end": float(prov[b - 1, 1]), "tok": (a, b)})
    for k, u in enumerate(utts):   # pad into the pauses, never overlap
        prev_end = utts[k - 1]["end"] if k else 0.0
        nxt = utts[k + 1]["start"] if k + 1 < len(utts) else clip["duration"]
        u["start"] = max(prev_end, u["start"] - 0.25, 0.0)
        u["end"] = min(nxt - 0.01 if k + 1 < len(utts) else nxt, u["end"] + 0.25)
        if u["end"] <= u["start"]:
            u["end"] = u["start"] + 0.05
    return utts


# ------------------------------------------------------------------ MFA
def _tg_escape(s: str) -> str:
    return s.replace('"', '""')


def write_textgrid(path: Path, dur: float, utts: list[dict], toks: list[str], tier: str = "spk") -> None:
    ivs, t = [], 0.0
    for u in utts:
        a, b = u["tok"]
        words = [T.norm(x) for x in toks[a:b]]
        text = " ".join(w for w in words if w)
        if u["start"] > t + 1e-4:
            ivs.append((t, u["start"], ""))
        ivs.append((u["start"], u["end"], text))
        t = u["end"]
    if t < dur:
        ivs.append((t, dur, ""))
    lines = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "", "xmin = 0", f"xmax = {dur}",
             "tiers? <exists>", "size = 1", "item []:", "    item [1]:", '        class = "IntervalTier"',
             f'        name = "{tier}"', "        xmin = 0", f"        xmax = {dur}",
             f"        intervals: size = {len(ivs)}"]
    for k, (a, b, txt) in enumerate(ivs, 1):
        lines += [f"        intervals [{k}]:", f"            xmin = {a}", f"            xmax = {b}",
                  f'            text = "{_tg_escape(txt)}"']
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mfa_available() -> bool:
    try:
        r = subprocess.run([CONDA, "run", "-n", "mfa", "mfa", "version"], capture_output=True, text=True, timeout=300)
        return r.returncode == 0
    except Exception:
        return False


def run_mfa(jobs: list[dict], n_jobs: int = 4) -> dict[str, list[tuple[str, float, float]]]:
    corpus, out = MFA_WORK / "corpus", MFA_WORK / "aligned"
    shutil.rmtree(corpus, ignore_errors=True)
    shutil.rmtree(out, ignore_errors=True)
    corpus.mkdir(parents=True)
    for j in jobs:
        d = corpus / j["id"]
        d.mkdir()
        wav = Path(j["audio"])
        if not wav.is_absolute():
            wav = Path(__file__).resolve().parents[1] / wav
        shutil.copy(wav, d / f"{j['id']}.wav")
        write_textgrid(d / f"{j['id']}.TextGrid", j["duration"], j["utts"], j["toks"])
    cmd = [CONDA, "run", "--no-capture-output", "-n", "mfa", "mfa", "align", str(corpus), "english_us_arpa",
           "english_us_arpa", str(out), "--clean", "--overwrite", "--output_format", "json", "-j", str(n_jobs),
           "--beam", "100", "--retry_beam", "400"]
    print("[mfa]", " ".join(cmd[4:8]), "...", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    (MFA_WORK / "mfa_log.txt").write_text(r.stdout + "\n" + r.stderr, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError(f"MFA failed (see {MFA_WORK / 'mfa_log.txt'})")
    result = {}
    for j in jobs:
        cands = list(out.rglob(f"{j['id']}.json"))
        if not cands:
            continue
        data = json.loads(cands[0].read_text(encoding="utf-8"))
        entries = []
        for name, tier in data.get("tiers", {}).items():
            if name.endswith("words"):
                entries += [(e[2], float(e[0]), float(e[1])) for e in tier.get("entries", [])]
        entries.sort(key=lambda e: e[1])
        result[j["id"]] = [e for e in entries if e[0] and e[0] not in ("<eps>", "sil", "sp")]
    return result


def map_word_times(toks: list[str], aligned: list[tuple[str, float, float]], dur: float) -> tuple[list[Word], float]:
    words = [Word(t) for t in toks]
    ops = align(toks, [a[0] for a in aligned])
    hit = 0
    for op, i, j in ops:
        if op == "match" or (op == "sub" and aligned[j][0] in ("<unk>", "spn")):
            words[i].start, words[i].end = aligned[j][1], aligned[j][2]
            hit += 1
    prov = provisional_times(words, dur)
    for w, (s, e) in zip(words, prov):
        w.start, w.end = float(s), float(e)
    return words, hit / max(len(toks), 1)


def ctc_word_times(clip: dict, toks: list[str], backend: str, utts: list[dict]) -> list[Word]:
    from subfix.timing import word_times
    cfg = Config.load(use_tuned=False)
    words = [Word(t) for t in toks]
    for u in utts:   # utterance bounds as provisional times
        a, b = u["tok"]
        sub = provisional_times(words[a:b], u["end"] - u["start"])
        for w, (s, e) in zip(words[a:b], sub):
            w.start, w.end = u["start"] + float(s), u["start"] + float(e)
    wav = str(ROOT / clip["audio"])
    timed, _ = word_times(words, wav, cfg, backend=backend)
    return timed


def agreement(a: list[Word], b: list[Word]) -> dict:
    d = np.array([abs(x.start - y.start) for x, y in zip(a, b)])
    return {"mean_abs_start_ms": float(d.mean() * 1000), "median_ms": float(np.median(d) * 1000),
            "within_50": float((d <= 0.05).mean()), "within_100": float((d <= 0.1).mean())}


def agreement_only(clips: list[dict]) -> None:
    from subfix.asr import transcribe
    rows = []
    for c in clips:
        meta_p = REFS / f"{c['id']}.meta.json"
        words_p = REFS / f"{c['id']}.words.json"
        if not (meta_p.exists() and words_p.exists()):
            continue
        meta = load_json(meta_p)
        if meta.get("aligner") != "mfa":
            continue
        ref = [Word(w["text"], w["start"], w["end"]) for w in load_json(words_p)]
        toks = [w.text for w in ref]
        if c.get("utterances"):
            utts = libri_utterances(c, toks)
        else:
            utts = ted_utterances(c, toks, transcribe(str(ROOT / c["audio"]), model="large-v3-turbo").words)
        ctc = ctc_word_times(c, toks, "ctc", utts)
        meta["agreement_vs_ctc"] = agreement(ref, ctc)
        save_json(meta_p, meta)
        rows.append((c["id"], meta["agreement_vs_ctc"]))
        print(f"[agreement] {c['id']}: {meta['agreement_vs_ctc']}", flush=True)
    if rows:
        import numpy as np
        keys = ["mean_abs_start_ms", "median_ms", "within_50", "within_100"]
        avg = {k: float(np.mean([r[1][k] for r in rows])) for k in keys}
        save_json(ROOT / "results" / "reference_agreement.json", {"files": dict(rows), "mean": avg})
        print("[agreement] mean over files:", avg)


def spacy_pos(words: list[Word], nlp=None) -> list[str]:
    """UD tags from spaCy for the reference cue breaks (lexical fallback without spaCy)."""
    if nlp is None:
        return pos_tags(words)
    from spacy.tokens import Doc
    doc = Doc(nlp.vocab, words=[T.core(w.text) or w.text for w in words])
    for _, proc in nlp.pipeline:
        doc = proc(doc)
    return [t.pos_ for t in doc]


def build_reference(clip: dict, words: list[Word], cfg: Config, nlp=None) -> None:
    cues = segment(words, cfg, spacy_pos(words, nlp))
    REFS.mkdir(parents=True, exist_ok=True)
    write_srt(cues, REFS / f"{clip['id']}.srt")
    save_json(REFS / f"{clip['id']}.words.json",
              [{"text": w.text, "start": round(w.start, 3), "end": round(w.end, 3)} for w in words])


def main() -> None:
    sys.path.insert(0, str(ROOT / "tools"))
    from fetch_data import all_clips
    from subfix.asr import transcribe
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--aligner", default="mfa", choices=["mfa", "ctc"])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--asr-model", default="large-v3-turbo")
    ap.add_argument("--no-agreement", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--agreement-only", action="store_true",
                    help="score existing MFA references against our CTC aligner (reference-error proxy)")
    args = ap.parse_args()
    clips = [c for c in all_clips() if not args.ids or c["id"] in args.ids]
    if args.agreement_only:
        agreement_only(clips)
        return
    if not args.force:
        clips = [c for c in clips if not (REFS / f"{c['id']}.srt").exists()]
    if not clips:
        print("[refs] nothing to do")
        return
    cfg = strict_config()
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer"])
    except Exception:
        nlp = None
    jobs = []
    for c in clips:
        toks = ref_tokens(c)
        if c.get("utterances"):
            utts = libri_utterances(c, toks)
        else:
            asr = transcribe(str(ROOT / c["audio"]), model=args.asr_model)
            utts = ted_utterances(c, toks, asr.words)
        jobs.append({**c, "toks": toks, "utts": utts, "audio": c["audio"]})
    aligner = args.aligner
    if aligner == "mfa" and not mfa_available():
        print("[refs] MFA unavailable -> falling back to our CTC aligner (timing scores will be flagged)")
        aligner = "ctc"
    mfa_out = run_mfa(jobs, args.jobs) if aligner == "mfa" else {}
    for j in jobs:
        meta = {"id": j["id"], "aligner": aligner, "n_tokens": len(j["toks"])}
        if aligner == "mfa" and j["id"] in mfa_out:
            words, cov = map_word_times(j["toks"], mfa_out[j["id"]], j["duration"])
            meta["mfa_coverage"] = cov
        else:
            if aligner == "mfa":
                print(f"[refs] {j['id']}: no MFA output, using our CTC aligner")
                meta["aligner"] = "ctc"
            words = ctc_word_times(j, j["toks"], "ctc", j["utts"])
        if not args.no_agreement and meta["aligner"] == "mfa":
            try:
                ctc = ctc_word_times(j, j["toks"], "ctc", j["utts"])
                meta["agreement_vs_ctc"] = agreement(words, ctc)
            except Exception as exc:
                meta["agreement_vs_ctc"] = {"error": str(exc)}
        build_reference(j, words, cfg, nlp)
        save_json(REFS / f"{j['id']}.meta.json", meta)
        print(f"[refs] {j['id']}: {meta}", flush=True)


if __name__ == "__main__":
    main()
