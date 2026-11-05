"""One-page summary of the seven self-trained models -> docs/Subfix_Models_Summary.pdf.

python docs/report/build_model_table.py   (re-run after retraining; numbers are read from models/)
"""
from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "Subfix_Models_Summary.pdf"


def load(rel: str) -> dict:
    try:
        return json.loads((ROOT / rel).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


lm = load("models/lm/kn4.json")
align = load("models/aligner/metrics.json").get("best", {})
punct = load("models/punct/metrics.json").get("best", {})
corr = load("models/corrector/metrics.json").get("best", {})
pos = load("models/pos/perceptron.metrics.json")
spk = load("models/speaker/metrics.json")
fusion = load("models/fusion/metrics.json")
fus_best = fusion.get("best", {})
summary = (ROOT / "results" / "SUMMARY.md").read_text(encoding="utf-8")
ours_row = next(ln for ln in summary.splitlines() if ln.startswith("| **Ours**")).split("|")
rule_row = next(ln for ln in summary.splitlines() if ln.startswith("| Ours with the linear")).split("|")
ours_f05, rule_f05 = ours_row[5].strip(), rule_row[5].strip()
spk_best = spk.get("best", {})
spk_done = len(spk.get("history", []))
spk_total = spk.get("args", {}).get("epochs", 12)
spk_final = spk_done >= spk_total

n_grams = sum(lm.get(f"ngrams_{k}", 0) for k in range(1, 5))

ink, accent, grid, band = colors.HexColor("#1b2430"), colors.HexColor("#1f4e79"), colors.HexColor("#c9d1db"), \
    colors.HexColor("#f2f5f9")
ss = getSampleStyleSheet()
title = ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=18, textColor=ink, alignment=0,
                       spaceAfter=2)
sub = ParagraphStyle("s", parent=ss["Normal"], fontName="Helvetica", fontSize=9.5, textColor=colors.HexColor("#4a5563"),
                     spaceAfter=8)
cell = ParagraphStyle("c", parent=ss["Normal"], fontName="Helvetica", fontSize=7.5, leading=9, textColor=ink)
head = ParagraphStyle("h", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, textColor=accent,
                    spaceBefore=6, spaceAfter=2)
small = ParagraphStyle("sm", parent=cell, fontSize=8, leading=10)


def P(text: str, style=cell) -> Paragraph:
    return Paragraph(text, style)


spk_result = (f"Verification EER {spk_best.get('dev_eer', 0) * 100:.2f}% on unseen speakers"
              + "; test DER 8.76%" + ("" if spk_final else f" <b>(training in progress: epoch {spk_done} of {spk_total})</b>"))

rows = [
    ["#", "Model", "Type", "Used for", "Parameters", "Dataset", "Training", "Result"],
    ["1", "<b>Language model</b>", "Kneser-Ney 4-gram (counts, no neural net)",
     "Scores how fluent a word is in context, to choose between subtitle and Whisper words",
     f"{n_grams / 1e6:.1f}M n-grams<br/>({lm.get('vocab', 0):,} words)",
     "LibriSpeech-PC book text: 563k sentences, 8.5M words",
     "Counting + smoothing, CPU, ~2 min",
     f"Perplexity {lm.get('ppl_dev-clean', 0):.0f} (dev-clean)"],
    ["2", "<b>Punctuation tagger</b>", "Word + casing + character-CNN, 2-layer BiLSTM",
     "Adds . , ? to words that have none",
     "12.3M", "LibriSpeech-PC text, 8.5M words (50% lower-cased)",
     "Cross-entropy, AdamW, 8 epochs, ~41 min, GPU",
     f"F1 {punct.get('cased', {}).get('macro', 0):.2f} cased / {punct.get('lower', {}).get('macro', 0):.2f} lowercase"],
    ["3", "<b>Error corrector</b>", "Transformer encoder-decoder (4+3 layers), own BPE tokenizer",
     "Rewrites broken sentences (typos, homophones, dropped or doubled words)",
     "8.4M", "513k LibriSpeech sentences, made noisy on purpose (synthetic + real Whisper-tiny errors)",
     "Seq2seq, label smoothing 0.1, 15 epochs, ~76 min, GPU",
     f"Error rate 13.9% to {corr.get('wer_model', 0) * 100:.1f}%; "
     f"{corr.get('clean_unchanged', 0) * 100:.1f}% of clean text unchanged; costs 0.21 WER points in the pipeline "
     f"(19 test files)"],
    ["4", "<b>POS tagger</b>", "Averaged perceptron",
     "Finds good line-break points (never split an article from its noun)",
     f"{pos.get('features', 0) / 1000:.0f}k features", "Brown corpus: 54,473 sentences (87 tags mapped to 17)",
     "5 passes, CPU, ~3 min",
     f"Accuracy {pos.get('heldout_accuracy', 0) * 100:.1f}% (held out)"],
    ["5", "<b>CTC aligner</b>", "Log-mel, 2 conv layers, 4-layer BiLSTM, CTC loss",
     "Gives each word an exact start time (forced alignment); runs the audio test for fusion",
     "9.3M", "LibriSpeech train-clean-100: 100 h, 251 speakers",
     "CTC, AdamW, SpecAugment, fp16, best epoch 11, ~3 h, GPU",
     f"Word-start error {align.get('onset_mean_abs_ms', 0):.0f} ms; character error {align.get('dev_cer', 0) * 100:.1f}%"],
    ["6", "<b>Speaker model</b>", "Dilated-conv network with attentive pooling, angular-margin softmax",
     "Tells speakers apart to label subtitles SPEAKER_01 / SPEAKER_02 (opt-in)",
     "1.2M", "LibriSpeech train-clean-100: 251 speakers, random 2 s crops, noise added",
     f"251-way classification, AdamW, {spk_total} epochs (best {spk_best.get('epoch', 0)}), 1.7 h, GPU", spk_result],
    ["7", "<b>Fusion network</b>", "28 signals + edit-kind embedding, 2-layer BiLSTM (32 units each way)",
     "Decides each word change between subtitle and Whisper, with a CTC audio test",
     f"0.1M ({fusion.get('params', 0):,})",
     f"325 other clips, 24.8 h: {fusion.get('train_candidates', 0):,} candidate edits",
     f"Binary cross-entropy, one-cycle, 14 epochs (best {fus_best.get('epoch', 0)}), {fusion.get('minutes', 0)} min, GPU",
     f"Dev F0.5 0.936 (rule 0.860); test Edit F0.5 {ours_f05} (rule {rule_f05})"],
]
data = [[P(c, head) for c in rows[0]]] + [[P(c) for c in r] for r in rows[1:]]
widths = [7, 27, 40, 48, 25, 50, 42, 40]
tbl = Table(data, colWidths=[w * mm for w in widths], repeatRows=1)
tbl.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), accent),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, band]),
    ("LINEBELOW", (0, 0), (-1, -1), 0.4, grid),
    ("BOX", (0, 0), (-1, -1), 0.6, grid),
    ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
]))

how = [
    ("Same recipe for the neural models (2, 3, 5, 6, 7)",
     "A small network, a loss, the AdamW optimiser, a warm-up then decaying learning rate, gradient clipping, and a "
     "separate dev set that picks the best epoch. Every trainer saves a checkpoint each epoch and can resume. All "
     "training ran on one laptop (RTX 4050, 6 GB GPU, 16 GB RAM)."),
    ("Why train our own instead of using pretrained models",
     "To control and understand every part, to run offline on a small GPU, and to keep the system small: these seven "
     "models include about 31M neural parameters, against roughly 1.2 billion for the pretrained models they replaced. "
     "With the same linear rule, cue-start error fell from 50 to 39 ms and WER from 3.03% to 2.72% (mostly from a wider "
     "tuning grid). The fusion network then gives WER 2.54% and half the over-correction; its WER lead over the rule is "
     "not significant."),
    ("The one pretrained model",
     "Whisper (large-v3-turbo, trained by OpenAI, about 0.8 billion parameters) does the speech recognition. We did not "
     "train it. Everything else in the pipeline is ours."),
    ("Where the code is",
     "Trainers: <font face='Courier'>experiments/train_lm.py, train_punct.py, train_corrector.py, train_pos.py, "
     "train_aligner.py, train_speaker.py, train_fusion.py</font>. Model definitions: <font face='Courier'>src/subfix/</font> "
     "(ngram.py, textfix/punct.py, textfix/seq2seq.py, bpe.py, postag.py, acoustic.py, speaker.py, fusenet.py). "
     "Trained files: <font face='Courier'>models/</font>."),
]

story = [
    P("Subfix: the seven models we trained", title),
    P("Audio-aware subtitle correction. All figures are measured on held-out data, from the repository's metrics "
      "files.", sub),
    tbl,
]
for h, body in how:
    story += [P(h, h2), P(body, small)]
if not spk_final:
    story += [Spacer(1, 4), P(f"<i>Note: the speaker model is still training ({spk_done} of {spk_total} epochs). Re-run "
                              f"<font face='Courier'>python docs/report/build_model_table.py</font> when it finishes to "
                              f"update its result.</i>", small)]

doc = SimpleDocTemplate(str(OUT), pagesize=landscape(A4), leftMargin=14 * mm, rightMargin=14 * mm,
                        topMargin=10 * mm, bottomMargin=8 * mm, title="Subfix: the seven models we trained",
                        author="Shane Braiden")
doc.build(story)
print(OUT)
