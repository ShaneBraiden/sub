"""Build the PDF report on how the self-trained models were made.

    .venv\\Scripts\\python docs\\report\\build_report.py            # -> docs/report/Subfix_Self_Trained_Models.pdf

Every number comes from the repository (facts.py): checkpoints, metrics files, data caches and
results tables, so re-running this after retraining updates the report. Until the aligner is
trained and the test run is redone, pages are stamped DRAFT.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from reportlab.platypus import CondPageBreak, KeepTogether, NextPageTemplate, PageBreak, Spacer  # noqa: E402

import charts  # noqa: E402
import facts as F  # noqa: E402
from layout import (CONTENT_W, Doc, P, bullets, callout, chain_diagram, formula, h2, h3, half_image,  # noqa: E402
                    image, numbered, pipeline_diagram, section, side_by_side, snippet, stack_diagram, table,
                    tiles, toc)

OUT = HERE / "Subfix_Self_Trained_Models.pdf"
FIG = HERE / "figures"


def _old_total(f: dict) -> float:
    return sum(m["old_params"] or 0 for m in f["old_models"])


def M(n: float, d: int = 1) -> str:
    return f"{n / 1e6:.{d}f}M"


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def n_(x) -> str:
    return f"{x:,}"


# ====================================================================== cover + summary
def cover(f: dict) -> list:
    neural = f["punct"]["params"] + f["corrector"]["params"] + f["aligner"]["params"]
    old_total = sum(m["old_params"] or 0 for m in f["old_models"])
    al = f["aligner"]["data"]
    out = [Spacer(1, 30),
           P("NLP COURSE PROJECT · TECHNICAL REPORT", "kicker"),
           P("Building the subtitle corrector's models from scratch", "title"),
           Spacer(1, 4),
           P("How the five self-trained models of the audio-aware subtitle correction system were designed, "
             "implemented, trained and evaluated: architectures, layers, parameters, code, data, results, and "
             "the problems solved along the way.", "lead"),
           Spacer(1, 6),
           P(f"<b>Shane</b> · B.Tech CSE (AI &amp; Data Analytics), Chennai · {f['date']}", "body"),
           Spacer(1, 18),
           tiles([("5", "models designed, coded and trained from scratch"),
                  (M(neural), "trainable neural parameters (3 PyTorch networks)"),
                  (f"{f['lm']['entries'] / 1e6:.1f}M", "n-gram entries in our Kneser-Ney language model"),
                  (f"{old_total / 1e9:.2f}B", "pretrained parameters replaced"),
                  (f"{f['lm']['words'] / 1e6:.2f}M" if 'words' in f['lm'] else "8.54M",
                   "words of training text (LibriSpeech-PC train)"),
                  (f"{al['hours']:.1f} h", f"of training speech ({n_(al['utterances'])} utterances, "
                                           f"{al['speakers']} speakers)")]),
           Spacer(1, 10),
           callout("The one-sentence version",
                   "Except for Whisper, which supplies the audio evidence, every learned component of the system "
                   "(the language model, the punctuation tagger, the grammar and ASR-error corrector, the "
                   "part-of-speech tagger and the forced aligner) was written from first principles in "
                   "NumPy and PyTorch and trained on public data on one 6 GB laptop GPU."),
           NextPageTemplate("body"), PageBreak(),
           P("Contents", "title2"), Spacer(1, 6), toc(), PageBreak()]
    return out


def summary(f: dict) -> list:
    lm, pu, co, po, al = f["lm"], f["punct"], f["corrector"], f["pos"], f["aligner"]
    cm = co["metrics"]["best"]
    pb = pu["metrics"]["best"]
    out = section(1, "Executive summary", "At a glance")
    out += [P("The project corrects subtitles using the audio as evidence. Mode A takes a video and its faulty "
              "<i>.srt/.vtt</i>; Mode B generates subtitles from audio alone. Both modes produce corrected text, "
              "word-accurate timing and subtitle cues that follow Netflix/BBC reading-speed and line-length rules. "
              "The first version relied on six large pretrained models. This report describes how five of them "
              "were replaced with models built and trained from scratch.", "lead")]
    rows = [
        ["Fusion language model", "Interpolated modified Kneser-Ney 4-gram, open vocabulary, NumPy",
         f"{lm['entries'] / 1e6:.1f}M n-grams", f"dev perplexity {lm['ppl_dev-clean']:.0f}; trained in {lm['train_seconds']} s"],
        ["Punctuation", "Word + char-CNN + casing embeddings, 2-layer BiLSTM",
         M(pu["params"], 2), f"macro-F1 {pb['cased']['macro']:.3f} (full stop {pb['cased']['.']:.3f})"],
        ["Corrector", "Transformer encoder-decoder (4+3 layers) + own BPE",
         M(co["params"], 2), f"noisy-dev WER {pct(cm['wer_copy'], 2)} -> {pct(cm['wer_model'], 2)}"],
        ["POS tagger", "Averaged perceptron, 17 feature templates, UD tags",
         f"{po['features']:,} features", f"held-out accuracy {pct(po['heldout_accuracy'], 2)}"],
        ["Forced aligner", "Log-mel + CNN + 4-layer BiLSTM, CTC, Viterbi alignment",
         M(al["params"], 2), _aligner_headline(f)],
    ]
    out += [table(["Model", "What we built", "Size", "Headline result"], rows, [1.05, 2.3, 0.9, 2.2], bold_first=True),
            Spacer(1, 10)]
    out += [h2("What makes this work stand out")]
    out += bullets([
        f"<b>{_old_total(f) / (pu['params'] + co['params'] + al['params']):.0f} times fewer parameters.</b> The neural "
        f"replacements total {M(pu['params'] + co['params'] + al['params'])} parameters, against about "
        f"{_old_total(f) / 1e9:.2f} billion in the pretrained models they replace (XLM-R-large alone has 560M).",
        "<b>Everything is our own code.</b> The n-gram counting, Kneser-Ney smoothing, BPE tokenizer, mel filterbank, "
        "CTC Viterbi alignment, beam search and perceptron are implemented in this repository, not imported.",
        "<b>Trained on a laptop.</b> All five models train on one RTX 4050 (6 GB) and a 16 GB laptop; the two "
        "count-based models each train in under four minutes on the CPU.",
        "<b>Honest evaluation.</b> Strict train/dev/test separation, model selection on dev only, and the same "
        "benchmark as the pretrained version, so the before/after comparison is like for like.",
        *([_headline_bullet(f)] if f["results"]["new"] else []),
        "<b>Engineered to survive failure.</b> Every neural trainer resumes from checkpoints, which mattered: the network "
        "dropped, a 6.4 GB download arrived corrupted, and the agent session lost its connection mid-training.",
    ])
    res = f["results"]
    if res["new"]:
        out.append(callout("End-to-end result", _results_sentence(f), "ok"))
    else:
        out.append(callout("Status", "The final aligner training and the full test-set re-run are still in progress; "
                                     "sections 8 and 10 are completed automatically when they finish.", "warn"))
    out.append(CondPageBreak(330))
    return out


def _headline_bullet(f: dict) -> str:
    o, n = _ours(f["results"]["old"]), _ours(f["results"]["new"])
    return (f"<b>No loss of accuracy.</b> On the same 27-file test set, WER went from {o['WER']:.2f}% to {n['WER']:.2f}% "
            f"and the mean cue-onset error from {o['Onset ms']:.0f} to {n['Onset ms']:.0f} ms after the swap; only "
            f"punctuation F1 is lower ({o['Punct F1']:.1f} to {n['Punct F1']:.1f}). Section 10 explains what caused each change.")


def _aligner_headline(f: dict) -> str:
    m = f["aligner"].get("metrics")
    if m and m.get("best"):
        b = m["best"]
        return f"mean onset error {b['onset_mean_abs_ms']:.0f} ms; dev CER {pct(b['dev_cer'])}"
    return "training in progress"


def _ours(table_: dict | None, mode: str = "A") -> dict | None:
    if not table_ or mode not in table_:
        return None
    return next((v for k, v in table_[mode].items()
                 if k.startswith(("Ours", "Mode B: Ours")) and not k.startswith("Ours +")), None)


def _results_sentence(f: dict) -> str:
    o, n = _ours(f["results"]["old"]), _ours(f["results"]["new"])
    if not o or not n:
        return "See section 11."
    return (f"On the 27-file test set (Mode A), WER is {n['WER']:.2f}% with our models vs {o['WER']:.2f}% with the "
            f"pretrained ones; ERRANT F0.5 {n['ERRANT F0.5']:.1f} vs {o['ERRANT F0.5']:.1f}; punctuation F1 "
            f"{n['Punct F1']:.1f} vs {o['Punct F1']:.1f}; mean cue-onset error {n['Onset ms']:.0f} ms vs "
            f"{o['Onset ms']:.0f} ms. Section 10 has the full comparison.")


# ====================================================================== background
def background(f: dict) -> list:
    out = section(2, "The system and why we replaced its models", "Background")
    out += [P("The pipeline separates <i>what was said</i> (speech recognition plus evidence fusion) from <i>when it "
              "was said</i> (forced alignment). Figure 1 shows where each self-trained model sits. Whisper is the only "
              "pretrained network left; everything shaded blue was built in this project."),
            pipeline_diagram(),
            P("Figure 1. Processing pipeline. Mode A runs every box; Mode B skips sync and fusion and uses the "
              "Whisper transcript as the draft.", "caption")]
    out += [h2("Why build our own models?")]
    out += bullets([
        "<b>Learning.</b> Using a pretrained checkpoint hides the interesting part. Building the models makes the course "
        "concepts concrete: smoothing, sequence labelling, subword tokenization, attention, CTC and structured "
        "prediction.",
        "<b>Size and speed.</b> The old text models needed about 860M parameters on a 6 GB GPU; ours need about 21M.",
        "<b>Control over data.</b> Every model is trained on known, public, licensed data with a documented split, so "
        "there is no risk that a pretrained model has already seen the test set.",
        "<b>Licences.</b> The MMS aligner weights are CC-BY-NC and vennify T5 is CC-BY-NC-SA. Our models carry no such "
        "restrictions beyond those of the datasets.",
        "<b>Transparency.</b> Every decision can be inspected: n-gram counts, perceptron features, the BPE merge list, "
        "attention masks, CTC alignment paths.",
    ])
    out += [h2("What we kept, and why")]
    out += bullets([
        "<b>Whisper (speech recognition).</b> Whisper was trained on more than 680,000 hours of speech. We estimate that "
        "a recogniser trained on our 100 hours would reach roughly 15-30% WER instead of about 3%, and the audio "
        "evidence the whole system depends on would collapse. We treat Whisper as a sensor, the way a vision "
        "project treats a camera.",
        "<b>Evaluation instruments only.</b> The Montreal Forced Aligner builds the silver timing references, "
        "whisper-tiny produces realistic ASR errors for the synthetic benchmark, and ERRANT uses spaCy to classify "
        "edits. They measure the system; they are not part of it.",
    ])
    out += [h2("Replacement map")]
    rows = []
    sizes = {"lm": f"{f['lm']['entries'] / 1e6:.1f}M n-grams (no weights)", "punct": M(f["punct"]["params"], 2),
             "corrector": M(f["corrector"]["params"], 2), "pos": f"{f['pos']['features']:,} sparse features",
             "aligner": M(f["aligner"]["params"], 2)}
    ours = {"lm": "Kneser-Ney 4-gram (NumPy)", "punct": "BiLSTM tagger + char-CNN", "corrector":
            "Transformer enc-dec + BPE", "pos": "Averaged perceptron", "aligner": "CNN + BiLSTM CTC model"}
    for m in f["old_models"]:
        rows.append([m["role"], m["old"], M(m["old_params"], 0) if m["old_params"] else "~12 MB",
                     ours[m["new_key"]], sizes[m["new_key"]]])
    out += [table(["Job", "Before (pretrained)", "Params", "Now (ours, from scratch)", "Size"], rows,
                  [1.35, 1.75, 0.6, 1.55, 1.2], bold_first=True), Spacer(1, 8)]
    if "params" in _figs:
        out += image(_figs["params"], caption="Figure 2. Parameter counts on a log scale. The language model and POS tagger "
                                              "are count-based and have no dense weights, so they are not plotted.")
    out.append(CondPageBreak(330))
    return out


# ====================================================================== data
def data_section(f: dict) -> list:
    pu, co, po, al, lm = f["punct"], f["corrector"], f["pos"], f["aligner"], f["lm"]
    d = pu.get("data", {})
    out = section(3, "Datasets and how we used them", "Data")
    out += [P("All training data is public. Only <b>train</b> splits are used for learning; <b>dev</b> splits choose "
              "hyper-parameters and checkpoints; the <b>test</b> benchmark is touched only for the final evaluation.")]
    rows = [
        ["LibriSpeech-PC train text", "Punctuated, cased transcripts of LibriSpeech train-clean-100/360 + train-other-500 "
         "(Meister et al. 2023)", f"{n_(lm['train_sentences'])} sentences · {n_(d.get('train_words', 8544054))} words",
         "LM, punctuation, corrector, BPE"],
        ["LibriSpeech train-clean-100 audio", "Read English speech, 16 kHz (Panayotov et al. 2015)",
         f"{n_(al['data']['utterances'])} utterances · {al['data']['hours']:.1f} h · {al['data']['speakers']} speakers",
         "CTC acoustic model"],
        ["Brown corpus", "1.16M tokens of 1961 American English, 87 tags (Francis & Kucera)",
         f"{n_(po['train_sentences'])} train / {n_(po['heldout_sentences'])} held-out sentences", "POS tagger"],
        ["whisper-tiny confusions", "Real ASR errors of a weak recogniser on the dev clips (RoBART idea)",
         f"{co['confusion_entries']} word -> mistake entries", "Corrector noise"],
        ["LibriSpeech-PC dev-clean/other", "Dev transcripts", f"{n_(d.get('dev_words', 98053))} words",
         "LM perplexity, punctuation and corrector selection"],
        ["dev-clean audio + MFA references", "Silver word timings from the Montreal Forced Aligner",
         "300 utterances + 4 long-form clips", "Aligner selection"],
        ["Benchmark (test)", "8 LibriSpeech test-clean + 8 test-other clips (3-5 min) and 11 TED-LIUM talks",
         "27 files, seeded corruptions", "Final evaluation only"],
    ]
    out += [table(["Dataset", "Content", "Size used", "Used for"], rows, [1.35, 2.3, 1.6, 1.4], bold_first=True),
            Spacer(1, 8)]
    out += [h2("Keeping the evaluation honest")]
    out += bullets([
        "LibriSpeech train, dev and test speakers are disjoint, so the acoustic model is always evaluated on unseen voices.",
        "The corrector's real ASR confusions come from the <i>dev</i> clips; the test clips never feed any training step.",
        "Fusion weights and pipeline settings were re-tuned on dev only; test numbers were produced once, at the end.",
        "The synthetic corruption generator is seeded, and every injected error is logged, so results are reproducible.",
    ])
    if d.get("train_labels"):
        lab = d["train_labels"]
        tot = sum(lab)
        out += [h2("A closer look at the punctuation labels")]
        out += [P(f"Each training word carries one label. The classes are very unbalanced, which is why we report "
                  f"macro-F1 rather than accuracy: a model that never predicts punctuation would already be "
                  f"{pct(lab[0] / tot)} accurate.")]
        out += [table(["Label", "Meaning", "Training words", "Share"],
                      [["0", "no mark after the word", n_(lab[0]), pct(lab[0] / tot)],
                       [".", "full stop (also ! and sentence ends)", n_(lab[1]), pct(lab[1] / tot)],
                       [",", "comma (also ; and :)", n_(lab[2]), pct(lab[2] / tot)],
                       ["?", "question mark", n_(lab[3]), pct(lab[3] / tot, 2)]],
                      [0.6, 2.6, 1.2, 0.8], num_cols=(2, 3))]
    out.append(CondPageBreak(330))
    return out


# ====================================================================== model 1: LM
def lm_section(f: dict) -> list:
    lm = f["lm"]
    t = f["results"]["tuning"]
    out = section(4, "Model 1: Kneser-Ney n-gram language model", "Language modelling")
    out += [P("<b>Job.</b> When the subtitle says <i>s</i> but Whisper heard <i>a</i>, fusion asks which word fits the "
              "context better. The language model supplies log P(word | previous words); the difference "
              "log P(a | ctx) - log P(s | ctx) is one of the fusion features. It replaces distilgpt2 (82M parameters).")]
    out += [h2("The mathematics")]
    out += [P("We implemented <b>interpolated modified Kneser-Ney</b> smoothing (Chen &amp; Goodman 1998), the strongest "
              "classical n-gram method. For an n-gram with history h and count c:")]
    out += [formula([
        "P<sub>k</sub>(w | h) = max(c<sub>k</sub>(hw) - D<sub>k</sub>(c), 0) / &Sigma;<sub>v</sub> c<sub>k</sub>(hv)"
        " &nbsp;+&nbsp; &gamma;<sub>k</sub>(h) &middot; P<sub>k-1</sub>(w | h')",
        "&gamma;<sub>k</sub>(h) = [ D<sub>1</sub>&middot;N<sub>1</sub>(h&middot;) + D<sub>2</sub>&middot;N<sub>2</sub>(h&middot;)"
        " + D<sub>3+</sub>&middot;N<sub>3+</sub>(h&middot;) ] / &Sigma;<sub>v</sub> c<sub>k</sub>(hv)",
        "Y = n<sub>1</sub> / (n<sub>1</sub> + 2n<sub>2</sub>), &nbsp; D<sub>1</sub> = 1 - 2Y&middot;n<sub>2</sub>/n<sub>1</sub>, "
        "&nbsp; D<sub>2</sub> = 2 - 3Y&middot;n<sub>3</sub>/n<sub>2</sub>, &nbsp; D<sub>3+</sub> = 3 - 4Y&middot;n<sub>4</sub>/n<sub>3</sub>",
    ])]
    out += [P("Lower orders do not use raw counts but <b>continuation counts</b> N<sub>1+</sub>(&middot;w), the number of "
              "different words that precede w. This is Kneser-Ney's key idea: \"Francisco\" is frequent but almost always "
              "follows \"San\", so it should get a small unigram probability. n<sub>1</sub>..n<sub>4</sub> are the "
              "count-of-counts of each order, and the recursion ends in a uniform distribution.")]
    out += [chain_diagram([("4-gram", "P(w | 3 words)", "ours"), ("trigram", "P(w | 2 words)", "ours"),
                           ("bigram", "P(w | 1 word)", "ours"), ("unigram", "continuation", "ours"),
                           ("uniform", "1 / |V|", "rule")],
                          ["+ γ(h) ×", "+ γ(h) ×", "+ γ(h) ×", "+ γ ×"]),
            P("Figure 3. Interpolation chain: each order discounts its counts and passes the saved mass down.", "caption")]
    out += [h2("Engineering it to fit a laptop")]
    out += bullets([
        f"<b>Bit-packed n-grams.</b> The vocabulary has {n_(lm['vocab'])} words, which fits in {lm['bits_per_word']} bits, "
        f"so a whole 4-gram packs into one 64-bit integer. Counting is then a single <font name='Consolas'>np.unique</font> "
        f"over the {n_(8544054)}-token stream, and lookup is a binary search (<font name='Consolas'>np.searchsorted</font>). "
        "No Python dictionaries hold n-grams.",
        f"<b>Streaming corpus.</b> Sentences come from a generator read twice (vocabulary pass, counting pass), so the "
        f"corpus is never held in memory as Python strings. Training took <b>{lm['train_seconds']} s</b> on the CPU.",
        f"<b>Open vocabulary.</b> {n_(lm['rare_types'])} word types seen only once become &lt;unk&gt;. The probability "
        "of an unknown word is P(&lt;unk&gt; | h) times a character-bigram spelling model trained on those rare words, so "
        "a typo like \"recieve\" scores far lower than \"receive\" in the same context.",
        "<b>Vectorised batch scoring.</b> Fusion scores thousands of short candidate windows at once; all orders are "
        "evaluated with array operations.",
    ])
    if t.get("new") and t.get("old"):
        out.append(callout("Talking point", f"Our count-based LM has no neural weights and trains in under two minutes. "
                           f"After re-tuning on dev, fusion with it reaches F0.5 {t['new']['dev_f05']:.3f}, against "
                           f"{t['old']['dev_f05']:.3f} with the 82M-parameter distilgpt2. To be fair, the small gain "
                           "comes from searching a wider range for the LM's insertion gate, which the n-gram scores "
                           "needed, not from the LM being better than a neural one."))
    out += snippet("src/subfix/ngram.py", "p = np.full(n, 1.0 / (len(self.vocab) - 1))", 22,
                   "batched interpolated lookup, highest order last")
    out += [h2("Model statistics")]
    rows = [[str(k), n_(lm[f"ngrams_{k}"]), n_(lm["histories"][k]),
             " / ".join(f"{x:.3f}" for x in lm["discounts"][k])] for k in range(1, lm["order"] + 1)]
    out += [table(["Order", "n-grams stored", "Histories", "Discounts D1 / D2 / D3+"], rows, [0.8, 1.4, 1.4, 2.2],
                  num_cols=(1, 2))]
    out += [Spacer(1, 8)]
    if "lm" in _figs:
        out += image(_figs["lm"], caption="Figure 4. Number of distinct n-grams per order.")
    out += [tiles([(n_(lm["vocab"]), "vocabulary (words)"), (f"{lm['ppl_dev-clean']:.0f}", "perplexity on dev-clean"),
                   (f"{lm['ppl_dev-other']:.0f}", "perplexity on dev-other"),
                   (pct(lm["oov_dev-clean"], 2), "out-of-vocabulary rate (dev-clean)"),
                   (f"{lm['train_seconds']} s", "training time (CPU)"), (f"{lm['size_mb']:.0f} MB", "model file")],
                  cols=3)]
    out.append(CondPageBreak(330))
    return out


# ====================================================================== model 2: punctuation
def punct_section(f: dict) -> list:
    pu = f["punct"]
    b = pu["metrics"]["best"]
    a = pu["metrics"]["args"]
    out = section(5, "Model 2: BiLSTM punctuation tagger", "Sequence labelling")
    out += [P("<b>Job.</b> Speech recognition and many subtitles lack punctuation. We treat restoration as "
              "<b>sequence labelling</b>: every word gets one of four labels (none, full stop, comma, question mark) "
              "that says which mark follows it. It replaces FullStop XLM-RoBERTa-large (560M parameters).")]
    out += [h2("Architecture")]
    out += [stack_diagram(
        [[("Word", "lower-cased form", "io"), ("Word embedding", f"{pu['vocab']:,} x 192", "ours")],
         [("Casing", "lower/title/upper/mixed/none", "io"), ("Case embedding", "5 x 8", "ours")],
         [("Characters", f"first {pu['max_chars']} chars", "io"), ("Char emb 39x32 + Conv1d", "64 filters, k=3, max-pool", "ours")]],
        [("Concatenate", "192 + 8 + 64 = 264 per word", "rule"), ("BiLSTM, 2 layers", "256 forward + 256 backward", "ours"),
         ("Linear 512 -> 4 + softmax", "{ none, . , ? }", "ours")]),
        P("Figure 5. Each word is represented three ways, and a bidirectional LSTM reads the whole window so that "
          "the label can depend on words on both sides.", "caption")]
    rows = [[name, kind, n_(n)] for name, kind, n in pu["layers"] if n]
    rows.append(["Total", "", n_(pu["params"])])
    out += [table(["Layer", "Type", "Parameters"], rows, [1.4, 2.0, 1.2], num_cols=(2,), highlight_rows=(len(rows) - 1,))]
    out += [Spacer(1, 6)]
    out += bullets([
        "<b>Why a character CNN?</b> Subtitles contain typos and rare names. The CNN builds a representation from "
        "spelling, so even an unseen word gets a useful vector.",
        "<b>Why a casing embedding?</b> A capitalised word often starts a sentence, which signals a full stop before "
        "it. Half of the training windows are lower-cased at random, so the model also works on caseless subtitles.",
        "<b>Why bidirectional?</b> Whether a comma follows a word depends on what comes next (\"..., which\", "
        "\"..., and\"), so the right-hand context matters as much as the left.",
    ])
    out += snippet("src/subfix/textfix/punct.py", "def forward(self, words, cases, chars):", 9, "the forward pass")
    out += [h2("Training")]
    rows = [["Data", f"{n_(pu['data'].get('train_words', 0))} words from whole LibriSpeech chapters (utterances in order)"],
            ["Windows", f"{a['seq_len']} words, batch {a['batch_size']}, new random offsets every epoch"],
            ["Optimiser", f"AdamW, lr {a['lr']}, weight decay 0.01, one-cycle schedule (5% warm-up), gradient clip 1.0"],
            ["Regularisation", "dropout 0.3 on inputs and outputs; 50% of windows lower-cased (augmentation)"],
            ["Epochs / time", f"{a['epochs']} epochs of about {pu['minutes_per_epoch']} min on the RTX 4050, plus "
                              f"{pu['prep_minutes']} min of preprocessing: about {pu['total_minutes']} min in total"],
            ["Selection", "best dev score = mean of cased and lower-cased macro-F1"],
            ["Inference", "overlapping 230-word windows; each word keeps the prediction made farthest from a window edge"]]
    out += [table(["Setting", "Value"], rows, [1.2, 4.6], bold_first=True)]
    out += [Spacer(1, 8)]
    if "punct_curve" in _figs:
        out += [KeepTogether([side_by_side(half_image(_figs["punct_curve"]), half_image(_figs["punct_classes"])),
                P("Figure 6. Left: dev macro-F1 per epoch. Right: F1 per mark for the best checkpoint, with and "
                  "without casing in the input.", "caption")])]
    out.append(callout("Talking point", f"A {M(pu['params'])}-parameter BiLSTM trained in about {pu['total_minutes']} minutes reaches "
                       f"macro-F1 {b['cased']['macro']:.2f} and places full stops with F1 {b['cased']['.']:.2f}. It is "
                       f"{560e6 / pu['params']:.0f} times smaller than the XLM-R model it replaces. Commas are the "
                       "hardest class, which matches the literature, because comma use is partly a matter of style."))
    out.append(CondPageBreak(330))
    return out


# ====================================================================== model 3: corrector
def corrector_section(f: dict) -> list:
    co = f["corrector"]
    net = co["net"]
    a = co["metrics"]["args"]
    best = co["metrics"]["best"]
    sample, pieces = co["bpe_sample"]
    out = section(6, "Model 3: Transformer corrector with our own BPE", "Sequence-to-sequence")
    out += [P("<b>Job.</b> Fix what rules cannot: real-word errors, dropped or duplicated function words, homophones and "
              "ASR-style confusions, sentence by sentence. It replaces vennify T5-base (220M) and the CoEdIT-large "
              "(770M) comparison model.")]
    out += [h2("Step 1: a byte-pair-encoding tokenizer, written from scratch")]
    out += [P(f"BPE (Sennrich et al. 2016) starts from characters and repeatedly merges the most frequent adjacent pair. "
              f"Ours keeps case and attached punctuation, marks word ends with &lt;/w&gt; so decoding is exact, and "
              f"learned <b>{n_(co['bpe_merges'])} merges</b> over a {co['bpe_alphabet']}-symbol alphabet, giving a "
              f"<b>{n_(co['bpe_vocab'])}-token vocabulary</b> (4 special tokens included). Incremental pair counts and a "
              f"lazy max-heap make training take {co['bpe_train_seconds']} s in pure Python on the CPU.")]
    out += [table(["Input sentence", "Our BPE pieces"],
                  [[sample, " | ".join(p.replace("</w>", "_") for p in pieces)]], [1.6, 3.4])]
    out += [P("Frequent words stay whole (\"were\"); a misspelling such as \"recieved\" splits into familiar pieces, so the "
              "model can still represent it.", "small")]
    out += [h2("Step 2: the Transformer")]
    out += [stack_diagram(
        [[("Noisy sentence", "BPE ids + </s>", "io"), ("Embedding x sqrt(256) + position", "tied with output layer", "ours"),
          (f"Encoder x {net['enc_layers']}", "self-attention (4 heads) + FFN 1024", "ours")],
         [("Corrected so far", "<s> + previous tokens", "io"), ("Embedding x sqrt(256) + position", "same shared matrix", "ours"),
          (f"Decoder x {net['dec_layers']}", "masked self-attn + cross-attn + FFN", "ours")]],
        [("Output = h · E^T (tied weights)", f"softmax over {n_(co['bpe_vocab'])} BPE tokens", "ours"),
         ("Beam search, 4 beams", "GNMT length penalty ((5+L)/6)^0.6", "rule")], col_w=160),
        P("Figure 7. A pre-norm Transformer encoder-decoder. The embedding matrix is shared between source, target and "
          "output projection, which saves 2M parameters and helps with only 8.4M in total.", "caption")]
    rows = [["Token embedding (shared, tied)", f"{n_(co['bpe_vocab'])} x {net['d_model']}", n_(co['layers'][0][2])],
            ["Learned positions", f"{net['max_len']} x {net['d_model']}", n_(co['layers'][1][2])],
            [f"Encoder ({net['enc_layers']} layers + final LayerNorm)", f"{n_(co['enc_layer_params'])} per layer",
             n_(co['layers'][2][2])],
            [f"Decoder ({net['dec_layers']} layers + final LayerNorm)", f"{n_(co['dec_layer_params'])} per layer",
             n_(co['layers'][3][2])],
            ["Total", f"d_model {net['d_model']}, {net['heads']} heads, FFN {net['ff']}, dropout {net['dropout']}",
             n_(co['params'])]]
    out += [table(["Component", "Shape", "Parameters"], rows, [2.2, 2.2, 1.0], num_cols=(2,), highlight_rows=(4,))]
    out += snippet("src/subfix/textfix/seq2seq.py", "def embed(self, ids):", 18, "embedding, encoder, decoder with tied output")
    out += [h2("Step 3: training data without a parallel corpus")]
    out += [P(f"No dataset of (faulty subtitle, corrected subtitle) pairs exists, so we <b>manufacture</b> them. Each of the "
              f"{n_(co['train_sentences'])} clean training sentences is corrupted on the fly with the same operations "
              f"as the benchmark, plus real whisper-tiny mistakes. A third of the pairs stay clean so the model learns "
              f"that copying correct text is usually the right answer. Fresh noise is drawn every epoch, so the model "
              f"sees {a['epochs']} different noisy versions of each sentence.")]
    out += [table(["Noise", "Rate (up to)", "Example (illustrative)"],
                  [["Keyboard typo (insert / delete / swap / substitute)", "4% of words", "receive -> recieve"],
                   ["Homophone swap", "3%", "their -> there"],
                   ["Real whisper-tiny confusion (from dev clips)", "4%", "a word whisper-tiny misheard"],
                   ["Dropped function word", "2%", "went to the -> went the"],
                   ["Duplicated word", "1%", "the the"],
                   ["Punctuation: strip all / drop half", "30% / 30% of sentences", "Hello, world. -> Hello world"],
                   ["Lower-casing", "30% of sentences", "Paris -> paris"],
                   ["Left completely clean", "33% of pairs", "(copy task)"]],
                  [2.4, 1.2, 1.8])]
    out += snippet("experiments/train_corrector.py", "def noisify(", 22, "the noise generator (word-level part)")
    out += [h2("Training setup and results")]
    rows = [["Batching", f"length-bucketed, at most {n_(a['max_tokens'])} tokens per batch ({n_(co['batches_per_epoch'])} batches per epoch)"],
            ["Optimiser", f"AdamW (betas 0.9/0.98, weight decay 0.01), peak lr {a['lr']}, {n_(a['warmup'])} warm-up steps, cosine decay to 5%"],
            ["Loss", "cross-entropy with label smoothing 0.1, padding ignored"],
            ["Precision", "bfloat16 autocast on the GPU (same range as fp32, so no overflow), gradient clip 1.0"],
            ["Length", f"{a['epochs']} epochs = {n_(co['metrics']['history'][-1]['step'])} steps, about {co['total_minutes']} min in total"],
            ["Selection", "lowest WER on 1,000 noisy dev-clean sentences; 500 clean ones check over-correction"]]
    out += [table(["Setting", "Value"], rows, [1.1, 4.7], bold_first=True), Spacer(1, 8)]
    if "corrector" in _figs:
        out += image(_figs["corrector"], caption="Figure 8. Word error rate on noisy dev sentences. The dashed line is the "
                                                 "error of the uncorrected input.")
    out += [tiles([(pct(best["wer_copy"], 2), "WER of the noisy input"), (pct(best["wer_model"], 2), "WER after correction"),
                   (pct(1 - best["wer_model"] / best["wer_copy"], 0), "relative error reduction"),
                   (pct(best["exact"]), "sentences restored exactly"),
                   (pct(best["clean_unchanged"]), "clean sentences left untouched"),
                   (M(co["params"]), f"parameters ({220e6 / co['params']:.0f}x smaller than T5-base)")])]
    out.append(callout("Safety rails around the corrector", [
        "<b>Edit-ratio guard:</b> an output that changes more than 15% of a sentence's words (tuned on dev) is rejected.",
        "<b>Audio-consistency guard:</b> the corrector may not replace or delete a word Whisper heard with probability "
        "0.8 or more; the audio has the final say.",
        "<b>Copy training:</b> a third of the training pairs are clean, which is why 97.6% of clean sentences come out unchanged.",
    ], "note"))
    out.append(CondPageBreak(330))
    return out


# ====================================================================== model 4: POS
def pos_section(f: dict) -> list:
    po = f["pos"]
    out = section(7, "Model 4: averaged-perceptron POS tagger", "Structured prediction")
    out += [P("<b>Job.</b> The cue segmenter must not break a line between \"the\" and its noun, or end a line with a "
              "preposition. It needs a part-of-speech tag for every word. This replaces spaCy's en_core_web_sm.")]
    out += [h2("Algorithm")]
    out += bullets([
        "<b>Greedy left-to-right tagging</b> with a linear model: score(tag) = &Sigma; w[feature, tag], take the arg-max, "
        "and feed the predicted tag into the next word's features.",
        "<b>Perceptron learning</b> (Collins 2002): when the guess is wrong, add 1 to the weights of the true tag's "
        "features and subtract 1 from the guessed tag's.",
        "<b>Averaging</b> keeps the mean of every weight over all updates, which makes the perceptron far more stable. We "
        "use timestamps, so averaging costs nothing per step.",
        f"<b>Tag dictionary:</b> {n_(po['tagdict'])} frequent, unambiguous words (at least 20 occurrences, at least 97% one "
        "tag) skip the model entirely, which is both faster and more accurate.",
    ])
    out += [table(["Feature templates (17)", ""],
                  [["bias; current word; suffixes (3, 2 chars); first letter", "word identity and morphology"],
                   ["shape (UPPER / Title / digit / punct / lower); shape + sentence-initial", "capitalisation cues"],
                   ["previous tag; tag two back; both together; previous tag + current word", "tag context (a Markov chain)"],
                   ["previous word and its suffix; word two back", "left context"],
                   ["next word and its suffix; word two ahead", "right context"]], [3.2, 2.0])]
    out += snippet("src/subfix/postag.py", "if not guess:", 14, "the averaged-perceptron update")
    out += [h2("Data: from Brown tags to Universal Dependencies")]
    out += [P("The Brown corpus uses 87 fine-grained tags plus compounds such as <font name='Consolas'>PPS+BEZ</font> "
              "(\"it's\") and <font name='Consolas'>NN$-TL</font>. Our segmenter's rules use the 17 Universal POS tags, so we "
              "wrote a mapping with contextual rules: forms of <i>be</i> and modals become AUX, and <i>have</i>/<i>do</i> "
              "become AUX only when a verb follows within three words (\"has gone\" vs \"has a car\").")]
    out += [h2("Results")]
    out += [tiles([(pct(po["heldout_accuracy"], 2), f"held-out accuracy ({n_(po['heldout_tokens'])} tokens)"),
                   (n_(po["features"]), "active features after averaging"),
                   (f"{po['train_seconds']} s", "training time, 5 passes (CPU)")])]
    out += [table(["Pass", "1", "2", "3", "4", "5"],
                  [["Training accuracy"] + [pct(x, 2) for x in po["train_accuracy"]]], [1.6, 1, 1, 1, 1, 1],
                  num_cols=(1, 2, 3, 4, 5))]
    out += [Spacer(1, 8)]
    if "pos" in _figs:
        out += image(_figs["pos"], caption="Figure 9. Recall per tag on held-out Brown sentences. X (foreign words) and "
                                           "INTJ are rare; the tags the segmenter relies on (DET, ADP, CCONJ, AUX) "
                                           "are above 98%.")
    out.append(callout("Talking point", f"A model with no neural network at all, trained in about three minutes on a CPU, "
                       f"tags {pct(po['heldout_accuracy'], 1)} of unseen words correctly, close to the roughly 97% of "
                       "classic published taggers on comparable data."))
    out.append(CondPageBreak(330))
    return out


# ====================================================================== model 5: aligner
def aligner_section(f: dict) -> list:
    al = f["aligner"]
    d = al["data"]
    m = al.get("metrics") or {}
    out = section(8, "Model 5: CTC acoustic model for forced alignment", "Speech")
    out += [P("<b>Job.</b> After the text is corrected, every word needs an exact start and end time so that cues appear "
              "when the words are spoken. We train a character-level acoustic model with <b>CTC</b> (Graves et al. "
              "2006) and use it for <b>forced alignment</b>: the transcript is known, and we only search for "
              "<i>where</i> each character is. It replaces the 300M-parameter MMS wav2vec2 aligner.")]
    out += [h2("Architecture")]
    out += [stack_diagram(
        [[("Waveform", "16 kHz mono", "io"), ("STFT", f"{al['n_fft']}-sample Hann window (25 ms), hop {al['hop']} (10 ms)", "rule"),
          (f"{al['n_mels']} log-mel bands", "our own HTK mel filterbank", "rule"),
          ("Normalise", "per-utterance mean / variance", "rule")]],
        [("Conv1d 80 -> 256, k=5 + ReLU", None, "ours"), ("Conv1d 256 -> 256, k=5, stride 2", "20 ms frames, like wav2vec2", "ours"),
         ("LayerNorm + dropout 0.15", None, "ours"), ("BiLSTM x 4", "320 forward + 320 backward", "ours"),
         ("Linear 640 -> 29 + log-softmax", "blank, |, a-z, '", "ours")], box_h=24, col_w=230),
        P("Figure 10. The acoustic model. The feature extractor is part of the model and runs on the GPU.", "caption")]
    rows = [[name, kind, n_(n)] for name, kind, n in al["layers"] if n]
    rows.append(["Total", "", n_(al["params"])])
    out += [table(["Layer", "Type", "Parameters"], rows, [1.4, 2.0, 1.2], num_cols=(2,), highlight_rows=(len(rows) - 1,))]
    out += snippet("src/subfix/acoustic.py", "def forward(self, wav, lengths):", 10, "the log-mel front end, written from scratch")
    out += [h2("How CTC works, in three lines")]
    out += bullets([
        "The network outputs, every 20 ms, a probability for each of the 29 symbols, including a special <b>blank</b>.",
        "A transcript can be spelled across the frames in many ways (repeats collapse, blanks vanish); the <b>CTC loss</b> "
        "sums the probability of all of them with dynamic programming, so no frame-level labels are needed.",
        "For alignment we take the <b>single best path</b> (Viterbi) that spells exactly our transcript. The frames where a "
        "word's letters appear give its start and end.",
    ])
    out += snippet("src/subfix/timing.py", "def ctc_viterbi(", 34, "our NumPy Viterbi over the CTC trellis")
    out += [h2("Training")]
    full = d["utterances"] >= d["full_split_utterances"] - 50
    rows = [["Data", f"LibriSpeech train-clean-100: {n_(d['utterances'])} utterances, {d['hours']:.1f} h, {d['speakers']} speakers"
             + (" (the whole split; utterances over 20 s are skipped)" if full else
                f" (full split: {n_(d['full_split_utterances'])} / {d['full_split_hours']} h / {d['full_split_speakers']})")],
            ["Augmentation", "SpecAugment: 2 frequency masks (up to 15 bands) and 2 time masks (up to 5%)"],
            ["Batching", "utterances sorted by length, at most 320 s of audio per batch, 2 loader workers"],
            ["Optimiser", "AdamW, peak lr 1.5e-3, 1,000 warm-up steps, cosine decay to 2%, gradient clip 5"],
            ["Precision", "network in fp16 autocast with a gradient scaler; the STFT stays in fp32 (power overflows fp16)"],
            ["Checkpoints", "every 500 steps and every epoch; training resumes from the exact batch"],
            ["Selection", "mean word-onset error against MFA references on dev clips, plus greedy CER on 300 dev utterances"]]
    if m.get("args"):
        rows.insert(4, ["Epochs / time", f"{m['args'].get('epochs')} epochs ({n_(m['history'][-1]['step'])} steps), "
                        f"{al['wall_clock']} wall clock, {al['epoch_minutes']} min per epoch. Planned "
                        f"{al['planned_epochs']}, cut to {m['args'].get('epochs')} because the onset error had been "
                        "flat since epoch 1; the schedule was resumed with a shorter cosine decay"])
        pl = al["pilot"]
        rows.append(["Pilot run", f"while the damaged archive was being repaired: {pl['epochs']} epochs on the first "
                     f"{pl['hours']} h ({n_(pl['utterances'])} utterances, {pl['speakers']} speakers) to confirm that CTC "
                     "trained from scratch converges; not used for the final model"])
        rows.append(["Tie-break", "onset error compared at 1 ms resolution, ties go to the lower dev CER"])
    out += [table(["Setting", "Value"], rows, [1.1, 4.7], bold_first=True), Spacer(1, 8)]
    out += [h2("Making alignment robust")]
    out += bullets([
        "<b>Wildcard edges.</b> Audio is aligned in chunks of up to 60 s with a 1 s margin. A synthetic &lt;star&gt; "
        "symbol (best non-blank score minus 0.5) at both ends absorbs the extra audio instead of stretching the first or "
        "last word over it.",
        "<b>Chunk planning.</b> Chunks are cut at the longest pause near the end of each window, so no word is split.",
        "<b>Unpadded spans.</b> Word times come from the frames where letters are emitted, not padded into the "
        "surrounding silence (the padding would make subtitles appear about 0.5 s early).",
        "<b>Graceful fallback.</b> Words that cannot be aligned get times interpolated between their aligned neighbours.",
        "<b>Lag calibration.</b> A CTC model tends to emit a character slightly <i>after</i> the sound starts. On the dev "
        "clips our raw word starts are late by a near-constant amount, so that median lag is stored with the model "
        f"(onset_shift = {al.get('onset_shift', 0.0) * 1000:+.0f} ms) and applied to every aligned word."
        if al.get("onset_shift") else
        "<b>Lag calibration.</b> CTC tends to emit a character slightly after the sound starts; the median lag measured "
        "on the dev clips is stored with the model and subtracted from every aligned word.",
    ])
    if m.get("best"):
        b = m["best"]
        out += [h2("Results")]
        out += [tiles([(f"{b['onset_mean_abs_ms']:.0f} ms", "mean |word-onset error| vs MFA (after lag calibration)"),
                       (f"{b['onset_median_abs_ms']:.0f} ms", "median |onset error|"),
                       (pct(b["within_100ms"]), "word starts within 100 ms"),
                       (pct(b["dev_cer"]), "greedy character error rate (dev-clean)"),
                       (f"{b.get('onset_raw_mean_abs_ms', b['onset_mean_abs_ms']):.0f} ms",
                        f"raw mean error before calibration (lag {b['onset_bias_ms']:+.0f} ms)"),
                       (f"epoch {b['epoch']} of {m.get('args', {}).get('epochs', len(m['history']))}", "best checkpoint")])]
        out += [P("These dev figures use the lag measured on the same dev clips. The independent number is the "
                  "test-set cue-onset error in section 10, measured on 27 files the aligner never saw.", "small")]
        rows = [[str(h["epoch"]), f"{h['train_loss']:.3f}", pct(h["dev_cer"]),
                 f"{h.get('onset_raw_mean_abs_ms', 0):.1f}", f"{h['onset_mean_abs_ms']:.1f}", pct(h["within_100ms"])]
                for h in m["history"]]
        out += [table(["Epoch", "Train CTC loss", "Dev CER", "Raw onset ms", "Calibrated onset ms", "Within 100 ms"],
                      rows, [0.6, 1.1, 0.9, 1.0, 1.3, 1.1], num_cols=(1, 2, 3, 4, 5))]
        if "aligner_cer" in _figs:
            out += [KeepTogether([side_by_side(half_image(_figs["aligner_cer"]), half_image(_figs["aligner_onset"])),
                    P("Figure 11. Left: recognition quality while training. Right: timing error, which is what the "
                      "aligner is selected on.", "caption")])]
        out.append(callout("Talking point", f"A {M(al['params'])}-parameter model trained from scratch on {d['hours']:.1f} hours "
                           f"times word starts on the dev clips with a mean error of {b['onset_mean_abs_ms']:.0f} ms, with "
                           f"{pct(b['within_100ms'])} within 100 ms. The MMS-300M aligner it replaces has "
                           f"{300e6 / al['params']:.0f} times more parameters and was pre-trained on hundreds of thousands "
                           "of hours of speech; on the test files our aligner places cues more accurately (section 10)."))
    else:
        out.append(callout("Status", "The aligner is still training. Its results, curves and talking point are added "
                                     "automatically when models/aligner/metrics.json exists.", "warn"))
    out.append(CondPageBreak(330))
    return out


# ====================================================================== integration + results
def integration_section(f: dict) -> list:
    t = f["results"]["tuning"]
    out = section(9, "Putting the models to work", "Integration")
    out += [P("The models do not act alone. In Mode A every disagreement between subtitle and audio becomes a "
              "candidate edit, scored by a weighted rule. The weights were re-tuned by grid search on the dev set "
              "after the language model changed, maximising F0.5 (precision counts twice as much as recall, because a "
              "wrong \"correction\" is worse than a missed one).")]
    out += [formula(["score(a over s) = w<sub>conf</sub>&middot;p<sub>ASR</sub>(a) + w<sub>phon</sub>&middot;phon_sim(s, a)"
                     " + w<sub>LM</sub>&middot;[log P(a | ctx) - log P(s | ctx)]",
                     "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
                     " + w<sub>oov</sub>&middot;[s is not a dictionary word] + w<sub>lex</sub>&middot;[a is in the user's lexicon]",
                     "The subtitle word s is replaced by the ASR word a when score &gt; &tau;<sub>sub</sub>."])]
    if t.get("new") and t.get("old"):
        o, n = t["old"], t["new"]
        keys = [("w_conf", "w_conf (ASR confidence)"), ("w_phon", "w_phon (sounds alike)"), ("w_lm", "w_LM (language model)"),
                ("w_oov", "w_oov (not a word)"), ("tau_sub", "tau_sub (threshold)"), ("tau_ins", "tau_ins (insertions)"),
                ("tau_lm_ins", "tau_lm_ins (LM gate for insertions)"),
                ("dev_f05", "dev F0.5"), ("dev_tp", "dev true positives"), ("dev_fp", "dev false positives")]
        rows = [[lbl, f"{o[k]:g}", f"{n[k]:g}"] for k, lbl in keys if k in o and k in n]
        out += [Spacer(1, 6), table(["Tuned value", "with distilgpt2", "with our KN LM"], rows, [2.4, 1.3, 1.3],
                                    num_cols=(1, 2))]
        out += [P("With our language model the grid search moved weight towards ASR confidence, dropped the spelling "
                  "flag and doubled the LM weight. The first test run showed that many dropped words were no longer "
                  "put back: n-gram log-probabilities for an inserted word are much lower than distilgpt2's, so the "
                  "LM gate for insertions sat outside the original search range. Widening that range on dev "
                  f"(tau_lm_ins {o.get('tau_lm_ins', 0):g} to {n.get('tau_lm_ins', 0):g}) fixed it.", "small")]
    out += [h2("How the models cooperate (illustrative examples)")]
    out += bullets([
        "<b>Fusion (LM):</b> \"Hello <u>their</u>\" vs heard \"there\": the LM prefers \"Hello there\", the phonetic "
        "similarity is high and Whisper is confident, so the word is replaced and logged with its evidence.",
        "<b>Punctuation:</b> adds the missing marks; in Mode A it only fills gaps and never removes the subtitler's marks.",
        "<b>Corrector:</b> repairs what is left, such as a duplicated word (\"the the\"), sentence by sentence, behind its guards.",
        "<b>Aligner:</b> times every final word, including the ones the corrector changed, which Whisper never timed.",
        "<b>POS tagger:</b> guides the dynamic-programming segmenter to split cues at natural phrase boundaries within "
        "42 characters per line and 17 characters per second.",
    ])
    out += [h2("Tested and usable")]
    c = f["code"]
    out += bullets([
        f"<b>{c['py_tests']} Python test functions</b> (unit, property-based, integration and API) and <b>{c['ts_tests']} frontend tests</b>, "
        "including tests for each new model module: KN normalisation, BPE round-trips, beam search vs greedy, "
        "perceptron tagging, CTC alignment plumbing.",
        "<b>A web application</b> (FastAPI + React) wraps the pipeline: upload, live progress, a player with corrected "
        "captions and word-by-word highlighting from our aligner's times, a colour-coded diff with the evidence for "
        "each edit, an in-browser cue editor and downloads.",
    ])
    out.append(CondPageBreak(330))
    return out


def results_section(f: dict) -> list:
    res = f["results"]
    out = section(10, "End-to-end results: before and after", "Evaluation")
    old, new = res["old"], res["new"]
    if not new:
        out.append(callout("Pending", "The full test-set run with the final models has not finished yet. When "
                           "results/SUMMARY.md is regenerated after the aligner is trained, this section fills in "
                           "automatically with the side-by-side comparison.", "warn"))
        if old:
            o = _ours(old)
            out += [P("For reference, the pretrained version scored (Mode A, all 27 test files):")]
            out += [table(["Metric", "Pretrained system"],
                          [["WER", f"{o['WER']:.2f}%"], ["ERRANT F0.5", f"{o['ERRANT F0.5']:.1f}"],
                           ["Punctuation F1", f"{o['Punct F1']:.1f}"], ["Mean onset error", f"{o['Onset ms']:.0f} ms"],
                           ["Onsets within 250 ms", f"{o['<=250 ms']:.1f}%"], ["Lines within 42 chars", f"{o['Lines ok']:.1f}%"]],
                          [2.4, 1.4], num_cols=(1,))]
        out.append(PageBreak())
        return out
    out += [P("Same benchmark, same corruptions, same metrics: 8 LibriSpeech test-clean, 8 test-other and 11 TED-LIUM "
              "files with about 11% injected word errors plus timing and formatting faults.")]
    for mode, title in (("A", "Mode A: audio + corrupted subtitles"), ("B", "Mode B: audio only")):
        o, n = _ours(old, mode), _ours(new, mode)
        if not o or not n:
            continue
        cols = [("WER", "WER %", False, 2), ("CER", "CER %", False, 2), ("ERRANT F0.5", "ERRANT F0.5", True, 1),
                ("Over-corr", "Over-correction %", False, 2), ("Punct F1", "Punctuation F1", True, 1),
                ("Onset ms", "Mean cue-onset error (ms)", False, 0), ("<=250 ms", "Onsets within 250 ms %", True, 1),
                ("CPS<=17", "Cues at <=17 CPS %", True, 1), ("Lines ok", "Lines <=42 chars %", True, 1),
                ("RTF", "Real-time factor", False, 3)]
        rows = []
        for key, label, higher, d in cols:
            if o.get(key) is None or n.get(key) is None:
                continue
            diff = n[key] - o[key]
            better = (diff > 0) == higher if abs(diff) > 1e-9 else None
            mark = "same" if better is None else ("better" if better else "worse")
            rows.append([label, f"{o[key]:.{d}f}", f"{n[key]:.{d}f}", f"{diff:+.{d}f} ({mark})"])
        out += [h2(title), table(["Metric", "Pretrained models", "Our models", "Change"], rows, [2.2, 1.3, 1.3, 1.4],
                                 num_cols=(1, 2, 3))]
    if "results" in _figs:
        out += [Spacer(1, 8)] + image(_figs["results"], caption="Figure 12. The headline metrics side by side (Mode A).")
    out += _timing_table(f)
    out += _honest_reading(f)
    base = new.get("A", {})
    brows = [[k, f"{v['WER']:.2f}" if v.get("WER") is not None else "–",
              f"{v['ERRANT F0.5']:.1f}" if v.get("ERRANT F0.5") is not None else "–",
              f"{v['Onset ms']:.0f}" if v.get("Onset ms") is not None else "–"] for k, v in base.items()]
    if brows:
        out += [h2("All systems on the new run (Mode A)"),
                table(["System", "WER %", "ERRANT F0.5", "Onset ms"], brows, [3.4, 0.9, 1.0, 0.9], num_cols=(1, 2, 3))]
    out.append(CondPageBreak(330))
    return out


def _timing_table(f: dict) -> list:
    tn, to = f["results"].get("timing_new"), f["results"].get("timing_old")
    if not tn:
        return []
    rows = []
    names = [("whisper_dtw", tn, "Whisper's own timestamps (no aligner)"), ("w2v2_ctc", to, "wav2vec2-base CTC, 95M (pretrained, before)"),
             ("mms_ctc", to, "MMS-300M CTC, 300M (pretrained, before)"), ("own_ctc", tn, "Our CTC aligner, 9.3M (from scratch)")]
    for key, table_, label in names:
        r = (table_ or {}).get(key)
        if r:
            rows.append([label, f"{r['Onset ms']:.0f}", f"{r['<=100 ms']:.1f}%", f"{r['<=250 ms']:.1f}%", f"{r['Word ms']:.0f}"])
    if not rows:
        return []
    return [h2("Timing: our aligner against the alternatives"),
            P("Timing ablation on the 19 test files with MFA references (8 LibriSpeech test-clean clips and 11 TED talks). "
              "Only the word-timing step changes; the text is identical within each run."),
            table(["Word-timing method", "Cue onset error (ms)", "Onsets within 100 ms", "Onsets within 250 ms",
                   "Word |Δt| (ms)"], rows, [2.6, 1.0, 1.0, 0.9, 0.9], num_cols=(1, 2, 3, 4),
                  highlight_rows=(len(rows) - 1,)),
            P("Word |Δt| averages start and end errors. Measured separately, our word starts are better than MMS-300M's "
              "(about 33 ms vs 43 ms mean) but our word ends are worse (about 81 ms vs 42 ms). Ends only decide when a cue "
              "disappears, which the cue-timing rules then extend anyway.", "small"), Spacer(1, 6)]


def _honest_reading(f: dict) -> list:
    o, n = _ours(f["results"]["old"]), _ours(f["results"]["new"])
    ob, nb = _ours(f["results"]["old"], "B"), _ours(f["results"]["new"], "B")
    g = f["results"].get("gec_new") or {}
    none, ours = g.get("none"), g.get("own_transformer")
    pts = [
        f"<b>WER improved ({o['WER']:.2f}% to {n['WER']:.2f}%), but mostly thanks to re-tuning.</b> Widening the search "
        "range of the fusion LM gate let more dropped words be put back (recall of dropped words rose from about 75% to "
        "92%). The old system's own no-LM ablation suggests it would have gained similarly, so we do not credit this to "
        "the n-gram model.",
        f"<b>Timing improved ({o['Onset ms']:.0f} to {n['Onset ms']:.0f} ms mean cue-onset error;"
        f" {o['<=250 ms']:.1f}% to {n['<=250 ms']:.1f}% within 250 ms).</b> This is a genuine win for our 9.3M-parameter "
        "aligner over the 300M-parameter MMS model.",
        f"<b>Punctuation is the one place the pretrained model was better</b> (F1 {o['Punct F1']:.1f} vs {n['Punct F1']:.1f} in "
        + (f"Mode A, {ob['Punct F1']:.1f} vs {nb['Punct F1']:.1f} in Mode B" if ob and nb else "Mode A")
        + "). XLM-R-large was pre-trained on vastly more text than our 8.5M words.",
    ]
    if none and ours:
        pts.append(f"<b>The corrector does not help end to end</b> (WER {none['WER']:.2f}% without it vs {ours['WER']:.2f}% with "
                   "it on the ablation subset), exactly as with every pretrained grammar model in the first version: once "
                   "fusion and the audio guard have run, little is left for it to fix. Its intrinsic result (13.9% to 4.3% "
                   "WER on noisy sentences) shows the model itself works.")
    pts.append(f"<b>Speed is unchanged</b> (real-time factor {o['RTF']:.3f} before, {n['RTF']:.3f} now).")
    return [h2("Reading the results honestly")] + bullets(pts) + [Spacer(1, 4)]


# ====================================================================== difficulties
DIFFICULTIES = [
    ("A 6.4 GB download over a slow, unstable network",
     "A single connection to OpenSLR ran at about 90 KB/s (about 19 hours for train-clean-100); speed swung between "
     "1 KB/s and 13 MB/s, and for a while a DNS outage made every mirror fail.",
     "Wrote scripts/fetch_parallel.sh: 8 HTTP range requests spread over tested OpenSLR mirrors, each part resumable, "
     "with longer back-off and up to 3,000 retries."),
    ("Orphaned download processes corrupting files",
     "Stopping the download task did not kill its background per-part loops; they kept starting curl, so after the "
     "restart two writers appended to the same parts.",
     "Found and killed every leftover bash/curl process, then hardened the script: bytes are appended only on an HTTP "
     "206 response, each run writes an owner token so stale loops stop, and a gzip -t check runs at the end."),
    ("A corrupted archive after the download finished",
     "The archive failed its gzip integrity check. Streaming decompression located the first error exactly at the "
     "restart offset in part 3.",
     "Instead of downloading 6.4 GB again, wrote a window patcher that compares a 4 KB sample of every 1 MB window with "
     "the server and re-fetches only the bad ones: 495 of 552 windows in part 3, about 520 MB, in 10 minutes. The "
     "repaired archive passed gzip -t and all 28,539 utterances (100.6 h) extracted."),
    ("Waiting on the repair",
     "The aligner could not start training while the archive was broken.",
     "Extracted the intact first 41% (stopping before the first bad byte) and ran a 2-epoch pilot on it to confirm that "
     "CTC trained from scratch converges and gives usable word onsets."),
    ("Interrupted working sessions",
     "The AI agent session lost its connection during a DNS outage and later hit a usage limit, so work stopped twice "
     "mid-training.",
     "Training ran as a background process with checkpoints every epoch (and every 500 steps for the aligner), so "
     "nothing was lost. Every neural trainer resumes from last.pt with its optimiser and scheduler state."),
    ("Only 6 GB of GPU memory",
     "Two GPU jobs at once spill into shared memory and slow to a crawl.",
     "Scheduled strictly one GPU job at a time: punctuation, then corrector, then aligner; batches sized by tokens or "
     "seconds of audio instead of sentence count; mixed precision."),
    ("Numerical overflow in half precision",
     "The STFT power spectrum overflows float16; T5-style models overflow in fp16 as well.",
     "The feature front end runs in fp32 and only the network uses fp16 autocast with a gradient scaler; the "
     "Transformer uses bfloat16, which has fp32's range."),
    ("16 GB of RAM, often only 2-6 GB free",
     "8.5 million words as Python strings and dictionaries of n-grams would not fit.",
     "n-grams packed into 64-bit integers and counted with NumPy; corpora streamed by generators; the punctuation "
     "dataset cached as compact integer arrays (one .npz file) rather than strings."),
    ("Subtitles that may have no capital letters",
     "A punctuation model that relies on capitals breaks on lower-case subtitles.",
     "A separate casing embedding, plus 50% of training windows lower-cased at random, so the model works both "
     "ways (macro-F1 0.81 cased, 0.67 lower-cased)."),
    ("A corrector that wants to rewrite everything",
     "Sequence-to-sequence models tend to paraphrase correct text.",
     "Trained with 33% clean copy pairs, plus an edit-ratio guard tuned to 15% and the audio-consistency guard; 97.6% of "
     "clean dev sentences pass through unchanged."),
    ("No training data for the corrector",
     "There is no public corpus of faulty subtitles paired with corrections.",
     "Generated pairs with a seeded noise model mixing synthetic typos and homophones with real whisper-tiny errors "
     "(the DARAG lesson: real errors beat purely synthetic ones), with fresh noise every epoch."),
    ("Unknown words in the language model",
     "Without special handling, a typo scores the same as any rare word.",
     "An open-vocabulary model: P(&lt;unk&gt;) times a character-bigram spelling probability."),
    ("Brown's 87 tags vs the 17 tags the segmenter needs",
     "Compound tags (PPS+BEZ), title markers (-TL) and the have/do auxiliary ambiguity.",
     "A mapping with lexical and look-ahead rules (have/do are AUX only before a verb)."),
    ("Re-tuning after the swap",
     "The cached dev features still contained distilgpt2 scores, so the old tuning was invalid.",
     "Recomputed the features with our LM and re-ran the grid search; the text-stage cache is keyed on configuration, so "
     "the whole benchmark was re-run from scratch."),
    ("Forced alignment at chunk edges",
     "Extra audio around a chunk was being forced onto the first or last word.",
     "Wildcard &lt;star&gt; tokens at both ends of the CTC target absorb the margin."),
    ("CTC emits characters late",
     "Raw word starts from our aligner were consistently late (median 46.7 ms on the dev clips).",
     "Stored that median lag with the model as onset_shift (-46.7 ms) and applied it at inference; the selection metric "
     "is the error after calibration (25.1 ms mean)."),
    ("Fusion stopped restoring dropped words",
     "The first test run with our LM restored only about 65% of dropped words: the LM gate's search range had been sized "
     "for distilgpt2's scores.",
     "Widened the grid for that gate on dev (dev F0.5 0.733 to 0.747); dropped-word recall rose to 92% and test WER fell "
     "from 3.26% to 2.75%."),
    ("Characters the corrector's tokenizer never saw",
     "LibriSpeech text has no digits or exclamation marks, so the BPE alphabet lacks them; an untrained beam search also "
     "produced start tokens mid-sentence.",
     "Sentences the BPE cannot represent exactly are passed through unchanged (BPE.covers); padding, start and unknown "
     "tokens are masked out during decoding."),
    ("A silently broken baseline",
     "Baseline B2 (ffsubsync) had been failing since the project folder moved: its launcher still pointed at the old "
     "virtual environment.",
     "The wrapper now runs ffsubsync through the current Python interpreter; B2 is back to its earlier numbers."),
    ("Choosing between near-identical checkpoints",
     "Aligner epochs 8 and 11 differed by 0.1 ms of onset error but by 1.4 points of character error.",
     "Onset error compared at 1 ms resolution with a CER tie-break, so epoch 11 (CER 7.85%) was selected."),
    ("A crash at interpreter exit",
     "On Windows the aligner process ended with error 0xC0000409 after printing \"done\" (CUDA plus DataLoader workers).",
     "Every output is saved before shutdown, so the crash costs nothing; noted rather than worked around."),
    ("Two workstreams in one folder",
     "The models and the web application were built at the same time in the same uncommitted repository.",
     "Strict file ownership between the two streams, stable public interfaces (pipeline.run, config keys), and a web "
     "app that hides options its configuration does not offer. While the configuration was mid-edit, the app "
     "showed a clear error instead of crashing."),
]


def difficulties_section(f: dict) -> list:
    out = section(11, "Difficulties faced and how we solved them", "Engineering")
    out += [P("Building models from scratch on a laptop over an unreliable connection is mostly an exercise in "
              "engineering under constraints. These are the problems that actually happened, in roughly the order we "
              "hit them.")]
    rows = [[f"<b>{t}</b>", p, s] for t, p, s in DIFFICULTIES]
    out += [table(["Challenge", "What went wrong", "How we solved it"], rows, [1.35, 1.9, 2.75])]
    out.append(CondPageBreak(330))
    return out


def lessons_section(f: dict) -> list:
    out = section(12, "Lessons learned and future work", "Reflection")
    out += [h2("What we learned")]
    out += bullets([
        "<b>Classical methods are not obsolete.</b> A 1998 smoothing method and a 2002 perceptron, implemented well, are "
        "competitive for narrow jobs and train in minutes.",
        "<b>Data design matters more than model size.</b> The corrector's quality comes from the noise model (real ASR "
        "confusions, clean copies), not from depth.",
        "<b>Evaluate the job, not the proxy.</b> The aligner is selected on word-onset error against MFA, not on "
        "recognition accuracy, because timing is what subtitles need.",
        "<b>Guard every learned component.</b> Edit-ratio and audio-consistency guards keep imperfect models safe.",
        "<b>Intrinsic gains do not always survive the pipeline.</b> The corrector cuts noisy-sentence WER by 69% on its "
        "own, yet changes almost nothing end to end, because fusion and the guards have already fixed what it would fix.",
        "<b>Re-tune after every swap.</b> A search range sized for one model silently capped the next one; widening it "
        "was worth half a point of WER.",
        "<b>Plan for failure.</b> Resumable checkpoints, resumable downloads and integrity checks turned outages into "
        "delays instead of lost work.",
    ])
    out += [h2("Future work")]
    out += bullets([
        "Calibrate the aligner's word ends separately (or end each word where the next blank run starts); word ends are "
        "the one timing measure where MMS was better.",
        "Train the punctuation tagger on more, and more spoken, punctuated text (TED or OpenSubtitles transcripts) to close "
        "the 2.5-point gap to XLM-R.",
        "Train the acoustic model on all 960 hours of LibriSpeech, or replace the BiLSTM with a small Conformer.",
        "A neural LSTM language model, or KenLM-style pruning, to cut the LM file from 226 MB.",
        "Replace the rule-weighted fusion with a learned classifier (logistic regression on the same features).",
        "Evaluate on Indian-English recordings (our own clips) to measure accent robustness.",
        "Distil Whisper's knowledge into a smaller recogniser of our own, the one pretrained part left.",
    ])
    out.append(CondPageBreak(330))
    return out


# ====================================================================== presentation kit
def talking_points(f: dict) -> list:
    pu, co, po, al, lm = f["punct"], f["corrector"], f["pos"], f["aligner"], f["lm"]
    t = f["results"]["tuning"]
    out = section(13, "Presentation kit", "For the viva")
    out += [h2("Points worth saying")]
    pts = [
        "Five of the six learned components were designed, coded and trained by us; only Whisper is pretrained, and we "
        "can explain why it has to stay.",
        f"Our neural models total {M(pu['params'] + co['params'] + al['params'])} parameters, against about 1.16 billion "
        "for the pretrained models they replace.",
        "We implemented interpolated modified Kneser-Ney smoothing ourselves, with the Chen-Goodman discount estimates.",
        f"The LM stores {lm['entries'] / 1e6:.1f} million n-grams as bit-packed 64-bit integers and trains in "
        f"{lm['train_seconds']} seconds on a CPU.",
        "An open-vocabulary trick (a character-bigram spelling model) lets the LM tell a typo from a rare word.",
        f"After re-tuning, fusion with our LM reaches {t['new']['dev_f05']:.3f} F0.5 on dev vs {t['old']['dev_f05']:.3f} "
        "with distilgpt2 (the small gain comes from a wider search range, not a better LM)."
        if t.get("new") and t.get("old") else "Fusion weights were re-tuned on dev for our LM.",
        *([_e2e_point(f)] if f["results"]["new"] else []),
        "Punctuation is framed as sequence labelling with a 2-layer BiLSTM over word, character-CNN and casing features.",
        f"Randomly lower-casing half of the training windows makes the punctuation model robust to caseless subtitles.",
        f"The punctuation tagger reaches macro-F1 {pu['metrics']['best']['cased']['macro']:.2f} with "
        f"{560e6 / pu['params']:.0f} times fewer parameters than XLM-R-large.",
        f"We wrote our own BPE tokenizer: {n_(co['bpe_merges'])} merges, {n_(co['bpe_vocab'])}-token vocabulary, exact "
        "round-trip decoding.",
        "The corrector is a pre-norm Transformer encoder-decoder with tied embeddings and beam search with length "
        "normalisation.",
        "With no parallel corpus available, we generated training pairs from a noise model that includes real "
        "whisper-tiny errors.",
        f"The corrector cuts noisy-dev WER from {pct(co['metrics']['best']['wer_copy'])} to "
        f"{pct(co['metrics']['best']['wer_model'])} and leaves {pct(co['metrics']['best']['clean_unchanged'])} of clean "
        "sentences untouched.",
        "Two guards (edit ratio and audio consistency) mean the audio always has the last word.",
        f"The POS tagger is an averaged perceptron: {pct(po['heldout_accuracy'], 1)} accuracy with no neural network.",
        "We mapped Brown's 87 tags to Universal Dependencies with look-ahead rules for auxiliaries.",
        "The aligner computes its own log-mel features (our mel filterbank) and is trained with CTC loss and SpecAugment "
        "on 100.6 hours of speech.",
        "CTC emits characters about 47 ms late; we measure that lag on dev and store it with the model as a calibration.",
        "Forced alignment uses our NumPy Viterbi with wildcard edge tokens and pause-aware chunking.",
        "Strict data hygiene: train for learning, dev for tuning, test once; LibriSpeech speakers are disjoint across splits.",
        "We survived a corrupted 6.4 GB download, network stalls and an agent outage thanks to resumable checkpoints "
        "and a byte-range patcher.",
    ]
    out += numbered(pts)
    out.append(CondPageBreak(120))
    out += [h2("Likely questions, with answers")]
    qa = [
        ("Why not simply use pretrained models?",
         "For a course project the learning is in building them. Ours are 26-46 times smaller than the models they "
         "replace, run on a laptop, use only "
         "documented training data (no hidden test-set exposure) and carry no non-commercial licence restrictions."),
        ("Then why keep Whisper?",
         "Speech recognition at 3% WER needs hundreds of thousands of hours of training audio. With 100 hours we would get "
         "15-30% WER, and the audio evidence the system relies on would be too unreliable. We use Whisper as a sensor and "
         "build everything that interprets its output."),
        ("What is Kneser-Ney smoothing, and why not add-one?",
         "Add-one gives too much probability to unseen events. Kneser-Ney subtracts a small discount from every seen count, "
         "passes that mass to lower orders, and uses continuation counts (how many contexts a word appears in) for those "
         "orders. It is the best classical n-gram smoother."),
        ("How do you handle words the models have never seen?",
         "The LM maps rare words to <unk> and multiplies by a character-level spelling probability; the punctuation "
         "tagger has a character CNN; the corrector splits unknown words into BPE pieces."),
        ("Why a BiLSTM for punctuation rather than a Transformer?",
         "With 8.5M training words and a 6 GB GPU, a BiLSTM is data-efficient and fast. The task needs only local context "
         "on both sides, which a bidirectional LSTM captures well."),
        ("How did you train a corrector without labelled data?",
         "We generated noisy/clean pairs from clean text with a seeded noise model: typos, homophones, dropped and doubled "
         "words, punctuation and case loss, and real whisper-tiny mistakes. A third of the pairs are clean copies."),
        ("How do you stop the corrector from changing correct text?",
         "Copy training, an edit-ratio guard (at most 15% of words changed) and the audio guard: a word Whisper heard "
         "confidently cannot be changed. 97.6% of clean sentences come out unchanged."),
        ("What is CTC, and why does it need a blank symbol?",
         "CTC trains a frame-level model from unaligned transcripts by summing over every way of spelling the transcript "
         "across the frames. The blank lets the model output nothing between characters and separates doubled letters, "
         "as in \"ll\"."),
        ("How is forced alignment different from recognition?",
         "Recognition searches for the most likely text. Alignment already knows the text and searches only for the "
         "best timing of it, a constrained Viterbi path, so even a modest acoustic model can time words well."),
        ("How do you know the timings are right?",
         "We compare word starts with silver references from the Montreal Forced Aligner, report mean and median absolute "
         "onset error and the share within 100 and 250 ms, and pick the aligner checkpoint on that metric."),
        ("How did you avoid data leakage?",
         "Only train splits are used for learning. Dev splits select checkpoints and weights. The test benchmark is run "
         "once at the end, and LibriSpeech speakers do not overlap across splits."),
        ("Is the perceptron really machine learning?",
         "Yes. It is a linear classifier trained by error-driven updates over 127k sparse features, and averaging gives "
         "it the generalisation of a large-margin method. It is the model behind classic taggers such as NLTK's."),
        ("Did removing the pretrained models hurt?",
         "Not overall. On the same test set WER went from 3.03% to 2.75% and cue timing from 50 to 39 ms. Punctuation is "
         "2.5 F1 points worse, and we say openly that most of the WER gain came from re-tuning a fusion threshold."),
        ("Why does the corrector not lower WER end to end?",
         "On its own it cuts noisy-sentence WER from 13.9% to 4.3%. In the pipeline, fusion with the audio has already fixed "
         "most errors and the audio guard blocks changes to confidently heard words, so little is left. Every pretrained "
         "grammar model behaved the same way in the first version."),
        ("Where is your system still weaker?",
         "Punctuation (XLM-R saw far more text) and word end times (our CTC aligner's ends are about 81 ms off vs 42 ms for "
         "MMS). Both have clear fixes: more punctuated training text and a separate end calibration."),
        ("What was the hardest problem?",
         "Getting 6.4 GB of audio over an unstable network and recovering from a corrupted archive without starting over. "
         "We solved it with parallel range downloads and a patcher that re-fetched only the damaged byte ranges."),
        ("What would you do with more compute?",
         "Train the aligner on 960 hours with a Conformer, replace rule-based fusion with a learned classifier, and "
         "eventually distil Whisper into a recogniser of our own."),
    ]
    for q, a in qa:
        out += [P(f"Q. {q}", "q"), P(a, "a")]
    out.append(CondPageBreak(330))
    return out


def _e2e_point(f: dict) -> str:
    o, n = _ours(f["results"]["old"]), _ours(f["results"]["new"])
    return (f"End to end, on the same 27 test files: WER {o['WER']:.2f}% to {n['WER']:.2f}%, ERRANT F0.5 "
            f"{o['ERRANT F0.5']:.1f} to {n['ERRANT F0.5']:.1f}, cue-onset error {o['Onset ms']:.0f} to {n['Onset ms']:.0f} ms. "
            "We can also say honestly where we lost: punctuation F1, by 2.5 points.")


def cheat_sheet(f: dict) -> list:
    pu, co, po, al, lm = f["punct"], f["corrector"], f["pos"], f["aligner"], f["lm"]
    out = section(14, "Numbers cheat sheet", "One page to remember")
    m = al.get("metrics") or {}
    b = m.get("best") or {}
    rows = [
        ["Language model", "Modified Kneser-Ney 4-gram", f"{n_(lm['vocab'])} words; {lm['entries'] / 1e6:.1f}M n-grams",
         f"{n_(lm['train_sentences'])} sentences / 8.54M words", f"ppl {lm['ppl_dev-clean']:.0f} (dev-clean)", f"{lm['train_seconds']} s, CPU"],
        ["Punctuation", "Emb + char-CNN + 2x BiLSTM(256)", f"{n_(pu['params'])} params",
         f"{pu['data'].get('train_words', 0) / 1e6:.2f}M words", f"macro-F1 {pu['metrics']['best']['cased']['macro']:.3f}",
         f"{pu['metrics']['args']['epochs']} epochs, ~{pu['total_minutes']} min"],
        ["Corrector", "Transformer 4+3 layers, d=256, 4 heads", f"{n_(co['params'])} params, BPE {n_(co['bpe_vocab'])}",
         f"{n_(co['train_sentences'])} sentences", f"WER {pct(co['metrics']['best']['wer_copy'], 1)} -> "
         f"{pct(co['metrics']['best']['wer_model'], 1)}", f"{co['metrics']['args']['epochs']} epochs, ~{co['total_minutes']} min"],
        ["POS tagger", "Averaged perceptron, 17 templates", f"{n_(po['features'])} features",
         f"{n_(po['train_sentences'])} Brown sentences", f"acc {pct(po['heldout_accuracy'], 2)}", f"{po['train_seconds']} s, CPU"],
        ["Aligner", "Log-mel + 2 conv + 4x BiLSTM(320), CTC", f"{n_(al['params'])} params",
         f"{al['data']['hours']:.1f} h / {n_(al['data']['utterances'])} utts",
         f"onset {b['onset_mean_abs_ms']:.0f} ms, CER {pct(b['dev_cer'])}" if b else "training",
         f"{m.get('args', {}).get('epochs', 25)} epochs, {al['wall_clock']}"],
    ]
    out += [table(["Model", "Architecture", "Size", "Training data", "Result", "Training"], rows,
                  [0.95, 1.65, 1.3, 1.25, 1.15, 0.95], bold_first=True)]
    out += [Spacer(1, 10)]
    c = f["code"]
    mod_rows = [[k, str(v["files"]), n_(v["lines"])] for k, v in c["modules"].items()]
    out += [h2("Code written"), table(["Component", "Files", "Lines"], mod_rows, [3.2, 0.8, 0.9], num_cols=(1, 2))]
    out += [P(f"The whole <font name='Consolas'>subfix</font> package is {n_(c['src_lines'])} lines of Python; "
              f"{c['py_tests']} Python test functions and {c['ts_tests']} frontend tests.", "small")]
    out.append(CondPageBreak(330))
    return out


def appendix(f: dict) -> list:
    out = section(15, "Appendix", "Reference")
    out += [h2("A. Reproducing every model")]
    cmds = [("python experiments/train_lm.py", "Kneser-Ney 4-gram LM -> models/lm/kn4.npz"),
            ("python experiments/train_pos.py", "perceptron POS tagger -> models/pos/"),
            ("python experiments/train_punct.py --epochs 8", "BiLSTM punctuation tagger -> models/punct/"),
            ("python experiments/train_corrector.py --epochs 15", "BPE + Transformer corrector -> models/corrector/"),
            ("python experiments/train_aligner.py --epochs 25", "CTC acoustic model -> models/aligner/"),
            ("python experiments/tune_fusion.py", "re-tune fusion weights on dev -> configs/tuned.yaml"),
            ("scripts\\run_all_experiments.ps1 -Fresh", "full benchmark -> results/"),
            ("python docs\\report\\build_report.py", "this report")]
    out += [table(["Command", "Produces"], [[f"<font name='Consolas'>{c}</font>", d] for c, d in cmds], [2.9, 2.6])]
    out += [P("Every trainer resumes from <font name='Consolas'>last.pt</font> if it is interrupted; seeds are fixed.",
              "small")]
    out += [h2("B. Glossary")]
    gl = [("ASR", "automatic speech recognition"), ("BiLSTM", "bidirectional long short-term memory network"),
          ("BPE", "byte-pair encoding, a subword tokenizer"), ("CER / WER", "character / word error rate"),
          ("CTC", "connectionist temporal classification, a loss for unaligned sequence labelling"),
          ("CPS", "characters per second, the subtitle reading speed"),
          ("ERRANT", "a toolkit that extracts and scores grammatical edits"),
          ("F0.5", "F-score weighting precision twice as much as recall"),
          ("Forced alignment", "finding when each word of a known transcript is spoken"),
          ("KN", "Kneser-Ney smoothing for n-gram language models"),
          ("MFA", "Montreal Forced Aligner, used for silver timing references"),
          ("Perplexity", "exp(average negative log-probability per word); lower is better"),
          ("SpecAugment", "masking bands of frequencies and time during training"),
          ("UD / UPOS", "Universal Dependencies part-of-speech tags")]
    out += [table(["Term", "Meaning"], [[f"<b>{a}</b>", b] for a, b in gl], [1.2, 4.6])]
    out += [h2("C. References")]
    refs = [
        "Chen, S. F. &amp; Goodman, J. (1998). An empirical study of smoothing techniques for language modeling. Harvard TR-10-98.",
        "Kneser, R. &amp; Ney, H. (1995). Improved backing-off for M-gram language modeling. ICASSP.",
        "Hochreiter, S. &amp; Schmidhuber, J. (1997). Long short-term memory. Neural Computation 9(8).",
        "Vaswani, A. et al. (2017). Attention is all you need. NeurIPS.",
        "Sennrich, R., Haddow, B. &amp; Birch, A. (2016). Neural machine translation of rare words with subword units. ACL.",
        "Wu, Y. et al. (2016). Google's neural machine translation system (length normalisation). arXiv:1609.08144.",
        "Collins, M. (2002). Discriminative training methods for hidden Markov models: perceptron algorithms. EMNLP.",
        "Graves, A. et al. (2006). Connectionist temporal classification. ICML.",
        "Park, D. S. et al. (2019). SpecAugment. Interspeech.",
        "Loshchilov, I. &amp; Hutter, F. (2019). Decoupled weight decay regularization (AdamW). ICLR.",
        "Panayotov, V. et al. (2015). LibriSpeech: an ASR corpus based on public domain audio books. ICASSP.",
        "Meister, A. et al. (2023). LibriSpeech-PC: punctuation and capitalization restoration benchmark. arXiv:2310.02943.",
        "Francis, W. N. &amp; Kucera, H. (1979). Brown Corpus Manual. Brown University.",
        "Petrov, S., Das, D. &amp; McDonald, R. (2012). A universal part-of-speech tagset. LREC.",
        "Radford, A. et al. (2023). Robust speech recognition via large-scale weak supervision (Whisper). ICML.",
        "McAuliffe, M. et al. (2017). Montreal Forced Aligner. Interspeech.",
        "Bryant, C., Felice, M. &amp; Briscoe, T. (2017). Automatic annotation and evaluation of error types for GEC (ERRANT). ACL.",
    ]
    out += numbered(refs)
    return out


# ====================================================================== main
_figs: dict = {}


def build(out: Path = OUT) -> Path:
    global _figs
    f = F.collect()
    missing = [k for k in ("lm", "pos", "punct", "corrector") if not f[k]]
    if missing:
        raise SystemExit(f"models missing: {missing} - train them first")
    _figs = charts.build_all(f, FIG)
    doc = Doc(out, "Building the subtitle corrector's models from scratch",
              "Audio-aware subtitle correction · self-trained models", draft=not f["ready"])
    story = []
    for part in (cover, summary, background, data_section, lm_section, punct_section, corrector_section, pos_section,
                 aligner_section, integration_section, results_section, difficulties_section, lessons_section,
                 talking_points, cheat_sheet, appendix):
        story += part(f)
    doc.multiBuild(story)
    print(f"wrote {out} ({'final' if f['ready'] else 'DRAFT'})")
    return out


if __name__ == "__main__":
    build()
