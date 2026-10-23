# Presentation guide: Subfix (audio-aware subtitle correction)

Every number here comes from `results/SUMMARY.md`, `models/*/metrics.json` and
`docs/report/facts.py`. If you retrain or re-run anything, re-check them.

---

## 1. The project in one minute

**Problem.** Subtitles you find or receive are often wrong: typos, wrong homophones
("their/there"), dropped or doubled words, missing punctuation, and timing that drifts
seconds away from the speech. Speech recognition (Whisper) alone is also imperfect: it
mishears names, its timestamps are loose (~235 ms off), and its cues break reading-speed rules.

**Idea.** Use *both* noisy sources together. The subtitle file and the audio make
different mistakes, so a fusion step decides, word by word, which source to trust,
using acoustic confidence, phonetic similarity, a language model and spelling checks.
Then the words are re-timed against the audio with our own aligner, punctuated, and cut into
cues that obey broadcast rules (≤42 chars/line, ≤2 lines, ≤17–20 chars/second).

**Two modes.**
- Mode A: audio + a broken subtitle file → corrected, re-timed subtitles.
- Mode B: audio only → subtitles from scratch (Whisper + our own post-processing).

**Pipeline** (`src/subfix/pipeline.py:152`, function `run`):

```
media ──► preprocess ──► Whisper ASR ──┐
                                       ├─► align (word-level DP) ─► FUSE ─► corrector ─► punctuation
subtitle file ─► parse ─► sync (offset)┘                                                    │
                                                                                            ▼
             export SRT/VTT ◄── chapters ◄── segment into cues (DP + POS) ◄── CTC re-timing
```

---

## 2. The models: what we trained, how, and what was hard

Only **Whisper** (speech recognition) is a pretrained model in the main pipeline.
Five pretrained models (~1.16 billion parameters in total) were **replaced by models
we trained ourselves** (~30 million neural parameters plus a count-based LM):

| Role | Before (pretrained) | Now (self-trained) | Size |
|---|---|---|---|
| Language model for fusion | distilgpt2 (82M) | Kneser-Ney 4-gram | 13.4M n-grams |
| Punctuation | FullStop XLM-RoBERTa-large (560M) | BiLSTM tagger | 12.3M params |
| Error corrector | T5-base (220M) | Transformer encoder-decoder + own BPE | 8.4M params |
| POS tags (for line breaks) | spaCy en_core_web_sm | Averaged perceptron | 127k features |
| Word timing (forced alignment) | MMS-300M wav2vec2 (300M) | CTC BiLSTM acoustic model | 9.3M params |

All training ran on one laptop: RTX 4050 (6 GB), 16 GB RAM.

### 2.1 Kneser-Ney 4-gram language model
- **Code:** model `src/subfix/ngram.py` (class `KneserNeyLM`, line 45), wrapper `src/subfix/lm.py`,
  trainer `experiments/train_lm.py`. Output: `models/lm/kn4.npz` + `kn4.json`.
- **Data:** LibriSpeech-PC train text (train-clean-100/360 + train-other-500): 563,049 sentences, ~8.5M words.
  Shared loader: `experiments/corpus.py`.
- **How it works:** counts 1- to 4-grams; modified Kneser-Ney smoothing with three discounts per order
  (estimated from count-of-counts, `_discounts`, line 35). Unknown words get P(&lt;unk&gt;) × a character-bigram
  spelling probability, so a typo scores lower than a real rare word.
- **Result:** vocab 55,757; perplexity 283.7 (dev-clean), 251.8 (dev-other); trains in ~2 minutes.
- **Difficulty:** 8.5M words as Python strings and n-gram dicts did not fit in RAM →
  n-grams packed into 64-bit integers and counted with NumPy; corpus streamed by generators.
- **Difficulty:** the fusion thresholds had been tuned on distilgpt2 scores; after the swap the
  "restore dropped word" gate stopped firing (only ~65% of dropped words restored). Widening its
  dev-tuning grid fixed it: dropped-word recall 92%, test WER 3.26% → 2.75%.

### 2.2 Punctuation restoration (BiLSTM)
- **Code:** model `src/subfix/textfix/punct.py` (`build_net`, line 50), trainer `experiments/train_punct.py`.
  Output: `models/punct/model.pt`, `metrics.json`. Data cache `cache/train/punct_data.npz`.
- **Architecture:** for each word: word embedding (192) + casing embedding (8, five classes) +
  character CNN (64 filters over the spelling) → bidirectional LSTM → 4 labels per word:
  nothing / `.` / `,` / `?`.
- **Training:** LibriSpeech-PC (punctuated LibriSpeech) train splits, AdamW + OneCycle LR, 8 epochs,
  ~41 minutes.
- **Result:** macro-F1 0.81 on cased text, 0.67 on lower-case text (period 0.91, comma 0.69, question 0.83).
- **Difficulty:** many subtitles have no capitals; a model that leans on capitals breaks. Fix: separate
  casing embedding + 50% of training windows lower-cased at random, so it works both ways.
- **Honest note:** end-to-end punct F1 is 70.1 vs 72.6 with the 560M FullStop model. We gave up
  2.5 points for a model ~45× smaller.

### 2.3 Error corrector (small Transformer + our own BPE)
- **Code:** model `src/subfix/textfix/seq2seq.py` (`DEFAULT_NET`, line 11), tokenizer `src/subfix/bpe.py`,
  inference + guards `src/subfix/textfix/gec.py`, trainer `experiments/train_corrector.py`.
  Output: `models/corrector/model.pt`, `bpe.json`.
- **Architecture:** encoder 4 layers, decoder 3 layers, d_model 256, 4 heads, pre-norm, own BPE vocabulary.
- **Data problem:** no public corpus of "broken subtitle → fixed subtitle" pairs exists. We **generated**
  them (`noisify`, line 62): synthetic typos and homophones plus **real Whisper-tiny mistakes** harvested
  from dev audio (`harvest_confusions`, line 42), 513k training sentences, fresh noise every epoch.
- **Over-correction problem:** seq2seq models like to paraphrase correct text. Fix: 33% of pairs are
  clean copies (`make_pairs`, line 94), an edit-ratio guard (15%), and an audio-consistency guard.
  97.6% of clean dev sentences pass through unchanged.
- **Result:** on noisy dev text, WER 13.9% (copy) → 4.3% (model). bf16 training, ~76 minutes.
- **Honest note:** end-to-end it does not add value (ablation: with corrector WER 2.80, without 2.76),
  because fusion already fixes most errors before it. It is kept as an option, not a headline.

### 2.4 Part-of-speech tagger (averaged perceptron)
- **Code:** `src/subfix/postag.py` (`PerceptronTagger`, line 78), trainer `experiments/train_pos.py`.
  Output: `models/pos/perceptron.json.gz`.
- **Why it exists:** the cue segmenter must not break a line between "the" and its noun; POS tags
  tell it where good line breaks are.
- **Training:** Brown corpus, 54,473 sentences, 5 passes, ~3 minutes. Held-out accuracy **97.2%**.
- **Difficulty:** Brown has 87 tags incl. compound ones (`PPS+BEZ`) and title markers (`-TL`); we need the
  17 Universal tags. Wrote a mapping (`brown_to_ud`, line 50) with look-ahead rules
  (have/do are AUX only before a verb).

### 2.5 CTC acoustic aligner (the hardest one)
- **Code:** model `src/subfix/acoustic.py` (`build_acoustic`, line 58), alignment use `src/subfix/timing.py`,
  trainer `experiments/train_aligner.py`. Output: `models/aligner/model.pt` (stores `onset_shift`).
- **Architecture:** 80-band log-mel features → 2 conv layers (2× time subsampling, 20 ms frames)
  → 4-layer BiLSTM (320 hidden) → 29 characters + blank, trained with **CTC loss**
  (`CTCLoss`, line 213). SpecAugment for robustness (`spec_augment`, line 114).
- **Data:** LibriSpeech train-clean-100 (100.6 h, 251 speakers, disjoint from test speakers).
- **Training:** AdamW, fp16 autocast + GradScaler, batches sized by seconds of audio, checkpoint every
  500 steps; best epoch 11, ~3 hours.
- **How alignment works:** we know the words already, so CTC forced alignment finds when each character
  is spoken; the first character's frame is the word's start time.
- **Result:** dev CER 7.85%; word-start error **25.1 ms mean**, 89% within 50 ms, 98.6% within 100 ms.
  In the full pipeline, cue onsets are **39 ms** off vs 235 ms for raw Whisper (and 50 ms for MMS-300M).
- **Difficulty:** CTC emits characters ~47 ms late, consistently. Fix: measure the median lag on dev,
  store it in the checkpoint as `onset_shift`, subtract at inference.
- **Difficulty:** extra audio at chunk edges was forced onto the first/last word. Fix: wildcard `<star>`
  tokens at both ends of the target absorb it.
- **Difficulty:** fp16 overflows in the STFT power spectrum. Fix: features in fp32, only the network in fp16.
- **Known weakness:** word *ends* are early (~81 ms); only starts are calibrated.

### 2.6 Infrastructure difficulties (good "challenges" slide material)
- **6.4 GB dataset over an unstable network** (~90 KB/s single connection, DNS outages): wrote
  `scripts/fetch_parallel.sh` (8 parallel resumable range requests over several mirrors).
- **Corrupted archive** after restarts: instead of re-downloading 6.4 GB, compared 4 KB samples of each 1 MB
  window with the server and re-fetched only bad windows (~520 MB in 10 min).
- **While waiting**, ran a 2-epoch pilot on the intact first 41% to prove CTC-from-scratch converges.
- **6 GB GPU:** strictly one GPU job at a time; token/seconds-based batches; mixed precision.
- **Interrupted sessions:** every trainer resumes from `last.pt` (model + optimiser + scheduler).
- **Cache invalidation:** cached dev features still contained distilgpt2 scores → re-tuned and re-ran
  the whole benchmark from scratch (`scripts/run_all_experiments.ps1 -Fresh`).

The full list with explanations is `DIFFICULTIES` in `docs/report/build_report.py:774`.

---

## 3. Results (test split: LibriSpeech test-clean/other + TED-LIUM long-form)

| System (Mode A) | WER % | Onset error | Cues ≤250 ms | CPS ≤20 | Lines ok |
|---|---|---|---|---|---|
| Corrupted input | 10.88 | 9365 ms | 7.1% | 97.2% | 82.6% |
| B1: SymSpell + LanguageTool | 8.82 | 9362 ms | 7.1% | 97.4% | 82.9% |
| B2: ffsubsync (timing only) | 11.45 | 5633 ms | 27.3% | 97.0% | 82.5% |
| B3: raw Whisper | 3.10 | 235 ms | 59.2% | 93.5% | 27.5% |
| **Ours** | **2.75** | **39 ms** | **97.8%** | **99.9%** | **100%** |

Headlines:
- **75% relative WER reduction** (10.88% → 2.75%); target was ≥30%.
- Beats raw Whisper on WER **and** on timing (6× more precise) **and** on readability (100% vs 27.5% legal lines).
- Fewer over-corrections than Whisper replacement (1.41% vs 3.48%): we keep the subtitle author's correct words.
- Repairs 97% of spelling errors, 97% of homophones, 92% of dropped words, 85% of duplicates,
  82% of entity errors, but only 43% of "ASR-style" errors (the hardest class).
- Robust to corruption level: injected 5% / 10% / 20% → output WER 2.41 / 2.80 / 3.14%.
- Runs at RTF 0.056: one hour of video in ~3.4 minutes on the laptop GPU.
- Self-trained models beat the pretrained v1 on WER (3.03 → 2.75) and timing (50 → 39 ms).

Ablations that prove each piece matters (`results/ablations/`):
- Turn off the audio guard → WER 2.80 → 3.36.
- Use Whisper's own timestamps instead of our CTC aligner → onset 42 → 233 ms.
- Greedy line breaking instead of DP → only 61% of cues meet CPS ≤20 (vs 99.8%).

Extra outputs: **chapter markers** (WindowDiff 0.121 vs 0.216 TextTiling, `results/chapters/`),
optional **speaker labels** (pyannote), and a **web app** (FastAPI + React).

---

## 4. What makes it stand out

1. **Two-source fusion, not just "run Whisper".** Most tools either fix text (spell-checkers) or fix
   timing (ffsubsync) or replace everything with ASR. This uses the subtitle and the audio as two
   witnesses and arbitrates word by word, which is why it beats every baseline on every axis at once.
2. **Built our own models.** ~1.16B pretrained parameters replaced with ~30M self-trained ones, trained on a
   6 GB laptop GPU, and the system got *better* (WER 3.03 → 2.75, timing 50 → 39 ms).
3. **Timing precision.** 39 ms average cue-start error vs 235 ms for Whisper; 97.8% of cues within 250 ms.
4. **Broadcast-quality output.** 100% of cues follow line-length and line-count rules.
5. **Rigorous evaluation.** Two datasets, 3 baselines, 6 ablations, a corruption sweep, per-error-type
   recall, before/after comparison, and honestly reported negative results.
6. **A usable product.** Upload → see the diff → edit cues → download, in a browser.

## 5. Impact

- **Accessibility:** captions matter for deaf and hard-of-hearing viewers; wrong or late captions
  are the most common complaint. Correct + well-timed + readable cues directly address that.
- **Cost and privacy:** runs fully offline on a mid-range laptop; no cloud API, no per-minute fees,
  media never leaves the machine.
- **Speed:** ~3.4 minutes per hour of video vs hours of manual subtitle fixing.
- **Who benefits:** educators (lecture videos), small creators, archives with old/out-of-sync
  subtitle files, fansub/translation teams who need a clean, synced source.

Phrase impact as potential: we measured quality on benchmarks, not with real users.

## 6. Limitations (say them before the examiner does)

- Mode B (audio only) WER is 3.20% vs 3.10% for raw Whisper: slightly worse, but much better
  punctuation (77.7 vs 71.3 F1), timing and readability. Objective O2 is only partly met.
- Only 43% of ASR-type errors are repaired: when audio and subtitle are both unclear, there is nothing to arbitrate.
- The self-trained corrector does not improve end-to-end results.
- Aligner word ends are ~81 ms early; only starts are calibrated.
- English only; tested on read speech and TED talks, not noisy films or Indian-English accents yet.
- Chapter markers are weak on meetings (AMI), so an opt-in meeting mode exists.
- Speaker diarization needs a HuggingFace token and has not been run with the real pyannote model here.

---

## 7. Where the code is (map for your slides)

**Training scripts** (all resumable from `last.pt`):

| Model | Trainer | Model code | Saved to |
|---|---|---|---|
| 4-gram LM | `experiments/train_lm.py` | `src/subfix/ngram.py`, `src/subfix/lm.py` | `models/lm/` |
| Punctuation | `experiments/train_punct.py` | `src/subfix/textfix/punct.py` | `models/punct/` |
| Corrector | `experiments/train_corrector.py` | `src/subfix/textfix/seq2seq.py`, `src/subfix/bpe.py`, `src/subfix/textfix/gec.py` | `models/corrector/` |
| POS tagger | `experiments/train_pos.py` | `src/subfix/postag.py` | `models/pos/` |
| CTC aligner | `experiments/train_aligner.py` | `src/subfix/acoustic.py`, `src/subfix/timing.py` | `models/aligner/` |
| Shared text corpus | `experiments/corpus.py` | | |

**Pipeline (`src/subfix/`):**

| Stage | File |
|---|---|
| Orchestration | `pipeline.py` (`run`) |
| Audio preprocessing | `preprocess.py` |
| Whisper ASR | `asr.py` |
| Subtitle parse/write | `subio.py`, `export.py` |
| Global offset / drift fix | `sync.py` |
| Align subtitle words to ASR words | `align.py` |
| **Fusion (core idea)** | `fuse.py` (`Fuser`, line 81) |
| Spelling / truecasing | `textfix/spell.py`, `textfix/truecase.py` |
| Re-timing with own aligner | `timing.py` |
| Cue segmentation (DP + POS) | `segment.py` |
| Chapters | `chapters.py` |
| Speaker labels | `diarize.py` |
| CLI | `cli.py` |
| Settings | `config.py`, `configs/default.yaml`, `configs/tuned.yaml` |

**Evaluation:** `experiments/run_eval.py` (benchmark), `experiments/tune_fusion.py` (dev grid search),
`experiments/summarize.py` (tables), `experiments/error_analysis.py`, `experiments/eval_chapters.py`,
`metrics/` (WER, ERRANT, timing, format, segmentation). Benchmark creation: `tools/build_benchmark.py`,
`tools/corrupt.py` (error injection), `tools/make_refs.py` (MFA reference timings).
Run everything: `scripts/run_all_experiments.ps1 -Fresh`.

**Web app:** backend `app/backend/` (FastAPI, `app.py`, job queue `jobs.py`), frontend `frontend/src/`
(React + TypeScript). Start with `scripts/start_app.ps1`.

**Tests:** `tests/` (139 Python tests), `frontend/src/**/*.test.ts*` (30 frontend tests).

**Reports:** `docs/report/Subfix_Self_Trained_Models.pdf` (30-page model report),
`docs/report/ADA23DLU04_Project_Report.docx` (course report). Plots: `results/plots/`, `docs/report/figures/`.

---

## 8. Likely questions and short answers

- **Why not just use Whisper?** Whisper's WER is close, but its timing is 6× worse, only 27.5% of its
  cues are readable, and it over-corrects 2.5× more because it throws away the author's correct words.
- **Why train your own models when pretrained ones exist?** To understand and control every component,
  run on a 6 GB laptop, and show it can be done ~40× smaller. And it improved WER and timing.
- **Why is punctuation F1 lower than before?** FullStop is a 560M-parameter model; ours is 12M.
  We accept −2.5 F1 for that size reduction.
- **How do you know timing is right?** Reference timings from the Montreal Forced Aligner (MFA),
  a standard phonetic aligner, used as silver references.
- **How did you get training data for the corrector?** Generated it: real Whisper-tiny errors plus
  synthetic typos/homophones applied to clean LibriSpeech text.
- **What is CTC?** A loss that trains a network to output a character sequence from audio without
  knowing where each character is; at inference, forced alignment recovers when each character occurs.
- **What is Kneser-Ney?** An n-gram smoothing method that estimates how likely a word is to appear
  in new contexts (not just how frequent it is), which handles rare words better.
- **Is there data leakage?** No: training uses LibriSpeech train splits; tuning uses dev; results are on
  test-clean/test-other/TED test, with different speakers.
