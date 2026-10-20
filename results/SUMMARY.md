# Results summary (test split)

## ALL

Mode A (audio + corrupted subtitles):

| System | WER % | CER % | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Onset |Δt| ms | ≤250 ms % | CPS≤17 % | CPS≤20 % | Lines ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 10.88 | 5.14 | 0.0 | 0.0 | 0.00 | 30.6 | 9365 | 7.1 | 88.2 | 97.2 | 82.6 | 0.000 |
| B1: SymSpell + LanguageTool | 8.82 | 4.85 | 43.3 | 46.2 | 0.90 | 31.2 | 9362 | 7.1 | 88.6 | 97.4 | 82.9 | 0.037 |
| B2: ffsubsync (timing only) | 11.45 | 5.78 | 0.0 | 0.2 | 0.64 | 30.7 | 5633 | 27.3 | 86.2 | 97.0 | 82.5 | 0.008 |
| B3: raw Whisper replace | 3.10 | 1.84 | 69.5 | 70.7 | 3.48 | 71.3 | 235 | 59.2 | 80.4 | 93.5 | 27.5 | 0.031 |
| **Ours** (all self-trained models except Whisper) | 2.75 | 1.55 | 77.1 | 82.2 | 1.41 | 70.1 | 39 | 97.8 | 92.6 | 99.9 | 100.0 | 0.056 |

Mode B (audio only):

| System | WER % | CER % | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Onset |Δt| ms | ≤250 ms % | CPS≤17 % | CPS≤20 % | Lines ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Mode B: raw Whisper | 3.10 | 1.84 | – | – | – | 71.3 | 235 | 59.2 | 80.4 | 93.5 | 27.5 | 0.031 |
| Mode B: Ours | 3.20 | 1.88 | – | – | – | 77.7 | 42 | 97.3 | 92.1 | 99.8 | 100.0 | 0.053 |

## libri-test-clean

Mode A (audio + corrupted subtitles):

| System | WER % | CER % | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Onset |Δt| ms | ≤250 ms % | CPS≤17 % | CPS≤20 % | Lines ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 11.70 | 5.59 | 0.0 | 0.0 | 0.00 | 41.5 | 2497 | 0.9 | 91.8 | 97.7 | 86.1 | 0.000 |
| B1: SymSpell + LanguageTool | 9.51 | 5.16 | 38.3 | 43.6 | 0.85 | 41.4 | 2499 | 0.9 | 91.6 | 97.7 | 86.3 | 0.038 |
| B2: ffsubsync (timing only) | 11.95 | 5.82 | 0.0 | 0.0 | 0.30 | 41.5 | 459 | 46.3 | 90.4 | 97.5 | 86.0 | 0.004 |
| B3: raw Whisper replace | 2.13 | 0.70 | 63.5 | 81.9 | 1.62 | 68.4 | 273 | 46.2 | 91.8 | 98.9 | 21.2 | 0.029 |
| **Ours** (all self-trained models except Whisper) | 2.16 | 0.99 | 72.4 | 87.9 | 0.67 | 73.8 | 25 | 99.6 | 99.0 | 100.0 | 100.0 | 0.053 |

Mode B (audio only):

| System | WER % | CER % | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Onset |Δt| ms | ≤250 ms % | CPS≤17 % | CPS≤20 % | Lines ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Mode B: raw Whisper | 2.13 | 0.70 | – | – | – | 68.4 | 273 | 46.2 | 91.8 | 98.9 | 21.2 | 0.029 |
| Mode B: Ours | 2.27 | 0.77 | – | – | – | 76.1 | 26 | 99.8 | 97.6 | 100.0 | 100.0 | 0.050 |

## libri-test-other

Mode A (audio + corrupted subtitles):

| System | WER % | CER % | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Onset |Δt| ms | ≤250 ms % | CPS≤17 % | CPS≤20 % | Lines ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 12.18 | 5.53 | 0.0 | 0.0 | 0.00 | 19.7 | 2782 | 11.8 | 91.2 | 98.0 | 86.0 | 0.000 |
| B1: SymSpell + LanguageTool | 10.14 | 5.40 | 31.1 | 43.8 | 1.00 | 20.5 | 2779 | 11.7 | 91.6 | 98.0 | 86.0 | 0.032 |
| B2: ffsubsync (timing only) | 12.78 | 6.29 | 0.0 | 0.0 | 0.71 | 19.8 | 633 | 35.1 | 91.4 | 97.7 | 85.9 | 0.004 |
| B3: raw Whisper replace | 4.16 | 2.56 | 66.1 | 74.0 | 4.06 | 74.1 | 323 | 36.0 | 87.6 | 96.6 | 27.0 | 0.028 |
| **Ours** (all self-trained models except Whisper) | 2.48 | 1.10 | 73.0 | 86.5 | 1.27 | 67.1 | 27 | 99.8 | 91.6 | 100.0 | 100.0 | 0.051 |

Mode B (audio only):

| System | WER % | CER % | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Onset |Δt| ms | ≤250 ms % | CPS≤17 % | CPS≤20 % | Lines ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Mode B: raw Whisper | 4.16 | 2.56 | – | – | – | 74.1 | 323 | 36.0 | 87.6 | 96.6 | 27.0 | 0.028 |
| Mode B: Ours | 4.24 | 2.59 | – | – | – | 79.6 | 25 | 100.0 | 91.8 | 99.8 | 100.0 | 0.049 |

## ted-test

Mode A (audio + corrupted subtitles):

| System | WER % | CER % | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Onset |Δt| ms | ≤250 ms % | CPS≤17 % | CPS≤20 % | Lines ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 10.44 | 4.96 | 0.0 | 0.0 | 0.00 | – | 12774 | 7.5 | 86.5 | 96.8 | 80.8 | 0.000 |
| B1: SymSpell + LanguageTool | 8.41 | 4.67 | 50.0 | 47.3 | 0.88 | – | 12767 | 7.5 | 87.0 | 97.2 | 81.2 | 0.039 |
| B2: ffsubsync (timing only) | 11.08 | 5.66 | 0.0 | 0.3 | 0.70 | – | 8218 | 20.5 | 83.7 | 96.6 | 80.8 | 0.013 |
| B3: raw Whisper replace | 3.09 | 1.92 | 73.9 | 67.4 | 3.75 | – | 213 | 65.2 | 77.3 | 92.1 | 28.6 | 0.035 |
| **Ours** (all self-trained models except Whisper) | 2.93 | 1.77 | 81.6 | 79.7 | 1.60 | – | 46 | 96.9 | 91.4 | 99.8 | 100.0 | 0.062 |

Mode B (audio only):

| System | WER % | CER % | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Onset |Δt| ms | ≤250 ms % | CPS≤17 % | CPS≤20 % | Lines ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Mode B: raw Whisper | 3.09 | 1.92 | – | – | – | – | 213 | 65.2 | 77.3 | 92.1 | 28.6 | 0.035 |
| Mode B: Ours | 3.18 | 1.96 | – | – | – | – | 50 | 96.2 | 91.0 | 99.8 | 100.0 | 0.059 |

## Objectives (imp.md 1.2)

| # | Target | Result | Met |
|---|---|---|---|
| O1 | ≥30% relative WER reduction (Mode A) | 10.88% → 2.75% (75% relative) | yes |
| O2 | Mode B: lower WER and higher punct F1 than raw Whisper | WER 3.10% → 3.20%, punct F1 71.3 → 77.7 | partly/no |
| O3 | ≥90% of cue boundaries within ±250 ms | 97.8% of cue onsets | yes |
| O4 | ≥95% cues ≤42 chars/line, ≤2 lines, ≤17–20 CPS | lines 100.0%, 2-line 100.0%, CPS≤20 99.9%, CPS≤17 92.6% | yes (at 20 CPS) |
| O5 | Demo: upload → diff → download | web app (`app/backend/` + `frontend/`), tested by `tests/test_backend.py` and `npm test` | yes |

Per-error-type recall of **Ours** (share of injected errors repaired):

| asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|
| 42.9% (420/978) | 92.3% (679/736) | 85.3% (313/367) | 81.8% (108/132) | 96.7% (712/736) | 97.4% (1073/1102) |

## Self-trained models vs the earlier pretrained ones (ALL test files)

Same benchmark, ASR (Whisper large-v3-turbo) and pipeline; v1 rows are from `results/pretrained_v1/`.

| System | WER % | CER % | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Onset |Δt| ms | ≤250 ms % | CPS≤17 % | CPS≤20 % | Lines ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ours v1: distilgpt2 + FullStop XLM-R + T5-base GEC + MMS-300M + spaCy | 3.03 | 1.78 | 75.1 | 80.4 | 1.42 | 72.6 | 50 | 97.1 | 92.4 | 99.8 | 100.0 | 0.058 |
| Ours v1 + fine-tuned T5-small | 2.98 | 1.73 | 75.7 | 81.1 | 1.35 | 72.5 | 50 | 97.2 | 92.0 | 99.8 | 100.0 | 0.056 |
| **Ours** (all self-trained models except Whisper) | 2.75 | 1.55 | 77.1 | 82.2 | 1.41 | 70.1 | 39 | 97.8 | 92.6 | 99.9 | 100.0 | 0.056 |
| Mode B: Ours v1 (pretrained) | 3.24 | 1.93 | – | – | – | 79.5 | 53 | 96.8 | 91.7 | 99.8 | 100.0 | 0.063 |
| Mode B: Ours (self-trained) | 3.20 | 1.88 | – | – | – | 77.7 | 42 | 97.3 | 92.1 | 99.8 | 100.0 | 0.053 |

## Ablation: timing_source (ALL, libri-test-clean + ted-test)

| System | WER % | CER % | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Onset |Δt| ms | ≤250 ms % | CPS≤17 % | CPS≤20 % | Lines ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abl_timing_source_whisper_dtw | 2.80 | 1.63 | – | 81.3 | 1.44 | 73.8 | 233 | 63.4 | 91.3 | 99.9 | 100.0 | 0.044 |
| abl_timing_source_own_ctc | 2.80 | 1.63 | – | 81.3 | 1.44 | 73.8 | 42 | 97.4 | 92.8 | 99.8 | 100.0 | 0.058 |

Word-level timing vs MFA (mean |Δt| ms): whisper_dtw 90, own_ctc 57

## Ablation: segmenter (ALL, libri-test-clean + ted-test)

| System | WER % | CER % | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Onset |Δt| ms | ≤250 ms % | CPS≤17 % | CPS≤20 % | Lines ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abl_segmenter_greedy | 2.80 | 1.63 | – | 81.3 | 1.44 | 73.8 | 35 | 98.5 | 56.9 | 61.3 | 100.0 | 0.057 |
| abl_segmenter_dp | 2.80 | 1.63 | – | 81.3 | 1.44 | 73.8 | 42 | 97.4 | 92.8 | 99.8 | 100.0 | 0.058 |

## Ablation: audio_guard (ALL, libri-test-clean + ted-test)

| System | WER % | CER % | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Onset |Δt| ms | ≤250 ms % | CPS≤17 % | CPS≤20 % | Lines ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abl_audio_guard_guard_on | 2.80 | 1.63 | – | 81.3 | 1.44 | 73.8 | 42 | 97.4 | 92.8 | 99.8 | 100.0 | 0.058 |
| abl_audio_guard_guard_off | 3.36 | 1.92 | – | 76.9 | 1.95 | 73.8 | 42 | 97.3 | 93.2 | 99.8 | 100.0 | 0.063 |

## Ablation: fusion_signals (ALL, libri-test-clean + ted-test)

| System | WER % | CER % | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Onset |Δt| ms | ≤250 ms % | CPS≤17 % | CPS≤20 % | Lines ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abl_fusion_signals_full | 2.80 | 1.63 | – | 81.3 | 1.44 | 73.8 | 42 | 97.4 | 92.8 | 99.8 | 100.0 | 0.058 |
| abl_fusion_signals_no_conf | 2.78 | 1.64 | – | 80.9 | 1.48 | 76.1 | 42 | 97.3 | 92.5 | 99.8 | 100.0 | 0.059 |
| abl_fusion_signals_no_phon | 2.83 | 1.65 | – | 81.3 | 1.42 | 75.6 | 42 | 97.4 | 92.8 | 99.8 | 100.0 | 0.056 |
| abl_fusion_signals_no_lm | 2.69 | 1.55 | – | 80.2 | 1.67 | 76.0 | 42 | 97.4 | 92.7 | 99.9 | 100.0 | 0.055 |
| abl_fusion_signals_no_oov | 2.80 | 1.63 | – | 81.3 | 1.44 | 73.8 | 42 | 97.4 | 92.8 | 99.8 | 100.0 | 0.057 |

## Ablation: asr_size (ALL, libri-test-clean + ted-test)

| System | WER % | CER % | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Onset |Δt| ms | ≤250 ms % | CPS≤17 % | CPS≤20 % | Lines ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abl_asr_size_tiny_en | 4.06 | 2.20 | – | 78.2 | 1.19 | 74.9 | 40 | 97.6 | 92.6 | 99.8 | 100.0 | 1.872 |
| abl_asr_size_small_en | 3.18 | 1.83 | – | 81.1 | 1.25 | 77.4 | 41 | 97.6 | 92.7 | 99.8 | 100.0 | 0.064 |
| abl_asr_size_large_v3_turbo | 2.80 | 1.63 | – | 81.3 | 1.44 | 73.8 | 42 | 97.4 | 92.8 | 99.8 | 100.0 | 0.056 |

## Ablation: gec_model (ALL, libri-test-clean + ted-test)

| System | WER % | CER % | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Onset |Δt| ms | ≤250 ms % | CPS≤17 % | CPS≤20 % | Lines ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abl_gec_model_none | 2.76 | 1.61 | – | 81.7 | 1.38 | 74.3 | 41 | 97.5 | 92.8 | 99.8 | 100.0 | 0.054 |
| abl_gec_model_own_transformer | 2.80 | 1.63 | – | 81.3 | 1.44 | 73.8 | 42 | 97.4 | 92.8 | 99.8 | 100.0 | 0.058 |

## Corruption-rate sweep (libri-test-clean + ted-test)

| Injected rate | No-op WER % | Ours WER % |
|---|---|---|
| 5% | 5.29 | 2.41 |
| 10% | 9.87 | 2.80 |
| 20% | 17.46 | 3.14 |
