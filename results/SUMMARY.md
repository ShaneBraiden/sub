# Results summary (test split)

## ALL

Mode A (audio + corrupted subtitles):

| System | WER % | CER % | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Onset |Δt| ms | ≤250 ms % | CPS≤17 % | CPS≤20 % | Lines ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 10.88 | 5.14 | 0.0 | 0.0 | 0.00 | 30.6 | 9365 | 7.1 | 88.2 | 97.2 | 82.6 | 0.000 |
| B1: SymSpell + LanguageTool | 8.82 | 4.85 | 43.3 | 46.2 | 0.90 | 31.2 | 9362 | 7.1 | 88.6 | 97.4 | 82.9 | 0.035 |
| B2: ffsubsync (timing only) | 11.45 | 5.78 | 0.0 | 0.2 | 0.64 | 30.7 | 5633 | 27.3 | 86.2 | 97.0 | 82.5 | 0.005 |
| B3: raw Whisper replace | 3.10 | 1.84 | 69.5 | 70.7 | 3.48 | 71.3 | 235 | 59.2 | 80.4 | 93.5 | 27.5 | 0.031 |
| Ours with the linear fusion rule | 2.72 | 1.54 | 77.2 | 82.3 | 1.39 | 70.4 | 39 | 97.8 | 92.5 | 99.9 | 100.0 | 0.056 |
| **Ours** (all self-trained models except Whisper) | 2.54 | 1.32 | 79.9 | 87.3 | 0.70 | 71.8 | 36 | 98.3 | 91.8 | 99.8 | 100.0 | 0.081 |

Mode B (audio only):

| System | WER % | CER % | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Onset |Δt| ms | ≤250 ms % | CPS≤17 % | CPS≤20 % | Lines ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Mode B: raw Whisper | 3.10 | 1.84 | – | – | – | 71.3 | 235 | 59.2 | 80.4 | 93.5 | 27.5 | 0.031 |
| Mode B: Ours | 3.20 | 1.88 | – | – | – | 77.7 | 42 | 97.3 | 92.1 | 99.8 | 100.0 | 0.061 |

## libri-test-clean

Mode A (audio + corrupted subtitles):

| System | WER % | CER % | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Onset |Δt| ms | ≤250 ms % | CPS≤17 % | CPS≤20 % | Lines ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 11.70 | 5.59 | 0.0 | 0.0 | 0.00 | 41.5 | 2497 | 0.9 | 91.8 | 97.7 | 86.1 | 0.000 |
| B1: SymSpell + LanguageTool | 9.51 | 5.16 | 38.3 | 43.6 | 0.85 | 41.4 | 2499 | 0.9 | 91.6 | 97.7 | 86.3 | 0.036 |
| B2: ffsubsync (timing only) | 11.95 | 5.82 | 0.0 | 0.0 | 0.30 | 41.5 | 459 | 46.3 | 90.4 | 97.5 | 86.0 | 0.003 |
| B3: raw Whisper replace | 2.13 | 0.70 | 63.5 | 81.9 | 1.62 | 68.4 | 273 | 46.2 | 91.8 | 98.9 | 21.2 | 0.029 |
| Ours with the linear fusion rule | 2.16 | 0.99 | 72.4 | 88.0 | 0.65 | 74.6 | 24 | 99.6 | 99.0 | 100.0 | 100.0 | 0.049 |
| **Ours** (all self-trained models except Whisper) | 2.22 | 1.02 | 72.5 | 88.7 | 0.53 | 76.9 | 24 | 99.8 | 98.8 | 100.0 | 100.0 | 0.075 |

Mode B (audio only):

| System | WER % | CER % | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Onset |Δt| ms | ≤250 ms % | CPS≤17 % | CPS≤20 % | Lines ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Mode B: raw Whisper | 2.13 | 0.70 | – | – | – | 68.4 | 273 | 46.2 | 91.8 | 98.9 | 21.2 | 0.029 |
| Mode B: Ours | 2.27 | 0.77 | – | – | – | 76.1 | 26 | 99.8 | 97.6 | 100.0 | 100.0 | 0.056 |

## libri-test-other

Mode A (audio + corrupted subtitles):

| System | WER % | CER % | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Onset |Δt| ms | ≤250 ms % | CPS≤17 % | CPS≤20 % | Lines ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 12.18 | 5.53 | 0.0 | 0.0 | 0.00 | 19.7 | 2782 | 11.8 | 91.2 | 98.0 | 86.0 | 0.000 |
| B1: SymSpell + LanguageTool | 10.14 | 5.40 | 31.1 | 43.8 | 1.00 | 20.5 | 2779 | 11.7 | 91.6 | 98.0 | 86.0 | 0.031 |
| B2: ffsubsync (timing only) | 12.78 | 6.29 | 0.0 | 0.0 | 0.71 | 19.8 | 633 | 35.1 | 91.4 | 97.7 | 85.9 | 0.002 |
| B3: raw Whisper replace | 4.16 | 2.56 | 66.1 | 74.0 | 4.06 | 74.1 | 323 | 36.0 | 87.6 | 96.6 | 27.0 | 0.028 |
| Ours with the linear fusion rule | 2.44 | 1.07 | 72.9 | 86.6 | 1.27 | 67.0 | 26 | 99.8 | 91.6 | 100.0 | 100.0 | 0.048 |
| **Ours** (all self-trained models except Whisper) | 2.02 | 0.83 | 74.3 | 89.9 | 0.83 | 67.9 | 24 | 100.0 | 91.8 | 100.0 | 100.0 | 0.078 |

Mode B (audio only):

| System | WER % | CER % | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Onset |Δt| ms | ≤250 ms % | CPS≤17 % | CPS≤20 % | Lines ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Mode B: raw Whisper | 4.16 | 2.56 | – | – | – | 74.1 | 323 | 36.0 | 87.6 | 96.6 | 27.0 | 0.028 |
| Mode B: Ours | 4.24 | 2.59 | – | – | – | 79.6 | 25 | 100.0 | 91.8 | 99.8 | 100.0 | 0.058 |

## ted-test

Mode A (audio + corrupted subtitles):

| System | WER % | CER % | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Onset |Δt| ms | ≤250 ms % | CPS≤17 % | CPS≤20 % | Lines ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 10.44 | 4.96 | 0.0 | 0.0 | 0.00 | – | 12774 | 7.5 | 86.5 | 96.8 | 80.8 | 0.000 |
| B1: SymSpell + LanguageTool | 8.41 | 4.67 | 50.0 | 47.3 | 0.88 | – | 12767 | 7.5 | 87.0 | 97.2 | 81.2 | 0.038 |
| B2: ffsubsync (timing only) | 11.08 | 5.66 | 0.0 | 0.3 | 0.70 | – | 8218 | 20.5 | 83.7 | 96.6 | 80.8 | 0.008 |
| B3: raw Whisper replace | 3.09 | 1.92 | 73.9 | 67.4 | 3.75 | – | 213 | 65.2 | 77.3 | 92.1 | 28.6 | 0.035 |
| Ours with the linear fusion rule | 2.90 | 1.75 | 81.8 | 79.8 | 1.57 | – | 46 | 96.9 | 91.2 | 99.8 | 100.0 | 0.066 |
| **Ours** (all self-trained models except Whisper) | 2.72 | 1.49 | 86.9 | 86.2 | 0.71 | – | 41 | 97.6 | 90.2 | 99.7 | 100.0 | 0.087 |

Mode B (audio only):

| System | WER % | CER % | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Onset |Δt| ms | ≤250 ms % | CPS≤17 % | CPS≤20 % | Lines ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Mode B: raw Whisper | 3.09 | 1.92 | – | – | – | – | 213 | 65.2 | 77.3 | 92.1 | 28.6 | 0.035 |
| Mode B: Ours | 3.18 | 1.96 | – | – | – | – | 50 | 96.2 | 91.0 | 99.8 | 100.0 | 0.066 |

## Objectives (imp.md 1.2)

| # | Target | Result | Met |
|---|---|---|---|
| O1 | ≥30% relative WER reduction (Mode A) | 10.88% → 2.54% (77% relative) | yes |
| O2 | Mode B: lower WER and higher punct F1 than raw Whisper | WER 3.10% → 3.20%, punct F1 71.3 → 77.7 | partly/no |
| O3 | ≥90% of cue boundaries within ±250 ms | 98.3% of cue onsets | yes |
| O4 | ≥95% cues ≤42 chars/line, ≤2 lines, ≤17–20 CPS | lines 100.0%, 2-line 100.0%, CPS≤20 99.8%, CPS≤17 91.8% | yes (at 20 CPS) |
| O5 | Demo: upload → diff → download | web app (`app/backend/` + `frontend/`), tested by `tests/test_backend.py` and `npm test` | yes |

Per-error-type recall of **Ours** (share of injected errors repaired):

| asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|
| 43.1% (422/978) | 93.2% (686/736) | 84.5% (310/367) | 82.6% (109/132) | 97.1% (715/736) | 98.1% (1081/1102) |

## Self-trained models vs the earlier pretrained ones (ALL test files)

Same benchmark, ASR (Whisper large-v3-turbo) and pipeline; v1 rows are from `results/pretrained_v1/`.

| System | WER % | CER % | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Onset |Δt| ms | ≤250 ms % | CPS≤17 % | CPS≤20 % | Lines ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ours v1: distilgpt2 + FullStop XLM-R + T5-base GEC + MMS-300M + spaCy | 3.03 | 1.78 | 75.1 | 80.4 | 1.42 | 72.6 | 50 | 97.1 | 92.4 | 99.8 | 100.0 | 0.058 |
| Ours v1 + fine-tuned T5-small | 2.98 | 1.73 | 75.7 | 81.1 | 1.35 | 72.5 | 50 | 97.2 | 92.0 | 99.8 | 100.0 | 0.056 |
| **Ours** (all self-trained models except Whisper) | 2.54 | 1.32 | 79.9 | 87.3 | 0.70 | 71.8 | 36 | 98.3 | 91.8 | 99.8 | 100.0 | 0.081 |
| Mode B: Ours v1 (pretrained) | 3.24 | 1.93 | – | – | – | 79.5 | 53 | 96.8 | 91.7 | 99.8 | 100.0 | 0.063 |
| Mode B: Ours (self-trained) | 3.20 | 1.88 | – | – | – | 77.7 | 42 | 97.3 | 92.1 | 99.8 | 100.0 | 0.061 |

## The fusion network against the alternatives, paired over test files

Files resampled 10,000 times. The first two rows use all test files; the others use the ablation's libri-test-clean + ted-test. Negative WER and positive F0.5 favour the network.

| Network minus | WER points (95% CI) | Edit F0.5 points (95% CI) | Files with lower WER |
|---|---|---|---|
| Linear rule, all test files | -0.18 (-0.44 to +0.09) | +5.0 (+2.8 to +6.8) | 14 of 27 |
| Raw Whisper, all test files | -0.56 (-1.35 to +0.05) | +16.6 (+13.1 to +19.2) | 17 of 27 |
| Linear rule tuned on F0.5 alone | -0.53 (-0.82 to -0.22) | +6.4 (+3.9 to +8.3) | 15 of 19 |
| Logistic regression | +0.01 (-0.11 to +0.14) | +1.5 (+0.6 to +2.6) | 8 of 19 |
| MLP per candidate | -0.07 (-0.15 to +0.02) | +0.3 (-0.3 to +0.9) | 11 of 19 |
| Network + word embeddings | -0.08 (-0.15 to -0.01) | +0.7 (+0.2 to +1.3) | 11 of 19 |
| Network without CTC evidence | -0.25 (-0.36 to -0.13) | +0.5 (-0.3 to +1.4) | 12 of 19 |

## Ablation: fusion_method (ALL, libri-test-clean + ted-test)

| System | WER % | CER % | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Onset |Δt| ms | ≤250 ms % | CPS≤17 % | CPS≤20 % | Lines ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abl_fusion_method_rule_v1 | 2.80 | 1.63 | – | 81.3 | 1.44 | 73.8 | 42 | 97.4 | 92.8 | 99.8 | 100.0 | 0.064 |
| abl_fusion_method_rule_f05 | 3.16 | 1.79 | – | 80.3 | 1.31 | 76.0 | 42 | 97.3 | 92.8 | 99.9 | 100.0 | 0.063 |
| abl_fusion_method_rule | 2.77 | 1.62 | – | 81.4 | 1.41 | 74.6 | 42 | 97.4 | 92.7 | 99.8 | 100.0 | 0.063 |
| abl_fusion_method_logistic | 2.62 | 1.40 | – | 85.2 | 0.81 | 77.0 | 39 | 97.8 | 92.1 | 99.7 | 100.0 | 0.082 |
| abl_fusion_method_mlp | 2.70 | 1.47 | – | 86.4 | 0.65 | 75.9 | 40 | 97.8 | 92.0 | 99.8 | 100.0 | 0.082 |
| abl_fusion_method_words | 2.72 | 1.44 | – | 86.0 | 0.69 | 77.2 | 40 | 97.8 | 92.1 | 99.8 | 100.0 | 0.084 |
| abl_fusion_method_no_ctc | 2.88 | 1.59 | – | 86.2 | 0.61 | 77.2 | 41 | 97.6 | 91.5 | 99.8 | 100.0 | 0.062 |
| abl_fusion_method_neural | 2.63 | 1.41 | – | 86.7 | 0.68 | 76.9 | 38 | 98.0 | 91.8 | 99.8 | 100.0 | 0.085 |

## Ablation: timing_source (ALL, libri-test-clean + ted-test)

| System | WER % | CER % | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Onset |Δt| ms | ≤250 ms % | CPS≤17 % | CPS≤20 % | Lines ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abl_timing_source_whisper_dtw | 2.63 | 1.41 | – | 86.7 | 0.68 | 76.9 | 237 | 62.8 | 90.4 | 99.6 | 100.0 | 0.062 |
| abl_timing_source_own_ctc | 2.63 | 1.41 | – | 86.7 | 0.68 | 76.9 | 38 | 98.0 | 91.8 | 99.8 | 100.0 | 0.082 |

Word-level timing vs MFA (mean |Δt| ms): whisper_dtw 92, own_ctc 55

## Ablation: segmenter (ALL, libri-test-clean + ted-test)

| System | WER % | CER % | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Onset |Δt| ms | ≤250 ms % | CPS≤17 % | CPS≤20 % | Lines ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abl_segmenter_greedy | 2.63 | 1.41 | – | 86.7 | 0.68 | 76.9 | 33 | 98.6 | 56.0 | 61.0 | 100.0 | 0.083 |
| abl_segmenter_dp | 2.63 | 1.41 | – | 86.7 | 0.68 | 76.9 | 38 | 98.0 | 91.8 | 99.8 | 100.0 | 0.085 |

## Ablation: audio_guard (ALL, libri-test-clean + ted-test)

| System | WER % | CER % | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Onset |Δt| ms | ≤250 ms % | CPS≤17 % | CPS≤20 % | Lines ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abl_audio_guard_guard_on | 2.63 | 1.41 | – | 86.7 | 0.68 | 76.9 | 38 | 98.0 | 91.8 | 99.8 | 100.0 | 0.085 |
| abl_audio_guard_guard_off | 3.28 | 1.78 | – | 81.2 | 1.34 | 77.1 | 39 | 97.9 | 92.3 | 99.8 | 100.0 | 0.089 |

## Ablation: fusion_signals (ALL, libri-test-clean + ted-test)

| System | WER % | CER % | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Onset |Δt| ms | ≤250 ms % | CPS≤17 % | CPS≤20 % | Lines ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abl_fusion_signals_full | 2.77 | 1.62 | – | 81.4 | 1.41 | 74.6 | 42 | 97.4 | 92.7 | 99.8 | 100.0 | 0.063 |
| abl_fusion_signals_no_conf | 3.13 | 1.77 | – | 80.1 | 1.34 | 76.3 | 42 | 97.4 | 92.7 | 99.9 | 100.0 | 0.066 |
| abl_fusion_signals_no_phon | 2.72 | 1.58 | – | 81.3 | 1.47 | 75.0 | 42 | 97.4 | 92.7 | 99.8 | 100.0 | 0.063 |
| abl_fusion_signals_no_lm | 2.69 | 1.55 | – | 80.1 | 1.68 | 76.9 | 42 | 97.4 | 92.7 | 99.8 | 100.0 | 0.063 |
| abl_fusion_signals_no_oov | 2.77 | 1.62 | – | 81.4 | 1.41 | 74.6 | 42 | 97.4 | 92.7 | 99.8 | 100.0 | 0.065 |

## Ablation: asr_size (ALL, libri-test-clean + ted-test)

| System | WER % | CER % | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Onset |Δt| ms | ≤250 ms % | CPS≤17 % | CPS≤20 % | Lines ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abl_asr_size_tiny_en | 3.75 | 1.93 | – | 82.1 | 0.76 | 75.8 | 39 | 97.7 | 92.4 | 99.7 | 100.0 | 1.899 |
| abl_asr_size_small_en | 2.92 | 1.55 | – | 85.5 | 0.72 | 76.5 | 40 | 97.8 | 92.3 | 99.7 | 100.0 | 0.090 |
| abl_asr_size_large_v3_turbo | 2.63 | 1.41 | – | 86.7 | 0.68 | 76.9 | 38 | 98.0 | 91.8 | 99.8 | 100.0 | 0.083 |

## Ablation: gec_model (ALL, libri-test-clean + ted-test)

| System | WER % | CER % | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Onset |Δt| ms | ≤250 ms % | CPS≤17 % | CPS≤20 % | Lines ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abl_gec_model_none | 2.42 | 1.28 | – | 88.7 | 0.42 | 76.9 | 35 | 98.3 | 91.8 | 99.8 | 100.0 | 0.079 |
| abl_gec_model_own_transformer | 2.63 | 1.41 | – | 86.7 | 0.68 | 76.9 | 38 | 98.0 | 91.8 | 99.8 | 100.0 | 0.085 |

## Corruption-rate sweep (libri-test-clean + ted-test)

| Injected rate | No-op WER % | Linear rule WER % | Ours WER % |
|---|---|---|---|
| 5% | 5.29 | 2.39 | 2.11 |
| 10% | 9.87 | 2.78 | 2.68 |
| 20% | 17.46 | 3.06 | 2.81 |
