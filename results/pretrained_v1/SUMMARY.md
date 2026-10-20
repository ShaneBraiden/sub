# Results summary (test split)

## ALL

Mode A (audio + corrupted subtitles):

| System | WER % | CER % | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Onset |Δt| ms | ≤250 ms % | CPS≤17 % | CPS≤20 % | Lines ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 10.88 | 5.14 | 0.0 | 0.0 | 0.00 | 30.6 | 9365 | 7.1 | 88.2 | 97.2 | 82.6 | 0.000 |
| B1: SymSpell + LanguageTool | 8.82 | 4.85 | 43.3 | 46.2 | 0.90 | 31.2 | 9362 | 7.1 | 88.6 | 97.4 | 82.9 | 0.036 |
| B2: ffsubsync (timing only) | 11.45 | 5.78 | 0.0 | 0.2 | 0.64 | 30.7 | 5633 | 27.3 | 86.2 | 97.0 | 82.5 | 0.004 |
| B3: raw Whisper replace | 3.10 | 1.84 | 69.5 | 70.7 | 3.48 | 71.3 | 235 | 59.2 | 80.4 | 93.5 | 27.5 | 0.031 |
| **Ours** (fusion + punct + T5 GEC + MMS align + DP seg) | 3.03 | 1.78 | 75.1 | 80.4 | 1.42 | 72.6 | 50 | 97.1 | 92.4 | 99.8 | 100.0 | 0.058 |
| Ours + CoEdIT-large | 3.11 | 1.81 | 74.9 | 79.7 | 1.45 | 73.2 | 49 | 97.3 | 92.2 | 99.8 | 100.0 | 0.091 |
| Ours + fine-tuned T5-small | 2.98 | 1.73 | 75.7 | 81.1 | 1.35 | 72.5 | 50 | 97.2 | 92.0 | 99.8 | 100.0 | 0.056 |

Mode B (audio only):

| System | WER % | CER % | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Onset |Δt| ms | ≤250 ms % | CPS≤17 % | CPS≤20 % | Lines ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Mode B: raw Whisper | 3.10 | 1.84 | – | – | – | 71.3 | 235 | 59.2 | 80.4 | 93.5 | 27.5 | 0.031 |
| Mode B: Ours | 3.24 | 1.93 | – | – | – | 79.5 | 53 | 96.8 | 91.7 | 99.8 | 100.0 | 0.063 |

## libri-test-clean

Mode A (audio + corrupted subtitles):

| System | WER % | CER % | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Onset |Δt| ms | ≤250 ms % | CPS≤17 % | CPS≤20 % | Lines ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 11.70 | 5.59 | 0.0 | 0.0 | 0.00 | 41.5 | 2497 | 0.9 | 91.8 | 97.7 | 86.1 | 0.000 |
| B1: SymSpell + LanguageTool | 9.51 | 5.16 | 38.3 | 43.6 | 0.85 | 41.4 | 2499 | 0.9 | 91.6 | 97.7 | 86.3 | 0.036 |
| B2: ffsubsync (timing only) | 11.95 | 5.82 | 0.0 | 0.0 | 0.30 | 41.5 | 459 | 46.3 | 90.4 | 97.5 | 86.0 | 0.002 |
| B3: raw Whisper replace | 2.13 | 0.70 | 63.5 | 81.9 | 1.62 | 68.4 | 273 | 46.2 | 91.8 | 98.9 | 21.2 | 0.029 |
| **Ours** (fusion + punct + T5 GEC + MMS align + DP seg) | 2.52 | 1.29 | 73.2 | 85.1 | 0.69 | 81.8 | 36 | 99.6 | 98.8 | 100.0 | 100.0 | 0.058 |
| Ours + CoEdIT-large | 2.66 | 1.40 | 73.0 | 84.4 | 0.69 | 81.6 | 36 | 99.6 | 98.8 | 100.0 | 100.0 | 0.062 |
| Ours + fine-tuned T5-small | 2.54 | 1.31 | 73.6 | 85.3 | 0.67 | 81.6 | 36 | 99.6 | 98.8 | 100.0 | 100.0 | 0.054 |

Mode B (audio only):

| System | WER % | CER % | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Onset |Δt| ms | ≤250 ms % | CPS≤17 % | CPS≤20 % | Lines ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Mode B: raw Whisper | 2.13 | 0.70 | – | – | – | 68.4 | 273 | 46.2 | 91.8 | 98.9 | 21.2 | 0.029 |
| Mode B: Ours | 2.36 | 0.84 | – | – | – | 80.6 | 37 | 99.4 | 97.9 | 100.0 | 100.0 | 0.054 |

## libri-test-other

Mode A (audio + corrupted subtitles):

| System | WER % | CER % | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Onset |Δt| ms | ≤250 ms % | CPS≤17 % | CPS≤20 % | Lines ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 12.18 | 5.53 | 0.0 | 0.0 | 0.00 | 19.7 | 2782 | 11.8 | 91.2 | 98.0 | 86.0 | 0.000 |
| B1: SymSpell + LanguageTool | 10.14 | 5.40 | 31.1 | 43.8 | 1.00 | 20.5 | 2779 | 11.7 | 91.6 | 98.0 | 86.0 | 0.032 |
| B2: ffsubsync (timing only) | 12.78 | 6.29 | 0.0 | 0.0 | 0.71 | 19.8 | 633 | 35.1 | 91.4 | 97.7 | 85.9 | 0.002 |
| B3: raw Whisper replace | 4.16 | 2.56 | 66.1 | 74.0 | 4.06 | 74.1 | 323 | 36.0 | 87.6 | 96.6 | 27.0 | 0.028 |
| **Ours** (fusion + punct + T5 GEC + MMS align + DP seg) | 3.14 | 1.55 | 67.7 | 83.8 | 1.37 | 65.7 | 31 | 100.0 | 93.1 | 99.8 | 100.0 | 0.051 |
| Ours + CoEdIT-large | 3.15 | 1.52 | 68.2 | 83.8 | 1.35 | 66.9 | 31 | 100.0 | 92.9 | 99.8 | 100.0 | 0.079 |
| Ours + fine-tuned T5-small | 2.92 | 1.43 | 68.3 | 85.5 | 1.24 | 65.7 | 31 | 100.0 | 92.7 | 99.8 | 100.0 | 0.055 |

Mode B (audio only):

| System | WER % | CER % | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Onset |Δt| ms | ≤250 ms % | CPS≤17 % | CPS≤20 % | Lines ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Mode B: raw Whisper | 4.16 | 2.56 | – | – | – | 74.1 | 323 | 36.0 | 87.6 | 96.6 | 27.0 | 0.028 |
| Mode B: Ours | 4.35 | 2.71 | – | – | – | 78.8 | 32 | 100.0 | 93.8 | 99.8 | 100.0 | 0.055 |

## ted-test

Mode A (audio + corrupted subtitles):

| System | WER % | CER % | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Onset |Δt| ms | ≤250 ms % | CPS≤17 % | CPS≤20 % | Lines ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 10.44 | 4.96 | 0.0 | 0.0 | 0.00 | – | 12774 | 7.5 | 86.5 | 96.8 | 80.8 | 0.000 |
| B1: SymSpell + LanguageTool | 8.41 | 4.67 | 50.0 | 47.3 | 0.88 | – | 12767 | 7.5 | 87.0 | 97.2 | 81.2 | 0.038 |
| B2: ffsubsync (timing only) | 11.08 | 5.66 | 0.0 | 0.3 | 0.70 | – | 8218 | 20.5 | 83.7 | 96.6 | 80.8 | 0.007 |
| B3: raw Whisper replace | 3.09 | 1.92 | 73.9 | 67.4 | 3.75 | – | 213 | 65.2 | 77.3 | 92.1 | 28.6 | 0.035 |
| **Ours** (fusion + punct + T5 GEC + MMS align + DP seg) | 3.12 | 1.93 | 80.3 | 78.5 | 1.58 | – | 58 | 95.9 | 90.8 | 99.7 | 100.0 | 0.062 |
| Ours + CoEdIT-large | 3.19 | 1.96 | 79.7 | 77.5 | 1.64 | – | 56 | 96.1 | 90.6 | 99.7 | 100.0 | 0.122 |
| Ours + fine-tuned T5-small | 3.08 | 1.88 | 80.9 | 79.1 | 1.51 | – | 58 | 96.0 | 90.3 | 99.7 | 100.0 | 0.059 |

Mode B (audio only):

| System | WER % | CER % | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Onset |Δt| ms | ≤250 ms % | CPS≤17 % | CPS≤20 % | Lines ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Mode B: raw Whisper | 3.09 | 1.92 | – | – | – | – | 213 | 65.2 | 77.3 | 92.1 | 28.6 | 0.035 |
| Mode B: Ours | 3.20 | 1.99 | – | – | – | – | 61 | 95.5 | 89.9 | 99.8 | 100.0 | 0.074 |

## Objectives (imp.md 1.2)

| # | Target | Result | Met |
|---|---|---|---|
| O1 | ≥30% relative WER reduction (Mode A) | 10.88% → 3.03% (72% relative) | yes |
| O2 | Mode B: lower WER and higher punct F1 than raw Whisper | WER 3.10% → 3.24%, punct F1 71.3 → 79.5 | partly/no |
| O3 | ≥90% of cue boundaries within ±250 ms | 97.1% of cue onsets | yes |
| O4 | ≥95% cues ≤42 chars/line, ≤2 lines, ≤17–20 CPS | lines 100.0%, 2-line 100.0%, CPS≤20 99.8%, CPS≤17 92.4% | yes (at 20 CPS) |
| O5 | Demo: upload → diff → download | Gradio app (`app/gradio_app.py`), smoke-tested via `tests/smoke_gradio.py` | yes |

Per-error-type recall of **Ours** (share of injected errors repaired):

| asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|
| 42.7% (418/978) | 74.6% (549/736) | 86.6% (318/367) | 79.5% (105/132) | 96.2% (708/736) | 96.8% (1067/1102) |

## Ablation: timing_source (ALL, libri-test-clean + ted-test)

| System | WER % | CER % | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Onset |Δt| ms | ≤250 ms % | CPS≤17 % | CPS≤20 % | Lines ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abl_timing_source_whisper_dtw | 3.01 | 1.82 | – | 79.7 | 1.43 | 81.8 | 244 | 61.9 | 91.5 | 99.8 | 100.0 | 0.052 |
| abl_timing_source_w2v2_ctc | 3.01 | 1.82 | – | 79.7 | 1.43 | 81.8 | 60 | 96.9 | 92.3 | 99.9 | 100.0 | 0.058 |
| abl_timing_source_mms_ctc | 3.01 | 1.82 | – | 79.7 | 1.43 | 81.8 | 54 | 96.6 | 92.3 | 99.8 | 100.0 | 0.061 |

Word-level timing vs MFA (mean |Δt| ms): whisper_dtw 90, w2v2_ctc 53, mms_ctc 43

## Ablation: segmenter (ALL, libri-test-clean + ted-test)

| System | WER % | CER % | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Onset |Δt| ms | ≤250 ms % | CPS≤17 % | CPS≤20 % | Lines ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abl_segmenter_greedy | 3.01 | 1.82 | – | 79.7 | 1.43 | 81.8 | 42 | 98.5 | 56.8 | 60.6 | 100.0 | 0.062 |
| abl_segmenter_dp | 3.01 | 1.82 | – | 79.7 | 1.43 | 81.8 | 54 | 96.6 | 92.3 | 99.8 | 100.0 | 0.059 |

## Ablation: audio_guard (ALL, libri-test-clean + ted-test)

| System | WER % | CER % | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Onset |Δt| ms | ≤250 ms % | CPS≤17 % | CPS≤20 % | Lines ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abl_audio_guard_guard_on | 3.01 | 1.82 | – | 79.7 | 1.43 | 81.8 | 54 | 96.6 | 92.3 | 99.8 | 100.0 | 0.062 |
| abl_audio_guard_guard_off | 4.26 | 2.70 | – | 71.4 | 2.26 | 82.4 | 54 | 96.5 | 91.9 | 99.8 | 100.0 | 0.086 |

## Ablation: fusion_signals (ALL, libri-test-clean + ted-test)

| System | WER % | CER % | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Onset |Δt| ms | ≤250 ms % | CPS≤17 % | CPS≤20 % | Lines ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abl_fusion_signals_full | 3.01 | 1.82 | – | 79.7 | 1.43 | 81.8 | 54 | 96.6 | 92.3 | 99.8 | 100.0 | 0.062 |
| abl_fusion_signals_no_conf | 2.93 | 1.78 | – | 79.3 | 1.52 | 80.9 | 53 | 96.7 | 92.1 | 99.7 | 100.0 | 0.063 |
| abl_fusion_signals_no_phon | 2.94 | 1.77 | – | 79.5 | 1.50 | 81.2 | 54 | 96.6 | 92.0 | 99.8 | 100.0 | 11.595 |
| abl_fusion_signals_no_lm | 2.67 | 1.55 | – | 79.9 | 1.69 | 80.9 | 53 | 96.7 | 91.8 | 99.8 | 100.0 | 0.084 |
| abl_fusion_signals_no_oov | 3.01 | 1.82 | – | 79.8 | 1.42 | 81.8 | 54 | 96.6 | 92.3 | 99.8 | 100.0 | 0.054 |

## Ablation: asr_size (ALL, libri-test-clean + ted-test)

| System | WER % | CER % | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Onset |Δt| ms | ≤250 ms % | CPS≤17 % | CPS≤20 % | Lines ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abl_asr_size_tiny_en | 4.26 | 2.45 | – | 76.1 | 1.22 | 81.2 | 52 | 96.9 | 91.9 | 99.8 | 100.0 | 1.885 |
| abl_asr_size_small_en | 3.45 | 2.08 | – | 79.2 | 1.28 | 81.3 | 54 | 96.4 | 91.5 | 99.8 | 100.0 | 0.063 |
| abl_asr_size_large_v3_turbo | 3.01 | 1.82 | – | 79.7 | 1.43 | 81.8 | 54 | 96.6 | 92.3 | 99.8 | 100.0 | 0.051 |

## Ablation: gec_model (ALL, libri-test-clean + ted-test)

| System | WER % | CER % | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Onset |Δt| ms | ≤250 ms % | CPS≤17 % | CPS≤20 % | Lines ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abl_gec_model_none | 2.97 | 1.77 | – | 80.9 | 1.30 | 81.7 | 53 | 96.7 | 91.9 | 99.8 | 100.0 | 0.048 |
| abl_gec_model_vennify_t5_base | 3.01 | 1.82 | – | 79.7 | 1.43 | 81.8 | 54 | 96.6 | 92.3 | 99.8 | 100.0 | 0.058 |
| abl_gec_model_coedit_large | 3.10 | 1.87 | – | 78.8 | 1.47 | 81.6 | 53 | 96.8 | 92.1 | 99.8 | 100.0 | 0.092 |
| abl_gec_model_ft_t5_small | 2.99 | 1.78 | – | 80.3 | 1.37 | 81.6 | 54 | 96.7 | 91.9 | 99.8 | 100.0 | 0.053 |

## Corruption-rate sweep (libri-test-clean + ted-test)

| Injected rate | No-op WER % | Ours WER % |
|---|---|---|
| 5% | 5.29 | 2.51 |
| 10% | 9.87 | 3.00 |
| 20% | 17.46 | 3.77 |
