# Results: test

## ALL

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 10.88 | 5.14 | 0.0 | 0.0 | 0.00 | 30.6 | 93.8 | 9365 | 2.4 | 7.1 | – | 88.2 | 97.2 | 82.6 | 40.9 | 0.000 |
| B1: SymSpell + LanguageTool | 8.82 | 4.85 | 43.3 | 46.2 | 0.90 | 31.2 | 95.8 | 9362 | 2.4 | 7.1 | 9834 | 88.6 | 97.4 | 82.9 | 41.2 | 0.035 |
| B2: ffsubsync only (timing) | 11.45 | 5.78 | 0.0 | 0.2 | 0.64 | 30.7 | 93.8 | 5633 | 11.1 | 27.3 | – | 86.2 | 97.0 | 82.5 | 40.7 | 0.005 |
| B3: Raw Whisper replace | 3.10 | 1.84 | 69.5 | 70.7 | 3.48 | 71.3 | 95.0 | 235 | 36.2 | 59.2 | 86 | 80.4 | 93.5 | 27.5 | 14.1 | 0.031 |
| Ours with the linear fusion rule instead of the network | 2.72 | 1.54 | 77.2 | 82.3 | 1.39 | 70.4 | 97.8 | 39 | 95.7 | 97.8 | 56 | 92.5 | 99.9 | 100.0 | 92.5 | 0.056 |
| Ours: neural fusion (own KN LM, own CTC) + own punct + own corrector + own CTC align + DP seg | 2.54 | 1.32 | 79.9 | 87.3 | 0.70 | 71.8 | 97.7 | 36 | 96.4 | 98.3 | 55 | 91.8 | 99.8 | 100.0 | 91.8 | 0.081 |
| Mode B: raw Whisper | 3.10 | 1.84 | – | – | – | 71.3 | 95.0 | 235 | 36.2 | 59.2 | 86 | 80.4 | 93.5 | 27.5 | 14.1 | 0.031 |
| Mode B: Ours (own punct + own corrector + own CTC align + DP seg) | 3.20 | 1.88 | – | – | – | 77.7 | 96.2 | 42 | 95.3 | 97.3 | 55 | 92.1 | 99.8 | 100.0 | 92.1 | 0.061 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 0.0 (0/978) | 0.7 (5/736) | 0.0 (0/367) | 0.0 (0/132) | 0.0 (0/736) | 0.0 (0/1102) |
| B1: SymSpell + LanguageTool | 4.5 (44/978) | 2.4 (18/736) | 1.4 (5/367) | 13.6 (18/132) | 19.0 (140/736) | 75.5 (832/1102) |
| B2: ffsubsync only (timing) | 0.0 (0/978) | 0.7 (5/736) | 0.5 (2/367) | 0.0 (0/132) | 0.0 (0/736) | 0.0 (0/1102) |
| B3: Raw Whisper replace | 52.8 (516/978) | 98.2 (723/736) | 85.3 (313/367) | 81.8 (108/132) | 95.7 (704/736) | 96.0 (1058/1102) |
| Ours with the linear fusion rule instead of the network | 43.3 (423/978) | 93.5 (688/736) | 85.3 (313/367) | 81.8 (108/132) | 96.7 (712/736) | 97.5 (1074/1102) |
| Ours: neural fusion (own KN LM, own CTC) + own punct + own corrector + own CTC align + DP seg | 43.1 (422/978) | 93.2 (686/736) | 84.5 (310/367) | 82.6 (109/132) | 97.1 (715/736) | 98.1 (1081/1102) |
| Mode B: raw Whisper | – | – | – | – | – | – |
| Mode B: Ours (own punct + own corrector + own CTC align + DP seg) | – | – | – | – | – | – |

## libri-test-clean

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 11.70 | 5.59 | 0.0 | 0.0 | 0.00 | 41.5 | 92.3 | 2497 | 0.0 | 0.9 | – | 91.8 | 97.7 | 86.1 | 52.6 | 0.000 |
| B1: SymSpell + LanguageTool | 9.51 | 5.16 | 38.3 | 43.6 | 0.85 | 41.4 | 95.2 | 2499 | 0.0 | 0.9 | 2486 | 91.6 | 97.7 | 86.3 | 52.8 | 0.036 |
| B2: ffsubsync only (timing) | 11.95 | 5.82 | 0.0 | 0.0 | 0.30 | 41.5 | 92.4 | 459 | 17.9 | 46.3 | – | 90.4 | 97.5 | 86.0 | 51.7 | 0.003 |
| B3: Raw Whisper replace | 2.13 | 0.70 | 63.5 | 81.9 | 1.62 | 68.4 | 94.1 | 273 | 30.9 | 46.2 | 81 | 91.8 | 98.9 | 21.2 | 16.8 | 0.029 |
| Ours with the linear fusion rule instead of the network | 2.16 | 0.99 | 72.4 | 88.0 | 0.65 | 74.6 | 97.5 | 24 | 99.2 | 99.6 | 50 | 99.0 | 100.0 | 100.0 | 99.0 | 0.049 |
| Ours: neural fusion (own KN LM, own CTC) + own punct + own corrector + own CTC align + DP seg | 2.22 | 1.02 | 72.5 | 88.7 | 0.53 | 76.9 | 97.5 | 24 | 99.4 | 99.8 | 50 | 98.8 | 100.0 | 100.0 | 98.8 | 0.075 |
| Mode B: raw Whisper | 2.13 | 0.70 | – | – | – | 68.4 | 94.1 | 273 | 30.9 | 46.2 | 81 | 91.8 | 98.9 | 21.2 | 16.8 | 0.029 |
| Mode B: Ours (own punct + own corrector + own CTC align + DP seg) | 2.27 | 0.77 | – | – | – | 76.1 | 95.8 | 26 | 99.0 | 99.8 | 50 | 97.6 | 100.0 | 100.0 | 97.6 | 0.056 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 0.0 (0/149) | 0.0 (0/112) | 0.0 (0/54) | 0.0 (0/64) | 0.0 (0/112) | 0.0 (0/164) |
| B1: SymSpell + LanguageTool | 4.7 (7/149) | 0.9 (1/112) | 0.0 (0/54) | 18.8 (12/64) | 17.9 (20/112) | 75.0 (123/164) |
| B2: ffsubsync only (timing) | 0.0 (0/149) | 0.0 (0/112) | 0.0 (0/54) | 0.0 (0/64) | 0.0 (0/112) | 0.0 (0/164) |
| B3: Raw Whisper replace | 59.7 (89/149) | 100.0 (112/112) | 83.3 (45/54) | 84.4 (54/64) | 98.2 (110/112) | 96.3 (158/164) |
| Ours with the linear fusion rule instead of the network | 49.0 (73/149) | 95.5 (107/112) | 81.5 (44/54) | 84.4 (54/64) | 98.2 (110/112) | 98.8 (162/164) |
| Ours: neural fusion (own KN LM, own CTC) + own punct + own corrector + own CTC align + DP seg | 47.0 (70/149) | 96.4 (108/112) | 83.3 (45/54) | 84.4 (54/64) | 98.2 (110/112) | 98.8 (162/164) |
| Mode B: raw Whisper | – | – | – | – | – | – |
| Mode B: Ours (own punct + own corrector + own CTC align + DP seg) | – | – | – | – | – | – |

## libri-test-other

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 12.18 | 5.53 | 0.0 | 0.0 | 0.00 | 19.7 | 95.3 | 2782 | 4.8 | 11.8 | – | 91.2 | 98.0 | 86.0 | 51.5 | 0.000 |
| B1: SymSpell + LanguageTool | 10.14 | 5.40 | 31.1 | 43.8 | 1.00 | 20.5 | 96.4 | 2779 | 4.7 | 11.7 | 2814 | 91.6 | 98.0 | 86.0 | 51.5 | 0.031 |
| B2: ffsubsync only (timing) | 12.78 | 6.29 | 0.0 | 0.0 | 0.71 | 19.8 | 95.3 | 633 | 14.8 | 35.1 | – | 91.4 | 97.7 | 85.9 | 51.8 | 0.002 |
| B3: Raw Whisper replace | 4.16 | 2.56 | 66.1 | 74.0 | 4.06 | 74.1 | 95.9 | 323 | 23.9 | 36.0 | 82 | 87.6 | 96.6 | 27.0 | 15.7 | 0.028 |
| Ours with the linear fusion rule instead of the network | 2.44 | 1.07 | 72.9 | 86.6 | 1.27 | 67.0 | 98.1 | 26 | 97.9 | 99.8 | 51 | 91.6 | 100.0 | 100.0 | 91.6 | 0.048 |
| Ours: neural fusion (own KN LM, own CTC) + own punct + own corrector + own CTC align + DP seg | 2.02 | 0.83 | 74.3 | 89.9 | 0.83 | 67.9 | 98.0 | 24 | 98.6 | 100.0 | 51 | 91.8 | 100.0 | 100.0 | 91.8 | 0.078 |
| Mode B: raw Whisper | 4.16 | 2.56 | – | – | – | 74.1 | 95.9 | 323 | 23.9 | 36.0 | 82 | 87.6 | 96.6 | 27.0 | 15.7 | 0.028 |
| Mode B: Ours (own punct + own corrector + own CTC align + DP seg) | 4.24 | 2.59 | – | – | – | 79.6 | 96.7 | 25 | 98.4 | 100.0 | 51 | 91.8 | 99.8 | 100.0 | 91.8 | 0.058 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 0.0 (0/162) | 0.0 (0/107) | 0.0 (0/54) | 0.0 (0/68) | 0.0 (0/107) | 0.0 (0/162) |
| B1: SymSpell + LanguageTool | 4.3 (7/162) | 0.0 (0/107) | 0.0 (0/54) | 8.8 (6/68) | 18.7 (20/107) | 75.9 (123/162) |
| B2: ffsubsync only (timing) | 0.0 (0/162) | 0.0 (0/107) | 0.0 (0/54) | 0.0 (0/68) | 0.0 (0/107) | 0.0 (0/162) |
| B3: Raw Whisper replace | 75.9 (123/162) | 99.1 (106/107) | 79.6 (43/54) | 79.4 (54/68) | 96.3 (103/107) | 96.9 (157/162) |
| Ours with the linear fusion rule instead of the network | 67.9 (110/162) | 97.2 (104/107) | 79.6 (43/54) | 79.4 (54/68) | 100.0 (107/107) | 98.1 (159/162) |
| Ours: neural fusion (own KN LM, own CTC) + own punct + own corrector + own CTC align + DP seg | 72.8 (118/162) | 94.4 (101/107) | 77.8 (42/54) | 80.9 (55/68) | 99.1 (106/107) | 98.1 (159/162) |
| Mode B: raw Whisper | – | – | – | – | – | – |
| Mode B: Ours (own punct + own corrector + own CTC align + DP seg) | – | – | – | – | – | – |

## ted-test

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 10.44 | 4.96 | 0.0 | 0.0 | 0.00 | – | – | 12774 | 2.4 | 7.5 | – | 86.5 | 96.8 | 80.8 | 35.2 | 0.000 |
| B1: SymSpell + LanguageTool | 8.41 | 4.67 | 50.0 | 47.3 | 0.88 | – | – | 12767 | 2.4 | 7.5 | 12829 | 87.0 | 97.2 | 81.2 | 35.6 | 0.038 |
| B2: ffsubsync only (timing) | 11.08 | 5.66 | 0.0 | 0.3 | 0.70 | – | – | 8218 | 8.4 | 20.5 | – | 83.7 | 96.6 | 80.8 | 35.1 | 0.008 |
| B3: Raw Whisper replace | 3.09 | 1.92 | 73.9 | 67.4 | 3.75 | – | – | 213 | 39.2 | 65.2 | 87 | 77.3 | 92.1 | 28.6 | 13.4 | 0.035 |
| Ours with the linear fusion rule instead of the network | 2.90 | 1.75 | 81.8 | 79.8 | 1.57 | – | – | 46 | 94.4 | 96.9 | 58 | 91.2 | 99.8 | 100.0 | 91.2 | 0.066 |
| Ours: neural fusion (own KN LM, own CTC) + own punct + own corrector + own CTC align + DP seg | 2.72 | 1.49 | 86.9 | 86.2 | 0.71 | – | – | 41 | 95.2 | 97.6 | 57 | 90.2 | 99.7 | 100.0 | 90.2 | 0.087 |
| Mode B: raw Whisper | 3.09 | 1.92 | – | – | – | – | – | 213 | 39.2 | 65.2 | 87 | 77.3 | 92.1 | 28.6 | 13.4 | 0.035 |
| Mode B: Ours (own punct + own corrector + own CTC align + DP seg) | 3.18 | 1.96 | – | – | – | – | – | 50 | 93.8 | 96.2 | 58 | 91.0 | 99.8 | 100.0 | 91.0 | 0.066 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | homophone | spelling |
|---|---|---|---|---|---|
| Corrupted input (no-op) | 0.0 (0/667) | 1.0 (5/517) | 0.0 (0/259) | 0.0 (0/517) | 0.0 (0/776) |
| B1: SymSpell + LanguageTool | 4.5 (30/667) | 3.3 (17/517) | 1.9 (5/259) | 19.3 (100/517) | 75.5 (586/776) |
| B2: ffsubsync only (timing) | 0.0 (0/667) | 1.0 (5/517) | 0.8 (2/259) | 0.0 (0/517) | 0.0 (0/776) |
| B3: Raw Whisper replace | 45.6 (304/667) | 97.7 (505/517) | 86.9 (225/259) | 95.0 (491/517) | 95.7 (743/776) |
| Ours with the linear fusion rule instead of the network | 36.0 (240/667) | 92.3 (477/517) | 87.3 (226/259) | 95.7 (495/517) | 97.0 (753/776) |
| Ours: neural fusion (own KN LM, own CTC) + own punct + own corrector + own CTC align + DP seg | 35.1 (234/667) | 92.3 (477/517) | 86.1 (223/259) | 96.5 (499/517) | 97.9 (760/776) |
| Mode B: raw Whisper | – | – | – | – | – |
| Mode B: Ours (own punct + own corrector + own CTC align + DP seg) | – | – | – | – | – |
