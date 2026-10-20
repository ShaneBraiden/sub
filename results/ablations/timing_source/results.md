# Ablation: timing_source

## ALL

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| whisper_dtw | 2.80 | 1.63 | – | 81.3 | 1.44 | 73.8 | 97.5 | 233 | 38.2 | 63.4 | 90 | 91.3 | 99.9 | 100.0 | 91.3 | 0.044 |
| own_ctc | 2.80 | 1.63 | – | 81.3 | 1.44 | 73.8 | 97.5 | 42 | 95.3 | 97.4 | 57 | 92.8 | 99.8 | 100.0 | 92.8 | 0.058 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|---|
| whisper_dtw | 38.1 (311/816) | 91.6 (576/629) | 86.3 (270/313) | 84.4 (54/64) | 96.2 (605/629) | 97.2 (914/940) |
| own_ctc | 38.1 (311/816) | 91.6 (576/629) | 86.3 (270/313) | 84.4 (54/64) | 96.2 (605/629) | 97.2 (914/940) |

## libri-test-clean

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| whisper_dtw | 2.16 | 0.99 | – | 87.9 | 0.67 | 73.8 | 97.5 | 337 | 23.0 | 44.1 | 82 | 98.9 | 100.0 | 100.0 | 98.9 | 0.039 |
| own_ctc | 2.16 | 0.99 | – | 87.9 | 0.67 | 73.8 | 97.5 | 25 | 99.0 | 99.6 | 50 | 99.0 | 100.0 | 100.0 | 99.0 | 0.055 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|---|
| whisper_dtw | 49.0 (73/149) | 94.6 (106/112) | 81.5 (44/54) | 84.4 (54/64) | 98.2 (110/112) | 98.8 (162/164) |
| own_ctc | 49.0 (73/149) | 94.6 (106/112) | 81.5 (44/54) | 84.4 (54/64) | 98.2 (110/112) | 98.8 (162/164) |

## ted-test

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| whisper_dtw | 2.93 | 1.77 | – | 79.7 | 1.60 | – | – | 210 | 41.5 | 67.7 | 92 | 89.6 | 99.9 | 100.0 | 89.6 | 0.047 |
| own_ctc | 2.93 | 1.77 | – | 79.7 | 1.60 | – | – | 46 | 94.4 | 96.9 | 58 | 91.4 | 99.8 | 100.0 | 91.4 | 0.061 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | homophone | spelling |
|---|---|---|---|---|---|
| whisper_dtw | 35.7 (238/667) | 90.9 (470/517) | 87.3 (226/259) | 95.7 (495/517) | 96.9 (752/776) |
| own_ctc | 35.7 (238/667) | 90.9 (470/517) | 87.3 (226/259) | 95.7 (495/517) | 96.9 (752/776) |
