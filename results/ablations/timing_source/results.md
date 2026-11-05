# Ablation: timing_source

## ALL

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| whisper_dtw | 2.63 | 1.41 | – | 86.7 | 0.68 | 76.9 | 97.5 | 237 | 37.3 | 62.8 | 92 | 90.4 | 99.6 | 100.0 | 90.4 | 0.062 |
| own_ctc | 2.63 | 1.41 | – | 86.7 | 0.68 | 76.9 | 97.5 | 38 | 96.0 | 98.0 | 55 | 91.8 | 99.8 | 100.0 | 91.8 | 0.082 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|---|
| whisper_dtw | 37.3 (304/816) | 93.0 (585/629) | 85.6 (268/313) | 84.4 (54/64) | 96.8 (609/629) | 98.1 (922/940) |
| own_ctc | 37.3 (304/816) | 93.0 (585/629) | 85.6 (268/313) | 84.4 (54/64) | 96.8 (609/629) | 98.1 (922/940) |

## libri-test-clean

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| whisper_dtw | 2.22 | 1.02 | – | 88.7 | 0.53 | 76.9 | 97.5 | 343 | 22.7 | 43.3 | 83 | 98.5 | 100.0 | 100.0 | 98.5 | 0.055 |
| own_ctc | 2.22 | 1.02 | – | 88.7 | 0.53 | 76.9 | 97.5 | 24 | 99.4 | 99.8 | 50 | 98.8 | 100.0 | 100.0 | 98.8 | 0.073 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|---|
| whisper_dtw | 47.0 (70/149) | 96.4 (108/112) | 83.3 (45/54) | 84.4 (54/64) | 98.2 (110/112) | 98.8 (162/164) |
| own_ctc | 47.0 (70/149) | 96.4 (108/112) | 83.3 (45/54) | 84.4 (54/64) | 98.2 (110/112) | 98.8 (162/164) |

## ted-test

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| whisper_dtw | 2.72 | 1.49 | – | 86.2 | 0.71 | – | – | 214 | 40.5 | 67.0 | 95 | 88.6 | 99.5 | 100.0 | 88.6 | 0.067 |
| own_ctc | 2.72 | 1.49 | – | 86.2 | 0.71 | – | – | 41 | 95.2 | 97.6 | 57 | 90.2 | 99.7 | 100.0 | 90.2 | 0.089 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | homophone | spelling |
|---|---|---|---|---|---|
| whisper_dtw | 35.1 (234/667) | 92.3 (477/517) | 86.1 (223/259) | 96.5 (499/517) | 97.9 (760/776) |
| own_ctc | 35.1 (234/667) | 92.3 (477/517) | 86.1 (223/259) | 96.5 (499/517) | 97.9 (760/776) |
