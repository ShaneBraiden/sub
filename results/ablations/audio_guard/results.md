# Ablation: audio_guard

## ALL

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| guard_on | 2.63 | 1.41 | – | 86.7 | 0.68 | 76.9 | 97.5 | 38 | 96.0 | 98.0 | 55 | 91.8 | 99.8 | 100.0 | 91.8 | 0.085 |
| guard_off | 3.28 | 1.78 | – | 81.2 | 1.34 | 77.1 | 97.7 | 39 | 95.7 | 97.9 | 56 | 92.3 | 99.8 | 100.0 | 92.3 | 0.089 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|---|
| guard_on | 37.3 (304/816) | 93.0 (585/629) | 85.6 (268/313) | 84.4 (54/64) | 96.8 (609/629) | 98.1 (922/940) |
| guard_off | 38.1 (311/816) | 92.8 (584/629) | 83.7 (262/313) | 76.6 (49/64) | 96.3 (606/629) | 97.4 (916/940) |

## libri-test-clean

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| guard_on | 2.22 | 1.02 | – | 88.7 | 0.53 | 76.9 | 97.5 | 24 | 99.4 | 99.8 | 50 | 98.8 | 100.0 | 100.0 | 98.8 | 0.077 |
| guard_off | 2.91 | 1.23 | – | 83.6 | 0.95 | 77.1 | 97.7 | 24 | 99.4 | 99.6 | 50 | 98.6 | 100.0 | 100.0 | 98.6 | 0.087 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|---|
| guard_on | 47.0 (70/149) | 96.4 (108/112) | 83.3 (45/54) | 84.4 (54/64) | 98.2 (110/112) | 98.8 (162/164) |
| guard_off | 42.3 (63/149) | 96.4 (108/112) | 81.5 (44/54) | 76.6 (49/64) | 98.2 (110/112) | 98.8 (162/164) |

## ted-test

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| guard_on | 2.72 | 1.49 | – | 86.2 | 0.71 | – | – | 41 | 95.2 | 97.6 | 57 | 90.2 | 99.7 | 100.0 | 90.2 | 0.090 |
| guard_off | 3.35 | 1.89 | – | 80.6 | 1.42 | – | – | 42 | 94.9 | 97.5 | 57 | 90.8 | 99.7 | 100.0 | 90.8 | 0.091 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | homophone | spelling |
|---|---|---|---|---|---|
| guard_on | 35.1 (234/667) | 92.3 (477/517) | 86.1 (223/259) | 96.5 (499/517) | 97.9 (760/776) |
| guard_off | 37.2 (248/667) | 92.1 (476/517) | 84.2 (218/259) | 95.9 (496/517) | 97.2 (754/776) |
