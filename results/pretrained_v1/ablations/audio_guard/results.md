# Ablation: audio_guard

## ALL

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| guard_on | 3.01 | 1.82 | – | 79.7 | 1.43 | 81.8 | 97.4 | 54 | 92.5 | 96.6 | 43 | 92.3 | 99.8 | 100.0 | 92.3 | 0.062 |
| guard_off | 4.26 | 2.70 | – | 71.4 | 2.26 | 82.4 | 97.6 | 54 | 92.3 | 96.5 | 43 | 91.9 | 99.8 | 100.0 | 91.9 | 0.086 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|---|
| guard_on | 39.0 (318/816) | 73.6 (463/629) | 87.5 (274/313) | 81.2 (52/64) | 95.7 (602/629) | 96.6 (908/940) |
| guard_off | 37.0 (302/816) | 75.2 (473/629) | 85.9 (269/313) | 75.0 (48/64) | 94.9 (597/629) | 96.0 (902/940) |

## libri-test-clean

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| guard_on | 2.52 | 1.29 | – | 85.1 | 0.69 | 81.8 | 97.4 | 36 | 95.8 | 99.6 | 37 | 98.8 | 100.0 | 100.0 | 98.8 | 0.061 |
| guard_off | 3.84 | 1.99 | – | 75.9 | 1.73 | 82.4 | 97.6 | 36 | 96.2 | 99.6 | 37 | 98.8 | 100.0 | 100.0 | 98.8 | 0.085 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|---|
| guard_on | 49.0 (73/149) | 73.2 (82/112) | 81.5 (44/54) | 81.2 (52/64) | 97.3 (109/112) | 98.2 (161/164) |
| guard_off | 42.3 (63/149) | 75.0 (84/112) | 79.6 (43/54) | 75.0 (48/64) | 97.3 (109/112) | 98.2 (161/164) |

## ted-test

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| guard_on | 3.12 | 1.93 | – | 78.5 | 1.58 | – | – | 58 | 91.7 | 95.9 | 44 | 90.8 | 99.7 | 100.0 | 90.8 | 0.063 |
| guard_off | 4.35 | 2.85 | – | 70.4 | 2.37 | – | – | 58 | 91.5 | 95.8 | 45 | 90.3 | 99.7 | 100.0 | 90.3 | 0.087 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | homophone | spelling |
|---|---|---|---|---|---|
| guard_on | 36.7 (245/667) | 73.7 (381/517) | 88.8 (230/259) | 95.4 (493/517) | 96.3 (747/776) |
| guard_off | 35.8 (239/667) | 75.2 (389/517) | 87.3 (226/259) | 94.4 (488/517) | 95.5 (741/776) |
