# Ablation: audio_guard

## ALL

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| guard_on | 2.80 | 1.63 | – | 81.3 | 1.44 | 73.8 | 97.5 | 42 | 95.3 | 97.4 | 57 | 92.8 | 99.8 | 100.0 | 92.8 | 0.058 |
| guard_off | 3.36 | 1.92 | – | 76.9 | 1.95 | 73.8 | 97.7 | 42 | 95.2 | 97.3 | 57 | 93.2 | 99.8 | 100.0 | 93.2 | 0.063 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|---|
| guard_on | 38.1 (311/816) | 91.6 (576/629) | 86.3 (270/313) | 84.4 (54/64) | 96.2 (605/629) | 97.2 (914/940) |
| guard_off | 36.2 (295/816) | 91.4 (575/629) | 85.0 (266/313) | 76.6 (49/64) | 95.9 (603/629) | 96.7 (909/940) |

## libri-test-clean

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| guard_on | 2.16 | 0.99 | – | 87.9 | 0.67 | 73.8 | 97.5 | 25 | 99.0 | 99.6 | 50 | 99.0 | 100.0 | 100.0 | 99.0 | 0.054 |
| guard_off | 2.86 | 1.22 | – | 82.6 | 1.16 | 73.8 | 97.7 | 26 | 99.0 | 99.4 | 50 | 98.6 | 100.0 | 100.0 | 98.6 | 0.062 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|---|
| guard_on | 49.0 (73/149) | 94.6 (106/112) | 81.5 (44/54) | 84.4 (54/64) | 98.2 (110/112) | 98.8 (162/164) |
| guard_off | 43.0 (64/149) | 94.6 (106/112) | 79.6 (43/54) | 76.6 (49/64) | 98.2 (110/112) | 98.8 (162/164) |

## ted-test

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| guard_on | 2.93 | 1.77 | – | 79.7 | 1.60 | – | – | 46 | 94.4 | 96.9 | 58 | 91.4 | 99.8 | 100.0 | 91.4 | 0.061 |
| guard_off | 3.46 | 2.06 | – | 75.6 | 2.12 | – | – | 46 | 94.3 | 96.9 | 58 | 92.0 | 99.8 | 100.0 | 92.0 | 0.064 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | homophone | spelling |
|---|---|---|---|---|---|
| guard_on | 35.7 (238/667) | 90.9 (470/517) | 87.3 (226/259) | 95.7 (495/517) | 96.9 (752/776) |
| guard_off | 34.6 (231/667) | 90.7 (469/517) | 86.1 (223/259) | 95.4 (493/517) | 96.3 (747/776) |
