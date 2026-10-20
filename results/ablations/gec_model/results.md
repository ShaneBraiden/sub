# Ablation: gec_model

## ALL

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 2.76 | 1.61 | – | 81.7 | 1.38 | 74.3 | 97.4 | 41 | 95.3 | 97.5 | 57 | 92.8 | 99.8 | 100.0 | 92.8 | 0.054 |
| own_transformer | 2.80 | 1.63 | – | 81.3 | 1.44 | 73.8 | 97.5 | 42 | 95.3 | 97.4 | 57 | 92.8 | 99.8 | 100.0 | 92.8 | 0.058 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|---|
| none | 38.1 (311/816) | 91.6 (576/629) | 86.3 (270/313) | 84.4 (54/64) | 95.5 (601/629) | 97.2 (914/940) |
| own_transformer | 38.1 (311/816) | 91.6 (576/629) | 86.3 (270/313) | 84.4 (54/64) | 96.2 (605/629) | 97.2 (914/940) |

## libri-test-clean

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 2.15 | 0.99 | – | 88.2 | 0.67 | 74.3 | 97.4 | 25 | 99.0 | 99.6 | 50 | 99.0 | 100.0 | 100.0 | 99.0 | 0.053 |
| own_transformer | 2.16 | 0.99 | – | 87.9 | 0.67 | 73.8 | 97.5 | 25 | 99.0 | 99.6 | 50 | 99.0 | 100.0 | 100.0 | 99.0 | 0.053 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|---|
| none | 49.0 (73/149) | 94.6 (106/112) | 81.5 (44/54) | 84.4 (54/64) | 98.2 (110/112) | 98.8 (162/164) |
| own_transformer | 49.0 (73/149) | 94.6 (106/112) | 81.5 (44/54) | 84.4 (54/64) | 98.2 (110/112) | 98.8 (162/164) |

## ted-test

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 2.89 | 1.74 | – | 80.2 | 1.53 | – | – | 45 | 94.4 | 97.0 | 58 | 91.3 | 99.8 | 100.0 | 91.3 | 0.054 |
| own_transformer | 2.93 | 1.77 | – | 79.7 | 1.60 | – | – | 46 | 94.4 | 96.9 | 58 | 91.4 | 99.8 | 100.0 | 91.4 | 0.061 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | homophone | spelling |
|---|---|---|---|---|---|
| none | 35.7 (238/667) | 90.9 (470/517) | 87.3 (226/259) | 95.0 (491/517) | 96.9 (752/776) |
| own_transformer | 35.7 (238/667) | 90.9 (470/517) | 87.3 (226/259) | 95.7 (495/517) | 96.9 (752/776) |
