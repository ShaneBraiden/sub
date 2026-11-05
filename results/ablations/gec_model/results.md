# Ablation: gec_model

## ALL

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 2.42 | 1.28 | – | 88.7 | 0.42 | 76.9 | 97.4 | 35 | 96.4 | 98.3 | 55 | 91.8 | 99.8 | 100.0 | 91.8 | 0.079 |
| own_transformer | 2.63 | 1.41 | – | 86.7 | 0.68 | 76.9 | 97.5 | 38 | 96.0 | 98.0 | 55 | 91.8 | 99.8 | 100.0 | 91.8 | 0.085 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|---|
| none | 37.1 (303/816) | 93.0 (585/629) | 84.3 (264/313) | 84.4 (54/64) | 96.0 (604/629) | 98.1 (922/940) |
| own_transformer | 37.3 (304/816) | 93.0 (585/629) | 85.6 (268/313) | 84.4 (54/64) | 96.8 (609/629) | 98.1 (922/940) |

## libri-test-clean

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 2.20 | 1.02 | – | 88.9 | 0.51 | 76.9 | 97.4 | 24 | 99.4 | 99.8 | 50 | 98.8 | 100.0 | 100.0 | 98.8 | 0.077 |
| own_transformer | 2.22 | 1.02 | – | 88.7 | 0.53 | 76.9 | 97.5 | 24 | 99.4 | 99.8 | 50 | 98.8 | 100.0 | 100.0 | 98.8 | 0.076 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|---|
| none | 47.0 (70/149) | 96.4 (108/112) | 83.3 (45/54) | 84.4 (54/64) | 98.2 (110/112) | 98.8 (162/164) |
| own_transformer | 47.0 (70/149) | 96.4 (108/112) | 83.3 (45/54) | 84.4 (54/64) | 98.2 (110/112) | 98.8 (162/164) |

## ted-test

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 2.47 | 1.33 | – | 88.6 | 0.40 | – | – | 38 | 95.7 | 98.0 | 56 | 90.2 | 99.7 | 100.0 | 90.2 | 0.081 |
| own_transformer | 2.72 | 1.49 | – | 86.2 | 0.71 | – | – | 41 | 95.2 | 97.6 | 57 | 90.2 | 99.7 | 100.0 | 90.2 | 0.091 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | homophone | spelling |
|---|---|---|---|---|---|
| none | 34.9 (233/667) | 92.3 (477/517) | 84.6 (219/259) | 95.6 (494/517) | 97.9 (760/776) |
| own_transformer | 35.1 (234/667) | 92.3 (477/517) | 86.1 (223/259) | 96.5 (499/517) | 97.9 (760/776) |
