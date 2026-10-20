# tune_punct_A_fill

## ALL

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tune_punct_A_fill | 3.33 | 2.03 | – | 79.9 | 1.50 | 77.0 | 97.6 | 221 | 38.9 | 64.5 | 82 | 87.4 | 99.1 | 100.0 | 87.4 | 0.037 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|---|
| tune_punct_A_fill | 45.9 (123/268) | 61.1 (132/216) | 86.1 (93/108) | 89.0 (81/91) | 97.2 (210/216) | 97.2 (316/325) |

## libri-dev-clean

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tune_punct_A_fill | 2.65 | 1.35 | – | 86.2 | 0.88 | 77.0 | 97.6 | 269 | 28.0 | 51.0 | 79 | 97.4 | 100.0 | 100.0 | 97.4 | 0.039 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|---|
| tune_punct_A_fill | 56.1 (74/132) | 67.0 (63/94) | 83.0 (39/47) | 89.0 (81/91) | 100.0 (94/94) | 100.0 (142/142) |

## ted-validation

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tune_punct_A_fill | 3.86 | 2.57 | – | 73.9 | 1.98 | – | – | 182 | 47.9 | 75.7 | 84 | 79.2 | 98.4 | 100.0 | 79.2 | 0.035 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | homophone | spelling |
|---|---|---|---|---|---|
| tune_punct_A_fill | 36.0 (49/136) | 56.6 (69/122) | 88.5 (54/61) | 95.1 (116/122) | 95.1 (174/183) |
