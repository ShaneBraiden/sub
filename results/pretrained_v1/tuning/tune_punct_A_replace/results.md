# tune_punct_A_replace

## ALL

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tune_punct_A_replace | 3.05 | 1.80 | – | 81.7 | 1.36 | 65.4 | 96.7 | 51 | 93.4 | 97.7 | 42 | 89.0 | 99.4 | 100.0 | 89.0 | 0.049 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|---|
| tune_punct_A_replace | 45.1 (121/268) | 72.7 (157/216) | 86.1 (93/108) | 89.0 (81/91) | 98.1 (212/216) | 96.3 (313/325) |

## libri-dev-clean

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tune_punct_A_replace | 2.56 | 1.26 | – | 87.0 | 0.88 | 65.4 | 96.7 | 39 | 94.6 | 99.8 | 39 | 97.3 | 100.0 | 100.0 | 97.3 | 0.052 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|---|
| tune_punct_A_replace | 53.8 (71/132) | 72.3 (68/94) | 83.0 (39/47) | 89.0 (81/91) | 100.0 (94/94) | 100.0 (142/142) |

## ted-validation

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tune_punct_A_replace | 3.42 | 2.23 | – | 76.8 | 1.72 | – | – | 61 | 92.4 | 95.9 | 45 | 81.9 | 98.8 | 100.0 | 81.9 | 0.044 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | homophone | spelling |
|---|---|---|---|---|---|
| tune_punct_A_replace | 36.8 (50/136) | 73.0 (89/122) | 88.5 (54/61) | 96.7 (118/122) | 93.4 (171/183) |
