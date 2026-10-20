# tune_gec_0.15

## ALL

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tune_gec_0.15 | 3.17 | 1.87 | – | 80.7 | 1.45 | 70.6 | 96.5 | 51 | 93.6 | 97.7 | 42 | 88.7 | 99.4 | 100.0 | 88.7 | 0.066 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|---|
| tune_gec_0.15 | 45.5 (122/268) | 72.7 (157/216) | 86.1 (93/108) | 89.0 (81/91) | 98.1 (212/216) | 96.6 (314/325) |

## libri-dev-clean

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tune_gec_0.15 | 2.71 | 1.34 | – | 86.0 | 0.93 | 70.6 | 96.5 | 39 | 95.0 | 99.8 | 39 | 96.6 | 100.0 | 100.0 | 96.6 | 0.070 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|---|
| tune_gec_0.15 | 53.8 (71/132) | 72.3 (68/94) | 83.0 (39/47) | 89.0 (81/91) | 100.0 (94/94) | 100.0 (142/142) |

## ted-validation

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tune_gec_0.15 | 3.52 | 2.29 | – | 75.8 | 1.85 | – | – | 61 | 92.4 | 95.9 | 45 | 81.9 | 98.8 | 100.0 | 81.9 | 0.058 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | homophone | spelling |
|---|---|---|---|---|---|
| tune_gec_0.15 | 37.5 (51/136) | 73.0 (89/122) | 88.5 (54/61) | 96.7 (118/122) | 94.0 (172/183) |
