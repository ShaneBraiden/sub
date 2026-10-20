# tune_gec_0.25

## ALL

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tune_gec_0.25 | 3.18 | 1.90 | – | 80.4 | 1.48 | 70.6 | 96.5 | 52 | 93.6 | 97.6 | 42 | 88.6 | 99.5 | 100.0 | 88.6 | 0.065 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|---|
| tune_gec_0.25 | 45.5 (122/268) | 72.7 (157/216) | 86.1 (93/108) | 89.0 (81/91) | 98.1 (212/216) | 96.6 (314/325) |

## libri-dev-clean

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tune_gec_0.25 | 2.75 | 1.38 | – | 85.7 | 0.96 | 70.6 | 96.5 | 39 | 95.0 | 99.8 | 39 | 96.6 | 100.0 | 100.0 | 96.6 | 0.069 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|---|
| tune_gec_0.25 | 53.8 (71/132) | 72.3 (68/94) | 83.0 (39/47) | 89.0 (81/91) | 100.0 (94/94) | 100.0 (142/142) |

## ted-validation

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tune_gec_0.25 | 3.52 | 2.31 | – | 75.5 | 1.88 | – | – | 62 | 92.4 | 95.7 | 45 | 81.7 | 99.0 | 100.0 | 81.7 | 0.057 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | homophone | spelling |
|---|---|---|---|---|---|
| tune_gec_0.25 | 37.5 (51/136) | 73.0 (89/122) | 88.5 (54/61) | 96.7 (118/122) | 94.0 (172/183) |
