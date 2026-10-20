# Corruption rate 5%

## ALL

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 5.29 | 2.61 | – | 0.0 | 0.00 | 35.3 | 92.3 | 5571 | 2.9 | 7.0 | – | 86.7 | 97.2 | 86.1 | 40.6 | 0.000 |
| Ours: fusion + punct + T5 GEC + CTC align + DP seg | 2.51 | 1.56 | – | 68.1 | 1.67 | 74.5 | 97.6 | 54 | 92.4 | 96.7 | 43 | 91.6 | 99.7 | 100.0 | 91.6 | 0.059 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 0.0 (0/429) | 1.1 (3/285) | 0.0 (0/143) | 0.0 (0/28) | 0.0 (0/285) | 0.0 (0/429) |
| Ours: fusion + punct + T5 GEC + CTC align + DP seg | 39.2 (168/429) | 77.5 (221/285) | 91.6 (131/143) | 89.3 (25/28) | 95.4 (272/285) | 97.9 (420/429) |

## libri-test-clean

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 6.01 | 3.07 | – | 0.0 | 0.00 | 35.3 | 92.3 | 3433 | 0.9 | 2.4 | – | 90.6 | 97.6 | 90.8 | 56.3 | 0.000 |
| Ours: fusion + punct + T5 GEC + CTC align + DP seg | 2.31 | 1.30 | – | 75.3 | 1.13 | 74.5 | 97.6 | 39 | 96.4 | 99.6 | 37 | 97.4 | 99.6 | 100.0 | 97.4 | 0.060 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 0.0 (0/76) | 0.0 (0/50) | 0.0 (0/25) | 0.0 (0/28) | 0.0 (0/50) | 0.0 (0/76) |
| Ours: fusion + punct + T5 GEC + CTC align + DP seg | 47.4 (36/76) | 84.0 (42/50) | 76.0 (19/25) | 89.3 (25/28) | 98.0 (49/50) | 98.7 (75/76) |

## ted-test

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 5.14 | 2.51 | – | 0.0 | 0.00 | – | – | 6136 | 3.5 | 8.2 | – | 85.6 | 97.0 | 84.9 | 36.4 | 0.000 |
| Ours: fusion + punct + T5 GEC + CTC align + DP seg | 2.55 | 1.61 | – | 66.5 | 1.78 | – | – | 58 | 91.4 | 96.1 | 44 | 90.2 | 99.8 | 100.0 | 90.2 | 0.059 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | homophone | spelling |
|---|---|---|---|---|---|
| Corrupted input (no-op) | 0.0 (0/353) | 1.3 (3/235) | 0.0 (0/118) | 0.0 (0/235) | 0.0 (0/353) |
| Ours: fusion + punct + T5 GEC + CTC align + DP seg | 37.4 (132/353) | 76.2 (179/235) | 94.9 (112/118) | 94.9 (223/235) | 97.7 (345/353) |
