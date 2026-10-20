# Corruption rate 10%

## ALL

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 9.87 | 4.71 | – | 0.0 | 0.00 | 50.1 | 90.8 | 5700 | 0.4 | 1.3 | – | 87.1 | 97.0 | 84.3 | 41.6 | 0.000 |
| Ours: fusion + punct + T5 GEC + CTC align + DP seg | 3.00 | 1.83 | – | 78.6 | 1.45 | 78.6 | 96.8 | 53 | 92.8 | 96.7 | 43 | 92.1 | 99.8 | 100.0 | 92.1 | 0.070 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 0.0 (0/790) | 0.5 (3/571) | 0.0 (0/285) | 0.0 (0/58) | 0.0 (0/571) | 0.0 (0/856) |
| Ours: fusion + punct + T5 GEC + CTC align + DP seg | 39.2 (310/790) | 74.1 (423/571) | 84.2 (240/285) | 86.2 (50/58) | 96.1 (549/571) | 96.7 (828/856) |

## libri-test-clean

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 10.64 | 5.11 | – | 0.0 | 0.00 | 50.1 | 90.8 | 4269 | 0.2 | 0.7 | – | 92.3 | 98.0 | 87.7 | 55.8 | 0.000 |
| Ours: fusion + punct + T5 GEC + CTC align + DP seg | 2.31 | 1.18 | – | 86.1 | 0.66 | 78.6 | 96.8 | 34 | 96.6 | 99.8 | 38 | 98.4 | 100.0 | 100.0 | 98.4 | 0.080 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 0.0 (0/141) | 0.0 (0/101) | 0.0 (0/50) | 0.0 (0/58) | 0.0 (0/101) | 0.0 (0/150) |
| Ours: fusion + punct + T5 GEC + CTC align + DP seg | 48.2 (68/141) | 77.2 (78/101) | 74.0 (37/50) | 86.2 (50/58) | 98.0 (99/101) | 98.0 (147/150) |

## ted-test

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 9.70 | 4.63 | – | 0.0 | 0.00 | – | – | 6068 | 0.5 | 1.5 | – | 85.7 | 96.7 | 83.4 | 38.0 | 0.000 |
| Ours: fusion + punct + T5 GEC + CTC align + DP seg | 3.14 | 1.97 | – | 76.9 | 1.61 | – | – | 57 | 91.9 | 96.0 | 44 | 90.7 | 99.8 | 100.0 | 90.7 | 0.062 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | homophone | spelling |
|---|---|---|---|---|---|
| Corrupted input (no-op) | 0.0 (0/649) | 0.6 (3/470) | 0.0 (0/235) | 0.0 (0/470) | 0.0 (0/706) |
| Ours: fusion + punct + T5 GEC + CTC align + DP seg | 37.3 (242/649) | 73.4 (345/470) | 86.4 (203/235) | 95.7 (450/470) | 96.5 (681/706) |
