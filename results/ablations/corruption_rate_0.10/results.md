# Corruption rate 10%

## ALL

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 9.87 | 4.71 | – | 0.0 | 0.00 | 50.1 | 90.8 | 5700 | 0.4 | 1.3 | – | 87.1 | 97.0 | 84.3 | 41.6 | 0.000 |
| Ours: fusion (own KN LM) + own punct + own corrector + own CTC align + DP seg | 2.80 | 1.64 | – | 79.9 | 1.43 | 76.5 | 96.9 | 41 | 95.4 | 97.6 | 57 | 92.7 | 99.8 | 100.0 | 92.7 | 0.058 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 0.0 (0/790) | 0.5 (3/571) | 0.0 (0/285) | 0.0 (0/58) | 0.0 (0/571) | 0.0 (0/856) |
| Ours: fusion (own KN LM) + own punct + own corrector + own CTC align + DP seg | 37.5 (296/790) | 90.0 (514/571) | 84.2 (240/285) | 84.5 (49/58) | 97.2 (555/571) | 96.5 (826/856) |

## libri-test-clean

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 10.64 | 5.11 | – | 0.0 | 0.00 | 50.1 | 90.8 | 4269 | 0.2 | 0.7 | – | 92.3 | 98.0 | 87.7 | 55.8 | 0.000 |
| Ours: fusion (own KN LM) + own punct + own corrector + own CTC align + DP seg | 2.04 | 0.97 | – | 87.9 | 0.66 | 76.5 | 96.9 | 24 | 99.4 | 99.8 | 50 | 99.0 | 100.0 | 100.0 | 99.0 | 0.055 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 0.0 (0/141) | 0.0 (0/101) | 0.0 (0/50) | 0.0 (0/58) | 0.0 (0/101) | 0.0 (0/150) |
| Ours: fusion (own KN LM) + own punct + own corrector + own CTC align + DP seg | 48.9 (69/141) | 95.0 (96/101) | 74.0 (37/50) | 84.5 (49/58) | 99.0 (100/101) | 98.0 (147/150) |

## ted-test

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 9.70 | 4.63 | – | 0.0 | 0.00 | – | – | 6068 | 0.5 | 1.5 | – | 85.7 | 96.7 | 83.4 | 38.0 | 0.000 |
| Ours: fusion (own KN LM) + own punct + own corrector + own CTC align + DP seg | 2.96 | 1.78 | – | 78.1 | 1.59 | – | – | 44 | 94.5 | 97.1 | 58 | 91.3 | 99.8 | 100.0 | 91.3 | 0.061 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | homophone | spelling |
|---|---|---|---|---|---|
| Corrupted input (no-op) | 0.0 (0/649) | 0.6 (3/470) | 0.0 (0/235) | 0.0 (0/470) | 0.0 (0/706) |
| Ours: fusion (own KN LM) + own punct + own corrector + own CTC align + DP seg | 35.0 (227/649) | 88.9 (418/470) | 86.4 (203/235) | 96.8 (455/470) | 96.2 (679/706) |
