# Corruption rate 10%

## ALL

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 9.87 | 4.71 | – | 0.0 | 0.00 | 50.1 | 90.8 | 5700 | 0.4 | 1.3 | – | 87.1 | 97.0 | 84.3 | 41.6 | 0.000 |
| Ours: neural fusion (own KN LM, own CTC) + own punct + own corrector + own CTC align + DP seg | 2.68 | 1.47 | – | 85.3 | 0.72 | 75.3 | 97.0 | 37 | 96.1 | 98.2 | 55 | 92.2 | 99.8 | 100.0 | 92.2 | 0.083 |
| Ours with the linear fusion rule instead of the network | 2.78 | 1.63 | – | 80.1 | 1.42 | 76.6 | 96.9 | 41 | 95.4 | 97.6 | 57 | 92.6 | 99.8 | 100.0 | 92.6 | 0.062 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 0.0 (0/790) | 0.5 (3/571) | 0.0 (0/285) | 0.0 (0/58) | 0.0 (0/571) | 0.0 (0/856) |
| Ours: neural fusion (own KN LM, own CTC) + own punct + own corrector + own CTC align + DP seg | 36.8 (291/790) | 92.1 (526/571) | 82.1 (234/285) | 81.0 (47/58) | 96.8 (553/571) | 97.4 (834/856) |
| Ours with the linear fusion rule instead of the network | 37.5 (296/790) | 91.9 (525/571) | 83.9 (239/285) | 84.5 (49/58) | 97.2 (555/571) | 96.6 (827/856) |

## libri-test-clean

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 10.64 | 5.11 | – | 0.0 | 0.00 | 50.1 | 90.8 | 4269 | 0.2 | 0.7 | – | 92.3 | 98.0 | 87.7 | 55.8 | 0.000 |
| Ours: neural fusion (own KN LM, own CTC) + own punct + own corrector + own CTC align + DP seg | 2.09 | 1.03 | – | 88.4 | 0.44 | 75.3 | 97.0 | 23 | 100.0 | 100.0 | 50 | 99.2 | 100.0 | 100.0 | 99.2 | 0.078 |
| Ours with the linear fusion rule instead of the network | 2.06 | 0.98 | – | 88.1 | 0.64 | 76.6 | 96.9 | 24 | 99.4 | 99.8 | 50 | 99.0 | 100.0 | 100.0 | 99.0 | 0.059 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 0.0 (0/141) | 0.0 (0/101) | 0.0 (0/50) | 0.0 (0/58) | 0.0 (0/101) | 0.0 (0/150) |
| Ours: neural fusion (own KN LM, own CTC) + own punct + own corrector + own CTC align + DP seg | 46.1 (65/141) | 98.0 (99/101) | 72.0 (36/50) | 81.0 (47/58) | 99.0 (100/101) | 98.0 (147/150) |
| Ours with the linear fusion rule instead of the network | 48.9 (69/141) | 95.0 (96/101) | 74.0 (37/50) | 84.5 (49/58) | 99.0 (100/101) | 98.0 (147/150) |

## ted-test

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 9.70 | 4.63 | – | 0.0 | 0.00 | – | – | 6068 | 0.5 | 1.5 | – | 85.7 | 96.7 | 83.4 | 38.0 | 0.000 |
| Ours: neural fusion (own KN LM, own CTC) + own punct + own corrector + own CTC align + DP seg | 2.80 | 1.56 | – | 84.6 | 0.78 | – | – | 40 | 95.2 | 97.7 | 57 | 90.5 | 99.8 | 100.0 | 90.5 | 0.087 |
| Ours with the linear fusion rule instead of the network | 2.93 | 1.76 | – | 78.2 | 1.58 | – | – | 45 | 94.4 | 97.1 | 58 | 91.1 | 99.8 | 100.0 | 91.1 | 0.065 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | homophone | spelling |
|---|---|---|---|---|---|
| Corrupted input (no-op) | 0.0 (0/649) | 0.6 (3/470) | 0.0 (0/235) | 0.0 (0/470) | 0.0 (0/706) |
| Ours: neural fusion (own KN LM, own CTC) + own punct + own corrector + own CTC align + DP seg | 34.8 (226/649) | 90.9 (427/470) | 84.3 (198/235) | 96.4 (453/470) | 97.3 (687/706) |
| Ours with the linear fusion rule instead of the network | 35.0 (227/649) | 91.3 (429/470) | 86.0 (202/235) | 96.8 (455/470) | 96.3 (680/706) |
