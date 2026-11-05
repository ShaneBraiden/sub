# Corruption rate 5%

## ALL

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 5.29 | 2.61 | – | 0.0 | 0.00 | 35.3 | 92.3 | 5571 | 2.9 | 7.0 | – | 86.7 | 97.2 | 86.1 | 40.6 | 0.000 |
| Ours: neural fusion (own KN LM, own CTC) + own punct + own corrector + own CTC align + DP seg | 2.11 | 1.17 | – | 77.9 | 0.86 | 77.9 | 97.3 | 37 | 95.9 | 98.1 | 56 | 91.8 | 99.8 | 100.0 | 91.8 | 0.085 |
| Ours with the linear fusion rule instead of the network | 2.39 | 1.43 | – | 69.9 | 1.63 | 76.5 | 97.3 | 40 | 95.7 | 97.8 | 57 | 92.3 | 99.8 | 100.0 | 92.3 | 0.062 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 0.0 (0/429) | 1.1 (3/285) | 0.0 (0/143) | 0.0 (0/28) | 0.0 (0/285) | 0.0 (0/429) |
| Ours: neural fusion (own KN LM, own CTC) + own punct + own corrector + own CTC align + DP seg | 36.8 (158/429) | 91.9 (262/285) | 90.9 (130/143) | 78.6 (22/28) | 96.1 (274/285) | 98.4 (422/429) |
| Ours with the linear fusion rule instead of the network | 39.6 (170/429) | 91.6 (261/285) | 91.6 (131/143) | 85.7 (24/28) | 96.5 (275/285) | 97.9 (420/429) |

## libri-test-clean

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 6.01 | 3.07 | – | 0.0 | 0.00 | 35.3 | 92.3 | 3433 | 0.9 | 2.4 | – | 90.6 | 97.6 | 90.8 | 56.3 | 0.000 |
| Ours: neural fusion (own KN LM, own CTC) + own punct + own corrector + own CTC align + DP seg | 2.06 | 1.12 | – | 78.4 | 0.86 | 77.9 | 97.3 | 24 | 99.8 | 99.8 | 50 | 98.4 | 100.0 | 100.0 | 98.4 | 0.081 |
| Ours with the linear fusion rule instead of the network | 1.95 | 0.96 | – | 78.8 | 0.99 | 76.5 | 97.3 | 23 | 99.8 | 100.0 | 50 | 97.6 | 100.0 | 100.0 | 97.6 | 0.058 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 0.0 (0/76) | 0.0 (0/50) | 0.0 (0/25) | 0.0 (0/28) | 0.0 (0/50) | 0.0 (0/76) |
| Ours: neural fusion (own KN LM, own CTC) + own punct + own corrector + own CTC align + DP seg | 47.4 (36/76) | 96.0 (48/50) | 76.0 (19/25) | 78.6 (22/28) | 98.0 (49/50) | 98.7 (75/76) |
| Ours with the linear fusion rule instead of the network | 51.3 (39/76) | 90.0 (45/50) | 76.0 (19/25) | 85.7 (24/28) | 100.0 (50/50) | 98.7 (75/76) |

## ted-test

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 5.14 | 2.51 | – | 0.0 | 0.00 | – | – | 6136 | 3.5 | 8.2 | – | 85.6 | 97.0 | 84.9 | 36.4 | 0.000 |
| Ours: neural fusion (own KN LM, own CTC) + own punct + own corrector + own CTC align + DP seg | 2.13 | 1.18 | – | 77.8 | 0.87 | – | – | 40 | 95.0 | 97.7 | 57 | 90.3 | 99.7 | 100.0 | 90.3 | 0.087 |
| Ours with the linear fusion rule instead of the network | 2.49 | 1.53 | – | 67.9 | 1.77 | – | – | 44 | 94.8 | 97.3 | 59 | 91.1 | 99.7 | 100.0 | 91.1 | 0.066 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | homophone | spelling |
|---|---|---|---|---|---|
| Corrupted input (no-op) | 0.0 (0/353) | 1.3 (3/235) | 0.0 (0/118) | 0.0 (0/235) | 0.0 (0/353) |
| Ours: neural fusion (own KN LM, own CTC) + own punct + own corrector + own CTC align + DP seg | 34.6 (122/353) | 91.1 (214/235) | 94.1 (111/118) | 95.7 (225/235) | 98.3 (347/353) |
| Ours with the linear fusion rule instead of the network | 37.1 (131/353) | 91.9 (216/235) | 94.9 (112/118) | 95.7 (225/235) | 97.7 (345/353) |
