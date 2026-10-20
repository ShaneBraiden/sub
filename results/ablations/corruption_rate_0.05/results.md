# Corruption rate 5%

## ALL

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 5.29 | 2.61 | – | 0.0 | 0.00 | 35.3 | 92.3 | 5571 | 2.9 | 7.0 | – | 86.7 | 97.2 | 86.1 | 40.6 | 0.000 |
| Ours: fusion (own KN LM) + own punct + own corrector + own CTC align + DP seg | 2.41 | 1.44 | – | 69.8 | 1.65 | 76.6 | 97.3 | 40 | 95.8 | 97.8 | 57 | 92.3 | 99.8 | 100.0 | 92.3 | 0.061 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 0.0 (0/429) | 1.1 (3/285) | 0.0 (0/143) | 0.0 (0/28) | 0.0 (0/285) | 0.0 (0/429) |
| Ours: fusion (own KN LM) + own punct + own corrector + own CTC align + DP seg | 39.6 (170/429) | 90.2 (257/285) | 91.6 (131/143) | 85.7 (24/28) | 96.5 (275/285) | 97.9 (420/429) |

## libri-test-clean

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 6.01 | 3.07 | – | 0.0 | 0.00 | 35.3 | 92.3 | 3433 | 0.9 | 2.4 | – | 90.6 | 97.6 | 90.8 | 56.3 | 0.000 |
| Ours: fusion (own KN LM) + own punct + own corrector + own CTC align + DP seg | 1.97 | 0.98 | – | 78.7 | 0.99 | 76.6 | 97.3 | 23 | 99.8 | 100.0 | 50 | 97.6 | 100.0 | 100.0 | 97.6 | 0.063 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 0.0 (0/76) | 0.0 (0/50) | 0.0 (0/25) | 0.0 (0/28) | 0.0 (0/50) | 0.0 (0/76) |
| Ours: fusion (own KN LM) + own punct + own corrector + own CTC align + DP seg | 51.3 (39/76) | 88.0 (44/50) | 76.0 (19/25) | 85.7 (24/28) | 100.0 (50/50) | 98.7 (75/76) |

## ted-test

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 5.14 | 2.51 | – | 0.0 | 0.00 | – | – | 6136 | 3.5 | 8.2 | – | 85.6 | 97.0 | 84.9 | 36.4 | 0.000 |
| Ours: fusion (own KN LM) + own punct + own corrector + own CTC align + DP seg | 2.51 | 1.54 | – | 67.8 | 1.79 | – | – | 44 | 94.8 | 97.3 | 59 | 91.1 | 99.7 | 100.0 | 91.1 | 0.060 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | homophone | spelling |
|---|---|---|---|---|---|
| Corrupted input (no-op) | 0.0 (0/353) | 1.3 (3/235) | 0.0 (0/118) | 0.0 (0/235) | 0.0 (0/353) |
| Ours: fusion (own KN LM) + own punct + own corrector + own CTC align + DP seg | 37.1 (131/353) | 90.6 (213/235) | 94.9 (112/118) | 95.7 (225/235) | 97.7 (345/353) |
