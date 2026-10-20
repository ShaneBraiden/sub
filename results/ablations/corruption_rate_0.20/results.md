# Corruption rate 20%

## ALL

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 17.46 | 8.14 | – | 0.0 | 0.00 | 40.5 | 91.0 | 6336 | 2.1 | 5.6 | – | 86.2 | 95.8 | 77.7 | 37.2 | 0.000 |
| Ours: fusion (own KN LM) + own punct + own corrector + own CTC align + DP seg | 3.14 | 1.87 | – | 86.8 | 1.42 | 72.4 | 96.4 | 38 | 95.6 | 97.9 | 56 | 92.9 | 99.8 | 100.0 | 92.9 | 0.063 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 0.1 (1/884) | 0.4 (5/1144) | 0.0 (0/571) | 0.0 (0/117) | 0.1 (1/1144) | 0.1 (1/1713) |
| Ours: fusion (own KN LM) + own punct + own corrector + own CTC align + DP seg | 38.3 (339/884) | 89.8 (1027/1144) | 78.5 (448/571) | 88.0 (103/117) | 96.7 (1106/1144) | 97.5 (1671/1713) |

## libri-test-clean

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 20.15 | 9.39 | – | 0.0 | 0.00 | 40.5 | 91.0 | 3733 | 1.4 | 3.9 | – | 90.7 | 96.6 | 83.0 | 50.2 | 0.000 |
| Ours: fusion (own KN LM) + own punct + own corrector + own CTC align + DP seg | 2.74 | 1.48 | – | 91.4 | 0.40 | 72.4 | 96.4 | 24 | 99.4 | 100.0 | 50 | 98.0 | 100.0 | 100.0 | 98.0 | 0.057 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 0.0 (0/190) | 0.0 (0/201) | 0.0 (0/101) | 0.0 (0/117) | 0.0 (0/201) | 0.3 (1/301) |
| Ours: fusion (own KN LM) + own punct + own corrector + own CTC align + DP seg | 47.9 (91/190) | 93.0 (187/201) | 73.3 (74/101) | 88.0 (103/117) | 99.5 (200/201) | 97.7 (294/301) |

## ted-test

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 16.89 | 7.88 | – | 0.0 | 0.00 | – | – | 6987 | 2.2 | 6.1 | – | 85.1 | 95.5 | 76.3 | 34.0 | 0.000 |
| Ours: fusion (own KN LM) + own punct + own corrector + own CTC align + DP seg | 3.23 | 1.95 | – | 85.6 | 1.63 | – | – | 41 | 94.8 | 97.4 | 57 | 91.7 | 99.8 | 100.0 | 91.7 | 0.067 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | homophone | spelling |
|---|---|---|---|---|---|
| Corrupted input (no-op) | 0.1 (1/694) | 0.5 (5/943) | 0.0 (0/470) | 0.1 (1/943) | 0.0 (0/1412) |
| Ours: fusion (own KN LM) + own punct + own corrector + own CTC align + DP seg | 35.7 (248/694) | 89.1 (840/943) | 79.6 (374/470) | 96.1 (906/943) | 97.5 (1377/1412) |
