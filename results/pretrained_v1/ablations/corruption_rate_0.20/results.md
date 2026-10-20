# Corruption rate 20%

## ALL

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 17.46 | 8.14 | – | 0.0 | 0.00 | 40.5 | 91.0 | 6336 | 2.1 | 5.6 | – | 86.2 | 95.8 | 77.7 | 37.2 | 0.000 |
| Ours: fusion + punct + T5 GEC + CTC align + DP seg | 3.77 | 2.37 | – | 85.2 | 1.36 | 77.7 | 96.5 | 52 | 92.4 | 96.9 | 42 | 92.2 | 99.9 | 100.0 | 92.2 | 0.060 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 0.1 (1/884) | 0.4 (5/1144) | 0.0 (0/571) | 0.0 (0/117) | 0.1 (1/1144) | 0.1 (1/1713) |
| Ours: fusion + punct + T5 GEC + CTC align + DP seg | 39.0 (345/884) | 69.6 (796/1144) | 79.5 (454/571) | 85.5 (100/117) | 96.2 (1100/1144) | 97.2 (1665/1713) |

## libri-test-clean

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 20.15 | 9.39 | – | 0.0 | 0.00 | 40.5 | 91.0 | 3733 | 1.4 | 3.9 | – | 90.7 | 96.6 | 83.0 | 50.2 | 0.000 |
| Ours: fusion + punct + T5 GEC + CTC align + DP seg | 3.86 | 2.24 | – | 88.9 | 0.42 | 77.7 | 96.5 | 39 | 95.6 | 99.6 | 37 | 97.4 | 100.0 | 100.0 | 97.4 | 0.061 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 0.0 (0/190) | 0.0 (0/201) | 0.0 (0/101) | 0.0 (0/117) | 0.0 (0/201) | 0.3 (1/301) |
| Ours: fusion + punct + T5 GEC + CTC align + DP seg | 48.4 (92/190) | 68.2 (137/201) | 72.3 (73/101) | 85.5 (100/117) | 98.0 (197/201) | 96.7 (291/301) |

## ted-test

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 16.89 | 7.88 | – | 0.0 | 0.00 | – | – | 6987 | 2.2 | 6.1 | – | 85.1 | 95.5 | 76.3 | 34.0 | 0.000 |
| Ours: fusion + punct + T5 GEC + CTC align + DP seg | 3.75 | 2.40 | – | 84.3 | 1.55 | – | – | 55 | 91.7 | 96.3 | 43 | 91.0 | 99.8 | 100.0 | 91.0 | 0.060 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | homophone | spelling |
|---|---|---|---|---|---|
| Corrupted input (no-op) | 0.1 (1/694) | 0.5 (5/943) | 0.0 (0/470) | 0.1 (1/943) | 0.0 (0/1412) |
| Ours: fusion + punct + T5 GEC + CTC align + DP seg | 36.5 (253/694) | 69.9 (659/943) | 81.1 (381/470) | 95.8 (903/943) | 97.3 (1374/1412) |
