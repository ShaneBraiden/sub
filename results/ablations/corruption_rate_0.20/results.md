# Corruption rate 20%

## ALL

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 17.46 | 8.14 | – | 0.0 | 0.00 | 40.5 | 91.0 | 6336 | 2.1 | 5.6 | – | 86.2 | 95.8 | 77.7 | 37.2 | 0.000 |
| Ours: neural fusion (own KN LM, own CTC) + own punct + own corrector + own CTC align + DP seg | 2.81 | 1.50 | – | 91.1 | 0.67 | 75.7 | 96.5 | 38 | 95.9 | 97.9 | 56 | 92.3 | 99.8 | 100.0 | 92.3 | 0.078 |
| Ours with the linear fusion rule instead of the network | 3.06 | 1.81 | – | 87.0 | 1.40 | 72.3 | 96.4 | 38 | 95.6 | 97.8 | 56 | 92.6 | 99.8 | 100.0 | 92.6 | 0.061 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 0.1 (1/884) | 0.4 (5/1144) | 0.0 (0/571) | 0.0 (0/117) | 0.1 (1/1144) | 0.1 (1/1713) |
| Ours: neural fusion (own KN LM, own CTC) + own punct + own corrector + own CTC align + DP seg | 40.5 (358/884) | 94.8 (1084/1144) | 77.2 (441/571) | 88.9 (104/117) | 96.9 (1108/1144) | 97.7 (1674/1713) |
| Ours with the linear fusion rule instead of the network | 38.3 (339/884) | 92.4 (1057/1144) | 78.6 (449/571) | 88.0 (103/117) | 96.7 (1106/1144) | 97.5 (1671/1713) |

## libri-test-clean

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 20.15 | 9.39 | – | 0.0 | 0.00 | 40.5 | 91.0 | 3733 | 1.4 | 3.9 | – | 90.7 | 96.6 | 83.0 | 50.2 | 0.000 |
| Ours: neural fusion (own KN LM, own CTC) + own punct + own corrector + own CTC align + DP seg | 2.52 | 1.34 | – | 93.0 | 0.27 | 75.7 | 96.5 | 30 | 99.4 | 99.8 | 52 | 98.2 | 100.0 | 100.0 | 98.2 | 0.076 |
| Ours with the linear fusion rule instead of the network | 2.68 | 1.45 | – | 91.6 | 0.38 | 72.3 | 96.4 | 24 | 99.4 | 100.0 | 50 | 98.0 | 100.0 | 100.0 | 98.0 | 0.059 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 0.0 (0/190) | 0.0 (0/201) | 0.0 (0/101) | 0.0 (0/117) | 0.0 (0/201) | 0.3 (1/301) |
| Ours: neural fusion (own KN LM, own CTC) + own punct + own corrector + own CTC align + DP seg | 50.5 (96/190) | 98.0 (197/201) | 72.3 (73/101) | 88.9 (104/117) | 99.0 (199/201) | 97.7 (294/301) |
| Ours with the linear fusion rule instead of the network | 47.9 (91/190) | 95.5 (192/201) | 73.3 (74/101) | 88.0 (103/117) | 99.5 (200/201) | 97.7 (294/301) |

## ted-test

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Corrupted input (no-op) | 16.89 | 7.88 | – | 0.0 | 0.00 | – | – | 6987 | 2.2 | 6.1 | – | 85.1 | 95.5 | 76.3 | 34.0 | 0.000 |
| Ours: neural fusion (own KN LM, own CTC) + own punct + own corrector + own CTC align + DP seg | 2.87 | 1.54 | – | 90.6 | 0.76 | – | – | 40 | 95.0 | 97.5 | 56 | 90.9 | 99.8 | 100.0 | 90.9 | 0.080 |
| Ours with the linear fusion rule instead of the network | 3.13 | 1.88 | – | 85.9 | 1.61 | – | – | 41 | 94.7 | 97.3 | 57 | 91.4 | 99.8 | 100.0 | 91.4 | 0.062 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | homophone | spelling |
|---|---|---|---|---|---|
| Corrupted input (no-op) | 0.1 (1/694) | 0.5 (5/943) | 0.0 (0/470) | 0.1 (1/943) | 0.0 (0/1412) |
| Ours: neural fusion (own KN LM, own CTC) + own punct + own corrector + own CTC align + DP seg | 37.8 (262/694) | 94.1 (887/943) | 78.3 (368/470) | 96.4 (909/943) | 97.7 (1380/1412) |
| Ours with the linear fusion rule instead of the network | 35.7 (248/694) | 91.7 (865/943) | 79.8 (375/470) | 96.1 (906/943) | 97.5 (1377/1412) |
