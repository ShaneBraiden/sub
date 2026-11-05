# Results: dev

## ALL

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Mode B: Ours (own punct + own corrector + own CTC align + DP seg) | 3.17 | 1.67 | – | – | – | 76.4 | 96.5 | 37 | 95.3 | 97.2 | 55 | 89.4 | 99.6 | 100.0 | 89.2 | 0.052 |
| Ours: neural fusion (own KN LM, own CTC) + own punct + own corrector + own CTC align + DP seg | 2.46 | 1.31 | 76.9 | 87.0 | 0.81 | 77.2 | 97.6 | 32 | 96.5 | 98.4 | 54 | 87.5 | 99.0 | 100.0 | 87.4 | 0.074 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|---|
| Mode B: Ours (own punct + own corrector + own CTC align + DP seg) | – | – | – | – | – | – |
| Ours: neural fusion (own KN LM, own CTC) + own punct + own corrector + own CTC align + DP seg | 44.0 (118/268) | 95.8 (207/216) | 83.3 (90/108) | 86.8 (79/91) | 96.8 (209/216) | 98.5 (320/325) |

## libri-dev-clean

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Mode B: Ours (own punct + own corrector + own CTC align + DP seg) | 2.40 | 0.74 | – | – | – | 76.4 | 96.5 | 24 | 97.5 | 100.0 | 51 | 97.3 | 100.0 | 100.0 | 97.3 | 0.053 |
| Ours: neural fusion (own KN LM, own CTC) + own punct + own corrector + own CTC align + DP seg | 2.06 | 0.96 | 72.6 | 89.4 | 0.67 | 77.2 | 97.6 | 24 | 97.9 | 100.0 | 51 | 97.3 | 100.0 | 100.0 | 97.0 | 0.070 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | entity | homophone | spelling |
|---|---|---|---|---|---|---|
| Mode B: Ours (own punct + own corrector + own CTC align + DP seg) | – | – | – | – | – | – |
| Ours: neural fusion (own KN LM, own CTC) + own punct + own corrector + own CTC align + DP seg | 56.1 (74/132) | 97.9 (92/94) | 83.0 (39/47) | 86.8 (79/91) | 98.9 (93/94) | 100.0 (142/142) |

## ted-validation

| System | WER | CER | ERRANT F0.5 | Edit F0.5 | Over-corr % | Punct F1 | Case acc | Onset |Δt| ms | % onset ≤100ms | % onset ≤250ms | Word |Δt| ms | CPS≤17 % | CPS≤20 % | Lines ok % | All fmt ok % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Mode B: Ours (own punct + own corrector + own CTC align + DP seg) | 3.76 | 2.40 | – | – | – | – | – | 49 | 93.4 | 94.7 | 58 | 82.5 | 99.2 | 100.0 | 82.1 | 0.051 |
| Ours: neural fusion (own KN LM, own CTC) + own punct + own corrector + own CTC align + DP seg | 2.76 | 1.58 | 85.1 | 84.8 | 0.92 | – | – | 39 | 95.3 | 97.0 | 57 | 79.1 | 98.2 | 100.0 | 79.1 | 0.081 |

Per-error-type recall (% repaired, fixed/total):

| System | asr | drop | dup | homophone | spelling |
|---|---|---|---|---|---|
| Mode B: Ours (own punct + own corrector + own CTC align + DP seg) | – | – | – | – | – |
| Ours: neural fusion (own KN LM, own CTC) + own punct + own corrector + own CTC align + DP seg | 32.4 (44/136) | 94.3 (115/122) | 83.6 (51/61) | 95.1 (116/122) | 97.3 (178/183) |
