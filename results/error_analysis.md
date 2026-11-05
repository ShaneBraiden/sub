# Error analysis: `ours` on test

## 10 worst files (WER)

| File | Dataset | WER % | Edit F0.5 | Onset |Δt| ms |
|---|---|---|---|---|
| ted-test-JamesCameron | ted-test | 4.44 | 76.5 | 53 |
| ted-test-DanBarber_2010_S103 | ted-test | 4.17 | 71.4 | 15 |
| libri-test-other-8461-281231 | libri-test-other | 4.16 | 83.5 | 25 |
| ted-test-DanBarber | ted-test | 3.19 | 84.4 | 55 |
| libri-test-clean-6829-68769 | libri-test-clean | 3.17 | 84.5 | 27 |
| ted-test-BillGates | ted-test | 3.08 | 82.8 | 49 |
| libri-test-other-2414-159411 | libri-test-other | 2.99 | 86.7 | 35 |
| libri-test-clean-5142-33396 | libri-test-clean | 2.77 | 86.0 | 25 |
| ted-test-JaneMcGonigal | ted-test | 2.69 | 88.0 | 28 |
| ted-test-GaryFlake | ted-test | 2.55 | 89.7 | 24 |

## Most frequent residual substitutions (reference → output)

| Reference | Output | Count |
|---|---|---|
| 2 | 22 | 14 |
| the | a | 9 |
| brahman | brahmin | 8 |
| a | the | 7 |
| this | the | 7 |
| de | the | 6 |
| in | and | 5 |
| franz | france | 5 |
| that | the | 4 |
| and | in | 4 |
| is | was | 4 |
| bracy | tracy | 4 |
| could | can | 4 |
| it | that | 4 |
| 10000 | 0 | 4 |

## Most frequent residual deletions / insertions

Deleted (missing in output): i (19), the (15), is (12), a (10), are (9), you (9), and (8), have (8), of (7), in (6), it (5), to (5), this (5), that (5), not (4)

Inserted (extra in output): is (16), it (9), and (8), that (8), 100 (7), the (6), i (6), to (6), 6 (6), a (5), are (5), 19 (5), 5 (5), of (4), 3 (3)

## Origin of residual substitutions

* substitution also present in the corrupted subtitle: 468
* substitution introduced (ASR/correction): 102
