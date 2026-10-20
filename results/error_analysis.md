# Error analysis: `ours` on test

## 10 worst files (WER)

| File | Dataset | WER % | Edit F0.5 | Onset |Δt| ms |
|---|---|---|---|---|
| ted-test-JamesCameron | ted-test | 5.70 | 66.9 | 58 |
| libri-test-other-2414-159411 | libri-test-other | 4.93 | 78.9 | 35 |
| libri-test-other-8461-281231 | libri-test-other | 4.29 | 81.5 | 27 |
| ted-test-DanBarber_2010_S103 | ted-test | 4.17 | 71.4 | 15 |
| ted-test-BillGates | ted-test | 3.64 | 72.1 | 57 |
| libri-test-other-4294-35475 | libri-test-other | 3.03 | 84.0 | 34 |
| ted-test-GaryFlake | ted-test | 2.81 | 86.5 | 25 |
| libri-test-clean-5142-33396 | libri-test-clean | 2.77 | 85.7 | 26 |
| ted-test-DanBarber | ted-test | 2.74 | 84.3 | 66 |
| ted-test-DanielKahneman | ted-test | 2.69 | 82.1 | 44 |

## Most frequent residual substitutions (reference → output)

| Reference | Output | Count |
|---|---|---|
| brahman | brahmin | 15 |
| this | the | 8 |
| the | a | 8 |
| in | and | 7 |
| a | the | 7 |
| de | the | 6 |
| that | the | 5 |
| and | in | 5 |
| franz | france | 5 |
| could | can | 5 |
| 2 | 22 | 4 |
| ok | okay | 4 |
| any | anyone | 3 |
| bracy | tracy | 3 |
| it | that | 3 |

## Most frequent residual deletions / insertions

Deleted (missing in output): the (31), i (27), and (26), a (22), is (18), that (13), in (12), have (12), it (10), this (7), you (7), to (6), are (6), of (6), one (5)

Inserted (extra in output): is (9), and (7), that (7), a (6), it (6), 100 (6), the (5), i (4), 2 (4), 19 (3), 6 (3), 5 (3), he (2), of (2), to (2)

## Origin of residual substitutions

* substitution also present in the corrupted subtitle: 450
* substitution introduced (ASR/correction): 107
