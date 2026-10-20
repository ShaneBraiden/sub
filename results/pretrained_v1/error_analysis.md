# Error analysis: `ours` on test

## 10 worst files (WER)

| File | Dataset | WER % | Edit F0.5 | Onset |Δt| ms |
|---|---|---|---|---|
| ted-test-JamesCameron | ted-test | 6.00 | 64.2 | 88 |
| libri-test-other-2414-159411 | libri-test-other | 5.63 | 74.9 | 32 |
| libri-test-other-8461-281231 | libri-test-other | 4.68 | 80.4 | 28 |
| ted-test-DanBarber_2010_S103 | ted-test | 4.17 | 71.4 | 25 |
| ted-test-BillGates | ted-test | 3.73 | 72.4 | 61 |
| libri-test-other-4294-35475 | libri-test-other | 3.35 | 84.0 | 33 |
| libri-test-clean-5142-33396 | libri-test-clean | 3.19 | 83.2 | 33 |
| ted-test-DanBarber | ted-test | 2.98 | 84.4 | 78 |
| ted-test-DanielKahneman | ted-test | 2.96 | 80.2 | 66 |
| libri-test-clean-3575-170457 | libri-test-clean | 2.83 | 83.3 | 38 |

## Most frequent residual substitutions (reference → output)

| Reference | Output | Count |
|---|---|---|
| brahman | brahmin | 16 |
| the | a | 8 |
| this | the | 7 |
| in | and | 6 |
| a | the | 6 |
| could | can | 6 |
| and | in | 5 |
| franz | france | 5 |
| de | the | 5 |
| that | the | 4 |
| and | an | 4 |
| 2 | 22 | 4 |
| ok | okay | 4 |
| selves | cells | 4 |
| any | anyone | 3 |

## Most frequent residual deletions / insertions

Deleted (missing in output): the (40), and (31), i (30), a (24), that (20), is (20), in (14), it (12), you (12), to (11), this (11), if (9), as (7), are (7), have (7)

Inserted (extra in output): and (8), it (7), that (7), a (6), 100 (6), the (5), of (3), is (3), 6 (3), 2 (3), are (2), 19 (2), to (2), 3 (2), 5 (2)

## Origin of residual substitutions

* substitution also present in the corrupted subtitle: 442
* substitution introduced (ASR/correction): 107
