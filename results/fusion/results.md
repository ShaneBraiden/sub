# Neural fusion on the dev clips

Written by `experiments/fusion_results.py`. All of this is dev data, used to choose the network; the test numbers are in `results/test/` and `results/ablations/fusion_method/`.

## Variants (three seeds, fusion stage only)

129,063 training candidates in 968,758 aligned words. WER is the estimate after fusion alone (errors of the corrupted subtitle minus the gain of the accepted edits). Perfect decisions on the official corruption would give 1.30.

| Who decides | Params | F0.5 | Precision | Recall | WER % | AUC sub | AUC ins | AUC del |
|---|---|---|---|---|---|---|---|---|
| Linear rule (same selection score) | 9 | 0.860 | 0.864 | 0.845 | 2.73 | – | – | – |
| Linear rule tuned on F0.5 alone | 9 | 0.863 | 0.881 | 0.799 | 3.14 | – | – | – |
| **BiLSTM over the signals, 32 hidden units (shipped)** | 100,019 | 0.936 ± 0.001 | 0.966 ± 0.002 | 0.833 ± 0.005 | 1.99 ± 0.05 | 0.948 ± 0.002 | 0.934 ± 0.002 | 0.942 ± 0.004 |
| BiLSTM, 64 hidden units | 240,307 | 0.936 ± 0.000 | 0.968 ± 0.001 | 0.828 ± 0.002 | 2.04 ± 0.02 | 0.948 ± 0.002 | 0.934 ± 0.003 | 0.940 ± 0.003 |
| BiLSTM, 128 hidden units | 717,491 | 0.936 ± 0.000 | 0.967 ± 0.003 | 0.830 ± 0.008 | 2.02 ± 0.09 | 0.947 ± 0.002 | 0.933 ± 0.008 | 0.948 ± 0.006 |
| BiLSTM + word and character embeddings | 511,667 | 0.930 ± 0.001 | 0.963 ± 0.001 | 0.821 ± 0.002 | 2.15 ± 0.03 | 0.935 ± 0.005 | 0.911 ± 0.010 | 0.947 ± 0.005 |
| MLP per candidate (no context) | 25,267 | 0.932 ± 0.000 | 0.965 ± 0.004 | 0.819 ± 0.012 | 2.16 ± 0.11 | 0.939 ± 0.003 | 0.912 ± 0.004 | 0.929 ± 0.002 |
| Logistic regression | 87 | 0.922 ± 0.001 | 0.950 ± 0.003 | 0.824 ± 0.007 | 2.18 ± 0.06 | 0.903 ± 0.007 | 0.904 ± 0.001 | 0.931 ± 0.001 |
| BiLSTM without the CTC evidence | 100,019 | 0.923 ± 0.002 | 0.962 ± 0.004 | 0.793 ± 0.004 | 2.52 ± 0.02 | 0.946 ± 0.001 | 0.872 ± 0.015 | 0.915 ± 0.007 |

**Lowest WER % at a given precision** (over all thresholds; the rule reaches precision 0.864 at WER 2.73)

| Who decides | ≥ 0.90 | ≥ 0.92 | ≥ 0.94 | ≥ 0.95 | ≥ 0.96 | ≥ 0.97 | ≥ 0.98 |
|---|---|---|---|---|---|---|---|
| **BiLSTM over the signals, 32 hidden units (shipped)** | 1.90 | 1.90 | 1.91 | 1.93 | 1.95 | 2.14 | 2.57 |
| BiLSTM, 64 hidden units | 1.89 | 1.89 | 1.90 | 1.94 | 1.97 | 2.11 | 2.58 |
| BiLSTM, 128 hidden units | 1.89 | 1.89 | 1.91 | 1.94 | 1.97 | 2.12 | 2.66 |
| BiLSTM + word and character embeddings | 1.95 | 1.95 | 2.01 | 2.07 | 2.13 | 2.50 | 3.10 |
| MLP per candidate (no context) | 1.92 | 1.92 | 1.92 | 2.03 | 2.12 | 2.34 | 2.73 |
| Logistic regression | 1.99 | 1.99 | 2.06 | 2.20 | 2.56 | 3.06 | 3.89 |
| BiLSTM without the CTC evidence | 2.20 | 2.20 | 2.31 | 2.37 | 2.51 | 2.82 | – |

## Whole pipeline (first seed of each variant, all 12 dev clips)

| Who decides | WER % | CER % | Edit F0.5 | Over-corr. % | Punct F1 | RTF | entity recall | asr recall | homophone recall | spelling recall | drop recall | dup recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Linear rule (same selection score) | 2.85 | 1.62 | 80.7 | 1.58 | 77.1 | 0.054 | 89 | 46 | 98 | 97 | 94 | 86 |
| Linear rule tuned on F0.5 alone | 3.32 | 1.87 | 80.0 | 1.48 | 77.4 | 0.053 | 79 | 31 | 95 | 97 | 93 | 86 |
| **BiLSTM over the signals, 32 hidden units (shipped)** | 2.46 | 1.31 | 87.0 | 0.81 | 77.2 | 0.069 | 87 | 44 | 97 | 98 | 96 | 83 |
| BiLSTM, 64 hidden units | 2.58 | 1.39 | 86.6 | 0.80 | 76.7 | 0.069 | 87 | 42 | 97 | 98 | 94 | 82 |
| BiLSTM, 128 hidden units | 2.54 | 1.37 | 86.8 | 0.79 | 77.1 | 0.068 | 86 | 42 | 96 | 98 | 94 | 82 |
| BiLSTM + word and character embeddings | 2.55 | 1.36 | 86.1 | 0.78 | 77.2 | 0.069 | 86 | 41 | 98 | 98 | 95 | 83 |
| MLP per candidate (no context) | 2.64 | 1.43 | 86.0 | 0.75 | 77.5 | 0.069 | 85 | 37 | 98 | 98 | 96 | 82 |
| Logistic regression | 2.54 | 1.40 | 86.0 | 0.87 | 77.3 | 0.069 | 84 | 46 | 98 | 98 | 95 | 76 |
| BiLSTM without the CTC evidence | 2.88 | 1.60 | 85.0 | 0.80 | 76.9 | 0.053 | 84 | 41 | 95 | 98 | 93 | 68 |

## What single signals can tell (AUC: is this edit good?)

`train-clean-100` is the audio our CTC model trained on; the others are audio it never saw.

| Candidates | Dataset | n | Good | CTC evidence | Whisper confidence | LM |
|---|---|---|---|---|---|---|
| sub | dev-clean | 12,896 | 90% | 0.656 | 0.839 | 0.827 |
| sub | ted | 3,521 | 90% | 0.638 | 0.823 | 0.821 |
| sub | train-clean-100 | 72,404 | 90% | 0.700 | 0.821 | 0.835 |
| sub | benchmark libri | 1,390 | 89% | 0.800 | 0.739 | 0.892 |
| sub | benchmark ted | 1,146 | 91% | 0.677 | 0.819 | 0.866 |
| ins | dev-clean | 3,493 | 91% | 0.662 | 0.735 | 0.760 |
| ins | ted | 1,182 | 82% | 0.733 | 0.860 | 0.755 |
| ins | train-clean-100 | 19,081 | 89% | 0.701 | 0.764 | 0.824 |
| ins | benchmark libri | 302 | 89% | 0.843 | 0.897 | 0.844 |
| ins | benchmark ted | 404 | 85% | 0.845 | 0.763 | 0.764 |
| del | dev-clean | 2,550 | 66% | 0.728 | 0.773 | 0.705 |
| del | ted | 1,420 | 39% | 0.740 | 0.607 | 0.728 |
| del | train-clean-100 | 12,516 | 69% | 0.668 | 0.762 | 0.674 |
| del | benchmark libri | 171 | 80% | 0.657 | 0.601 | 0.674 |
| del | benchmark ted | 701 | 30% | 0.790 | 0.540 | 0.701 |
| duplicate del | dev-clean | 1,465 | 99% | 0.599 | 0.531 | 0.517 |
| duplicate del | ted | 611 | 80% | 0.959 | 0.710 | 0.737 |
| duplicate del | train-clean-100 | 8,148 | 99% | 0.585 | 0.587 | 0.625 |
| duplicate del | benchmark libri | 137 | 98% | 0.858 | 0.784 | 0.726 |
| duplicate del | benchmark ted | 299 | 57% | 0.969 | 0.542 | 0.734 |

## An error type the network never saw (whole pipeline, dev clips)

Trained on dev-clean + ted only. Each column is the recall of that error type: by the rule, by the network trained on all types, and by the network whose training corruptions left that type out.

| Error type | Rule | Network, all types | Network, type left out |
|---|---|---|---|
| entity | 89 | 82 | 84 |
| asr | 46 | 36 | 37 |
| homophone | 98 | 98 | 98 |
| spelling | 97 | 98 | 98 |
| drop | 94 | 96 | 95 |
| dup | 86 | 83 | 31 |

| Network trained without | WER % | Edit F0.5 | Over-corr. % |
|---|---|---|---|
| rule | 2.85 | 80.7 | 1.58 |
| all_types | 2.68 | 85.5 | 0.78 |
| no_entity | 2.54 | 85.7 | 0.83 |
| no_asr | 2.65 | 86.1 | 0.77 |
| no_homophone | 2.50 | 86.2 | 0.80 |
| no_spelling | 2.68 | 85.7 | 0.78 |
| no_drop | 2.52 | 85.5 | 0.80 |
| no_dup | 3.06 | 84.0 | 0.87 |
