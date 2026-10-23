# Speaker diarization on synthetic test-clean conversations

60 conversations (2.58 h), 2-5 unseen speakers each, no overlapping speech; DER with a 0.25 s collar. Thresholds tuned on dev-clean conversations.

| System | DER % | Miss % | FA % | Confusion % | Speaker count exact | Count error |
|---|---:|---:|---:|---:|---:|---:|
| MFCC statistics (untrained) | 18.44 | 1.35 | 0.0 | 17.09 | 40% | 0.75 |
| Own speaker model | 8.76 | 1.35 | 0.0 | 7.41 | 52% | 0.57 |
| Own speaker model, oracle K | 4.66 | 1.35 | 0.0 | 3.31 | 100% | 0.0 |

DER % by number of speakers:

| System | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|
| MFCC statistics (untrained) | 18.45 | 13.15 | 16.75 | 25.26 |
| Own speaker model | 14.0 | 7.41 | 6.68 | 6.96 |
| Own speaker model, oracle K | 4.91 | 2.04 | 7.23 | 4.41 |
