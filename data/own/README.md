# Your own recordings (e.g. Indian-English lectures)

Put each clip here as `<name>.wav` (or .flac/.mp3/.m4a/.mp4) plus `<name>.txt`, a verbatim transcript with
normal punctuation and casing. Then:

```powershell
.venv\Scripts\python toolsetch_data.py --own                              # registers data/clips/own/*.json
.venv\Scripts\python toolsuild_benchmark.py --stages asr refs corrupt manifest  # MFA refs + corruption
.venv\Scripts\python experimentsun_eval.py --manifest test --datasets own --systems noop b3_raw_whisper ours
```

The clips become the `own` dataset of the test manifest, reported separately (imp.md 6.2).
