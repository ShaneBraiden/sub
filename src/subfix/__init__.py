"""subfix: audio-aware subtitle correction (English).

Mode A: audio + faulty subtitles -> ASR evidence fusion -> text clean-up -> forced alignment -> re-segmentation.
Mode B: audio only -> ASR draft -> the same post-correction chain.
"""

__version__ = "0.1.0"
