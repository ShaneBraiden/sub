"""Our own CTC acoustic model for forced alignment, trained from scratch on LibriSpeech
train-clean-100 by experiments/train_aligner.py (replaces MMS-300M / wav2vec2-base-960h).

Features: 80-band log-mel filterbank (25 ms Hann window, 10 ms hop; mel filters built here),
normalised per chunk. Network: two 1-D convolutions (the second with stride 2 -> 20 ms frames,
the same rate as wav2vec2/MMS) and a 4-layer BiLSTM, with a linear CTC output over
blank, the word separator "|", a-z and the apostrophe.
"""
from __future__ import annotations

import numpy as np

SR = 16000
N_FFT = 400
HOP = 160
N_MELS = 80
SUBSAMPLE = 2
BLANK = 0
TOKENS = ["<blank>", "|"] + list("abcdefghijklmnopqrstuvwxyz") + ["'"]
FRAME_S = HOP * SUBSAMPLE / SR               # 0.02 s per output frame


def mel_filterbank(n_mels: int = N_MELS, n_fft: int = N_FFT, sr: int = SR, fmin: float = 20.0,
                   fmax: float | None = None) -> np.ndarray:
    """(n_mels, n_fft//2+1) triangular filters on the HTK mel scale."""
    fmax = fmax or sr / 2

    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)

    def mel2hz(m):
        return 700.0 * (10 ** (np.asarray(m) / 2595.0) - 1.0)

    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    edges = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, len(freqs)), dtype=np.float32)
    for m in range(n_mels):
        lo, c, hi = edges[m], edges[m + 1], edges[m + 2]
        up = (freqs - lo) / max(c - lo, 1e-9)
        down = (hi - freqs) / max(hi - c, 1e-9)
        fb[m] = np.maximum(0.0, np.minimum(up, down))
    return fb


def text_to_ids(words: list[str]) -> list[int]:
    """Target ids for a word list: letters/apostrophes, '|' between words; other chars dropped."""
    ids: list[int] = []
    for w in words:
        chars = [TOKENS.index(c) for c in w.lower() if c in TOKENS[2:]]
        if not chars:
            continue
        if ids:
            ids.append(1)
        ids.extend(chars)
    return ids


def build_acoustic(n_mels: int = N_MELS, conv: int = 256, hidden: int = 320, layers: int = 4,
                   dropout: float = 0.15):
    import torch
    from torch import nn

    class Frontend(nn.Module):
        """Waveform (B, S) -> normalised log-mel (B, T, n_mels) and lengths."""

        def __init__(self):
            super().__init__()
            self.register_buffer("window", torch.hann_window(N_FFT), persistent=False)
            self.register_buffer("fb", torch.from_numpy(mel_filterbank(n_mels)), persistent=False)

        def forward(self, wav, lengths):
            spec = torch.stft(wav.float(), N_FFT, HOP, window=self.window, return_complex=True, center=True)
            power = spec.real ** 2 + spec.imag ** 2                   # (B, F, T)
            mel = torch.log(torch.clamp(self.fb @ power, min=1e-10)).transpose(1, 2)
            n = lengths // HOP + 1
            mask = (torch.arange(mel.shape[1], device=mel.device)[None, :] < n[:, None]).unsqueeze(-1)
            cnt = mask.sum(1, keepdim=True).clamp(min=1)
            mean = (mel * mask).sum(1, keepdim=True) / cnt
            var = (((mel - mean) * mask) ** 2).sum(1, keepdim=True) / cnt
            return ((mel - mean) / torch.sqrt(var + 1e-5)) * mask, n

    class AcousticCTC(nn.Module):
        def __init__(self):
            super().__init__()
            self.frontend = Frontend()
            self.conv1 = nn.Conv1d(n_mels, conv, 5, padding=2)
            self.conv2 = nn.Conv1d(conv, conv, 5, stride=SUBSAMPLE, padding=2)
            self.norm = nn.LayerNorm(conv)
            self.lstm = nn.LSTM(conv, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                                dropout=dropout if layers > 1 else 0.0)
            self.drop = nn.Dropout(dropout)
            self.out = nn.Linear(2 * hidden, len(TOKENS))

        def frames(self, n_feat):
            return (n_feat - 1) // SUBSAMPLE + 1

        def forward_features(self, feats, n_feat):
            x = torch.relu(self.conv1(feats.transpose(1, 2)))
            x = torch.relu(self.conv2(x)).transpose(1, 2)
            x = self.drop(self.norm(x))
            n = self.frames(n_feat)
            packed = nn.utils.rnn.pack_padded_sequence(x, n.cpu(), batch_first=True, enforce_sorted=False)
            h, _ = self.lstm(packed)
            h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=x.shape[1])
            return torch.log_softmax(self.out(self.drop(h)).float(), dim=-1), n

        def forward(self, wav, lengths):
            feats, n_feat = self.frontend(wav, lengths)
            return self.forward_features(feats, n_feat)

    return AcousticCTC()


def spec_augment(feats, n_feat, freq_masks: int = 2, freq_width: int = 15, time_masks: int = 2,
                 time_frac: float = 0.05, generator=None):
    """In-place SpecAugment (Park et al. 2019) frequency and time masking on (B, T, F) features."""
    import torch
    B, T_, F = feats.shape
    for b in range(B):
        for _ in range(freq_masks):
            w = int(torch.randint(0, freq_width + 1, (1,), generator=generator))
            f0 = int(torch.randint(0, max(1, F - w), (1,), generator=generator))
            feats[b, :, f0:f0 + w] = 0.0
        L = int(n_feat[b])
        for _ in range(time_masks):
            w = int(torch.randint(0, max(1, int(time_frac * L)) + 1, (1,), generator=generator))
            t0 = int(torch.randint(0, max(1, L - w), (1,), generator=generator))
            feats[b, t0:t0 + w, :] = 0.0
    return feats


def greedy_decode(logp: np.ndarray) -> str:
    """Best-path CTC decoding of (T, V) log-probs to text."""
    best = logp.argmax(-1)
    out, prev = [], -1
    for t in best:
        if t != prev and t != BLANK:
            out.append(TOKENS[t])
        prev = t
    return "".join(out).replace("|", " ").strip()
