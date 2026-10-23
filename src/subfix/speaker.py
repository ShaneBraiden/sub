"""Our own speaker diarization: a small speaker-embedding network trained from scratch on
LibriSpeech train-clean-100 (251 speakers) by experiments/train_speaker.py, plus clustering.
No pretrained model is involved (replaces pyannote).

Embedding network (x-vector style, Snyder et al. 2018, ~1.4 M parameters): 64-band log-mel
(25 ms / 10 ms, mean-normalised per window) -> five 1-D convolutions (TDNN layers with dilation
1, 2, 3, 1, 1) -> attentive statistics pooling (weighted mean + std over time) -> a 192-d embedding.
It is trained as a 251-way speaker classifier with an additive angular margin softmax (AAM, Deng et
al. 2019), so embeddings of the same voice point in the same direction (cosine similarity).

Diarization of a recording (OwnDiarizer):
  1. speech activity: frame log-energy above an adaptive threshold between the recording's noise
     floor and its loud speech, short gaps filled, short blips dropped
  2. embeddings of overlapping windows (1.5 s, hop 0.5 s) inside the speech
  3. agglomerative clustering (average linkage, cosine distance) cut at a distance threshold tuned
     on synthetic dev-clean conversations (stored with the model), or at num_speakers if given,
     then a few k-means style passes that move each window to its nearest speaker centroid
  4. every 10 ms speech frame takes the speaker whose windows over it are most similar to that
     speaker's centroid; turns shorter than min_turn are absorbed by their neighbours
The result has one speaker at a time (no overlapping speech), so turns == exclusive turns.
"""
from __future__ import annotations

import logging

import numpy as np

from .acoustic import mel_filterbank

log = logging.getLogger(__name__)

SR = 16000
N_FFT = 400
HOP = 160
N_MELS = 64
FRAME_S = HOP / SR                       # 0.01 s


# ------------------------------------------------------------------ network
def build_speaker_net(n_mels: int = N_MELS, channels: int = 256, pool_channels: int = 768, emb: int = 192):
    import torch
    from torch import nn

    class Frontend(nn.Module):
        """Waveform (B, S) -> log-mel (B, n_mels, T), mean-normalised over each window."""

        def __init__(self):
            super().__init__()
            self.register_buffer("window", torch.hann_window(N_FFT), persistent=False)
            self.register_buffer("fb", torch.from_numpy(mel_filterbank(n_mels)), persistent=False)

        def forward(self, wav):
            spec = torch.stft(wav.float(), N_FFT, HOP, window=self.window, return_complex=True, center=True)
            mel = torch.log(torch.clamp(self.fb @ (spec.real ** 2 + spec.imag ** 2), min=1e-8))
            return mel - mel.mean(-1, keepdim=True)

    def tdnn(i, o, k, d):
        return nn.Sequential(nn.Conv1d(i, o, k, dilation=d, padding=d * (k // 2)), nn.ReLU(), nn.BatchNorm1d(o))

    class SpeakerNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.frontend = Frontend()
            self.layers = nn.Sequential(tdnn(n_mels, channels, 5, 1), tdnn(channels, channels, 3, 2),
                                        tdnn(channels, channels, 3, 3), tdnn(channels, channels, 1, 1),
                                        tdnn(channels, pool_channels, 1, 1))
            self.attn = nn.Sequential(nn.Conv1d(pool_channels, 128, 1), nn.Tanh(), nn.Conv1d(128, pool_channels, 1))
            self.pool_norm = nn.BatchNorm1d(2 * pool_channels)
            self.embed = nn.Linear(2 * pool_channels, emb)
            self.emb_norm = nn.BatchNorm1d(emb)

        def forward(self, wav):
            """Waveform (B, S) at 16 kHz -> embeddings (B, emb) (not length-normalised)."""
            h = self.layers(self.frontend(wav))                  # (B, C, T)
            w = torch.softmax(self.attn(h), dim=-1)
            mu = (h * w).sum(-1)
            sd = torch.sqrt(((h ** 2) * w).sum(-1) - mu ** 2 + 1e-5)
            return self.emb_norm(self.embed(self.pool_norm(torch.cat([mu, sd], 1))))

    return SpeakerNet()


def build_aam_head(emb: int, n_classes: int, margin: float = 0.2, scale: float = 30.0):
    """Additive angular margin softmax: logits s*cos(theta + m) for the target class, s*cos(theta) otherwise."""
    import torch
    from torch import nn

    class AAMSoftmax(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(n_classes, emb) * 0.01)
            self.margin, self.scale = margin, scale

        def forward(self, x, labels=None):
            cos = nn.functional.linear(nn.functional.normalize(x), nn.functional.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
            if labels is None:
                return self.scale * cos
            target = torch.cos(torch.acos(cos) + self.margin)
            onehot = nn.functional.one_hot(labels, cos.shape[1]).bool()
            return self.scale * torch.where(onehot, target, cos)

    return AAMSoftmax()


class SpeakerEmbedder:
    """Our trained embedding network: windows of 16 kHz audio -> L2-normalised embeddings."""

    def __init__(self, path, device: str = "cpu"):
        import torch
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
        self.net = build_speaker_net(**ck.get("arch", {}))
        self.net.load_state_dict(ck["model"])
        self.net.eval().to(device)
        self.device = device
        self.threshold = float(ck.get("threshold", 0.6))     # dev-tuned clustering distance
        self.name = str(path)

    def __call__(self, windows: list[np.ndarray], batch: int = 64) -> np.ndarray:
        import torch
        out = []
        with torch.inference_mode():
            for k in range(0, len(windows), batch):
                chunk = windows[k:k + batch]
                n = max(len(w) for w in chunk)
                x = np.zeros((len(chunk), n), dtype=np.float32)
                for i, w in enumerate(chunk):          # short windows are repeated to the batch length
                    x[i] = np.resize(w, n)
                out.append(self.net(torch.from_numpy(x).to(self.device)).float().cpu().numpy())
        e = np.concatenate(out) if out else np.zeros((0, 1), np.float32)
        return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-8)


class MFCCStats:
    """Untrained baseline embedding: mean and std of 20 MFCCs over the window (no learning)."""

    threshold = 0.3
    name = "mfcc-stats"

    def __init__(self, n_mfcc: int = 20):
        self.fb = mel_filterbank(40)
        k = np.arange(40)
        self.dct = np.cos(np.pi / 40 * (k[None, :] + 0.5) * np.arange(1, n_mfcc + 1)[:, None]).astype(np.float32)

    def __call__(self, windows: list[np.ndarray], batch: int = 0) -> np.ndarray:
        win = np.hanning(N_FFT).astype(np.float32)
        rows = []
        for w in windows:
            w = np.pad(w, (0, max(0, N_FFT - len(w))))
            idx = np.arange(0, len(w) - N_FFT + 1, HOP)[:, None] + np.arange(N_FFT)[None, :]
            power = np.abs(np.fft.rfft(w[idx] * win, axis=1)) ** 2
            mfcc = np.log(np.maximum(power @ self.fb.T, 1e-8)) @ self.dct.T
            rows.append(np.concatenate([mfcc.mean(0), mfcc.std(0)]))
        e = np.asarray(rows, dtype=np.float32)
        if len(e):
            e = (e - e.mean(0)) / (e.std(0) + 1e-5)          # standardise within the recording
        return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-8)


# ------------------------------------------------------------------ speech activity
def speech_regions(audio: np.ndarray, sr: int = SR, rel: float = 0.3, min_speech: float = 0.25,
                   min_gap: float = 0.3) -> list[tuple[float, float]]:
    """Energy-based speech activity: (start, end) seconds. A frame is speech when its log-energy
    lies above noise + rel * (loud - noise), with noise / loud = 5th / 95th percentile."""
    win = int(0.025 * sr)
    n = 1 + max(len(audio) - win, 0) // HOP
    if n <= 1 or not np.any(audio):
        return []
    idx = np.arange(n)[:, None] * HOP + np.arange(win)[None, :]
    frames = np.pad(audio, (0, max(0, idx.max() + 1 - len(audio))))[idx]
    energy = 10 * np.log10(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-10)
    energy = np.convolve(np.pad(energy, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    noise, loud = np.percentile(energy, 5), np.percentile(energy, 95)
    speech = energy > noise + rel * (loud - noise)
    regions, k = [], 0
    while k < n:
        if not speech[k]:
            k += 1
            continue
        a = k
        while k < n and speech[k]:
            k += 1
        regions.append([a * FRAME_S, k * FRAME_S + 0.015])
    merged: list[list[float]] = []
    for r in regions:
        if merged and r[0] - merged[-1][1] < min_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    dur = len(audio) / sr
    return [(round(a, 3), round(min(b, dur), 3)) for a, b in merged if b - a >= min_speech]


# ------------------------------------------------------------------ clustering
def cluster(emb: np.ndarray, threshold: float, num_speakers: int = 0, min_speakers: int = 0,
            max_speakers: int = 0, refine: int = 3) -> np.ndarray:
    """Speaker label per embedding: average-linkage clustering on cosine distance, cut at
    `threshold` (or at exactly num_speakers), then k-means style refinement on the unit sphere."""
    n = len(emb)
    if n == 0:
        return np.zeros(0, dtype=int)
    if n == 1 or num_speakers == 1:
        return np.zeros(n, dtype=int)
    from scipy.cluster.hierarchy import fcluster, linkage
    Z = linkage(emb, method="average", metric="cosine")
    if num_speakers > 0:
        labels = fcluster(Z, t=min(num_speakers, n), criterion="maxclust")
    else:
        labels = fcluster(Z, t=threshold, criterion="distance")
        k = len(set(labels))
        lo, hi = max(min_speakers, 1), max_speakers if max_speakers > 0 else n
        if k < lo or k > hi:
            labels = fcluster(Z, t=min(max(k, lo), hi, n), criterion="maxclust")
    labels = np.unique(labels, return_inverse=True)[1]
    k = labels.max() + 1
    for _ in range(refine):
        cent = np.stack([emb[labels == c].mean(0) for c in range(k)])
        cent /= np.maximum(np.linalg.norm(cent, axis=1, keepdims=True), 1e-8)
        new = (emb @ cent.T).argmax(1)
        if len(set(new)) < k or np.array_equal(new, labels):     # never lose a speaker in refinement
            break
        labels = new
    return labels


# ------------------------------------------------------------------ diarizer
class OwnDiarizer:
    """Diarization with our embedding model (or any embedder with the same call signature)."""

    def __init__(self, embedder, threshold: float = 0.0, window: float = 1.5, hop: float = 0.5,
                 vad_rel: float = 0.3, min_turn: float = 0.5):
        self.embedder = embedder
        self.threshold = threshold if threshold > 0 else embedder.threshold
        self.window, self.hop, self.vad_rel, self.min_turn = window, hop, vad_rel, min_turn

    def windows(self, regions, n_samples: int, sr: int):
        """(start, end) seconds of the embedding windows covering the speech regions."""
        out = []
        for a, b in regions:
            if b - a <= self.window:
                out.append((a, b))
                continue
            t = a
            while t + self.window < b:
                out.append((t, t + self.window))
                t += self.hop
            out.append((max(a, b - self.window), b))
        return out

    def embed(self, audio: np.ndarray, sr: int):
        """Speech regions, embedding windows and their embeddings (the expensive part)."""
        audio = np.asarray(audio, dtype=np.float32)
        regions = speech_regions(audio, sr, self.vad_rel)
        wins = [(a, b) for a, b in self.windows(regions, len(audio), sr) if b - a >= 0.2]
        emb = self.embedder([audio[int(a * sr):int(b * sr)] for a, b in wins]) if wins else None
        return {"duration": len(audio) / sr, "regions": regions, "wins": wins, "emb": emb}

    def __call__(self, audio: np.ndarray, sr: int, num_speakers: int = 0, min_speakers: int = 0,
                 max_speakers: int = 0):
        return self.decode(self.embed(audio, sr), num_speakers, min_speakers, max_speakers)

    def decode(self, e: dict, num_speakers: int = 0, min_speakers: int = 0, max_speakers: int = 0,
               threshold: float | None = None):
        """Cluster the embeddings of embed() and turn them into speaker turns."""
        from .diarize import Diarization, Turn
        threshold = self.threshold if threshold is None else threshold
        info = {"model": self.embedder.name, "backend": "own", "threshold": round(threshold, 4),
                "exclusive": "own", "num_speakers": num_speakers}
        regions, wins, emb = e["regions"], e["wins"], e["emb"]
        if not wins:
            return Diarization.build([], [], info)
        labels = cluster(emb, threshold, num_speakers, min_speakers, max_speakers)
        k = int(labels.max()) + 1
        cent = np.stack([emb[labels == c].mean(0) for c in range(k)])
        cent /= np.maximum(np.linalg.norm(cent, axis=1, keepdims=True), 1e-8)
        sims = emb @ cent.T                                     # (windows, speakers)
        # frame scores: mean similarity of the windows covering each 10 ms frame
        n_fr = int(np.ceil(e["duration"] / FRAME_S)) + 1
        score = np.zeros((n_fr, k), np.float32)
        count = np.zeros(n_fr, np.float32)
        for (a, b), s in zip(wins, sims):
            i, j = int(a / FRAME_S), int(b / FRAME_S)
            score[i:j] += s
            count[i:j] += 1
        turns: list[Turn] = []
        for a, b in regions:
            i, j = int(a / FRAME_S), max(int(b / FRAME_S), int(a / FRAME_S) + 1)
            seg = score[i:j] / np.maximum(count[i:j, None], 1)
            lab = seg.argmax(1)
            runs = _runs(lab)
            runs = _absorb_short(runs, seg, int(round(self.min_turn / FRAME_S)))
            turns += [Turn(a + p * FRAME_S, min(a + q * FRAME_S, b), f"S{c}") for p, q, c in runs]
        info["speakers_found"] = k
        return Diarization.build(turns, list(turns), info)

    def close(self) -> None:
        self.embedder = None


def _runs(lab: np.ndarray) -> list[list[int]]:
    runs: list[list[int]] = []
    for t, c in enumerate(lab.tolist()):
        if runs and runs[-1][2] == c:
            runs[-1][1] = t + 1
        else:
            runs.append([t, t + 1, c])
    return runs


def _absorb_short(runs: list[list[int]], seg: np.ndarray, min_len: int) -> list[list[int]]:
    """Give each run shorter than min_len (shortest first) to the neighbour its frames prefer."""
    runs = [list(r) for r in runs]
    while len(runs) > 1:
        k = min(range(len(runs)), key=lambda i: runs[i][1] - runs[i][0])
        p, q, _ = runs[k]
        if q - p >= min_len:
            break
        cands = [i for i in (k - 1, k + 1) if 0 <= i < len(runs)]
        best = max(cands, key=lambda i: float(seg[p:q, runs[i][2]].sum()))
        runs[best][0], runs[best][1] = min(runs[best][0], p), max(runs[best][1], q)
        del runs[k]
        merged: list[list[int]] = []
        for r in runs:
            if merged and merged[-1][2] == r[2]:
                merged[-1][1] = r[1]
            else:
                merged.append(r)
        runs = merged
    return runs
