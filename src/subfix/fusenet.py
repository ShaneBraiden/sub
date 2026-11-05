"""Neural evidence fusion: a network decides each fusion candidate (cfg.fusion.method = neural).

The linear rule of subfix.fuse weighs Whisper's confidence, phonetic similarity and the n-gram LM.
It has no way to check the audio itself, so it cannot tell a repeat the speaker really said
("I I think", which Whisper tends to drop) from a duplicate typo. This module adds that check and
learns the decision from data, trained from scratch by experiments/train_fusion.py:

  * CTC audio evidence (CtcEvidence): log P(audio | text after the edit) - log P(audio | text
    before it), from our own CTC acoustic model over the words around the edit (the idea of
    checking a transcript against CTC posteriors follows Zhang et al. 2022, WenetSpeech),
  * the signals the rule already sees, plus word frequency, casing and timing (FEATURES),
  * a BiLSTM over the aligned word sequence, so a decision sees its neighbours (a run of low
    confidence, a repeated phrase, a stretch where subtitle and audio disagree).

The output is P(the edit improves the subtitle) per candidate, one output per candidate kind (sub,
ins, del), compared with a dev-tuned threshold: a keep/edit tagger in the spirit of GECToR
(Omelianchuk et al. 2020), where a higher threshold trades recall for precision. Hard rules stay:
sound events are never changed, fillers never inserted, and split/merge keep their OOV rule.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from . import text as T
from .acoustic import BLANK, FRAME_S, text_to_ids
from .preprocess import SR
from .subio import Word
from .utils.device import resolve_device

log = logging.getLogger(__name__)

KINDS = ["match", "sub", "ins", "del", "split", "merge"]
SCORED = ("sub", "ins", "del")           # candidate kinds the network decides, one output each
CHARS = "abcdefghijklmnopqrstuvwxyz0123456789'"
MAX_CHARS = 16
DEFAULT_MODEL = "models/fusion"
CTC_FEATURES = ("ctc_ok", "ctc_delta", "ctc_per_char", "ctc_fit_before", "ctc_fit_after")
FEATURES = ("has_s", "has_a", "p", "phon", "cost", "lm", "oov_s", "oov_a", "dup", "span_conf", "filler", "func",
            "dur", "gap", "dt", "freq_s", "freq_a", "upper_s", "upper_a", "prev_end", "caseless", "len_s", "len_a",
            ) + CTC_FEATURES
FEATURE_INDEX = {name: k for k, name in enumerate(FEATURES)}


# ------------------------------------------------------------------ CTC audio evidence
def clip_emissions(aligner, audio: np.ndarray, chunk_s: float = 30.0, overlap_s: float = 2.0) -> np.ndarray:
    """CTC log-probabilities of a whole clip, (frames, tokens). Chunks overlap by 2 * overlap_s and
    each keeps the frames farthest from its edges (a BiLSTM is unreliable near a chunk edge)."""
    hop = int(round(FRAME_S * SR))
    chunk, ov = int(chunk_s * SR), int(overlap_s * SR)
    step = chunk - 2 * ov
    if len(audio) < 2 * hop:
        return np.zeros((0, 29), np.float32)
    parts, start = [], 0
    while True:
        end = min(len(audio), start + chunk)
        lp = aligner.emissions(audio[start:end])
        first = 0 if start == 0 else ov // hop
        if end == len(audio):
            parts.append(lp[first:])
            break
        parts.append(lp[first:(start + step + ov) // hop - start // hop])
        start += step
    return np.concatenate(parts)


class CtcEvidence:
    """The log-likelihood ratio of a text edit under our CTC acoustic model.

    `context` Whisper words on each side give the audio window (cut `margin` seconds wider) and the
    shared text, so the ratio measures only the edited words. Windows longer than `max_window`
    seconds (a passage Whisper skipped) give no evidence (ctc_ok = 0)."""

    def __init__(self, aligner, audio: np.ndarray | None = None, context: int = 2, margin: float = 0.15,
                 max_window: float = 10.0, batch_size: int = 256, emissions: np.ndarray | None = None):
        self.aligner = aligner
        self.audio = audio
        self.context = context
        self.margin = margin
        self.max_frames = int(max_window / FRAME_S)
        self.batch_size = batch_size
        self._lp = emissions

    @property
    def lp(self) -> np.ndarray:
        if self._lp is None:
            self._lp = clip_emissions(self.aligner, self.audio)
        return self._lp

    def _request(self, c, aligned: list[int | None], sub: list[Word], asr: list[Word]):
        """(first frame, last frame, target ids before the edit, target ids after it), or None."""
        K = self.context
        if c.kind in ("sub", "ins"):
            j_first, j_next = c.j, c.j + 1
        elif c.kind == "del":
            left = next((aligned[m] for m in range(c.i - 1, -1, -1) if aligned[m] is not None), None)
            right = next((aligned[m] for m in range(c.i + 1, len(sub)) if aligned[m] is not None), None)
            if left is None and right is None:
                return None
            j_first = left + 1 if left is not None else right
            j_next = right if right is not None else left + 1
        else:
            return None
        left_idx = list(range(max(0, j_first - K), j_first))
        right_idx = list(range(j_next, min(len(asr), j_next + K)))
        spoken = lambda k: T.spoken_form(asr[k].text)  # noqa: E731
        ctx_l, ctx_r = [spoken(k) for k in left_idx], [spoken(k) for k in right_idx]
        if c.kind == "sub":
            before = ctx_l + [T.spoken_form(sub[c.i].text)] + ctx_r
            after = ctx_l + [spoken(c.j)] + ctx_r
        elif c.kind == "ins":
            before = ctx_l + ctx_r
            after = ctx_l + [spoken(c.j)] + ctx_r
        else:
            before = ctx_l + [T.spoken_form(sub[c.i].text)] + ctx_r
            after = ctx_l + ctx_r
        anchors = left_idx + right_idx + ([c.j] if c.kind != "del" else [])
        if not anchors:
            return None
        f0 = max(0, int((min(asr[k].start for k in anchors) - self.margin) / FRAME_S))
        f1 = min(len(self.lp), int(math.ceil((max(asr[k].end for k in anchors) + self.margin) / FRAME_S)))
        ids_b, ids_a = text_to_ids(before), text_to_ids(after)
        if not ids_b or not ids_a or f1 - f0 > self.max_frames or f1 - f0 < max(len(ids_b), len(ids_a)) + 2:
            return None
        return f0, f1, ids_b, ids_a

    def score(self, cands: list, units: list[tuple], sub: list[Word], asr: list[Word]) -> np.ndarray:
        """(len(cands), len(CTC_FEATURES)) evidence features; all zero (ctc_ok = 0) without evidence."""
        import torch
        from rapidfuzz.distance import Levenshtein
        out = np.zeros((len(cands), len(CTC_FEATURES)), np.float32)
        aligned: list[int | None] = [None] * len(sub)
        for kind, i, j, i2, _ in units:
            if kind != "ins" and kind != "del" and i is not None:
                aligned[i] = j
                if i2 is not None:
                    aligned[i2] = j
        reqs = [(n, r) for n, c in enumerate(cands) for r in [self._request(c, aligned, sub, asr)] if r is not None]
        if not reqs:
            return out
        device = next(self.aligner.model.parameters()).device if hasattr(self.aligner, "model") else "cpu"
        reqs.sort(key=lambda x: x[1][1] - x[1][0])           # similar lengths share a padded batch
        lp = torch.from_numpy(self.lp)
        for start in range(0, len(reqs), self.batch_size):
            batch = reqs[start:start + self.batch_size]
            tmax = max(r[1] - r[0] for _, r in batch)
            x = torch.zeros(tmax, 2 * len(batch), lp.shape[1])
            in_len, targets, tgt_len = [], [], []
            for b, (_, (f0, f1, ids_b, ids_a)) in enumerate(batch):
                for h, ids in enumerate((ids_b, ids_a)):
                    x[:f1 - f0, 2 * b + h] = lp[f0:f1]
                    in_len.append(f1 - f0)
                    targets.extend(ids)
                    tgt_len.append(len(ids))
            nll = torch.nn.functional.ctc_loss(x.to(device), torch.tensor(targets, device=device),
                                               torch.tensor(in_len, device=device),
                                               torch.tensor(tgt_len, device=device), blank=BLANK, reduction="none",
                                               zero_infinity=False).cpu().numpy()
            for b, (n, (f0, f1, ids_b, ids_a)) in enumerate(batch):
                s_b, s_a = -float(nll[2 * b]), -float(nll[2 * b + 1])
                if not (math.isfinite(s_b) and math.isfinite(s_a)):
                    continue
                best = float(lp[f0:f1].max(dim=-1).values.sum())
                frames = f1 - f0
                delta = s_a - s_b
                changed = max(1, Levenshtein.distance(ids_b, ids_a))
                out[n] = (1.0, np.clip(delta, -30, 30) / 10, np.clip(delta / changed, -10, 10) / 5,
                          np.clip((s_b - best) / frames, -3, 0.5), np.clip((s_a - best) / frames, -3, 0.5))
        return out


# ------------------------------------------------------------------ unit table (network input)
@dataclass
class UnitTable:
    kinds: np.ndarray            # (n,) index into KINDS
    s_forms: list[str]           # lowercase subtitle word of each unit ("" if none)
    a_forms: list[str]           # lowercase ASR word of each unit ("" if none)
    num: np.ndarray              # (n, len(FEATURES))
    cand: np.ndarray             # (n,) index into the candidate list, -1 for match units


def unit_table(units: list[tuple], cands: list, sub: list[Word], asr: list[Word], spell=None, caseless: bool = False,
               evidence: CtcEvidence | None = None) -> UnitTable:
    """One row per aligned unit, in order. `cands` comes from Fuser.featurize (one per non-match unit)."""
    n = len(units)
    num = np.zeros((n, len(FEATURES)), np.float32)
    kinds = np.zeros(n, np.int8)
    cand = np.full(n, -1, np.int32)
    s_forms, a_forms = [""] * n, [""] * n
    ctc = evidence.score(cands, units, sub, asr) if evidence is not None else None
    ix = FEATURE_INDEX

    def set_word(row: np.ndarray, tok: str, form: str, name: str) -> None:
        row[ix[f"oov_{name}"]] = bool(spell.is_oov(tok)) if spell is not None else 0.0
        row[ix[f"freq_{name}"]] = math.log10(1 + spell.frequency(form)) / 10 if spell is not None else 0.0
        row[ix[f"upper_{name}"]] = T.core(tok)[:1].isupper()
        row[ix[f"len_{name}"]] = min(len(form), 20) / 10

    next_cand = 0
    for u, (kind, i, j, _, _) in enumerate(units):
        row = num[u]
        kinds[u] = KINDS.index(kind)
        row[ix["caseless"]] = caseless
        s = sub[i] if i is not None else None
        a = asr[j] if j is not None else None
        if s is not None:
            s_forms[u] = T.norm(s.text)
            row[ix["has_s"]] = 1.0
            set_word(row, s.text, s_forms[u], "s")
            row[ix["prev_end"]] = i > 0 and T.ends_sentence(sub[i - 1].text)
        if a is not None:
            a_forms[u] = T.norm(a.text)
            row[ix["has_a"]] = 1.0
            row[ix["p"]] = a.prob
            set_word(row, a.text, a_forms[u], "a")
            row[ix["dur"]] = float(np.clip(a.end - a.start, 0, 2))
            if j > 0:
                row[ix["gap"]] = float(np.clip(a.start - asr[j - 1].end, 0, 3))
        if s is not None and a is not None and s.start is not None:
            row[ix["dt"]] = float(np.clip(s.start - a.start, -5, 5)) / 5
        if kind == "match":
            row[ix["phon"]] = 1.0
            continue
        c = cands[next_cand]
        cand[u] = next_cand
        ft = c.feats
        row[ix["phon"]] = ft.get("phon", 0.0)
        row[ix["cost"]] = ft.get("cost", 1.0)
        row[ix["lm"]] = float(np.clip(ft.get("lm", 0.0), -10, 10)) / 5
        row[ix["dup"]] = ft.get("dup", False)
        row[ix["span_conf"]] = ft.get("span_conf", 0.0)
        row[ix["filler"]] = ft.get("filler", False)
        row[ix["func"]] = ft.get("func", False)
        if ctc is not None:
            row[[ix[name] for name in CTC_FEATURES]] = ctc[next_cand]
        next_cand += 1
    return UnitTable(kinds, s_forms, a_forms, num, cand)


def char_ids(form: str) -> list[int]:
    """Character ids of a lowercase form (0 = pad, 1 = unknown character)."""
    return [CHARS.index(ch) + 2 if ch in CHARS else 1 for ch in form[:MAX_CHARS]]


# ------------------------------------------------------------------ network
def build_net(n_words: int = 0, n_num: int = len(FEATURES), arch: str = "signals", word_dim: int = 48, char_dim: int = 16,
              char_filters: int = 48, unit_dim: int = 128, hidden: int = 32, layers: int = 2, dropout: float = 0.2):
    """arch: signals = BiLSTM over the signals (the shipped network); words = the same with word and
    character-CNN embeddings of both words (they overfit); mlp = per-candidate MLP on the signals;
    linear = logistic regression on the signals."""
    import torch
    from torch import nn

    class FusionNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.arch = arch
            n_out = len(SCORED)
            if arch == "linear":
                self.out = nn.Linear(n_num, n_out)
                return
            self.drop = nn.Dropout(dropout)
            self.kind = nn.Embedding(len(KINDS), 8)
            in_dim = n_num + 8
            if arch == "words":
                self.word = nn.Embedding(n_words, word_dim, padding_idx=0)
                self.char = nn.Embedding(len(CHARS) + 2, char_dim, padding_idx=0)
                self.conv = nn.Conv1d(char_dim, char_filters, kernel_size=3, padding=1)
                in_dim += 2 * (word_dim + char_filters)
            self.unit = nn.Sequential(nn.Linear(in_dim, unit_dim), nn.ReLU(), nn.Dropout(dropout))
            head_in = unit_dim + n_num
            if arch != "mlp":
                self.lstm = nn.LSTM(unit_dim, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                                    dropout=dropout if layers > 1 else 0.0)
                head_in += 2 * hidden
            self.head = nn.Sequential(nn.Linear(head_in, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, n_out))

        def token(self, words, chars):
            B, L, C = chars.shape
            ce = self.char(chars.view(B * L, C)).transpose(1, 2)
            ce = torch.relu(self.conv(ce)).masked_fill(chars.view(B * L, 1, C) == 0, 0.0)
            return torch.cat([self.word(words), ce.max(dim=2).values.view(B, L, -1)], dim=-1)

        def forward(self, kinds, num, s_word=None, s_char=None, a_word=None, a_char=None, lengths=None):
            """(B, L, len(SCORED)) logits; lengths = true sequence lengths of a padded batch."""
            if self.arch == "linear":
                return self.out(num)
            parts = [self.kind(kinds), num]
            if self.arch == "words":
                parts += [self.token(s_word, s_char), self.token(a_word, a_char)]
            x = self.unit(torch.cat(parts, dim=-1))
            feats = [x, num]
            if self.arch != "mlp":
                if lengths is not None:
                    packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                                               enforce_sorted=False)
                    h, _ = self.lstm(packed)
                    h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=x.shape[1])
                else:
                    h, _ = self.lstm(x)
                feats.append(h)
            return self.head(self.drop(torch.cat(feats, dim=-1)))

    return FusionNet()


# ------------------------------------------------------------------ inference
class NeuralFusion:
    """Loads a trained FusionNet and decides fusion candidates in place."""

    LEXICON_FACTOR = 0.5     # user-supplied domain terms face half the threshold (training data has no lexicon)

    def __init__(self, model_dir: str = DEFAULT_MODEL, device: str = "auto", threshold: float = 0.0):
        import torch
        from .lm import resolve_model_path
        self.torch = torch
        self.device = resolve_device(device)
        path = resolve_model_path(model_dir) / "model.pt"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: train it with python experiments/train_fusion.py")
        self.path = path
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.vocab: list[str] = ck["vocab"]
        self.ids = {w: i for i, w in enumerate(self.vocab)}
        self.arch = ck.get("net", {}).get("arch", "signals")
        self.net = build_net(len(self.vocab), **ck.get("net", {}))
        self.net.load_state_dict(ck["model"])
        self.net.to(self.device).eval()
        self.drop = [FEATURE_INDEX[name] for name in ck.get("drop", [])]
        self.uses_ctc = "ctc_delta" not in ck.get("drop", [])
        self.thresholds = {k: threshold for k in SCORED} if threshold > 0 else dict(ck["thresholds"])

    def _token_arrays(self, forms: list[str]) -> tuple[np.ndarray, np.ndarray]:
        wid = np.zeros(len(forms), np.int64)
        chars = np.zeros((len(forms), MAX_CHARS), np.int64)
        for k, form in enumerate(forms):
            if form:
                wid[k] = self.ids.get(form, 1)
                ci = char_ids(form) or [1]
                chars[k, :len(ci)] = ci
        return wid, chars

    def probabilities(self, table: UnitTable) -> np.ndarray:
        """(n, len(SCORED)) P(edit improves the subtitle) for every unit of the table."""
        torch = self.torch
        num = table.num.copy()
        num[:, self.drop] = 0.0
        to = lambda a: torch.from_numpy(a).unsqueeze(0).to(self.device)  # noqa: E731
        args = [to(table.kinds.astype(np.int64)), to(num)]
        if self.arch == "words":
            s_w, s_c = self._token_arrays(table.s_forms)
            a_w, a_c = self._token_arrays(table.a_forms)
            args += [to(s_w), to(s_c), to(a_w), to(a_c)]
        with torch.inference_mode():
            return torch.sigmoid(self.net(*args))[0].cpu().numpy()

    def decide(self, cands: list, units: list[tuple], sub: list[Word], asr: list[Word], spell=None,
               caseless: bool = False, evidence: CtcEvidence | None = None) -> None:
        if not cands:
            return
        table = unit_table(units, cands, sub, asr, spell, caseless, evidence if self.uses_ctc else None)
        probs = self.probabilities(table)
        for u in np.flatnonzero(table.cand >= 0):
            c = cands[table.cand[u]]
            if c.kind not in SCORED or c.rule in ("event", "filler"):
                continue                      # hard rules: split/merge, sound events and fillers
            c.score = float(probs[u, SCORED.index(c.kind)])
            tau = self.thresholds[c.kind] * (self.LEXICON_FACTOR if c.feats.get("lex") else 1.0)
            c.accept, c.rule = c.score > tau, "neural"

    def close(self) -> None:
        self.net = None
