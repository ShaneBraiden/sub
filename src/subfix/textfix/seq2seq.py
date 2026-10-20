"""Small Transformer encoder-decoder (Vaswani et al. 2017) for subtitle correction, trained from
scratch by experiments/train_corrector.py. Pre-norm layers, shared source/target BPE
embeddings tied to the output projection, learned positions; decoding by batched beam search
with length normalisation."""
from __future__ import annotations

import math

from ..bpe import BOS, EOS, PAD, UNK

DEFAULT_NET = dict(d_model=256, heads=4, ff=1024, enc_layers=4, dec_layers=3, dropout=0.1, max_len=160)


def build_seq2seq(vocab_size: int, d_model: int = 256, heads: int = 4, ff: int = 1024, enc_layers: int = 4,
                  dec_layers: int = 3, dropout: float = 0.1, max_len: int = 160):
    import torch
    from torch import nn

    class Seq2Seq(nn.Module):
        def __init__(self):
            super().__init__()
            self.max_len = max_len
            self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
            self.pos = nn.Embedding(max_len, d_model)
            self.scale = math.sqrt(d_model)
            enc = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, batch_first=True, norm_first=True)
            dec = nn.TransformerDecoderLayer(d_model, heads, ff, dropout, batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(enc, enc_layers, norm=nn.LayerNorm(d_model),
                                                 enable_nested_tensor=False)
            self.decoder = nn.TransformerDecoder(dec, dec_layers, norm=nn.LayerNorm(d_model))
            self.drop = nn.Dropout(dropout)
            nn.init.normal_(self.emb.weight, std=d_model ** -0.5)
            with torch.no_grad():
                self.emb.weight[PAD].zero_()

        def embed(self, ids):
            pos = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
            return self.drop(self.emb(ids) * self.scale + self.pos(pos))

        def encode(self, src):
            mask = src == PAD
            return self.encoder(self.embed(src), src_key_padding_mask=mask), mask

        def decode(self, tgt, memory, mem_mask):
            T = tgt.shape[1]
            causal = torch.triu(torch.ones((T, T), dtype=torch.bool, device=tgt.device), diagonal=1)
            h = self.decoder(self.embed(tgt), memory, tgt_mask=causal, tgt_is_causal=True,
                             tgt_key_padding_mask=tgt == PAD, memory_key_padding_mask=mem_mask)
            return h @ self.emb.weight.t()

        def forward(self, src, tgt_in):
            memory, mem_mask = self.encode(src)
            return self.decode(tgt_in, memory, mem_mask)

        @torch.inference_mode()
        def generate(self, src, num_beams: int = 4, max_len: int | None = None, len_alpha: float = 0.6):
            """Batched beam search; returns a list of token-id lists (without BOS/EOS)."""
            B = src.shape[0]
            K = max(1, num_beams)
            max_len = min(max_len or self.max_len, self.max_len)
            memory, mem_mask = self.encode(src)
            memory = memory.repeat_interleave(K, 0)
            mem_mask = mem_mask.repeat_interleave(K, 0)
            seqs = torch.full((B * K, 1), BOS, dtype=torch.long, device=src.device)
            scores = torch.zeros(B, K, device=src.device)
            scores[:, 1:] = float("-inf")                        # start from one live beam
            done = torch.zeros(B * K, dtype=torch.bool, device=src.device)
            finished: list[list[tuple[float, list[int]]]] = [[] for _ in range(B)]
            V = self.emb.num_embeddings
            for step in range(max_len - 1):
                logp = torch.log_softmax(self.decode(seqs, memory, mem_mask)[:, -1].float(), dim=-1)
                logp[:, [PAD, BOS, UNK]] = float("-inf")
                logp[done] = float("-inf")
                cand = (scores.view(-1, 1) + logp).view(B, K * V)
                top, idx = cand.topk(2 * K, dim=1)
                beam, tok = idx // V, idx % V
                new_seqs, new_scores = [], torch.full((B, K), float("-inf"), device=src.device)
                for b in range(B):
                    kept = 0
                    for s, bm, t in zip(top[b].tolist(), beam[b].tolist(), tok[b].tolist()):
                        if s == float("-inf"):
                            break
                        prev = seqs[b * K + bm]
                        if t == EOS:
                            L = len(prev)
                            finished[b].append((s / (((5 + L) / 6) ** len_alpha), prev[1:].tolist()))
                            continue
                        if kept < K:
                            new_seqs.append(torch.cat([prev, prev.new_tensor([t])]))
                            new_scores[b, kept] = s
                            kept += 1
                    while kept < K:                               # pad dead beams
                        new_seqs.append(torch.cat([seqs[b * K], seqs.new_tensor([PAD])]))
                        kept += 1
                seqs = torch.stack(new_seqs)
                scores = new_scores
                done = torch.isinf(scores.view(-1))
                best_alive = scores.max(dim=1).values
                if all(len(finished[b]) >= K and
                       max(f[0] for f in finished[b]) > best_alive[b].item() / (((5 + step + 2) / 6) ** len_alpha)
                       for b in range(B)):
                    break
            out = []
            for b in range(B):
                if finished[b]:
                    out.append(max(finished[b], key=lambda x: x[0])[1])
                else:
                    k = int(scores[b].argmax())
                    out.append([t for t in seqs[b * K + k, 1:].tolist() if t != PAD])
            return out

    return Seq2Seq()
