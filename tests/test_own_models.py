"""Unit tests for the self-trained model code (toy data; the trained weights are not needed)."""
import random

import numpy as np
import pytest

from subfix.bpe import BPE, EOS
from subfix.lm import lm_words, sentence_pieces
from subfix.ngram import KneserNeyLM
from subfix.postag import PerceptronTagger, brown_to_ud, split_token
from subfix.textfix.punct import Featurizer, label_of


@pytest.fixture(scope="module")
def toy_lm():
    rng = random.Random(0)
    vocab = "the cat dog sat on mat a ran fast slowly big red".split()
    sents = [[rng.choice(vocab) for _ in range(rng.randint(3, 9))] for _ in range(2000)]
    sents += [["the", "cat", "sat", "on", "the", "mat"]] * 100
    return KneserNeyLM.train(lambda: iter(sents), order=4, min_count=1, log=lambda *a: None)


@pytest.mark.parametrize("hist", [["the", "cat", "sat"], ["on", "the"], ["unseen", "big"], []])
def test_kneser_ney_distributions_sum_to_one(toy_lm, hist):
    H = toy_lm.order - 1
    h = [toy_lm.word_id(x) for x in hist][-H:]
    h = [-1] * (H - len(h)) + h
    cand = np.arange(1, len(toy_lm.vocab))           # every word except <s>
    p = np.exp(toy_lm.logprobs(np.array([h] * len(cand)), cand))
    assert p.sum() == pytest.approx(1.0, abs=1e-5)


def test_kneser_ney_prefers_seen_order_and_penalises_oov(toy_lm):
    good, bad = toy_lm.score_sequences([(["the", "cat", "sat", "on", "the", "mat"], True, True),
                                        (["mat", "the", "on", "sat", "cat", "the"], True, True)])
    assert good > bad + 5
    known, typo = toy_lm.score_sequences([(["the", "cat"], False, False), (["the", "cqat"], False, False)])
    assert known > typo


def test_kneser_ney_save_load_roundtrip(toy_lm, tmp_path):
    toy_lm.save(tmp_path / "lm.npz")
    lm2 = KneserNeyLM.load(tmp_path / "lm.npz")
    seq = [(["the", "big", "dog", "ran"], False, True)]
    assert lm2.score_sequences(seq) == pytest.approx(toy_lm.score_sequences(seq))


def test_lm_tokenization_and_sentence_pieces():
    assert lm_words("Hello,") == ["hello"]
    assert lm_words("25") == ["twenty", "five"]
    assert lm_words("--") == []
    pieces = sentence_pieces("went home. Then we slept".split())
    assert pieces == [(["went", "home"], False, True), (["then", "we", "slept"], True, False)]


def test_bpe_roundtrip_and_coverage():
    words = ("the cat sat on the mat. The cats sat, the mats sat? " * 50).split()
    bpe = BPE.train(words, vocab_size=60, min_char_count=1, log=lambda *a: None)
    text = "The cat sat on the mats."
    ids = bpe.encode(text)
    assert bpe.decode(ids) == text
    assert len(ids) < len(text.replace(" ", ""))       # merges happened
    assert bpe.covers(text) and not bpe.covers("Zebra!")


def test_bpe_save_load(tmp_path):
    bpe = BPE.train("a b ab abc abc abd".split() * 20, vocab_size=30, min_char_count=1, log=lambda *a: None)
    bpe.save(tmp_path / "bpe.json")
    assert BPE.load(tmp_path / "bpe.json").encode("abc abd") == bpe.encode("abc abd")


def test_brown_to_ud_mapping():
    sent = [("The", "AT"), ("Fulton", "NP-TL"), ("jury", "NN"), ("has", "HVZ"), ("said", "VBN"),
            ("that", "CS"), ("he", "PPS"), ("had", "HVD"), ("a", "AT"), ("dog", "NN"), ("and", "CC"),
            ("didn't", "DOD*"), ("go", "VB"), ("to", "TO"), ("Atlanta's", "NP$"), ("2", "CD"), (".", ".")]
    tags = [t for _, t in brown_to_ud(sent)]
    assert tags == ["DET", "PROPN", "NOUN", "AUX", "VERB", "SCONJ", "PRON", "VERB", "DET", "NOUN", "CCONJ",
                    "AUX", "VERB", "PART", "PROPN", "NUM", "PUNCT"]


def test_perceptron_learns_and_splits_punctuation():
    data = [[("the", "DET"), ("dog", "NOUN"), ("runs", "VERB"), (".", "PUNCT")],
            [("a", "DET"), ("cat", "NOUN"), ("sleeps", "VERB"), (".", "PUNCT")],
            [("the", "DET"), ("cat", "NOUN"), ("runs", "VERB"), (",", "PUNCT")]] * 30
    tagger = PerceptronTagger()
    tagger.train(data, n_iter=3, log=lambda *a: None)
    assert tagger.tag(["a", "dog", "sleeps", "."]) == ["DET", "NOUN", "VERB", "PUNCT"]
    assert tagger.tag_tokens(["a", "dog", "sleeps."]) == ["DET", "NOUN", "VERB"]
    assert split_token('"Hello,"') == ("Hello", ['"'], [",", '"'])


def test_punct_labels_and_features():
    assert [label_of(t) for t in ["word", "word.", "word,", "word?", "word!", "word;", "'word.'"]] == \
        ["0", ".", ",", "?", ".", ",", "."]
    f = Featurizer(["<pad>", "<unk>", "hello", "world"])
    wid, case, chars = f(["Hello", "WORLD", "zzz"])
    assert wid.tolist() == [2, 3, 1]
    assert case.tolist() == [2, 3, 1]                  # title, upper, lower
    assert chars.shape == (3, 20) and chars[2, :3].tolist() == [27, 27, 27]


def test_seq2seq_learns_to_copy():
    torch = pytest.importorskip("torch")
    from subfix.bpe import BOS
    from subfix.textfix.seq2seq import build_seq2seq
    torch.manual_seed(0)
    m = build_seq2seq(30, d_model=32, heads=2, ff=64, enc_layers=1, dec_layers=1, dropout=0.0, max_len=12)
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    for _ in range(300):
        x = torch.randint(4, 30, (32, 4))
        src = torch.cat([x, torch.full((32, 1), EOS)], 1)
        tgt = torch.cat([torch.full((32, 1), BOS), x, torch.full((32, 1), EOS)], 1)
        loss = torch.nn.functional.cross_entropy(m(src, tgt[:, :-1]).reshape(-1, 30), tgt[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    x = torch.randint(4, 30, (4, 4))
    out = m.generate(torch.cat([x, torch.full((4, 1), EOS)], 1), num_beams=3, max_len=8)
    assert sum(o == r for o, r in zip(out, x.tolist())) >= 3


def test_acoustic_frontend_and_targets():
    torch = pytest.importorskip("torch")
    from subfix.acoustic import FRAME_S, TOKENS, build_acoustic, mel_filterbank, text_to_ids
    fb = mel_filterbank()
    assert fb.shape == (80, 201) and (fb.sum(axis=1) > 0).all()
    assert text_to_ids(["Hi", "it's", "42"]) == [TOKENS.index("h"), TOKENS.index("i"), 1, TOKENS.index("i"),
                                                 TOKENS.index("t"), TOKENS.index("'"), TOKENS.index("s")]
    model = build_acoustic(conv=16, hidden=16, layers=1).eval()
    wav = torch.randn(2, 16000)
    lp, n = model(wav, torch.tensor([16000, 8000]))
    assert lp.shape[0] == 2 and lp.shape[2] == len(TOKENS)
    assert int(n[0]) == pytest.approx(1.0 / FRAME_S, abs=1) and int(n[1]) < int(n[0])
    assert torch.allclose(lp.exp().sum(-1), torch.ones_like(lp[..., 0]), atol=1e-4)


def test_ctc_aligner_wrapper_with_untrained_model():
    torch = pytest.importorskip("torch")
    from subfix.acoustic import build_acoustic
    from subfix.timing import CTCAligner
    torch.manual_seed(0)
    aligner = CTCAligner(device="cpu", model=build_acoustic(conv=16, hidden=16, layers=2))
    audio = np.random.default_rng(0).normal(0, 0.1, 32000).astype(np.float32)
    ids, owner = aligner._targets(["Hi,", "it's"])
    assert owner == [0, 0, -1, 1, 1, 1, 1] and ids[2] == aligner.sep
    spans = aligner.align(audio, ["hello", "world"])
    assert len(spans) == 2
    starts = [s[0] for s in spans if s is not None]
    assert starts == sorted(starts) and all(0.0 <= s <= 2.0 for s in starts)
