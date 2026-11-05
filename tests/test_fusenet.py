"""Unit tests for the neural fusion code (toy data and random weights; trained models not needed)."""
import itertools

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from experiments.fusion_data import edit_gains  # noqa: E402
from experiments.summarize import paired_bootstrap  # noqa: E402
from experiments.tune_fusion import Tuner  # noqa: E402
from experiments.train_fusion import GRID, WER_WEIGHT, f05, threshold_counts, tune_thresholds  # noqa: E402
from subfix import text as T  # noqa: E402
from subfix.acoustic import BLANK, FRAME_S, TOKENS, text_to_ids  # noqa: E402
from subfix.align import align_words  # noqa: E402
from subfix.config import Config  # noqa: E402
from subfix.fuse import Candidate, Fuser  # noqa: E402
from subfix.fusenet import (CTC_FEATURES, FEATURE_INDEX, FEATURES, KINDS, MAX_CHARS, SCORED, CtcEvidence,  # noqa: E402
                            NeuralFusion, build_net, char_ids, clip_emissions, unit_table)
from subfix.lm import StubLM  # noqa: E402
from subfix.preprocess import SR  # noqa: E402
from subfix.subio import Word  # noqa: E402


class StubSpell:
    def is_oov(self, tok):
        return T.norm(tok) in {"teh", "recieve"}

    def frequency(self, tok):
        return 0 if self.is_oov(tok) else 1000


def words(text, src="sub", prob=1.0, step=0.5):
    return [Word(t, k * step, k * step + 0.4, prob, src, {"cue": 0, "line": 0}) for k, t in enumerate(text.split())]


def candidates(sub_text, asr_text, prob=0.9):
    cfg = Config()
    sub, asr = words(sub_text), words(asr_text, "asr", prob)
    fz = Fuser(cfg, lm=StubLM(), spell=StubSpell())
    units = fz.units(align_words(sub, asr, cfg, use_time=False), sub, asr)
    return sub, asr, units, [fz.decide(c) for c in fz.featurize(units, sub, asr)], fz


# ------------------------------------------------------------------ thresholds
def toy_decisions(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    out = rng.integers(0, len(SCORED), n)
    gain = np.where(rng.random(n) < 0.85, 1, -1)
    prob = np.where(gain > 0, rng.beta(6, 1.5, n), rng.beta(1.5, 4, n))
    return prob, gain, out, np.zeros(n, bool)


def test_threshold_search_matches_brute_force():
    prob, gain, out, base = toy_decisions()
    err, n_ref = 1.1 * (gain > 0).sum(), 20000.0
    tau = tune_thresholds(prob, gain, out, base, err, n_ref)
    best = -np.inf
    for a, b, c in itertools.product(GRID[::2], GRID[::2], GRID[::2]):
        acc = prob > np.array([a, b, c])[out]
        tp, fp = (acc & (gain > 0)).sum(), (acc & (gain <= 0)).sum()
        best = max(best, f05(tp, fp, err) - WER_WEIGHT * 100 * (err - gain[acc].sum()) / n_ref)
    acc = prob > np.array([tau[k] for k in SCORED])[out]
    tp, fp = (acc & (gain > 0)).sum(), (acc & (gain <= 0)).sum()
    assert f05(tp, fp, err) - WER_WEIGHT * 100 * (err - gain[acc].sum()) / n_ref >= best - 1e-12


def test_threshold_counts_add_the_decisions_that_stay_with_the_rule():
    prob, gain, out, _ = toy_decisions(200)
    out[:20] = -1
    base = np.zeros(200, bool)
    base[:10] = True
    tp, fp, net = threshold_counts(prob, gain, out, base)
    assert tp.shape == (len(GRID),) * len(SCORED)
    acc = (prob > GRID[0]) & (out >= 0) | base
    assert tp[0, 0, 0] == (acc & (gain > 0)).sum() and fp[0, 0, 0] == (acc & (gain <= 0)).sum()
    assert net[0, 0, 0] == gain[acc].sum()


# ------------------------------------------------------------------ labels
def test_gain_rewards_only_edits_that_reach_the_reference():
    sub, asr, _, cands, _ = candidates("I saw teh cat", "I saw the cat")
    assert edit_gains(cands, sub, asr, "i saw the cat".split()) == [1]
    # the subtitle was right and the ASR wrong: the same edit now harms
    sub, asr, _, cands, _ = candidates("I saw that cat", "I saw the cat")
    assert edit_gains(cands, sub, asr, "i saw that cat".split()) == [-1]


def test_gain_neutral_when_the_edit_swaps_one_error_for_another():
    sub, asr, _, cands, _ = candidates("I saw teh cat", "I saw tea cat")
    assert edit_gains(cands, sub, asr, "i saw the cat".split()) == [0]


def test_gain_of_duplicate_deletion_depends_on_what_was_said():
    sub, asr, _, cands, _ = candidates("I I think so", "I think so")
    dels = [c for c in cands if c.kind == "del"]
    assert dels
    assert edit_gains(dels, sub, asr, "i think so".split()) == [1]       # a typo: delete
    assert edit_gains(dels, sub, asr, "i i think so".split()) == [-1]    # really said twice: keep


def test_gain_of_insertion():
    sub, asr, _, cands, _ = candidates("I saw cat", "I saw the cat")
    ins = [c for c in cands if c.kind == "ins"]
    assert edit_gains(ins, sub, asr, "i saw the cat".split()) == [1]
    assert edit_gains(ins, sub, asr, "i saw cat".split()) == [-1]


# ------------------------------------------------------------------ CTC evidence
class StubAligner:
    """Emissions that spell a given text, one letter per frame group, with blanks between letters."""

    def __init__(self, text=""):
        self.text = text

    def emissions(self, audio):
        n = (len(audio) // 160) // 2 + 1
        lp = np.full((n, len(TOKENS)), -8.0, np.float32)
        lp[:, BLANK] = -0.01
        return lp


def spelled(text, frames_per_token=4, frames=None):
    """(T, V) log-probs that say `text`: each token for a few frames, then blanks."""
    ids = text_to_ids(text.split())
    total = frames or (len(ids) * (frames_per_token + 2) + 4)
    lp = np.full((total, len(TOKENS)), -10.0, np.float32)
    lp[:, BLANK] = -0.01
    t = 2
    for tok in ids:
        lp[t:t + frames_per_token] = -10.0
        lp[t:t + frames_per_token, tok] = -0.01
        t += frames_per_token + 2
    return lp


def test_clip_emissions_keep_a_continuous_frame_count():
    audio = np.zeros(int(100.3 * SR), np.float32)
    lp = clip_emissions(StubAligner(), audio)
    assert abs(len(lp) - 100.3 / FRAME_S) <= 3         # no frames lost or doubled at the chunk joins
    assert len(clip_emissions(StubAligner(), np.zeros(int(12.0 * SR), np.float32))) == 601


def _evidence_for(said, sub_text, asr_text):
    sub, asr, units, cands, _ = candidates(sub_text, asr_text)
    lp = spelled(said)
    for k, w in enumerate(asr):     # Whisper's words span the whole utterance
        w.start, w.end = k * len(lp) * FRAME_S / len(asr), (k + 1) * len(lp) * FRAME_S / len(asr)
    ev = CtcEvidence(StubAligner(), emissions=lp, margin=0.0)
    return cands, ev.score(cands, units, sub, asr)


def test_ctc_evidence_separates_real_repeats_from_typed_duplicates():
    delta = CTC_FEATURES.index("ctc_delta")
    ok = CTC_FEATURES.index("ctc_ok")
    cands, ev = _evidence_for("i think so", "I I think so", "I think so")
    k = next(n for n, c in enumerate(cands) if c.kind == "del")
    assert ev[k, ok] == 1 and ev[k, delta] > 0          # audio has one "I": deleting the copy fits better
    cands, ev = _evidence_for("i i think so", "I I think so", "I think so")
    k = next(n for n, c in enumerate(cands) if c.kind == "del")
    assert ev[k, ok] == 1 and ev[k, delta] < 0          # audio has two: deleting one fits worse


def test_ctc_evidence_prefers_the_spoken_substitution():
    cands, ev = _evidence_for("i saw the cat", "I saw dog cat", "I saw the cat")
    k = next(n for n, c in enumerate(cands) if c.kind == "sub")
    assert ev[k, CTC_FEATURES.index("ctc_delta")] > 0
    cands, ev = _evidence_for("i saw dog cat", "I saw dog cat", "I saw the cat")
    k = next(n for n, c in enumerate(cands) if c.kind == "sub")
    assert ev[k, CTC_FEATURES.index("ctc_delta")] < 0


def test_too_long_a_window_gives_no_evidence():
    sub, asr, units, cands, _ = candidates("I saw teh cat", "I saw the cat")
    for k, w in enumerate(asr):
        w.start, w.end = k * 8.0, k * 8.0 + 0.4
    ev = CtcEvidence(StubAligner(), emissions=spelled("i saw the cat", frames=2000), margin=0.0, max_window=10.0)
    assert ev.score(cands, units, sub, asr)[0, CTC_FEATURES.index("ctc_ok")] == 0


# ------------------------------------------------------------------ unit table + network
def test_unit_table_has_one_row_per_unit_and_links_candidates():
    sub, asr, units, cands, _ = candidates("I saw teh cat", "I saw the cat")
    t = unit_table(units, cands, sub, asr, StubSpell(), caseless=True)
    assert t.num.shape == (len(units), len(FEATURES))
    assert (t.cand >= 0).sum() == len(cands)
    u = int(np.flatnonzero(t.cand >= 0)[0])
    assert KINDS[t.kinds[u]] == "sub" and (t.s_forms[u], t.a_forms[u]) == ("teh", "the")
    assert t.num[u, FEATURE_INDEX["oov_s"]] == 1 and t.num[u, FEATURE_INDEX["oov_a"]] == 0
    assert t.num[u, FEATURE_INDEX["p"]] == pytest.approx(0.9)
    assert (t.num[:, FEATURE_INDEX["caseless"]] == 1).all()
    assert (t.num[:, FEATURE_INDEX["ctc_ok"]] == 0).all()     # no evidence given


def test_char_ids_truncate_and_mark_unknown_characters():
    assert len(char_ids("x" * 40)) == MAX_CHARS
    assert char_ids("é")[0] == 1


@pytest.mark.parametrize("arch", ["signals", "words", "mlp", "linear"])
def test_net_output_shape_and_padding(arch):
    net = build_net(50, arch=arch).eval()
    B, L = 2, 7
    kinds = torch.zeros(B, L, dtype=torch.long)
    num = torch.rand(B, L, len(FEATURES))
    extra = []
    if arch == "words":
        w = torch.ones(B, L, dtype=torch.long)
        c = torch.ones(B, L, MAX_CHARS, dtype=torch.long)
        extra = [w, c, w, c]
    with torch.no_grad():
        out = net(kinds, num, *extra)
        assert out.shape == (B, L, len(SCORED))
        if arch in ("signals", "words"):
            # padding the second sequence must not change what the first one sees
            short = net(kinds[:1, :4], num[:1, :4], *[x[:1, :4] for x in extra])
            packed = net(kinds, num, *extra, lengths=torch.tensor([4, 7]))
            assert torch.allclose(packed[0, :4], short[0], atol=1e-5)


# ------------------------------------------------------------------ decisions
def checkpoint(tmp_path, bias: float, thresholds=None, arch="signals", drop=()):
    """A saved network whose every output is pushed towards `bias` (random weights otherwise)."""
    torch.manual_seed(0)
    vocab = ["<pad>", "<unk>", "the", "cat"]
    net = build_net(len(vocab), arch=arch)
    last = net.out if arch == "linear" else net.head[-1]
    torch.nn.init.zeros_(last.weight)
    torch.nn.init.constant_(last.bias, bias)
    torch.save({"model": net.state_dict(), "vocab": vocab, "net": {"arch": arch}, "drop": list(drop),
                "thresholds": thresholds or {k: 0.5 for k in SCORED}}, tmp_path / "model.pt")
    return str(tmp_path)


def fuse_with(neural, sub_text, asr_text, events=()):
    cfg = Config()
    sub, asr = words(sub_text), words(asr_text, "asr", 0.99)
    for w in sub:
        if w.text in events:
            w.meta["event"] = True
    res = Fuser(cfg, lm=StubLM(), spell=StubSpell(), neural=neural).fuse(sub, asr, align_words(sub, asr, cfg, use_time=False))
    return " ".join(w.text for w in res.words), res


def test_neural_decision_overrides_the_rule(tmp_path):
    yes = NeuralFusion(checkpoint(tmp_path, 5.0), "cpu")
    out, res = fuse_with(yes, "I saw that cat", "I saw the cat")
    assert out == "I saw the cat"
    e = next(e for e in res.edits if e["op"] == "sub")
    assert e["rule"] == "neural" and e["decision"] == "accept" and e["score"] > 0.9
    no = NeuralFusion(checkpoint(tmp_path, -5.0), "cpu")
    assert fuse_with(no, "I saw that cat", "I saw the cat")[0] == "I saw that cat"


def test_thresholds_are_per_kind_and_can_be_overridden(tmp_path):
    p = checkpoint(tmp_path, 0.0, {"sub": 0.9, "ins": 0.1, "del": 0.9})        # P = 0.5 everywhere
    net = NeuralFusion(p, "cpu")
    assert fuse_with(net, "I saw that cat", "I saw the cat")[0] == "I saw that cat"
    assert fuse_with(net, "I saw cat", "I saw the cat")[0] == "I saw the cat"
    assert fuse_with(NeuralFusion(p, "cpu", threshold=0.4), "I saw that cat", "I saw the cat")[0] == "I saw the cat"


def test_hard_rules_outrank_the_network(tmp_path):
    yes = NeuralFusion(checkpoint(tmp_path, 5.0), "cpu")
    assert "[MUSIC]" in fuse_with(yes, "I [MUSIC] saw it", "I saw it", events=("[MUSIC]",))[0]
    assert "[MUSIC]" in fuse_with(yes, "[MUSIC] I saw it", "the I saw it", events=("[MUSIC]",))[0]
    assert "um" not in fuse_with(yes, "I saw it", "I um saw it")[0]


def test_lexicon_terms_face_a_lower_threshold(tmp_path):
    net = NeuralFusion(checkpoint(tmp_path, 0.0, {"sub": 0.7, "ins": 0.7, "del": 0.7}), "cpu")
    sub, asr = words("I saw parish"), words("I saw Paris", "asr", 0.99)
    cfg = Config()
    ops = align_words(sub, asr, cfg, use_time=False)
    plain = Fuser(cfg, lm=StubLM(), spell=StubSpell(), neural=net).fuse(sub, asr, ops)
    assert plain.words[-1].text == "parish"
    with_term = Fuser(cfg, lm=StubLM(), spell=StubSpell(), lexicon=["Paris"], neural=net).fuse(sub, asr, ops)
    assert with_term.words[-1].text != "parish"


def test_ablation_checkpoint_needs_no_audio_evidence(tmp_path):
    net = NeuralFusion(checkpoint(tmp_path, 5.0, drop=CTC_FEATURES), "cpu")
    assert net.uses_ctc is False
    assert fuse_with(net, "I saw that cat", "I saw the cat")[0] == "I saw the cat"


def test_no_network_keeps_the_rule():
    cfg = Config()
    sub, asr = words("I saw that cat"), words("I saw the cat", "asr", 0.99)
    res = Fuser(cfg, lm=StubLM(), spell=StubSpell()).fuse(sub, asr, align_words(sub, asr, cfg, use_time=False))
    assert all(e["rule"] != "neural" for e in res.edits)


def test_missing_model_falls_back_to_the_rule(tmp_path):
    from subfix.models import Models
    cfg = Config.load(use_tuned=False, overrides={"fusion.model": str(tmp_path / "nope"), "device": "cpu"})
    assert Models(cfg).fusion() is None


def test_rule_tuner_uses_the_selection_score_of_the_network():
    # 50 sure edits, then two unsure ones: one repairs five words, the other changes nothing. Taking
    # them lowers F0.5 and WER, so they are worth it only if the score looks at WER.
    data = {"gold_total": 100, "n_ref": 500, "base": [0, 0, 0],
            "rows": {"sub": [[0.9, 0, 0, 0, 0, 1]] * 50 + [[0.6, 0, 0, 0, 0, 5], [0.6, 0, 0, 0, 0, 0]], "ins": [], "del": []}}
    start = dict(w_conf=1.0, w_phon=0.0, w_lm=0.0, w_oov=0.0, tau_sub=0.5, tau_ins=0.5, tau_lm_ins=0.0,
                 del_span_conf=0.9, tau_lm_del=0.0)
    f05_only = Tuner(data, wer_weight=0.0).tune(start)
    assert (f05_only["dev_tp"], f05_only["dev_fp"]) == (50, 0)
    assert f05_only["dev_wer"] == pytest.approx(10.0)
    both = Tuner(data, wer_weight=WER_WEIGHT).tune(start)
    assert (both["dev_tp"], both["dev_fp"]) == (51, 1)
    assert both["dev_wer"] == pytest.approx(9.0)
    assert both["dev_f05"] == pytest.approx(f05(51, 1, 100), abs=1e-4)
    assert both["dev_f05"] < f05_only["dev_f05"]


def test_paired_bootstrap_signs_and_pooled_scores():
    rows = [{"sub": 5, "del": 1, "ins": 1 + k, "n_ref_words": 100 + 10 * k, "edit_tp": 10, "edit_fp": 2, "edit_fn": 3} for k in range(6)]
    better = [dict(r, sub=2, edit_fp=0) for r in rows]
    assert paired_bootstrap(rows, rows)["wer"] == (0.0, 0.0, 0.0)
    d = paired_bootstrap(better, rows)
    assert d["wer"][1] <= d["wer"][0] <= d["wer"][2] < 0
    assert 0 < d["edit_f05"][1] <= d["edit_f05"][0] <= d["edit_f05"][2]
    n = sum(r["n_ref_words"] for r in rows)
    assert d["wer"][0] == pytest.approx(-3 * len(rows) / n)
    assert d["edit_f05"][0] == pytest.approx(f05(60, 0, 78) - f05(60, 12, 78))


def test_candidate_dataclass_still_has_what_decide_reads():
    c = Candidate("sub", 0, 0)
    assert c.rule == "" and c.accept is False
