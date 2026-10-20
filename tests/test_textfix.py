import pytest

from subfix.subio import Word
from subfix.textfix.common import retokenize
from subfix.textfix.gec import guarded_gec, sentence_spans
from subfix.textfix.punct import add_mark, restore
from subfix.textfix.spell import SpellChecker
from subfix.textfix.truecase import mark_sentence_starts, truecase


def W(texts, meta=None):
    return [Word(t, k * 1.0, k * 1.0 + 0.5, 1.0, "sub", dict(meta or {})) for k, t in enumerate(texts)]


def test_guard_accepts_small_edits_rejects_paraphrase():
    src = "She go to school every day."
    assert guarded_gec(src, "She goes to school every day.", 0.25) == "She goes to school every day."
    assert guarded_gec(src, "She attends classes daily at her school.", 0.25) == src
    assert guarded_gec(src, "", 0.25) == src


def test_sentence_spans_break_on_punct_events_and_length():
    ws = W(["Hi.", "How", "are", "you?", "[music]", "fine"])
    ws[4].meta["event"] = True
    assert sentence_spans(ws, 40) == [(0, 1), (1, 4), (4, 5), (5, 6)]
    assert sentence_spans(W(["a"] * 5), 2) == [(0, 2), (2, 4), (4, 5)]


def test_retokenize_keeps_times_and_logs_edits():
    old = W(["She", "go", "to", "school."])
    new, edits = retokenize(old, ["She", "goes", "to", "school."], "gec")
    assert [w.text for w in new] == ["She", "goes", "to", "school."]
    assert new[1].start == old[1].start and new[1].source == "gec"
    assert edits == [{"stage": "gec", "op": "sub", "old": "go", "new": "goes", "index": 1}]
    new2, edits2 = retokenize(old, ["She", "will", "go", "to", "school."], "gec")
    assert new2[1].start is None and edits2[0]["op"] == "ins"


def test_truecase_caseless_uses_asr_hints():
    ws = W(["hello", "world.", "i", "went", "to", "paris."])
    ws[5].meta["case_hint"] = "Paris"
    out, _ = truecase(ws, caseless_source=True)
    assert " ".join(w.text for w in out) == "Hello world. I went to Paris."


def test_truecase_lowercases_orphaned_sentence_start():
    class Freq:
        def frequency(self, w):
            return 10 ** 9
    ws = W(["We", "went", "home.", "So", "tired."])
    mark_sentence_starts(ws)
    ws[2].text = "home,"           # punctuation model removed the period
    out, _ = truecase(ws, caseless_source=False, spell=Freq())
    assert " ".join(w.text for w in out) == "We went home, so tired."


class StubRestorer:
    def __init__(self, labels):
        self.labels = labels

    def predict(self, words, chunk_words=230, overlap=5):
        return [(lab, 0.9) for lab in self.labels]


def test_punct_fill_keeps_existing_marks():
    ws = W(["hello", "there!", "how", "are", "you"])
    out, _ = restore(ws, StubRestorer(["0", ".", "0", "0", "?"]), "fill")
    assert [w.text for w in out] == ["hello", "there!", "how", "are", "you?"]


def test_punct_replace_and_final_period():
    ws = W(["hello", "there,", "friend"])
    out, _ = restore(ws, StubRestorer(["0", "0", "0"]), "replace")
    assert [w.text for w in out] == ["hello", "there", "friend."]


def test_add_mark_before_closing_quote():
    assert add_mark('word"', ".") == 'word."'


@pytest.fixture(scope="module")
def spell():
    return SpellChecker()


def test_symspell_corrections(spell):
    assert spell.suggest("recieve") == "receive"
    assert spell.suggest("Tommorow,") == "Tomorrow,"
    assert spell.suggest("hello") is None
    assert spell.is_oov("alot") and not spell.is_oov("don't") and not spell.is_oov("42")


def test_symspell_skips_mid_sentence_names(spell):
    out, edits = spell.correct_words(W(["I", "met", "Grzegorz", "and", "recieved", "gifts."]))
    assert [w.text for w in out] == ["I", "met", "Grzegorz", "and", "received", "gifts."]
    assert len(edits) == 1


def test_audio_guard_blocks_edits_to_confirmed_words():
    from subfix.textfix.common import protector
    old = W(["compared", "with", "that", "which", "nothing"])
    for w in old:
        w.meta.update(asr_text=w.text, asr_prob=0.97)
    old[4].meta["asr_prob"] = 0.4                       # low-confidence word may still change
    new, edits = retokenize(old, ["Compared", "to", "that", "which", "is", "in", "the", "water."],
                            "gec", protect=protector(0.8))
    assert [w.text for w in new] == ["Compared", "with", "that", "which", "is", "in", "the", "water."]
    assert any(e["op"] == "blocked" for e in edits)
    assert new[0].text == "Compared"                    # casing-only change allowed
