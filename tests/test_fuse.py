from subfix import text as T
from subfix.align import align_words
from subfix.config import Config
from subfix.fuse import Fuser, passthrough, transplant
from subfix.lm import StubLM
from subfix.subio import Word


class StubSpell:
    VOCAB = {"i", "saw", "left", "it", "over", "there", "their", "the", "cat", "want", "to", "go", "home",
             "think", "so", "have", "a", "lot", "of", "time", "hello", "um", "car", "now", "parish"}

    def is_oov(self, tok):
        n = T.norm(tok)
        return bool(n) and n not in self.VOCAB

    def frequency(self, tok):
        return 10 ** 9 if T.norm(tok) in self.VOCAB else 0


def words(text, prob=0.95, src="sub", events=()):
    out = []
    for k, t in enumerate(text.split()):
        meta = {"cue": 0, "line": 0}
        if t in events:
            meta["event"] = True
        out.append(Word(t, k * 0.5, k * 0.5 + 0.4, prob if src == "asr" else 1.0, src, meta))
    return out


def fuse(sub_text, asr_text, prob=0.95, good=(), events=(), cfg=None):
    cfg = cfg or Config()
    sub, asr = words(sub_text, events=events), words(asr_text, prob, "asr")
    ops = align_words(sub, asr, cfg, use_time=False)
    res = Fuser(cfg, lm=StubLM(good_phrases=good), spell=StubSpell()).fuse(sub, asr, ops)
    return " ".join(w.text for w in res.words), res


def test_homophone_substitution_accepted_with_audio_evidence():
    out, res = fuse("I left it over their", "I left it over there", good=["over there"])
    assert out == "I left it over there"
    e = [e for e in res.edits if e["op"] == "sub"][0]
    assert e["decision"] == "accept" and e["score"] > 1.0


def test_low_confidence_substitution_rejected():
    out, _ = fuse("I left it over their", "I left it over there", prob=0.2)
    assert out == "I left it over their"


def test_duplicate_word_deleted():
    out, res = fuse("the the cat", "the cat")
    assert out == "the cat"
    assert any(e["rule"] == "duplicate" for e in res.edits)


def test_filler_never_inserted():
    out, _ = fuse("I think so", "I um think so", good=["i um think"])
    assert out == "I think so"


def test_dropped_function_word_restored():
    out, _ = fuse("I want go home", "I want to go home", good=["want to go"])
    assert out == "I want to go home"


def test_weak_insertion_rejected():
    out, _ = fuse("I want go home", "I want to go home", prob=0.3, good=["want to go"])
    assert out == "I want go home"


def test_split_spelling_fixed_only_when_oov():
    out, _ = fuse("I have alot of time", "I have a lot of time")
    assert out == "I have a lot of time"


def test_sound_event_kept():
    out, _ = fuse("[music] hello", "hello", events=("[music]",))
    assert out == "[music] hello"


def test_punctuation_moves_behind_inserted_word():
    out, _ = fuse("I want go.", "I want to go", good=["want to go"])
    assert out == "I want to go."


def test_transplant_keeps_subtitle_affixes_and_casing():
    assert transplant("There,", "their") == "Their,"
    assert transplant('"parish."', "Paris") == '"Paris."'
    assert transplant("the", "A") == "a"


def test_passthrough_keeps_words():
    sub = words("hello there")
    assert [w.text for w in passthrough(sub)] == ["hello", "there"]
