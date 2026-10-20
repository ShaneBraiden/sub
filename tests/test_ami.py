"""tools/fetch_ami.py: AMI NXT word + topic layers -> one timed word stream with topic boundaries."""
from __future__ import annotations

import xml.etree.ElementTree as ET

from tools.fetch_ami import convert, split_of, topic_names

NS = 'xmlns:nite="http://nite.sourceforge.net/"'


def words_xml(meeting: str, spk: str, rows: list[tuple]) -> str:
    out = [f'<nite:root nite:id="{meeting}.{spk}.words" {NS}>']
    for k, (tag, text, st, en, punc) in enumerate(rows):
        times = f' starttime="{st}" endtime="{en}"' if st is not None else ""
        p = ' punc="true"' if punc else ""
        body = f">{text}</{tag}>" if text else "/>"
        out.append(f'<{tag} nite:id="{meeting}.{spk}.words{k}"{times}{p}{body}')
    return "\n".join(out + ["</nite:root>"])


def fixture(tmp_path):
    a = [("w", "Okay", 1.0, 1.3, False), ("w", ".", 1.3, 1.3, True),
         ("w", "Today", 1.5, 1.8, False), ("w", "budget", 1.8, 2.2, False), ("w", ".", 2.2, 2.2, True),
         ("vocalsound", "", 2.3, 2.5, False),
         ("w", "Then", 40.0, 40.3, False), ("w", "design", 40.3, 40.8, False), ("w", ".", 40.8, 40.8, True),
         ("w", "later", 45.0, 45.4, False)]                    # > 1 s pause: a new utterance without a full stop
    b = [("w", "Yeah", 2.0, 2.2, False), ("w", ".", 2.2, 2.2, True),
         ("w", "Colours", 41.0, 41.5, False), ("w", ".", 41.5, 41.5, True)]
    (tmp_path / "M.A.words.xml").write_text(words_xml("M", "A", a), encoding="utf-8")
    (tmp_path / "M.B.words.xml").write_text(words_xml("M", "B", b), encoding="utf-8")
    # topic 1 also claims B's "Colours" (a stray word of an earlier topic after topic 2 has started);
    # topic 2 holds a nested subtopic
    (tmp_path / "M.topic.xml").write_text(f"""<nite:root nite:id="M.topic" {NS}>
  <topic nite:id="M.topic.1" other_description="budget talk">
    <nite:pointer role="scenario_topic_type" href="default-topics.xml#id(top.4)"/>
    <nite:child href="M.A.words.xml#id(M.A.words0)..id(M.A.words5)"/>
    <nite:child href="M.B.words.xml#id(M.B.words0)..id(M.B.words1)"/>
    <nite:child href="M.B.words.xml#id(M.B.words2)..id(M.B.words3)"/>
  </topic>
  <topic nite:id="M.topic.2">
    <nite:pointer role="scenario_topic_type" href="default-topics.xml#id(top.28)"/>
    <topic nite:id="M.topic.3" other_description="colours">
      <nite:child href="M.A.words.xml#id(M.A.words6)..id(M.A.words7)"/>
    </topic>
    <nite:child href="M.A.words.xml#id(M.A.words8)"/>
    <nite:child href="M.A.words.xml#id(M.A.words9)"/>
  </topic>
</nite:root>""", encoding="utf-8")
    (tmp_path / "default-topics.xml").write_text(f"""<topicname nite:id="top.0" name="te" {NS}>
  <topicname nite:id="top.4" name="other"/>
  <topicname nite:id="top.2" name="Top level"><topicname nite:id="top.28" name="discussion"/></topicname>
</topicname>""", encoding="utf-8")
    return {"A": tmp_path / "M.A.words.xml", "B": tmp_path / "M.B.words.xml"}


def test_stream_topics_and_boundaries(tmp_path):
    files = fixture(tmp_path)
    names = topic_names(tmp_path / "default-topics.xml")
    rec = convert("M", files, tmp_path / "M.topic.xml", names)
    # utterances of both speakers ordered by start; punctuation attached; a 4 s pause starts an utterance
    assert [w[0] for w in rec["words"]] == ["Okay.", "Today", "budget.", "Yeah.", "Then", "design.", "Colours.", "later"]
    assert [w[3] for w in rec["words"]] == ["A", "A", "A", "B", "A", "A", "B", "A"]
    assert rec["words"][4][1:3] == [40.0, 40.3]
    # two top-level topics; the stray late word of topic 1 stays inside topic 2 (no extra boundary)
    assert [(t["index"], t["first_word"], t["start"]) for t in rec["topics"]] == [(1, 0, 1.0), (2, 4, 40.0)]
    assert [(t["label"], t["type"]) for t in rec["topics"]] == [("budget talk", "other"), ("discussion", "discussion")]
    assert rec["topics"][1]["end"] == 45.4 and rec["merged_topics"] == 0


def test_split_follows_the_meeting_list():
    def meeting(**attrs):
        return ET.Element("meeting", {k: v for k, v in attrs.items() if v is not None})
    assert split_of(meeting(type="scenario", seen_type="training")) == "train"
    assert split_of(meeting(type="scenario", seen_type="development")) == "dev"
    assert split_of(meeting(type="scenario", visibility="unseen")) == "test"
    assert split_of(meeting(type="nonscenario")) == "nonscenario"
