"""AMI meeting topic annotations: real topic boundaries for the chapter-marker evaluation.

Downloads the AMI manual annotations (ami_public_manual_1.6.2.zip, 22.9 MB, CC BY 4.0, University
of Edinburgh) into data/raw/ami/, extracts the word and topic layers, and writes one JSON per
annotated meeting (139) to data/ami/meetings/, plus data/ami/manifest.json.

Per meeting, the transcript becomes one timed word stream like an ASR output: each speaker's words
(manual transcription with forced-aligned times) are cut into utterances at sentence ends and at
pauses over 1 s, punctuation tokens are attached to the word before them, and the utterances of all
speakers are ordered by start time. Every word carries the top-level topic it belongs to (AMI topics
are word ranges in each speaker's file, `id(a)..id(b)` in file order, nested for subtopics); a word
outside every topic takes the topic of the word before it. Topics are numbered in order of their
start, and the topic sequence along the stream is made monotone (running maximum): a topic starts at
the first word labelled with it or a later topic, so a word of an earlier topic that turns up after a
later one has begun adds no boundary, and a topic overtaken that way before its own first word is
merged into the later one.

The split comes from corpusResources/meetings.xml: scenario meetings `train` (seen/training, 95),
`dev` (seen/development, 20) and `test` (unseen, 20), and `nonscenario` (4).

  python -I tools/fetch_ami.py            # download (resumable) + convert
"""
from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
SIZE = 22887865
RAW = ROOT / "data" / "raw" / "ami"
ZIP = RAW / "ami_public_manual_1.6.2.zip"
SRC = RAW / "ami_public_manual_1.6.2"
OUT = ROOT / "data" / "ami"
KEEP = ("topics/", "words/")
KEEP_FILES = ("ontologies/default-topics.xml", "corpusResources/meetings.xml", "LICENCE.txt", "00README_MANUAL.txt")
NITE = "{http://nite.sourceforge.net/}"
_HREF = re.compile(r"^(?P<file>[^#]+)#id\((?P<a>[^)]+)\)(?:\.\.id\((?P<b>[^)]+)\))?$")
SENT_END = (".", "?", "!")
UTT_PAUSE = 1.0


# ------------------------------------------------------------------ download + extract
def fetch() -> None:
    from fetch_data import download
    if not ZIP.exists() or ZIP.stat().st_size != SIZE:
        download([URL], ZIP, size=SIZE)
    if (SRC / "corpusResources" / "meetings.xml").exists():
        return
    with zipfile.ZipFile(ZIP) as z:
        if z.testzip() is not None:
            raise SystemExit(f"{ZIP} is corrupt: delete it and run again")
        keep = [i for i in z.infolist() if not i.is_dir()
                and (i.filename.startswith(KEEP) or i.filename in KEEP_FILES)]
        if any(i.filename.startswith(("/", "\\")) or ".." in i.filename or i.file_size > 50 << 20 for i in keep):
            raise SystemExit("unexpected member names or sizes in the AMI archive")
        z.extractall(SRC, members=keep)


# ------------------------------------------------------------------ parsing
def read_words(path: Path) -> tuple[dict[str, int], list[dict]]:
    """(element id -> position in file order, every element as {id, w?, text, start, end, punc})."""
    order, elems = {}, []
    for k, el in enumerate(ET.parse(path).getroot()):
        eid = el.get(f"{NITE}id")
        order[eid] = k
        st, en = el.get("starttime"), el.get("endtime")
        elems.append({"id": eid, "w": el.tag == "w", "text": (el.text or "").strip(),
                      "start": float(st) if st else None, "end": float(en) if en else None,
                      "punc": el.get("punc") == "true"})
    return order, elems


def topic_names(path: Path) -> dict[str, str]:
    return {el.get(f"{NITE}id"): el.get("name") for el in ET.parse(path).getroot().iter("topicname")}


def _ranges(topic: ET.Element) -> list[tuple[str, str, str]]:
    """(speaker, first id, last id) of every word range under a topic, subtopics included."""
    out = []
    for child in topic.iter(f"{NITE}child"):
        m = _HREF.match(child.get("href", ""))
        if not m:
            continue
        speaker = m["file"].split(".")[1]
        out.append((speaker, m["a"], m["b"] or m["a"]))
    return out


def convert(meeting: str, files: dict[str, Path], topic_file: Path, names: dict[str, str]) -> dict:
    words = {spk: read_words(p) for spk, p in files.items()}
    root = ET.parse(topic_file).getroot()
    labels: dict[tuple[str, int], int] = {}          # (speaker, element position) -> top-level topic
    tops = []
    for t, topic in enumerate(e for e in root if e.tag == "topic"):
        ptr = topic.find(f"{NITE}pointer")
        kind = names.get(ptr.get("href", "").split("#id(")[-1].rstrip(")")) if ptr is not None else None
        desc = topic.get("other_description")
        tops.append({"label": desc or kind or "", "type": kind})
        for spk, a, b in _ranges(topic):
            order = words[spk][0]
            for k in range(order[a], order[b] + 1):
                labels[(spk, k)] = t

    # utterances per speaker: [word dicts] cut at sentence ends and long pauses
    utts = []
    for spk, (_, elems) in words.items():
        cur, last_end = [], None
        for k, e in enumerate(elems):
            if not e["w"] or not e["text"]:
                continue
            if e["punc"]:
                if cur:
                    cur[-1]["text"] += e["text"]
                    if e["text"] in SENT_END:
                        utts.append(cur)
                        cur = []
                continue
            start = e["start"] if e["start"] is not None else last_end
            if start is None:
                continue
            end = e["end"] if e["end"] is not None else start
            if cur and last_end is not None and start - last_end > UTT_PAUSE:
                utts.append(cur)
                cur = []
            cur.append({"text": e["text"], "start": start, "end": max(end, start), "spk": spk,
                        "topic": labels.get((spk, k))})
            last_end = max(end, start)
        if cur:
            utts.append(cur)
    utts.sort(key=lambda u: (u[0]["start"], u[0]["spk"]))
    stream = [w for u in utts for w in u]

    # topics in order of their first word in time; words outside every topic inherit the previous one
    first_time = {}
    for w in stream:
        if w["topic"] is not None:
            first_time[w["topic"]] = min(first_time.get(w["topic"], 1e18), w["start"])
    ordered = sorted(first_time, key=first_time.get)
    rank = {t: r for r, t in enumerate(ordered)}
    # running maximum of the topic rank along the stream; a segment = a run of one maximum
    seg_of, high = [], 0
    for w in stream:
        if w["topic"] is not None:
            high = max(high, rank[w["topic"]])
        seg_of.append(high)
    starts = [i for i in range(len(stream)) if i == 0 or seg_of[i] != seg_of[i - 1]]
    topics = []
    for j, i in enumerate(starts):
        end_i = starts[j + 1] if j + 1 < len(starts) else len(stream)
        info = tops[ordered[seg_of[i]]] if ordered else {"label": "", "type": None}
        topics.append({"index": j + 1, "first_word": i, "start": stream[i]["start"],
                       "end": stream[end_i - 1]["end"], "label": info["label"], "type": info["type"]})
    return {"id": meeting, "words": [[w["text"], round(w["start"], 3), round(w["end"], 3), w["spk"]] for w in stream],
            "topics": topics, "merged_topics": len(ordered) - len(topics)}


def split_of(meeting: ET.Element) -> str:
    if meeting.get("type") != "scenario":
        return "nonscenario"
    return {"training": "train", "development": "dev"}.get(meeting.get("seen_type") or "", "test")


def main() -> int:
    fetch()
    names = topic_names(SRC / "ontologies" / "default-topics.xml")
    meetings = {m.get("observation"): m for m in ET.parse(SRC / "corpusResources" / "meetings.xml").getroot().iter("meeting")}
    (OUT / "meetings").mkdir(parents=True, exist_ok=True)
    manifest = []
    for topic_file in sorted((SRC / "topics").glob("*.topic.xml")):
        mid = topic_file.name.split(".")[0]
        files = {p.name.split(".")[1]: p for p in sorted((SRC / "words").glob(f"{mid}.*.words.xml"))}
        rec = convert(mid, files, topic_file, names)
        meta = meetings.get(mid)
        rec["split"] = split_of(meta) if meta is not None else "nonscenario"
        rec["duration"] = float(meta.get("duration")) if meta is not None and meta.get("duration") else rec["words"][-1][2]
        (OUT / "meetings" / f"{mid}.json").write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        manifest.append({"id": mid, "split": rec["split"], "duration": rec["duration"], "words": len(rec["words"]),
                         "topics": len(rec["topics"]), "speakers": len(files)})
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    by = {}
    for m in manifest:
        by.setdefault(m["split"], []).append(m)
    for s, ms in sorted(by.items()):
        print(f"{s}: {len(ms)} meetings, {sum(m['duration'] for m in ms) / 3600:.1f} h, "
              f"{sum(m['topics'] for m in ms) / len(ms):.1f} topics per meeting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
