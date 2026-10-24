"""HTTP API of app.backend with a fake pipeline runner (no models needed).

The fake builds a real RunResult from the uploaded subtitles (Mode A) or a canned transcript
(Mode B), so build_result(), the diff and the SRT/VTT export are the real code paths.
"""
from __future__ import annotations

import io
import json
import threading
import time

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.backend.app import create_app
from app.backend.runner import build_result
from app.backend.schemas import JobOptions, option_defaults, to_overrides
from app.backend.settings import Settings
from app.backend.store import JobStore
from subfix.asr import AsrResult
from subfix.chapters import Chapter
from subfix.config import Config
from subfix.diarize import Diarization, Turn, assign_speakers, speaker_report
from subfix.export import write_outputs
from subfix.pipeline import RunResult
from subfix.subio import Cue, Word, cues_to_words, read_subs

SRT = """1
00:00:00,500 --> 00:00:02,000
Hello their, and welcome.

2
00:00:02,500 --> 00:00:04,000
Today we talk about it.
"""


def wav_bytes(seconds: float = 4.5, sr: int = 16000) -> bytes:
    t = np.arange(int(seconds * sr)) / sr
    buf = io.BytesIO()
    sf.write(buf, (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def fake_result(wav: str, subs, diarized: bool = False, chapters: bool = False) -> RunResult:
    cfg = Config.load()
    asr_words = [Word(t, 0.6 + 0.4 * k, 0.9 + 0.4 * k, 0.95, "asr") for k, t in
                 enumerate("Hello there, and welcome. Today we talk about it.".split())]
    asr = AsrResult(asr_words, [{"start": 0.6, "end": 4.0, "text": " Hello there, and welcome. Today we talk about it."}],
                    {"model": "fake"})
    final = [w.copy(source="fused") for w in asr_words]
    if subs:
        orig_cues = read_subs(subs)
        orig_words = cues_to_words(orig_cues)
        final[1].meta.update(orig_text="their", asr_text="there,", asr_prob=0.95, rule="phonetic")
        mode = "A"
    else:
        orig_cues = orig_words = None
        mode = "B"
    diar = None
    if diarized:      # host says hello, guest answers
        diar = Diarization.build([Turn(0.5, 2.15, "host"), Turn(2.15, 4.2, "guest")])
        final = assign_speakers(final, diar)
        cues = [Cue(1, 0.6, 2.1, "SPEAKER_00: Hello there, and welcome.", final[:4], ["SPEAKER_00"], ["SPEAKER_00: "]),
                Cue(2, 2.2, 4.0, "SPEAKER_01: Today we talk about it.", final[4:], ["SPEAKER_01"], ["SPEAKER_01: "])]
    else:
        cues = [Cue(1, 0.6, 2.1, "Hello there, and welcome.", final[:4]),
                Cue(2, 2.2, 4.0, "Today we talk about it.", final[4:])]
    return RunResult(mode=mode, cues=cues, words=final, orig_cues=orig_cues, orig_words=orig_words, asr=asr,
                     sync=None, edits=[{"stage": "fusion", "op": "sub", "decision": "accept", "old": "their",
                                        "new": "there,", "prob": np.float32(0.95)}],
                     stage_times={"asr": 0.1, "timing": 0.05}, audio_duration=4.5, wav=wav, config=cfg.to_dict(),
                     diarization=diar, speakers=speaker_report(final, cues, diar, 4.5) if diar else None,
                     chapters=[Chapter(1, 0.0, 2.15, "Greeting", ["welcome"], "Hello there, and welcome.", (0, 4)),
                               Chapter(2, 2.15, 4.5, "Topic", ["talk"], "Today we talk about it.", (4, 9), 0.4)]
                     if chapters else None)


class FakeRunner:
    def __init__(self, gate: threading.Event | None = None, fail: bool = False):
        self.gate = gate
        self.fail = fail
        self.calls: list[dict] = []

    def defaults(self):
        return {"max_cps": 17.0, "max_chars": 42, "punctuation": True}

    def loaded_models(self):
        return []

    def device_info(self):
        return {"device": "cpu", "name": None}

    def run(self, job, media, subs, out_dir, progress):
        self.calls.append(job)
        for key, frac in (("preprocess", 0.02), ("asr", 0.05), ("punct", 0.45)):
            progress(key, frac)
            if self.gate is not None:
                assert self.gate.wait(10)
        if self.fail:
            raise RuntimeError("boom")
        progress("timing", 0.75)
        res = fake_result(str(media), subs, diarized=bool(job["options"].get("diarize")),
                          chapters=bool(job["options"].get("chapters")))
        progress("export", 0.97)
        paths = write_outputs(res, out_dir, "clip.corrected")
        result = build_result(res, Config.load())
        result["files"] = {k: p.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for k, p in paths.items()}
        result["wav"] = str(media)
        return result


@pytest.fixture
def make_client(tmp_path):
    clients = []

    def make(runner=None, **settings):
        s = Settings(data_dir=tmp_path / "server", frontend_dir=tmp_path / "dist", **settings)
        s.data_dir.mkdir(parents=True, exist_ok=True)
        client = TestClient(create_app(s, runner or FakeRunner()))
        client.__enter__()
        clients.append(client)
        return client

    yield make
    for c in clients:
        c.__exit__(None, None, None)


def submit(client, subs: str | None = None, options: dict | None = None, name: str = "clip.wav"):
    files = {"media": (name, wav_bytes(), "audio/wav")}
    if subs is not None:
        files["subs"] = ("clip.srt", subs.encode(), "application/x-subrip")
    return client.post("/api/jobs", files=files, data={"options": json.dumps(options or {})})


def wait_status(client, job_id, statuses=("done", "failed", "cancelled"), timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        snap = client.get(f"/api/jobs/{job_id}").json()
        if snap["status"] in statuses:
            return snap
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} stuck in {snap['status']}")


def test_build_result_is_plain_json():
    res = fake_result("x.wav", None)
    out = build_result(res, Config.load())
    json.dumps(out, allow_nan=False)     # numpy scalars and NaN are converted
    assert out["mode"] == "B"
    assert [c["text"] for c in out["cues"]] == ["Hello there, and welcome.", "Today we talk about it."]
    assert out["cues"][0]["words"][0] == ["Hello", 0.6, 0.9]
    assert out["original_cues"][0]["text"].startswith("Hello there")
    assert out["asr_raw"]["text"] == " ".join(c["text"] for c in out["original_cues"])
    assert len(out["asr_raw"]["segments"]) == len(out["original_cues"])
    assert out["edits"][0]["prob"] == pytest.approx(0.95, abs=1e-6)


def test_options_and_system(make_client):
    c = make_client()
    opts = c.get("/api/options").json()
    assert opts["defaults"]["max_cps"] == 17.0
    keys = {f["key"] for f in opts["fields"]}
    assert {"max_cps", "segmentation", "terms"} <= keys
    seg = next(f for f in opts["fields"] if f["key"] == "segmentation")
    assert seg["choices"] == ["dp", "greedy", "keep"]
    assert opts["stages"][0]["key"] == "preprocess"
    assert "diarize" in [s["key"] for s in opts["stages"]]
    assert opts["diarization"] == {"model": False}     # the fake runner cannot diarize
    labels = next(f for f in opts["fields"] if f["key"] == "speaker_labels")
    assert labels["choices"] == ["every", "change", "none"]
    sysinfo = c.get("/api/system").json()
    assert sysinfo["device"]["device"] == "cpu" and sysinfo["queued"] == 0


def test_mode_b_roundtrip(make_client):
    c = make_client()
    r = submit(c)
    assert r.status_code == 201, r.text
    job = r.json()
    assert job["mode"] == "B" and job["media_name"] == "clip.wav"
    snap = wait_status(c, job["id"])
    assert snap["status"] == "done", snap
    assert [s["key"] for s in snap["stages"]] == ["preprocess", "asr", "punct", "timing", "export"]
    assert snap["summary"]["cues"] == 2

    res = c.get(f"/api/jobs/{job['id']}/result").json()
    assert res["mode"] == "B" and "wav" not in res
    assert res["rows"][0]["new"][0]["t"] == "Hello"
    assert set(res["files"]) == {"srt", "vtt", "json", "html"}

    srt = c.get(f"/api/jobs/{job['id']}/files/srt")
    assert srt.status_code == 200 and "Today we talk about it." in srt.text
    assert "attachment" in srt.headers["content-disposition"]
    assert c.get(f"/api/jobs/{job['id']}/files/nope").status_code == 404

    media = c.get(f"/api/jobs/{job['id']}/media", headers={"Range": "bytes=0-99"})
    assert media.status_code == 206 and len(media.content) == 100

    wf = c.get(f"/api/jobs/{job['id']}/waveform?bins=200").json()
    assert len(wf["peaks"]) == 200 and max(wf["peaks"]) == 1.0 and wf["duration"] == pytest.approx(4.5)
    assert "job" in c.get(f"/api/jobs/{job['id']}/log").text

    listing = c.get("/api/jobs").json()
    assert listing["total"] == 1 and listing["jobs"][0]["id"] == job["id"]


def test_mode_a_and_manual_edits(make_client):
    c = make_client()
    job = submit(c, SRT).json()
    assert job["mode"] == "A" and job["subs_name"] == "clip.srt"
    assert wait_status(c, job["id"])["status"] == "done"
    res = c.get(f"/api/jobs/{job['id']}/result").json()
    assert res["mode"] == "A" and res["edited_cues"] is None
    assert any(t["k"] == "sub" for t in res["rows"][0]["new"])
    assert res["original_cues"][0]["text"] == "Hello their, and welcome."

    cues = [{"start": 2.2, "end": 4.0, "text": "Today we talk\nabout it."},
            {"start": 0.6, "end": 2.1, "text": "Hello there, and welcome!"}]
    r = c.put(f"/api/jobs/{job['id']}/cues", json={"cues": cues})
    assert r.status_code == 200, r.text
    res = c.get(f"/api/jobs/{job['id']}/result").json()
    assert [x["text"] for x in res["edited_cues"]] == ["Hello there, and welcome!", "Today we talk\nabout it."]
    edited = c.get(f"/api/jobs/{job['id']}/files/edited_srt").text
    assert "welcome!" in edited and "about it." in edited
    assert c.get(f"/api/jobs/{job['id']}").json()["edited"]

    bad = c.put(f"/api/jobs/{job['id']}/cues", json={"cues": [{"start": 3, "end": 2, "text": "x"}]})
    assert bad.status_code == 422
    assert c.delete(f"/api/jobs/{job['id']}/cues").status_code == 204
    assert c.get(f"/api/jobs/{job['id']}/files/edited_srt").status_code == 404
    assert c.get(f"/api/jobs/{job['id']}/result").json()["edited_cues"] is None


def test_upload_validation(make_client):
    c = make_client()
    assert submit(c, name="clip.txt").status_code == 400
    r = c.post("/api/jobs", files={"media": ("a.wav", wav_bytes()), "subs": ("a.ass", b"x")})
    assert r.status_code == 400
    assert submit(c, subs="not a subtitle").status_code == 400
    assert submit(c, options={"max_cps": 99}).status_code == 422
    assert submit(c, options={"unknown": 1}).status_code == 422
    assert submit(c, options={"segmentation": "keep"}).status_code == 422
    assert c.get("/api/jobs").json()["total"] == 0          # nothing half-created
    assert c.get("/api/jobs/nope").status_code == 404


def test_options_reach_the_runner(make_client):
    runner = FakeRunner()
    c = make_client(runner)
    job = submit(c, options={"max_cps": 20, "grammar_model": False, "terms": " Chennai "}).json()
    wait_status(c, job["id"])
    assert runner.calls[0]["options"] == {"max_cps": 20.0, "grammar_model": False, "terms": "Chennai"}

    r = c.post(f"/api/jobs/{job['id']}/rerun", json={"max_cps": 15})
    assert r.status_code == 201, r.text
    new = r.json()
    assert new["id"] != job["id"] and new["media_name"] == "clip.wav"
    assert wait_status(c, new["id"])["status"] == "done"
    assert runner.calls[1]["options"] == {"max_cps": 15.0, "terms": ""}


def test_queue_cancel_and_delete(make_client):
    gate = threading.Event()
    c = make_client(FakeRunner(gate=gate))
    first = submit(c).json()
    second = submit(c).json()
    wait_status(c, first["id"], ("running",))
    snap2 = c.get(f"/api/jobs/{second['id']}").json()
    assert snap2["status"] == "queued" and snap2["queue_position"] == 1
    assert c.delete(f"/api/jobs/{first['id']}").status_code == 409

    assert c.post(f"/api/jobs/{second['id']}/cancel").json()["status"] == "cancelled"
    running = c.post(f"/api/jobs/{first['id']}/cancel").json()
    assert running["cancelling"]
    gate.set()
    assert wait_status(c, first["id"])["status"] == "cancelled"
    assert c.post(f"/api/jobs/{first['id']}/cancel").status_code == 409
    assert c.get(f"/api/jobs/{first['id']}/result").status_code == 409

    assert c.delete(f"/api/jobs/{first['id']}").status_code == 204
    assert c.get(f"/api/jobs/{first['id']}").status_code == 404
    assert c.get("/api/jobs").json()["total"] == 1


def test_failure_is_reported(make_client):
    c = make_client(FakeRunner(fail=True))
    job = submit(c).json()
    snap = wait_status(c, job["id"])
    assert snap["status"] == "failed" and snap["error"] == "RuntimeError: boom"
    assert "Traceback" in c.get(f"/api/jobs/{job['id']}/log").text


def test_events_stream_until_done(make_client):
    c = make_client()
    job = submit(c).json()
    events = []
    with c.stream("GET", f"/api/jobs/{job['id']}/events") as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    assert events and events[-1]["status"] == "done"


def test_interrupted_jobs_are_requeued(tmp_path, make_client):
    data = tmp_path / "server"
    (data / "jobs" / "old").mkdir(parents=True)
    (data / "jobs" / "old" / "media.wav").write_bytes(wav_bytes())
    store = JobStore(data / "jobs.sqlite3")
    store.create(id="old", status="running", media_name="old.wav", media_file="media.wav", mode="B", options={})
    store.close()
    c = make_client()
    assert wait_status(c, "old")["status"] == "done"


def test_frontend_hosting(tmp_path, make_client):
    c = make_client()
    assert "not built" in c.get("/").text
    assert c.get("/api/nothing").status_code == 404
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<div id=root></div>", encoding="utf-8")
    (dist / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    assert c.get("/jobs/abc").text == "<div id=root></div>"          # SPA fallback
    js = c.get("/assets/app.js")
    assert js.text == "console.log(1)" and "immutable" in js.headers["cache-control"]
    assert c.get("/../jobs.sqlite3").text == "<div id=root></div>"   # no path traversal


def test_speaker_options_map_to_the_config():
    cfg = Config.load()
    d = option_defaults(cfg)
    assert d["diarize"] is False and d["num_speakers"] == 0 and d["speaker_labels"] == "every"
    over = to_overrides(JobOptions(diarize=True, num_speakers=3, speaker_labels="change"), cfg)
    assert over == {"diarize.enabled": True, "diarize.num_speakers": 3, "diarize.label": "change"}
    assert cfg.override(over).diarize.num_speakers == 3
    with pytest.raises(ValueError):
        JobOptions(num_speakers=50)


def test_diarized_job_result_downloads_and_edits(make_client):
    c = make_client()
    job = submit(c, options={"diarize": True, "num_speakers": 2}).json()
    assert wait_status(c, job["id"])["status"] == "done"
    res = c.get(f"/api/jobs/{job['id']}/result").json()
    sp = res["speakers"]
    assert [r["speaker"] for r in sp["speakers"]] == ["SPEAKER_00", "SPEAKER_01"]
    assert sp["totals"]["speaker_changes"] == 1 and len(sp["turns"]) == 2
    assert res["cues"][0]["speakers"] == ["SPEAKER_00"] and res["cues"][0]["labels"] == ["SPEAKER_00: "]
    assert res["rows"][1]["speakers"] == ["SPEAKER_01"] and res["rows"][1]["overlap"] is False
    assert res["rows"][0]["cps"] == pytest.approx(22 / 1.5)          # the label is not counted
    assert {"rttm", "speakers"} <= set(res["files"])
    rttm = c.get(f"/api/jobs/{job['id']}/files/rttm")
    assert rttm.status_code == 200 and rttm.text.startswith("SPEAKER clip.corrected 1 0.500")
    assert c.get(f"/api/jobs/{job['id']}/files/speakers").json()["totals"]["speakers"] == 2

    # fixing a wrong speaker by editing its label moves the voice tag too
    cues = [{"start": 0.6, "end": 2.1, "text": "SPEAKER_00: Hello there, and welcome."},
            {"start": 2.2, "end": 4.0, "text": "SPEAKER_00: Today we talk\nabout it."}]
    assert c.put(f"/api/jobs/{job['id']}/cues", json={"cues": cues}).status_code == 200
    edited = c.get(f"/api/jobs/{job['id']}/result").json()["edited_cues"]
    assert edited[1]["speakers"] == ["SPEAKER_00", "SPEAKER_00"] and edited[1]["labels"] == ["SPEAKER_00: ", ""]
    vtt = c.get(f"/api/jobs/{job['id']}/files/edited_vtt").text
    assert "<v SPEAKER_00>SPEAKER_00: Today we talk\nabout it.</v>" in vtt


def test_plain_job_has_no_speaker_data(make_client):
    c = make_client()
    job = submit(c).json()
    wait_status(c, job["id"])
    res = c.get(f"/api/jobs/{job['id']}/result").json()
    assert res["speakers"] is None and "speakers" not in res["cues"][0] and "speakers" not in res["rows"][0]
    assert c.get(f"/api/jobs/{job['id']}/files/rttm").status_code == 404


def test_chapter_options_map_to_the_config():
    cfg = Config.load()
    d = option_defaults(cfg)
    assert d["chapters"] is True and d["num_chapters"] == 0 and d["min_chapter"] == cfg.chapters.min_duration
    assert d["chapter_mode"] == "talk"
    over = to_overrides(JobOptions(chapters=True, chapter_mode="meeting", num_chapters=4, min_chapter=90), cfg)
    assert over == {"chapters.enabled": True, "chapters.mode": "meeting", "chapters.num_chapters": 4,
                    "chapters.min_duration": 90.0}
    assert cfg.override(over).chapters.mode == "meeting"
    with pytest.raises(ValueError):
        JobOptions(chapter_mode="lecture")
    assert cfg.override(over).chapters.num_chapters == 4
    with pytest.raises(ValueError):
        JobOptions(min_chapter=5)


def test_chapters_in_the_result_and_downloads(make_client):
    c = make_client()
    job = submit(c, options={"chapters": True}).json()
    snap = wait_status(c, job["id"])
    assert snap["status"] == "done" and snap["summary"]["chapters"] == 2
    res = c.get(f"/api/jobs/{job['id']}/result").json()
    assert [(ch["title"], ch["start"], ch["summary"]) for ch in res["chapters"]] == [
        ("Greeting", 0.0, "Hello there, and welcome."), ("Topic", 2.15, "Today we talk about it.")]
    assert {"chapters_vtt", "chapters_txt", "ffmetadata", "chapters_json"} <= set(res["files"])
    vtt = c.get(f"/api/jobs/{job['id']}/files/chapters_vtt")
    assert vtt.status_code == 200 and vtt.headers["content-type"].startswith("text/vtt")
    assert "00:00:02.150 --> 00:00:04.500\nTopic" in vtt.text
    assert c.get(f"/api/jobs/{job['id']}/files/chapters_txt").text == "0:00 Greeting\n0:02 Topic\n"
    ff = c.get(f"/api/jobs/{job['id']}/files/ffmetadata")
    assert ff.headers["content-type"].startswith("text/plain") and "START=2150\nEND=4500\ntitle=Topic" in ff.text
    assert c.get(f"/api/jobs/{job['id']}/files/chapters_json").json()[1]["keywords"] == ["talk"]


def test_job_without_chapters(make_client):
    c = make_client()
    job = submit(c).json()
    wait_status(c, job["id"])
    res = c.get(f"/api/jobs/{job['id']}/result").json()
    assert res["chapters"] is None and "chapters_vtt" not in res["files"]
    assert c.get(f"/api/jobs/{job['id']}/files/ffmetadata").status_code == 404
