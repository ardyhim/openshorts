"""Parity tests for the openshorts-transcript agent skill.

The skill must emit the same contracts the app emits. These tests import the
skill script by path and compare its export with
manual_clip_selection.build_transcript_export, and its YouTube helpers with
free_tools. That is the drift alarm: change one side without the other and the
suite goes red.
"""
import importlib.util
import io
import json
from pathlib import Path

import pytest

import manual_clip_selection

try:  # free_tools imports FastAPI, absent from the minimal CI install.
    import free_tools
except Exception:  # noqa: BLE001 - the YouTube parity tests skip instead
    free_tools = None

SCRIPT = (Path(__file__).resolve().parents[1]
          / "skills" / "openshorts-transcript" / "scripts"
          / "openshorts_transcript.py")


def _load_skill():
    spec = importlib.util.spec_from_file_location("openshorts_transcript", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


skill = _load_skill()


@pytest.fixture(autouse=True)
def _clean_clip_env(monkeypatch):
    for name in ("CLIP_MIN_SECONDS", "CLIP_MAX_SECONDS",
                 "CLIP_TARGET_MIN", "CLIP_TARGET_MAX"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def transcript():
    return {
        "text": "Halo semuanya. Ini rahasia besar.",
        "language": "id",
        "segments": [
            {"start": 0.0, "end": 12.5, "text": "Halo semuanya.",
             "words": [{"word": " Halo", "start": 0.0, "end": 0.5},
                       {"word": " semuanya.", "start": 0.5, "end": 1.2}]},
            {"start": 12.5, "end": 42.0, "text": "Ini rahasia besar.",
             "words": [{"word": " Ini", "start": 12.5, "end": 12.8},
                       {"word": " rahasia", "start": 12.8, "end": 13.4},
                       {"word": " besar.", "start": 13.4, "end": 14.0}]},
        ],
    }


SRT = """1
00:00:01,000 --> 00:00:03,500
<i>Hello</i> world,
this is line two.

2
00:00:04,000 --> 00:00:06,250
Second cue.
"""

VTT = """WEBVTT

NOTE this is a comment

00:01.000 --> 00:03.500 line:0 position:10%
<v Roger>Hello</v> <c.yellow>world</c>

00:04.000 --> 00:06.250
Second cue.
"""


class TestSubtitleParsing:
    def test_srt_multiline_and_markup(self):
        source = skill.load_subtitle(SRT, language="en", filename="sample.srt")
        segments = source["transcript"]["segments"]
        assert [s["text"] for s in segments] == [
            "Hello world, this is line two.", "Second cue."]
        assert segments[0]["start"] == 1.0
        assert segments[0]["end"] == 3.5
        assert source["duration"] == 6.25

    def test_srt_words_keep_the_app_leading_space_convention(self):
        source = skill.load_subtitle(SRT, language="en", filename="sample.srt")
        words = source["transcript"]["segments"][0]["words"]
        assert [w["word"] for w in words] == [
            " Hello", " world,", " this", " is", " line", " two."]
        assert all(w["word"].startswith(" ") for w in words)
        assert all(w["end"] > w["start"] for w in words)
        assert words[0]["start"] == 1.0
        assert words[-1]["end"] == 3.5

    def test_vtt_skips_notes_and_strips_cue_tags(self):
        source = skill.load_subtitle(VTT, language="en", filename="sample.vtt")
        segments = source["transcript"]["segments"]
        assert [s["text"] for s in segments] == [
            "Hello world", "Second cue."]
        assert segments[0]["start"] == 1.0
        assert segments[0]["end"] == 3.5

    def test_timestamped_txt_uses_next_start(self):
        text = "[00:01] First line\n[00:03] Second line\n"
        source = skill.load_subtitle(text, filename="sample.txt", duration=10)
        segments = source["transcript"]["segments"]
        assert [(s["start"], s["end"], s["text"]) for s in segments] == [
            (1.0, 3.0, "First line"), (3.0, 10.0, "Second line")]

    def test_timestamped_txt_without_last_end_needs_duration(self):
        text = "[00:01] Only line\n"
        source = skill.load_subtitle(text, filename="sample.txt")
        assert source["implicit_last_end"] is True
        with pytest.raises(skill.TranscriptError) as exc:
            skill.to_export(source, type("Args", (), {
                "duration": None, "job_id": None, "language": None,
                "min_seconds": None, "max_seconds": None})())
        assert exc.value.code == "ambiguous_duration"
        with pytest.raises(skill.TranscriptError) as exc:
            skill.to_pipeline(source)
        assert exc.value.code == "ambiguous_duration"

    def test_srt_content_with_a_txt_extension_is_sniffed(self):
        source = skill.load_subtitle(SRT, filename="captions.txt", language="en")
        assert [s["text"] for s in source["transcript"]["segments"]] == [
            "Hello world, this is line two.", "Second cue."]

    def test_plain_text_only_becomes_text(self):
        source = skill.load_subtitle("Just words, no timestamps.",
                                     filename="sample.txt")
        assert source["transcript"]["segments"] == []
        assert skill.to_text(source) == "Just words, no timestamps."
        with pytest.raises(skill.TranscriptError) as exc:
            skill.to_srt(source)
        assert exc.value.code == "no_timestamps"


class TestExportParity:
    def test_matches_manual_clip_selection(self, transcript):
        ours = skill.build_transcript_export(
            transcript, 42.0, job_id="job-1", language="id")
        theirs = manual_clip_selection.build_transcript_export(
            transcript, 42.0, job_id="job-1", language="id")
        ours.pop("generated_at")
        theirs.pop("generated_at")
        assert ours == theirs

    def test_matches_on_a_long_transcript(self):
        segments = []
        for index in range(40):
            start = index * 4.0
            segments.append({
                "start": start,
                "end": start + 3.5,
                "text": f"This is transcript segment number {index}.",
            })
        transcript = {"text": " ".join(s["text"] for s in segments),
                      "language": "en", "segments": segments}
        ours = skill.build_transcript_export(transcript, 160.0)
        theirs = manual_clip_selection.build_transcript_export(transcript, 160.0)
        ours.pop("generated_at")
        theirs.pop("generated_at")
        assert ours == theirs
        assert len(ours["windows"]) == len(theirs["windows"]) > 1

    def test_clip_count_targets_match(self):
        for count in (1, 2, 3, 5, 10, 24):
            assert skill.clip_count_targets(count) == (
                manual_clip_selection.clip_count_targets(count))

    def test_matches_with_custom_bounds(self, transcript, monkeypatch):
        monkeypatch.setenv("CLIP_MIN_SECONDS", "20")
        monkeypatch.setenv("CLIP_MAX_SECONDS", "35")
        ours = skill.build_transcript_export(transcript, 42.0)
        theirs = manual_clip_selection.build_transcript_export(transcript, 42.0)
        ours.pop("generated_at")
        theirs.pop("generated_at")
        assert ours == theirs
        assert ours["clip_duration_seconds"] == {"min": 20.0, "max": 35.0}

    def test_export_from_srt_has_app_schema(self, tmp_path):
        path = tmp_path / "sample.srt"
        path.write_text(SRT, encoding="utf-8")
        args = type("Args", (), {
            "job_id": None, "language": "en", "min_seconds": None,
            "max_seconds": None, "duration": None})()
        source = skill.load_source(str(path), args)
        payload = skill.to_export(source, args)
        assert payload["schema"] == "openshorts.transcript-export.v1"
        assert payload["language"] == "en"
        assert payload["duration_seconds"] == 6.25
        assert len(payload["transcript"]["segments"]) == 2
        assert payload["windows"]
        assert payload["output_contract"]["schema"] == "openshorts.clip-selection.v1"
        json.dumps(payload, allow_nan=False)


class TestPipelineContract:
    def test_pipeline_has_text_language_segments_and_words(self, tmp_path):
        path = tmp_path / "sample.srt"
        path.write_text(SRT, encoding="utf-8")
        result = skill.main([str(path), "--format", "pipeline",
                             "--language", "en"])
        assert result == 0
        # main writes to stdout; call the loader separately for the shape.
        source = skill.load_subtitle(SRT, filename="sample.srt", language="en")
        pipeline = skill.to_pipeline(source)
        assert set(pipeline) == {"text", "language", "segments"}
        assert pipeline["language"] == "en"
        assert set(pipeline["segments"][0]) == {"start", "end", "text", "words"}
        assert pipeline["segments"][0]["words"]

    def test_json_pipeline_round_trips_word_timings(self, transcript):
        source = skill.load_json(transcript)
        pipeline = skill.to_pipeline(source)
        assert pipeline["language"] == "id"
        assert pipeline["segments"][0]["words"] == transcript["segments"][0]["words"]

    def test_youtube_tool_json_converts_to_pipeline(self):
        youtube = {
            "video_id": "dQw4w9WgXcQ",
            "title": "Example",
            "channel": "Channel",
            "duration": 12.0,
            "languages": [{"code": "en", "kind": "manual",
                           "request": "en", "name": "English"}],
            "language": "en",
            "kind": "manual",
            "segments": [
                {"start": 0.0, "dur": 2.0, "text": "Hello world."},
                {"start": 2.0, "dur": 3.0, "text": "Second line."},
            ],
            "cached": True,
        }
        source = skill.load_json(youtube)
        pipeline = skill.to_pipeline(source)
        assert pipeline["segments"][0]["end"] == 2.0
        assert pipeline["segments"][1]["end"] == 5.0
        assert pipeline["segments"][0]["words"][0]["word"].startswith(" ")
        assert skill.to_youtube(source)["cached"] is True
        assert skill.to_youtube(source)["language"] == "en"

    def test_youtube_format_falls_back_to_the_transcript_language(self, transcript):
        youtube = {
            "video_id": "dQw4w9WgXcQ", "title": "Example", "duration": 12.0,
            "segments": [{"start": 0.0, "dur": 2.0, "text": "Hello."}],
        }
        source = skill.load_json(youtube, language=transcript["language"])
        assert skill.to_youtube(source)["language"] == "id"


class TestYouTubeParity:
    pytestmark = pytest.mark.skipif(
        free_tools is None, reason="free_tools needs FastAPI")

    def test_extract_video_id_matches_free_tools(self):
        urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://www.youtube.com/shorts/dQw4w9WgXcQ",
            "https://www.youtube.com/live/dQw4w9WgXcQ",
            "dQw4w9WgXcQ",
            "https://example.com/watch?v=dQw4w9WgXcQ",
            "https://www.youtube.com/playlist?list=PL123",
        ]
        for url in urls:
            assert skill.extract_video_id(url) == free_tools.extract_video_id(url)

    def test_parse_json3_matches_free_tools(self):
        raw = json.dumps({"events": [
            {"tStartMs": 0, "dDurationMs": 5000,
             "segs": [{"utf8": "Hello "}, {"utf8": "world\n"}]},
            {"tStartMs": 1200, "dDurationMs": 4000,
             "segs": [{"utf8": "overlap"}]},
            {"tStartMs": 5000, "segs": [{"utf8": "\n"}]},
        ]})
        assert skill.parse_json3(raw) == free_tools.parse_json3(raw)

    def test_choose_track_matches_free_tools(self):
        info = {
            "language": "es",
            "subtitles": {"en": [{"name": "English", "ext": "json3"}]},
            "automatic_captions": {
                "es-orig": [{"name": "Spanish (auto-generated)", "ext": "json3"}],
                "fr-orig": [{"name": "French (auto-generated)", "ext": "json3"}],
            },
        }
        for lang in (None, "en", "es", "es-orig", "fr-orig", "de"):
            assert skill.choose_track(info, lang) == free_tools.choose_track(info, lang)


class TestCli:
    def run(self, capsys, *argv):
        assert skill.main(list(argv)) == 0
        return capsys.readouterr().out

    def test_srt_to_text_with_timestamps(self, capsys, tmp_path):
        path = tmp_path / "sample.srt"
        path.write_text(SRT, encoding="utf-8")
        output = self.run(capsys, str(path), "--format", "text", "--timestamps")
        assert output.splitlines() == [
            "[0:01] Hello world, this is line two.",
            "[0:04] Second cue.",
        ]

    def test_srt_to_srt(self, capsys, tmp_path):
        path = tmp_path / "sample.srt"
        path.write_text(SRT, encoding="utf-8")
        output = self.run(capsys, str(path), "--format", "srt")
        assert "1\n00:00:01,000 --> 00:00:03,500\nHello world, this is line two." in output
        assert "2\n00:00:04,000 --> 00:00:06,250\nSecond cue." in output

    def test_srt_to_export(self, capsys, tmp_path):
        path = tmp_path / "sample.srt"
        path.write_text(SRT, encoding="utf-8")
        payload = json.loads(self.run(capsys, str(path), "--format", "export",
                                      "--language", "en"))
        assert payload["schema"] == "openshorts.transcript-export.v1"
        assert payload["language"] == "en"
        assert payload["duration_seconds"] == 6.25

    def test_out_writes_a_file(self, capsys, tmp_path):
        path = tmp_path / "sample.srt"
        path.write_text(SRT, encoding="utf-8")
        out = tmp_path / "nested" / "transcript.json"
        self.run(capsys, str(path), "--format", "export", "--out", str(out))
        assert json.loads(out.read_text(encoding="utf-8"))["schema"] == (
            "openshorts.transcript-export.v1")

    def test_bad_input_is_a_user_error(self, capsys):
        assert skill.main(["does-not-exist.srt"]) == 2
        assert "No such file" in capsys.readouterr().err

    def test_stdin_srt_is_sniffed(self, capsys, monkeypatch):
        monkeypatch.setattr("sys.stdin", io.StringIO(SRT))
        payload = json.loads(self.run(capsys, "-", "--format", "pipeline",
                                      "--language", "en"))
        assert len(payload["segments"]) == 2
        assert payload["segments"][0]["text"] == (
            "Hello world, this is line two.")

    def test_stdin_timestamped_txt_is_sniffed(self, capsys, monkeypatch):
        monkeypatch.setattr("sys.stdin", io.StringIO(
            "[00:01] First line\n[00:03] Second line\n"))
        payload = json.loads(self.run(capsys, "-", "--format", "pipeline",
                                      "--language", "en", "--duration", "10"))
        assert [(s["start"], s["end"]) for s in payload["segments"]] == [
            (1.0, 3.0), (3.0, 10.0)]

    def test_stdin_plain_text_needs_text_format(self, capsys, monkeypatch):
        monkeypatch.setattr("sys.stdin", io.StringIO("Just words."))
        assert skill.main(["-", "--format", "text"]) == 0
        assert capsys.readouterr().out.strip() == "Just words."
        monkeypatch.setattr("sys.stdin", io.StringIO("Just words."))
        assert skill.main(["-", "--format", "export"]) == 2
        assert "no timestamps" in capsys.readouterr().err
