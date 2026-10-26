"""Tests for the Gemini-free transcript export / external-AI import contract."""
import json

import pytest

import manual_clip_selection as manual


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("CLIP_MIN_SECONDS", raising=False)
    monkeypatch.delenv("CLIP_MAX_SECONDS", raising=False)


@pytest.fixture
def transcript():
    return {
        "language": "id",
        "segments": [
            {"start": 0.0, "end": 12.5, "text": "Halo semuanya. Ini rahasia besar."},
            {"start": 12.5, "end": 42.0, "text": "Trik ini membuat hidup jauh lebih mudah."},
        ],
    }


def _payload(**overrides):
    clip = {"start": 1.0, "end": 31.0}
    clip.update(overrides)
    return {"clips": [clip]}


class TestBuildTranscriptExport:
    def test_carries_transcript_windows_and_prompt(self, transcript):
        export = manual.build_transcript_export(
            transcript, 42.0, job_id="job-1", language="id")

        assert export["schema"] == manual.EXPORT_SCHEMA
        assert export["job_id"] == "job-1"
        assert export["language"] == "id"
        assert export["duration_seconds"] == 42.0
        assert len(export["transcript"]["segments"]) == 2
        assert export["transcript"]["segments"][0]["text"].startswith("Halo")
        assert export["windows"]
        assert export["output_contract"]["schema"] == manual.SELECTION_SCHEMA
        assert '"clips"' in export["prompt"]
        assert manual.SELECTION_SCHEMA in export["prompt"]
        assert export["recommended_clip_count"]["min"] >= 2

    def test_duration_falls_back_to_last_segment(self, transcript):
        export = manual.build_transcript_export(transcript, 0)
        assert export["duration_seconds"] == 42.0
        assert "42.000 seconds" in export["prompt"]

    def test_invalid_duration_falls_back_to_segments(self, transcript):
        export = manual.build_transcript_export(transcript, float("nan"))
        assert export["duration_seconds"] == 42.0

    def test_uses_clip_duration_bounds(self, transcript, monkeypatch):
        monkeypatch.setenv("CLIP_MIN_SECONDS", "20")
        monkeypatch.setenv("CLIP_MAX_SECONDS", "35")
        export = manual.build_transcript_export(transcript, 42)
        assert export["clip_duration_seconds"] == {"min": 20.0, "max": 35.0}
        assert "20-35 seconds" in export["prompt"]

    def test_inverted_bounds_fall_back_to_defaults(self, transcript):
        export = manual.build_transcript_export(
            transcript, 42, min_secs=80, max_secs=20)
        assert export["clip_duration_seconds"] == {"min": 15.0, "max": 60.0}

    def test_drops_unusable_segments(self):
        export = manual.build_transcript_export({
            "segments": [
                {"start": 0, "end": 5, "text": "  "},
                {"start": 8, "end": 8, "text": "zero"},
                {"start": 2, "end": 4, "text": "keep"},
            ]
        }, 10)
        assert export["transcript"]["segments"] == [
            {"start": 2.0, "end": 4.0, "text": "keep"}
        ]

    def test_serializes_to_json_without_naNs(self, transcript):
        export = manual.build_transcript_export(transcript, float("inf"))
        json.dumps(export, allow_nan=False)


class TestParseSelectionPayload:
    def test_minimal_clip_gets_app_defaults(self, transcript):
        result = manual.parse_selection_payload(
            _payload(), duration=42.0, transcript=transcript)

        assert result["schema"] == manual.SELECTION_SCHEMA
        assert result["warnings"] == []
        clip = result["clips"][0]
        assert clip == {
            "start": 1.0,
            "end": 31.0,
            "source_window_id": "",
            "predicted_score": 70,
            "video_description_for_tiktok": "Halo semuanya.",
            "video_description_for_instagram": "Halo semuanya.",
            "video_title_for_youtube_short": "Halo semuanya.",
            "viral_hook_text": "Halo semuanya.",
            "why": "",
        }

    def test_normalized_output_is_importable_again(self, transcript):
        # app.py validates once, writes manual_selection.json, then main.py
        # re-validates the same file in --clips-file mode. Round-tripping must
        # not change a single field.
        first = manual.parse_selection_payload(
            _payload(score=88, title="Judul", hook="Hook pendek",
                     description_tiktok="TikTok desc", reason="Karena jelas"),
            duration=42.0, transcript=transcript)
        second = manual.parse_selection_payload(
            first, duration=42.0, transcript=transcript)
        assert second["clips"] == first["clips"]

    def test_accepts_markdown_fences(self, transcript):
        raw = "```json\n" + json.dumps(_payload()) + "\n```"
        result = manual.parse_selection_payload(
            raw, duration=42.0, transcript=transcript)
        assert len(result["clips"]) == 1

    def test_accepts_fences_wrapped_in_prose(self, transcript):
        raw = ("Sure! Here is the selection:\n\n```json\n"
               + json.dumps(_payload()) + "\n```\n\nLet me know if you want changes.")
        result = manual.parse_selection_payload(
            raw, duration=42.0, transcript=transcript)
        assert len(result["clips"]) == 1

    def test_accepts_bare_json_wrapped_in_prose(self):
        raw = "Here you go: " + json.dumps(_payload()) + " Hope it helps."
        result = manual.parse_selection_payload(raw, duration=42.0)
        assert len(result["clips"]) == 1

    def test_accepts_a_bare_list(self):
        result = manual.parse_selection_payload(
            [{"start": 0, "end": 20}], duration=42.0)
        assert len(result["clips"]) == 1

    def test_accepts_nested_result_wrapper(self):
        result = manual.parse_selection_payload(
            {"result": _payload()}, duration=42.0)
        assert len(result["clips"]) == 1

    def test_accepts_the_app_metadata_shape(self, transcript):
        # A user may paste a metadata.json from a previous job.
        raw = {
            "shorts": [{
                "start": 2,
                "end": 22,
                "predicted_score": 91,
                "video_title_for_youtube_short": "Judul pendek",
                "viral_hook_text": "Hook dari metadata",
                "video_description_for_tiktok": "TikTok caption",
                "video_description_for_instagram": "IG caption",
                "why": "Momen lucu",
            }]
        }
        result = manual.parse_selection_payload(
            raw, duration=42.0, transcript=transcript)
        clip = result["clips"][0]
        assert clip["predicted_score"] == 91
        assert clip["video_title_for_youtube_short"] == "Judul pendek"
        assert clip["viral_hook_text"] == "Hook dari metadata"
        assert clip["video_description_for_tiktok"] == "TikTok caption"
        assert clip["video_description_for_instagram"] == "IG caption"
        assert clip["why"] == "Momen lucu"

    def test_accepts_timestamp_and_metadata_aliases(self):
        raw = {"clips": [{
            "start_time": "1.5",
            "end_seconds": "25",
            "youtube_title": "Judul",
            "hook_text": "Hook",
            "explanation": "Alasan",
            "tiktok": "TT",
            "instagram": "IG",
            "window_id": "w2",
        }]}
        result = manual.parse_selection_payload(raw, duration=42.0)
        clip = result["clips"][0]
        assert clip["start"] == 1.5
        assert clip["end"] == 25.0
        assert clip["video_title_for_youtube_short"] == "Judul"
        assert clip["viral_hook_text"] == "Hook"
        assert clip["why"] == "Alasan"
        assert clip["video_description_for_tiktok"] == "TT"
        assert clip["video_description_for_instagram"] == "IG"
        assert clip["source_window_id"] == "w2"

    def test_clips_are_sorted_by_start(self):
        raw = {"clips": [
            {"start": 30, "end": 50},
            {"start": 0, "end": 20},
        ]}
        result = manual.parse_selection_payload(raw, duration=60.0)
        assert [c["start"] for c in result["clips"]] == [0.0, 30.0]

    def test_clamps_negative_start_and_past_end(self, transcript):
        result = manual.parse_selection_payload(
            _payload(start=-3, end=50), duration=42.0, transcript=transcript)
        clip = result["clips"][0]
        assert clip["start"] == 0.0
        assert clip["end"] == 42.0
        assert any("clamped to 0" in w for w in result["warnings"])
        assert any("past the source" in w for w in result["warnings"])

    def test_warns_about_duration_outside_the_band(self):
        result = manual.parse_selection_payload(
            _payload(start=0, end=10), duration=42.0)
        assert any("below the requested minimum" in w for w in result["warnings"])

        result = manual.parse_selection_payload(
            _payload(start=0, end=40), duration=42.0, max_secs=30)
        assert any("above the requested maximum" in w for w in result["warnings"])

    def test_score_is_clamped_and_invalid_scores_default(self):
        result = manual.parse_selection_payload(
            {"clips": [
                {"start": 0, "end": 10, "score": 150},
                {"start": 12, "end": 22, "score": -5},
                {"start": 24, "end": 34, "score": "nope"},
            ]},
            duration=42.0)
        assert [c["predicted_score"] for c in result["clips"]] == [100, 0, 70]
        assert any("using 70" in w for w in result["warnings"])
        assert any("clipped to 0-100" in w for w in result["warnings"])

    def test_boolean_score_is_rejected_as_a_number(self):
        result = manual.parse_selection_payload(
            _payload(score=True), duration=42.0)
        assert result["clips"][0]["predicted_score"] == 70
        assert any("score is not a number" in w for w in result["warnings"])

    def test_title_is_truncated_on_a_word_boundary(self):
        title = " ".join(["kata"] * 40)
        result = manual.parse_selection_payload(
            _payload(title=title), duration=42.0)
        clipped = result["clips"][0]["video_title_for_youtube_short"]
        assert len(clipped) <= 100
        assert clipped.split()[-1] == "kata"

    def test_hook_is_capped_at_ten_words(self):
        hook = "satu dua tiga empat lima enam tujuh delapan sembilan sepuluh sebelas"
        result = manual.parse_selection_payload(
            _payload(hook=hook), duration=42.0)
        assert result["clips"][0]["viral_hook_text"] == (
            "satu dua tiga empat lima enam tujuh delapan sembilan sepuluh")

    def test_blank_title_falls_back_to_the_transcript(self, transcript):
        result = manual.parse_selection_payload(
            _payload(title="   "), duration=42.0, transcript=transcript)
        assert result["clips"][0]["video_title_for_youtube_short"] == "Halo semuanya."

    def test_blank_description_falls_back_to_the_title(self):
        result = manual.parse_selection_payload(
            _payload(title="Judul", description_tiktok=" "), duration=42.0)
        clip = result["clips"][0]
        assert clip["video_description_for_tiktok"] == "Judul"
        assert clip["video_description_for_instagram"] == "Judul"


class TestParseSelectionErrors:
    @pytest.mark.parametrize("payload", [
        "",
        "   ",
        "not json",
        "```json\n{train: 1}\n```",
    ])
    def test_invalid_json(self, payload):
        with pytest.raises(manual.SelectionError, match="not valid JSON"):
            manual.parse_selection_payload(payload, duration=42.0)

    @pytest.mark.parametrize("payload", [
        {},
        {"nothing": []},
        {"clips": []},
    ])
    def test_missing_or_empty_clip_list(self, payload):
        with pytest.raises(manual.SelectionError):
            manual.parse_selection_payload(payload, duration=42.0)

    def test_clip_must_be_an_object(self):
        with pytest.raises(manual.SelectionError, match="Clip 1 is not a JSON object"):
            manual.parse_selection_payload([42], duration=42.0)

    def test_missing_start(self):
        with pytest.raises(manual.SelectionError, match="Missing required field: start"):
            manual.parse_selection_payload({"clips": [{"end": 10}]}, duration=42.0)

    def test_missing_end(self):
        with pytest.raises(manual.SelectionError, match="Missing required field: end"):
            manual.parse_selection_payload({"clips": [{"start": 1}]}, duration=42.0)

    @pytest.mark.parametrize("value", [True, False, "abc", None])
    def test_non_numeric_timestamp(self, value):
        with pytest.raises(manual.SelectionError):
            manual.parse_selection_payload(
                {"clips": [{"start": value, "end": 10}]}, duration=42.0)

    @pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
    def test_non_finite_timestamp(self, value):
        with pytest.raises(manual.SelectionError, match="finite"):
            manual.parse_selection_payload(
                {"clips": [{"start": value, "end": 10}]}, duration=42.0)

    def test_end_must_be_after_start(self):
        with pytest.raises(manual.SelectionError, match="must be greater than start"):
            manual.parse_selection_payload(
                {"clips": [{"start": 10, "end": 10}]}, duration=42.0)

    def test_end_clamped_before_start_is_rejected(self):
        with pytest.raises(manual.SelectionError, match="must be greater than start"):
            manual.parse_selection_payload(
                {"clips": [{"start": 50, "end": 60}]}, duration=42.0)

    def test_clip_shorter_than_a_renderable_minimum(self):
        with pytest.raises(manual.SelectionError, match="too short to render"):
            manual.parse_selection_payload(
                {"clips": [{"start": 1, "end": 1.5}]}, duration=42.0)

    def test_unknown_duration(self):
        with pytest.raises(manual.SelectionError, match="duration is unknown"):
            manual.parse_selection_payload(_payload(), duration=0)

    @pytest.mark.parametrize("duration", [None, "nan", "inf"])
    def test_non_finite_duration(self, duration):
        with pytest.raises(manual.SelectionError, match="duration is unknown"):
            manual.parse_selection_payload(_payload(), duration=duration)

    def test_too_many_clips(self):
        raw = {"clips": [{"start": i, "end": i + 1}
                         for i in range(manual.MAX_CLIPS + 1)]}
        with pytest.raises(manual.SelectionError, match=f"at most {manual.MAX_CLIPS}"):
            manual.parse_selection_payload(raw, duration=1000)
