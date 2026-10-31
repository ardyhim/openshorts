# OpenShorts transcript skill

An [Agent Skill](https://agentskills.io) that turns a YouTube URL or a subtitle
file into the same transcript payloads the OpenShorts app produces — and,
optionally, the `clip-selection.json` the app imports.

It mirrors four contracts:

- `openshorts.transcript-export.v1` — the app's downloadable transcript export
  (the default).
- The word-level `source_transcript.json` shape the clipping pipeline consumes.
- The public YouTube transcript generator response from
  `GET /api/tools/youtube-transcript`.
- `openshorts.clip-selection.v1` — the clip picks the app's manual import
  accepts.

Plus readable TXT and SRT output for editors.

## Install

```bash
# Claude Code and most agents
cp -r skills/openshorts-transcript ~/.claude/skills/

# OpenCode
cp -r skills/openshorts-transcript ~/.config/opencode/skills/
```

Anything else: drop the folder where that agent reads skills from. The skill
needs Python 3.9+ and `yt-dlp` for YouTube input (`pip install yt-dlp`).
Subtitle input uses only the standard library. Install `py3langid` if you want
automatic language detection on subtitle input (`pip install py3langid`).

## Use

```bash
# App transcript export (default)
python scripts/openshorts_transcript.py "<youtube-url>" -o transcript.json
python scripts/openshorts_transcript.py captions.srt -o transcript.json

# Word-level pipeline transcript
python scripts/openshorts_transcript.py captions.vtt --format pipeline -o source_transcript.json

# The public YouTube tool response
python scripts/openshorts_transcript.py "<youtube-url>" --format youtube --compact

# Human-readable
python scripts/openshorts_transcript.py captions.srt --format text --timestamps
python scripts/openshorts_transcript.py captions.srt --format srt -o captions.srt
```

## Clips in one pass

When the user wants clip moments, the agent produces both files:

```bash
# 1. Transcript export (duration, windows, prompt, output contract)
python scripts/openshorts_transcript.py "<youtube-url>" -o video.transcript.json

# 2. The agent reads the export and writes video.clip-selection.json
#    following clip-selection.md, then validates it:
python scripts/openshorts_transcript.py video.transcript.json \
  --check-selection video.clip-selection.json
```

`--check-selection` mirrors the app's `parse_selection_payload()`: it reports
`valid`, `warnings`, `errors` and per-clip timings, and exits `2` when the file
cannot be imported. The Python script cannot judge which moments are good — the
agent does that from the transcript — but it guarantees the result will import.

Useful options:

- `--lang es` / `--lang en-orig` chooses a YouTube caption track.
- `--list-languages` prints the tracks available for a video.
- `--language id` sets the output language when it cannot be inferred.
- `--duration 123.4` supplies the source duration for a TXT with an open last
  cue.
- `--cookies cookies.txt` and `--proxy <url>` handle YouTube restrictions.
- `-` reads stdin (JSON, SRT/VTT, or plain text for `--format text`).
- `--check-selection FILE` validates a `clip-selection.json`; pair it with a
  transcript source or `--duration`.

## Parity

The skill is tested against the app's own functions:

```bash
python -m pytest tests/test_openshorts_transcript_skill.py -q
```

The export is compared field-for-field with
`manual_clip_selection.build_transcript_export()`, the YouTube helpers with
`free_tools.extract_video_id()`, `choose_track()` and `parse_json3()`, and the
selection validator against the import rules in
`manual_clip_selection.parse_selection_payload()`.

## Accuracy note

SRT, VTT and YouTube captions have line-level timings. The `pipeline` format
estimates word timings from those lines so the payload has the shape the app
expects. It is not faster-whisper word timing; use the app's transcription
pipeline when exact per-word timing matters.
