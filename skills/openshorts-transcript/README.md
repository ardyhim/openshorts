# OpenShorts transcript skill

An [Agent Skill](https://agentskills.io) that turns a YouTube URL or a subtitle
file into the same transcript payloads the OpenShorts app produces.

It mirrors three contracts:

- `openshorts.transcript-export.v1` — the app's downloadable transcript export
  (the default).
- The word-level `source_transcript.json` shape the clipping pipeline consumes.
- The public YouTube transcript generator response from
  `GET /api/tools/youtube-transcript`.

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

Useful options:

- `--lang es` / `--lang en-orig` chooses a YouTube caption track.
- `--list-languages` prints the tracks available for a video.
- `--language id` sets the output language when it cannot be inferred.
- `--duration 123.4` supplies the source duration for a TXT with an open last
  cue.
- `--cookies cookies.txt` and `--proxy <url>` handle YouTube restrictions.
- `-` reads stdin (JSON, SRT/VTT, or plain text for `--format text`).

## Parity

The skill is tested against the app's own functions:

```bash
python -m pytest tests/test_openshorts_transcript_skill.py -q
```

The export is compared field-for-field with
`manual_clip_selection.build_transcript_export()`, and the YouTube helpers with
`free_tools.extract_video_id()`, `choose_track()` and `parse_json3()`.

## Accuracy note

SRT, VTT and YouTube captions have line-level timings. The `pipeline` format
estimates word timings from those lines so the payload has the shape the app
expects. It is not faster-whisper word timing; use the app's transcription
pipeline when exact per-word timing matters.
