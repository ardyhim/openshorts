---
name: openshorts-transcript
description: Generate the transcript JSON the OpenShorts app produces from a YouTube URL/ID or an SRT, VTT, TXT or transcript JSON, and optionally the clip-selection JSON the app imports. Use when the user asks for a transcript, captions, subtitles, a YouTube transcript, an OpenShorts transcript export, the text/language/segments/words shape the clipping pipeline consumes, or clip-worthy moments / an `openshorts.clip-selection.v1` file from a transcript.
version: 1.1.0
homepage: https://github.com/mutonby/openshorts/tree/main/skills/openshorts-transcript
metadata:
  openclaw:
    emoji: "📝"
  hermes:
    category: media
    tags: [transcript, captions, subtitles, youtube, openshorts]
---

# OpenShorts transcript

Produce the transcript data the OpenShorts app produces, and — when the user
wants clips — the `openshorts.clip-selection.v1` file the app imports. The
transcript half is deterministic; the selection half is your editorial
judgment.

The script reads a YouTube URL/ID or a subtitle/transcript file and writes one
of the app's own contracts. Run it from this skill's directory:

```bash
python scripts/openshorts_transcript.py "<youtube-url>" --format export -o transcript.json
python scripts/openshorts_transcript.py captions.srt --format export -o transcript.json
```

Omit `-o` to write to stdout. No API key is needed.

## Choose the format

`export` is the default and the app's downloadable transcript artifact. Change
only when the user asks for one of the other contracts:

| The user wants | `--format` | What comes back |
|---|---|---|
| A transcript to hand to another AI, import into the clipper, or download like the app | `export` | `openshorts.transcript-export.v1` |
| The word-level transcript the pipeline stores as `source_transcript.json` | `pipeline` | `{text, language, segments[{start, end, text, words}]}` |
| The public YouTube transcript generator response | `youtube` | `{video_id, title, channel, duration, languages, language, kind, segments[{start, dur, text}]}` |
| Readable text to paste or print | `text` | one line per cue, optionally with `--timestamps` |
| A subtitle file | `srt` | SubRip cues |

If the user says only "get the transcript", use `export`. If they say
"word-level", "with words", or name `source_transcript.json`, use `pipeline`.
If they name the YouTube transcript tool or want caption-track metadata, use
`youtube`.

Clip selection is not a `--format`: the script prepares and validates, and you
write the selection after reading the export. The next section is the exact
workflow.

## Transcript plus clip selection

When the user asks for clip moments, highlights, a clip list, or a
`clip-selection.json` — not just the transcript — produce both files in one
run:

```bash
python scripts/openshorts_transcript.py "<source>" --format export \
  -o <name>.transcript.json
```

Then read the whole export and write `<name>.clip-selection.json` following
[`clip-selection.md`](clip-selection.md):

- Use `transcript.segments[]` for content and cut points, `windows[]` for
  navigation, and follow the export's `prompt` and `output_contract`.
- Output only `openshorts.clip-selection.v1` JSON: absolute seconds, clips best
  first, no overlap, inside `clip_duration_seconds`, count inside
  `recommended_clip_count`.
- The script does not choose clips. That is your editorial judgment; never
  invent a moment you did not read.

Validate before returning:

```bash
python scripts/openshorts_transcript.py <name>.transcript.json \
  --check-selection <name>.clip-selection.json
```

`valid: true` means the app's import endpoint will accept the file. Fix every
`errors` entry and every field-length/overlap warning. Return both paths, or
only the selection file when that is all the user asked to see.

## Workflow

1. **Identify the input.** A YouTube link or bare 11-character ID, an
   `.srt`/`.vtt`/`.txt`/`.json` file, or `-` for stdin. A plain text file with
   no timestamps can only produce `--format text`.
2. **Pick the track when relevant.** For a YouTube video with several caption
   tracks, pass `--lang` (for example `es` or `en-orig`). Run
   `--list-languages` first when the user cares which track they get.
3. **Generate.** Run the command and keep the output byte-for-byte: do not add
   keys, reorder segments, round timestamps again, or wrap JSON in prose.
4. **Add clips when asked.** If the user wants clip moments, follow "Transcript
   plus clip selection": select from the export, write the selection JSON, and
   validate it with `--check-selection`.
5. **Check before returning.**
   - `export`: `schema == "openshorts.transcript-export.v1"`,
     `duration_seconds > 0`, `transcript.segments` non-empty, `windows`
     non-empty.
   - `pipeline`: top level is exactly `text`, `language`, `segments`; every
     segment has `start`, `end`, `text`, `words`; words use the app's
     leading-space convention.
   - `youtube`: every segment has `start`, `dur`, `text`; `language` and
     `kind` are set.
   - `text`/`srt`: the first cue starts at the first spoken word, not at 0.
   - `clip-selection`: `--check-selection` reports `valid: true`.
6. **Return the file or the exact output.** With `-o`, tell the user the path.
   Without it, return the contents and nothing else unless they asked for an
   explanation.

## Input rules

- Timestamps are **absolute seconds from the start of the source**.
- Use `--language <code>` when the language is known. Otherwise the export
  auto-detects from the text when `py3langid` is installed (the classifier the
  app's Parakeet path uses), and says `unknown` when it is not.
- A timestamped TXT whose last line has no end time needs `--duration` before it
  can become an export or pipeline JSON; without it the script refuses rather
  than inventing a final timecode.
- YouTube input reads the captions YouTube already has. It does not download or
  transcribe audio. Videos with no captions fail with `no_captions`.
- If YouTube blocks the request, retry with `--cookies cookies.txt` or
  `--proxy <url>`.

## Accuracy

SRT, VTT and YouTube captions carry **line** timings, not word timings. The
`pipeline` format estimates per-word times by distributing each cue over its
words. That is suitable for caption layout and the app's contracts, but it is
not speech-recognition word timing. Say so when the user asks for word-level
accuracy. The app's own `source_transcript.json` comes from faster-whisper or
Parakeet and is exact at the word level; for that, run OpenShorts itself
(`python main.py -u "<url>" --prepare-manual`) and read
`.transcript_checkpoint.json`.

See `reference.md` for the transcript payloads and error codes,
`clip-selection.md` for the selection contract and editorial checklist, and the
source files this skill mirrors.
