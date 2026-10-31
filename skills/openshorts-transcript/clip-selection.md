# Clip selection

The editorial half of a transcript run. The Python script cannot judge which
moments are good; you do, after reading the transcript. This file is the
contract the OpenShorts app imports and the checklist to apply before writing.

## Output

Return only valid JSON: no fences, no commentary, no extra keys.

```json
{
  "schema": "openshorts.clip-selection.v1",
  "clips": [
    {
      "start": 596.4,
      "end": 644.4,
      "score": 95,
      "title": "A specific title, max 100 characters",
      "hook": "The strongest hook, max 10 words",
      "reason": "One sentence, max 20 words, naming what makes the moment work.",
      "description_tiktok": "1-2 sentences that tease the payoff. #hashtag #hashtag #hashtag",
      "description_instagram": "1-2 sentences that tease the payoff. #hashtag #hashtag #hashtag"
    }
  ]
}
```

`start` and `end` are **absolute seconds from the start of the source**, as
JSON numbers. Clips are best first.

## Hard rules

- `schema` exactly `openshorts.clip-selection.v1`.
- `0 <= start < end <= duration_seconds`.
- Respect `clip_duration_seconds` and `recommended_clip_count` from the export.
  Defaults are 15-60 seconds and 5-8 clips when the input has no export.
- At least 1 second per clip; the app rejects anything shorter.
- Never cut in the middle of a word. Prefer `transcript.segments[]` boundaries.
- No two clips may cover the same moment or share the same payoff.
- `title` ≤ 100 characters, `hook` ≤ 10 words, `reason` one sentence ≤ 20
  words. The app truncates title and hook, so an over-limit field silently
  loses text.
- Both descriptions: 1-2 sentences that tease the payoff, plus 3-5 relevant
  single-token hashtags.
- Write title, hook, reason and descriptions in the source language unless the
  user asks for another language.
- Do not invent quotes, timecodes, or visual claims. If the timestamps are
  missing or unreliable, say so instead of fabricating seconds.

## How to read an export

`openshorts.transcript-export.v1` carries everything you need:

- `language` - output language.
- `duration_seconds` - upper bound for every `end`.
- `clip_duration_seconds` - allowed clip length.
- `recommended_clip_count` - how many clips to return.
- `prompt` and `output_contract` - the app's own instructions. Follow them
  when they are more specific than this file; the hard rules above still win.
- `transcript.segments[]` - timed text; use it for content and cut points.
- `windows[]` - grouped consecutive segments with `id`, `start`, `end` and
  `text`; use them for navigation and context. A clip may start and end
  anywhere, not only on a window boundary.
- `transcript.text` - may be a short placeholder; prefer the segments.

For a plain SRT/VTT/TXT, first build an export (the script's default format)
so duration, windows and the recommended count come from the same logic the app
uses.

## What makes a clip work

Look for, in the whole transcript:

- a hook that creates a question, tension, or "wait, what?"
- a story that pays off: reveal, reversal, confession, punchline, lesson
- real emotion: frustration, joy, embarrassment, relief, pride
- humor, banter, awkwardness, an unexpected reply
- surprise, contradiction, a belief being overturned
- practical value: a clear explanation, demo, or fix
- relatability: "that's me" or "send this to someone"
- a line that survives outside its context

These are lenses, not categories. A strong moment often fits none cleanly.

**Cold-open rule:** the first two seconds must carry a hook that works on a
viewer who has seen nothing else. Start early enough that the stakes are clear,
and end after the payoff, not mid-thought.

Read the whole source before choosing. A line that looks strong alone may be
setup for a payoff 90 seconds later. Do not select from the first window only.

## Validate before returning

```bash
python scripts/openshorts_transcript.py transcript.json \
  --check-selection clip-selection.json
```

The report is JSON: `valid`, `clips`, `duration_seconds`, `warnings`, `errors`
and a per-clip `items` list. Exit code is `0` when importable and `2` when not.

- Fix every `errors` entry.
- Fix field-length, missing-field and overlap `warnings` before returning.
  Other warnings (clamped end, below requested minimum) describe what the app
  will do, not a broken file.
- `--check-selection` can run without a transcript when you pass `--duration`.

## Import into the app

The app imports this file with:

```http
POST /api/jobs/{job_id}/analysis/import
Content-Type: application/json
```

The request body may be the JSON itself or a multipart `file` field. The route
is available in self-host/free mode only, and the job must already have a
transcript. The app validates the timestamps, renders the clips, and returns
the updated job.
