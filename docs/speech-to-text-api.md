# Speech to text

The `whisper-x` service runs WhisperX `large-v3-turbo` on port `:9191`. It
transcribes an audio file, adds word timings, and assigns speaker labels. The
route is `POST /transcribe` with multipart form data.

Diarization runs by default on this deployment. A warm request took 4.1 s for an
8.7 s two-speaker clip. The first request after a restart took 21 s while the
model loaded. A 41.5 s narration took 8.7 s to transcribe.

## Quick start

`scripts/transcribe_client.py` uploads the file, prints the transcript with
speaker labels, and can print word timings.

```bash
./scripts/transcribe_client.py two_speakers.wav --language en
```

```text
language: en
speakers: 2
duration: 8.48s
[00:00.03 - 00:00.37] SPEAKER_00: Hello.
[00:00.99 - 00:02.38] SPEAKER_00: This is the first speaker.
[00:03.00 - 00:04.34] SPEAKER_00: The weather is nice today.
[00:04.62 - 00:06.39] SPEAKER_01: And this is the second speaker.
[00:06.41 - 00:08.48] SPEAKER_01: I think it might rain later.
transcribed in 4.10s
```

Use `--words` for word timings and confidence scores, `--text-only` for a plain
transcript, `--json` for the raw response, and `--out transcript.json` to save it.

## Request

```bash
curl http://192.168.1.10:9191/transcribe \
  -F file=@two_speakers.wav \
  -F language=en
```

The file field is named `file`. The older `audio_file` name returns a validation
error.

| Field | Meaning |
|---|---|
| `file` | Audio file. Required. WAV and MP3 were tested. |
| `language` | Language code such as `en`, or `auto`. The default is `auto`. |
| `num_speakers` | Exact number of speakers. |
| `min_speakers` | Lower bound when the count is unknown. |
| `max_speakers` | Upper bound when the count is unknown. |

The response contains `status`, `language`, `segments`, `word_segments`,
`speakers`, `num_speakers` and `diarization_device`. Each segment has `start`,
`end`, `text`, `speaker`, and `words`. Each word has `start`, `end`, `word`,
`score` and `speaker`.

## Speaker settings

Use `language` when you know it. On the 8.7 s clip, `language=en` took 4.08 s and
`auto` took 4.49 s, with the same text.

Use `num_speakers` when the count is known and stable. When it is not, leave it
out and set `min_speakers` and `max_speakers` instead. A wrong count can split
one voice into two labels or merge two voices into one.

The labels are anonymous (`SPEAKER_00`, `SPEAKER_01`). The service does not know
names. Map labels to people in your application if you need them.

The clean two-speaker test gave the same labels with and without
`num_speakers=2`, and the extra hint saved about 0.03 s. The hint matters more on
noisy or overlapping audio.

## Word timings

Word timings are always present. `--words` prints them with the model's
confidence score:

```text
[00:00.99 - 00:02.38] SPEAKER_00: This is the first speaker.
    00:00.99 - 00:01.18 This score=0.873
    00:01.25 - 00:01.35 is score=0.686
    ...
```

Scores below about 0.5 point to unclear audio or a word cut at a boundary.
Check those words before using the transcript for anything strict.

## Long audio

The 41.5 s narration returned 8 segments in 8.7 s. The response includes the full
transcript, so one request is fine for short recordings. For very long files,
split at silence and transcribe the parts in order. That keeps each request
smaller and lets you recover from a failed chunk.

## Troubleshooting

- `422` or a missing file error: send the audio in a field named `file`.
- Connection refused: start the audio profile with `./sparkstack.sh up audio`.
- The first real request is slow: WhisperX loads the aligner and diarizer on
  demand. `./sparkstack.sh test` checks `/health` only, so send a short clip after
  a restart to warm the transcription path.
- Speaker labels swap between similar voices: pass `num_speakers`, or use
  `min_speakers` and `max_speakers` on longer recordings.
- Language detection is wrong on a short clip: pass the language code directly.
- An unsupported format fails: convert to WAV or MP3 first.

The service definition lives in `docker-compose.yaml`. General stack setup is
covered in the [README](../README.md).
