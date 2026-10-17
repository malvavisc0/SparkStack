# Text to speech

The `tts-voicedesign` service runs Qwen3-TTS on port `:9192`. It serves the
OpenAI speech route with 10 preset voices and 24 kHz mono audio.

A 3.8 s clip took 2.3 s to render. A 41.5 s narration took 23.9 s. Speech
generation runs faster than real time, so a long script is practical, but it is
not instant.

## Quick start

`scripts/tts_client.py` sends the request and writes the audio file.

```bash
./scripts/tts_client.py "SparkStack is up." --out hello.wav
./scripts/tts_client.py "Small files are useful for agents." --format mp3 --out hello.mp3
./scripts/tts_client.py --list-voices
```

The client prints the audio duration, file size, render time and format details.

## Voices

`GET /speakers` returns the preset ids:

```text
vd_british_male
vd_british_female
vd_british_male_casual
vd_british_female_warm
vd_german_male
vd_german_female
vd_german_male_casual
vd_german_female_warm
vd_spanish_female
vd_spanish_male
```

The default is `vd_british_male`. The German and Spanish voice ids match their
language. Use the matching voice and language for non-English text.

## Request

```bash
curl http://192.168.1.10:9192/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input": "SparkStack is up.", "voice": "vd_british_male"}' \
  --output hello.wav
```

| Field | Default | Notes |
|---|---|---|
| `input` | required | Text to speak. |
| `voice` | `vd_british_male` | One of the preset ids. |
| `response_format` | `wav` | `wav`, `mp3`, `pcm` or `zip`. |
| `speed` | 1.0 | 0.8 produced 7.18 s of audio for a fixed sentence, 1.2 produced 5.48 s. |
| `language` | unset | Full language name: `English`, `German`, `Spanish`, and so on. |
| `instruct` | unset | Voice direction, such as "speak slowly and warmly". |
| `max_new_tokens` | unset | Optional generation cap. |

## Language names

Pass a full language name, not a short code. Supported names are `auto`,
`Chinese`, `English`, `French`, `German`, `Italian`, `Japanese`, `Korean`,
`Portuguese`, `Russian` and `Spanish`.

A short code such as `en` returns HTTP 200 with an empty 44-byte WAV and an error
in the server log. The client rejects short codes before sending the request.
If you omit `language`, the voice decides.

## Formats

`wav` is 24 kHz, mono, 16-bit PCM. The streaming encoder writes `0xFFFFFFFF` in
the RIFF and data size fields, so some players show an unknown duration. The
audio data is valid. The client calculates the duration from the data chunk.

`mp3` is much smaller. A 2.2 s clip was 10.7 KB as MP3 and about 100 KB as WAV.
Use MP3 for transport and WAV or PCM when another program needs raw audio.

`pcm` is headerless 24 kHz mono 16-bit samples.

`zip` contains `audio.mp3` and `timer.json` with word timings. Use it when a
client needs both the audio and the timing data. The first `zip` request after a
restart loads the forced aligner and took about 90 s in a test; later requests
were about 1 s.

## Instruct and speed

The service accepts an `instruct` field to change delivery without changing the voice id:

```bash
./scripts/tts_client.py "That is a great idea." \
  --voice vd_british_female_warm \
  --instruct "Speak in a warm, excited tone." \
  --out excited.wav
```

`speed` changes the length of the generated audio. The default is 1.0, 0.8
produced a longer clip, and 1.2 produced a shorter one. The client passes the
value through.

## Long text and round trips

A 110-word script produced 41.5 s of audio in 23.9 s. The server handled the
whole input in one request. For scripts longer than a few minutes, split at
paragraphs and join the audio afterward.

To test the full loop, synthesize a sentence and transcribe it:

```bash
./scripts/tts_client.py "The quick brown fox jumps over the lazy dog." --out round.wav
./scripts/transcribe_client.py round.wav --language en --text-only
```

## Troubleshooting

- The output is an empty 44-byte WAV: `language` used a short code or an
  unsupported value. Use a full name or omit the field.
- Voice not found: run `./scripts/tts_client.py --list-voices`.
- Connection refused: start the audio profile with `./sparkstack.sh up audio`.
- The first request can be slower while CUDA graphs and caches settle. The model
  loads at container startup; `./sparkstack.sh test` checks health only.
- A player shows an unknown WAV duration: the header uses streaming sizes. Use a
  player that reads the data chunk, or convert the file.
- A long script sounds cut off: split it into shorter requests and check the
  end of each clip.

The service definition lives in `docker-compose.yaml`. General stack setup is
covered in the [README](../README.md).
