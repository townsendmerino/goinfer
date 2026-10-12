---
title: "Transcripts that match the reference"
area: "Speech"
order: 21
summary: "OpenAI's transcription endpoint from a Whisper written in Go. On whisper-small its text equals HuggingFace's on all 73 test clips."
stand: "goinfer answers OpenAI's `/v1/audio/transcriptions` from a Whisper written in Go, in the same binary that serves chat. Its output is compared with HuggingFace transformers on the same checkpoint, clip by clip."
measured: 2026-10-11
reviewed: ""
facts:
  - {label: "endpoint", value: "OpenAI's POST /v1/audio/transcriptions, five response formats"}
  - {label: "against transformers, whisper-small", value: "73/73 clips with the same text, 6.70% word error rate on both sides"}
  - {label: "timestamps", value: "13 segments of a 70 s recording, the same tokens and times"}
  - {label: "checked so far", value: "whisper-small, English, WAV files"}
doesnt:
  - title: "It has been checked on one Whisper size and one language."
    text: "Every comparison here used `openai/whisper-small` and English recordings. The other sizes load the same way, large-v3 included, but none has been compared with the reference, and no other language has been transcribed in a check."
  - title: "It reads WAV files only."
    text: "The file has to be a 16-bit PCM WAV, at any sample rate and channel count, and at most 600 s long. An mp3 or an m4a is refused with a 400, because goinfer does not decode them. Convert first, for example with ffmpeg."
  - title: "It doesn't do everything OpenAI's endpoint does."
    text: "`prompt`, `stream` and word-level timestamps are refused, each by name, instead of being ignored. A `temperature` above 0 samples from goinfer's own seeded random stream, which was not compared with the reference; the comparisons are of greedy decoding, where the most likely token is always taken."
  - title: "It isn't fast, and it wasn't timed against anything."
    text: "Whisper runs in 32-bit floating point on the CPU, one transcription at a time per model. On the Linux PC the 73 clips took 225 s, 3.1 s a clip. No other engine was timed in the same session, so this page gives no ratio. With an NVIDIA card the encoder runs on the GPU; on Apple and WebGPU builds it stays on the CPU."
  - title: "Matching the reference is not the same as being right."
    text: "The word error rate is 6.70% on both sides: goinfer makes whisper-small's mistakes, the same ones in the same places. The check shows the Go code is a faithful Whisper. It says nothing about how good whisper-small is on your audio."
  - title: "The chat models that take audio are checked less."
    text: "Qwen3-ASR and Voxtral Mini take a clip inside a chat message. In 32-bit floating point Qwen3-ASR equals its reference on the same 73 clips, but the 4-bit weights goinfer serves by default read 5.13% word error rate against 4.09%. Voxtral Mini was compared on two clips, 5.9 s and 36 s."
figures:
  - {text: "73/73", source: "docs/tasks/task-multimodal-support-2026-10.md"}
  - {text: "6.70%", source: "docs/tasks/task-multimodal-support-2026-10.md"}
  - {text: "0/73", source: "docs/tasks/task-multimodal-support-2026-10.md"}
  - {text: "225 s", source: "docs/tasks/task-multimodal-support-2026-10.md"}
  - {text: "3.1 s", source: "docs/tasks/task-multimodal-support-2026-10.md"}
  - {text: "1e-9", source: "docs/tasks/task-multimodal-support-2026-10.md"}
  - {text: "1.000000000", source: "docs/tasks/task-multimodal-support-2026-10.md"}
  - {text: "600 s", source: "docs/server.md"}
  - {text: "2.4", source: "docs/server.md"}
  - {text: "-1.0", source: "docs/server.md"}
  - {text: "1.74 s", source: "CHANGELOG.md"}
  - {text: "0.13 s", source: "CHANGELOG.md"}
  - {text: "5.13%", source: "CHANGELOG.md"}
  - {text: "4.09%", source: "CHANGELOG.md"}
  - {text: "5.9 s", source: "CHANGELOG.md"}
  - {text: "36 s", source: "CHANGELOG.md"}
sources:
  - "docs/server.md"
  - "docs/tasks/task-multimodal-support-2026-10.md"
  - "CHANGELOG.md"
  - "internal/whisper/transcribe.go"
  - "internal/whisper/decoder.go"
  - "internal/whisper/longform.go"
  - "internal/serveapp/speech.go"
  - "scripts/s14f_grade.py"
  - "scripts/s14i_served.py"
---

## The problem

Speech to text usually arrives as a second stack. The language model runs in one program, and Whisper runs in Python beside it, with PyTorch, or in a C++ port with its own build.

Rewriting Whisper is also easy to get almost right. Most of what makes its output its own is not the network but the rules around it: which tokens are allowed after a timestamp, when a window of silence is skipped, when a poor result is retried at a higher temperature, how a recording longer than 30 s is cut into windows. Break one rule and the transcript of a clean recording still reads well. Nothing errors.

## What goinfer does

Point the server at a Whisper checkpoint and it answers OpenAI's transcription endpoint. The Whisper is written in Go, like the rest of goinfer. There is no Python in the process and nothing to compile.

```sh
goinfer-serve --model ~/models/whisper-small
curl http://127.0.0.1:8080/v1/audio/transcriptions -F file=@clip.wav -F response_format=verbose_json
```

A client written for OpenAI's endpoint can be pointed at it. The response formats are `json`, `text`, `srt`, `vtt` and `verbose_json`, which carries each segment's start and end time. The language is detected, or you name it with `language`. A recording longer than 30 s is transcribed window by window, up to 600 s.

By default a request runs OpenAI's decode policy. A window is tried greedily first, which means the most likely token is taken at every step. It is tried again at a higher temperature if its text compresses too well (a ratio over 2.4, the sign of a repeating loop) or its average log-probability is under -1.0. A silent window is skipped.

A Whisper model can be served beside chat models from the same process. It appears in `/v1/models` with the capability `audio.transcriptions`.

## How it works

Each stage is compared with HuggingFace `transformers`, the standard Python library for these models, running the same checkpoint in 32-bit floating point. The reference is Python; goinfer is the thing under test.

- **The network.** On whisper-small, the scores goinfer's decoder gives every possible next token have a cosine of 1.000000000 with the reference's (1 means identical), and the greedy token sequences are equal.
- **Timestamps and long recordings.** On a 70 s recording made of test clips, goinfer and the reference produce the same 13 segments: the same tokens, and start and end times equal to 1e-9 of a second.
- **The decode policy.** With OpenAI's thresholds set on both sides, the segments are identical on the 70 s recording and on one that is mostly silence, where a window is skipped.
- **Through the server.** The same recordings sent over HTTP come back with the same segments the reference produced.

A check is not trusted until it has failed on deliberately broken code. The decoder's comparison was run with six planted defects, such as positions shifted by one or the final normalisation removed, and it failed on each.

The checks have found real bugs. Qwen3-ASR, one of the chat models that take audio, first read 71 of 73 clips equal to its reference. The cause was a stop token read wrongly, and with it fixed all 73 matched.

## What was measured

Whisper-small against `transformers` 5.15.0 in 32-bit floating point, on 73 short LibriSpeech recordings (read English, a public test set), each side transcribing greedily and detecting the language itself. Measured on a Linux PC (Ryzen 7 3700X, RTX 2070 SUPER 8 GB, 62 GB of RAM), first on 2026-10-10 and again on 2026-10-11 after the code was made faster.

| What was compared | Result |
|---|---|
| The text of each transcript | the same, byte for byte, on 73/73 clips |
| Word error rate against the published transcripts | 6.70% for goinfer and 6.70% for the reference |
| The language each side detected | differs on 0/73 clips |
| A 70 s recording with timestamps | the same 13 segments, times equal to 1e-9 |

Word error rate is the share of words a transcript gets wrong against a human one. The two sides have the same rate because they produce the same text.

The 73 clips took 225 s, 3.1 s a clip, on that machine's CPU. With `--backend cuda` the encoder, the half of Whisper that reads the audio, runs on the GPU: 0.13 s for a 30 s window against 1.74 s on the CPU, and the 73 transcripts are unchanged.

## Use it

- Download a Whisper checkpoint in HuggingFace's layout, for example `openai/whisper-small`, and start `goinfer-serve --model <that directory>`.
- Send a WAV file to `/v1/audio/transcriptions`. Add `-F response_format=srt` or `vtt` for subtitles, and `-F language=en` to skip detection.
- To give a chat model audio instead, serve Qwen3-ASR, Voxtral Mini or a small Gemma 4 and put an `input_audio` part in the message. The [Models page](/models/#speech) lists them.
- The request fields, the limits and what is refused are in [the server docs](https://github.com/townsendmerino/goinfer/blob/main/docs/server.md), under "Speech to text".
