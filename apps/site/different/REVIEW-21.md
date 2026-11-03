# REVIEW-21: "Transcripts that match the reference"

Draft: `site/different/21-transcripts-that-match-the-reference.md` (body about 800 words counting the code block and table). `reviewed:` left empty. Builds clean with `-drafts`; a build without `-drafts` leaves it out. It takes number 21 because 17 and 20 are reserved in the plan (`docs/completed/task-site-2026-09.md`, section 4a). Its area, "Speech", is new.

## Claims and sources

| Claim | Source |
|---|---|
| `POST /v1/audio/transcriptions`; formats `json`, `text`, `srt`, `vtt`, `verbose_json`; `language` or detection; longer than 30 s window by window; at most 600 s; 16-bit PCM WAV only, mp3 and m4a a 400; `prompt`, `stream` and word-level timestamps refused by name; one transcription at a time per model; listed in `/v1/models` with `audio.transcriptions`; served beside chat models | `docs/server.md`, the "Speech to text" paragraphs; `internal/serveapp/speech.go` |
| OpenAI's decode policy by default: compression ratio over 2.4, average log-probability under -1.0, a silent window skipped; a `temperature` above 0 is one attempt from goinfer's own seeded stream | `docs/server.md`, same paragraphs |
| 73/73 transcripts byte-equal to transformers, WER 6.70% on both sides, detected language differs on 0/73 | `docs/tasks/task-multimodal-support-2026-10.md`, "G-S14f3: PASS" (2026-10-10, nobara-pc night run) and "G-S14t2, end to end: PASS" (2026-10-11) |
| The reference is transformers 5.15.0 in float32; the clips are `hf-internal-testing/librispeech_asr_dummy` | `docs/measurements/multimodal-support-2026-10/run-s14f-night.sh` (header); the task doc's G-S14c4 "Data" line |
| Decoder scores at cosine 1.000000000, greedy sequences equal | task doc, "G-S14f2, `openai/whisper-small`, the LibriSpeech clip: PASS" |
| 70 s recording: the same 13 segments, tokens and times to 1e-9 | task doc, "G-S14g3, `openai/whisper-small`: PASS first run" |
| Decode policy: identical segments on the 70 s recording and on a mostly silent one, where a window is skipped | task doc, "G-S14h2, `openai/whisper-small` with OpenAI's decode policy: PASS" |
| Over HTTP the segments equal the references | task doc, "G-S14i3, served on whisper-small: PASS"; `scripts/s14i_served.py` |
| Six planted decoder defects, each red (positions shifted, no final norm among them) | task doc, "G-S14t1, exactness: PASS" and "G-S14f1, tiny: PASS on the first build" |
| Qwen3-ASR first read 71 of 73, a stop-token bug; then 73 of 73 | task doc, "G-S14c4a: PASS" |
| 225 s for the 73 clips, 3.1 s a clip | task doc, "G-S14t2, end to end: PASS" |
| Encoder on CUDA: 1.74 s to 0.13 s for a 30 s window, the 73 transcripts unchanged; CUDA only | `CHANGELOG.md`, "Whisper and Voxtral's audio encoder on the GPU under `--backend cuda`"; `cuda/audio_encoder.go` is the only device registration |
| Qwen3-ASR at int4 5.13% WER against float32's 4.09%; Voxtral Mini compared on a 5.9 s and a 36 s clip | `CHANGELOG.md`, "speech to text" entry |

## Things to decide or check

1. **The machine for the 225 s figure is inferred.** The G-S14f3 night run names nobara-pc. The 2026-10-11 re-read that gives 225 s (G-S14t2) does not name its box; it was the same session's work and it is compared there with the nobara figures, so I wrote "on that machine's CPU". If it ran elsewhere, that sentence and the "It isn't fast" limit need the right machine.
2. **No speed comparison, on purpose.** The record has transformers' arm at 310 s, but from the night before, and a cross-session ratio is not a ratio. The page gives goinfer's own time and says nothing was timed beside it. If you want a comparison, it needs a same-session interleaved run first.
3. **The title's claim is narrower than it sounds.** "Match the reference" is true of whisper-small, English, greedy decoding. The first limit says so; the summary names whisper-small.
4. **The Qwen3-ASR bug as the example in "How it works".** It is the one real bug the clip-by-clip comparison caught in this work that I could source in a sentence, but it is a chat model's, not Whisper's. Cut the paragraph if you would rather the page stayed on Whisper.
5. **`goinfer-serve` in the example.** The server doc's own example is `go run ./cmd/serve`; I used the released binary's name, as the other writeups do. Speech is not in v0.22.0, so until the next release that command needs a build from source. The Models page section says "in releases after v0.22.0"; this page does not, because a writeup is published at a release.
6. **Not covered:** Gemma 4's audio (it has its own record under S5), and EmbeddingGemma 2's audio embeddings.

## Alongside this draft

- The Models page has a "Speech" filter and a "Speech to text" section (`site/internal/site/templates/models.html`), and the build fails without the section (`site/internal/site/verify.go`).
- Three family summaries changed in the matrix generator (`decoder/capability_matrix_test.go:siteDocs`): Voxtral no longer says "not yet served", Qwen3-ASR says its check was on English clips, and Gemma 4 carries the audio task for its E-models.
- The filter's empty-state text no longer says "Only two families take images today"; nine do.
