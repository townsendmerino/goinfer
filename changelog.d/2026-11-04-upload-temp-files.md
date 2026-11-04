- **Large uploads no longer leave temp files behind, and an over-long audio clip is refused before it is decoded.** `POST
  /v1/audio/transcriptions` left a temp file the size of the upload for every form over 32 MiB, and `POST /v1/files` did the
  same through `server.NewHandler`. A clip over its limit is now refused from its WAV header, on the chat routes' audio too: a
  256 MiB clip at 4 kHz allocated about 2.5 GiB before its refusal. An over-limit chunked body on `/v1/files`,
  `/v1/audio/transcriptions` or `/v1/systemone` now answers 413, not 400.
