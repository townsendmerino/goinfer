- **A census of what `serve` sends a client** (SB7). `TestWireCensus` replays 38 requests through the in-process server
  and checks each response's status, headers and body byte for byte against goldens, SSE streams included. The requests
  cover chat, completions, responses, messages, tool calls, errors and a mid-stream cancel. A layout test holds
  `internal/serveapp`'s test files that import `decoder` at 40, to fall as the shell moves onto a fake engine. No
  behavior changes.
