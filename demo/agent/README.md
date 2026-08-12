# stdlib-agent — ask the Go standard library anything, fully local

The combined demo for the whole stack: a RAG coding agent over the Go
stdlib where **goinfer** runs the model, **ken** runs the retrieval, and
**aikit** underlies both — offline, pure Go, no Python, no API key, no
vector DB, no install.

Two front ends over one shared core:

| command | what |
|---|---|
| [`cmd/agent-web`](cmd/agent-web) ⭐ | browser chat in the Claude-app idiom — streaming answers, the constrained tool call as an expandable chip, ken's results in a collapsible card. One embedded HTML page, no CDNs. |
| [`cmd/stdlib-agent`](cmd/stdlib-agent) | terminal REPL — same loop, good for ssh boxes and asciinema |

```
you> where does net/http decide to reuse a keep-alive connection?

[ken search: "http transport keep-alive connection reuse"]     ← model's own
net/http/transport.go:1234 …                                     constrained
                                                                 tool call
The reuse decision is in (*persistConn).readLoop …
(net/http/transport.go:1234-1267): …
```

## Why this demo exists

Each project's marquee claim, exercised in one artifact:

| project | claim shown here |
|---|---|
| [goinfer](../..) | an LLM in one static binary; **constrained decoding** — the model *cannot* emit a malformed tool call |
| [ken](https://github.com/townsendmerino/ken) | hybrid code search (BM25 + Model2Vec + RRF) with a 35K-chunk stdlib index **baked into the server binary** |
| [aikit](https://github.com/townsendmerino/aikit) | the shared pure-Go primitives both are built on (`linalg`, `embed`, `bm25`, `fuse`, …) |

The web UI is itself part of the pitch: the tool-call chip you expand *is*
the logit mask, the citation card *is* ken's index — agent legibility as a
feature, in the gemma-web mold (stdlib-only server, one `//go:embed` page).

## How it works

```
┌──────────────────┐   constrained JSON    ┌──────────────────────┐
│ agent core       │ ──{"action":"search", │ ken-demo-go-stdlib   │
│ (goinfer +       │    "query":"…"}──────▶│ (MCP subprocess,     │
│  Qwen2.5-Coder)  │ ◀──file:line chunks───│  index + model baked)│
└───────┬──────────┘    over MCP stdio     └──────────────────────┘
   CLI ◀┴▶ web (NDJSON stream)
```

Each turn is two generation passes (see [`agent/agent.go`](agent/agent.go)):

1. **DECIDE** — decoding is constrained to a JSON Schema
   (`constrain.JSONSchema` → token-level logit mask), so the model is
   physically unable to emit an unparseable tool call. The JSON it emits is
   forwarded verbatim as an MCP `tools/call` — the same wire format Claude
   or Cursor would send ken.
2. **ANSWER** — retrieved chunks are spliced into the prompt and the model
   streams a grounded, `file:line`-cited answer.

## Build & run

### Quickstart — just see the loop (no ken setup)

`cmd/ken-stub` is a ~50-line MCP `search` server that returns canned
`file:line` go-stdlib chunks, so you can exercise the full DECIDE → search →
ANSWER loop with **only a model** — no ken repo, no index build. (It's also the
asset-free fixture for smoke-testing the agent wiring.) Real retrieval is the
full setup below; the stub is just the on-ramp.

```bash
GOWORK=off go build -o /tmp/ken-stub ./cmd/ken-stub
GOWORK=off go run ./cmd/agent-web \
    --model ~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf --ken /tmp/ken-stub
# → http://127.0.0.1:8484   (swap agent-web → stdlib-agent for the REPL)
```

### Full demo — real retrieval

Two binaries, both self-contained:

```bash
# 1. The ken side: index + embedding model baked into an MCP server.
#    (In the ken repo — full steps incl. corpus assembly: demos/README.md)
ken build-index /tmp/go-stdlib-curated -o demos/go-stdlib/index.bin \
    --mode=hybrid --chunker=regex --model ~/.ken/model
CGO_ENABLED=0 go build -tags=kendemo -o ken-demo-go-stdlib ./demos/go-stdlib

# 2. The agent side (this dir): bake the LLM into both commands…
./build-embed.sh ~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
./dist/agent-web-$(go env GOOS)-$(go env GOARCH) --ken ./ken-demo-go-stdlib
# → agent-web listening on http://127.0.0.1:8484

# …or the dev loop, model from disk. This module is outside goinfer's go.work,
# so either run `go work use ./demo/agent` once, or prefix with GOWORK=off:
GOWORK=off go run ./cmd/agent-web    --model <model.gguf> --ken <ken-demo-go-stdlib>
GOWORK=off go run ./cmd/stdlib-agent --model <model.gguf> --ken <ken-demo-go-stdlib>
```

Useful flags (both commands): `--ken-top-k` (chunks per search, default 4 —
lower = much faster prefill), `--freq-penalty` (answer-phase repetition/loop
guard, default 0.3; small models loop without it), `--presence-penalty`,
`--temp` (answer phase; the decide phase is always greedy), `--max`, `--quant`;
agent-web adds `--addr` (default `127.0.0.1:8484`).

### Images (Gemma 3 VL)

Run with a Gemma 3 VL model + its vision tower and **agent-web takes images** —
drop one on the page, paste from the clipboard, or click 📎:

```bash
GOWORK=off go run ./cmd/agent-web \
    --model ~/models/gemma-3-4b-it --vision ~/models/gemma-3-4b-it --ken /tmp/ken-stub
```

`--vision` is auto-discovered when `--model` is a VL checkpoint dir. An image turn
skips the ken search (the image is the context) and answers it through goinfer's
pure-Go vision path (SigLIP encoder + projector → the decoder's embed-by-vector
seam). **Heads-up:** the SigLIP prefill is CPU-heavy — expect a minute or two per
image (the UI shows "analyzing image…"); a faster int8 tower is the planned
follow-on (`docs/completed/task-cpu-vision-prefill.md`).

## Demo script

The 14 vetted queries in ken's
[`demos/go-stdlib/QUERIES.md`](https://github.com/townsendmerino/ken/blob/main/demos/go-stdlib/QUERIES.md)
come with canonical answers — every Go dev can verify the citations
against their own `$GOROOT/src`. Good opening trio:

1. "where does the http server decide a request body is too large?"
2. "how does sync.Once avoid the double-checked locking bug?"
3. "what triggers a goroutine stack to grow?"

## Scaffold status / TODO

- [x] compiles + `go vet` clean against goinfer's current APIs (build with
      `GOWORK=off`, or `go work use ./demo/agent`, until it's added to go.work —
      see the last item). End-to-end run against a real ken-demo-go-stdlib
      binary still unverified.
- [ ] tune `decideSystem` on the 0.5B (it may over- or under-search);
      consider always-search as a `--mode=rag` escape hatch
- [ ] web: full markdown + syntax highlighting (renderMarkdownLite is
      deliberately tiny), conversation persistence (JSON files or
      modernc.org/sqlite — keep no-cgo), multi-conversation sidebar
- [ ] port chat's `prequant` embed mode (faster cold start, lower RAM)
- [ ] add to goinfer's go.work + CI build (separate module keeps it out of
      the root `go build ./...`, like ken's `chunk/treesitter`)
- [ ] record the README gif — web UI, with/without retrieval A/B as the closer

<!-- citation-lint: allow-path net EXTERNAL — the Go standard library, not a file in any repo here -->
