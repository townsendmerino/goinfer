# Task (goinfer): add a "bigger-model" tier of canned /demo prompts

> **For:** Claude Code, in `~/tmcode/goinfer`.
> **Why:** both size tiers run the same `demo/chat` binary, so they share one
> `demos` table. The existing prompts are tuned to the 0.5B (tiny, verifiable).
> Add a harder set that the 1.5B (~20 tok/s, genuinely more capable) shows off —
> multi-step bugs, small algorithms, concurrency, a correct conceptual answer,
> richer structured extraction. Keep every prompt brief (CPU-bound: ~20 tok/s, so
> a 400-token answer is ~20 s on screen).

## 1. Tag existing demos, add the new tier

Add a tier marker to the `demo` struct (e.g. `tier string` with values `"fast"`
and `"big"`, or a `big bool`). Tag the existing seven as **fast**; tag the new
ones as **big**. `json`/`extract` work on both — leave them `fast` (they're the
party trick at any size).

Append these to the `demos` table (prompt strings are paste-ready; keep the
brevity instructions — they bound runtime):

```go
{name: "race", tier: "big", prompt: "What's the bug and the fix?\n\nfunc main() {\n\tfor i := 0; i < 3; i++ {\n\t\tgo func() { fmt.Println(i) }()\n\t}\n\ttime.Sleep(time.Second)\n}"},
{name: "lru", tier: "big", prompt: "Implement an LRU cache in Go with O(1) Get and Put. Code only, brief."},
{name: "pool", tier: "big", prompt: "Write a worker pool in Go: N goroutines consume a jobs channel and send results on another channel, coordinated with sync.WaitGroup. Concise."},
{name: "test", tier: "big", prompt: "Write IsBalanced(s string) bool that checks balanced (), [], and {}, plus a table-driven test for it. Concise."},
{name: "niltl", tier: "big", prompt: "Explain the difference between a nil slice and an empty slice in Go, with a one-line example of each. Two sentences."},
{name: "wrap", tier: "big", prompt: "Write a Go function that opens a file and returns a wrapped error with %w, then show a caller using errors.Is. Concise."},
{name: "extract", tier: "fast", json: true, prompt: "Extract repo, version, language, and license as a JSON object from:\nken v0.4.0 is an MIT-licensed Go code-search tool."},
```

(Adjust field names/order to match the actual `demo` struct. If a `json` field
doesn't exist on the struct as shown, wire `extract` the same way the existing
`json` demo sets one-shot JSON-constrained mode.)

## 2. Show the tier in `/demos`

In the `/demos` listing, print the tag so users know which flatter the bigger
model, e.g.:

```
  8  race     What's the bug and the fix? …            [1.5B]
  9  lru      Implement an LRU cache in Go …           [1.5B]
```

Map `tier=="big"` → `[1.5B]`, `tier=="fast"` → no tag (or `[0.5B]`). Keep it a
single dim-colored suffix like the existing `[json]` marker. `/demo <n|name>`
runs them unchanged — both binaries can run every demo; the tag is guidance, not
a gate.

## 3. README

In `demo/chat/README.md`, split the `/demos` table into two short groups —
"works great on the 0.5B" (the fast set) and "shows off the 1.5B" (the big set) —
or add a tier column. One line of context: every demo runs on either binary; the
big ones just look better on the 1.5B. Keep the `json`/`extract` note about
guaranteed-valid JSON.

## 4. Done

- [ ] `demo` struct has a tier marker; existing 7 tagged `fast`, new 6 tagged
      `big`, `extract` added (`fast`, json-constrained).
- [ ] `/demos` shows the `[1.5B]` tag; `/demo <name>` runs all of them.
- [ ] README `/demos` section reflects the two tiers.
- [ ] `gofmt` / `go vet` / `go test ./...` green; `go run ./demo/chat --model …`
      and `/demos` + a couple `/demo` runs work on both 0.5B and 1.5B.
```
