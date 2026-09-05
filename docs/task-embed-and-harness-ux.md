# Task: the first hour — embedding goinfer in a Go program, and pointing existing tools at `serve`

> **Status: SCOPED 2026-09-02. Phases 0 and 2 SHIPPED the same day** (`pull` exported with
> `hf:`/`demo:` refs; the banner and `serve check`; P-20 measured and fixed). Phase 1 (the
> facade) is unstarted and now unblocked — G4 was what gated it, and it passes.
>
> **Superseded status line:** Phase 0 SHIPPED the same day (`pull` exported; `hf:`/`demo:`
> references accepted by `--model` on both binaries). The ordering was changed on review — see
> §6 — so mode 3 (banner, `serve check`) comes before the facade, and P-20 is measured before
> the facade is designed around it. Everything else below is unstarted. Design record for the two modes of use where
> "pure Go, one static binary, no toolchain" is a differentiator no peer has, and where the
> audience the repo names — a Go engineer, not an ML person — actually lives: **mode 2, embed it**
> (`go get`, in-process) and **mode 3, point my tools at it** (`serve` behind Claude Code, dsh,
> Open-WebUI). Sibling: `task-fit-to-hardware.md` (mode 4), which owns "will it fit"; this doc
> owns what happens in the hour after it does. Reads `task-model-pull.md` (shipped 2026-09-02) as
> the step that made both hours possible, and the 2026-09-02 audit's serving tranches (C-06, C-07,
> C-08, M-17, M-18, M-19, M-21, M-22, M-23 fixed the same day) as the floor this builds on.

**The two people.**

- **Mode 2 — the embedder.** A Go developer writing a CLI, a desktop tool, an edge service, who
  wants generation or structured output *in-process*, deployed by copying a file. They expect what
  they get from any other Go library: a small stable API, a `context.Context`, an iterator for
  streaming, typed errors, and a struct that comes back filled in. The README promises them "a Go
  struct the model cannot violate".
- **Mode 3 — the harness user.** A developer who already has Claude Code, dsh, Open-WebUI,
  Continue or their own agent loop, and wants it pointed at a local model with zero cloud. They
  expect drop-in compatibility — tool calls, streaming, usage accounting, stop sequences — and a
  turn that finishes in the time the harness waits.

**Why they were never designed for.** The lab mode (measurement, parity, the primer) has a
persona, a script and gates, and it works. These two have none: the "first hour" for mode 2 is
`internal/chatapp/main.go` (632 lines, the reference implementation of "use the library"), and
for mode 3 it is a harness discovering the audit's serving findings one request at a time
(`docs/server.md`'s dsh recipe says it directly: "set expectations, don't let the harness discover
them"). The audit fixed the crashes; it did not add the script.

---

## 0. What must not change

- **The Hard tier is the promise** (`docs/api-tiers.md`): `decoder.Load`/`Generate`, the
  tokenizer, `chat`, `constrain`, and `serve`'s routes and flags. Everything below is **additive
  and Experimental until v1.0**; nothing here renames or removes a Hard name.
- **No new module dependency in the root.** A facade is stdlib + the packages that exist.
- **Parity-gated numerics are untouched.** A convenience layer calls the same `Generate`, the
  same masker, the same template; it cannot change a logit.
- **Compatibility is the promise for `serve`; field order and extensions are not**
  (`docs/api-tiers.md`). A "doctor" tests the promise, it does not extend the surface.

## 1. Mode 2 today — the inventory

What a Go program has to do to get a filled-in struct out of a model, read off `internal/chatapp`
and the README example (`README.md:76-88`):

1. ~~Find a checkpoint (now: `goinfer-chat pull`, but `internal/modelpull` is `internal` — a
   library caller cannot reach it).~~ **CLOSED by phase 0:** the package is exported as
   `goinfer/pull` (Experimental), and `--model hf:owner/repo:quant` / `demo:tier` fetches on
   first use, so a library caller and a CLI user take the same first step.
2. `decoder.Load(dir, decoder.Options{...})` (`decoder/model.go:198`), choosing a quant and a
   backend, and knowing that four default-ON behaviours are set through `os.Setenv` rather than
   `Options` (N-42).
3. Load the tokenizer separately; detect the chat template (`chat.Detect`, `chat/chat.go:112`);
   render turns to a string; encode.
4. Build `constrain.GrammarFromStruct(Person{})`, then the masker
   `constrain.NewMasker(g, toks, eos).StopWhenComplete().Process`, which needs the token-bytes
   table and the EOS set from somewhere.
5. `m.Generate(ctx, ids, maxTokens, sp)` → `(<-chan int, *Generation)` (`decoder/model.go:807`);
   drain the channel; decode incrementally with UTF-8 holdback; stop on the template's stop ids;
   check `Generation.Err()` after the channel closes.
6. `json.Unmarshal` — which the README says "always succeeds" and the audit found does not for
   embedded structs, `time.Time`, unsigned ints, enums containing `<`, a top-level integer (M-27,
   M-28, M-29), and refuses the canonical no-argument tool schema (M-30).

Five steps remaining, one package the user must not know about (the env plumbing),
and the promise at step 6 broken in five places. Every one of those steps is code the demos
already carry; the design below is a matter of moving it under one name.

## 2. Mode 2 — the facade

A new package (working name `llm`, at `github.com/townsendmerino/goinfer/llm`; §F.1 on the
name), Experimental, thin, whose whole job is the six steps above.

```go
m, err := llm.Open(ctx, "hf:Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF:q4_k_m")   // path or ref; pulls if absent
defer m.Close()

for tok, err := range m.Chat(ctx, []llm.Message{{Role: "user", Content: "reverse a slice in place"}}) {
    if err != nil { return err }
    fmt.Print(tok.Text)
}

p, err := llm.Into[Person](ctx, m, "Extract the person from: …")   // the README's promise, literally
```

- **`Open`** takes a path *or* an `hf:` reference, resolves the reference through `modelpull`
  (exported as `goinfer/pull`, the library the CLI and the web UI already share — a third client
  costs nothing), applies `task-fit-to-hardware.md`'s plan for the backend and quant unless the
  caller pins them in `llm.Options`, loads the tokenizer, detects the template. One call, one error.
- **`Chat`** is `iter.Seq2[Token, error]` (range-over-func; the module is at go 1.27): the
  channel, the UTF-8 holdback, the stop ids and the post-close `Err()` all live inside it. `Token`
  carries text and id; `m.Complete` is the raw-prompt sibling.
- **`Into[T]`** is `GrammarFromStruct` + the masker + `StopWhenComplete` + `Unmarshal` in one
  generic call. Its acceptance corpus is the audit's list: an embedded struct, a `time.Time`, a
  `uint8`, an enum with `<`, a top-level `int`, a `[]string`, a nested optional object, a schema
  with no properties — each a test that fails today (M-27–M-30) and must pass before the function
  is exported. If `Unmarshal` fails after a constrained generation, that is a bug in `constrain`,
  and `Into` says so in the error rather than retrying.
- **`Options`** carries what today reaches the decoder only through the environment: fast
  attention, fused attention, expert-major prefill, the MoE cache, KV precision, context —
  the `KVPrecision` pattern `Options` already uses (`decoder/model.go:151-163`). With the field
  present, the `serve` flags stop transporting through `os.Setenv` (N-42's last paragraph) and a
  library caller and a multi-model `serve` can differ per model.
- **Typed errors** for the three things a caller must branch on: context length exceeded,
  checkpoint not found/gated (from `pull`), backend declined (with the plan's numbers).
- **This is what the bindings export.** `task-bindings.md`'s c-archive for mobile and sidecar
  for desktop both want exactly this surface; building the facade first means the bindings wrap
  one thing instead of re-deriving the six steps in C.

**The README example becomes the facade example**, ≤ 25 lines, and it is compiled in CI against
the *tagged* release with `GOWORK=off` — which is the standalone-build gate M-34 found missing,
now with a reason to exist beyond release hygiene.

## 3. Mode 3 — the harness contract

### 3.1 What a harness needs, and where each need stands

| need | status after the 2026-09-02 tranches | open |
|---|---|---|
| one command from nothing to a running endpoint | `serve -web` can start with no model and pull one (shipped) | `serve -model hf:…` one-command run (§3.2) |
| streaming that survives a silent minute | C-06 (one SSE writer), M-17 (write deadline), M-19 (Anthropic heartbeat) fixed | — |
| tool calls that round-trip | M-18 (Responses loop) fixed | M-20 Gemma-4 tool template; N-18 `tool_choice` required/any; M-26 usage chunk on tool streams |
| structured output through the API | works for objects | M-27 top-level scalars; M-30 no-arg tool schema |
| an 8k-token turn that finishes | CUDA prefill 270 tok/s measured (`docs/server.md:173-200`); CPU ~30; **resident prefix reuse shipped 2026-09-02 (3358e6b)** — agent turn 3 went 9.13 → 0.42 s, so a turn now costs its suffix, not its history | reuse is token-id bookkeeping only (`residentReuseLen`, `decoder/resident_reuse.go`) and nothing rewinds a recurrent state: on the Gated-DeltaNet/Mamba hybrids a reused prefix runs against the state decode left behind, and it was found silently corrupting Qwen3.5 output on CUDA (2026-09-02 run on the 8 GB box; its QUEUE §A entry was still uncommitted there when this line was written) — needs a family exclusion until the state is snapshotted with the prefix; also single-conversation (QUEUE §A) |
| knowing what it will do before the first request | the banner prints resolved decode/prefill paths (`internal/serveapp/main.go:960-932`) | the rest of §3.3 |
| finding out it does not work, fast | `-require-backend` (`:354`) | nothing exercises the *routes* a harness uses |

### 3.2 One command

`serve -model hf:Qwen/Qwen2.5-7B-Instruct-GGUF:q4_k_m` — the `hf:` form `pull` already parses,
pulled if absent (sha256-verified, the cache dir it already uses), then loaded through the plan.
`goinfer-chat hf:…` likewise. This is `ollama run`'s shape without a registry: the reference is
explicit and auditable, which `task-model-pull.md` §2.1 argued for and this does not change.
`task-model-pull.md` left `serve pull` out "to keep the blast radius on main.go at zero"; the
`hf:` form keeps it at one function call and gives mode 3 the same first minute mode 1 got.

### 3.3 The banner is the UI

A harness user reads exactly one thing: the lines `serve` prints before it says it is listening.
Today those name the model and the resolved decode/prefill paths. The design adds the rest, in one
block, in this order: model and quant; backend and **placement** (from the plan — resident /
expert-cached N / paged / CPU, with the byte line); context cap and KV precision; **session reuse:
on/off** — and when off, why ("hybrid family: reuse excluded until the recurrent state is rewound
with the prefix; each turn re-prefills, ~N s at 8k tokens on this machine" — since 3358e6b the
resident path reuses too, so the why-line is for the exclusions, not the path); routes enabled
(`/v1/chat/completions`, `/v1/messages`, `/v1/responses`, embeddings,
vision, `-web`); features (tools: yes/no per template — Gemma-4 says "partial" until M-20 closes;
structured output; speculative); the expected-rate band if one exists (`task-fit-to-hardware.md`
§5). Every line is a fact the runtime already knows; the banner is where it stops being private.

### 3.4 `serve check` — the doctor

A subcommand (or `-check`) that starts the server, then drives it through **the conversation a
harness would**, over the real routes, and prints a per-feature verdict with a number:

```
models list ............ ok   1 model
chat, streamed ......... ok   TTFT 0.41 s · 38.2 tok/s · usage present
tools, OpenAI .......... ok   call → result → answer in 2 turns
tools, Anthropic ....... ok   tool_use → tool_result → end_turn
tools, Responses ....... ok   function_call_output consumed
structured output ...... ok   {"type":"integer"} → 366
stop sequences ......... ok   split across tokens, not leaked
long prompt (8k) ....... ok   TTFT 4.9 s        ← the number a harness user needs before choosing a model
count_tokens ........... ok   matches usage
```

Each row is a scripted exchange against the tiny fixtures in CI (the shape ten of the 38
`serveapp` test files already have, skipping today on `GOINFER_SERVE_MODEL`) and against a real
model on the box. A red row names the audit ID or the doc section that explains it. This is the
G-class gate for mode 3: a route that a harness uses and nothing exercises is the "gate that
cannot fail" one level up.

### 3.5 Recipes, one per harness, with expectations

`docs/integrations/<harness>.md`, each ≤ 40 lines: the exact settings block, the one `serve`
line, the model class that passed, and **the expectation line** — TTFT at the harness's turn
size on a named machine, with provenance. The dsh recipe exists (`docs/server.md:173`) and is
the template; Claude Code (`/v1/messages`, the three env vars at `docs/server.md:123`),
Open-WebUI and Continue/Cline get the same shape. A recipe is retired when `serve check` covers
what it says.

## 4. What each person never has to know

**Mode 2:** that `tokenizer`, `chat` and `constrain` are separate packages; the token-bytes
table; EOS sets; UTF-8 holdback; `Generation.Err()` after channel close; any `GOINFER_*`
variable; quant names; that `internal/chatapp` is the real example.

**Mode 3:** slot counts and placements (mode 4 owns them); that the resident path re-prefills on
the families where reuse is excluded;
that Gemma-4 tool calls render differently after the first turn (until M-20); that `-web` is off
by default (the banner says how to turn it on); which of the five routes a given harness speaks.

## 5. Gates — pre-registered

- **G1 · the facade example compiles standalone.** The README's ≤ 25-line program builds with
  `GOWORK=off` against the last tag in CI (M-34), and runs against `testdata/llama-tiny` producing
  a filled struct. Fails on any Hard-tier rename by construction.
- **G2 · `Into[T]` corpus.** The eight shapes in §2, each round-tripping on the tiny fixture with
  `Unmarshal` succeeding and `DisallowUnknownFields` on. Today's expected state: five red.
- **G3 · `serve check` in CI on the tiny fixtures**, every row green; on the box against one
  real model per backend before a tag (§C1 gains a row).
- **G4 · constrained decoding cost.** `Into[T]` on a resident GPU model must cost ≤ 1.5× the
  unconstrained token time on the same prompt. Measured paired; below 1.5× ships, above 2× the
  facade documents the cost instead of hiding it.

  > **MEASURED 2026-09-02 — G37 in `docs/QUEUE.md`.** P-20's "3–10×" was an estimate and reads
  > pessimistic: the mask is **39.7 ns/token / 6.0 ms per step** at `fsStr` on V=151,936, the
  > bottom of P-20's own predicted band. P-20's cheapest lever (`isEOS` was a `map[int]bool`
  > probed 151,936× per step) was unspent; an indexed `[]bool` took it **−15%**, interleaved A/B,
  > 3/3 pairs. Against this box's post-G36 1.5B token that is **1.81–1.97×** — inside the
  > ambiguous band and at the top of it, so G4 is neither passed nor failed: `Into[T]` ships with
  > the cost stated, or after the remaining L-07 levers.
  >
  > **UPDATE, same day — G4 now PASSES on this model class.** An exact per-vocabulary bitmap for
  > the plain-string state (96.88% of ids answerable by one bit test, proved exhaustively equal to
  > the full walk) took `fsStr` **6.03 → 0.35 ms, ~17×**. Weighted over every step of a real
  > document the mask is **1.299 ms/step = 1.21×**, under the 1.5× bar. `Into[T]` can be the
  > headline rather than a documented caveat — for the 1.5B class; the per-class caveat below
  > still holds.
  >
  > **And a single number cannot settle G4.** The mask cost is O(V) and CONSTANT in model size,
  > so the ratio worsens as decode gets faster — the same 6.0 ms against a ~2 ms step is ~4×.
  > The bar has to be stated per model class. Two further costs are excluded and unmeasured: a
  > non-nil `LogitProcessor` disables speculative decoding and the on-device greedy argmax, so
  > the real ratio on those configurations is higher.
- **G5 · a harness turn.** dsh's recorded run (277 s, 2026-08-26) and a Claude Code
  `/v1/messages` tool loop are re-run after the M-20/N-18/M-26 fixes and the numbers land in the
  recipes; a recipe with no number is not published.
- **G6 · the banner tells the truth.** A test that parses the banner and compares every line
  against the runtime's own state (placement, ctx, sessions, routes) — the M-07 class (doc says
  exact, code does not) applied to the one document every user reads.

## 6. Phasing — each independently droppable

> **Ordering revised on review, 2026-09-02.** As written, phase 1 (the facade, mode 2) preceded
> phase 2 (banner + `serve check`, mode 3). Reversed, for two reasons. **Audience:** mode 3 has a
> named consumer with a recorded run (dsh, 277 s, 2026-08-26) and a recipe already in
> `docs/server.md`; no mode-2 consumer is named anywhere in the repo, so its demand is inferred.
> **Reversibility:** the banner and the doctor commit to no API, while the facade is a public
> surface that the bindings will wrap — a wrong guess there propagates into C.
>
> And **G4's number is measured before the facade is designed**, not in phase 3. P-20's "3–10×"
> is explicitly an ESTIMATE ("plausibly 3–10× slower"), and the audit lists it among the items
> whose measurement is cheap. It decides whether `Into[T]` — the facade's headline and the
> README's literal promise — is a flagship or a footnote, so it cannot stay a guess underneath
> a design that assumes it.

0. ~~**Export `pull`** and the `hf:` reference in `-model` and `goinfer-chat`.~~ **SHIPPED
   2026-09-02.** `internal/modelpull` → `goinfer/pull`, listed Experimental in `api-tiers.md`;
   `pull.Resolve`/`IsRef` behind `--model` on both binaries. A plain path is returned untouched
   and stays Hard — `TestResolve_pathsAreUntouched` holds that line, because otherwise every
   existing `--model` silently changes meaning.
1. **The facade**: `Open`, `Chat`, `Complete`, `Into[T]`, `Options`, typed errors; the README
   example; G1 + G2 (which means closing M-27–M-30 first — they are the corpus).
2. ~~**The banner** (§3.3) and **`serve check`** (§3.4) with G3 + G6; the recipes (§3.5) with G5.~~
   **BANNER + `serve check` SHIPPED 2026-09-02.** The banner reports decode/prefill path,
   context + KV precision, session reuse *and why it is off*, and features; built as a pure
   function of resolved state so **G6** asserts it against the runtime rather than trusting it.
   `serve check` drives a RUNNING server (a client, not an embedded server) and prints a number
   per row — the long-prompt TTFT row is the one a harness user actually needs. Each defect it
   claims to catch has a test that makes it go red against an httptest server, so **G3** runs in
   CI with no model.

   **§3.5 + G5, 2026-09-02 — PARTLY done, and the split is deliberate.** `docs/integrations/claude-code.md`
   is published with measured numbers: the `/v1/messages` tool loop round-trips (`glob` → `read`
   → answer, 3 turns, 2.83 s, ends `end_turn`), `usage` is present on a **streamed tool call**
   (M-26's fix, confirmed), and the expectation line is **TTFT 8.85 s for a 2,293-token agent
   turn ≈ 259 tok/s prefill** — which independently corroborates `server.md`'s 270 tok/s figure.
   Qwen2.5-7B-Instruct int4 CUDA-resident, RTX 2070 SUPER, driver 595.91.07.

   **Only that one recipe is published**, because §3.5's own rule is that a recipe with no
   number is not published and Open-WebUI / Continue have none yet.

   **The dsh half of G5 was NOT run** (decided 2026-09-02): dsh is not installed here and
   `server.md` documents its install as an ordeal (npm's resolver hangs; 19 peer plugins added
   one error at a time). The 277 s figure therefore stands un-re-measured. The Claude Code loop
   exercises the same routes and the same tool round-trip, so what is missing is dsh's harness,
   not the server behaviour.

   **Two of G5's three preconditions are still open**, which the recipe states rather than
   hides: N-18 (`tool_choice` `any`/`required` does not force a call with 2+ tools — `forcedTool`
   handles `none` and a named function, and `required` falls through to the single-tool case)
   and M-20 (Gemma-4 tool rendering). Only M-26 was fixed, and the measurement confirms it.
3. **Constrained-decoding speed** (L-07) so `Into[T]` is usable on the GPU backends — G4.
4. ~~**Session anchors** (L-05) and longest-prefix reuse (L-15) so an agent loop stops paying a
   cold prefill per turn on the resident path.~~ **SHIPPED 2026-09-02, by a different route than
   either L-05 or L-15 proposed.** Both of those improve the CPU-side session path, which a
   resident model bypasses entirely — so neither would have helped here. The fix was prefix
   reuse NATIVELY on the resident positional KV, where the "both cannot be the source of truth"
   conflict does not exist: agent turn 3 went **9.13 s → 0.42 s (21.7×)**, whole loop
   27.00 → 9.82 s, against L-05's pre-registered ≥2× bar. Gated on token-identity vs cold
   prefill. See the commit and `docs/integrations/claude-code.md`.
5. **Bindings** (`task-bindings.md`) wrap the facade, not the six steps.

## 7. Open questions

- **F.1 The facade's name and place.** `llm` as a sub-package keeps the root module import-light
  and the Hard tier untouched; a root package (`import "github.com/townsendmerino/goinfer"`) reads
  better in a README but binds the module path to one API forever. Lean sub-package, Experimental,
  promote at v1.0 if it earns it.
- **F.2 `Into[T]` on a chat model vs a raw prompt.** Whether it renders the chat template with a
  system line asking for JSON, or takes the caller's messages verbatim. Lean: messages verbatim
  plus an optional system hint; the grammar does the enforcing either way.
- **F.3 Where `-web` stops.** The web UI is a client of the same routes, so structured output
  and tool calling could be exposed there for free; whether it should is a scope call, not a
  design one. Lean: chat and pull only — it is the first-run surface, not a product.
- **F.4 The Anthropic-side check.** Claude Code applies its own idle timeout to a silent
  stream; the heartbeat fix (M-19) should make G5 pass, but the timeout value is unrecorded.
  Measure it in the G5 run rather than assume it.

## Sources

`docs/api-tiers.md` (the Hard tier; the surfaces the facade must not touch) · `README.md:76-88`
(the six-step example the facade replaces) · `internal/chatapp/main.go` (the 632-line reference
implementation of mode 2) · `decoder/model.go:151-163`, `:193`, `:802` (`Options`, `Load`,
`Generate`) · `chat/chat.go:112` (`Detect`) · `pull/pull.go` (the library `pull`
exports) · `internal/serveapp/main.go:328`, `:354`, `:927-932` (`-web`, `-require-backend`, the
resolved-path banner) · `docs/server.md:109-133`, `:173-200` (Claude Code and dsh today) ·
`docs/scoping-dsh-goinfer.md` (Tier 0–2) · `docs/task-model-pull.md` (shipped; `hf:` refs, the
cache dir, the web UI's contract) · `docs/task-bindings.md` (what the facade is for downstream) ·
`docs/audit-2026-09-02.md` C-06, C-07, C-08, M-07, M-17–M-30, M-34, N-18, N-42, P-20, L-05, L-07,
L-15 (the floor and the open items this builds on) · `task-fit-to-hardware.md` (placement, the
plan, the banner's byte lines).
