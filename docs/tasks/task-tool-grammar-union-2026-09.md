# Task: constrained tool calls when there is more than one tool (T0–T7) — 2026-09

> **Status: SCOPED 2026-09-15, unstarted. T0 is a measurement and gates the rest.**
>
> goinfer can already make a tool call structurally impossible to malform — but only when the
> choice of tool is unambiguous. In the case every agent harness actually produces (a dozen tools,
> `tool_choice: auto`) the decode is unconstrained and a botched call is possible. This doc closes
> that, and closes **N-18** with the same primitive.
>
> Siblings: [`task-embed-and-harness-ux.md`](task-embed-and-harness-ux.md) owns mode 3 (point my
> tools at it) and lists N-18 as one of G5's open preconditions; `docs/server.md`'s "What
> `tool_choice` actually constrains" paragraph is the user-facing statement this work rewrites;
> [`task-web-ui-2026-09.md`](task-web-ui-2026-09.md) W19 and W23 are downstream consumers.

---

## 1. Why

**The failure.** With two or more tools and `tool_choice` left at `auto`, the model free-runs.
`ParseToolCalls` (`chat/tools.go:93`) then parses whatever came out, and malformed JSON yields no
call — which the harness reads as a prose answer where it expected a tool result. It retries,
re-reads files it has already read, and burns the context window on the retry loop. The damage is
not one bad turn; it is that a bad turn looks like a *different kind* of turn.

**Outside evidence, unprompted.** A developer migrating large prompts from the frontier APIs to a
self-hosted stack (Ollama + opencode, 2026-09) named this as a session-killer in exactly those
terms — parsing tool-call responses "shoves so much raw data into context that it destroys sessions
like a burst pipe at your dinner party" — alongside context-window exhaustion. He is describing
goinfer's mode-3 user, on a competitor, hitting a failure class goinfer is one step away from
eliminating. Treat it as a single data point about what the pain *is*, not as a measurement of how
often it happens; T0 is that measurement.

**What the product currently claims.** `docs/server.md:144` states the position accurately and
narrowly: a named tool "rides constrained decoding, so its call cannot be malformed … `any`/`required`
constrain only when there is exactly ONE tool … with two or more the model decides freely and the
output is NOT grammar-constrained." The goal of this doc is to make the second half of that
sentence unnecessary.

**Why it is worth doing at all.** "Tool calls cannot fail to parse" is a claim about a *class* of
failure, provable by construction rather than by benchmark, and it holds on a 1.5B model as firmly
as on a 70B one — which is the opposite of every other lever in this repo, where the small-model
case is where things fall apart. Neither Ollama nor llama.cpp's default path makes it.

> **Correction 2026-09-24 (read from source, not measured):** the last sentence is wrong for llama.cpp. At `427291b`
> (2026-09-05) its differential autoparser derives `<tool_call>` from a Qwen2.5-style template and, under `tool_choice: auto`,
> builds a LAZY grammar triggered on that word whose body is a choice over the supplied tool names (literals) followed by each
> tool's parameter schema (`common/chat-auto-parser-generator.cpp`, `standard_json_tools` in `common/chat-peg-parser.cpp`) —
> i.e. T1 + T2's wrapper arming, already shipped. T1 is therefore parity with llama.cpp, not a lead. Ollama 0.32.5 does not
> constrain: `tools/tools.go` parses after the template-derived tag by scanning for a supplied tool name and an arguments
> object. Neither accepts a call whose wrapper is missing on a Qwen2.5 template; llama.cpp does for Qwen3-Coder's first call
> (optional `<tool_call>`, triggered on the complete `<function=NAME>` of a supplied tool). See the T0 record's peer section.

## 2. What exists today, cited

- **The grammar machinery is built and in use.** `constrain.ToolCallGrammar`
  (`constrain/tool_grammar.go:30`) composes the family wrapper around a JSON-Schema-constrained
  value, assembling the schema document inline as
  `{"name":{"const":<name>}, <argsKey>:<paramSchema>}`. The model physically cannot emit anything
  else.
- **It is only reached when unambiguous.** `internal/serveapp/tools.go` applies it "tight when
  unambiguous" — a forced function, or a lone tool. A named `tool_choice` that cannot be
  constrained is a 400 rather than a silent unconstrained decode (audit M-05). With 2+ tools and
  `auto`, nothing is applied.
- **The blocker is one missing schema keyword.** N candidate tools is a disjunction, and
  `constrain/schema.go` names `oneOf` in the keywords it deliberately does **not** enforce — "the
  package contract is that these are a compile error, not a silent no-op, so a caller can't believe
  a constraint is in force that isn't (M27)."
- **The grammar interface is small and already supports lookahead.** `Grammar`
  (`constrain/constrain.go:23`) is `TryBytes` / `Commit` / `CanEnd` / `Reset` / `Clone` — five
  methods, with `Clone` existing for non-mutating multi-step lookahead. Anything implementing it
  drops into `constrain.NewMasker` unchanged.
- **Masking already enables grammar-fused speculative decode.** `gr.masker` is what turns it on
  (`internal/serveapp/tools.go`), so a union masker that respects the interface inherits it.
- **Only four templates have a constrainable call form.** `ToolCallWrapper` (`chat/tools.go:78`)
  covers `chatml`/`mellum2` (`<tool_call>\n` … `\n</tool_call>`), `llama3` (**empty prefix and
  suffix**, `parameters` as the args key), and `mistral` (`[TOOL_CALLS] `, one-element array).
  Everything else returns `ok=false` and is untouched by this work.
- **N-18 is open and cited from the recipe.** `docs/QUEUE.md:201`: `tool_choice` `required`/`any`
  "forces nothing with 2+ tools". It is one of G5's stated preconditions.

## 3. Ground rules

1. **The model keeps its choice.** Under `auto` a prose answer is legal, and must stay legal. This
   work constrains the *shape of a call once the model commits to making one* — it must never force
   a call that was not going to happen. A design that cannot preserve this is the wrong design.
2. **No general `oneOf`.** See §9 — the narrow route exists precisely because tool calls carry a
   discriminator in a fixed position, and widening the schema contract is a much larger change with
   a much larger blast radius.
3. **`schemaGrammar` is not modified.** The union delegates to it unchanged, so every argument
   schema behaves exactly as it does today under a named `tool_choice`.
4. **Parity-gated.** Constrained decoding changes which tokens are legal, therefore which tokens
   are sampled. T6 owns the evidence that a call already valid unconstrained is unchanged.
5. **Additive.** Families without a wrapper, and requests with 0 or 1 tools, take the paths they
   take today, byte for byte.

## 4. T0 — measure the failure rate first. This gates T1.

The work is a couple of days; it should not start on one blog post and an argument.

- **Question:** across the checkpoints goinfer actually serves, on realistic agent turns with 5–15
  tools and `tool_choice: auto`, what fraction of intended calls come out unparseable by
  `ParseToolCalls`?
- **Method:** replay a fixed set of turns (the opencode two-turn task from R14 and the Claude Code
  agent loop are both already recorded) against 0.5B / 1.5B / 7B and one MoE, at int4 and int8,
  greedy and at temperature 0.7. Count: parsed calls, unparsed-but-intended (a `<tool_call>` opener
  with a body that fails to parse), and calls to a name not in the tool list.
- **Kill:** below **2%** unparsed across every small-model cell, this is a documentation fix, not an
  engineering one — write the number into `docs/server.md` and stop. Above **10%** on any cell that
  a harness would realistically use, T1 ships without further argument. In between, it is a judgement
  call to be made with the number in hand, not before.
- The temperature cells matter more than the greedy ones: harnesses do not all pin temperature, and
  this failure should get worse as sampling loosens. If it does not, that is itself a finding.
- Record in `docs/measurements/`, as usual.

## 5. T1 — a union grammar over N tools

The insight that keeps this small: **every branch is identical until the tool name, and the tool
name decides everything after it.** So the alternation is live across one string and nowhere else.

1. Wrapper prefix, then `{"name":"` — one literal, shared by every branch, exactly as today.
2. **The name: a prefix trie over the N tool names.** `TryBytes` admits a byte iff it extends at
   least one live name. This is the only genuinely new machinery in the item, and it is a
   string-prefix set, not general alternation.
3. At the closing quote **exactly one name is complete**, so select that tool's `schemaGrammar` and
   delegate the rest of the object to it, unmodified.
4. Args key and suffix: literal, per family.

New API, mirroring the existing one:

```go
func ToolCallsGrammar(prefix, suffix, argsKey string, array bool, tools []ToolSpec) (Grammar, error)
```

`Clone` must clone the live-name set and, once past the discriminator, the delegate — otherwise
grammar-fused speculative decode reads the wrong branch.

**Provable by construction:** every string the union accepts is a well-formed call to exactly one
supplied tool. That is the property to state in the test name.

## 6. T2 — arming: the part with a real design problem

The grammar cannot be active from the first token, or `auto` would force a call every turn
(ground rule 1). It has to arm on the model committing to a call.

- **chatml / mellum2 / mistral: straightforward.** Watch for the wrapper prefix (`<tool_call>\n`,
  `[TOOL_CALLS] `). Until it appears the model is unconstrained; once it does, lock to the union.
  This is the correct semantics as well as the convenient one — the model still chooses whether to
  call, and can no longer choose to call badly.
- **llama3: there is no prefix to arm on.** `ToolCallWrapper` returns empty prefix *and* empty
  suffix — a llama3 call is a bare JSON object, so nothing distinguishes the start of a call from
  a prose answer that happens to begin with `{`. Three options, and **this needs a decision before
  T1 is written, because it may change the trigger interface**:
  - (a) arm on `{"` plus the args-key-shaped opening, accepting that a prose answer starting with a
    JSON object gets constrained into a call. Wrong occasionally, and silently.
  - (b) leave llama3 to the named/`required` paths only, where the call is already forced, and
    state the limitation. Narrow, correct, and the default recommendation.
  - (c) arm on `{"name":"` followed by a byte that extends a live tool name — the trie itself is
    the trigger, and a prose `{` that does not continue into a real tool name releases the
    constraint. Most precise; needs the masker to support release-on-mismatch, which it does not
    today. **Only viable if the release path can be made exact** — a grammar that cannot back out
    is worse than no grammar.
- The trigger-arming wrapper is new either way: a `Grammar` that passes everything through until a
  literal (or trie prefix) matches, then delegates. Small, and worth having on its own.

## 7. T3 — `tool_choice: required` / `any` with 2+ tools (closes N-18)

Once T1 exists this is nearly free: force the wrapper prefix rather than waiting for it, then run
the union. The model must call something, and cannot call it badly.

- Removes N-18 from `docs/QUEUE.md:201` and from G5's precondition list.
- `docs/server.md:144`'s paragraph gets rewritten: the "with two or more the model decides freely"
  sentence goes away.
- Named `tool_choice` behaviour is unchanged.

## 8. T4–T7

- **T4 — family coverage census.** `ToolCallWrapper` covers four template names; the registry has
  36 families. Which families resolve to `chatml`/`mellum2`/`llama3`/`mistral`, and which of the
  rest have a call form that *could* be expressed as prefix/suffix/argsKey but has not been?
  Publish the answer in `docs/capability-matrix.md`'s neighbourhood rather than leaving "tools:
  yes/no" to imply constrainability. A family with tool support but no constrainable form is a
  different row from a family with neither.
- **T5 — parallel calls, decided not discovered.** Some families emit several calls in one turn;
  the existing `array` flag only covers Mistral's one-element wrapper. Either the union accepts a
  repeated wrapper (each independently constrained) or it constrains the first and releases — pick
  one, write it down, test it. Do not let it be whatever falls out.
- **T6 — gates.**
  - *Construction:* property test over generated tool sets — every accepted string parses to a call
    to exactly one supplied tool; no accepted string names an absent tool. Fuzz the tool names
    (unicode, `<`, `&` — M-29's escaping trap lives here).
  - *Parity:* a call that was already valid under an unconstrained decode is byte-identical under
    the union, same seed. Red before the change on a deliberately broken trie.
  - *Non-forcing:* with `auto` and a prompt that should not call a tool, the output is unchanged
    from today across the whole T0 matrix. This is the gate that protects ground rule 1, and it is
    the one most likely to fail quietly.
  - *Speculative:* grammar-fused spec decode produces identical output with and without the union
    masker, since `Clone` is doing real work on that path.
  - *Re-measure:* T0's matrix re-run. The unparsed-call rate must be **zero** on constrainable
    families, not merely lower — anything else means the grammar has a hole.
- **T7 — the docs.** `docs/server.md:144` (the claim), `docs/QUEUE.md:201` (N-18 closed),
  `docs/integrations/claude-code.md` and the opencode recipe (both currently warn about this),
  `task-embed-and-harness-ux.md` (G5's preconditions). The README line is worth writing carefully:
  this is a claim about a failure class, and it should be stated no more broadly than the family
  census in T4 supports.

## 9. The alternative, considered and rejected

**Add `oneOf` to the schema compiler.** General, and it would serve `response_format` too. Rejected
as the route for *this* problem: the masker is incremental, so alternation means carrying a set of
live branches through every node type until the input discriminates, then collapsing — a change to
the automaton itself, in a package whose contract is deliberately narrow and whose failure mode is a
caller believing a constraint is in force that is not (M27). Estimated at a week or more against T1's
couple of days, with far more surface to get wrong.

It is the right change eventually, and if someone builds it, T1's trie becomes an optimisation of
the general case rather than a workaround. It should not be the thing standing between the product
and a tool call that cannot fail to parse.

## 10. Not in scope, stated

- **Making the model choose a *better* tool.** This work constrains form, not judgement. A
  well-formed call to the wrong tool is exactly as likely afterwards.
- **Families with no constrainable call form.** They keep today's behaviour; T4 says which they are.
- **Anything about MCP.** Tool *transport* is the harness's problem.
- **`response_format` / structured output.** Shares the machinery, has its own items (W23).

## Sources

`constrain/tool_grammar.go:30` (`ToolCallGrammar`, the single-tool case) ·
`constrain/constrain.go:23` (the `Grammar` interface, including `Clone`) · `constrain/schema.go`
(the keyword contract, and `oneOf` among the unenforced) · `internal/serveapp/tools.go` ("tight
when unambiguous", and `gr.masker` enabling grammar-fused spec) · `chat/tools.go:78` (the four
wrapper families, and llama3's empty prefix) · `chat/tools.go:93` (`ParseToolCalls`, where a
malformed call becomes a prose answer) · `docs/server.md:144` (the user-facing claim this rewrites)
· `docs/QUEUE.md:201` (N-18, open) ·
[`task-embed-and-harness-ux.md`](task-embed-and-harness-ux.md) (mode 3, G5's preconditions) ·
[`task-web-ui-2026-09.md`](task-web-ui-2026-09.md) W19, W23 (downstream consumers)

<!-- doc-reviewed: 2026-09-15 -->
