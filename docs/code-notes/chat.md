# chat: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `chat`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## package.byteexact

Moved from `chat/chat.go` (the package doc comment) on 2026-10-10.

```text
Package chat renders a conversation into the exact prompt string a model's
chat template expects — no Jinja engine. goinfer loads a handful of families
(Gemma 3/4, ChatML/Qwen, Llama-3, Mistral, Ministral 3); each has a small native Go
renderer here, checked against HuggingFace's apply_chat_template (see the
testdata/chat_goldens fixtures).

WHAT "BYTE-EXACT" COVERS, precisely (N-37 — the claim here used to be unqualified):

  - The `sys_user` and `sys_multi` shapes are byte-exact for every family EXCEPT Harmony
    (gpt-oss)'s multi-turn case: a prior ASSISTANT turn re-rendered into history is emitted
    as `<|start|>assistant<|message|>{content}<|end|>` with no `<|channel|>` marker at all —
    this file's own Harmony() doc comment says the channel is declared, never optional
    ("gpt-oss always answers on a channel... Channel must be included for every message"), so
    a real gpt-oss conversation's prior turns would carry `<|channel|>final` and this
    rendering diverges from it. NOT fixed here: unlike ChatML's no-system case there is no
    harmony golden to render the correct channel marker against (checked 2026-09-11 — none
    exists in testdata/chat_goldens), so guessing the exact byte sequence would risk being
    wrong in a way that looks right, the same reasoning that kept the other three families'
    no-system goldens unmade below. Single-turn harmony (no prior assistant turn) is
    unaffected.
  - The NO-SYSTEM shape: a ChatML template that declares a default system message (Qwen 2.5's
    "You are Qwen, created by Alibaba Cloud…", Qwen2.5-VL's "You are a helpful assistant.") gets
    it, read from the checkpoint's own template by Detect (default_system.go; owner, 2026-10-09),
    so the no-system rendering is that template's byte for byte (TestDetect_chatMLDefaultSystemIsTheTemplates).
    The bare ChatML() renderer, shared with families that declare no default, still emits no
    system turn: TestChatML_noSystem_documentedDivergence pins that against the same golden.
  - Tool rendering is byte-exact for GEMMA 4 ONLY, whose tool syntax is a micro-language the
    model parses. For the JSON families the embedded tool JSON's spacing follows Jinja's
    tojson and is checked structurally — see TestRenderTools_declarations (M-20).

Detect picks the renderer from the GGUF/HF tokenizer.chat_template string
(fingerprinted against the known families); for a bare checkpoint with no
template, it falls back to the special-token heuristic. An unrecognized
template is an explicit error — the caller then does a raw completion.
```

Note added when this moved (2026-10-10): the Harmony exception in the first bullet above is no longer true. `harmonySegments` renders
a prior assistant turn on the `final` channel, and `TestHarmony_conversationMatchesHF` pins it against
`testdata/chat_think_goldens/harmony_history.json`. The last bullet is also narrower than the code: the Harmony, Qwen3.5 XML and
Qwen2.5-grouped tool renderings are byte-exact against their own goldens (`harmony_tools.json`, `qwen35_tools.json`,
`tools_qwen25_grouped.json`). The code comment was rewritten to match.

## Detect.fingerprints

Moved from `chat/chat.go` (the comment above the SmolLM3 case in `Detect`) on 2026-10-10.

```text
M-36 (audit-2026-09-10): three fingerprints checked BEFORE their generic siblings,
same "more specific first" discipline already used above (Harmony-before-Gemma4,
Mellum2-before-ChatML) — each of these three otherwise matches a generic branch's
substring test and would get silently misrendered rather than refused or routed
correctly. All three verified 2026-09-16 against real, live checkpoint templates,
not guessed:
  - SmolLM3 (HuggingFaceTB/SmolLM3-3B): <|im_start|> markers (matches ChatML's own
    test) but always emits its own "## Metadata" system preamble the caller never
    asked for — declined rather than silently rendered as plain ChatML.
  - Olmo 3 (allenai/Olmo-3-7B-Instruct): <|im_start|> markers too, but its tool
    syntax is <functions>/<function_calls> XML, not ChatML/Qwen's Hermes
    <tool_call> JSON dialect — a tool-calling request would render/parse the wrong
    shape under generic ChatML, so this declines too rather than guessing.
  - Ministral 3 (mistralai/Ministral-3-8B-Instruct-2512): contains "[INST]" (matches
    Mistral()'s own test) but is [SYSTEM_PROMPT]-based, not v0.3's system-folded-
    into-the-last-user-turn shape — routed to the real Ministral() renderer instead
    of silently misrendering under the wrong Mistral version.
```

## Detect.bareMinistral

Moved from `chat/chat.go` (the comment above the bare-checkpoint Ministral case in `Detect`) on 2026-10-10.

```text
Ministral 3 (and the Mistral 3 VL saves) ship no chat template; their tekken vocab carries [SYSTEM_PROMPT] as
a control token, which Mistral v0.3's [INST]-only vocab does not (S10: without this a Ministral 3 image request
was refused, "no chat template for vision", and text fell back to raw completion).
```

## Harmony

Moved from `chat/templates.go` (the comment above `Harmony`) on 2026-10-10.

```text
Harmony (gpt-oss) — "<|start|>{role}<|message|>{content}<|end|>", with a REQUIRED
system message that gpt-oss's template synthesizes rather than taking from the caller,
and a generation prompt of a bare "<|start|>assistant".

Three things make this family unlike the others here, all of them load-bearing:

 1. THE SYSTEM MESSAGE IS SYNTHESIZED, NOT PASSED THROUGH. gpt-oss's own template emits a
    fixed identity line, a knowledge cutoff, TODAY'S DATE, a reasoning-effort line and the
    valid-channel declaration — whether or not the caller supplied a system prompt. A
    caller-supplied system prompt is a DEVELOPER message in harmony, which is a separate
    role, so it is rendered as one rather than replacing the preamble.
 2. THE DATE IS LIVE. `Current date:` comes from strftime_now in the upstream template, so
    it is read from timeNow (the same injectable clock Llama-3's preamble uses) and a
    byte-exactness test must pin the clock.
 3. THE CHANNEL SET IS DECLARED, NOT OPTIONAL. "Channel must be included for every message"
    — gpt-oss always answers on a channel (analysis / commentary / final), so there is no
    non-thinking form of this prompt the way Qwen3 and Gemma-4 both have. Callers that want
    only the answer must strip the analysis channel from the OUTPUT; it cannot be suppressed
    in the prompt.

Reasoning effort defaults to "medium", matching the upstream template's own default.

STOPS ARE UPSTREAM'S, NOT "EVERY END MARKER". gpt-oss-20b's generation_config.json lists
eos_token_id [200002, 199999, 200012] = <|return|>, <|endoftext|>, <|call|>. <|end|> is NOT a stop:
it closes each MESSAGE, and one reply is several — the analysis message ends in <|end|>, then the
model opens <|start|>assistant<|channel|>final<|message|> and ends the turn with <|return|> (or
<|call|> for a tool call). Stopping on <|end|> ended every reply after its thinking: through serve,
gpt-oss streamed only its analysis channel and never an answer (found 2026-09-14 by the web UI's W6
capture, docs/tasks/task-web-ui-2026-09.md).
```

## Phi3

Moved from `chat/templates.go` (the comment above `Phi3`) on 2026-10-10.

```text
Phi3 — microsoft/Phi-3-mini-4k-instruct's current template: per turn
"<|{role}|>\n{content}<|end|>\n", a leading system turn when given, generation prompt
"<|assistant|>\n". No BOS in the template, and the HF tokenizer adds none
(add_bos_token false).

Before this renderer existed Detect matched no Phi-3 template, so chat and serve fed Phi-3
a raw completion with no turn markers at all: the model wrote its own "<|assistant|>" and
answered the benchmark's filler prompt with newlines (found 2026-09-25,
docs/measurements/peer-claim-2026-09-25.md).
```

## GlmOCR

Moved from `chat/templates.go` (the comment above `GlmOCR`) on 2026-10-10.

```text
GlmOCR — zai-org/GLM-OCR's template (the GLM-4.1V family's `[gMASK]<sop>` shape): "[gMASK]<sop>", an optional
"<|system|>\n{system}", then per turn "<|user|>\n{content}" or "<|assistant|>\n<think></think>\n{content}" (a prior
assistant turn; the template writes an EMPTY think block there when the turn carries no reasoning, and this renderer
carries none), and the generation prompt "<|assistant|>\n". No end-of-turn marker: the next role marker closes a turn,
and a reply ends at <|endoftext|> or the next <|user|> (the checkpoint's eos_token_id is [59246, 59253], hence the stops).

Verified byte-for-byte against HF's apply_chat_template on the checkpoint at revision 2e85a628
(testdata/chat_goldens/glm_ocr.json, text turns), and for the image shape by the O3 gate
(multimodal.GlmOcrImageBlock + the golden's input_ids). NOT rendered: tools (`<|observation|>` turns and the
`<tool_call>` XML), `enable_thinking`'s `<think></think>` generation-prompt suffix and `/nothink` — the OCR model is
prompted with a task string, not a conversation, and none of those is a path it is used on. Detect matches only this
checkpoint's template (the image markers are in the fingerprint), so GLM-4.5's text template, which shares the
`[gMASK]<sop>` opening and is rendered differently, is not captured by it.
```

## Ministral

Moved from `chat/templates.go` (the comment above `Ministral`) on 2026-10-10.

```text
Ministral — Ministral 3's own template (mistralai/Ministral-3-8B-Instruct-2512's real
chat_template.jinja, fetched and read 2026-09-16, not guessed from Mistral v0.3's shape): "<s>"
once, an EXPLICIT system message as its own "[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT]" block —
no "[INST] " space before user content (unlike v0.3's Mistral(), which has one), and no eos
leading space before an assistant turn either.

M-36 (audit-2026-09-10): this family was previously misdetected as Mistral() via the shared
"[INST]" substring — a different template version with a different id stream on every turn
(no [SYSTEM_PROMPT] at all, "[INST] " WITH a space, system folded into the last user turn
instead of rendered as its own block).

System placement: the real template emits [SYSTEM_PROMPT]...[/SYSTEM_PROMPT] wherever a
role=="system" message naturally falls in the conversation (validated to be first or absent by
its own role-ordering pass in the common case) — this renders it FIRST, right after "<s>" and
before any turn, which is exactly what the real template produces when the caller's system
message is the conversation's first message, the only shape goinfer's system-as-a-separate-
parameter Render(system, turns) interface can represent (there is no per-turn system Turn).

NOT implemented: the real template's default system message (injected only when the caller
provides NONE at all — a long, Mistral-product-branded prompt naming "Ministral-3-8B-
Instruct-2512" and "Le Chat" by identity, with {today}/{yesterday} date substitutions) is
deliberately NOT replicated here, the same documented divergence ChatML() already has for
Qwen's own default system message (chat.go's own "byte-exact" scope note) — injecting
Mistral's own product identity into a self-hosted goinfer response would be actively wrong, not
just an omission. The no-system-message case is therefore NOT byte-exact for this family
(goldens below cover explicit-system cases only, same convention as ChatML's).

Tool calling is NOT wired for this family (SupportsTools() declines it, chat/tools.go): the
real wire format ([TOOL_CALLS]name[ARGS]{json} per call, no JSON-array wrapping) differs from
the existing "mistral" tool dialect (a JSON array of call objects) enough that reusing it would
silently mis-render/mis-parse rather than simply be incomplete — declining is the honest choice
until that dialect is built for real, not a guess dressed up as support.

Segments (M25): every marker is a single control token of Ministral 3's tokenizer (<s> 1, </s> 2, [INST] 3,
[/INST] 4, [SYSTEM_PROMPT] 17, [/SYSTEM_PROMPT] 18), so they are sp() and the system and turn texts ct(): a marker a
user types stays literal, and an image block in a user turn is a content gap the vision splice can find (S10: as one
Special segment the whole prompt hid it, and every Pixtral request was refused).
```

## Template.ParseToolCallsFor

Moved from `chat/tools.go` (the comment above `Template.ParseToolCallsFor`) on 2026-10-10.

```text
ParseToolCallsFor is ParseToolCalls with the request's tool list in hand, which
lets it recover one more shape: a BARE call — the call object with its wrapper
left off — on the families whose wrapper is `<tool_call>` (AcceptsBareToolCall).

Why it exists: Qwen2.5-Coder at 0.5B, 1.5B and 7B practically never writes the
wrapper under tool_choice "auto"; it emits `{"name": ..., "arguments": {...}}` on
its own, which ParseToolCalls reads as prose, so the harness never gets a call.
Measured in docs/measurements/tool-call-failure-t0-2026-09-23.md: 0 wrapped calls
in 1,200 samples, and the same on Ollama. llama.cpp makes the same allowance for
Qwen3-Coder's first call.

It is deliberately narrow, because a prose answer that happens to be JSON must
stay prose:
  - only when ParseToolCalls found no call — a wrapped call is never re-read;
  - only when the output's first non-space byte opens a JSON object (a bare call
    has an empty lead, and prose-then-JSON is left alone);
  - the object's "name" must be a string EXACTLY equal to a supplied tool's name,
    and its "arguments" (or "parameters") must be a JSON object — so this can
    never produce a call to a tool the caller did not offer;
  - the FIRST object only. The 0.5B emits runs of speculative calls one after
    another; executing all of them would be worse than the prose it replaces.

Every other output, and every other family, gets exactly what ParseToolCalls
returns.
```

## tojsonEscape

Moved from `chat/tools.go` (the comment above `tojsonEscape`) on 2026-10-10.

```text
tojsonEscape applies the one HTML-safety substitution encoding/json's default escaping
(SetEscapeHTML's true default, on by construction here since we never turn it off) leaves out.
N-80 (docs/audit-2026-09-10.md): encoding/json already turns the raw bytes for less-than,
greater-than and ampersand into their six-character backslash-u-NNNN escapes — the same
substitution Jinja2's own htmlsafe_json_dumps (what the `| tojson` filter these chat templates
use calls) makes for those three. The fourth one, an apostrophe, is the one encoding/json has
no flag for, so it was missing here — a tool description or default value containing one
rendered one byte different from what the reference Jinja template would produce. (The
audit's own citation for this said Jinja renders it as the HTML entity for an apostrophe;
verified 2026-09-16 against jinja2's actual source (src/jinja2/utils.py,
htmlsafe_json_dumps) and it is the same backslash-u-NNNN form as the other three, not an
entity — correcting the claim rather than reproducing it.)
```

## renderGemma4Tools.system

Moved from `chat/gemma4_tools.go` (the comment above the system text in `renderGemma4Tools`) on 2026-10-10.

```text
No separator between the system text and the first declaration: upstream's template renders
"<|turn>system\n{system}<|tool>declaration:…" (measured against HF with a system prompt, 2026-09-30 —
the golden's only cases had none, so the "\n" that used to be written here was never compared).
```

## renderGemma4Tools.turns

Moved from `chat/gemma4_tools.go` (the comment above `openModelTurn` in `renderGemma4Tools`) on 2026-10-10.

```text
M-20: THE TOOL RESPONSES SIT INSIDE THE MODEL TURN THAT MADE THE CALL. goinfer used to
close the turn with "<turn|>\n" after the calls and then re-open a
"<|turn>model\n<|channel>thought\n<channel|>" scaffold after the response; the upstream
template does neither. Both are visible in testdata/chat_goldens/tools_gemma4.json's
`call_result` case, which was committed and never read by any test.

So the turn is closed lazily: an assistant turn whose calls are answered by tool turns
stays open until something that is not a tool response follows it.
```

## gemmaValue

Moved from `chat/gemma4_tools.go` (the comment above `gemmaValue`) on 2026-10-10.

```text
gemmaValue renders one argument value in Gemma's micro-language.

M-20: the default arm used to json.Marshal, so an object-typed parameter came out as
{"limit":5} — JSON, not Gemma syntax — inside a body the model reads as Gemma syntax. Upstream
renders a mapping as {k:v,…} with the same quoting as the top level, which is what the
declaration half of this file already does for schemas (gemmaSchema). Any tool with an
object-typed or array-typed parameter hit this.
```

## gemmaParseValue

Moved from `chat/gemma4_tools.go` (the comment above `gemmaParseValue`) on 2026-10-10.

```text
gemmaParseValue is gemmaValue's inverse for one rendered value: <|"|>-quoted → string,
true/false → bool, {…} → object, […] → array, else a number, else the bare text.

M-20: nested values had no case at all, so an object argument came back as two broken string
keys. Recursive here for the same reason gemmaValue is recursive — a renderer and a parser
that disagree about nesting produce arguments the tool silently mis-receives.
```

## splitGemmaPairs

Moved from `chat/gemma4_tools.go` (the comment above `splitGemmaPairs`) on 2026-10-10.

```text
splitGemmaPairs splits on commas that are neither inside a <|"|>…<|"|> string nor inside a
nested {…} / […].

M-20: it used to track quoting only, so opts:{limit:5,sort:<|"|>asc<|"|>} split at the INNER
comma and parsed to {"opts":"{limit:5","sort":"asc}"} — two keys, both wrong, no error.
```

## renderGemma4NativeTools

Moved from `chat/gemma4_tools.go` (the comment above `renderGemma4NativeTools`) on 2026-10-10.

```text
renderGemma4NativeTools renders system + turns + tool declarations the way Gemma 4's CANONICAL chat template does — a port of its
message loop, not a variation on renderGemma4Tools — and is what `-tool-format template` selects for such a checkpoint. The variable
names are the template's (prev_message_type, prev_non_tool_role, continues_into_next) so the two can be read side by side. Three
things differ from goinfer's own rendering, all of them the template's:

  - an assistant message's TEXT is written after its calls and their results, not before the calls;
  - the model turn is CLOSED (`<turn|>`) right after that text, when the message has results and text;
  - the generation prompt writes no turn header after a tool result, whether or not the turn was closed.

The last two make a prompt that ends `…<tool_response|>text<turn|>\n` with nothing after it when an agent client replays a turn that
had a preamble. Whether that is better or worse for the model than goinfer's own order is a question for a measurement, not for this
file: the format is opt-in.
```

Note added when this moved (2026-10-10): "the format is opt-in" is no longer true. `Detect` sets `nativeByDefault` for a canonical Gemma 4
template, so `auto` selects this renderer (`docs/measurements/gemma4-tool-text-order-2026-09-30/RESULTS.md` is the adoption). The code
comment was rewritten to say so.

## fenced_tool_calls.go.WithLenientToolCalls

Removed from the comment above `Template.WithLenientToolCalls` in `chat/fenced_tool_calls.go` on 2026-10-10 (the rule's motivation).

```text
the rule exists because Qwen2.5-Coder-7B, told to edit a file by opencode, answered
with the call in a ```json block twice and no edit happened.
```

## harmony_parse.go.header

Removed from the file-level comment of `chat/harmony_parse.go` on 2026-10-10 (the measurement and the pre-parser behaviour).

```text
Measured on
gpt-oss-20b-MXFP4 2026-09-30, docs/measurements/harmony-parser-2026-09-30/: the decoded stream carries these markers as literal
text and never carries the turn stops, which are stop ids. Before this parser every marker reached `content`.
```

The same file's comment above `harmonyCall` said "nothing surfaces it yet"; that went stale when `ReplySplitter.ToolCalls` and
`harmonyToolCalls` landed, and the code comment now says so.

## default_system.go.chatMLDefaultSystem

Removed from the comment above `chatMLDefaultSystem` in `chat/default_system.go` on 2026-10-10 (the decision it records).

```text
(owner, 2026-10-09: "every
template with one")
```

## history.go.dates

Dates dropped from two comments in `chat/history.go` on 2026-10-10: the history-rule header said the three real templates were read
on 2026-09-30, and the Mellum2.1 branch of `detectHistoryKind` said the template was read on 2026-10-07 (JetBrains/Mellum2.1-12B-A2.5B-Thinking).

## reasoning.go.dates

Dates dropped from `chat/reasoning.go` on 2026-10-10: the three ChatML generation-prompt shapes in `detectChatMLReasoning` were read from
real checkpoints on 2026-09-30, and the earlier Gemma 4 template in `detectOldGemma4Reasoning` was read from the E2B GGUF on 2026-10-01.
The file header's "Measured on the real templates, not assumed." is the same fact; the per-checkpoint pin is `TestThinkModes_matchHF`.

## Open work: segmenting the tool-rendered prompts and Mistral

Two follow-ups that the code comments in `chat/tools.go` and `chat/templates.go` used to name, kept here because no doc or queue entry owns them.
The code comments state the limitation (no injection hardening of those content spans).

```text
(RenderToolsSegments)  With tools declared it returns the tool-rendered prompt as a single Special segment — identical to the
whole-string Encode path (no regression); segmenting the per-family tool templates so their content spans are hardened too is a
follow-up.

(Mistral)  Statically deciding the special/content split would risk changing the tokenization of legitimate prompts, so this renderer
emits ONE Special segment (identical to whole-string Encode — no regression) and forgoes the injection hardening the others get.
Splitting it safely needs the loaded tokenizer's added-vocabulary, a follow-up.
```
