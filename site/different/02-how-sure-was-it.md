---
title: "How sure was it?"
area: "Structured output"
order: 2
summary: "Each enum, boolean and integer field of a constrained JSON answer comes back with the model's own probability over the answers the schema allowed."
stand: "Ask for a JSON answer, and each choice, yes/no and count in it comes back with the model's own probability over the answers your schema allowed."
measured: 2026-09-27
reviewed: 2026-09-29
raw: true
facts:
  - {label: "turn it on", value: "`\"goinfer_confidence\": true`"}
  - {label: "costs", value: "1–4% of each token's time"}
  - {label: "fields", value: "enum, boolean, integer"}
  - {label: "calibrated", value: "no"}
doesnt:
  - title: "It isn't the probability that the answer is right."
    text: "It's the model's probability over what the schema allowed, which is a narrower thing. It was tested for ranking (is a low number wrong more often?), not for calibration, and every field it reports says `\"calibrated\": false`. A 0.9 doesn't mean nine times in ten."
  - title: "It doesn't cover numbers or free text."
    text: "Only enum, boolean and integer fields are reported. Number and string fields are not: the test set produced too few wrong numbers and names to judge them."
  - title: "It rests on one small test."
    text: "60 support tickets, one kind of task, written by us, and two models from one family (Qwen2.5). A harder labelled set would say more, and might move the integer result either way."
  - title: "It doesn't work everywhere a schema does."
    text: "The server refuses the flag with a 400, rather than dropping it silently, on requests with tools or images, on `/v1/jobs` and batches, and on `/v1/responses` and `/v1/messages`. It also turns off grammar-fused speculative decoding for that request: the speed-up that guesses several tokens ahead and checks them in one step."
  - title: "It doesn't make a small model good."
    text: "A 1.5B-parameter model that is unsure is often right to be. The number tells you where to look, not what the answer should have been."
figures:
  - {text: "0.847", source: "docs/measurements/confidence-c0-2026-09-27.md"}
  - {text: "0.967", source: "docs/measurements/confidence-c0-2026-09-27.md"}
  - {text: "0.727", source: "docs/measurements/confidence-c0-2026-09-27.md"}
  - {text: "0.905", source: "docs/measurements/confidence-c0-2026-09-27.md"}
  - {text: "0.680", source: "docs/measurements/confidence-c0-2026-09-27.md"}
  - {text: "0.942", source: "docs/measurements/confidence-c0-2026-09-27.md"}
  - {text: "4.20%", source: "docs/measurements/confidence-c0-2026-09-27.md"}
  - {text: "0.570 ms", source: "docs/measurements/confidence-c0-2026-09-27.md"}
  - {text: "13.56 ms", source: "docs/measurements/confidence-c0-2026-09-27.md"}
  - {text: "1.44%", source: "docs/measurements/confidence-c0-2026-09-27.md"}
  - {text: "0.542", source: "docs/measurements/confidence-c0-2026-09-27.md"}
  - {text: "37.55 ms", source: "docs/measurements/confidence-c0-2026-09-27.md"}
  - {text: "0.65", source: "docs/tasks/parked/task-constrained-confidence.md"}
  - {text: "0.55", source: "docs/tasks/parked/task-constrained-confidence.md"}
  - {text: "0.29 ms", source: "docs/tasks/parked/task-constrained-confidence.md"}
  - {text: "0.28 ms", source: "docs/tasks/parked/task-constrained-confidence.md"}
sources:
  - "docs/server.md"
  - "docs/measurements/confidence-c0-2026-09-27.md"
  - "docs/tasks/parked/task-constrained-confidence.md"
  - "constrain/confidence.go"
  - "examples/confidence"
---
<h2>The problem</h2>
<p class="lead">A constrained answer always looks sure of itself.</p>
<p>goinfer can already <a href="/different/04-a-go-struct-the-model-cant-break/">force a model's output into a JSON schema</a>, so the reply always parses and every field holds a value the schema allows. That's useful, and it hides something. A model that knew the answer and a model that picked between two options by a hair produce the same clean JSON. Nothing in the reply tells them apart.</p>
<p>If you're sorting support tickets, or reading invoices, or deciding which requests an agent may act on alone, that difference is the one you care about.</p>

<h2>What goinfer does</h2>
<p>A model writes its answer one token at a time; a token is a word or a piece of one. At every step of a constrained answer, the model scores every possible next token, goinfer works out which of them the schema allows, and it throws the other scores away. With confidence turned on, it also reads what the model thought at the one point where each field was decided, and reports it next to the answer.</p>

<div class="demo" aria-label="Example">
  <div class="demo-head">
    <span class="t">example · Qwen2.5-Coder 1.5B · triage schema</span>
    <div class="seg" id="tix" role="group" aria-label="Ticket">
      <button type="button" data-i="0" aria-pressed="true">Two lamps</button>
      <button type="button" data-i="1">Charged twice</button>
      <button type="button" data-i="2">Broken link</button>
    </div>
  </div>
  <div class="ticket"><span class="l">ticket</span><span id="tt"></span></div>
  <ul class="fields" id="flds"></ul>
  <div class="route">
    <label for="th">Send to a person when any field is below <input type="range" id="th" min="0.5" max="0.95" step="0.05" value="0.7"><output id="tho" for="th">0.70</output></label>
    <div class="verdict" id="verdict"></div>
  </div>
  <div class="demo-foot">Illustrative values, in the shape the server returns. The measured results are further down.</div>
</div>

<p>That threshold is the intended use. The number ranks answers, and a low one is wrong more often, so you can route on it: handle the confident ones, and send the rest to a person.</p>

<h2>Why only the deciding token</h2>
<p>Most tokens in a constrained answer aren't a choice at all. Once a model has started writing <code>"sh</code>, the schema allows only <code>shipping</code>, so the rest of the word and its closing quote have probability 1.0. Averaging those in would make every enum look about two-thirds certain before the model had a say.</p>
<div class="fig">
  <div class="toks" aria-label="Tokens of one enum value">
    <div class="tk"><code>"category":</code><small>structure</small></div>
    <div class="tk free"><code>"sh</code><small>decides</small></div>
    <div class="tk"><code>ipping</code><small>forced</small></div>
    <div class="tk"><code>"</code><small>forced</small></div>
  </div>
  <p class="figcap">An enum value is three tokens and one decision. goinfer reads the model at the free token and adds up the probability by the option each allowed token would spell. Where the tokens split is illustrative.</p>
</div>
<p>So a field's confidence comes only from tokens where the schema left the model a real choice. A field the schema forces outright, like an enum with one option, isn't reported, because its value says nothing about the model. An integer reports the least certain of its digits.</p>

<h2>What was measured</h2>
<p>Before anything was built, the question was whether this number means anything. We wrote 60 support tickets whose correct labels follow from rules in the prompt. On 2026-09-27 we ran two models over them on a MacBook Pro (M1 Pro, 16 GB), on its GPU through Metal. The models were Qwen2.5-Coder 1.5B Instruct and Qwen2.5 7B Instruct, with weights stored in about 4 bits each (Q4_K_M quantization). Each always took its most likely next token (greedy decoding). The bar was set in writing before the graded runs (pre-registered): for each field kind, the confidence must separate right answers from wrong ones with an AUROC of at least 0.65. Below 0.55 the kind would fail; in between, it would be left undecided.</p>
<p class="note">AUROC, in plain terms: pick one right answer and one wrong answer at random. How often did the right one carry the higher confidence? 0.5 is a coin toss. 1.0 is perfect.</p>

<div class="fig" aria-label="AUROC by field kind">
  <div class="scale" id="scale"></div>
  <div class="legend"><span><i></i>1.5B: counts, enough mistakes to grade</span><span><i class="h"></i>7B: too few mistakes to count</span><span>shaded: below 0.55 would have failed</span></div>
</div>

<div class="tablewrap"><table>
  <thead><tr><th>field kind</th><th>1.5B AUROC (right / wrong)</th><th>7B AUROC (right / wrong)</th><th>result</th></tr></thead>
  <tbody>
    <tr><td>enum</td><td class="n">0.847 (48 / 12)</td><td class="n">0.967 (55 / 5)</td><td class="pass">reported</td></tr>
    <tr><td>boolean</td><td class="n">0.727 (38 / 22)</td><td class="n">0.905 (58 / 2)</td><td class="pass">reported</td></tr>
    <tr><td>integer</td><td class="n">0.680 (43 / 17)</td><td class="n">0.942 (57 / 3)</td><td class="pass">reported</td></tr>
    <tr><td>number</td><td class="n">— (60 / 0)</td><td class="n">— (60 / 0)</td><td class="park">not reported yet</td></tr>
    <tr><td>string</td><td class="n">— (57 / 3)</td><td class="n">— (60 / 0)</td><td class="park">not reported yet</td></tr>
  </tbody>
</table></div>
<p>A model counted for a field kind only if it had at least 8 right and 8 wrong answers of that kind. The 7B got almost everything right, which is good for the 7B and useless for grading: a confidence can't be judged against mistakes that didn't happen. Its numbers point the same way and decide nothing. Integer cleared the bar on one model by 0.03, on 17 wrong answers. Numbers and free text need a harder test set, so they aren't reported until one exists.</p>

<div class="stats">
  <div class="stat"><div class="n">4.20%</div><div class="l">of a token's time on the 1.5B (0.570 ms of 13.56 ms). Median 1.88%.</div></div>
  <div class="stat"><div class="n">1.44%</div><div class="l">of a token's time on the 7B (0.542 ms of 37.55 ms). Median 0.72%.</div></div>
</div>
<p>Those two figures are from the test, which read the model at every position. The version that shipped reads only the positions that decide a value and skips free text, where nearly every token is allowed. Its own cost test added 0.29 ms at an object key, 0.28 ms at an enum value and nothing inside a free string, about 2% (1.5B) and 0.8% (7B) of a token at the positions it reads. That test ran on a busy machine, so read it as indicative. It's off unless you ask for it, and the answer itself doesn't change: the same request without the flag returns the same content.</p>

<h2>Use it</h2>
<div class="code">
  <div class="seg" id="codetabs" role="group" aria-label="Language">
    <button type="button" data-c="http" aria-pressed="true">HTTP</button>
    <button type="button" data-c="go">Go</button>
  </div>
<pre id="c-http"><span class="cm"># with goinfer-serve -model &lt;model.gguf&gt; running, any /v1/chat/completions
# or /v1/completions request with a json_schema</span>
curl -s localhost:8080/v1/chat/completions -d '{
  "messages": [{"role": "user", "content": "Triage this ticket: …"}],
  "response_format": {"type": "json_schema", "json_schema": {
    "name": "triage", "schema": { … }}},
  "goinfer_confidence": true
}'

<span class="cm"># the response gains one record per reported field</span>
"goinfer_confidence": [
  {"path": "category", "kind": "enum", "value": "shipping",
   "confidence": 0.60, "free_tokens": 1, "calibrated": false,
   "distribution": {"billing": 0.02, "shipping": 0.60, "technical": 0.37,
             "other": 0.01}},
  …
]</pre>
<pre id="c-go" hidden><span class="cm">// the masker that forces the schema also records what it saw</span>
mask := constrain.NewMasker(g, constrain.TokenBytes(vocab, tok.TokenText), stop).
    StopWhenComplete().
    CaptureConfidence(constrain.ConfidenceOptions{})

ch, gen := m.Generate(ctx, ids, 128, decoder.SamplingParams{
    LogitProcessor: mask.Process, StopIDs: stop})
<span class="cm">// … collect the generated ids into out …</span>

fields, err := mask.FieldConfidence(out)
for _, f := range fields {
    if f.Confidence &lt; 0.7 {
  escalate(f.Path, f.Value) <span class="cm">// route on it; don't read it as P(right)</span>
    }
}
<span class="cm">// full program: examples/confidence. From a clone of the repo:
// go run ./examples/confidence &lt;model.gguf&gt; "&lt;ticket&gt;"</span></pre>
</div>

<p>If all you need is a pick between a few options, the server's <a href="/different/05-decisions-without-generating/">decisions</a> endpoint answers from a single read of the prompt, with no generation at all, in the request shape some decision APIs already use.</p>
