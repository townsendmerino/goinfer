package chat

import (
	"testing"
	"time"
)

// The harmony (gpt-oss) renderer, pinned against HuggingFace's own template.
//
// gpt-oss was previously UNREACHABLE through chat.Detect: its HF checkpoint ships no
// chat_template at all (length 0) and its markers matched none of Detect's branches, so every
// gpt-oss prompt fell through to the raw-completion path. That is a real gap — goinfer supports
// the family for inference while being unable to hold a conversation with it.
//
// The expected string below is NOT hand-written. It is what
// `tokenizer.apply_chat_template(..., chat_template=<the 16.7 KB template from the MXFP4 GGUF>)`
// produced, and goinfer's render was verified to encode to the SAME 78 token ids against the
// real gpt-oss vocab. Byte equality here plus that id check is what makes the renderer
// trustworthy; a hand-written expectation would only test that the code does what I typed.
//
// The clock is pinned because harmony's system preamble carries a LIVE date (strftime_now
// upstream), so an unpinned test would pass only on the day it was written.
func TestHarmony_byteExact(t *testing.T) {
	defer func() { timeNow = time.Now }()
	timeNow = func() time.Time { return time.Date(2026, 8, 16, 12, 0, 0, 0, time.UTC) }

	const want = "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n" +
		"Knowledge cutoff: 2024-06\n" +
		"Current date: 2026-08-16\n\n" +
		"Reasoning: medium\n\n" +
		"# Valid channels: analysis, commentary, final. Channel must be included for every message." +
		"<|end|><|start|>user<|message|>Write a Python function that returns the nth Fibonacci number." +
		"<|end|><|start|>assistant"

	got := Harmony().Render("", []Turn{
		{Role: "user", Content: "Write a Python function that returns the nth Fibonacci number."},
	})
	if got != want {
		t.Errorf("harmony render differs from HuggingFace's\n got: %q\nwant: %q", got, want)
	}
}

// Detect must resolve harmony, and must NOT confuse it with Gemma 4. The two are one
// character apart in the marker Detect keys on: Gemma tests "<|channel>", harmony's is
// "<|channel|>", and "<|channel|>" does NOT contain "<|channel>" because the trailing pipe
// breaks the substring. That near-miss is why the ordering is asserted rather than assumed —
// a Detect that returned Gemma4() here would render gpt-oss through turn markers and produce
// a plausible, wrong prompt rather than an error.
func TestDetect_harmonyNotGemma4(t *testing.T) {
	// bare checkpoint (no chat_template) — the vocab-marker route, which is what the real
	// gpt-oss HF checkpoint hits, since it ships no template at all.
	harmonyVocab := map[string]bool{"<|start|>": true, "<|message|>": true, "<|channel|>": true, "<|end|>": true}
	got, err := Detect(Meta{HasToken: func(s string) bool { return harmonyVocab[s] }})
	if err != nil {
		t.Fatalf("Detect(bare gpt-oss vocab): %v", err)
	}
	if got.Name() != "harmony" {
		t.Errorf("bare gpt-oss vocab resolved to %q, want harmony", got.Name())
	}

	// Gemma 4 must still win on its own markers, which is the regression this ordering risks.
	g4, err := Detect(Meta{ChatTemplate: "…{{ '<|turn>' }}…{{ '<|channel>thought' }}…"})
	if err != nil {
		t.Fatalf("Detect(gemma4 template): %v", err)
	}
	if g4.Name() != "gemma4" {
		t.Errorf("gemma4 template resolved to %q — harmony's branch stole it", g4.Name())
	}
}
