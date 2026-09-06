//go:build darwin && goinfer_testhooks

package metal

import (
	"os"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// seedPrompt derives the probe from the MODEL'S OWN tokenizer and VERIFIES it by decoding —
// never a hardcoded id literal.
//
// This gate used to carry invented ids that nobody had ever decoded. Gemma's read
// "<bos>ath হই of carry Bত্ব忽视ardRep" — not Gemma tokens at all — so every Gemma parity number
// on either backend was measured on gibberish, and the resulting "Gemma is noisier than the
// control" was a rationalization of a confound in the test data. The control's were no better
// ("The history of_init with a text of a **"): real tokens, near-nonsense text. Flat logits from
// nonsense produce exactly the extra near-ties that got blamed on architecture.
//
// Both models now probe the SAME sentence, so they are actually comparable. Encode is used where
// the vocab has merge ranks; Gemma's GGUF ships scores instead, so its pieces are looked up
// directly — either way the result is decoded back and logged, so a bad prompt cannot hide again.
func seedPrompt(t *testing.T, path, text string) []int {
	t.Helper()
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Skipf("tokenizer: %v", err)
	}
	ids, err := tk.Encode(text, true)
	if err != nil { // decode-only vocab (no merges): resolve pieces via the vocab itself
		ids = nil
		if bos, ok := tk.TokenID("<bos>"); ok {
			ids = append(ids, bos)
		}
		for word := range strings.FieldsSeq(text) {
			id, ok := tk.TokenID("▁" + word) // SPM marks a leading space with ▁
			if !ok {
				t.Skipf("vocab lookup failed for %q — cannot build a verified prompt", word)
			}
			ids = append(ids, id)
		}
	}
	got, derr := tk.Decode(ids)
	if derr != nil {
		t.Fatalf("decode-back: %v", derr)
	}
	t.Logf("prompt %q → %v → decodes to %q", text, ids, got)
	return ids
}

// observeCos folds one position's logit cosine into the running min. NaN/Inf is COUNTED, never
// fed to the `< minCos` reduction: `NaN < x` is false in Go, so a degenerate (NaN) cosine — the
// signature of the worst bugs — would otherwise never update minCos and would sail through the
// floor as if parity held. This is the exact vacuity parity-coverage-policy.md § Falsifiable
// names, and TestParity_NaNCosineFailsTheGate breaks it on purpose to prove this guard fires.
// residentParity drives Metal resident decode against the CPU forward in greedy LOCKSTEP — the
// shipped metal convention (model_test.go): the CPU's argmax drives both sides, so they walk a
// coherent trajectory instead of an arbitrary id sequence full of near-ties.
//
// On the bar: Metal has NO like-for-like CPU reference. BuildResident requires an int8 load and
// re-quantizes to its own W4A8 (group=32, scale=max/7), which no CPU load reproduces — so this
// is always int4-GPU vs int8-CPU. CUDA's 3%-near-tie bar does NOT transfer; it compares
// int4-vs-int4. Measured here on the KNOWN-GOOD dense path, CUDA's bar fails. That is why the
// control is committed: the bar is read off it, not assumed — the same lesson CUDA learned when
// a cosine >= 0.999 draft failed its own shipped path. The threshold is the usual bug.
func residentParity(t *testing.T, path string, seed []int, steps int) parityStats {
	t.Helper()
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s", path)
	}
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	mg, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load (metal): %v", err)
	}
	rf := mg.ResidentForwardForTest()
	if rf == nil { // without this, a silent CPU fallback would pass every assertion trivially
		t.Fatal("metal resident DECLINED — admission says it should be admitted")
	}
	mcpu, err := decoder.Load(path, decoder.Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load (cpu): %v", err)
	}
	_, nL, _, nKV, hd, _, _ := mcpu.Dims()
	cache := decoder.NewKVCache(nL, nKV, hd, 0, 1024)

	st := parityStats{steps: steps, minCos: 1}
	tok := seed[0]
	for i := range steps {
		cpuL, err := mcpu.ForwardForTest(tok, cache)
		if err != nil {
			t.Fatalf("cpu forward: %v", err)
		}
		// The GPU side goes through EmbedResidentForTest — that is what applies Gemma's ×√hidden
		// embed scale, so FeatEmbedScale is under test only via this call.
		gpuL, err := rf.Forward(mg.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("gpu forward: %v", err)
		}
		// Skip the <bos> sink positions in the metric. Gemma's <bos> is an ATTENTION SINK whose
		// value vector is trained near-zero (|V| 9.4 vs 129 for an ordinary token), so a cosine
		// there is dominated by rounding — and the position after it attends to that sink. Both
		// read as catastrophic (-0.047) while the model is fine: measured dNLL decays 15.9 -> 0.06
		// nats as real keys accumulate, and it generates " Paris." correctly. Gating on min-cosine
		// over these positions reported the two places the metric is meaningless.
		if i >= 2 {
			st.observeCos(cosF(cpuL, gpuL))
		}
		ca, ga := argmaxF(cpuL), argmaxF(gpuL)
		if ca == ga {
			st.exact++
		} else {
			lo, hi := cpuL[0], cpuL[0]
			for _, v := range cpuL {
				if v < lo {
					lo = v
				}
				if v > hi {
					hi = v
				}
			}
			gap := float64(cpuL[ca]-cpuL[ga]) / (float64(hi-lo) + 1e-9)
			if gap > st.worstTie {
				st.worstTie = gap
			}
			if gap > 0.03 {
				st.hard++
			}
		}
		if i+1 < len(seed) { // seed the prompt, then greedy-continue off the CPU
			tok = seed[i+1]
		} else {
			tok = ca
		}
	}
	t.Logf("%s: %d/%d argmax-exact, worst near-tie %.3f%%, %d gaps >3%%, min cosine %.6f",
		path, st.exact, st.steps, st.worstTie*100, st.hard, st.minCos)
	return st
}

// The one sentence both models probe, so the subject and the control are comparable.
const probeText = "The capital of France is"

// TestDenseResidentParity is the CONTROL: the same harness on the known-good SHIPPED dense Qwen
// path, so "is this cosine / this many near-ties good?" has an answer measured on THIS box
// rather than assumed. Without it a Gemma number cannot be judged.
func TestDenseResidentParity(t *testing.T) {
	requireHeavyModel(t)
	path := os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	st := residentParity(t, path, seedPrompt(t, path, probeText), 24)
	assertParity(t, "dense control", st, 0.95)
}

// TestGemma3ResidentParity is the Metal Gemma 3 gate — judged by the SAME bar the control meets.
// Needs the checkpoint (~4 GB at int8, loaded twice → budget ~10 GB); skips without it.
func TestGemma3ResidentParity(t *testing.T) {
	requireHeavyModel(t)
	// Dormant until the kernels are validated: metal ships the Gemma kernels but does not yet
	// DECLARE the features, so gemma3 declines to CPU. Skip rather than fail — and the moment
	// the declaration lands this becomes a live gate (residentParity t.Fatals on a decline,
	// which is what catches a silent fallback).
	if !decoder.ResidentBackendFeatures("metal")[decoder.FeatSandwichNorm] {
		t.Skip("metal does not declare the Gemma features yet (kernels dormant) — see docs/task-metal-gemma.md")
	}
	path := os.ExpandEnv("$HOME/models/gemma-3-4b-it-Q4_K_M.gguf")
	st := residentParity(t, path, seedPrompt(t, path, probeText), 24)
	assertParity(t, "gemma3", st, 0.88) // int4-hostile: floor 0.92 (quantbar), 0.88 with margin
}
