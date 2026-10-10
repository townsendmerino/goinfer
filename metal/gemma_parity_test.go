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
// never a hardcoded id literal: ids nobody has decoded can be gibberish, and a parity number measured on gibberish is a
// confound (flat logits from nonsense produce exactly the extra near-ties that get blamed on architecture).
//
// Both models probe the SAME sentence, so they are comparable. Encode is used where
// the vocab has merge ranks; Gemma's GGUF ships scores instead, so its pieces are looked up
// directly — either way the result is decoded back and logged, so a bad prompt cannot hide.
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

// residentParity drives Metal resident decode against the CPU forward in greedy LOCKSTEP — the
// shipped metal convention (model_test.go): the CPU's argmax drives both sides, so they walk a
// coherent trajectory instead of an arbitrary id sequence full of near-ties.
//
// On the bar: CUDA's 3%-near-tie bar does NOT transfer to Metal, so the bar is read off the committed control
// (TestDenseResidentParity), not assumed; a threshold is the usual bug. It was set when Metal had no like-for-like CPU
// reference: BuildResident re-quantized an int8 load to its own W4A8 (group=32, scale=max/7), which no CPU load reproduces,
// so the pair was int4-GPU vs int8-CPU. Since slice 1 of docs/tasks/task-metal-int8-2026-10.md a dense int8 model runs its
// int8 weights natively, so for those this is int8 against int8, and TestW8Native_F2 holds the pair to the int4 pair's
// agreement.
func residentParity(t *testing.T, path string, seed []int, steps int) parityStats {
	t.Helper()
	return residentParityAt(t, path, "int8int8", seed, steps)
}

// residentParityAt is residentParity with both sides loaded at quant.
func residentParityAt(t *testing.T, path, quant string, seed []int, steps int) parityStats {
	t.Helper()
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s", path)
	}
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	mg, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: quant})
	if err != nil {
		t.Fatalf("load (metal): %v", err)
	}
	defer mg.Close()
	rf := mg.ResidentForwardForTest()
	if rf == nil { // without this, a silent CPU fallback would pass every assertion trivially
		skipIfMemoryDeclined(t, mg)
		t.Fatalf("metal resident DECLINED (%s) — admission says it should be admitted", mg.ResidentDecline())
	}
	mcpu, err := decoder.Load(path, decoder.Options{Quant: quant})
	if err != nil {
		t.Fatalf("load (cpu): %v", err)
	}
	defer mcpu.Close()
	_, nL, _, nKV, hd, _, _ := mcpu.Dims()
	cache := decoder.NewKVCache(nL, nKV, hd, 0, 1024, nil)

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
		// Skip the <bos> sink positions in the metric. Gemma's <bos> is an ATTENTION SINK whose value vector is trained
		// near-zero, so a cosine there is dominated by rounding — and the position after it attends to that sink. Both read
		// as catastrophic while the model is fine (it generates " Paris." correctly); gating on min-cosine over these
		// positions reports the two places the metric is meaningless.
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
	t.Logf("%s at %s (%s): %d/%d argmax-exact, worst near-tie %.3f%%, %d gaps >3%%, min cosine %.6f",
		path, quant, mg.DecodePath(), st.exact, st.steps, st.worstTie*100, st.hard, st.minCos)
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
	// Skipped unless Metal declares the Gemma features (it does): residentParity t.Fatals on a decline, which is what
	// catches a silent fallback.
	if !decoder.ResidentBackendFeatures("metal")[decoder.FeatSandwichNorm] {
		t.Skip("metal does not declare the Gemma features yet (kernels dormant) — see docs/task-metal-gemma.md")
	}
	path := os.ExpandEnv("$HOME/models/gemma-3-4b-it-Q4_K_M.gguf")
	st := residentParity(t, path, seedPrompt(t, path, probeText), 24)
	assertParity(t, "gemma3", st, 0.88) // int4-hostile: floor 0.92 (quantbar), 0.88 with margin
}
