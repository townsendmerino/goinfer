//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// OP1 on CUDA, the resident-level tests of the attention softcap (docs/tasks/task-forward-architecture-2026-10.md §4.6a). The softcap's
// evidence is the kernel gates against float64 (TestAttnSoftcap_kernelsAgainstFloat64, with the cap dropped as the control) and the lane
// test below; the tiny Gemma 2 fixtures are diagnostics, because their scores sit far inside the cap, where cap·tanh(s/cap) is nearly the
// identity and a dropped cap can leave them green.

var gemma2Tiny = []string{"gemma2-tiny", "gemma2-hd-tiny"}

// gemma2TinyCosine runs one fixture through the CUDA resident token by token and through the CPU, both at quant, and returns the worst
// logit cosine over 16 positions and the resident's reported cap. capOff removes the attention softcap from BOTH sides.
func gemma2TinyCosine(t *testing.T, fx, quant string, capOff bool) (worst float64, capReported float32) {
	t.Helper()
	was := decoder.SetAttnSoftcapOffForTest(capOff)
	defer decoder.SetAttnSoftcapOffForTest(was)
	ckpt := filepath.Join("..", "testdata", fx)
	mRes, err := decoder.Load(ckpt, decoder.Options{Backend: "cuda", Quant: quant})
	if err != nil {
		t.Fatalf("load cuda: %v", err)
	}
	defer mRes.Close()
	rf := mRes.ResidentForwardForTest()
	if rf == nil {
		t.Fatalf("%s at %s did not go resident: decode path %q; decline: %s", fx, quant, mRes.DecodePath(), mRes.ResidentDecline())
	}
	mCPU, err := decoder.Load(ckpt, decoder.Options{Backend: "cpu", Quant: quant})
	if err != nil {
		t.Fatalf("load cpu: %v", err)
	}
	defer mCPU.Close()
	_, _, _, _, _, _, vocab := mCPU.Dims()
	cache := mCPU.NewCache(16)
	rf.Reset()
	worst = 1.0
	for i := range 16 {
		tok := (i*37 + 11) % vocab
		lr, err := rf.Forward(mRes.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("resident forward[%d]: %v", i, err)
		}
		lc, err := mCPU.ForwardForTest(tok, cache)
		if err != nil {
			t.Fatalf("cpu forward[%d]: %v", i, err)
		}
		cos, _ := cosF32(lc, lr)
		worst = min(worst, cos)
	}
	return worst, mRes.AttnLogitSoftcapResident()
}

// TestGemma2ResidentParityCUDA is G2-T, a diagnostic: both tiny fixtures at both quants, the cap on both sides and, beside it, the
// baseline of the same measurement with the cap removed on both sides, so the capped number is read against what the fixture's own
// int4 and int8int8 noise gives. Only a gross floor is asserted; the numbers are in the log and in the record.
func TestGemma2ResidentParityCUDA(t *testing.T) {
	requireCUDADevice(t)
	for _, fx := range gemma2Tiny {
		for _, quant := range []string{"int4", "int8int8"} {
			t.Run(fx+"/"+quant, func(t *testing.T) {
				capped, capV := gemma2TinyCosine(t, fx, quant, false)
				base, baseV := gemma2TinyCosine(t, fx, quant, true)
				if capV == 0 || baseV != 0 {
					t.Fatalf("the arms do not differ in the cap: capped arm reports %v, baseline arm %v", capV, baseV)
				}
				t.Logf("G2-T %s %s: worst logit cosine %.6f with the cap (%v), %.6f with it removed on both sides", fx, quant, capped, capV, base)
				if capped < 0.99 {
					t.Errorf("worst logit cosine %.6f < 0.99, the gross floor of a diagnostic", capped)
				}
			})
		}
	}
}

// TestGemma2Softcap_prefillBitIdentical: the batched prefill's exact path (attn_batched with the cap) writes the same K and V rows and the
// same last-token logits as decoding the prompt one token at a time (attention, or split-KV, with the cap): both forms are one
// attn_score, so a cap applied in one and not the other would differ here. The prompt runs past the 4-key sliding window.
func TestGemma2Softcap_prefillBitIdentical(t *testing.T) {
	requireCUDADevice(t)
	for _, fx := range gemma2Tiny {
		t.Run(fx, func(t *testing.T) {
			mc, err := decoder.Load(filepath.Join("..", "testdata", fx), decoder.Options{Backend: "cuda", Quant: "int4"})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer mc.Close()
			rf, ok := mc.ResidentForwardForTest().(*cudaResident)
			if !ok {
				t.Fatalf("not resident: %q, decline %s", mc.DecodePath(), mc.ResidentDecline())
			}
			if rf.attnCap == 0 || !rf.prefillReady {
				t.Fatalf("attnCap %v, prefillReady %v: this test no longer exercises the capped batched prefill", rf.attnCap, rf.prefillReady)
			}
			_, nLayers, _, _, _, _, vocab := mc.Dims()
			const n = 40
			embs := make([][]float32, n)
			for i := range n {
				embs[i] = append([]float32(nil), mc.EmbedResidentForTest((i*53+7)%(vocab-1))...)
			}
			rf.Reset()
			for i := range n - 1 {
				if e := rf.ForwardNoLogits(embs[i], i); e != nil {
					t.Fatalf("sequential pos %d: %v", i, e)
				}
			}
			seq, err := rf.Forward(embs[n-1], n-1)
			if err != nil {
				t.Fatal(err)
			}
			seq = append([]float32(nil), seq...)
			seqK, seqV := make([][]float32, nLayers), make([][]float32, nLayers)
			for l := range nLayers {
				seqK[l], seqV[l] = rf.readKVForTest(l, n)
			}
			rf.Reset()
			bat, err := rf.PrefillLast(context.Background(), embs, 0)
			if err != nil {
				t.Fatalf("PrefillLast: %v", err)
			}
			for l := range nLayers {
				bk, bv := rf.readKVForTest(l, n)
				for i := range bk {
					if bk[i] != seqK[l][i] || bv[i] != seqV[l][i] {
						t.Fatalf("layer %d: K or V row differs between the batched prefill and the sequential decode (element %d)", l, i)
					}
				}
			}
			for i := range seq {
				if bat[i] != seq[i] {
					t.Fatalf("last-token logit %d: batched %v, sequential %v", i, bat[i], seq[i])
				}
			}
		})
	}
}

// TestGemma2Softcap_lanesCUDA is G2-L: the lane selectors keep a capped layer off every kernel that has no cap. The tiny fixtures have head
// dimension 32, which flash-decode and the fused kernels never serve, so each selector is asked at a head dimension it does serve, with the
// resident's own state set up as a served one would be, and each is checked twice: with the cap off it says yes (the control: the test can
// see the lane), with the cap on it says no.
func TestGemma2Softcap_lanesCUDA(t *testing.T) {
	requireCUDADevice(t)
	t.Setenv("GOINFER_CUDA_FLASH_DECODE", "16") // the test binary turns the lane off by default; this turns it on so the control can see it
	t.Setenv("GOINFER_CUDA_FLASH_DECODE_MIN_KEYS", "0")
	mc, err := decoder.Load(filepath.Join("..", "testdata", "gemma2-tiny"), decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer mc.Close()
	rf, ok := mc.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatalf("not resident: %q, decline %s", mc.DecodePath(), mc.ResidentDecline())
	}
	capV := rf.attnCap
	if capV == 0 {
		t.Fatal("the fixture's resident has no attention cap")
	}
	rf.layers[0].hd, rf.layers[0].nKV = 64, 2 // a geometry both lanes serve (G = 2)
	rf.fastAttn, rf.forceExactKernels, rf.passPromptLen = true, false, 4096
	if rf.bAttnFused64 == (Pipeline{}) || rf.faCombine == (Pipeline{}) {
		t.Fatalf("the fused (%v) and flash-decode (%v) pipelines did not load: the control cannot see the lanes", rf.bAttnFused64 != (Pipeline{}), rf.faCombine != (Pipeline{}))
	}
	fa := func() bool { return rf.faEligible(0) }
	fused := func() bool { _, use := rf.useAttnFused(64, 4096); return use }
	rf.attnCap = 0
	if !fa() || !fused() {
		t.Fatalf("VOID control: with no cap the flash-decode selector says %v and the fused-prefill selector %v; both should say yes at hd 64", fa(), fused())
	}
	rf.attnCap = capV
	if fa() {
		t.Error("a capped layer is eligible for the flash-decode lane (and its verify rows, which ask the same question)")
	}
	if fused() {
		t.Error("a capped layer takes the fused prefill kernels")
	}
	// an image block on a capped model is refused by name
	if _, _, err := rf.prefillCore(context.Background(), make([][]float32, 4), 0, tailLastLogits, [][2]int{{0, 4}}, nil, nil); err == nil || !strings.Contains(err.Error(), "attention softcap") {
		t.Errorf("an image block on a capped model: got %v, want a refusal naming the attention softcap", err)
	}
	// the block drafter is declined by name
	if _, err := rf.AttachDrafter(nil); err == nil || !strings.Contains(err.Error(), "attention softcap") {
		t.Errorf("AttachDrafter on a capped target: got %v, want a decline naming the attention softcap", err)
	}
}
