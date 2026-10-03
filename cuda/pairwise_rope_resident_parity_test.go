//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// WHY THIS GATE EXISTS. TestCohereResidentParityCUDA (cohere_resident_parity_test.go) loads the
// committed cohere-tiny / cohere2-tiny, whose weights are ~0.02 std: attention is nearly UNIFORM
// there, a softmax over near-equal scores is blind to which key is which, and so a WRONG ROTATION
// (the NeoX half-split kernels run on a GPT-J pairwise family) leaves the logits at cosine 0.9997.
// The real checkpoints were not blind: Command-R7B and Aya-expanse-8B at int4 on this very resident,
// per-position resident-vs-CPU worst cosine -0.075 / -0.041 (docs/measurements/cuda-pairwise-rope-2026-10-01.md).
//
// So this gate PEAKS the attention: it derives, at test time and in a temp dir, a checkpoint from the
// committed fixture with every 2-D weight scaled by peakedScale (0.02 -> ~0.25 std, the factor the
// owner's session measured: cosine 0.06 with NeoX kernels against 0.9997 flat). Deterministic, no new
// binary in the tree, and the same bytes the CPU path loads.
//
// IT MUST BE ABLE TO FAIL, and proves that on every run: after the real measurement it swaps the NeoX
// rope pipelines (rope_kv / rope_kv_batched) into the SAME resident and re-measures; that arm must
// fall below the bar, or the gate has gone blind (the discrimination control, not a mock).
//
// Three paths, because they are three kernels: sequential decode (rope_kv_pw), batched prefill
// (rope_kv_batched_pw), and decode AFTER a batched prefill (reads the K the batched kernel stored).
// The prompt is 40 tokens: past 32 positions, and past cohere2-tiny's 8-token sliding window.

const (
	peakedScale   = 12.5 // 0.02-std fixture weights -> ~0.25
	pairwisePos   = 40
	pairwiseSplit = 32 // batched prefill length; positions [32,40) decode after it
	cosBar        = 0.995
	relBar        = 0.15 // vs 0.10 on the flat gate: peaked attention amplifies int8 requantization noise slightly; the NeoX arm reads far outside either
)

// peakedCheckpoint copies fixture src into a temp dir with every 2-D F32 tensor scaled by factor.
// safetensors layout is [8-byte LE header length][JSON header][raw data]; scaling in place leaves the
// header and every offset untouched.
func peakedCheckpoint(t testing.TB, src string, factor float32) string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(src, "model.safetensors"))
	if err != nil {
		t.Fatalf("read fixture weights: %v", err)
	}
	n := binary.LittleEndian.Uint64(raw[:8])
	var hdr map[string]json.RawMessage
	if err := json.Unmarshal(raw[8:8+n], &hdr); err != nil {
		t.Fatalf("safetensors header: %v", err)
	}
	data := raw[8+n:]
	scaled := 0
	for name, h := range hdr {
		if name == "__metadata__" {
			continue
		}
		var e struct {
			Dtype   string   `json:"dtype"`
			Shape   []int    `json:"shape"`
			Offsets [2]int64 `json:"data_offsets"`
		}
		if err := json.Unmarshal(h, &e); err != nil {
			t.Fatalf("tensor %s header: %v", name, err)
		}
		if len(e.Shape) != 2 {
			continue // norms stay as the fixture has them
		}
		if e.Dtype != "F32" {
			t.Fatalf("tensor %s dtype %s: the peaked-checkpoint transform only handles F32", name, e.Dtype)
		}
		for off := e.Offsets[0]; off < e.Offsets[1]; off += 4 {
			v := math.Float32frombits(binary.LittleEndian.Uint32(data[off:])) * factor
			binary.LittleEndian.PutUint32(data[off:], math.Float32bits(v))
		}
		scaled++
	}
	if scaled < 8 {
		t.Fatalf("scaled only %d tensors — the transform did not find the weights", scaled)
	}
	dst := t.TempDir()
	if err := os.WriteFile(filepath.Join(dst, "model.safetensors"), raw, 0o644); err != nil {
		t.Fatal(err)
	}
	for _, f := range []string{"config.json", "generation_config.json"} {
		if b, err := os.ReadFile(filepath.Join(src, f)); err == nil {
			if err := os.WriteFile(filepath.Join(dst, f), b, 0o644); err != nil {
				t.Fatal(err)
			}
		}
	}
	return dst
}

type pairwiseMetrics struct {
	decodeCos, decodeRel   float64 // sequential decode, every position of the prompt
	prefillCos, prefillRel float64 // batched prefill, last-token logits
	mixedCos, mixedRel     float64 // decode at positions [split, n) AFTER a batched prefill of [0, split)
}

func (m pairwiseMetrics) ok() bool {
	return m.decodeCos >= cosBar && m.prefillCos >= cosBar && m.mixedCos >= cosBar &&
		m.decodeRel <= relBar && m.prefillRel <= relBar && m.mixedRel <= relBar
}

// measurePairwiseParity runs the three paths on rf (a resident) against the CPU model at the same quant.
func measurePairwiseParity(t *testing.T, mRes, mCPU *decoder.Model, cr *cudaResident, prompt []int) pairwiseMetrics {
	t.Helper()
	m := pairwiseMetrics{decodeCos: 1, prefillCos: 1, mixedCos: 1}
	embs := make([][]float32, len(prompt))
	for i, tok := range prompt {
		embs[i] = mRes.EmbedResidentForTest(tok)
	}
	// CPU reference logits at every position.
	cache := mCPU.NewCache(len(prompt))
	cpu := make([][]float32, len(prompt))
	for i, tok := range prompt {
		lc, err := mCPU.ForwardForTest(tok, cache)
		if err != nil {
			t.Fatalf("cpu forward[%d]: %v", i, err)
		}
		cpu[i] = append([]float32(nil), lc...)
	}
	// 1. sequential decode.
	cr.Reset()
	for i := range prompt {
		lr, err := cr.Forward(embs[i], i)
		if err != nil {
			t.Fatalf("resident forward[%d]: %v", i, err)
		}
		cos, _ := cosF32(lr, cpu[i])
		m.decodeCos = math.Min(m.decodeCos, cos)
		m.decodeRel = math.Max(m.decodeRel, relL2(lr, cpu[i]))
	}
	// 2. batched prefill of the whole prompt, last-token logits.
	cr.Reset()
	got, err := cr.PrefillLast(context.Background(), embs, 0)
	if err != nil {
		t.Fatalf("PrefillLast: %v (the batched path must not decline for this family)", err)
	}
	last := len(prompt) - 1
	m.prefillCos, _ = cosF32(got, cpu[last])
	m.prefillRel = relL2(got, cpu[last])
	// 3. batched prefill of [0, split), then sequential decode of the rest: reads the K/V the batched
	// pairwise kernel wrote, with rope_kv_pw rotating the new queries.
	cr.Reset()
	if _, err := cr.PrefillLast(context.Background(), embs[:pairwiseSplit], 0); err != nil {
		t.Fatalf("PrefillLast(split): %v", err)
	}
	for i := pairwiseSplit; i < len(prompt); i++ {
		lr, err := cr.Forward(embs[i], i)
		if err != nil {
			t.Fatalf("resident forward after prefill [%d]: %v", i, err)
		}
		cos, _ := cosF32(lr, cpu[i])
		m.mixedCos = math.Min(m.mixedCos, cos)
		m.mixedRel = math.Max(m.mixedRel, relL2(lr, cpu[i]))
	}
	return m
}

// forceNeoXRope rebinds cr's rope pipelines (decode, batched prefill and, when loaded, m-RoPE prefill) to the NeoX
// kernels (the dispatch every family used before rope_pairwise.cu) and returns a restore func. Test-only mutation: the production binding
// stays in backend.go.
func forceNeoXRope(t *testing.T, cr *cudaResident) func() {
	t.Helper()
	gmod, err := cr.dev.CompileLibrary(gemvFwdPTX)
	if err != nil {
		t.Fatalf("compile gemv_fwd.ptx: %v", err)
	}
	pbmod, err := cr.dev.CompileLibrary(prefillBatchedPTX)
	if err != nil {
		t.Fatalf("compile prefill_batched.ptx: %v", err)
	}
	dec, err := cr.dev.NewComputePipeline(gmod, "rope_kv")
	if err != nil {
		t.Fatal(err)
	}
	bat, err := cr.dev.NewComputePipeline(pbmod, "rope_kv_batched")
	if err != nil {
		t.Fatal(err)
	}
	oldDec, oldBat, oldMR := cr.ropeKV, cr.bRopeKV, cr.bRopeKVMRoPE
	cr.ropeKV, cr.bRopeKV = dec, bat
	if cr.mropePrefillReady { // m-RoPE families (GLM-OCR) also bind the m-RoPE prefill kernel
		mmod, err := cr.dev.CompileLibrary(ropeMRopePrefillPTX)
		if err != nil {
			t.Fatalf("compile rope_mrope_prefill.ptx: %v", err)
		}
		if cr.bRopeKVMRoPE, err = cr.dev.NewComputePipeline(mmod, "rope_kv_mrope_batched"); err != nil {
			t.Fatal(err)
		}
	}
	return func() { cr.ropeKV, cr.bRopeKV, cr.bRopeKVMRoPE = oldDec, oldBat, oldMR }
}

// TestPairwiseRoPEResidentParityCUDA: Cohere (cohere-tiny) and Cohere2 (cohere2-tiny, sliding window +
// a NoPE global layer) on the CUDA resident vs the CPU at the same quantization, with peaked attention.
func TestPairwiseRoPEResidentParityCUDA(t *testing.T) {
	for _, fx := range []string{"cohere-tiny", "cohere2-tiny"} {
		t.Run(fx, func(t *testing.T) {
			src := filepath.Join("..", "testdata", fx)
			requireDeviceAndFixture(t, src)
			ckpt := peakedCheckpoint(t, src, peakedScale)
			mRes, err := decoder.Load(ckpt, decoder.Options{Backend: "cuda", Quant: "int8int8"})
			if err != nil {
				t.Fatalf("load cuda: %v", err)
			}
			defer mRes.Close()
			rf := mRes.ResidentForwardForTest()
			cr, ok := rf.(*cudaResident)
			if !ok {
				t.Fatalf("%s did not go CUDA-resident (%T) — decline: %s", fx, rf, mRes.ResidentDecline())
			}
			if !cr.pairwiseRoPE {
				t.Fatalf("resident.pairwiseRoPE=false for %s: the pairwise kernels are not bound", fx)
			}
			mCPU, err := decoder.Load(ckpt, decoder.Options{Backend: "cpu", Quant: "int8int8"})
			if err != nil {
				t.Fatalf("load cpu: %v", err)
			}
			defer mCPU.Close()
			_, _, _, _, _, _, vocab := mCPU.Dims()
			prompt := make([]int, pairwisePos)
			for i := range prompt {
				prompt[i] = (i*37 + 3) % vocab
			}

			got := measurePairwiseParity(t, mRes, mCPU, cr, prompt)
			t.Logf("PAIRWISE kernels: decode %d pos worst cos %.6f relL2 %.4f | batched prefill last cos %.6f relL2 %.4f | decode after prefill [%d,%d) worst cos %.6f relL2 %.4f",
				pairwisePos, got.decodeCos, got.decodeRel, got.prefillCos, got.prefillRel, pairwiseSplit, pairwisePos, got.mixedCos, got.mixedRel)
			if !got.ok() {
				t.Errorf("resident diverges from CPU (bars: cosine >= %.3f, relL2 <= %.2f): %+v", cosBar, relBar, got)
			}

			// Discrimination control: the SAME resident with the NeoX rope pipelines must read red.
			restore := forceNeoXRope(t, cr)
			neox := measurePairwiseParity(t, mRes, mCPU, cr, prompt)
			restore()
			t.Logf("NeoX control:      decode worst cos %.6f relL2 %.4f | batched prefill last cos %.6f relL2 %.4f | decode after prefill worst cos %.6f relL2 %.4f",
				neox.decodeCos, neox.decodeRel, neox.prefillCos, neox.prefillRel, neox.mixedCos, neox.mixedRel)
			if neox.decodeCos >= cosBar || neox.prefillCos >= cosBar || neox.mixedCos >= cosBar {
				t.Errorf("the gate is BLIND: with the NeoX rope kernels forced, a pairwise family still reads within the bar (%+v). Peak the attention harder (peakedScale) before trusting this gate", neox)
			}
		})
	}
}
