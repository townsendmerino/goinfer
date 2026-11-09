//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"math"
	"math/rand"
	"path/filepath"
	"strings"
	"testing"

	gpu "github.com/townsendmerino/goinfer/backend/cuda/internal/gpudev"
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
	ckpt := filepath.Join("../../testdata", fx)
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
			mc, err := decoder.Load(filepath.Join("../../testdata", fx), decoder.Options{Backend: "cuda", Quant: "int4"})
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

// TestGemma2Softcap_lanesCUDA is G2-L: the lane selectors keep a capped layer off every kernel that has no cap, and off the flash-decode
// lane unless the load asked for it (decoder.Options.FlashDecodeSoftcap; OP1-o2), when it takes the capped variant (decode_fa_cap). The
// tiny fixtures have head dimension 32, which flash-decode and the fused kernels never serve, so each selector is asked at a head dimension
// it does serve, with the resident's own state set up as a served one would be: with the cap off both say yes (the control: the test can
// see the lanes); with the cap on, the default load's flash-decode selector says no, the opted-in one yes, and the fused-prefill one no.
func TestGemma2Softcap_lanesCUDA(t *testing.T) {
	requireCUDADevice(t)
	t.Setenv("GOINFER_CUDA_FLASH_DECODE", "16") // the test binary turns the lane off by default; this turns it on so the control can see it
	t.Setenv("GOINFER_CUDA_FLASH_DECODE_MIN_KEYS", "0")
	mc, err := decoder.Load(filepath.Join("../../testdata/gemma2-tiny"), decoder.Options{Backend: "cuda", Quant: "int4"})
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
	if rf.faSoftcap {
		t.Fatal("a default load opted a capped model into the flash-decode lane")
	}
	if fa() {
		t.Error("a capped layer on a default load is eligible for the flash-decode lane: the exact split-KV lane is the default (its verify rows ask the same question)")
	}
	rf.faSoftcap = true // what decoder.Options.FlashDecodeSoftcap sets; TestGemma2Softcap_flashDecodeVsF64 reads it through a load
	if !fa() || rf.faModule != "decode_fa_cap" {
		t.Errorf("a capped layer, opted in: flash-decode eligible %v on module %q; want eligible on decode_fa_cap", fa(), rf.faModule)
	}
	rf.faSoftcap = false
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

// flashRefCap is flashRef with Gemma 2's attention softcap on every score: cap·tanh(scale·q·k / cap) before the softmax, in f64.
func flashRefCap(q, k, v []float32, nH, nKV, hd, winStart, nKeys int, scale, cap float32) []float64 {
	G, kvDim := nH/nKV, nKV*hd
	out := make([]float64, nH*hd)
	sc := make([]float64, nKeys-winStart)
	for h := range nH {
		kvh := h / G
		mx := math.Inf(-1)
		for s := winStart; s < nKeys; s++ {
			var d float64
			for i := range hd {
				d += float64(q[h*hd+i]) * float64(k[s*kvDim+kvh*hd+i])
			}
			d = float64(cap) * math.Tanh(d*float64(scale)/float64(cap))
			sc[s-winStart] = d
			mx = math.Max(mx, d)
		}
		var den float64
		for i := range sc {
			sc[i] = math.Exp(sc[i] - mx)
			den += sc[i]
		}
		for s := winStart; s < nKeys; s++ {
			w := sc[s-winStart] / den
			for i := range hd {
				out[h*hd+i] += w * float64(v[s*kvDim+kvh*hd+i])
			}
		}
	}
	return out
}

// TestGemma2Softcap_flashDecodeVsF64 is OP1-o2's resident-level check (docs/completed/task-forward-architecture-2026-10.md §4.6a, "Owed,
// not in OP1"; the kernel gate at both KV types is the kernels package's TestAttnSoftcap_flashDecodeAgainstFloat64): a capped model
// loaded with decoder.Options.FlashDecodeSoftcap carries the opt-in to its resident and loads the lane's capped variant (decode_fa_cap), and
// that lane, through the resident's own launch path, driven on
// synthetic K/V at head dimensions 64, 128 and 256 (the lane's), matches an f64 oracle that caps every score, within the uncapped lane's
// bound (TestFlashDecodeVsF64: 1e-4 of the largest output), at several key counts, window starts and split counts, deterministically.
// Scores are drawn large enough that the fixture's cap (5) bends most of them, and the control requires it: on most cases the uncapped
// oracle is far from the capped one, and the lane with its cap lifted to 1e30 is far from the capped oracle. The multi-row verify lane,
// loaded from the same module, equals the M=1 lane at every row bit for bit.
func TestGemma2Softcap_flashDecodeVsF64(t *testing.T) {
	requireCUDADevice(t)
	t.Setenv("GOINFER_CUDA_FLASH_DECODE", "16")
	m, err := decoder.Load(filepath.Join("../../testdata/gemma2-tiny"), decoder.Options{Backend: "cuda", Quant: "int4", FlashDecodeSoftcap: true})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	rf, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatalf("not resident: %s", m.ResidentDecline())
	}
	if !rf.faSoftcap {
		t.Fatal("decoder.Options.FlashDecodeSoftcap did not reach the resident")
	}
	capV := rf.attnCap
	if capV == 0 || rf.faModule != "decode_fa_cap" || rf.faCombine == (Pipeline{}) || rf.faCombineRows == (Pipeline{}) {
		t.Fatalf("cap %v, lane module %q, combine loaded %v, rows loaded %v: the capped lane did not load", capV, rf.faModule,
			rf.faCombine != (Pipeline{}), rf.faCombineRows != (Pipeline{}))
	}
	const nKV = 2
	nH := rf.nH
	if nH%nKV != 0 {
		t.Fatalf("nH %d", nH)
	}
	// The lane's partial buffer is sized at load for the fixture's head dimension (32); a model the lane serves sizes it for its own. Size
	// it for the largest geometry driven here, every row of a verify batch at 16 splits, as such a load would.
	if e := rf.do(func() error { rf.faBuf = rf.tf(faMaxRows * nH * 16 * faWarps * (256 + 4)); return nil }); e != nil {
		t.Fatal(e)
	}
	rng := rand.New(rand.NewSource(5))
	worst, cases, bent, lifted := 0.0, 0, 0, 0
	for _, hd := range []int{64, 128, 256} {
		kvDim := nKV * hd
		scale := float32(1 / math.Sqrt(float64(hd)))
		for _, nKeys := range []int{1, 9, 65, 300, 1000} {
			q, k, v := make([]float32, nH*hd), make([]float32, nKeys*kvDim), make([]float32, nKeys*kvDim)
			for i := range q {
				q[i] = 3 * float32(rng.NormFloat64()) // scores of a few times the cap
			}
			for i := range k {
				k[i], v[i] = float32(rng.NormFloat64()), float32(rng.NormFloat64())
			}
			for _, winStart := range []int{0, nKeys / 3} {
				ref := flashRefCap(q, k, v, nH, nKV, hd, winStart, nKeys, scale, capV)
				plain := flashRef(q, k, v, nH, nKV, hd, winStart, nKeys, scale)
				var refMax, capGap float64
				for i, x := range ref {
					refMax = math.Max(refMax, math.Abs(x))
					capGap = math.Max(capGap, math.Abs(x-plain[i]))
				}
				refMax = math.Max(refMax, 1e-12)
				rel := func(got []float32) (r float64) {
					for i := range got {
						if math.IsNaN(float64(got[i])) {
							t.Fatalf("NaN: hd=%d nKeys=%d win=%d elem %d", hd, nKeys, winStart, i)
						}
						r = math.Max(r, math.Abs(float64(got[i])-ref[i]))
					}
					return r / refMax
				}
				for _, S := range []int{1, 4, 16} {
					rf.faSplit = S
					got := runFlash(t, rf, q, k, v, hd, nKV, winStart, nKeys, scale)
					again := runFlash(t, rf, q, k, v, hd, nKV, winStart, nKeys, scale)
					for i := range got {
						if got[i] != again[i] {
							t.Fatalf("NOT DETERMINISTIC: hd=%d nKeys=%d win=%d S=%d elem %d", hd, nKeys, winStart, S, i)
						}
					}
					e := rel(got)
					worst = math.Max(worst, e)
					cases++
					if e > 1e-4 {
						t.Errorf("hd=%d nKeys=%d win=%d S=%d: %.3g of max|ref| %.3g from the capped f64 oracle (bound 1e-4)", hd, nKeys, winStart, S, e, refMax)
					}
					if capGap/refMax > 1e-2 { // the cap moves this case's output: the lane must follow it
						bent++
						rf.attnCap = 1e30
						off := runFlash(t, rf, q, k, v, hd, nKV, winStart, nKeys, scale)
						rf.attnCap = capV
						if rel(off) > 1e-3 {
							lifted++
						}
					}
				}
			}
		}
	}
	t.Logf("%d cases, worst %.3g of max|ref| from the capped oracle; %d where the cap moves the output by over 1%%, %d of them red with the cap lifted",
		cases, worst, bent, lifted)
	if bent < cases/2 || lifted != bent {
		t.Fatalf("control: the cap moved %d of %d cases and lifting it reddened %d: the test cannot see the cap", bent, cases, lifted)
	}

	// The verify lane: each row of a multi-row launch equals the M=1 lane at its position, bit for bit, under the cap.
	const hd, maxKeys = 128, 700
	kvDim, qDim := nKV*hd, nH*hd
	scale := float32(1 / math.Sqrt(float64(hd)))
	q, k, v := make([]float32, faMaxRows*qDim), make([]float32, maxKeys*kvDim), make([]float32, maxKeys*kvDim)
	for i := range q {
		q[i] = 3 * float32(rng.NormFloat64())
	}
	for i := range k {
		k[i], v[i] = float32(rng.NormFloat64()), float32(rng.NormFloat64())
	}
	var qb, kb, vb, cb, kt Buffer
	if e := rf.do(func() error {
		qb, kb, vb, cb = rf.tf(len(q)), rf.tf(len(k)), rf.tf(len(v)), rf.tf(faMaxRows*qDim)
		kt = rf.kvIdentityTab(t, maxKeys)
		for _, u := range []struct {
			b Buffer
			x []float32
		}{{qb, q}, {kb, k}, {vb, v}} {
			if e := gpu.Upload(u.b, u.x); e != nil {
				return e
			}
		}
		return nil
	}); e != nil {
		t.Fatalf("setup: %v", e)
	}
	saved := rf.attnScale
	rf.attnScale = scale
	defer func() { rf.attnScale = saved }()
	rows := 0
	for _, S := range []int{4, 16} {
		rf.faSplit = S
		for _, n0 := range []int{1, 9, 300, 650} {
			for _, M := range []int{1, 3, 9} {
				full := make([]float32, faMaxRows*qDim)
				if e := rf.do(func() error {
					for _, run := range faRowRuns(n0, M, 0, S) {
						if e := rf.flashDecodeRowsLaunch(qb, kb, vb, kt, cb, hd, nKV, run); e != nil {
							return e
						}
					}
					if e := rf.stream.Sync(); e != nil {
						return e
					}
					return gpu.Download(cb, full)
				}); e != nil {
					t.Fatalf("multi-row launch: %v", e)
				}
				for i := range M {
					one := runFlash(t, rf, q[i*qDim:(i+1)*qDim], k, v, hd, nKV, 0, n0+i, scale)
					for j := range one {
						if math.Float32bits(one[j]) != math.Float32bits(full[i*qDim+j]) {
							t.Fatalf("S=%d n0=%d M=%d row %d elem %d: multi-row %v, M=1 %v", S, n0, M, i, j, full[i*qDim+j], one[j])
						}
					}
					rows++
				}
			}
		}
	}
	t.Logf("%d verify rows equal the M=1 capped lane bit for bit", rows)
}
