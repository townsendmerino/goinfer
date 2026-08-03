//go:build darwin

package metal

import (
	"os"
	"runtime"
	"strconv"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// forwardPagedCaptureForTest replicates forwardLogitsPaged's per-layer walk but captures the residual
// r.x AFTER each layer (readable post-submit+wait), so a paged-Metal-vs-CPU per-layer cosine trace can
// localize the 26B divergence to a specific layer. Caller holds the OS thread + filled r.x.
func (r *Resident) forwardPagedCaptureForTest(pos int) [][]float32 {
	r.uPos.SetU32(uint32(pos))
	r.uNKeys.SetU32(uint32(pos + 1))
	g := r.g4moe
	cap := make([][]float32, 0, r.nL)
	for l := 0; l < r.nL; l++ {
		L := &r.layers[l]
		if L.g4moe != nil && L.g4moe.pool != nil {
			e := r.q.Begin()
			r.encodeAttention(e, l)
			r.encodeG4Phase1(e, L)
			e.End()
			idx := g.rIdx.U32s()
			slots := make([]expertSlot, g.topK)
			for j := 0; j < g.topK; j++ {
				slots[j] = L.g4moe.pool.ensureResident(int(idx[j]))
			}
			e2 := r.q.Begin()
			r.encodeG4Phase2Paged(e2, slots)
			r.encodeG4Join(e2, L)
			e2.End()
		} else {
			e := r.q.Begin()
			r.encodeLayer(e, l)
			e.End()
		}
		cap = append(cap, append([]float32(nil), r.x.Floats()...))
	}
	return cap
}

// TestGemma4_26B_localize is the Step-6 divergence localizer: paged-Metal vs CPU per-layer hidden
// cosine on a SINGLE-token forward (pos 0 — no attention history / sliding window / KV, so a drop is
// the layer's own compute). Quant noise degrades smoothly; a composition bug STEPS. Names the first
// discontinuity and correlates it with local/global + K=V + geometry. Heavy + paged.
func TestGemma4_26B_localize(t *testing.T) {
	requireHeavyModel(t)
	const giw = "/Users/francistownsend-merino/models/gemma4-26b-int4.giw"
	if _, err := os.Stat(giw); err != nil {
		t.Skipf("no .giw")
	}
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")
	t.Setenv("GOINFER_METAL_MOE_SLOTS", strconv.Itoa(32))
	m, err := decoder.Load(giw, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()

	// #4 (cheap, static): attnGeom key {hd,nKV,half,kEqV} collision — two layers sharing a key but
	// differing in something the key doesn't carry (rope base is per-layer L.invf, window per-layer
	// L.uWindow, so those are safe; a collision that shares a pipeline WHERE SHAPE differs is the bug).
	type key struct{ hd, nKV, half, kEqV int }
	seen := map[key][]int{}
	for l := 0; l < 64; l++ {
		half := len(m.RopeInvFreqLayerResident(l))
		if half == 0 {
			continue
		}
		kv := 0
		if m.VFromKResident(l) {
			kv = 1
		}
		k := key{m.HeadDimAtResident(l), m.KVHeadsAtResident(l), half, kv}
		seen[k] = append(seen[k], l)
	}
	t.Logf("geometry map (hd,nKV,half,kEqV → layers):")
	for k, ls := range seen {
		t.Logf("  {hd=%d nKV=%d half=%d kEqV=%d} → layers %v", k.hd, k.nKV, k.half, k.kEqV, ls)
	}

	r, err := BuildResident(m)
	if err != nil {
		t.Fatalf("BuildResident: %v", err)
	}
	defer r.Close()

	// RoPE is IDENTITY at pos 0 (angle 0) — a wrong rotary fraction/base is invisible there. So trace
	// the LAST position of a short prompt (RoPE active) and compare to pos 0. If the global layers
	// crater at pos>0 while locals stay clean, the defect is the global-layer RoPE (fraction/base/tail).
	prompt := twoGeomPrompt[:4] // positions 0..3
	last := len(prompt) - 1
	nL1 := r.nL + 1

	decoder.SetGemma4HiddenCaptureForTest(true)
	defer decoder.SetGemma4HiddenCaptureForTest(false)
	cpuCache := m.NewCache(len(prompt))
	for _, tk := range prompt {
		if _, err := m.ForwardForTest(tk, cpuCache); err != nil {
			t.Fatalf("cpu: %v", err)
		}
	}
	cpuAll := decoder.Gemma4HiddenCaptureForTest() // per token: [post-embed, after-L0, ...]
	if len(cpuAll) < len(prompt)*nL1 {
		t.Fatalf("cpu captured %d states, want >= %d", len(cpuAll), len(prompt)*nL1)
	}
	cpu := cpuAll[last*nL1:] // last position's [post-embed, after-L0, ...]

	metal := func() [][]float32 {
		runtime.LockOSThread() // paged capture drives autorelease pools (thread-local)
		defer runtime.UnlockOSThread()
		var capLast [][]float32
		for pos, tk := range prompt {
			copy(r.x.Floats(), m.EmbedResidentForTest(tk))
			c := r.forwardPagedCaptureForTest(pos) // builds KV; keep the last position's per-layer
			if pos == last {
				capLast = c
			}
		}
		return capLast
	}()

	t.Logf("=== per-layer paged-Metal vs CPU cosine at pos %d (RoPE active) ===", last)
	prev := 1.0
	firstDrop := -1
	var worstLocal, worstGlobal = 1.0, 1.0
	for l := 0; l < r.nL; l++ {
		c, _ := cosMaxAbs(cpu[l+1], metal[l])
		if m.VFromKResident(l) {
			if c < worstGlobal {
				worstGlobal = c
			}
		} else if c < worstLocal {
			worstLocal = c
		}
		isGlobal := m.VFromKResident(l)
		tag := "dense "
		if isGlobal {
			tag = "GLOBAL(K=V hd=" + strconv.Itoa(m.HeadDimAtResident(l)) + ")"
		} else if m.LayerIsLocalResident(l) {
			tag = "local "
		}
		mark := ""
		if c < prev-0.10 { // a step down of >0.10 cosine = discontinuity
			mark = "  <<< DISCONTINUITY"
			if firstDrop < 0 {
				firstDrop = l
			}
		}
		if l < 12 || isGlobal || c < 0.9 || mark != "" { // always print globals + first 12 + any low/step
			t.Logf("  L%02d %-18s cosine %.4f%s", l, tag, c, mark)
		}
		prev = c
	}
	t.Logf("worst cosine at pos %d — LOCAL layers %.4f | GLOBAL(K=V) layers %.4f", last, worstLocal, worstGlobal)
	if firstDrop >= 0 {
		t.Logf("FIRST DISCONTINUITY at layer %d (local=%v kEqV=%v hd=%d nKV=%d)", firstDrop,
			m.LayerIsLocalResident(firstDrop), m.VFromKResident(firstDrop), m.HeadDimAtResident(firstDrop), m.KVHeadsAtResident(firstDrop))
	} else {
		t.Logf("NO step discontinuity — divergence is smooth (quant drift), not a composition bug")
	}
}
