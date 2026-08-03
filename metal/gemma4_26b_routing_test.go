//go:build darwin

package metal

import (
	"os"
	"runtime"
	"strconv"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// forwardPagedCaptureIdxForTest runs the paged forward at position pos and returns the top-k expert
// idx the Metal router SELECTED at each MoE layer (read off g.rIdx after each layer's router, before
// staging). Test 1 of the routing-divergence probe.
func (r *Resident) forwardPagedCaptureIdxForTest(pos int) [][]int {
	r.uPos.SetU32(uint32(pos))
	r.uNKeys.SetU32(uint32(pos + 1))
	g := r.g4moe
	var idxs [][]int
	for l := 0; l < r.nL; l++ {
		L := &r.layers[l]
		if L.g4moe != nil && L.g4moe.pool != nil {
			e := r.q.Begin()
			r.encodeAttention(e, l)
			r.encodeG4Phase1(e, L)
			e.End()
			raw := g.rIdx.U32s()
			cur := make([]int, g.topK)
			slots := make([]expertSlot, g.topK)
			for j := 0; j < g.topK; j++ {
				cur[j] = int(raw[j])
				slots[j] = L.g4moe.pool.ensureResident(int(raw[j]))
			}
			idxs = append(idxs, cur)
			e2 := r.q.Begin()
			r.encodeG4Phase2Paged(e2, slots)
			r.encodeG4Join(e2, L)
			e2.End()
		} else {
			e := r.q.Begin()
			r.encodeLayer(e, l)
			e.End()
		}
	}
	return idxs
}

// TestGemma4_26B_routingAgreement is Test 1: does the paged-Metal router SELECT the same top-8
// experts as CPU-int4, per MoE layer per position, on [1,7,42,100]? Hypothesis: routing is a DISCRETE
// function of a continuously-drifting input — accumulated int4/f16 drift eventually FLIPS a top-8
// selection, and from that layer on the two backends compute different things → cosine CLIFFS (which
// a dense stack, having no discrete decision, cannot do). If the first idx divergence coincides with
// the L11→L14 collapse, that's the cause.
func TestGemma4_26B_routingAgreement(t *testing.T) {
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
	r, err := BuildResident(m)
	if err != nil {
		t.Fatalf("BuildResident: %v", err)
	}
	defer r.Close()

	moeLayers := moeLayerIdx(r) // paged MoE layer indices
	nMoE := len(moeLayers)
	prompt := []int{1, 7, 42, 100}

	// CPU-int4 router selections (token-outer, layer-inner: decision d → token d/nMoE, layer d%nMoE).
	decoder.SetRouterCaptureForTest(true)
	defer decoder.SetRouterCaptureForTest(false)
	cc := m.NewCache(len(prompt))
	for _, tk := range prompt {
		if _, err := m.ForwardForTest(tk, cc); err != nil {
			t.Fatalf("cpu: %v", err)
		}
	}
	idxCPU, _ := decoder.RouterCaptureForTest()
	marginCPU := decoder.RouterMarginForTest() // per decision: min selected prob − max rejected prob

	// Paged-Metal router selections per position.
	metalIdx := make([][][]int, len(prompt))
	func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		for pos, tk := range prompt {
			copy(r.x.Floats(), m.EmbedResidentForTest(tk))
			metalIdx[pos] = r.forwardPagedCaptureIdxForTest(pos)
		}
	}()

	overlap := func(a []int, b []int) int {
		s := map[int]bool{}
		for _, x := range a {
			s[x] = true
		}
		n := 0
		for _, x := range b {
			if s[x] {
				n++
			}
		}
		return n
	}
	topK := len(idxCPU[0])
	t.Logf("=== router top-%d agreement CPUint4 vs paged-Metal, %d MoE layers, prompt %v ===", topK, nMoE, prompt)
	firstFlip := -1
	for li := 0; li < nMoE; li++ {
		row := ""
		for pos := range prompt {
			cpu := idxCPU[pos*nMoE+li]
			mtl := metalIdx[pos][li]
			ov := overlap(cpu, mtl)
			row += " " + strconv.Itoa(ov)
			if ov < topK && firstFlip < 0 {
				firstFlip = li
			}
		}
		// print early layers + any with a mismatch
		anyMiss := false
		for pos := range prompt {
			if overlap(idxCPU[pos*nMoE+li], metalIdx[pos][li]) < topK {
				anyMiss = true
			}
		}
		if li < 16 || anyMiss {
			mark := ""
			if anyMiss {
				mark = "  <-- FLIP"
			}
			t.Logf("  MoE-layer %2d  overlap/pos [%s ] /%d%s", moeLayers[li], row, topK, mark)
		}
	}
	// TEST 2: are the flips at TIGHT boundaries (sensitivity, expected) or gross (router bug)? Compare
	// the CPU top-k margin (min-selected − max-rejected prob) at FLIPPED vs MATCHED decisions. If the
	// flipped decisions have a much SMALLER margin, they are near-ties a tiny input drift crosses —
	// numeric sensitivity, not a defect (and per-layer cosine vs CPU is then an INVALID instrument past
	// the first flip). Step 5a already proved the router KERNEL is bit-exact on matched input, so the
	// flips are input-drift-driven by construction; the margins confirm they sit at the boundary.
	var sumFlip, sumMatch float64
	var nFlip, nMatch int
	for pos := range prompt {
		for li := 0; li < nMoE; li++ {
			d := pos*nMoE + li
			if d >= len(marginCPU) {
				continue
			}
			mg := float64(marginCPU[d])
			if overlap(idxCPU[d], metalIdx[pos][li]) < topK {
				sumFlip += mg
				nFlip++
			} else {
				sumMatch += mg
				nMatch++
			}
		}
	}
	meanFlip, meanMatch := 0.0, 0.0
	if nFlip > 0 {
		meanFlip = sumFlip / float64(nFlip)
	}
	if nMatch > 0 {
		meanMatch = sumMatch / float64(nMatch)
	}
	t.Logf("TEST 2 — CPU top-8 boundary margin: FLIPPED decisions mean %.5f (n=%d) vs MATCHED mean %.5f (n=%d)",
		meanFlip, nFlip, meanMatch, nMatch)
	if firstFlip >= 0 {
		t.Logf("VERDICT: routing diverges from MoE-layer %d (model layer %d), NOT localized to L11-14 — accumulated. "+
			"If flipped-margin << matched-margin: SENSITIVITY (discrete flips of near-tied experts under int4/int8 input "+
			"drift), expected — and per-layer cosine vs CPU is INVALID past the first flip. If margins are comparable/large: "+
			"a router bug.", firstFlip, moeLayers[firstFlip])
	}
}
