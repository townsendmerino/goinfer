package decoder

import (
	"testing"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/internal/diag/sb2census"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

// metalDecodeScratchByFamily is the family-keyed price of Metal's decode scratch that the operations' per-row declarations replace:
// the oracle TestMetalDecodeScratch_fromTheDeclarations holds MetalDecodeScratchBytes to.
func (m *Model) metalDecodeScratchByFamily(slots int) int64 {
	H, nL, nH, _, _, I, V := m.Dims()
	if H <= 0 || V <= 0 {
		return 0
	}
	var f32, bytes, u32s int64 // f32 elements, bytes, and 4-byte uniforms
	w := m.Weights()
	type geomKey struct {
		hd, nKV, half int
		kEqV          bool
	}
	geoms := map[geomKey]bool{}
	nHhd, kvDim, fa, attn, layerFFN := 0, 0, 0, 0, 0
	maxFFNI := I
	nE, k, inter, sharedInter, _, _, _, _, _, _, moeOK := m.MoEResidentParams()
	g4 := m.HasGemma4MoEResident()
	moeOK = moeOK && !g4
	q4k := m.Quant() == "q4k"
	for l := range nL {
		mixer, hasFFN := m.ResidentLayerOps(l)
		if mixer == MixerAttention {
			hd, kv := m.HeadDimAtResident(l), m.KVHeadsAtResident(l)
			kEqV := m.VFromKResident(l) && m.KVSrcAtResident(l) == l
			geoms[geomKey{hd, kv, len(m.RopeInvFreqLayerResident(l)), kEqV}] = true
			nHhd, kvDim = max(nHhd, nH*hd), max(kvDim, kv*hd)
			fa = max(fa, metalAttnFAPartialFloats(hd, nH, kv))
			attn++
		}
		if l >= len(w.Layers) {
			continue
		}
		lw := &w.Layers[l]
		_, isG4 := m.Gemma4MoEResidentLayer(l)
		denseFFN := hasFFN && !isG4 && !(moeOK && len(lw.Experts) > 0) && !m.NonGatedMLPResident()
		if denseFFN {
			if fi := lw.GateProj.Rows(); fi != 0 {
				maxFFNI = max(maxFFNI, fi)
				if fi != I {
					layerFFN++
				}
			}
		}
		// The q4k lane's segments, one row-count uniform a run of parts of one kind in each fused group, on the dense attention
		// and SwiGLU layers it runs (buildResident refuses a q4k load with any other).
		if q4k && mixer == MixerAttention && denseFFN {
			for _, group := range [][]*linalg.WeightMat{{&lw.QProj, &lw.KProj, &lw.VProj}, {&lw.OProj}, {&lw.DownProj}, {&lw.GateProj, &lw.UpProj}} {
				u32s += int64(m.metalQ4KRuns(group))
			}
		}
	}
	if nHhd == 0 { // no attention layer: the attention rows are sized at 4, never dispatched
		nHhd, kvDim = 4, 4
	}
	gu := maxFFNI
	if moeOK {
		gu = max(gu, inter, sharedInter)
	}
	var g4b Gemma4MoEResidentBundle
	if g4 {
		for l := range nL {
			if b, ok := m.Gemma4MoEResidentLayer(l); ok {
				g4b = b
				break
			}
		}
		gu = max(gu, g4b.DenseInter, g4b.MoeInter)
	}
	// The core: x, aSc; the fused q | k | v; the fused gate | up; ctx, cSc; oO, mSc; dSc, dO; the logits, the argmax partials, tok;
	// the sampler's block keys and indices and its draw; and the fused attention's partial.
	f32 += int64(H + 1 + nHhd + 2*kvDim + 2*gu + nHhd + 1 + H + 1 + 1 + H + V + 2*((V+7)/8) + 1 + 2*((V+1023)/1024) + 1 + fa)
	aq := H
	if q4k {
		aq = 4 * H
		f32 += int64(H + gu) // the lane's f32 activation rows
	}
	bytes += int64(aq + nHhd + H + 2*H + 2*H + 2*nHhd + gu) // aq, cq, mq; the f16 twins of aq, mq and cq; dq
	u32s += 22                                              // the core's eleven, uAct, uAddOne, uWindow, uLNHasBias, and the sampler's seven
	if fa > 0 {
		u32s += 2
	}
	if m.QKNormWholeResident() {
		u32s += 2
	}
	u32s += int64(7*len(geoms) + 2*attn + layerFFN) // each geometry's seven; each attention layer's sink flag and window; a distinct FFN width's
	if P := m.Gemma4PLEDimResident(); P > 0 {
		f32 += int64(nL*P + P + 1)
		bytes += int64(P)
		u32s++
	}
	if _, hk, hv, nk, nv, attnGate, ok := m.Qwen35ResidentParams(); ok { // the Gated DeltaNet mixer and gated attention's [query ‖ gate]
		keyDim, valueDim := nk*hk, nv*hv
		f32 += int64(2*(2*keyDim+valueDim) + 2*nv + 2*nv + 2*keyDim + 3*valueDim + 1)
		if attnGate {
			f32 += int64(2*nHhd + nHhd)
		}
		bytes += int64(valueDim)
		u32s += 12
		if eps, oEps := m.metalDeltaNetEpsByFamily(); oEps != 0 && oEps != eps {
			u32s++
		}
	}
	if nh, hd, ds, ng, _, normG, _, ok := m.NemotronResidentParams(); ok && normG > 0 { // the Mamba-2 mixer
		dInner, gSize := nh*hd, ng*ds
		f32 += int64(2*dInner + 2*gSize + nh + dInner + 2*gSize + 2*dInner + 1)
		bytes += int64(dInner)
		u32s += 13
	}
	if qLoRA, rank, nope, rope, vHead, _, _, _, ok := m.MLAResidentParams(); ok { // Multi-head Latent Attention
		qkHead, latDim := nope+rope, rank+rope
		f32 += int64(max(qLoRA, 1) + 1 + nH*qkHead + latDim + nH*latDim + nH*rank + nH*vHead + 1)
		bytes += int64(max(qLoRA, 4) + nH*vHead + 4*(3+5+5+7+3)) // qaQ, ctxQ, and the five packed shapes
		u32s += 5
	}
	if moeOK { // the generic MoE's router, shared expert and k slots
		_, _, gptOss := m.GptOssActResident()
		f32 += int64(nE + k + 1 + H)
		bytes += int64(4 * k)
		u32s += int64(10 + k)
		if gptOss {
			u32s += 3
		}
		if !gptOss && (2*inter)%8 == 0 && H%8 == 0 {
			f32 += int64(k*2*inter + k + k*H)
			bytes += int64(k * inter)
		}
		if metalMoEPaged(slots, nE) {
			bytes += int64(4 * k)
		}
	}
	if g4 { // Gemma 4's parallel dense ‖ MoE block
		f32 += int64(2*g4b.NE + 2*g4b.TopK + 4*H)
		bytes += int64(8 * g4b.TopK)
		u32s += int64(10 + g4b.TopK)
		if metalMoEPaged(slots, g4b.NE) {
			bytes += int64(4 * g4b.TopK)
		}
	}
	f32 += int64(metalScratchBatchRows * (H + V)) // batchX, batchLogits
	u32s += 3 * metalScratchBatchRows             // each batch row's position uniforms
	u32s += 12 + 7                                // the greedy chain's two tokens and position sets; a sampled chain's temperature, keys and draws
	f32 += int64(2 * V)                           // a sampled chain's two logits rows
	return 4*f32 + bytes + 4*u32s
}

// metalDeltaNetEps is the model's Gated DeltaNet operation's norm epsilons (program.DeltaNetOpParams): Metal holds a uniform of its own
// for the output norm's where that differs from the model's.
func (m *Model) metalDeltaNetEpsByFamily() (eps, oEps float64) {
	for _, ops := range m.Program().Layers {
		for _, o := range ops {
			if o.Kind == program.OpDeltaNet {
				return o.Delta.Eps, o.Delta.ONormEps
			}
		}
	}
	return 0, 0
}

// TestMetalDecodeScratch_fromTheDeclarations is MM5b's derivation (docs/tasks/task-forward-architecture-2026-10.md §4.9c): Metal's
// decode scratch priced from the program's per-row declarations (MetalDecodeScratchBytes) is the family-keyed price it replaces, on
// every fixture and option cell layer D built resident on Metal (TestMemGate_metal's rows), unpaged and at a paged slot count. Each
// is loaded on the CPU without its KV options.
func TestMetalDecodeScratch_fromTheDeclarations(t *testing.T) {
	d, err := sb2census.ReadLayerD("..", "metal")
	if err != nil {
		t.Fatal(err)
	}
	rows := 0
	for _, f := range d.Fixtures {
		path, err := sb2census.Path("..", f)
		if err != nil {
			t.Fatal(err)
		}
		for _, c := range sb2census.Cells {
			if !d.Resident[f+" "+c.Name] {
				continue
			}
			o := Options{Quant: c.Quant, ActQuantGroup: c.ActGroup} // the KV's precision is not decode's scratch, and the CPU delivers no int8 KV
			if c.SSM != "" {
				o.Knobs = &Knobs{"GOINFER_SSM_RESIDENT": c.SSM}
			}
			m, err := Load(path, o)
			if err != nil {
				t.Errorf("%s %s: %v", f, c.Name, err)
				continue
			}
			slotCounts := []int{0}
			if nE, k, _ := m.moeGeometry(); k > 0 && k < nE {
				slotCounts = append(slotCounts, k) // a paged slot count
			}
			for _, slots := range slotCounts {
				if got, want := m.MetalDecodeScratchBytes(slots), m.metalDecodeScratchByFamily(slots); got != want {
					t.Errorf("%s %s at %d slots: priced from the declarations %d bytes, by family %d (%+d)", f, c.Name, slots, got, want, got-want)
				}
			}
			rows += len(slotCounts)
			m.Close()
		}
	}
	if rows < 300 {
		t.Errorf("only %d prices compared", rows)
	}
	t.Logf("%d prices compared", rows)
}
