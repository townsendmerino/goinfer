package decoder

import (
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// TestValidateShapes_catchesArchMismatch is the break-it-first gate for audit C-06:
// a .giw whose tensor rows disagree with the architecture must be REJECTED, because
// the forward writes tensor.Rows() floats into a config-sized scratch slice (moeMLP's
// make([]float32, NumExperts); the qDim/kvDim/vocab decodeScratch buffers) — a heap
// write past the slice from caller-supplied bytes.
func TestValidateShapes_catchesArchMismatch(t *testing.T) {
	const (
		vocab, hidden, nH, nKV, hd, nE, inter, moeInter = 32, 8, 2, 1, 4, 4, 16, 12
	)
	arch := &Architecture{
		VocabSize: vocab, HiddenDim: hidden, NumHeads: nH, NumKVHeads: nKV, HeadDim: hd,
		IntermediateDim: inter,
		NumLayers:       1, // one layer in the bundle below; the C-06 layer-count check needs them to agree
		MoE:             &MoEConfig{NumExperts: nE, TopK: 2, IntermediateDim: moeInter},
	}
	f := func(rows, cols int) linalg.WeightMat {
		return linalg.WrapF32(make([]float32, rows*cols), rows, cols)
	}
	good := func() *Weights {
		return &Weights{
			arch:   arch,
			Embed:  f(vocab, hidden),
			LMHead: f(vocab, hidden),
			Layers: []LayerWeights{{
				QProj: f(nH*hd, hidden), KProj: f(nKV*hd, hidden), VProj: f(nKV*hd, hidden), OProj: f(hidden, nH*hd),
				GateProj: f(inter, hidden), UpProj: f(inter, hidden), DownProj: f(hidden, inter), Router: f(nE, hidden),
			}},
		}
	}
	// The honest bundle passes.
	if e := validateShapes(good(), arch); e != nil {
		t.Fatalf("valid bundle rejected: %v", e)
	}
	// Each tampered dimension must be caught.
	for _, tc := range []struct {
		name   string
		mutate func(*Weights)
		want   string
	}{
		{"router rows = NumExperts+K", func(w *Weights) { w.Layers[0].Router = f(nE+2, hidden) }, "Router"},
		{"embed vocab wrong", func(w *Weights) { w.Embed = f(vocab+1, hidden) }, "Embed"},
		{"lmhead vocab wrong", func(w *Weights) { w.LMHead = f(vocab*2, hidden) }, "LMHead"},
		// The embedding lookup widens one row into a HiddenDim scratch: a table declared wider than that is a heap write.
		{"embed cols wider than hidden", func(w *Weights) { w.Embed = f(vocab, hidden+8) }, "Embed: 16 cols"},
		{"embed cols narrower than hidden", func(w *Weights) { w.Embed = f(vocab, hidden-1) }, "Embed: 7 cols"},
		{"lmhead cols wrong", func(w *Weights) { w.LMHead = f(vocab, hidden*2) }, "LMHead: 16 cols"},
		{"posembed cols wrong", func(w *Weights) { w.PosEmbed = f(16, hidden+1) }, "PosEmbed: 9 cols"},
		{"qproj rows wrong", func(w *Weights) { w.Layers[0].QProj = f(nH*hd+1, hidden) }, "QProj"},
		{"kproj rows wrong", func(w *Weights) { w.Layers[0].KProj = f(nKV*hd+3, hidden) }, "KProj"},
		{"oproj rows wrong", func(w *Weights) { w.Layers[0].OProj = f(hidden+1, nH*hd) }, "OProj"},
		{"gateproj rows wrong", func(w *Weights) { w.Layers[0].GateProj = f(inter+1, hidden) }, "GateProj"},
		{"upproj rows wrong", func(w *Weights) { w.Layers[0].UpProj = f(inter*2, hidden) }, "UpProj"},
		{"expert gate rows wrong", func(w *Weights) {
			w.Layers[0].Experts = []expertWeights{{Gate: f(moeInter+1, hidden)}}
		}, "expert 0 Gate"},
		{"expert down rows wrong", func(w *Weights) {
			w.Layers[0].Experts = []expertWeights{{Down: f(hidden+1, moeInter)}}
		}, "expert 0 Down"},
		{"too few layers (OOB read)", func(w *Weights) { w.Layers = nil }, "layer count"},
		{"too many layers", func(w *Weights) { w.Layers = append(w.Layers, LayerWeights{}) }, "layer count"},
	} {
		w := good()
		tc.mutate(w)
		e := validateShapes(w, arch)
		if e == nil {
			t.Errorf("%s: not rejected", tc.name)
			continue
		}
		if !strings.Contains(e.Error(), tc.want) {
			t.Errorf("%s: error %q does not name %q", tc.name, e.Error(), tc.want)
		}
	}
}

// TestValidateShapes_gemma4PLE is a synthetic gate for the model-level Per-Layer-Embedding tail:
// PerLayerModelProj and PerLayerProjNorm were checked against the wrong axis (HiddenDim instead
// of NumLayers*HiddenSizePerLayerInput) and only a real gemma4-E2B checkpoint — a heavy,
// asset-gated test — ever exercised the branch, so both bugs shipped unnoticed. This runs on
// every invocation with no checkpoint required.
func TestValidateShapes_gemma4PLE(t *testing.T) {
	const hidden, pleDim, nLayers = 16, 4, 2
	pleTotal := nLayers * pleDim
	arch := &Architecture{
		HiddenDim: hidden, NumHeads: 1, NumKVHeads: 1, HeadDim: 4, IntermediateDim: 8,
		NumLayers: nLayers,
		gemma4:    &gemma4Params{HiddenSizePerLayerInput: pleDim},
	}
	f := func(rows, cols int) linalg.WeightMat {
		return linalg.WrapF32(make([]float32, rows*cols), rows, cols)
	}
	good := func() *Weights {
		layers := make([]LayerWeights, nLayers)
		for i := range layers {
			layers[i] = completeDenseLayer(arch, i)
			layers[i].PLEGate, layers[i].PLEProj, layers[i].PostPLENorm = f(pleDim, hidden), f(hidden, pleDim), make([]float32, hidden)
		}
		return &Weights{
			arch:               arch,
			Layers:             layers,
			PerLayerTokenEmbed: f(1, pleTotal), // only Rows()>0 gates the branch; shape unchecked
			PerLayerModelProj:  f(pleTotal, hidden),
			PerLayerProjNorm:   make([]float32, pleDim),
		}
	}
	if e := validateShapes(good(), arch); e != nil {
		t.Fatalf("valid gemma4 PLE bundle rejected: %v", e)
	}
	for _, tc := range []struct {
		name   string
		mutate func(*Weights)
		want   string
	}{
		{"PerLayerModelProj rows swapped with hidden", func(w *Weights) { w.PerLayerModelProj = f(hidden, pleTotal) }, "PerLayerModelProj"},
		{"PerLayerModelProj cols wrong", func(w *Weights) { w.PerLayerModelProj = f(pleTotal, hidden+1) }, "PerLayerModelProj"},
		{"PerLayerProjNorm sized to HiddenDim instead of pleDim", func(w *Weights) { w.PerLayerProjNorm = make([]float32, hidden) }, "PerLayerProjNorm"},
	} {
		w := good()
		tc.mutate(w)
		e := validateShapes(w, arch)
		if e == nil {
			t.Errorf("%s: not rejected", tc.name)
			continue
		}
		if !strings.Contains(e.Error(), tc.want) {
			t.Errorf("%s: error %q does not name %q", tc.name, e.Error(), tc.want)
		}
	}
}

// completeDenseLayer is layer i of a generic attention + gated-FFN arch with every projection present at its arch shape.
func completeDenseLayer(arch *Architecture, i int) LayerWeights {
	hd, h := arch.headDimAt(i), arch.HiddenDim
	q, kv, ffn := arch.headsAt(i)*hd, arch.kvHeadsAt(i)*hd, arch.ffnAt(i)
	return LayerWeights{
		QProj: mat(q, h), KProj: mat(kv, h), VProj: mat(kv, h), OProj: mat(h, q),
		GateProj: mat(ffn, h), UpProj: mat(ffn, h), DownProj: mat(h, ffn),
	}
}

// shapeCase is one tampering of a valid bundle and the tensor name the refusal must carry.
type shapeCase struct {
	name   string
	mutate func(*Weights)
	want   string
}

// requireRefused checks good passes validateShapes and each case fails it with a *SerializeError naming its tensor.
func requireRefused(t *testing.T, arch *Architecture, good func() *Weights, cases []shapeCase) {
	t.Helper()
	if e := validateShapes(good(), arch); e != nil {
		t.Fatalf("valid bundle rejected: %v", e)
	}
	for _, tc := range cases {
		w := good()
		tc.mutate(w)
		e := validateShapes(w, arch)
		if e == nil {
			t.Errorf("%s: not rejected", tc.name)
			continue
		}
		if !strings.Contains(e.Error(), tc.want) {
			t.Errorf("%s: error %q does not name %q", tc.name, e.Error(), tc.want)
		}
	}
}

// A projection's input width (Cols) is read from an arch-sized activation, and a projection the layer's program multiplies
// by must be present: either wrong is an out-of-bounds read or a garbage forward from a bundle that loaded clean.
func TestValidateShapes_colsAndMandatoryProjections(t *testing.T) {
	arch := &Architecture{
		VocabSize: 32, HiddenDim: 8, NumHeads: 2, NumKVHeads: 1, HeadDim: 4, IntermediateDim: 16, NumLayers: 1,
		MoE: &MoEConfig{NumExperts: 4, TopK: 2, IntermediateDim: 12, SharedIntermediateDim: 12},
	}
	h, q, kv, ffn, moeI := 8, 8, 4, 16, 12
	good := func() *Weights {
		lw := completeDenseLayer(arch, 0)
		lw.Router = mat(4, h)
		lw.Experts = []expertWeights{{Gate: mat(moeI, h), Up: mat(moeI, h), Down: mat(h, moeI)}}
		lw.SharedExpert = expertWeights{Gate: mat(moeI, h), Up: mat(moeI, h), Down: mat(h, moeI)}
		lw.SharedGate = mat(1, h)
		return &Weights{arch: arch, Embed: mat(32, h), Layers: []LayerWeights{lw}}
	}
	l0 := func(w *Weights) *LayerWeights { return &w.Layers[0] }
	requireRefused(t, arch, good, []shapeCase{
		{"QProj cols wrong", func(w *Weights) { l0(w).QProj = mat(q, h+1) }, "layer 0 QProj: 9 cols"},
		{"KProj cols wrong", func(w *Weights) { l0(w).KProj = mat(kv, 2*h) }, "layer 0 KProj: 16 cols"},
		{"VProj cols wrong", func(w *Weights) { l0(w).VProj = mat(kv, h-1) }, "layer 0 VProj: 7 cols"},
		{"OProj cols wrong", func(w *Weights) { l0(w).OProj = mat(h, q+4) }, "layer 0 OProj: 12 cols"},
		{"GateProj cols wrong", func(w *Weights) { l0(w).GateProj = mat(ffn, h+2) }, "layer 0 GateProj: 10 cols"},
		{"DownProj cols wrong", func(w *Weights) { l0(w).DownProj = mat(h, ffn+1) }, "layer 0 DownProj: 17 cols"},
		{"Router cols wrong", func(w *Weights) { l0(w).Router = mat(4, h+1) }, "layer 0 Router: 9 cols"},
		{"expert Down cols wrong", func(w *Weights) { l0(w).Experts[0].Down = mat(h, moeI+1) }, "layer 0 expert 0 Down: 13 cols"},
		{"expert Gate cols wrong", func(w *Weights) { l0(w).Experts[0].Gate = mat(moeI, h+3) }, "layer 0 expert 0 Gate: 11 cols"},
		{"shared expert Down cols wrong", func(w *Weights) { l0(w).SharedExpert.Down = mat(h, moeI-1) }, "layer 0 SharedExpert Down: 11 cols"},
		{"SharedGate cols wrong", func(w *Weights) { l0(w).SharedGate = mat(1, h+1) }, "layer 0 SharedGate: 9 cols"},
		{"GProj cols wrong", func(w *Weights) { l0(w).GProj = mat(2, h+1) }, "layer 0 GProj: 9 cols"},
		{"QProj absent", func(w *Weights) { l0(w).QProj = linalg.WeightMat{} }, "layer 0 QProj: absent"},
		{"VProj absent", func(w *Weights) { l0(w).VProj = linalg.WeightMat{} }, "layer 0 VProj: absent"},
		{"OProj absent", func(w *Weights) { l0(w).OProj = linalg.WeightMat{} }, "layer 0 OProj: absent"},
		{"Router absent on an expert layer", func(w *Weights) { l0(w).Router = linalg.WeightMat{} }, "layer 0 Router: absent"},
		{"dense DownProj absent on a dense layer", func(w *Weights) {
			l0(w).Experts, l0(w).Router = nil, linalg.WeightMat{}
			l0(w).DownProj = linalg.WeightMat{}
		}, "layer 0 DownProj: absent"},
	})
}

// The hybrid mixers are flat tensors their forwards index at arch widths and structs their forwards dereference: a layer
// whose program runs one the bundle lacks is a nil dereference at the first forward, and a short one an out-of-bounds read.
func TestValidateShapes_hybridMixers(t *testing.T) {
	const h = 8
	t.Run("qwen3.5 DeltaNet and gated attention", func(t *testing.T) {
		arch := &Architecture{
			VocabSize: 16, HiddenDim: h, NumHeads: 2, NumKVHeads: 1, HeadDim: 4, IntermediateDim: 16, NumLayers: 2,
			qwen35:        &qwen35Params{ConvKernel: 3, KeyHeadDim: 4, ValueHeadDim: 4, NumKeyHeads: 1, NumValueHeads: 2, AttnGate: true},
			layerIsLinear: func(i int) bool { return i == 0 },
		}
		keyDim, valueDim, numV := 4, 8, 2
		convDim := 2*keyDim + valueDim
		good := func() *Weights {
			l0, l1 := completeDenseLayer(arch, 0), completeDenseLayer(arch, 1)
			l0.QProj, l0.KProj, l0.VProj, l0.OProj = linalg.WeightMat{}, linalg.WeightMat{}, linalg.WeightMat{}, linalg.WeightMat{}
			l1.QProj, l1.KProj, l1.VProj, l1.OProj = linalg.WeightMat{}, linalg.WeightMat{}, linalg.WeightMat{}, linalg.WeightMat{}
			l0.delta = &deltaNetWeights{
				inProjQKV: mat(convDim, h), inProjZ: mat(valueDim, h), outProj: mat(h, valueDim),
				inProjB: make([]float32, numV*h), inProjA: make([]float32, numV*h), convW: make([]float32, convDim*3),
				dtBias: make([]float32, numV), negExpA: make([]float32, numV), normW: make([]float32, 4),
			}
			l1.qattn = &qwenAttnWeights{qProj: mat(16, h), kProj: mat(4, h), vProj: mat(4, h), oProj: mat(h, 8)}
			return &Weights{arch: arch, Layers: []LayerWeights{l0, l1}}
		}
		requireRefused(t, arch, good, []shapeCase{
			{"DeltaNet absent", func(w *Weights) { w.Layers[0].delta = nil }, "layer 0: the program runs a Gated DeltaNet"},
			{"gated attention absent", func(w *Weights) { w.Layers[1].qattn = nil }, "layer 1: the program runs a gated attention"},
			{"inProjQKV rows wrong", func(w *Weights) { w.Layers[0].delta.inProjQKV = mat(convDim-1, h) }, "delta.inProjQKV: 15 rows"},
			{"inProjZ cols wrong", func(w *Weights) { w.Layers[0].delta.inProjZ = mat(valueDim, h+1) }, "delta.inProjZ: 9 cols"},
			{"convW short", func(w *Weights) { w.Layers[0].delta.convW = make([]float32, convDim) }, "delta.convW"},
			{"negExpA absent", func(w *Weights) { w.Layers[0].delta.negExpA = nil }, "delta.negExpA: absent"},
			{"single-width gated q_proj", func(w *Weights) { w.Layers[1].qattn.qProj = mat(8, h) }, "qattn.qProj: 8 rows"},
			{"gated oProj cols wrong", func(w *Weights) { w.Layers[1].qattn.oProj = mat(h, 16) }, "qattn.oProj: 16 cols"},
		})
	})
	t.Run("MLA and KDA", func(t *testing.T) {
		arch := &Architecture{
			VocabSize: 16, HiddenDim: h, NumHeads: 2, NumKVHeads: 2, HeadDim: 4, IntermediateDim: 16, NumLayers: 2,
			mla:           &mlaParams{KVLoRARank: 4, QKNopeHeadDim: 2, QKRopeHeadDim: 2, VHeadDim: 3, GateGranularity: "head_wise"},
			kda:           &kdaParams{HeadDim: 2, NumHeads: 2, ConvKernel: 3},
			layerIsLinear: func(i int) bool { return i == 1 },
		}
		good := func() *Weights {
			l0, l1 := completeDenseLayer(arch, 0), completeDenseLayer(arch, 1)
			for _, l := range []*LayerWeights{&l0, &l1} {
				l.QProj, l.KProj, l.VProj, l.OProj = linalg.WeightMat{}, linalg.WeightMat{}, linalg.WeightMat{}, linalg.WeightMat{}
			}
			l0.mla = &mlaWeights{
				qProj: make([]float32, 2*4*h), kvAProj: make([]float32, (4+2)*h), kvALayernorm: make([]float32, 4),
				kvBProj: make([]float32, 2*(2+3)*4), oProj: make([]float32, h*2*3), gProj: make([]float32, 2*h),
			}
			l1.kda = &kdaWeights{
				qProj: mat(4, h), kProj: mat(4, h), vProj: mat(4, h), fProj: mat(4, h), gProj: mat(4, h), oProj: mat(h, 4),
				qConvW: make([]float32, 12), kConvW: make([]float32, 12), vConvW: make([]float32, 12),
				dtBias: make([]float32, 4), aLog: make([]float32, 2), bProj: make([]float32, 2*h), oNormW: make([]float32, 2),
			}
			return &Weights{arch: arch, Layers: []LayerWeights{l0, l1}}
		}
		requireRefused(t, arch, good, []shapeCase{
			{"MLA absent", func(w *Weights) { w.Layers[0].mla = nil }, "layer 0: the program runs a MLA attention"},
			{"KDA absent", func(w *Weights) { w.Layers[1].kda = nil }, "layer 1: the program runs a KDA attention"},
			{"kvBProj short", func(w *Weights) { w.Layers[0].mla.kvBProj = make([]float32, 8) }, "mla.kvBProj"},
			{"direct qProj absent", func(w *Weights) { w.Layers[0].mla.qProj = nil }, "mla.qProj: absent"},
			{"quantized oProj cols wrong", func(w *Weights) {
				w.Layers[0].mla.oProj, w.Layers[0].mla.oQ = nil, mat(h, 2*3+1)
			}, "mla.oProj: 7 cols"},
			{"element-wise gate on a head-wise arch", func(w *Weights) { w.Layers[0].mla.gProj = make([]float32, 6*h) }, "mla.gProj"},
			{"kvALayernorm absent", func(w *Weights) { w.Layers[0].mla.kvALayernorm = nil }, "mla.kvALayernorm: absent"},
			{"KDA qConvW short", func(w *Weights) { w.Layers[1].kda.qConvW = make([]float32, 4) }, "kda.qConvW"},
			{"KDA oProj cols wrong", func(w *Weights) { w.Layers[1].kda.oProj = mat(h, 5) }, "kda.oProj: 5 cols"},
		})
	})
	t.Run("Granite Mamba-2", func(t *testing.T) {
		arch := &Architecture{
			VocabSize: 16, HiddenDim: h, NumHeads: 2, NumKVHeads: 1, HeadDim: 4, IntermediateDim: 16, NumLayers: 2,
			granite:      &graniteParams{NHeads: 2, HeadDim: 2, DState: 2, NGroups: 1, DConv: 3},
			layerIsMamba: func(i int) bool { return i == 0 },
		}
		dInner := 4
		convDim, projDim := dInner+2*2, 2*dInner+2*2+2
		good := func() *Weights {
			l0, l1 := completeDenseLayer(arch, 0), completeDenseLayer(arch, 1)
			l0.QProj, l0.KProj, l0.VProj, l0.OProj = linalg.WeightMat{}, linalg.WeightMat{}, linalg.WeightMat{}, linalg.WeightMat{}
			l0.mamba = &mamba2Weights{
				inProj: make([]float32, projDim*h), convW: make([]float32, convDim*3), convB: make([]float32, convDim),
				aLog: make([]float32, 2), d: make([]float32, 2), dtBias: make([]float32, 2), normW: make([]float32, dInner),
				outProj: make([]float32, h*dInner),
			}
			return &Weights{arch: arch, Layers: []LayerWeights{l0, l1}}
		}
		requireRefused(t, arch, good, []shapeCase{
			{"Mamba-2 absent", func(w *Weights) { w.Layers[0].mamba = nil }, "layer 0: the program runs a Mamba-2 mixer"},
			{"inProj short", func(w *Weights) { w.Layers[0].mamba.inProj = make([]float32, projDim) }, "mamba.inProj"},
			{"quantized inProj rows wrong", func(w *Weights) {
				w.Layers[0].mamba.inProj, w.Layers[0].mamba.inQ = nil, mat(projDim+1, h)
			}, "mamba.inProj: 15 rows"},
			{"convB short", func(w *Weights) { w.Layers[0].mamba.convB = make([]float32, 1) }, "mamba.convB"},
			{"attention layer QProj absent", func(w *Weights) { w.Layers[1].QProj = linalg.WeightMat{} }, "layer 1 QProj: absent"},
		})
	})
}

// Gemma 4's dense||MoE layer and its PLE branch dereference every norm and projection unconditionally once the program runs
// them, so a bundle missing any of them must be refused at load, not panic in the decode goroutine.
func TestValidateShapes_gemma4MoEAndPLEComplete(t *testing.T) {
	const h, ffn, nE, moeI, pleDim = 8, 16, 4, 12, 4
	vec := func() []float32 { return make([]float32, h) }
	moeArch := &Architecture{
		VocabSize: 32, HiddenDim: h, NumHeads: 2, NumKVHeads: 1, HeadDim: 4, IntermediateDim: ffn, NumLayers: 1,
		MoE:    &MoEConfig{NumExperts: nE, TopK: 2, IntermediateDim: moeI},
		gemma4: &gemma4Params{GlobalHeadDim: 4, GlobalRotaryDim: 4},
	}
	moeGood := func() *Weights {
		lw := completeDenseLayer(moeArch, 0)
		lw.GateProj, lw.UpProj, lw.DownProj = linalg.WeightMat{}, linalg.WeightMat{}, linalg.WeightMat{}
		mo := &gemma4MoEWeights{
			preFFNNorm: vec(), postFFNNorm1: vec(), preFFNNorm2: vec(), postFFNNorm2: vec(), postFFNNorm: vec(),
			mlpGate: mat(ffn, h), mlpUp: mat(ffn, h), mlpDown: mat(h, ffn),
			routerProj: mat(nE, h), routerScale: vec(), perExpertScale: make([]float32, nE),
			denseInter: ffn, moeInter: moeI, nE: nE, topK: 2,
		}
		for range nE {
			mo.expertsGateUp = append(mo.expertsGateUp, mat(2*moeI, h))
			mo.expertsDown = append(mo.expertsDown, mat(h, moeI))
		}
		lw.gemma4moe = mo
		return &Weights{arch: moeArch, Embed: mat(32, h), Layers: []LayerWeights{lw}}
	}
	mo := func(w *Weights) *gemma4MoEWeights { return w.Layers[0].gemma4moe }
	requireRefused(t, moeArch, moeGood, []shapeCase{
		{"dense mlpGate absent", func(w *Weights) { mo(w).mlpGate = linalg.WeightMat{} }, "layer 0 gemma4moe mlpGate: absent"},
		{"dense mlpUp rows wrong", func(w *Weights) { mo(w).mlpUp = mat(ffn+1, h) }, "layer 0 gemma4moe mlpUp: 17 rows"},
		{"dense mlpDown cols wrong", func(w *Weights) { mo(w).mlpDown = mat(h, ffn-1) }, "layer 0 gemma4moe mlpDown: 15 cols"},
		{"preFFNNorm absent", func(w *Weights) { mo(w).preFFNNorm = nil }, "layer 0 gemma4moe preFFNNorm: absent"},
		{"postFFNNorm short", func(w *Weights) { mo(w).postFFNNorm = make([]float32, h-1) }, "layer 0 gemma4moe postFFNNorm: len 7"},
		{"postFFNNorm2 absent", func(w *Weights) { mo(w).postFFNNorm2 = nil }, "layer 0 gemma4moe postFFNNorm2: absent"},
	})

	pleArch := &Architecture{
		VocabSize: 32, HiddenDim: h, NumHeads: 2, NumKVHeads: 1, HeadDim: 4, IntermediateDim: ffn, NumLayers: 1,
		gemma4: &gemma4Params{GlobalHeadDim: 4, GlobalRotaryDim: 4, HiddenSizePerLayerInput: pleDim},
	}
	pleGood := func() *Weights {
		lw := completeDenseLayer(pleArch, 0)
		lw.PLEGate, lw.PLEProj, lw.PostPLENorm = mat(pleDim, h), mat(h, pleDim), vec()
		return &Weights{arch: pleArch, Embed: mat(32, h), Layers: []LayerWeights{lw},
			PerLayerTokenEmbed: mat(32, pleDim), PerLayerModelProj: mat(pleDim, h), PerLayerProjNorm: make([]float32, pleDim)}
	}
	l0 := func(w *Weights) *LayerWeights { return &w.Layers[0] }
	requireRefused(t, pleArch, pleGood, []shapeCase{
		{"PLEGate absent", func(w *Weights) { l0(w).PLEGate = linalg.WeightMat{} }, "layer 0 PLEGate: absent"},
		{"PLEProj absent", func(w *Weights) { l0(w).PLEProj = linalg.WeightMat{} }, "layer 0 PLEProj: absent"},
		{"PostPLENorm absent", func(w *Weights) { l0(w).PostPLENorm = nil }, "layer 0 PostPLENorm: absent"},
	})
}
