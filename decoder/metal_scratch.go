package decoder

import (
	"strings"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

// metalScratchBatchRows is the batch rows a Metal build allocates for ForwardBatch before any call grows them (ensureBatchCap(16)).
const metalScratchBatchRows = 16

// metalAttnFAPartialFloats is the fused attention's partial buffer a Metal build holds for a layer of head width hd with nH query and
// nKV K/V heads: nH × (hd + 2) floats for each of 32 splits, where attention_fa runs (head width 128, or 64 at a group of 7, and a
// group of at most 8), else none. It is backend/metal's attnFAPartialFloats, which TestMemGate_metal holds this price to.
func metalAttnFAPartialFloats(hd, nH, nKV int) int {
	if nKV <= 0 || nH%nKV != 0 {
		return 0
	}
	g := nH / nKV
	if !(hd == 128 || hd == 64 && g == 7) || g > 8 {
		return 0
	}
	return nH * (hd + 2) * 32
}

// MetalDecodeScratchBytes is what a Metal resident build of m leases on the scratch line for decode, the most it holds before a
// ForwardBatch grows its rows, at slots expert slots per layer (0 = every expert resident), as ResidentNeedBytes takes them: the
// decode list (backend/metal's decodeShape.buffers), the 16 batch rows (batchRows), and the greedy chain's buffers with a sampled
// chain's (chainBuffers). It is Metal's rule on the program's per-row declarations (metalDecodeRows): each declared row once, at the
// widest layer that declares it, and Metal's own terms beside them, read by the row's name (metalRowTerms). TestMemGate_metal holds
// it to the build's own list on every row. 0 for a model with no hidden or vocabulary width.
func (m *Model) MetalDecodeScratchBytes(slots int) int64 {
	H, _, nH, _, _, I, V := m.Dims()
	if H <= 0 || V <= 0 {
		return 0
	}
	d := m.metalDecodeRows(H, V)
	W := d.widest
	var f32, bytes, u32s int64 // f32 elements, bytes, and 4-byte uniforms
	for name, n := range W {
		if !metalSizedByRule[name] {
			f32 += int64(n)
		}
		if t := metalRowTerms[name]; t != nil {
			a, b, c := t(m, d, slots)
			f32, bytes, u32s = f32+a, bytes+b, u32s+c
		}
	}
	// The attention rows at the widest attention layer (4 where none runs: sized, never dispatched), and the FFN's fused gate | up at
	// the widest of every gate and up declared and the model's own width.
	nHhd, kvDim := W["q"], max(W["k"], W["v"])
	if nHhd == 0 {
		nHhd, kvDim = 4, 4
	}
	gu := max(I, W["gate"], W["up"], W["shared-gate"], W["shared-up"])
	q4k := m.Quant() == "q4k"
	aq := H
	if q4k {
		aq = 4 * H           // the q4k lane's head reads the f32 final norm from aq's slot 0
		f32 += int64(H + gu) // the lane's f32 activation rows
	}
	// q | k | v, gate | up, the context, a sublayer's second output row, the four activation scales, the argmax partials and token,
	// and the sampler's block keys and indices and its draw.
	f32 += int64(nHhd + 2*kvDim + 2*gu + nHhd + W["a"] + 4 + 2*((V+7)/8) + 1 + 2*((V+1023)/1024) + 1)
	// A sublayer input's int8 rows (attention's and the FFN's) and their f16 twins; the context's int8 row and f16 twin; the
	// activated FFN row's int8 row.
	bytes += int64(aq + H + 2*H + 2*H + nHhd + 2*nHhd + gu)
	u32s += 22 // the core's eleven, uAct, uAddOne, uWindow, uLNHasBias, and the sampler's seven
	type geomKey struct {
		hd, nKV, half int
		kEqV          bool
	}
	geoms := map[geomKey]bool{}
	fa, attn, layerFFN := 0, 0, 0
	w := m.Weights()
	for l, rows := range d.layers {
		s := d.shapes[l]
		if rows["q"] > 0 { // an attention layer: its geometry, its sink flag and window, and the fused attention's partial
			geoms[geomKey{s.HeadDim, s.KVHeads, len(m.RopeInvFreqLayerResident(l)), m.VFromKResident(l) && d.ownsKV[l]}] = true
			fa = max(fa, metalAttnFAPartialFloats(s.HeadDim, nH, s.KVHeads))
			attn++
		}
		if g := rows["gate"]; g != 0 && g != I && !d.routed[l] { // a dense FFN of its own width: its width's uniform
			layerFFN++
		}
		if q4k && l < len(w.Layers) { // the q4k lane's segments: one row-count uniform a run of parts of one kind in each fused group
			lw := &w.Layers[l]
			for _, group := range [][]*linalg.WeightMat{{&lw.QProj, &lw.KProj, &lw.VProj}, {&lw.OProj}, {&lw.DownProj}, {&lw.GateProj, &lw.UpProj}} {
				u32s += int64(m.metalQ4KRuns(group))
			}
		}
	}
	f32 += int64(fa)
	if fa > 0 {
		u32s += 2
	}
	if d.qkWhole {
		u32s += 2
	}
	u32s += int64(7*len(geoms) + 2*attn + layerFFN) // each geometry's seven; each attention layer's sink flag and window
	f32 += int64(metalScratchBatchRows * (H + V))   // batchX, batchLogits
	u32s += 3 * metalScratchBatchRows               // each batch row's position uniforms
	u32s += 12 + 7                                  // the greedy chain's two tokens and position sets; a sampled chain's temperature, keys and draws
	f32 += int64(2 * V)                             // a sampled chain's two logits rows
	return 4*f32 + bytes + 4*u32s
}

// MetalDecodeRowWidths is decode's per-row activations as m's program declares them, each name at the widest layer that declares
// it (program.Buffers' per-row entries and program.HeadBuffers): what a Metal build holds its decode rows to. A fresh map each call.
func (m *Model) MetalDecodeRowWidths() map[string]int {
	H, _, _, _, _, _, V := m.Dims()
	return m.metalDecodeRows(H, V).widest
}

// metalSizedByRule is the declared rows Metal does not hold as one f32 row of their width, which MetalDecodeScratchBytes sizes by its
// own rule: a sublayer's input (held only as int8 and f16 rows), the parallel block's second output (held in the FFN's output row),
// the attention rows (fused, with a floor) and every gate and up (one fused row at their widest).
var metalSizedByRule = map[string]bool{"x": true, "b": true, "q": true, "k": true, "v": true, "ctx": true, "gate": true, "up": true,
	"shared-gate": true, "shared-up": true}

// metalDecodeRows is decode's per-row activations as m's program declares them (program.Buffers' per-row entries and
// program.HeadBuffers): each name at the widest layer that declares it, each layer's own with its shape, whether the layer holds KV
// rows of its own, and whether it routes experts.
type metalDecodeRows struct {
	widest  map[string]int
	layers  []map[string]int
	shapes  []program.LayerShape
	ownsKV  []bool
	routed  []bool
	qkWhole bool // a QK-norm takes one statistic over every head
}

func (m *Model) metalDecodeRows(H, V int) metalDecodeRows {
	a, p := m.w.arch, m.Program()
	d := metalDecodeRows{widest: map[string]int{}}
	for l, ops := range p.Layers {
		s := layerShapeOf(a, l, ops)
		rows, owns, routed := map[string]int{}, false, false
		for _, o := range ops {
			d.qkWhole = d.qkWhole || o.Kind == program.OpQKNorm && o.QKNorm.Whole
			for _, b := range program.Buffers(o, s) {
				switch b.Unit {
				case program.PerRow:
					rows[b.Name] = max(rows[b.Name], b.Elems)
					d.widest[b.Name] = max(d.widest[b.Name], b.Elems)
					routed = routed || strings.HasSuffix(b.Name, "-router")
				case program.PerPosition:
					owns = true
				}
			}
		}
		d.layers, d.shapes, d.ownsKV, d.routed = append(d.layers, rows), append(d.shapes, s), append(d.ownsKV, owns), append(d.routed, routed)
	}
	for _, b := range program.HeadBuffers(p, program.HeadShape{Hidden: H, Vocab: V, PerLayerEmbed: d.widest["ple-gate"]}) {
		d.widest[b.Name] = max(d.widest[b.Name], b.Elems)
	}
	return d
}

// metalRowTerms is what a Metal build holds beside a declared row, by the row's name, as f32 elements, bytes and 4-byte uniforms:
// the int8 row and scale of a row its kernels quantize, a floor, a mixer's uniforms, a kernel's own rows, and the paged experts'
// slot indices at slots.
var metalRowTerms = map[string]func(m *Model, d metalDecodeRows, slots int) (f32, bytes, u32s int64){
	"ple-gate": func(m *Model, d metalDecodeRows, _ int) (int64, int64, int64) {
		return 1, int64(d.widest["ple-gate"]), 1
	},
	"dn-gated": func(m *Model, d metalDecodeRows, _ int) (int64, int64, int64) {
		u := int64(12)
		for _, ops := range m.Program().Layers {
			for _, o := range ops {
				if e := o.Delta.ONormEps; e != 0 && e != o.Delta.Eps {
					u = 13 // the output norm's own epsilon (an operation's parameter, which only the mixer that has it carries)
				}
			}
		}
		return 1, int64(d.widest["dn-gated"]), u
	},
	"mb-gated": func(m *Model, d metalDecodeRows, _ int) (int64, int64, int64) {
		return 1, int64(d.widest["mb-gated"]), 13
	},
	"mla-q": func(m *Model, d metalDecodeRows, _ int) (int64, int64, int64) {
		qLoRA := d.widest["mla-qa"]
		// q's LoRA row at 1 where there is none, the two scales; its int8 row at 4, the context's, and the five packed shapes
		return int64(max(qLoRA, 1) - qLoRA + 2), int64(max(qLoRA, 4) + d.widest["mla-ctx"] + 4*(3+5+5+7+3)), 5
	},
	"moe-router": func(m *Model, d metalDecodeRows, slots int) (int64, int64, int64) {
		return metalMoETerms(m, d, slots, "moe", true, 0)
	},
	"gptoss-router": func(m *Model, d metalDecodeRows, slots int) (int64, int64, int64) {
		return metalMoETerms(m, d, slots, "gptoss", false, 3)
	},
	"g4moe-router": func(m *Model, d metalDecodeRows, slots int) (int64, int64, int64) {
		nE, k, H := d.widest["g4moe-router"], d.widest["g4moe-experts"], d.widest["h"]
		bytes := int64(0)
		if metalMoEPaged(slots, nE) {
			bytes = int64(4 * k)
		}
		return int64(nE + 2*k + H), bytes, int64(10 + k) // the router's second set of rows and its input
	},
}

// metalMoETerms is what Metal's generic MoE holds beside its declared router rows (named by prefix): the shared expert's gate and
// output rows, its uniforms (extra beyond its base set of 10 + k), the k-slot kernels' rows where kSlots allows them and the widths
// divide, and the paged slot indices at slots.
func metalMoETerms(m *Model, d metalDecodeRows, slots int, prefix string, kSlots bool, extra int) (f32, bytes, u32s int64) {
	nE, k, H := d.widest[prefix+"-router"], d.widest[prefix+"-experts"], d.widest["h"]
	inter := 0 // the routed experts' width
	for l, rows := range d.layers {
		if d.routed[l] {
			inter = max(inter, rows["up"])
		}
	}
	f32, u32s = int64(1+H), int64(10+k+extra)
	if kSlots && (2*inter)%8 == 0 && H%8 == 0 {
		f32 += int64(k*2*inter + k + k*H)
		bytes += int64(k * inter)
	}
	if metalMoEPaged(slots, nE) {
		bytes += int64(4 * k)
	}
	return f32, bytes, u32s
}

// metalQ4KRuns is how many segments Metal's q4k lane uploads a fused group as: one a run of consecutive parts of one kind (the file's
// Q4_K, Q5_K or Q6_K blocks, else int8), as backend/metal's q4kLaneProjOn splits it. A part with no rows is not one the layer holds:
// it is not uploaded, so it neither starts a run nor splits one.
func (m *Model) metalQ4KRuns(group []*linalg.WeightMat) int {
	kind := func(wm *linalg.WeightMat) string {
		if wm.Kind() == "q4k" {
			return "q4k"
		}
		if k := m.KQuantNative("metal", wm); k != "" {
			return k
		}
		return "int8"
	}
	runs, prev := 0, ""
	for _, wm := range group {
		if wm.Rows() == 0 {
			continue
		}
		if k := kind(wm); runs == 0 || k != prev {
			runs++
			prev = k
		}
	}
	return runs
}

// metalMoEPaged reports whether a Metal build holding slots expert slots per layer pages its nE routed experts: a slot count above
// 0 and under the expert count, as backend/metal's MoE builds decide it.
func metalMoEPaged(slots, nE int) bool { return slots > 0 && slots < nE }
