package memprice

import (
	"strings"

	"github.com/townsendmerino/aikit/linalg"

	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/kvmgr"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

const (
	MetalDefaultKVSlots   = 2
	MetalCtxDefault       = 4096
	MetalFitDefaultCtx    = 16384
	MetalCtxCeiling       = 32768
	MetalKVWindowKeep     = 448
	MetalKVWindowPass     = 2048
	MetalKVWindowSlack    = 4
	MetalScratchBatchRows = 16
)

// MetalWindowBlocksFor returns the 64-row blocks one slot's window group holds when bounded.
func MetalWindowBlocksFor(window int) int {
	return (window-1+max(MetalKVWindowKeep, MetalKVWindowPass)+kvmgr.BlockRows-1)/kvmgr.BlockRows + MetalKVWindowSlack
}

// MetalRowBytesFor returns one position's row of b in Metal's KV pools when delivered at precision delivered.
func MetalRowBytesFor(b program.Buf, delivered string) int64 {
	switch {
	case b.Prec == program.PrecF32 || delivered == mem.KVF32:
		return int64(b.Elems) * 4
	case delivered == mem.KVI8:
		return int64(b.Elems) + int64(b.Heads)*4
	}
	return int64(b.Elems) * 2
}

// MetalKVTables returns how many block tables a Metal resident holds for each slot.
func MetalKVTables(a *darch.Architecture) int {
	if a == nil {
		return 1
	}
	win, other := false, false
	for _, g := range a.KVGeometry().Groups {
		if g.Retention == darch.KVRetainWindow {
			win = true
		} else {
			other = true
		}
	}
	if win && other {
		return 2
	}
	return 1
}

// MetalKVPartsFor returns one slot's KV on Metal at precision delivered and its pools' null blocks.
func MetalKVPartsFor(a *darch.Architecture, ctx int, delivered string, bufs func(l int) []program.Buf, winBlocks int) (slot, null int64) {
	blocks := int64((ctx + kvmgr.BlockRows - 1) / kvmgr.BlockRows)
	win := map[int]bool{}
	if winBlocks > 0 {
		for _, g := range a.KVGeometry().Groups {
			if g.Retention == darch.KVRetainWindow {
				for _, l := range g.Layers {
					win[l] = true
				}
			}
		}
	}
	var perBlock, perBlockWin int64
	for l := range a.NumLayers {
		for _, b := range bufs(l) {
			if b.Unit != program.PerPosition {
				continue
			}
			if win[l] {
				perBlockWin += kvmgr.BlockRows * MetalRowBytesFor(b, delivered)
			} else {
				perBlock += kvmgr.BlockRows * MetalRowBytesFor(b, delivered)
			}
		}
	}
	if perBlock+perBlockWin == 0 {
		return 0, 0
	}
	return blocks*perBlock + min(blocks, int64(winBlocks))*perBlockWin + int64(MetalKVTables(a))*blocks*4, perBlock + perBlockWin
}

// MetalKVBytesFor prices ResidentKVBytes("metal") from the architecture alone (before any model or window-pool registration exists).
func MetalKVBytesFor(a *darch.Architecture, ctx int, delivered string) int64 {
	if a == nil || ctx <= 0 {
		return 0
	}
	slot, null := MetalKVPartsFor(a, ctx, delivered, LayerBuffersOf(a), 0)
	return slot + null
}

// MetalKVCap returns one conversation's raised context cap on a Metal load of slots KV slots at share per-slot positions.
func MetalKVCap(slots, share, maxPositions int, pinned bool, extraPerPos int64, delivered string) int {
	if slots < 2 || pinned || extraPerPos != 0 || delivered == mem.KVI8 {
		return share
	}
	c := min(slots*((share+kvmgr.BlockRows-1)/kvmgr.BlockRows)*kvmgr.BlockRows, MetalCtxCeiling)
	if maxPositions > 0 {
		c = min(c, maxPositions)
	}
	return max(c, share)
}

// MetalKVTableRaiseBytes returns the extra block-table bytes across slots when one conversation's cap is raised from ctx to capCtx.
func MetalKVTableRaiseBytes(a *darch.Architecture, slots, ctx, capCtx int, hasKV bool) int64 {
	if ctx <= 0 || !hasKV {
		return 0
	}
	blocks := func(n int) int64 { return int64((n + kvmgr.BlockRows - 1) / kvmgr.BlockRows) }
	return int64(MetalKVTables(a)) * int64(slots) * (blocks(capCtx) - blocks(ctx)) * 4
}

// MetalAttnFAPartialFloats returns Metal's fused-attention partial buffer floats for a layer of head width hd with nH query and nKV K/V heads.
func MetalAttnFAPartialFloats(hd, nH, nKV int) int {
	if nKV <= 0 || nH%nKV != 0 {
		return 0
	}
	g := nH / nKV
	if !(hd == 128 || hd == 64 && g == 7) || g > 8 {
		return 0
	}
	return nH * (hd + 2) * 32
}

// MetalMoEPaged reports whether a Metal build holding slots expert slots per layer pages its nE routed experts.
func MetalMoEPaged(slots, nE int) bool { return slots > 0 && slots < nE }

// MetalQ4KRuns counts consecutive runs of one segment kind in a fused projection group on Metal's q4k lane.
func MetalQ4KRuns(group []*linalg.WeightMat, kquantKind func(*linalg.WeightMat) string) int {
	kind := func(wm *linalg.WeightMat) string {
		if wm.Kind() == "q4k" {
			return "q4k"
		}
		if kquantKind != nil {
			if k := kquantKind(wm); k != "" {
				return k
			}
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

var metalSizedByRule = map[string]bool{"x": true, "b": true, "q": true, "k": true, "v": true, "ctx": true, "gate": true, "up": true,
	"shared-gate": true, "shared-up": true}

// MetalDecodeRows holds decode's per-row activations declared by a program: each name at its widest layer and per-layer metadata.
type MetalDecodeRows struct {
	Widest  map[string]int
	Layers  []map[string]int
	Shapes  []program.LayerShape
	OwnsKV  []bool
	Routed  []bool
	QKWhole bool
}

// ComputeMetalDecodeRows derives MetalDecodeRows from a and p for hidden width H and vocab size V.
func ComputeMetalDecodeRows(a *darch.Architecture, p *program.Program, H, V int) MetalDecodeRows {
	d := MetalDecodeRows{Widest: map[string]int{}}
	for l, ops := range p.Layers {
		s := LayerShapeOf(a, l, ops)
		rows, owns, routed := map[string]int{}, false, false
		for _, o := range ops {
			d.QKWhole = d.QKWhole || o.Kind == program.OpQKNorm && o.QKNorm.Whole
			for _, b := range program.Buffers(o, s) {
				switch b.Unit {
				case program.PerRow:
					rows[b.Name] = max(rows[b.Name], b.Elems)
					d.Widest[b.Name] = max(d.Widest[b.Name], b.Elems)
					routed = routed || strings.HasSuffix(b.Name, "-router")
				case program.PerPosition:
					owns = true
				}
			}
		}
		d.Layers, d.Shapes, d.OwnsKV, d.Routed = append(d.Layers, rows), append(d.Shapes, s), append(d.OwnsKV, owns), append(d.Routed, routed)
	}
	for _, b := range program.HeadBuffers(p, program.HeadShape{Hidden: H, Vocab: V, PerLayerEmbed: d.Widest["ple-gate"]}) {
		d.Widest[b.Name] = max(d.Widest[b.Name], b.Elems)
	}
	return d
}

// MetalRowTerms maps declared per-row buffer names to Metal's extra f32 elements, bytes, and 4-byte uniforms.
var MetalRowTerms = map[string]func(p *program.Program, d MetalDecodeRows, slots int) (f32, bytes, u32s int64){
	"ple-gate": func(_ *program.Program, d MetalDecodeRows, _ int) (int64, int64, int64) {
		return 1, int64(d.Widest["ple-gate"]), 1
	},
	"dn-gated": func(p *program.Program, d MetalDecodeRows, _ int) (int64, int64, int64) {
		u := int64(12)
		for _, ops := range p.Layers {
			for _, o := range ops {
				if e := o.Delta.ONormEps; e != 0 && e != o.Delta.Eps {
					u = 13
				}
			}
		}
		return 1, int64(d.Widest["dn-gated"]), u
	},
	"mb-gated": func(_ *program.Program, d MetalDecodeRows, _ int) (int64, int64, int64) {
		return 1, int64(d.Widest["mb-gated"]), 13
	},
	"mla-q": func(_ *program.Program, d MetalDecodeRows, _ int) (int64, int64, int64) {
		qLoRA := d.Widest["mla-qa"]
		return int64(max(qLoRA, 1) - qLoRA + 2), int64(max(qLoRA, 4) + d.Widest["mla-ctx"] + 4*(3+5+5+7+3)), 5
	},
	"moe-router": func(_ *program.Program, d MetalDecodeRows, slots int) (int64, int64, int64) {
		return MetalMoETerms(d, slots, "moe", true, 0)
	},
	"gptoss-router": func(_ *program.Program, d MetalDecodeRows, slots int) (int64, int64, int64) {
		return MetalMoETerms(d, slots, "gptoss", false, 3)
	},
	"g4moe-router": func(_ *program.Program, d MetalDecodeRows, slots int) (int64, int64, int64) {
		nE, k, H := d.Widest["g4moe-router"], d.Widest["g4moe-experts"], d.Widest["h"]
		bytes := int64(0)
		if MetalMoEPaged(slots, nE) {
			bytes = int64(4 * k)
		}
		return int64(nE + 2*k + H), bytes, int64(10 + k)
	},
}

// MetalMoETerms returns Metal's generic MoE scratch terms beside its declared router rows.
func MetalMoETerms(d MetalDecodeRows, slots int, prefix string, kSlots bool, extra int) (f32, bytes, u32s int64) {
	nE, k, H := d.Widest[prefix+"-router"], d.Widest[prefix+"-experts"], d.Widest["h"]
	inter := 0
	for l, rows := range d.Layers {
		if d.Routed[l] {
			inter = max(inter, rows["up"])
		}
	}
	f32, u32s = int64(1+H), int64(10+k+extra)
	if kSlots && (2*inter)%8 == 0 && H%8 == 0 {
		f32 += int64(k*2*inter + k + k*H)
		bytes += int64(k * inter)
	}
	if MetalMoEPaged(slots, nE) {
		bytes += int64(4 * k)
	}
	return f32, bytes, u32s
}

// MetalDecodeScratchBytes prices what a Metal resident build leases on the scratch line for decode at slots expert slots per layer.
func MetalDecodeScratchBytes(w *darch.Weights, p *program.Program, quant string, slots int, kquantKind func(*linalg.WeightMat) string) int64 {
	if w == nil || w.Arch == nil || p == nil {
		return 0
	}
	a := w.Arch
	H, I, V := a.HiddenDim, a.IntermediateDim, a.VocabSize
	if H <= 0 || V <= 0 {
		return 0
	}
	d := ComputeMetalDecodeRows(a, p, H, V)
	W := d.Widest
	var f32, bytes, u32s int64
	for name, n := range W {
		if !metalSizedByRule[name] {
			f32 += int64(n)
		}
		if t := MetalRowTerms[name]; t != nil {
			af, ab, au := t(p, d, slots)
			f32, bytes, u32s = f32+af, bytes+ab, u32s+au
		}
	}
	nHhd, kvDim := W["q"], max(W["k"], W["v"])
	if nHhd == 0 {
		nHhd, kvDim = 4, 4
	}
	gu := max(I, W["gate"], W["up"], W["shared-gate"], W["shared-up"])
	q4k := quant == "q4k"
	aq := H
	if q4k {
		aq = 4 * H
		f32 += int64(H + gu)
	}
	f32 += int64(nHhd + 2*kvDim + 2*gu + nHhd + W["a"] + 4 + 2*((V+7)/8) + 1 + 2*((V+1023)/1024) + 1)
	bytes += int64(aq + H + 2*H + 2*H + nHhd + 2*nHhd + gu)
	u32s += 22
	type geomKey struct {
		hd, nKV, half int
		kEqV          bool
	}
	geoms := map[geomKey]bool{}
	fa, attn, layerFFN := 0, 0, 0
	for l, rows := range d.Layers {
		s := d.Shapes[l]
		if rows["q"] > 0 {
			geoms[geomKey{s.HeadDim, s.KVHeads, len(LayerRopeInvFreq(a, l)), VFromK(w, l) && d.OwnsKV[l]}] = true
			fa = max(fa, MetalAttnFAPartialFloats(s.HeadDim, s.Heads, s.KVHeads))
			attn++
		}
		if g := rows["gate"]; g != 0 && g != I && !d.Routed[l] {
			layerFFN++
		}
		if q4k && rows["q"] > 0 && rows["gate"] > 0 && !d.Routed[l] && l < len(w.Layers) {
			lw := &w.Layers[l]
			for _, group := range [][]*linalg.WeightMat{{&lw.QProj, &lw.KProj, &lw.VProj}, {&lw.OProj}, {&lw.DownProj}, {&lw.GateProj, &lw.UpProj}} {
				u32s += int64(MetalQ4KRuns(group, kquantKind))
			}
		}
	}
	f32 += int64(fa)
	if fa > 0 {
		u32s += 2
	}
	if d.QKWhole {
		u32s += 2
	}
	u32s += int64(7*len(geoms) + 2*attn + layerFFN)
	f32 += int64(MetalScratchBatchRows * (H + V))
	u32s += 3 * MetalScratchBatchRows
	u32s += 12 + 7
	f32 += int64(2 * V)
	return 4*f32 + bytes + 4*u32s
}

// MetalDerivedDenseBytes returns the dense-line buffers a Metal resident build derives beyond the load's tensors.
func MetalDerivedDenseBytes(w *darch.Weights, p *program.Program) int64 {
	if w == nil || w.Arch == nil {
		return 0
	}
	a := w.Arch
	nH := a.NumHeads
	dnetOK := a.Qwen35 != nil
	attnGate := dnetOK && a.Qwen35.AttnGate
	isMoE := a.MoE != nil && a.MoE.NumExperts > 0
	nE := 0
	if isMoE {
		nE = a.MoE.NumExperts
	}
	nonGated := false
	if p != nil {
		if s, err := program.ShapeOf(p); err == nil {
			nonGated = s.NonGated
		}
	}
	var floats, bytes int64
	maxHd, firstHalf := 0, -1
	for l := range w.Layers {
		lw := &w.Layers[l]
		mixer, hasFFN := LayerOps(a, l)
		if mixer == program.MixerAttention {
			hd := a.HeadDimAt(l)
			kvDim := a.KVHeadsAt(l) * hd
			maxHd = max(maxHd, hd)
			if len(lw.AttnSinks) == 0 {
				floats++
			}
			half := len(LayerRopeInvFreq(a, l))
			if half > 0 {
				floats += int64(half) + 1
			}
			if firstHalf < 0 {
				firstHalf = half
			}
			if !attnGate {
				if lw.QBias == nil {
					floats += int64(nH*hd+2*kvDim) - int64(len(lw.KBias)+len(lw.VBias))
				}
				if a.Gemma4KVSrcAt(l) == l && VFromK(w, l) {
					bytes += metalW4A8Bytes(lw.KProj.Rows(), lw.KProj.Cols())
				}
			}
		}
		g4 := lw.Gemma4MoE
		switch {
		case !hasFFN:
		case g4 != nil && a.Gemma4 != nil:
			floats += int64(g4.NE) + 1 - int64(len(g4.RouterScale))
		case isMoE && len(lw.Experts) > 0:
			if len(lw.RouterBias) == 0 {
				floats += int64(nE)
			}
		case nonGated:
			if len(lw.UpBias) == 0 {
				floats += int64(lw.UpProj.Rows())
			}
			if len(lw.DownBias) == 0 {
				floats += int64(lw.DownProj.Rows())
			}
		}
		if g4 == nil && a.Gemma4 != nil {
			if s := lw.LayerScalar; s != 0 && s != 1 {
				floats++
			}
		}
	}
	floats += int64(len(a.RoPEInvFreq(0)))
	if maxHd == 0 {
		maxHd = 4
	}
	floats += int64(maxHd)
	if firstHalf > 0 {
		floats += int64(len(MRoPEAxis(a, firstHalf)))
	}
	return 4*floats + bytes
}

func metalW4A8Bytes(rows, cols int) int64 {
	n := int64(rows) * int64(cols)
	return n/2 + 2*(n/32)
}
