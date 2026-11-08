package program

import "slices"

// The memory each operation declares beyond its weights (MM5, docs/tasks/task-forward-architecture-2026-10.md §4.9b and §4.9c): a load's
// KV, recurrent state and per-row activations are its layers' declarations and the head's, priced and allocated from the same list.

// Unit is what a buffer's size counts by.
type Unit uint8

const (
	PerPosition Unit = iota + 1 // one row per position of a sequence: the KV cache
	PerSlot                     // one copy per sequence: recurrent and conv state
	PerRow                      // one copy per row of a step: an activation an operation writes beyond the residual stream, f32
)

// Prec is a buffer's element type.
type Prec uint8

const (
	PrecKV  Prec = iota + 1 // the KV cache's precision as the backend delivers it (f16, f32, or int8 with one f32 scale per head)
	PrecF32                 // f32 whatever the KV precision: MLA's latent row, recurrent state
)

// Buf is one buffer an operation needs beyond its weights: Elems elements per Unit. Heads is a KV row's head count, which int8's
// per-head scales count by; Window, when not 0, is the most recent positions a windowed layer reads; Checkpointed marks state a rewind
// checkpoint copies.
type Buf struct {
	Name         string
	Unit         Unit
	Prec         Prec
	Elems, Heads int
	Window       int
	Checkpointed bool
}

// LayerShape is one layer's widths as the weights fix them, which its operations' buffers are sized by. Only the fields of the
// layer's operations are read.
type LayerShape struct {
	KVHeads, HeadDim int // attention's K and V rows: KVHeads × HeadDim each
	MLALatent        int // MLA's latent row: the KV LoRA rank plus the rope-carrying width
	DeltaNet         struct{ KeyHeads, KeyHeadDim, ValueHeads, ValueHeadDim, ConvKernel int }
	Mamba2           struct{ Heads, HeadDim, DState, ConvDim, ConvKernel int }
	KDA              struct{ Heads, HeadDim, ConvKernel int }
	ShortConv        struct{ Dim, LCache int }
	// The per-row activations' widths: the residual width, the query heads, and the dense FFN's width (its up rows).
	Hidden, Heads, FFN int
	MLA                struct{ QLoRA, Rank, QKHead, VHead int } // the q LoRA rank (0 for none), the KV LoRA rank, a q·k head, a value head
	Gemma4MoE          struct{ Experts, TopK, ExpertFFN int }   // Gemma 4's MoE block: its routed experts, top k, and an expert's FFN width
	PLE                int                                      // Gemma 4's per-layer-embedding width
}

// HeadShape is the widths of what no layer owns: the residual stream, the vocabulary, and the embedding stage's per-layer inputs.
type HeadShape struct {
	Hidden, Vocab int
	PLE           int // Gemma 4's per-layer-embedding width; read only where the program's embedding stage has the branch
}

// HeadBuffers is the per-row activations no layer owns, for p at shape s: the residual stream, a sublayer's input and output, the
// parallel block's second output where a layer writes it, the logits, and the per-layer-embedding inputs where the embedding has them.
func HeadBuffers(p *Program, s HeadShape) []Buf {
	row := func(name string, n int) Buf { return Buf{Name: name, Unit: PerRow, Prec: PrecF32, Elems: n} }
	out := []Buf{row("h", s.Hidden), row("x", s.Hidden), row("a", s.Hidden)}
	for _, ops := range p.Layers {
		if slices.ContainsFunc(ops, func(o Op) bool { return o.Out == RegB }) {
			out = append(out, row("b", s.Hidden))
			break
		}
	}
	out = append(out, row("logits", s.Vocab))
	if p.Embed.PLE {
		out = append(out, row("ple-in", len(p.Layers)*s.PLE))
	}
	return out
}

// Buffers is what op needs beyond its weights on a layer of shape s, nil for none.
func Buffers(op Op, s LayerShape) []Buf {
	kv := func(window int) []Buf {
		w := s.KVHeads * s.HeadDim
		return []Buf{
			{Name: "k", Unit: PerPosition, Prec: PrecKV, Elems: w, Heads: s.KVHeads, Window: window},
			{Name: "v", Unit: PerPosition, Prec: PrecKV, Elems: w, Heads: s.KVHeads, Window: window},
		}
	}
	state := func(name string, elems int, ckpt bool) Buf {
		return Buf{Name: name, Unit: PerSlot, Prec: PrecF32, Elems: elems, Checkpointed: ckpt}
	}
	rows := func(bufs []Buf, nameElems ...any) []Buf {
		for i := 0; i < len(nameElems); i += 2 {
			bufs = append(bufs, Buf{Name: nameElems[i].(string), Unit: PerRow, Prec: PrecF32, Elems: nameElems[i+1].(int)})
		}
		return bufs
	}
	q, kvw := s.Heads*s.HeadDim, s.KVHeads*s.HeadDim
	switch op.Kind {
	case OpProjQKV:
		return rows(nil, "q", q, "k", kvw, "v", kvw)
	case OpAttention:
		return rows(kv(op.Attn.Window), "ctx", q)
	case OpGatedAttn:
		return rows(kv(op.Gated.Attn.Window), "q-gate", 2*q, "q", q, "k", kvw, "v", kvw, "attn-gate", q, "ctx", q)
	case OpGemma4Attn:
		if !op.G4.OwnsKV {
			return rows(nil, "q", q, "ctx", q) // a KV-shared layer reads its source layer's rows
		}
		return rows(kv(op.G4.Window), "q", q, "k", kvw, "v", kvw, "ctx", q)
	case OpMLA:
		m := s.MLA
		return rows([]Buf{{Name: "latent", Unit: PerPosition, Prec: PrecF32, Elems: s.MLALatent}},
			"mla-qa", m.QLoRA, "mla-q", s.Heads*m.QKHead, "mla-kv", s.MLALatent, "mla-q-abs", s.Heads*s.MLALatent, "mla-wsum", s.Heads*m.Rank,
			"mla-ctx", s.Heads*m.VHead)
	case OpDeltaNet:
		d := s.DeltaNet
		keys, vals := d.KeyHeads*d.KeyHeadDim, d.ValueHeads*d.ValueHeadDim
		return rows([]Buf{state("window", (d.ConvKernel-1)*(2*keys+vals), true), state("state", d.ValueHeads*d.KeyHeadDim*d.ValueHeadDim, true)},
			"dn-mixed", 2*keys+vals, "dn-conv", 2*keys+vals, "dn-beta", d.ValueHeads, "dn-decay", d.ValueHeads, "dn-ba", 2*d.ValueHeads,
			"dn-q", keys, "dn-k", keys, "dn-core", vals, "dn-z", vals, "dn-gated", vals)
	case OpMamba2:
		m := s.Mamba2
		inner := m.Heads * m.HeadDim
		return rows([]Buf{state("window", (m.ConvKernel-1)*m.ConvDim, true), state("state", m.Heads*m.HeadDim*m.DState, true)},
			"mb-proj", inner+m.ConvDim+m.Heads, "mb-conv", m.ConvDim, "mb-y", inner, "mb-gated", inner)
	case OpFFN:
		switch op.FFN.Form {
		case FfnGated, FfnGemma4Dense:
			return rows(nil, "gate", s.FFN, "up", s.FFN)
		case FfnNonGated:
			return rows(nil, "up", s.FFN)
		}
		if prefix, ok := moeRows[op.FFN.Form]; ok {
			mo, gated := op.FFN.MoE, op.FFN.Form != FfnNemotronMoE // Nemotron's experts are not gated
			ffn := func(out []Buf, name string, n int) []Buf {
				if gated {
					out = rows(out, name+"gate", n)
				}
				return rows(out, name+"up", n)
			}
			out := ffn(nil, "", mo.IntermediateDim)
			if mo.SharedIntermediateDim > 0 {
				out = ffn(out, "shared-", mo.SharedIntermediateDim)
			}
			return rows(out, prefix+"-router", mo.NumExperts, prefix+"-experts", mo.TopK, prefix+"-weights", mo.TopK)
		}
	case OpGemma4MoE:
		g, ffn := s.Gemma4MoE, max(s.FFN, s.Gemma4MoE.ExpertFFN)
		return rows(nil, "gate", ffn, "up", ffn, "g4moe-router", g.Experts, "g4moe-experts", g.TopK, "g4moe-weights", g.TopK,
			"g4moe-dense-out", s.Hidden, "g4moe-moe-out", s.Hidden, "g4moe-router-in", s.Hidden)
	case OpPLE:
		return rows(nil, "ple-gate", s.PLE)
	case OpKDA:
		k := s.KDA
		win := (k.ConvKernel - 1) * k.Heads * k.HeadDim
		return []Buf{state("window-q", win, false), state("window-k", win, false), state("window-v", win, false), state("state", k.Heads*k.HeadDim*k.HeadDim, false)}
	case OpShortConv:
		return []Buf{state("window", (s.ShortConv.LCache-1)*s.ShortConv.Dim, false)}
	}
	return nil
}

// moeRows names each MoE FFN form's router rows (its logits, the chosen experts, their weights), so a backend that holds them per
// form can tell them apart.
var moeRows = map[FFNForm]string{FfnMoE: "moe", FfnGptOss: "gptoss", FfnLlama4MoE: "llama4moe", FfnNemotronMoE: "nemotronmoe"}

// LayerBuffers is every buffer a layer's operations declare, in their order.
func LayerBuffers(ops []Op, s LayerShape) []Buf {
	var out []Buf
	for _, o := range ops {
		out = append(out, Buffers(o, s)...)
	}
	return out
}
