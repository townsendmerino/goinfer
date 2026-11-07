package program

// The memory each operation declares beyond its weights (MM5, docs/tasks/task-forward-architecture-2026-10.md §4.9b): a load's KV and
// recurrent state are the sum of its layers' declarations, priced and allocated from the same list.

// Unit is what a buffer's size counts by.
type Unit uint8

const (
	PerPosition Unit = iota + 1 // one row per position of a sequence: the KV cache
	PerSlot                     // one copy per sequence: recurrent and conv state
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
// layer's mixer are read.
type LayerShape struct {
	KVHeads, HeadDim int // attention's K and V rows: KVHeads × HeadDim each
	MLALatent        int // MLA's latent row: the KV LoRA rank plus the rope-carrying width
	DeltaNet         struct{ KeyHeads, KeyHeadDim, ValueHeads, ValueHeadDim, ConvKernel int }
	Mamba2           struct{ Heads, HeadDim, DState, ConvDim, ConvKernel int }
	KDA              struct{ Heads, HeadDim, ConvKernel int }
	ShortConv        struct{ Dim, LCache int }
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
	switch op.Kind {
	case OpAttention:
		return kv(op.Attn.Window)
	case OpGatedAttn:
		return kv(op.Gated.Attn.Window)
	case OpGemma4Attn:
		if !op.G4.OwnsKV {
			return nil // a KV-shared layer reads its source layer's rows
		}
		return kv(op.G4.Window)
	case OpMLA:
		return []Buf{{Name: "latent", Unit: PerPosition, Prec: PrecF32, Elems: s.MLALatent}}
	case OpDeltaNet:
		d := s.DeltaNet
		keys, vals := d.KeyHeads*d.KeyHeadDim, d.ValueHeads*d.ValueHeadDim
		return []Buf{state("window", (d.ConvKernel-1)*(2*keys+vals), true), state("state", d.ValueHeads*d.KeyHeadDim*d.ValueHeadDim, true)}
	case OpMamba2:
		m := s.Mamba2
		return []Buf{state("window", (m.ConvKernel-1)*m.ConvDim, true), state("state", m.Heads*m.HeadDim*m.DState, true)}
	case OpKDA:
		k := s.KDA
		win := (k.ConvKernel - 1) * k.Heads * k.HeadDim
		return []Buf{state("window-q", win, false), state("window-k", win, false), state("window-v", win, false), state("state", k.Heads*k.HeadDim*k.HeadDim, false)}
	case OpShortConv:
		return []Buf{state("window", (s.ShortConv.LCache-1)*s.ShortConv.Dim, false)}
	}
	return nil
}

// LayerBuffers is every buffer a layer's operations declare, in their order.
func LayerBuffers(ops []Op, s LayerShape) []Buf {
	var out []Buf
	for _, o := range ops {
		out = append(out, Buffers(o, s)...)
	}
	return out
}
