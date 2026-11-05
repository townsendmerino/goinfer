package kvmgr

// The geometry: what a model's KV is made of, as the manager needs to know it. decoder derives it from an
// Architecture at load; nothing configures it.

// Retention is how long a layer group keeps a position.
type Retention uint8

const (
	// RetainFull keeps every position.
	RetainFull Retention = iota
	// RetainWindow keeps the last Span positions: a layer at position p reads [p-Span+1, p].
	RetainWindow
	// RetainChunk keeps the current Span-sized chunk: a layer at position p reads from the start of p's chunk.
	RetainChunk
)

func (r Retention) String() string {
	switch r {
	case RetainWindow:
		return "window"
	case RetainChunk:
		return "chunk"
	}
	return "full"
}

// Group is a set of layers with the same KV row and the same retention, so they share one block table.
type Group struct {
	Retention Retention
	Span      int   // the window or chunk length in positions; 0 for RetainFull
	Layers    []int // layer indices, ascending
	RowWidth  int   // elements in one position's K (and V) row for one layer: KV heads x head dim, or the MLA latent width
	Latent    bool  // the row is an MLA latent, not K and V
	KEqualsV  bool  // the layers' V equals their K, so a store may keep one of them
}

// State is one kind of recurrent state: per sequence, mutated in place per token, with no row per position.
type State struct {
	Kind   string // "deltanet", "mamba2", "shortconv", "kda"
	Layers []int
}

// Geometry is a model's KV shape.
type Geometry struct {
	NumLayers int
	Groups    []Group
	// Shared maps a layer that stores no KV of its own to the layer whose table it reads.
	Shared map[int]int
	States []State
}

// Recurrent reports whether a sequence carries recurrent state. Such a sequence can be continued only from exactly
// where it stopped: its state has no history to rewind to.
func (g Geometry) Recurrent() bool { return len(g.States) > 0 }

// GroupOf returns the index of the group that holds layer's rows, following Shared, or -1 for a layer with none
// (a recurrent layer).
func (g Geometry) GroupOf(layer int) int {
	if src, ok := g.Shared[layer]; ok {
		layer = src
	}
	for gi, grp := range g.Groups {
		for _, l := range grp.Layers {
			if l == layer {
				return gi
			}
		}
	}
	return -1
}

// FirstBlock is the first logical block a group still needs for a sequence that will next write position pos: block
// 0 for a full group, the block holding the oldest position a window layer at pos reads, the block holding the start
// of pos's chunk. One block of slack is kept behind a window, so a rollback inside a verify width stays readable.
func (grp Group) FirstBlock(pos int) int {
	switch grp.Retention {
	case RetainWindow:
		first := (pos-grp.Span+1)/BlockRows - 1
		return max(first, 0)
	case RetainChunk:
		if grp.Span <= 0 {
			return 0
		}
		return (pos / grp.Span) * grp.Span / BlockRows
	}
	return 0
}

// BlocksHeld is how many logical blocks a group holds for a sequence of n positions.
func (grp Group) BlocksHeld(n int) int {
	if n <= 0 {
		return 0
	}
	total := (n + BlockRows - 1) / BlockRows
	return total - grp.FirstBlock(n)
}
