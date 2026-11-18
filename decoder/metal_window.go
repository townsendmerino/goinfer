package decoder

import "sync"

// The Metal window group's pool. A window group keeps only each
// sequence's recent positions, so its pool can be a bounded number of blocks a slot instead of the context's. Whether a build may
// bound it depends on the build's own prefill routes (every pass that writes the group must fit the bound), which only the Metal
// backend knows, so the backend registers the function that decides and both the build and the plan call it.

const (
	// MetalKVWindowKeep is how many positions a Metal window group keeps behind the start of the window a write reads: the rollback
	// budget, so an edit late in a prompt still finds its window held.
	MetalKVWindowKeep = 448
	// MetalKVWindowPass is the most rows one pass may write into a bounded window group: the cap on a blocked prefill's block, and
	// the longest image turn or prompt pass the pool takes in one (a longer pass declines and the decoder runs it sequentially).
	MetalKVWindowPass = 2048
	// MetalKVWindowSlack is the blocks a bounded pool holds beyond the window, the pass and what the keep adds: the block being
	// written, the alignment of the base to a block, and the blocks a release has not yet freed.
	MetalKVWindowSlack = 4
)

// MetalKVWindowBlocks is the 64-row blocks one slot's window group holds when its pool is bounded: the window less one position, the
// larger of the retained rollback and one pass, in whole blocks, and the slack.
func MetalKVWindowBlocks(window int) int {
	return (window-1+max(MetalKVWindowKeep, MetalKVWindowPass)+KVBlockRows-1)/KVBlockRows + MetalKVWindowSlack
}

// MetalWindowPoolFunc decides, for a loaded model, how many blocks one slot's window group holds on Metal: MetalKVWindowBlocks of its
// window when every pass that writes the group is bounded by MetalKVWindowPass rows, 0 when the group keeps the context's blocks.
type MetalWindowPoolFunc func(m *Model) int

var (
	metalWindowMu   sync.Mutex
	metalWindowPool MetalWindowPoolFunc
)

// RegisterMetalWindowPool is called once by the Metal backend, at its init; a second registration panics. A binary without the Metal
// backend has none, and prices Metal's window pool at the context's blocks: more than a build allocates, the safe direction.
func RegisterMetalWindowPool(f MetalWindowPoolFunc) {
	metalWindowMu.Lock()
	defer metalWindowMu.Unlock()
	if metalWindowPool != nil {
		panic("decoder: the Metal window pool is registered twice")
	}
	metalWindowPool = f
}

// MetalWindowPoolRegistered reports whether a Metal backend registered its window pool function.
func MetalWindowPoolRegistered() bool {
	metalWindowMu.Lock()
	defer metalWindowMu.Unlock()
	return metalWindowPool != nil
}

// MetalWindowBlocks is the blocks a slot's window group holds on Metal for m, 0 when it keeps the context's blocks or no Metal backend
// registered.
func (m *Model) MetalWindowBlocks() int {
	metalWindowMu.Lock()
	f := metalWindowPool
	metalWindowMu.Unlock()
	if f == nil {
		return 0
	}
	return f(m)
}
