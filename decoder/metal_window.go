package decoder

import (
	"sync"

	"github.com/townsendmerino/goinfer/internal/runtime/memprice"
)

const (
	// MetalKVWindowKeep is how many positions a Metal window group keeps behind the start of the window a write reads.
	MetalKVWindowKeep = memprice.MetalKVWindowKeep
	// MetalKVWindowPass is the most rows one pass may write into a bounded window group.
	MetalKVWindowPass = memprice.MetalKVWindowPass
	// MetalKVWindowSlack is the blocks a bounded pool holds beyond the window, the pass and what the keep adds.
	MetalKVWindowSlack = memprice.MetalKVWindowSlack
)

// MetalKVWindowBlocks is the 64-row blocks one slot's window group holds when its pool is bounded.
func MetalKVWindowBlocks(window int) int { return memprice.MetalWindowBlocksFor(window) }

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
