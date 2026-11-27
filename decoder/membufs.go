package decoder

import (
	"github.com/townsendmerino/goinfer/internal/runtime/memprice"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

// LayerBuffers is what layer l of the loaded model declares beyond its weights (program.Buffers): the list a backend sizes its KV pools
// from and the plan prices. It is read once per model where the model keeps a fit memo; a caller must not modify it.
func (m *Model) LayerBuffers(l int) []program.Buf {
	a := m.w.Arch
	memo := m.fitOf
	if memo == nil {
		return memprice.LayerBuffers(a, l)
	}
	memo.mu.Lock()
	defer memo.mu.Unlock()
	if memo.bufs == nil {
		memo.bufs = make([][]program.Buf, a.NumLayers)
		for i := range a.NumLayers {
			memo.bufs[i] = memprice.LayerBuffers(a, i)
		}
	}
	return memo.bufs[l]
}
