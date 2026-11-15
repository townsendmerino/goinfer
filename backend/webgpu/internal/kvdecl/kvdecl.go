// Package kvdecl holds WebGPU's check of a layer's declared K/V rows against what its build allocates. It takes plain numbers and the
// declaration (internal/runtime/program), and imports no decoder, so the check is testable without a device or a model.
package kvdecl

import (
	"fmt"

	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

// Check holds layer l's declared per-position rows (decoder.Model.LayerBuffers) to what its build allocates, before the caches are sized
// from its geometry: on a layer that owns an attention cache, K and V of kvDim elements in nKV heads at the KV precision; on an MLA
// layer (latent) one latent of kvDim elements declared f32; on a layer that reads another's cache (owns false), none. A row missing,
// extra, declared twice, or of another width, head count or precision is refused: the plan prices the declaration, so caches sized
// otherwise would not be what fit priced. A window is not compared: WebGPU holds the full context on every layer.
func Check(l int, bufs []program.Buf, owns, latent bool, kvDim, nKV int) error {
	rows := map[string]program.Buf{}
	for _, b := range bufs {
		if b.Unit != program.PerPosition {
			continue
		}
		if _, again := rows[b.Name]; again {
			return fmt.Errorf("gpu: layer %d: the program declares %s rows twice", l, b.Name)
		}
		rows[b.Name] = b
	}
	need := map[string]program.Buf{}
	switch {
	case !owns:
	case latent:
		need["latent"] = program.Buf{Prec: program.PrecF32, Elems: kvDim}
	default:
		kv := program.Buf{Prec: program.PrecKV, Elems: kvDim, Heads: nKV}
		need["k"], need["v"] = kv, kv
	}
	for name, b := range rows {
		w, ok := need[name]
		switch {
		case !ok:
			return fmt.Errorf("gpu: layer %d: the program declares %s rows, which its build has no cache for", l, name)
		case b.Elems != w.Elems || b.Prec != w.Prec || (w.Heads > 0 && b.Heads != w.Heads):
			return fmt.Errorf("gpu: layer %d: the program declares %s rows of %d elements in %d heads at precision %d; its build allocates %d in %d at %d",
				l, name, b.Elems, b.Heads, b.Prec, w.Elems, w.Heads, w.Prec)
		}
	}
	for name := range need {
		if _, ok := rows[name]; !ok {
			return fmt.Errorf("gpu: layer %d: the program declares no %s rows", l, name)
		}
	}
	return nil
}
