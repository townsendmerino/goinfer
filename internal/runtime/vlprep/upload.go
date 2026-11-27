package vlprep

import (
	"fmt"

	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/decoder/resident"
	"github.com/townsendmerino/goinfer/internal/runtime/cpuengine"
	"github.com/townsendmerino/goinfer/internal/runtime/kvmgr"
)

// UploadPrefillKV uploads every non-shared layer's live K/V rows from cache into the resident backend via uploadLayer,
// computing and applying the sliding-window ring floor on slots (or invalidating IDs and raising the floor on error).
func UploadPrefillKV(arch *darch.Architecture, cache *cpuengine.Cache, kv resident.KVSubsystem, slots *kvmgr.Slots, forgetIDs func(), uploadLayer func(layer, base int, k, v []float32) error) (err error) {
	floor := 0
	for l := 0; l < arch.NumLayers; l++ {
		if r := cache.Rings[l]; r != nil && arch.Gemma4KVSrcAt(l) == l {
			if _, _, base := cache.LayerKV(l); base > 0 {
				floor = max(floor, base+r.W-1)
			}
		}
	}
	defer func() {
		if err == nil {
			slots.SetFloor(slots.Bound(), floor)
			return
		}
		forgetIDs()
		slots.RaiseFloor(slots.Bound(), floor)
	}()
	if kv != nil && kv.KVUploadPassAvailable() {
		if err := kv.BeginKVUpload(); err != nil {
			return err
		}
		defer func() {
			if e := kv.EndKVUpload(err != nil); err == nil {
				err = e
			}
		}()
	}
	for l := 0; l < arch.NumLayers; l++ {
		if arch.Gemma4KVSrcAt(l) != l {
			continue
		}
		k, v, base := cache.LayerKV(l)
		if len(k) == 0 {
			continue
		}
		if err := uploadLayer(l, base, k, v); err != nil {
			return fmt.Errorf("decoder: resident prefill upload layer %d (base=%d): %w", l, base, err)
		}
	}
	return nil
}
