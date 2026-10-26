package vision

import (
	"errors"
	"sync"
)

// errNoReload is ReleaseBlocks's refusal for a tower built from a caller's TensorSource: nothing here can open it again.
var errNoReload = errors.New("vision: this tower was built from a caller's tensor source and cannot reload its blocks; they stay")

// blockRelease is what lets a tower drop its encoder blocks from the host once a device holds them (ReleaseBlocks) and
// load them again on the host's next use (ensure): reload is the loader call that built the tower, which gives the same
// values again. The blocks are the bulk of a tower; the head and tail it keeps are a few MB.
type blockRelease[E any] struct {
	mu       sync.Mutex
	reload   func() (*E, error)
	released bool
}

func newBlockRelease[E any](reload func() (*E, error)) *blockRelease[E] {
	return &blockRelease[E]{reload: reload}
}

// release marks the blocks gone, after drop has cleared them; a nil r refuses.
func (r *blockRelease[E]) release(drop func()) error {
	if r == nil {
		return errNoReload
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	drop()
	r.released = true
	return nil
}

// ensure loads the blocks again when they were released, handing the fresh tower to take; a nil r, or blocks never
// released, is a no-op.
func (r *blockRelease[E]) ensure(take func(*E)) error {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if !r.released {
		return nil
	}
	fresh, err := r.reload()
	if err != nil {
		return err
	}
	take(fresh)
	r.released = false
	return nil
}

// isReleased reports whether the blocks are off the host now.
func (r *blockRelease[E]) isReleased() bool {
	if r == nil {
		return false
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.released
}

// ReleaseBlocks drops the tower's encoder blocks from the host, for a caller that has copied them to a device. The next
// host forward or weight export loads them again from the checkpoint the tower was loaded from, with the same values. It
// must not run concurrently with a forward or an export. A tower built from a caller's TensorSource
// (LoadQwen3VisionEncoderFrom) cannot reload and refuses, keeping its blocks.
func (e *Qwen3VisionEncoder) ReleaseBlocks() error {
	return e.rel.release(func() { e.blocks = nil; e.maps.dropReloads() })
}

func (e *Qwen3VisionEncoder) ensureBlocks() error {
	return e.rel.ensure(func(f *Qwen3VisionEncoder) { e.blocks = f.blocks; e.maps.adopt(&f.maps) })
}

// BlocksReleased reports whether ReleaseBlocks has dropped the blocks and no host use has loaded them since.
func (e *Qwen3VisionEncoder) BlocksReleased() bool { return e.rel.isReleased() }

// ReleaseBlocks is Qwen3VisionEncoder.ReleaseBlocks for GLM-OCR's tower.
func (e *GlmOcrVisionEncoder) ReleaseBlocks() error {
	return e.rel.release(func() { e.blocks = nil; e.maps.dropReloads() })
}

func (e *GlmOcrVisionEncoder) ensureBlocks() error {
	return e.rel.ensure(func(f *GlmOcrVisionEncoder) { e.blocks = f.blocks; e.maps.adopt(&f.maps) })
}

// BlocksReleased is Qwen3VisionEncoder.BlocksReleased for GLM-OCR's tower.
func (e *GlmOcrVisionEncoder) BlocksReleased() bool { return e.rel.isReleased() }

// ReleaseBlocks is Qwen3VisionEncoder.ReleaseBlocks for Qwen2.5-VL's tower. A resident encoder (EnableResident) keeps
// running on its device; only the CPU path and GPUWeights reload.
func (e *QwenVisionEncoder) ReleaseBlocks() error {
	return e.rel.release(func() { e.blocks = nil; e.maps.dropReloads() })
}

func (e *QwenVisionEncoder) ensureBlocks() error {
	return e.rel.ensure(func(f *QwenVisionEncoder) { e.blocks = f.blocks; e.maps.adopt(&f.maps) })
}

// BlocksReleased is Qwen3VisionEncoder.BlocksReleased for Qwen2.5-VL's tower.
func (e *QwenVisionEncoder) BlocksReleased() bool { return e.rel.isReleased() }

// ReleaseBlocks is Qwen3VisionEncoder.ReleaseBlocks for Gemma 4's tower (its encoder layers).
func (e *Gemma4Encoder) ReleaseBlocks() error {
	return e.rel.release(func() { e.layers = nil; e.maps.dropReloads() })
}

func (e *Gemma4Encoder) ensureBlocks() error {
	return e.rel.ensure(func(f *Gemma4Encoder) { e.layers = f.layers; e.maps.adopt(&f.maps) })
}

// BlocksReleased is Qwen3VisionEncoder.BlocksReleased for Gemma 4's tower.
func (e *Gemma4Encoder) BlocksReleased() bool { return e.rel.isReleased() }
