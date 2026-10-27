//go:build gpu && goinfer_testhooks

package gpu

import (
	"fmt"
	"github.com/townsendmerino/goinfer/gpu/internal/kernels"
)

// KScalesForTest reads back layer l's int8-KV K-scale buffer from a WebGPU resident decoder: one f32 per (position, KV head),
// laid out [pos*nKV + head]. It errors when the resident's KV is not int8, so a caller can never mistake an f32 cache for an
// unwritten scale buffer. It is the seam that pins the scale being written at the true position, not the rope position.
func KScalesForTest(rf any, l int) (scales []float32, nKV int, err error) {
	r, ok := rf.(*residentDecoder)
	if !ok {
		return nil, 0, fmt.Errorf("gpu: KScalesForTest: %T is not a WebGPU resident decoder", rf)
	}
	if !r.rm.KvI8 {
		return nil, 0, fmt.Errorf("gpu: KScalesForTest: the resident KV cache is not int8")
	}
	if l < 0 || l >= len(r.rm.Layers) || r.rm.Layers[l].KScale == nil {
		return nil, 0, fmt.Errorf("gpu: KScalesForTest: layer %d has no K-scale buffer", l)
	}
	b := r.rm.Layers[l].KScale
	s, err := kernels.ContextReadbackRaw(r.c, b, int(b.GetSize()/4))
	return s, r.nKV, err
}
