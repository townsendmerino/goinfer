//go:build cuda

package cuda

import (
	"math"
	"runtime"
	"sync"
)

// softcapParallelMin is the vocabulary size above which splitting the softcap across cores pays. Below it the fan-out
// loses, which is why there is a threshold rather than an unconditional split (the measurement:
// docs/code-notes/cuda.md#softcapParallelMin).
const softcapParallelMin = 32768

// applySoftcap applies softcap*tanh(x/softcap) elementwise and in place to a logit vector. It runs on the sampling path
// only: ForwardArgmax reduces on-device and reads back 4 bytes, so greedy decoding never pays it.
//
// Bit-identity is structural: every output element is a pure function of the single input element at the same index (no
// reduction, no accumulation, no ordering freedom), so splitting the range cannot change a bit whatever the worker
// count. That is why this stays float64 math.Tanh rather than a device kernel or a float32 approximation: both would be
// faster and neither would be the same number the CPU path produces (decoder/forwardn.go, decoder/model.go).
//
// Sibling set: the identical loop also lives in decoder/forwardn.go, decoder/model.go, metal/model.go and
// gpu/softcap.go. The decoder and metal copies are deliberately left alone (decoder/ is under a numerics freeze and
// metal/ is on hold; docs/QUEUE.md B6 records the pair).
func applySoftcap(logits []float32, sc float32) {
	if sc <= 0 {
		return
	}
	n := len(logits)
	if n < softcapParallelMin {
		for j, v := range logits {
			logits[j] = sc * float32(math.Tanh(float64(v/sc)))
		}
		return
	}
	w := min(runtime.GOMAXPROCS(0), n)
	chunk := (n + w - 1) / w
	var wg sync.WaitGroup
	for lo := 0; lo < n; lo += chunk {
		hi := min(lo+chunk, n)
		wg.Add(1)
		go func(part []float32) {
			defer wg.Done()
			for j, v := range part {
				part[j] = sc * float32(math.Tanh(float64(v/sc)))
			}
		}(logits[lo:hi])
	}
	wg.Wait()
}

// applyLogitScale multiplies a logit vector by scale in place: logits_scaling (FeatLogitScale) through
// decoder.Model.LogitScaleResident. A multiply is memory-bound, not compute-bound, so this is a single serial pass with
// no parallel threshold. scale==0 or ==1 is the no-op case of every family without the feature, kept here so a caller
// need not branch.
func applyLogitScale(logits []float32, scale float32) {
	if scale == 0 || scale == 1 {
		return
	}
	for j, v := range logits {
		logits[j] = v * scale
	}
}
