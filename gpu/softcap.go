//go:build gpu

package gpu

import (
	"math"
	"runtime"
	"sync"
)

// softcapParallelMin is the vocabulary size above which splitting the softcap across cores pays —
// the same crossover cuda/softcap.go measured (this is a host-side CPU loop regardless of which
// GPU backend produced the logits, so the same threshold applies unchanged).
const softcapParallelMin = 32768

// applySoftcap applies softcap·tanh(x/softcap) elementwise and in place to a logit vector —
// Gemma 2/3/4's final-logit softcap (FeatFinalLogitSoftcap), via decoder.Model.
// FinalLogitSoftcapResident. This is the WebGPU twin of cuda/softcap.go's applySoftcap /
// metal/model.go's softcapParallel — SIX sites now carry this identical loop
// (decoder/forwardn.go, decoder/model.go, cuda/prefill.go, cuda/resident.go, metal/model.go,
// and this one); see cuda/softcap.go's own comment for the bit-identity argument (every output
// element is a pure function of the single input element at the same index, so splitting the
// range cannot change a bit).
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
