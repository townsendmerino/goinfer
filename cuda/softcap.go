//go:build cuda

package cuda

import (
	"math"
	"runtime"
	"sync"
)

// softcapParallelMin is the vocabulary size above which splitting the softcap across cores pays.
// MEASURED (this box, GOMAXPROCS 16, RTX 2070 SUPER host, 2026-08-12) — the crossover is real and
// the small end is a LOSS, which is why there is a threshold rather than an unconditional fan-out:
//
//	vocab      serial    parallel   speedup
//	  8 192   45.7 us    48.2 us     0.95x   <- slower
//	 32 768  166.3 us   117.4 us     1.42x
//	 65 536  422.9 us   207.9 us     2.03x
//	131 072  747.5 us   371.9 us     2.01x
//	262 144    1.47 ms   639.8 us    2.30x   <- Gemma 3/4
const softcapParallelMin = 32768

// applySoftcap applies softcap·tanh(x/softcap) elementwise and in place to a logit vector.
//
// This runs on the SAMPLING path only. ForwardArgmax reduces the argmax on-device and reads back
// 4 bytes, never materialising the logit vector, so greedy decoding does not pay this at all — the
// cost lands exactly on the path that also does the ~1 MB readback.
//
// BIT-IDENTITY IS STRUCTURAL, not argued. Every output element is a pure function of the single
// input element at the same index: there is no reduction, no accumulation, and therefore no ordering
// or reassociation freedom for the split to exercise differently. Splitting the range cannot change
// a bit, whatever the worker count or the split points. That is why this stays float64 `math.Tanh`
// rather than becoming a device kernel or a float32 approximation — both would be faster and neither
// would be the same number as the CPU path produces (decoder/forwardn.go, decoder/model.go).
//
// SIBLING SET. Five sites carry this identical loop: decoder/forwardn.go, decoder/model.go,
// cuda/prefill.go, cuda/resident.go and metal/model.go. Both cuda/ callers now share this helper.
// The other three are unchanged and deliberately so — decoder/ is under the 6edd1ca numerics freeze
// and metal/ is on hold — which is recorded in docs/QUEUE.md B6 so the pair is not left implicit.
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
	w := runtime.GOMAXPROCS(0)
	if w > n {
		w = n
	}
	chunk := (n + w - 1) / w
	var wg sync.WaitGroup
	for lo := 0; lo < n; lo += chunk {
		hi := lo + chunk
		if hi > n {
			hi = n
		}
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
