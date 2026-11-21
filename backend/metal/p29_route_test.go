//go:build darwin

package metal

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime/debug"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestP29_q4kPromptRoutePerSlotCount: the route a q4k-lane prompt takes at one resident KV slot (phi3's load on a 16 GB Mac,
// where the memory guard grants one) and at two. The lane takes the batched f16 pass (prefillOK), and a one-slot resident still
// builds the step (P29), so at either slot count a prompt below the step's bound runs the step, batchMaxSeqs rows at a time, and
// a 64-token prompt, past it, runs the pass. On the 0.5B q4_k_m from ~/models.
func TestP29_q4kPromptRoutePerSlotCount(t *testing.T) {
	home, _ := os.UserHomeDir()
	path := filepath.Join(home, "models", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no model at %s", path)
	}
	const n = 64
	for _, slots := range []int{1, 2} {
		m, err := decoder.Load(path, decoder.Options{Quant: "q4k", ResidentContext: 512, ResidentKVSlots: slots})
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		r, err := buildResident(m)
		if err != nil {
			m.Close()
			t.Fatalf("build resident: %v", err)
		}
		a := &metalResident{r: r, hidden: r.H}
		floor := metalFastPrefillFloorFor(r.knobValue("GOINFER_METAL_FAST_PREFILL_FLOOR"))
		short, long := a.promptStepOK(8, 0, floor), a.promptStepOK(n, 0, floor)
		fmt.Fprintf(os.Stderr, "P29 counter: %d KV slot(s): q4k lane %v, pass %v, step built %v (batchIneligible %q); an 8-token prompt on the step %v, a %d-token one %v\n",
			r.kvSlotCount(), r.q4kLane, r.prefillOK, r.batch != nil, r.batchIneligible(), short, n, long)
		if !r.q4kLane || !r.prefillOK || r.kvSlotCount() != slots || !short || long {
			t.Errorf("%d slot(s): q4k lane %v, pass %v, %d KV slots, step route at 8 tokens %v, at %d %v; want lane on, pass on, %d slots, the step at 8 and the pass at %d",
				slots, r.q4kLane, r.prefillOK, r.kvSlotCount(), short, n, long, slots, n)
		}
		r.Close()
		m.Close()
		debug.FreeOSMemory()
	}
}
