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

// TestP29_q4kPromptRoutePerSlotCount: the route a q4k-lane prompt takes, and the rows each step carries, at one resident KV
// slot (phi3's load on a 16 GB Mac, where the memory guard grants one) and at two. The q4k lane never runs the batched pass
// (prefillOK), so the batched step is its only multi-row prefill. A one-slot resident builds the step for its own prompt
// (P29), so at either slot count a prompt runs the step, batchMaxSeqs rows per step. On the 0.5B q4_k_m from ~/models.
func TestP29_q4kPromptRoutePerSlotCount(t *testing.T) {
	home, _ := os.UserHomeDir()
	path := filepath.Join(home, "models", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no model at %s", path)
	}
	const n = 64
	for _, c := range []struct {
		slots    int
		wantStep bool
	}{{1, true}, {2, true}} {
		m, err := decoder.Load(path, decoder.Options{Quant: "q4k", ResidentContext: 512, ResidentKVSlots: c.slots})
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		r, err := buildResident(m)
		if err != nil {
			m.Close()
			t.Fatalf("build resident: %v", err)
		}
		a := &metalResident{r: r, hidden: r.H}
		step := a.promptStepOK(n, 0, metalFastPrefillFloorFor(r.knobValue("GOINFER_METAL_FAST_PREFILL_FLOOR")))
		rows, route := 1, "the sequential loop (decoder's per-token Forward)"
		if step {
			rows, route = batchMaxSeqs, "the batched step (prefillByStep)"
		}
		fmt.Fprintf(os.Stderr, "P29 counter: %d KV slot(s): q4k lane %v, pass %v, step built %v (batchIneligible %q); a %d-token prompt runs %s, %d row(s) per step\n",
			r.kvSlotCount(), r.q4kLane, r.prefillOK, r.batch != nil, r.batchIneligible(), n, route, rows)
		if !r.q4kLane || r.prefillOK || r.kvSlotCount() != c.slots || step != c.wantStep {
			t.Errorf("%d slot(s): q4k lane %v, pass %v, %d KV slots, step route %v; want lane on, no pass, %d slots, step %v",
				c.slots, r.q4kLane, r.prefillOK, r.kvSlotCount(), step, c.slots, c.wantStep)
		}
		r.Close()
		m.Close()
		debug.FreeOSMemory()
	}
}
