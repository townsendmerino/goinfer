//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestHLGenerate_gatedChain is HL-G3 (docs/tasks/task-forward-architecture-2026-10.md §6.1): greedy Generate on Metal
// under a gated processor, with the chain against greedyChainOff (today's path: ForwardArgmax while the gate is closed,
// Forward while it is open). The gate opens twice; on every third open step the processor bans the row's argmax, so the
// chain must redo the forward it queued. The tokens must be equal; with the chain on, every emitted token's forward is
// the chain's and every open step's row is the chain's, and at least one redo ran, or the cell is void. Cells: the
// committed tiny fixture alone and under MC3 (two slots, the generation alone and so held), and the 0.5B and 1.5B
// bench checkpoints from ~/models (the 1.5B as its Metal .giw, which the load-time fit guard admits with less free
// memory than the .gguf needs while it is read).
func TestHLGenerate_gatedChain(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	home, _ := os.UserHomeDir()
	for _, c := range []struct {
		name, path string
		slots      int
		prompt     []int
	}{
		{"fixture", mc3ChainFixture, 1, mc3ChainPrompt(5, 24)},
		{"fixture-mc3", mc3ChainFixture, 2, mc3ChainPrompt(5, 24)},
		{"0.5b", filepath.Join(home, "models", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"), 1, hlTextPrompt(48)},
		{"1.5b", filepath.Join(home, "models", "qwen2.5-coder-1.5b-instruct-q4_k_m.int4.metal.v16.giw"), 1, hlTextPrompt(48)},
	} {
		t.Run(c.name, func(t *testing.T) { hlGenerateCell(t, c.path, c.slots, c.prompt) })
	}
}

func hlTextPrompt(n int) []int {
	p := make([]int, n)
	for i := range p {
		p[i] = 1000 + (i*37+5)%15000
	}
	return p
}

func hlGenerateCell(t *testing.T, path string, slots int, prompt []int) {
	if _, err := os.Stat(path); err != nil {
		t.Skipf("%s: %v", path, err)
	}
	const n = 64
	open := func(gen []int) bool { l := len(gen); return (l >= 5 && l < 12) || (l >= 30 && l < 40) }
	type arm struct {
		ids                 []int
		calls, bans         int
		served, rows, redos int
	}
	run := func(off bool) arm {
		greedyChainOff = off
		defer func() { greedyChainOff = false }()
		m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 1024, ResidentKVSlots: slots})
		if err != nil {
			t.Fatalf("load %s: %v", path, err)
		}
		defer m.Close()
		if slots > 1 {
			if got := m.EnableResidentConcurrency(slots); got != slots {
				t.Fatalf("EnableResidentConcurrency(%d) = %d", slots, got)
			}
		}
		a, ok := m.ResidentForwardForTest().(*metalResident)
		if !ok {
			t.Fatalf("no metal resident: %s", m.ResidentDecline())
		}
		if !off && !a.GreedyChainAvailable() {
			t.Fatalf("%s cannot run the greedy chain (%s): the cell would test nothing", filepath.Base(path), a.r.greedyChainWhyNot())
		}
		var o arm
		sp := decoder.SamplingParams{
			LogitProcessor: func(gen []int, logits []float32) {
				if o.calls++; o.calls%3 == 0 {
					logits[argmaxF32(logits)] = float32(math.Inf(-1))
					o.bans++
				}
			},
			LogitProcessorGate: open,
		}
		s0, r0, d0 := a.r.chainServed, a.r.chainRows, a.r.chainRedos
		ch, g := m.Generate(context.Background(), prompt, n, sp)
		for id := range ch {
			o.ids = append(o.ids, id)
		}
		if err := g.Err(); err != nil {
			t.Fatalf("generate (chain off %v): %v", off, err)
		}
		o.served, o.rows, o.redos = a.r.chainServed-s0, a.r.chainRows-r0, a.r.chainRedos-d0
		return o
	}
	ref, got := run(true), run(false)
	t.Logf("%s, %d slots: %d tokens; chain on: %d served, %d rows, %d redos, %d processor calls (%d bans); off: %d served, %d calls",
		filepath.Base(path), slots, len(got.ids), got.served, got.rows, got.redos, got.calls, got.bans, ref.served, ref.calls)
	if ref.served != 0 {
		t.Fatalf("greedyChainOff still served %d tokens", ref.served)
	}
	if !slices.Equal(got.ids, ref.ids) {
		t.Fatalf("with the chain the tokens are %v, without it %v", got.ids, ref.ids)
	}
	if got.calls != ref.calls || got.bans != ref.bans {
		t.Fatalf("the processor ran %d times (%d bans) with the chain and %d (%d) without", got.calls, got.bans, ref.calls, ref.bans)
	}
	if got.served < len(got.ids) {
		t.Fatalf("the chain served %d of %d emitted tokens' forwards", got.served, len(got.ids))
	}
	if got.rows != got.calls {
		t.Fatalf("the processor ran %d times and the chain returned %d rows: an open step left the chain", got.calls, got.rows)
	}
	if got.redos == 0 || got.redos > got.bans {
		t.Fatalf("%d redos for %d bans: want at least one and no more than the bans (void without one)", got.redos, got.bans)
	}
}
