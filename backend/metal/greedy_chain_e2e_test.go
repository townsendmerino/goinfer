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

// TestGreedyChain_generateMatchesFullLogits is C-B01 through the decoder: greedy Generate on Metal with the chain
// (ResidentGreedyChain, the default where it is exact) against the same request with the chain off (greedyChainOff:
// the full-logits ForwardEmbPipe path production ran before). One generation of 90 tokens, past the chain's pool drain;
// then two turns, the second's prompt the first's prompt and reply plus three tokens, so the resident reuses the
// prefix and the forward the chain queued past turn one's last token must not leak into turn two. (A Session is not
// the multi-turn path on a resident: a plain session decodes on the CPU path, session.go.) The chain must have served
// every turn's tokens, or the comparison is of a path with itself.
//
// Tied and untied heads both, and an untied head whose device embedding table does not fit: the chain must then decline
// (decode keeps the full-logits path, the tokens unchanged) rather than allocate past the memory guard.
func TestGreedyChain_generateMatchesFullLogits(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	for _, c := range []struct {
		name         string
		tied, noRoom bool
	}{{"tied", true, false}, {"untied", false, false}, {"untied-no-room", false, true}} {
		t.Run(c.name, func(t *testing.T) { greedyChainGenerate(t, c.tied, c.noRoom, decoder.SamplingParams{}) })
	}
}

// TestSampleChain_generateMatchesForwardSample is C-P02 through the decoder: temperature-only sampling on Metal with the
// sampled chain (ResidentSampleChain) against the same request with it off, where every token is ForwardSample's draw.
// The same generations as the greedy test, at temperature 0.9 with a fixed seed: the tokens must be equal, every token
// device-drawn on both arms, and all of the chain arm's served by the chain.
func TestSampleChain_generateMatchesForwardSample(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	sp := decoder.SamplingParams{Temperature: 0.9, Seed: 11}
	for _, c := range []struct {
		name string
		tied bool
	}{{"tied", true}, {"untied", false}} {
		t.Run(c.name, func(t *testing.T) { greedyChainGenerate(t, c.tied, false, sp) })
	}
}

func greedyChainGenerate(t *testing.T, tied, noRoom bool, sp decoder.SamplingParams) {
	path := writeMC3FixtureTied(t, 4096, tied)
	prompt := make([]int, 24)
	for i := range prompt {
		prompt[i] = (i*613 + 101) % mfVocab
	}
	type out struct {
		one, turn1, turn2 []int
		served, reused    int
		deviceSampled     int
	}
	run := func(off bool) out {
		greedyChainOff = off
		defer func() { greedyChainOff = false }()
		m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 1024, ResidentKVSlots: 1})
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		defer m.Close()
		a, ok := m.ResidentForwardForTest().(*metalResident)
		if !ok {
			t.Fatalf("no metal resident: %s", m.ResidentDecline())
		}
		if noRoom && !off {
			live := metalLiveAvailable
			metalLiveAvailable = func() int64 { return 1 << 20 } // 1 MB free: the table cannot fit
			defer func() { metalLiveAvailable = live }()
		}
		ctx := context.Background()
		var o out
		gen := func(p []int, n int) []int {
			ch, g := m.Generate(ctx, p, n, sp)
			var ids []int
			for id := range ch {
				ids = append(ids, id)
			}
			if err := g.Err(); err != nil {
				t.Fatalf("generate: %v", err)
			}
			o.reused = g.PrefillReused
			o.deviceSampled += g.DeviceSampled
			return ids
		}
		o.one = gen(prompt, 90)
		o.turn1 = gen(prompt, 30)
		ext := append(append(append([]int(nil), prompt...), o.turn1...), 777, 778, 779)
		o.turn2 = gen(ext, 30)
		o.served = a.r.chainServed
		return o
	}
	ref, got := run(true), run(false)
	t.Logf("tied %v, no room %v: chain served %d tokens (off: %d); one %d tokens, turns %d + %d; turn two reused %d of %d prompt tokens (off: %d)",
		tied, noRoom, got.served, ref.served, len(got.one), len(got.turn1), len(got.turn2), got.reused, len(prompt)+30+3, ref.reused)
	if ref.served != 0 {
		t.Fatalf("greedyChainOff still served %d tokens", ref.served)
	}
	if noRoom {
		if got.served != 0 {
			t.Fatalf("the chain served %d tokens with no room for its embedding table: it should have declined", got.served)
		}
	} else if want := 90 + 30 + 30; got.served < want-3 { // each generation's first token comes from its prompt's logits
		t.Fatalf("the chain served %d tokens of the three generations' %d: it did not serve them all", got.served, want)
	}
	if got.reused == 0 || got.reused != ref.reused {
		t.Fatalf("turn two reused %d prompt tokens with the chain, %d without: the test needs the resident's prefix reuse on both", got.reused, ref.reused)
	}
	if sp.Temperature > 0 {
		// Every decode token is a device draw on both arms (the first of each generation is the prompt's, host-drawn).
		if want := 90 + 30 + 30 - 3; got.deviceSampled < want || ref.deviceSampled < want {
			t.Fatalf("device-drawn tokens: %d with the chain, %d without, want >= %d on both: a sampled arm left the device draw",
				got.deviceSampled, ref.deviceSampled, want)
		}
	}
	if !slices.Equal(got.one, ref.one) {
		t.Fatalf("one generation: the chain's tokens %v differ from the full-logits path's %v", got.one, ref.one)
	}
	if !slices.Equal(got.turn1, ref.turn1) || !slices.Equal(got.turn2, ref.turn2) {
		t.Fatalf("two turns: the chain's turns %v / %v differ from the full-logits path's %v / %v", got.turn1, got.turn2, ref.turn1, ref.turn2)
	}
}

// TestHLR_gatedChainMatchesTodaysPath is HL-R2 (docs/measurements/hl-2026-11/registration.md): greedy Generate on Metal under a gated
// processor, with the chain against greedyChainOff (today's path: ForwardArgmax while the gate is closed, Forward while it is open).
// The gate opens twice and, on every third open step, the processor bans the row's argmax, so the host's pick differs from the
// device's. The tokens must be equal, and the processor must run on the same steps. With the chain on, it serves the closed steps (a
// chained forward for every emitted token that is not an open step's) and at least one opening must have stopped it, or the cell is
// void. Cells: the committed fixture, and the 0.5B and 1.5B bench checkpoints from ~/models (the 1.5B as its Metal .giw).
func TestHLR_gatedChainMatchesTodaysPath(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	home, _ := os.UserHomeDir()
	text := func(n int) []int {
		p := make([]int, n)
		for i := range p {
			p[i] = 1000 + (i*37+5)%15000
		}
		return p
	}
	for _, c := range []struct {
		name, path string
		prompt     []int
	}{
		{"fixture", mc3ChainFixture, mc3ChainPrompt(5, 24)},
		{"0.5b", filepath.Join(home, "models", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"), text(48)},
		{"1.5b", filepath.Join(home, "models", "qwen2.5-coder-1.5b-instruct-q4_k_m.int4.metal.v17.giw"), text(48)},
	} {
		t.Run(c.name, func(t *testing.T) { hlrCell(t, c.path, c.prompt) })
	}
}

func hlrCell(t *testing.T, path string, prompt []int) {
	if _, err := os.Stat(path); err != nil {
		t.Skipf("%s: %v", path, err)
	}
	const n = 64
	open := func(gen []int) bool { l := len(gen); return (l >= 5 && l < 12) || (l >= 30 && l < 40) }
	type arm struct {
		ids          []int
		calls, bans  int
		served       int
		chainStopped int
	}
	run := func(off bool) arm {
		greedyChainOff = off
		defer func() { greedyChainOff = false }()
		m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 1024, ResidentKVSlots: 1})
		if err != nil {
			t.Fatalf("load %s: %v", path, err)
		}
		defer m.Close()
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
					logits[argmaxF(logits)] = float32(math.Inf(-1))
					o.bans++
				}
			},
			LogitProcessorGate: open,
		}
		s0 := a.r.chainServed
		ch, g := m.Generate(context.Background(), prompt, n, sp)
		for id := range ch {
			o.ids = append(o.ids, id)
		}
		if err := g.Err(); err != nil {
			t.Fatalf("generate (chain off %v): %v", off, err)
		}
		o.served = a.r.chainServed - s0
		return o
	}
	ref, got := run(true), run(false)
	t.Logf("%s: %d tokens; chain on: %d served, %d processor calls (%d bans); off: %d served, %d calls",
		filepath.Base(path), len(got.ids), got.served, got.calls, got.bans, ref.served, ref.calls)
	if ref.served != 0 {
		t.Fatalf("greedyChainOff still served %d tokens", ref.served)
	}
	if !slices.Equal(got.ids, ref.ids) {
		t.Fatalf("with the chain the tokens are %v, without it %v", got.ids, ref.ids)
	}
	if got.calls != ref.calls || got.bans != ref.bans {
		t.Fatalf("the processor ran %d times (%d bans) with the chain and %d (%d) without", got.calls, got.bans, ref.calls, ref.bans)
	}
	if got.bans == 0 {
		t.Fatalf("no open step banned the argmax: the cell would not show a host pick that differs from the device's")
	}
	// the chain serves the closed steps: every emitted token that is not an open step's, less the one from the prompt's logits
	if want := len(got.ids) - got.calls - 1; got.served < want {
		t.Fatalf("the chain served %d tokens' forwards, want at least %d (%d emitted, %d open steps)", got.served, want, len(got.ids), got.calls)
	}
	if got.served >= len(got.ids) {
		t.Fatalf("the chain served %d of %d emitted tokens: an open step stayed on the chain", got.served, len(got.ids))
	}
}
