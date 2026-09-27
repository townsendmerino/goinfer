//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"math/rand"
	"os"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestMetalKVSlots_interleavedMatchesAlone is MC1's gate (docs/tasks/task-concurrency-2026-09.md): with two resident KV
// slots, two conversations that share only a lead, interleaved turn by turn through the production Generate path,
// must each emit exactly the token ids they emit when served alone, and reuse exactly as much of their own history
// on every turn. A control — the same interleaving on one slot — must thrash, so this test can see the failure it
// guards against.
//
// GOINFER_METAL_KVSLOTS_MODEL=<checkpoint> runs the same scenario on a real checkpoint (int4, 48 tokens per turn)
// instead of the synthetic tiny model; its prompts are ids from a fixed range, not text.
func TestMetalKVSlots_interleavedMatchesAlone(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	dir, quant, maxTok, base := os.Getenv("GOINFER_METAL_KVSLOTS_MODEL"), "int4", 48, 1000
	if dir == "" {
		w := genTinyWeights(rand.New(rand.NewSource(21)))
		dir = t.TempDir()
		writeDense(t, dir, w)
		quant, maxTok, base = "int8int8", 12, 0
	}

	lead := []int{base + 1, base + 2, base + 3} // the shared "chat template" lead
	first := map[string][]int{
		"A": append(slices.Clone(lead), base+10, base+11, base+12, base+13),
		"B": append(slices.Clone(lead), base+20, base+21, base+22, base+23),
	}
	type turn = kvSlotTurn
	run := func(m *decoder.Model, order []string) map[string][]turn {
		prompts := map[string][]int{"A": slices.Clone(first["A"]), "B": slices.Clone(first["B"])}
		out := map[string][]turn{}
		for _, c := range order {
			ch, gen := m.Generate(context.Background(), prompts[c], maxTok, decoder.SamplingParams{})
			var ids []int
			for id := range ch {
				ids = append(ids, id)
			}
			if err := gen.Err(); err != nil {
				t.Fatalf("%s turn %d: %v", c, len(out[c])+1, err)
			}
			out[c] = append(out[c], turn{ids, gen.PrefillReused})
			// the next user turn extends the conversation with its reply and two new ids
			prompts[c] = append(append(slices.Clone(prompts[c]), ids...), base+30+len(out[c]), base+40+len(out[c]))
		}
		return out
	}
	load := func(slots int) *decoder.Model {
		m, err := decoder.Load(dir, decoder.Options{Backend: "metal", Quant: quant, ResidentKVSlots: slots})
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		if got := m.ResidentKVSlots(); got != slots {
			m.Close()
			t.Fatalf("ResidentKVSlots() = %d with %d requested — the resident did not allocate them", got, slots)
		}
		return m
	}
	interleaved := []string{"A", "B", "A", "B", "A", "B"}

	mTwo := load(2)
	inter := run(mTwo, interleaved)
	mTwo.Close()
	alone := map[string][]turn{}
	for _, c := range []string{"A", "B"} { // each conversation alone on a fresh one-slot model
		m := load(1)
		alone[c] = run(m, []string{c, c, c})[c]
		m.Close()
	}
	mOne := load(1)
	thrash := run(mOne, interleaved) // the control
	mOne.Close()

	for _, c := range []string{"A", "B"} {
		for i := range alone[c] {
			a, b := alone[c][i], inter[c][i]
			if !slices.Equal(a.ids, b.ids) {
				t.Errorf("%s turn %d: interleaved on 2 slots emitted %v, alone %v — not bit-identical", c, i+1, b.ids, a.ids)
			}
			if a.reused != b.reused {
				t.Errorf("%s turn %d: interleaved on 2 slots reused %d, alone %d", c, i+1, b.reused, a.reused)
			}
			if i > 0 && b.reused <= len(lead) {
				t.Errorf("%s turn %d: reused only %d (the lead is %d) — its own history was not kept", c, i+1, b.reused, len(lead))
			}
			if i > 0 && thrash[c][i].reused > len(lead) {
				t.Errorf("control: %s turn %d reused %d on ONE slot interleaved — the control no longer thrashes, so "+
					"this test cannot see a slot regression", c, i+1, thrash[c][i].reused)
			}
		}
		t.Logf("%s: reused per turn alone %v, interleaved on 2 slots %v, interleaved on 1 slot %v", c,
			reusedOf(alone[c]), reusedOf(inter[c]), reusedOf(thrash[c]))
	}
}

// kvSlotTurn is one generation of TestMetalKVSlots_interleavedMatchesAlone: its token ids and PrefillReused.
type kvSlotTurn struct {
	ids    []int
	reused int
}

func reusedOf(ts []kvSlotTurn) []int {
	out := make([]int, len(ts))
	for i, t := range ts {
		out[i] = t.reused
	}
	return out
}

// TestKVSlotsWithin pins MC1's fit-guard clamp (docs/tasks/task-concurrency-2026-09.md MC1 gate 4) without allocating a
// real clamp's worth of KV on a 16 GB machine: the largest count up to the request that fits the budget, never below 1.
func TestKVSlotsWithin(t *testing.T) {
	const mb = int64(1 << 20)
	for _, tc := range []struct {
		name               string
		want               int
		budget, base, slot int64
		got                int
	}{
		{"all fit", 4, 10000 * mb, 2000 * mb, 117 * mb, 4},
		{"clamped to what fits", 8, 2500 * mb, 2000 * mb, 117 * mb, 5}, // 2000 + 4*117 = 2468 <= 2500; +5*117 > 2500
		{"only the first fits", 4, 2050 * mb, 2000 * mb, 117 * mb, 1},
		{"even the first over budget: still 1, the decline is elsewhere", 4, 1000 * mb, 2000 * mb, 117 * mb, 1},
		{"request of 0 means 1", 0, 10000 * mb, 2000 * mb, 117 * mb, 1},
	} {
		if got := kvSlotsWithin(tc.want, tc.budget, tc.base, tc.slot); got != tc.got {
			t.Errorf("%s: kvSlotsWithin(%d, ...) = %d, want %d", tc.name, tc.want, got, tc.got)
		}
	}
}
