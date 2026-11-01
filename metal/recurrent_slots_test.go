//go:build darwin

package metal

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// deltaNetSlotFixtures are the tracked Gated-DeltaNet checkpoints the per-slot state is checked on (OP4m-d4a,
// docs/tasks/task-forward-architecture-2026-10.md §4.9).
var deltaNetSlotFixtures = []string{"../testdata/qwen35-tiny", "../testdata/qwen3next-tiny"}

// recurrentSlotsLoad loads dir at int4 with opts' slot request.
func recurrentSlotsLoad(t *testing.T, dir string, opts decoder.Options) *decoder.Model {
	t.Helper()
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	opts.Quant, opts.ResidentContext = "int4", 256
	m, err := decoder.Load(dir, opts)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	return m
}

// recurrentSlotsBuild builds m's Metal resident.
func recurrentSlotsBuild(t *testing.T, m *decoder.Model) *metalResident {
	t.Helper()
	rf, ok, err := (&metalBackend{}).BuildResident(m)
	if err != nil || !ok {
		t.Fatalf("build: %v", err)
	}
	a := rf.(*metalResident)
	t.Cleanup(func() { a.r.Close() })
	return a
}

// recurrentSlotsResident builds dir's resident with `slots` KV slots asked for explicitly (not the caller's default).
func recurrentSlotsResident(t *testing.T, dir string, slots int) *metalResident {
	t.Helper()
	return recurrentSlotsBuild(t, recurrentSlotsLoad(t, dir, decoder.Options{ResidentKVSlots: slots}))
}

// recurrentSeqs is two sequences of n embeddings each, from fixed token ids.
func recurrentSeqs(r *resident, n int) (a, b [][]float32) {
	seed := uint32(86420)
	rnd := func() int {
		seed ^= seed << 13
		seed ^= seed >> 17
		seed ^= seed << 5
		return int(seed % uint32(min(20000, r.V)))
	}
	for range n {
		a, b = append(a, mc3Emb(r, rnd())), append(b, mc3Emb(r, rnd()))
	}
	return a, b
}

// recurrentSolo decodes embs as one fresh sequence on the bound slot and returns each position's logits.
func recurrentSolo(a *metalResident, embs [][]float32) [][]float32 {
	a.Reset()
	out := make([][]float32, len(embs))
	for i, e := range embs {
		out[i] = append([]float32(nil), a.r.ForwardEmb(e, i)...)
	}
	return out
}

// recurrentInterleaved decodes A on slot 0 and B on slot 1 one token each in turn, and counts the logits that differ from
// refA and refB bit for bit.
func recurrentInterleaved(t *testing.T, a *metalResident, A, B, refA, refB [][]float32) int {
	t.Helper()
	bind := func(s int) {
		if err := a.r.useKVSlot(s); err != nil {
			t.Fatal(err)
		}
	}
	bind(0)
	a.Reset()
	bind(1)
	a.Reset()
	differ := 0
	count := func(got, want []float32) {
		for j := range want {
			if j >= len(got) || math.Float32bits(got[j]) != math.Float32bits(want[j]) {
				differ++
			}
		}
	}
	for i := range A {
		bind(0)
		count(a.r.ForwardEmb(A[i], i), refA[i])
		bind(1)
		count(a.r.ForwardEmb(B[i], i), refB[i])
	}
	return differ
}

// TestRecurrentSlots_deltaNetSequencesKeepTheirOwnState is OP4m-d4a's D4a-1 and D4a-2: on a Gated-DeltaNet model that asked
// for two KV slots, the resident has two, and two sequences decoded one token each in turn on slots 0 and 1 each equal the
// same sequence decoded alone on a one-slot resident, every logit of every position.
func TestRecurrentSlots_deltaNetSequencesKeepTheirOwnState(t *testing.T) {
	for _, dir := range deltaNetSlotFixtures {
		t.Run(filepath.Base(dir), func(t *testing.T) {
			two := recurrentSlotsResident(t, dir, 2)
			if n := two.r.kvSlotCount(); n != 2 {
				t.Fatalf("%d KV slots on a Gated-DeltaNet model that asked for 2", n)
			}
			one := recurrentSlotsResident(t, dir, 1)
			A, B := recurrentSeqs(one.r, 12)
			refA, refB := recurrentSolo(one, A), recurrentSolo(one, B)
			if d := recurrentInterleaved(t, two, A, B, refA, refB); d != 0 {
				t.Errorf("interleaved on two slots: %d logits differ from each sequence's solo run", d)
			}
		})
	}
}
