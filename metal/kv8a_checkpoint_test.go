//go:build goinfer_testhooks

package metal

import (
	"context"
	"math"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	gpu "github.com/townsendmerino/goinfer/metal/internal/gpudev"
)

// KV8a's Metal gates M1, M2(a) and M4(a) (docs/measurements/kv8a-metal-2026-10/README.md), on qwen3next-tiny (Gated-DeltaNet) and
// nemotron-tiny (Mamba-2).

var kv8aFixtures = []string{"qwen3next-tiny", "nemotron-tiny"}

func kv8aLoad(t *testing.T, fixture string) (*decoder.Model, *metalResident) {
	t.Helper()
	m, err := decoder.Load(filepath.Join("..", "testdata", fixture), decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 4096})
	if err != nil {
		t.Fatalf("load %s: %v", fixture, err)
	}
	t.Cleanup(func() { m.Close() })
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Fatalf("%s not Metal-resident: %s", fixture, m.ResidentDecline())
	}
	if a.RecurrentCheckpointBytes() == 0 {
		t.Fatalf("%s kept no recurrent checkpoint: %s", fixture, a.r.ckptDeclined)
	}
	return m, a
}

func kv8aIDs(seed, n, vocab int) []int {
	o := make([]int, n)
	for i := range o {
		o[i] = (i*131 + seed*977 + (i/7)*13 + 5) % (vocab - 1)
	}
	return o
}

// kv8aPrefill prefills ids from start as the decoder does: one PrefillLast, or a forward per token where the resident declines it
// (Mamba-2 has no Metal prefill kernel). It returns the last row's logits.
func kv8aPrefill(t *testing.T, m *decoder.Model, a *metalResident, ids []int, start int) []float32 {
	t.Helper()
	rows := make([][]float32, len(ids))
	for i, x := range ids {
		rows[i] = m.EmbedResidentForTest(x)
	}
	if lg, err := a.PrefillLast(context.Background(), rows, start); err == nil {
		return lg
	}
	var lg []float32
	var err error
	for i, r := range rows {
		if lg, err = a.Forward(r, start+i); err != nil {
			t.Fatal(err)
		}
	}
	return lg
}

// kv8aRec reads every recurrent buffer the checkpoint covers, in its order (conv window, then state, per layer), after the work in
// flight.
func kv8aRec(a *metalResident) [][]float32 {
	a.r.stopExec()
	out := make([][]float32, len(a.r.ckptSave))
	for i, c := range a.r.ckptSave {
		out[i] = append([]float32(nil), c.Src.Floats()...)
	}
	return out
}

// kv8aCkpt reads the checkpoint copies themselves.
func kv8aCkpt(a *metalResident) [][]float32 {
	a.r.stopExec()
	out := make([][]float32, len(a.r.ckptSave))
	for i, c := range a.r.ckptSave {
		out[i] = append([]float32(nil), c.Dst.Floats()...)
	}
	return out
}

// kv8aKV reads the first n positions of every attention layer's own K and V rows as bits (f16 or f32 caches).
func kv8aKV(a *metalResident, n int) [][]float32 {
	a.r.stopExec()
	var out [][]float32
	for l := range a.r.layers {
		L := &a.r.layers[l]
		if L.geom == nil || L.kvShared || a.r.kvI8 {
			continue
		}
		for _, b := range []Buffer{a.r.kc[l], a.r.vc[l]} {
			if a.r.kvF32 {
				out = append(out, append([]float32(nil), b.Floats()[:n*L.geom.kvDim]...))
				continue
			}
			h := b.U16s()[:n*L.geom.kvDim]
			v := make([]float32, len(h))
			for i, x := range h {
				v[i] = math.Float32frombits(uint32(x))
			}
			out = append(out, v)
		}
	}
	return out
}

func kv8aSame(a, b [][]float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if len(a[i]) != len(b[i]) {
			return false
		}
		for j := range a[i] {
			if math.Float32bits(a[i][j]) != math.Float32bits(b[i][j]) {
				return false
			}
		}
	}
	return true
}

// kv8aDecode runs n decode steps from pos on fixed ids: they move the live state and write rows past pos.
func kv8aDecode(t *testing.T, m *decoder.Model, a *metalResident, pos, n, vocab int) {
	t.Helper()
	for i := range n {
		if _, err := a.Forward(m.EmbedResidentForTest((i*29+3)%(vocab-1)), pos+i); err != nil {
			t.Fatalf("decode step at %d: %v", pos+i, err)
		}
	}
}

// statesOnly is the restore without the conv windows (odd entries are the states): the plant.
func statesOnly(a *metalResident) []gpu.DeviceCopy {
	var c []gpu.DeviceCopy
	for i := 1; i < len(a.r.ckptRestore); i += 2 {
		c = append(c, a.r.ckptRestore[i])
	}
	return c
}

// M1: the restored state equals the saved state byte for byte, every recurrent layer, conv windows included; a restore that skips
// the conv windows (the plant) must differ.
func TestKV8aMetal_checkpointBitExact(t *testing.T) {
	requireMetal(t)
	for _, fx := range kv8aFixtures {
		t.Run(fx, func(t *testing.T) {
			m, a := kv8aLoad(t, fx)
			_, _, _, _, _, _, vocab := m.Dims()
			const n = 300
			kv8aPrefill(t, m, a, kv8aIDs(1, n, vocab), 0)
			saved := kv8aRec(a)
			if err := a.SaveRecurrentCheckpoint(); err != nil {
				t.Fatal(err)
			}
			kv8aDecode(t, m, a, n, 8, vocab)
			if kv8aSame(kv8aRec(a), saved) {
				t.Fatal("8 decode steps left the state as saved: the case cannot see a restore")
			}
			if err := a.RestoreRecurrentCheckpoint(); err != nil {
				t.Fatal(err)
			}
			ok := kv8aSame(kv8aRec(a), saved)
			t.Logf("%s: checkpoint %d bytes (%.2f MiB) in %d buffers; restored == saved: %v", fx, a.RecurrentCheckpointBytes(), mib(a.RecurrentCheckpointBytes()), len(a.r.ckptSave), ok)
			if !ok {
				t.Fatal("the restored state differs from the saved one")
			}
			kv8aDecode(t, m, a, n, 8, vocab)
			if err := a.r.copyCheckpoint(statesOnly(a)); err != nil {
				t.Fatal(err)
			}
			if kv8aSame(kv8aRec(a), saved) {
				t.Error("a restore without the conv windows matched the saved state: the comparison cannot see a window")
			}
		})
	}
}

// M2(a): turn 1 prefilled and saved at c, 8 decode steps, then turn 2 (turn 1's prompt, a reply that differs from the decoded one,
// and a new message) restored at c and prefilled from there equals a cold prefill of turn 2: the last row's logits, every recurrent
// buffer, and the attention K/V rows, bit for bit. Both turns under Metal's fast-prefill floor, and both over it; under the 16-row
// floor the reply and the message are 4 tokens each. Plants: a restore without the conv windows, and a prefill from c-1 after the
// restore; both must differ.
func TestKV8aMetal_restoreEqualsColdPrefill(t *testing.T) {
	requireMetal(t)
	sizes := []struct {
		name               string
		n1, reply, message int
	}{{"under the fast floor", 6, 4, 4}, {"over the fast floor", 300, 15, 45}}
	for _, fx := range kv8aFixtures {
		for _, sz := range sizes {
			t.Run(fx+"/"+sz.name, func(t *testing.T) {
				mc, rc := kv8aLoad(t, fx)
				_, _, _, _, _, _, vocab := mc.Dims()
				a := kv8aIDs(1, sz.n1, vocab)
				b := append(append(append([]int(nil), a...), kv8aIDs(2, sz.reply, vocab)...), kv8aIDs(3, sz.message, vocab)...)
				n1, n2 := len(a), len(b)
				if (n1 < metalFastPrefillFloor) != (n2 < metalFastPrefillFloor) {
					t.Fatalf("turns of %d and %d straddle the fast-prefill floor %d", n1, n2, metalFastPrefillFloor)
				}
				lc := kv8aPrefill(t, mc, rc, b, 0)
				cold := append([][]float32{append([]float32(nil), lc...)}, append(kv8aRec(rc), kv8aKV(rc, n2)...)...)

				warm := func(plant string) [][]float32 {
					m, r := kv8aLoad(t, fx)
					kv8aPrefill(t, m, r, a, 0)
					if err := r.SaveRecurrentCheckpoint(); err != nil {
						t.Fatal(err)
					}
					kv8aDecode(t, m, r, n1, 8, vocab)
					from := n1
					var err error
					switch plant {
					case "no conv window":
						err = r.r.copyCheckpoint(statesOnly(r))
					case "from c-1":
						err, from = r.RestoreRecurrentCheckpoint(), n1-1
					default:
						err = r.RestoreRecurrentCheckpoint()
					}
					if err != nil {
						t.Fatal(err)
					}
					lw := kv8aPrefill(t, m, r, b[from:], from)
					return append([][]float32{append([]float32(nil), lw...)}, append(kv8aRec(r), kv8aKV(r, n2)...)...)
				}
				for _, plant := range []string{"", "no conv window", "from c-1"} {
					same := kv8aSame(warm(plant), cold)
					t.Logf("%s n1=%d n2=%d plant=%q: restore then suffix bit-identical to a cold prefill: %v", fx, n1, n2, plant, same)
					switch {
					case plant == "" && !same:
						t.Errorf("the restored turn 2 differs from a cold prefill of it")
					case plant != "" && same:
						t.Errorf("plant %q matched a cold prefill: the comparison cannot see it", plant)
					}
				}
			})
		}
	}
}

// M4(a), through the caller: after a generation, the checkpoint holds the state a cold prefill of the prompt leaves, not a state
// decode has moved, so the save ran after the prefill and before the first decode step. Plant: the same comparison against the live
// state after the generation (which decode moved) must differ.
func TestKV8aMetal_saveRunsBeforeDecode(t *testing.T) {
	requireMetal(t)
	for _, fx := range kv8aFixtures {
		t.Run(fx, func(t *testing.T) {
			m, a := kv8aLoad(t, fx)
			_, _, _, _, _, _, vocab := m.Dims()
			prompt := kv8aIDs(4, 120, vocab)
			ch, g := m.Generate(context.Background(), prompt, 8, decoder.SamplingParams{Temperature: 0})
			for range ch {
			}
			if err := g.Err(); err != nil {
				t.Fatal(err)
			}
			if !g.CheckpointSaved {
				t.Fatal("the generation saved no checkpoint")
			}
			ckpt, live := kv8aCkpt(a), kv8aRec(a)
			mc, rc := kv8aLoad(t, fx)
			kv8aPrefill(t, mc, rc, prompt, 0)
			cold := kv8aRec(rc)
			t.Logf("%s: checkpoint == cold prefill state: %v; live state after decode == cold: %v", fx, kv8aSame(ckpt, cold), kv8aSame(live, cold))
			if !kv8aSame(ckpt, cold) {
				t.Error("the checkpoint is not the state a cold prefill of the prompt leaves: the save did not run between prefill and decode")
			}
			if kv8aSame(live, cold) {
				t.Error("the live state after 8 decode steps equals the prefill's: the comparison cannot see a late save")
			}
		})
	}
}

// A load the memory guard cannot also fit the checkpoint into keeps none and says why, and the model still loads resident.
func TestKV8aMetal_noRoomKeepsNone(t *testing.T) {
	requireMetal(t)
	saved := ckptFits
	ckptFits = func(*decoder.Model, int64) (bool, string) { return false, "no room (test)" }
	defer func() { ckptFits = saved }()
	m, err := decoder.Load(filepath.Join("..", "testdata", "qwen3next-tiny"), decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 4096})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Fatalf("not resident: %s", m.ResidentDecline())
	}
	if a.RecurrentCheckpointBytes() != 0 || a.r.ckptDeclined == "" || len(a.r.ckptSave) != 0 {
		t.Fatalf("%d bytes kept, declined %q: want none kept and a reason", a.RecurrentCheckpointBytes(), a.r.ckptDeclined)
	}
	if err := a.SaveRecurrentCheckpoint(); err == nil {
		t.Error("a save with no checkpoint kept reported success")
	}
}
