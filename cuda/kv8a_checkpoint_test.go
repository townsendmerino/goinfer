//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"math"
	"path/filepath"
	"strings"
	"testing"

	gpu "github.com/townsendmerino/goinfer/cuda/internal/gpudev"
	"github.com/townsendmerino/goinfer/decoder"
)

// KV8a's CUDA gates 1 and 2(a) (docs/measurements/kv8a-2026-10/README.md), on qwen3next-tiny (DeltaNet) and nemotron-tiny (Mamba-2).

var kv8aFixtures = []string{"qwen3next-tiny", "nemotron-tiny"}

func kv8aLoad(t *testing.T, fixture string) (*decoder.Model, *cudaResident) {
	t.Helper()
	if fixture == "nemotron-tiny" {
		t.Cleanup(nemotronFeaturesForTest())
	}
	m, err := decoder.Load(filepath.Join("..", "testdata", fixture), decoder.Options{Backend: "cuda", Quant: "int4", ResidentContext: 4096})
	if err != nil {
		t.Fatalf("load %s: %v", fixture, err)
	}
	t.Cleanup(func() { m.Close() })
	r, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatalf("%s not CUDA-resident: %s", fixture, m.ResidentDecline())
	}
	if r.RecurrentCheckpointBytes() == 0 {
		t.Fatalf("%s kept no recurrent checkpoint: %s", fixture, r.ckptDeclined)
	}
	return m, r
}

func kv8aIDs(seed, n, vocab int) []int {
	o := make([]int, n)
	for i := range o {
		o[i] = (i*131 + seed*977 + (i/7)*13 + 5) % (vocab - 1)
	}
	return o
}

func kv8aEmb(m *decoder.Model, ids []int) [][]float32 {
	rows := make([][]float32, len(ids))
	for i, x := range ids {
		rows[i] = m.EmbedResidentForTest(x)
	}
	return rows
}

// kv8aPrefill prefills ids from start as the decoder does: one batched PrefillLast, or, where the resident declines it (Mamba-2 on
// CUDA runs the sequential path), a forward per token. It returns the last row's logits.
func kv8aPrefill(t *testing.T, m *decoder.Model, r *cudaResident, ids []int, start int) []float32 {
	t.Helper()
	lg, err := r.PrefillLast(context.Background(), kv8aEmb(m, ids), start)
	if err == nil {
		return lg
	}
	if !strings.Contains(err.Error(), "declined") {
		t.Fatal(err)
	}
	for i, x := range ids {
		if lg, err = r.Forward(m.EmbedResidentForTest(x), start+i); err != nil {
			t.Fatal(err)
		}
	}
	return lg
}

// kv8aRec downloads every recurrent buffer the checkpoint covers, in its order (conv window, then state, per layer).
func kv8aRec(t *testing.T, r *cudaResident) [][]float32 {
	t.Helper()
	out := make([][]float32, len(r.ckptSave))
	if err := r.do(func() error {
		for i, c := range r.ckptSave {
			out[i] = make([]float32, c.Src.Len())
			if e := gpu.Download(c.Src, out[i]); e != nil {
				return e
			}
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	return out
}

// kv8aKV reads the first n positions of every attention layer's own K and V rows (f32 layers) in the bound slot, through its table.
func kv8aKV(t *testing.T, r *cudaResident, n int) [][]float32 {
	t.Helper()
	var out [][]float32
	if err := r.do(func() error {
		if e := r.stream.Sync(); e != nil {
			return e
		}
		for l := range r.layers {
			Ly := &r.layers[l]
			if Ly.noKV() || Ly.kvShared || Ly.kvHalf || Ly.kvDim == 0 {
				continue
			}
			k, v, e := r.kvRowsForTest(r.kvSlot, l, 0, n)
			if e != nil {
				return e
			}
			out = append(out, k, v)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
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

// kv8aDecode runs n greedy-free decode steps from pos, feeding fixed ids: they move the live state and write rows past pos.
func kv8aDecode(t *testing.T, m *decoder.Model, r *cudaResident, pos, n, vocab int) {
	t.Helper()
	for i := range n {
		if _, err := r.Forward(m.EmbedResidentForTest((i*29+3)%(vocab-1)), pos+i); err != nil {
			t.Fatalf("decode step at %d: %v", pos+i, err)
		}
	}
}

// Gate 1: the restored state equals the saved state byte for byte, every recurrent layer, conv windows included; a restore that skips
// the conv windows (the plant) must differ.
func TestKV8a_checkpointBitExact(t *testing.T) {
	requireCUDADevice(t)
	for _, fx := range kv8aFixtures {
		t.Run(fx, func(t *testing.T) {
			m, r := kv8aLoad(t, fx)
			_, _, _, _, _, _, vocab := m.Dims()
			const n = 300
			kv8aPrefill(t, m, r, kv8aIDs(1, n, vocab), 0)
			saved := kv8aRec(t, r)
			if err := r.SaveRecurrentCheckpoint(); err != nil {
				t.Fatal(err)
			}
			kv8aDecode(t, m, r, n, 8, vocab)
			if kv8aSame(kv8aRec(t, r), saved) {
				t.Fatal("8 decode steps left the state as saved: the case cannot see a restore")
			}
			if err := r.RestoreRecurrentCheckpoint(); err != nil {
				t.Fatal(err)
			}
			ok := kv8aSame(kv8aRec(t, r), saved)
			t.Logf("%s: checkpoint %d bytes (%.2f MiB) in %d buffers; restored == saved: %v", fx, r.RecurrentCheckpointBytes(), mib(r.RecurrentCheckpointBytes()), len(r.ckptSave), ok)
			if !ok {
				t.Fatal("the restored state differs from the saved one")
			}
			// PLANT: restore only the states (odd entries), not the conv windows.
			kv8aDecode(t, m, r, n, 8, vocab)
			var statesOnly []gpu.DeviceCopy
			for i := 1; i < len(r.ckptRestore); i += 2 {
				statesOnly = append(statesOnly, r.ckptRestore[i])
			}
			if err := r.copyCheckpoint(statesOnly); err != nil {
				t.Fatal(err)
			}
			if kv8aSame(kv8aRec(t, r), saved) {
				t.Error("a restore without the conv windows matched the saved state: the comparison cannot see a window")
			}
		})
	}
}

// Gate 2(a): turn 1 prefilled and saved at c, 8 decode steps, then turn 2 (turn 1's prompt, a reply that differs from the decoded one,
// and a new message) restored at c and prefilled from there equals a cold prefill of turn 2: the last row's logits, every recurrent
// buffer, and the attention K/V rows, bit for bit. Both turns under the fast-prefill floor, and both over it. Plants: a restore
// without the conv windows, and a prefill from c-1 after the restore; both must differ.
func TestKV8a_restoreEqualsColdPrefill(t *testing.T) {
	requireCUDADevice(t)
	for _, fx := range kv8aFixtures {
		for _, n1 := range []int{300, 700} {
			t.Run(fx+"/"+map[int]string{300: "under the fast floor", 700: "over the fast floor"}[n1], func(t *testing.T) {
				mc, rc := kv8aLoad(t, fx)
				_, _, _, _, _, _, vocab := mc.Dims()
				a := kv8aIDs(1, n1, vocab)
				b := append(append(append([]int(nil), a...), kv8aIDs(2, 15, vocab)...), kv8aIDs(3, 45, vocab)...)
				n2 := len(b)
				if (n1 < fastPrefillFloor) != (n2 < fastPrefillFloor) {
					t.Fatalf("turns of %d and %d straddle the fast-prefill floor %d", n1, n2, fastPrefillFloor)
				}
				lc := kv8aPrefill(t, mc, rc, b, 0)
				cold := append([][]float32{append([]float32(nil), lc...)}, append(kv8aRec(t, rc), kv8aKV(t, rc, n2)...)...)

				warm := func(plant string) [][]float32 {
					m, r := kv8aLoad(t, fx)
					kv8aPrefill(t, m, r, a, 0)
					var err error
					if err := r.SaveRecurrentCheckpoint(); err != nil {
						t.Fatal(err)
					}
					kv8aDecode(t, m, r, n1, 8, vocab)
					from := n1
					switch plant {
					case "no conv window":
						var statesOnly []gpu.DeviceCopy
						for i := 1; i < len(r.ckptRestore); i += 2 {
							statesOnly = append(statesOnly, r.ckptRestore[i])
						}
						err = r.copyCheckpoint(statesOnly)
					case "from c-1":
						err, from = r.RestoreRecurrentCheckpoint(), n1-1
					default:
						err = r.RestoreRecurrentCheckpoint()
					}
					if err != nil {
						t.Fatal(err)
					}
					lw := kv8aPrefill(t, m, r, b[from:], from)
					return append([][]float32{append([]float32(nil), lw...)}, append(kv8aRec(t, r), kv8aKV(t, r, n2)...)...)
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

// A load whose free VRAM cannot hold the checkpoint keeps none and says why, and the model still loads resident.
func TestKV8a_noRoomKeepsNone(t *testing.T) {
	requireCUDADevice(t)
	saved := ckptFreeVRAM
	ckptFreeVRAM = func(*cudaResident) (uint64, error) { return 1 << 20, nil }
	defer func() { ckptFreeVRAM = saved }()
	m, err := decoder.Load(filepath.Join("..", "testdata", "qwen3next-tiny"), decoder.Options{Backend: "cuda", Quant: "int4", ResidentContext: 4096})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	r, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatalf("not resident: %s", m.ResidentDecline())
	}
	if r.RecurrentCheckpointBytes() != 0 || r.ckptDeclined == "" || len(r.ckptSave) != 0 {
		t.Fatalf("%d bytes kept, declined %q: want none kept and a reason", r.RecurrentCheckpointBytes(), r.ckptDeclined)
	}
	if err := r.SaveRecurrentCheckpoint(); err == nil {
		t.Error("a save with no checkpoint kept reported success")
	}
}
