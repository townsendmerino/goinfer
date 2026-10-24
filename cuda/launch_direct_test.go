//go:build cuda && goinfer_testhooks

package cuda

import (
	"math"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestLaunchDirect_bitIdentical: the same positions decoded with launches issued from the resident's own thread
// (Queue.LaunchDirect, the default) and through gocudrv's executor (Queue.Launch) give byte-equal logits, on a dense
// fixture and a MoE one, and the direct path allocates no more per token.
func TestLaunchDirect_bitIdentical(t *testing.T) {
	for _, dir := range []string{"../testdata/llama-tiny", "../testdata/qwen3moe-tiny"} {
		t.Run(dir[len("../testdata/"):], func(t *testing.T) {
			requireDeviceAndFixture(t, dir)
			mc, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4"})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer mc.Close()
			r := mustResident(t, mc)
			if r.launchViaExecutor {
				t.Fatal("a fresh resident must launch directly by default")
			}
			_, _, _, _, _, _, vocab := mc.Dims()
			const n = 12
			emb := make([][]float32, n)
			for i := range emb {
				emb[i] = mc.EmbedResidentForTest((i*2654435761 + 7) % (vocab - 1))
			}
			decode := func(viaExecutor bool) [][]float32 {
				r.launchViaExecutor = viaExecutor
				out := make([][]float32, n)
				for i := range n {
					lg, err := r.Forward(emb[i], i)
					if err != nil {
						t.Fatalf("forward (viaExecutor=%v) pos %d: %v", viaExecutor, i, err)
					}
					out[i] = append([]float32(nil), lg...)
				}
				return out
			}
			ex := decode(true)
			di := decode(false)
			for i := range n {
				if len(ex[i]) != len(di[i]) {
					t.Fatalf("pos %d: %d logits via the executor, %d direct", i, len(ex[i]), len(di[i]))
				}
				for j := range ex[i] {
					if math.Float32bits(ex[i][j]) != math.Float32bits(di[i][j]) {
						t.Fatalf("pos %d logit %d: %v via the executor, %v direct", i, j, ex[i][j], di[i][j])
					}
				}
			}

			allocs := func(viaExecutor bool) float64 {
				r.launchViaExecutor = viaExecutor
				return testing.AllocsPerRun(5, func() {
					if _, err := r.ForwardArgmax(emb[0], n); err != nil {
						t.Fatalf("argmax: %v", err)
					}
				})
			}
			aEx, aDi := allocs(true), allocs(false)
			r.launchViaExecutor = false
			t.Logf("%d positions byte-equal; allocations a token: %.0f via the executor, %.0f direct", n, aEx, aDi)
			if aDi > aEx {
				t.Errorf("the direct path allocates more a token (%.0f) than the executor path (%.0f)", aDi, aEx)
			}
		})
	}
}
