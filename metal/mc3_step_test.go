//go:build darwin

package metal

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime/debug"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// mc3Resident loads the S1 checkpoint with `slots` resident KV slots at a ctx-token resident context, for the plain dense
// W4A8 decode path S1 covers.
func mc3Resident(t *testing.T, slots, ctx int) (*decoder.Model, *resident) {
	t.Helper()
	if os.Getenv("GOINFER_METAL_MC3") != "1" {
		t.Skip("set GOINFER_METAL_MC3=1 (loads a real checkpoint)")
	}
	home, _ := os.UserHomeDir()
	path := os.Getenv("GOINFER_METAL_MC3_MODEL")
	if path == "" {
		path = filepath.Join(home, "models", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
	if strings.HasPrefix(path, "/Volumes/") || strings.HasPrefix(path, "/srv/models") {
		t.Fatalf("%s is on the archive, not the bench set (CLAUDE.md): a timing from it measures the disk", path)
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s: %v", path, err)
	}
	m, err := decoder.Load(path, decoder.Options{Quant: "int4", ResidentContext: ctx, ResidentKVSlots: slots})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("build resident: %v", err)
	}
	if r.moe != nil || r.g4moe != nil || r.sandwich || r.postOnly || r.parallelBlock || r.kvI8 || r.layerNorm ||
		r.decodeLaneW4F16 || r.nonGatedMLP || r.outBias || r.loraLayers != nil || r.qkNorm || r.learnedPos {
		r.Close()
		t.Skip("MC3 S1 covers the plain dense W4A8 decode path only")
	}
	for _, L := range r.layers {
		if L.qGate || L.delta != nil || L.geom == nil || L.geom.kEqV {
			r.Close()
			t.Skip("MC3 S1 covers the plain dense W4A8 decode path only")
		}
	}
	if got := len(r.kvSlotBufs); got < slots {
		r.Close()
		t.Fatalf("resident allocated %d KV slots, %d requested (the fit clamp)", got, slots)
	}
	if r.batch == nil {
		r.Close()
		t.Fatalf("resident built no batched step (batchIneligible: %q)", r.batchIneligible())
	}
	// Close both and hand the heap back before the next load: the fit guard prices live memory, and a second load in the
	// same process otherwise sees the first model's host weights still held and (correctly) refuses.
	t.Cleanup(func() {
		r.Close()
		m.Close()
		debug.FreeOSMemory()
	})
	return m, r
}

// mc3Emb is token id's (scaled) input embedding.
func mc3Emb(r *resident, id int) []float32 {
	e := make([]float32, r.H)
	r.embed.Row(id, e)
	if r.embedScale > 1 {
		for i := range e {
			e[i] *= r.embedScale
		}
	}
	return e
}

// mc3Fill writes history ids at positions 0.. into KV slot `slot` through production's layer-major batch path, in
// chunks of 64.
func mc3Fill(t *testing.T, r *resident, slot int, ids []int) {
	t.Helper()
	if err := r.useKVSlot(slot); err != nil {
		t.Fatal(err)
	}
	for c := 0; c < len(ids); c += 64 {
		var embs [][]float32
		for _, id := range ids[c:min(c+64, len(ids))] {
			embs = append(embs, mc3Emb(r, id))
		}
		if _, err := r.ForwardBatch(embs, c); err != nil {
			t.Fatalf("fill slot %d: %v", slot, err)
		}
	}
}

// mc3Step runs production's batched step (forwardMulti) and fails the test on an error.
func mc3Step(t *testing.T, r *resident, seqs []batchSeq) [][]float32 {
	t.Helper()
	out, err := r.forwardMulti(seqs)
	if err != nil {
		t.Fatalf("forwardMulti: %v", err)
	}
	return out
}

// TestMC3Step_bitIdentical is S1's identity check: 4 sequences at different depths, each on its own KV slot, stepped
// together 12 times teacher-forced, against production's single-token forward (ForwardEmb) run per sequence on a TWIN
// slot filled with the same history. Every logit must match bit for bit, every step.
//
//	GOINFER_METAL_MC3=1 go test -count=1 -run '^TestMC3Step_bitIdentical$' -v ./metal/
func TestMC3Step_bitIdentical(t *testing.T) { mc3Identity(t, []int{5, 23, 40, 300}, 1024) }

// TestMC3Step_bitIdenticalDeep is the same check where production's single-token step plans attention_fa (at or above
// attnFADepthFloor keys): two sequences past the floor, one crossing it during the 12 steps, one shallow — so the
// batched step must reproduce each sequence's own attention plan, both kinds in one command buffer.
func TestMC3Step_bitIdenticalDeep(t *testing.T) {
	mc3Identity(t, []int{attnFADepthFloor + 400, attnFADepthFloor - 6, 100, attnFADepthFloor + 64}, 2048)
}

func mc3Identity(t *testing.T, depths []int, ctx int) {
	const steps = 12
	B := len(depths)
	_, r := mc3Resident(t, 2*B, ctx)
	seed := uint32(1234567)
	rnd := func() int { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return int(seed % 20000) }
	for m, D := range depths {
		ids := make([]int, D)
		for i := range ids {
			ids[i] = rnd()
		}
		mc3Fill(t, r, m, ids)
		mc3Fill(t, r, B+m, ids)
	}
	totalDiff := 0
	for st := 0; st < steps; st++ {
		seqs := make([]batchSeq, B)
		ref := make([][]float32, B)
		for m := range B {
			seqs[m] = batchSeq{slot: m, pos: depths[m] + st, emb: mc3Emb(r, rnd())}
			if err := r.useKVSlot(B + m); err != nil {
				t.Fatal(err)
			}
			ref[m] = append([]float32(nil), r.ForwardEmb(seqs[m].emb, seqs[m].pos)...)
		}
		got := mc3Step(t, r, seqs)
		for m := range B {
			diff, worst := 0, 0.0
			for i := range got[m] {
				if math.Float32bits(got[m][i]) != math.Float32bits(ref[m][i]) {
					diff++
					worst = math.Max(worst, math.Abs(float64(got[m][i]-ref[m][i])))
				}
			}
			if diff != 0 {
				t.Errorf("step %d sequence %d (pos %d): %d of %d logits differ from production (max |diff| %.3g)", st, m, seqs[m].pos, diff, r.V, worst)
			}
			totalDiff += diff
		}
	}
	t.Logf("%d sequences x %d steps at depths %v: %d logits differ from production's single-token forward", B, steps, depths, totalDiff)
}

// TestMC3Step_throughput is S1's in-sequence cost (exploratory): one batched step of B = 1, 2, 4, 8 sequences, each on
// its own slot at depth D, against production's single-token decode at the same depth, GPU time per command buffer,
// arms interleaved per rep with rotating order. Reports aggregate = B x production / step(B).
//
//	GOINFER_METAL_MC3=1 [GOINFER_METAL_MC3_DEPTHS=128,512] go test -count=1 -run '^TestMC3Step_throughput$' -v ./metal/
func TestMC3Step_throughput(t *testing.T) {
	const maxB = 8
	_, r := mc3Resident(t, maxB+1, 1024)
	depths := []int{128, 512}
	if v := os.Getenv("GOINFER_METAL_MC3_DEPTHS"); v != "" {
		depths = nil
		for _, f := range strings.Split(v, ",") {
			var n int
			fmt.Sscan(strings.TrimSpace(f), &n)
			depths = append(depths, n)
		}
	}
	const reps, tokens = 7, 12
	t0 := time.Now()
	hb := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[mc3-s1 %6.1fs] %s\n", time.Since(t0).Seconds(), fmt.Sprintf(format, a...))
	}
	seed := uint32(7654321)
	rnd := func() int { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return int(seed % 20000) }
	med := func(xs []float64) float64 { v := append([]float64(nil), xs...); sort.Float64s(v); return v[len(v)/2] }
	for _, D := range depths {
		for sl := 0; sl <= maxB; sl++ {
			ids := make([]int, D)
			for i := range ids {
				ids[i] = rnd()
			}
			mc3Fill(t, r, sl, ids)
		}
		hb("depth %d: %d slots filled", D, maxB+1)
		arms := []int{0, 1, 2, 4, 8} // 0 = production single-token decode on slot maxB
		per := map[int][]float64{}   // arm -> per-rep median GPU ms per step
		for rep := 0; rep < reps; rep++ {
			for k := range arms {
				a := arms[(k+rep)%len(arms)]
				var ms []float64
				for tok := 0; tok < tokens; tok++ {
					emb := mc3Emb(r, rnd())
					if a == 0 {
						if err := r.useKVSlot(maxB); err != nil {
							t.Fatal(err)
						}
						r.ForwardEmb(emb, D) // rewrites position D each token: the depth stays D
						ms = append(ms, (r.gpuEnd-r.gpuStart)*1e3)
						continue
					}
					seqs := make([]batchSeq, a)
					for m := range a {
						seqs[m] = batchSeq{slot: m, pos: D, emb: emb}
					}
					mc3Step(t, r, seqs)
					ms = append(ms, (r.gpuEnd-r.gpuStart)*1e3)
				}
				per[a] = append(per[a], med(ms))
			}
		}
		line := fmt.Sprintf("depth %d: production %.3f ms/token", D, med(per[0]))
		for _, a := range arms[1:] {
			var ratio []float64
			for i := range per[a] {
				ratio = append(ratio, float64(a)*per[0][i]/per[a][i])
			}
			sort.Float64s(ratio)
			line += fmt.Sprintf("  | B=%d step %.3f ms (%.2fx a token), aggregate %.3fx [%.3f-%.3f]", a, med(per[a]), med(per[a])/med(per[0]), med(ratio), ratio[0], ratio[len(ratio)-1])
		}
		hb("%s", line)
	}
}
