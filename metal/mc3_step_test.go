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

// mc3Resident is the resident the MC3 identity checks run on, with `slots` resident KV slots at a ctx-token resident
// context: the S1 checkpoint under GOINFER_METAL_MC3=1 (mc3RealResident), and otherwise the generated fixture
// (mc3_fixture_test.go), so the checks run by default (E-G01, docs/audit-metal-2026-09-30.md). A timing calls
// mc3RealResident instead: on the fixture it would time nothing real.
func mc3Resident(t *testing.T, slots, ctx int) (*decoder.Model, *resident) {
	t.Helper()
	if os.Getenv("GOINFER_METAL_MC3") == "1" {
		return mc3RealResident(t, slots, ctx)
	}
	m, r := mc3FixtureResident(t, slots, ctx)
	if got := len(r.kvSlotBufs); got < slots || r.batch == nil {
		t.Fatalf("the MC3 fixture built %d KV slots (%d requested) and batched step %v (batchIneligible: %q)", got, slots, r.batch != nil, r.batchIneligible())
	}
	return m, r
}

// mc3LoadCheckpoint loads the checkpoint the MC3 checks run on under GOINFER_METAL_MC3=1 (GOINFER_METAL_MC3_MODEL, else the
// 1.5B in ~/models) with `slots` resident KV slots at a ctx-token resident context, and builds its resident. It skips
// without the variable or the file.
func mc3LoadCheckpoint(t *testing.T, slots, ctx int) (*decoder.Model, *resident) {
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
		m.Close()
		t.Fatalf("build resident: %v", err)
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

// mc3RealResident is mc3LoadCheckpoint for the plain dense W4A8 decode path S1 covers.
func mc3RealResident(t *testing.T, slots, ctx int) (*decoder.Model, *resident) {
	t.Helper()
	m, r := mc3LoadCheckpoint(t, slots, ctx)
	if r.moe != nil || r.g4moe != nil || r.sandwich || r.postOnly || r.parallelBlock || r.kvI8 || r.layerNorm ||
		r.decodeLaneW4F16 || r.nonGatedMLP || r.outBias || r.loraLayers != nil || r.qkNorm || r.learnedPos {
		t.Skip("MC3 S1 covers the plain dense W4A8 decode path only")
	}
	for _, L := range r.layers {
		if L.qGate || L.delta != nil || L.geom == nil || L.geom.kEqV {
			t.Skip("MC3 S1 covers the plain dense W4A8 decode path only")
		}
	}
	if got := len(r.kvSlotBufs); got < slots {
		t.Fatalf("resident allocated %d KV slots, %d requested (the fit clamp)", got, slots)
	}
	if r.batch == nil {
		t.Fatalf("resident built no batched step (batchIneligible: %q)", r.batchIneligible())
	}
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
	out, _, err := r.forwardMulti(seqs)
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

func mc3Identity(t *testing.T, depths []int, ctx int) { mc3IdentityWith(t, depths, ctx, nil) }

// mc3IdentityWith is mc3Identity with setup run on the resident before any step (to force a path).
func mc3IdentityWith(t *testing.T, depths []int, ctx int, setup func(r *resident)) {
	const steps = 12
	B := len(depths)
	_, r := mc3Resident(t, 2*B, ctx)
	if setup != nil {
		setup(r)
	}
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

// TestMC3Step_rowsPathBitIdentical is S4's identity check: with qkv and gate|up forced onto per-row production GEMVs
// at every batch size, and then forced onto the fragment at every size, a batched step at B = 2, 3 and 4 matches
// production's single-token forward on every logit. Which one calibrateRows picks changes speed only.
//
//	GOINFER_METAL_MC3=1 go test -tags goinfer_testhooks -count=1 -run '^TestMC3Step_rowsPathBitIdentical$' -v ./metal/
func TestMC3Step_rowsPathBitIdentical(t *testing.T) {
	for _, c := range []struct {
		name string
		rows int
	}{{"per-row", batchMaxSeqs}, {"fragment", 0}} {
		for _, depths := range [][]int{{5, 300}, {23, 40, 300}, {5, 23, 40, 300}} {
			t.Run(fmt.Sprintf("%s/B=%d", c.name, len(depths)), func(t *testing.T) {
				mc3IdentityWith(t, depths, 1024, func(r *resident) { r.batch.rowsQKV, r.batch.rowsGU = c.rows, c.rows })
			})
		}
	}
}

// TestMC3Step_throughput is S1's in-sequence cost (exploratory): one batched step of B = 1, 2, 4, 8 sequences, each on
// its own slot at depth D, against production's single-token decode at the same depth, GPU time per command buffer,
// arms interleaved per rep with rotating order. Reports aggregate = B x production / step(B).
//
//	GOINFER_METAL_MC3=1 [GOINFER_METAL_MC3_DEPTHS=128,512] go test -count=1 -run '^TestMC3Step_throughput$' -v ./metal/
func TestMC3Step_throughput(t *testing.T) {
	const maxB = 8
	_, r := mc3RealResident(t, maxB+1, 1024)
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

// TestMC3Step_drawsMatchForwardSample: rows that carry a device draw return exactly the id production's ForwardSample
// draws on a twin slot for the same (temperature, seed, draw), in steps that mix drawing rows with logits rows — two
// temperatures, and a draw counter that advances every step, teacher-forced 12 steps.
//
//	GOINFER_METAL_MC3=1 go test -count=1 -run '^TestMC3Step_drawsMatchForwardSample$' -v ./metal/
func TestMC3Step_drawsMatchForwardSample(t *testing.T) {
	const steps = 12
	depths := []int{7, 31, 64, 200}
	temps := []float64{0.8, 0, 1.3, 0} // 0: a logits row
	B := len(depths)
	_, r := mc3Resident(t, 2*B, 1024)
	if !r.SampleAvailable() {
		t.Skip("this resident cannot draw on-device")
	}
	seed := uint32(24681357)
	rnd := func() int { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return int(seed % 20000) }
	for m, D := range depths {
		ids := make([]int, D)
		for i := range ids {
			ids[i] = rnd()
		}
		mc3Fill(t, r, m, ids)
		mc3Fill(t, r, B+m, ids)
	}
	differ, draws := 0, 0
	for st := 0; st < steps; st++ {
		seqs := make([]batchSeq, B)
		wantID := make([]int, B)
		wantLogits := make([][]float32, B)
		for m := range B {
			seqs[m] = batchSeq{slot: m, pos: depths[m] + st, emb: mc3Emb(r, rnd())}
			if err := r.useKVSlot(B + m); err != nil {
				t.Fatal(err)
			}
			if temps[m] > 0 {
				d := &decoder.ResidentBatchDraw{Temperature: temps[m], Seed: uint64(9000 + m), Draw: uint64(st)}
				seqs[m].draw = d
				id, err := r.ForwardSample(seqs[m].emb, seqs[m].pos, d.Temperature, d.Seed, d.Draw)
				if err != nil {
					t.Fatal(err)
				}
				wantID[m] = id
			} else {
				wantLogits[m] = append([]float32(nil), r.ForwardEmb(seqs[m].emb, seqs[m].pos)...)
			}
		}
		logits, ids, err := r.forwardMulti(seqs)
		if err != nil {
			t.Fatal(err)
		}
		for m := range B {
			if temps[m] > 0 {
				draws++
				if ids[m] != wantID[m] || logits[m] != nil {
					t.Errorf("step %d row %d: drew %d (logits %v), ForwardSample drew %d", st, m, ids[m], logits[m] != nil, wantID[m])
					differ++
				}
				continue
			}
			for i := range logits[m] {
				if math.Float32bits(logits[m][i]) != math.Float32bits(wantLogits[m][i]) {
					t.Errorf("step %d row %d: logits differ from ForwardEmb at %d", st, m, i)
					differ++
					break
				}
			}
			if ids[m] != -1 {
				t.Errorf("step %d row %d: a logits row reported id %d", st, m, ids[m])
			}
		}
	}
	t.Logf("%d steps: %d device draws and %d logits rows checked, %d differ", steps, draws, steps*B-draws, differ)
}

// TestMC3StepBreakdown: EXPLORATORY, for MC3 S3 (step polish). Where a batched step's GPU time goes, in sequence: the
// step is timed whole, then with one category's kernels replaced by an empty kernel (so its dispatches still launch
// but do no work), and with every kernel empty (what is left is dispatch overhead alone). Categories: the batched
// matmuls (mc3_bt / mc3_btd / mc3_lm), attention, and the small per-row kernels (norm+quant, RoPE, KV store, ctx
// quant, SwiGLU, the packers). Median of 15 steps per arm, arms interleaved.
//
//	GOINFER_METAL_MC3=1 [GOINFER_METAL_MC3_DEPTHS=128,512] go test -count=1 -run '^TestMC3StepBreakdown$' -v ./metal/
func TestMC3StepBreakdown(t *testing.T) {
	_, r := mc3RealResident(t, 9, 1024)
	b := r.batch
	lib, err := r.d.CompileLibrary("kernel void mc3_noop() {}", MSL3_1)
	if err != nil {
		t.Fatal(err)
	}
	noop, err := r.d.NewComputePipeline(lib, "mc3_noop")
	if err != nil {
		t.Fatal(err)
	}
	cats := map[string][]*Pipeline{
		"matmul":    {&b.bt, &b.btd, &b.lm},
		"attention": {&r.pAttn, &r.pAttnFA, &r.pAttnFACombine, &r.pAttnRows},
		"per-row":   {&r.pRms, &r.pRope2, &r.pKv, &r.pQv, &r.pSw, &b.pack, &b.packLM, &r.pRmsRows, &r.pQvRows, &r.pSwRows, &r.pRope2Rows, &r.pKvRows},
	}
	orig := map[*Pipeline]Pipeline{}
	for _, ps := range cats {
		for _, p := range ps {
			orig[p] = *p
		}
	}
	set := func(names ...string) {
		for p, o := range orig {
			*p = o
		}
		for _, n := range names {
			for _, p := range cats[n] {
				*p = noop
			}
		}
	}
	defer set()
	depths := []int{128}
	if v := os.Getenv("GOINFER_METAL_MC3_DEPTHS"); v != "" {
		depths = nil
		for _, f := range strings.Split(v, ",") {
			var n int
			fmt.Sscan(strings.TrimSpace(f), &n)
			depths = append(depths, n)
		}
	}
	med := func(xs []float64) float64 { v := append([]float64(nil), xs...); sort.Float64s(v); return v[len(v)/2] }
	arms := []struct {
		name string
		off  []string
	}{{"full", nil}, {"-matmul", []string{"matmul"}}, {"-attention", []string{"attention"}}, {"-per-row", []string{"per-row"}},
		{"all empty", []string{"matmul", "attention", "per-row"}}}
	for _, D := range depths {
		for sl := 0; sl < 8; sl++ {
			ids := make([]int, D)
			for i := range ids {
				ids[i] = 1000 + 7*i + sl
			}
			set()
			mc3Fill(t, r, sl, ids)
		}
		for _, B := range []int{2, 4, 8} {
			ms := map[string][]float64{}
			for rep := 0; rep < 15; rep++ {
				for k := range arms {
					a := arms[(k+rep)%len(arms)]
					set(a.off...)
					seqs := make([]batchSeq, B)
					for m := range B {
						seqs[m] = batchSeq{slot: m, pos: D, emb: mc3Emb(r, 300+rep)}
					}
					if _, _, err := r.forwardMulti(seqs); err != nil {
						t.Fatal(err)
					}
					ms[a.name] = append(ms[a.name], (r.gpuEnd-r.gpuStart)*1e3)
				}
			}
			full := med(ms["full"])
			line := fmt.Sprintf("depth %d B=%d: step %.2f ms GPU", D, B, full)
			for _, a := range arms[1:4] {
				line += fmt.Sprintf(" | %s %.2f", a.name[1:], full-med(ms[a.name]))
			}
			line += fmt.Sprintf(" | dispatch overhead (all empty) %.2f", med(ms["all empty"]))
			fmt.Fprintf(os.Stderr, "[mc3-s3] %s\n", line)
		}
	}
	set()
}

// TestMC3StepWallVsGPU: EXPLORATORY, for an encode-ahead batched step. How much of a batched step's wall time is not
// GPU time — encoding its ~28 × 13 dispatches, commit and wait, and the host's logits copies — which is the most an
// encode-ahead executor (pre-encoding step n+1 while the GPU runs step n, as production decode does) could hide. In
// sequence, depth 128, median of 20 steps per arm: the batched step at B = 2, 4, 8, and production's single-token
// decode synchronous (ForwardEmb) and through its encode-ahead executor (ForwardEmbPipe).
//
//	GOINFER_METAL_MC3=1 [GOINFER_METAL_MC3_MODEL=…] go test -tags goinfer_testhooks -count=1 -run '^TestMC3StepWallVsGPU$' -v ./metal/
func TestMC3StepWallVsGPU(t *testing.T) {
	const maxB, D, steps = 8, 128, 20
	_, r := mc3RealResident(t, maxB+1, 1024)
	seed := uint32(97531)
	rnd := func() int { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return int(seed % 20000) }
	for sl := 0; sl <= maxB; sl++ {
		ids := make([]int, D)
		for i := range ids {
			ids[i] = rnd()
		}
		mc3Fill(t, r, sl, ids)
	}
	med := func(xs []float64) float64 { v := append([]float64(nil), xs...); sort.Float64s(v); return v[len(v)/2] }
	report := func(name string, wall, gpu []float64) {
		w, g := med(wall), med(gpu)
		fmt.Fprintf(os.Stderr, "[mc3-wall] %-24s wall %7.3f ms  GPU %7.3f ms  not-GPU %6.3f ms (%.1f%% of wall)\n", name, w, g, w-g, 100*(w-g)/w)
	}
	for _, B := range []int{2, 4, 8} {
		var wall, gpu []float64
		for range steps + 2 {
			seqs := make([]batchSeq, B)
			for m := range B {
				seqs[m] = batchSeq{slot: m, pos: D, emb: mc3Emb(r, rnd())}
			}
			t0 := time.Now()
			mc3Step(t, r, seqs)
			wall = append(wall, time.Since(t0).Seconds()*1e3)
			gpu = append(gpu, (r.gpuEnd-r.gpuStart)*1e3)
		}
		report(fmt.Sprintf("batched step B=%d", B), wall[2:], gpu[2:])
	}
	for _, pipe := range []bool{false, true} {
		if err := r.useKVSlot(maxB); err != nil {
			t.Fatal(err)
		}
		var wall, gpu []float64
		for range steps + 2 {
			emb := mc3Emb(r, rnd())
			t0 := time.Now()
			if pipe {
				r.ForwardEmbPipe(emb, D)
			} else {
				r.ForwardEmb(emb, D)
			}
			wall = append(wall, time.Since(t0).Seconds()*1e3)
			gpu = append(gpu, (r.gpuEnd-r.gpuStart)*1e3)
		}
		r.stopExec()
		name := "production token (sync)"
		if pipe {
			name = "production token (pipe)"
		}
		report(name, wall[2:], gpu[2:])
	}
}

// TestMC3Verify_sameSlotRowsBitIdentical is the first gate of Metal spec verify on the step kernels
// (docs/tasks/task-concurrency-2026-09.md, MC4): a step whose rows are CONSECUTIVE positions of one sequence on one slot
// (a verify round) gives every logit and every K/V element bit-identical to production decoding those tokens one after
// another. Twin slots are filled with the same history; M = 2, 4, 8 rows at depths 128 and 2048, and a run straddling
// attnFADepthFloor, so one step mixes per-head and attention_fa rows.
//
//	GOINFER_METAL_MC3=1 [GOINFER_METAL_MC3_MODEL=…] go test -tags goinfer_testhooks -count=1 -run '^TestMC3Verify_sameSlotRowsBitIdentical$' -v ./metal/
func TestMC3Verify_sameSlotRowsBitIdentical(t *testing.T) {
	_, r := mc3Resident(t, 2, 4096)
	seed := uint32(424242)
	rnd := func() int { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return int(seed % 20000) }
	kv := func(slot, from, n int) [][]uint16 {
		if err := r.useKVSlot(slot); err != nil {
			t.Fatal(err)
		}
		var out [][]uint16
		for l := range r.layers {
			d := r.layers[l].geom.kvDim
			o := r.kvHostOff(l, 2) + from*d
			out = append(out, append([]uint16(nil), r.kc[l].U16s()[o:o+n*d]...), append([]uint16(nil), r.vc[l].U16s()[o:o+n*d]...))
		}
		return out
	}
	cases := []struct{ D, M int }{{128, 2}, {128, 4}, {128, 8}, {2048, 2}, {2048, 4}, {2048, 8}, {attnFADepthFloor - 4, 8}}
	total := 0
	for _, c := range cases {
		ids := make([]int, c.D)
		for i := range ids {
			ids[i] = rnd()
		}
		mc3Fill(t, r, 0, ids)
		mc3Fill(t, r, 1, ids)
		embs := make([][]float32, c.M)
		for i := range embs {
			embs[i] = mc3Emb(r, rnd())
		}
		if err := r.useKVSlot(0); err != nil {
			t.Fatal(err)
		}
		var ref [][]float32
		for i, e := range embs {
			ref = append(ref, append([]float32(nil), r.ForwardEmb(e, c.D+i)...))
		}
		refKV := kv(0, c.D, c.M)
		seqs := make([]batchSeq, c.M)
		for i := range seqs {
			seqs[i] = batchSeq{slot: 1, pos: c.D + i, emb: embs[i]}
		}
		got := mc3Step(t, r, seqs)
		gotKV := kv(1, c.D, c.M)
		lg, kd := 0, 0
		for i := range got {
			for j := range got[i] {
				if math.Float32bits(got[i][j]) != math.Float32bits(ref[i][j]) {
					lg++
				}
			}
		}
		for i := range refKV {
			for j := range refKV[i] {
				if refKV[i][j] != gotKV[i][j] {
					kd++
				}
			}
		}
		fmt.Fprintf(os.Stderr, "[mc3-verify] depth %d, %d rows on one slot: %d of %d logits differ, %d K/V elements differ\n", c.D, c.M, lg, c.M*r.V, kd)
		if lg != 0 || kd != 0 {
			t.Errorf("depth %d, M = %d: the same-slot step differs from sequential decode (%d logits, %d K/V elements)", c.D, c.M, lg, kd)
		}
		total += lg + kd
	}
	t.Logf("%d differing values over %d cases", total, len(cases))
}

// TestMC3Verify_rowCost is the second gate: what each extra verified row costs on the step kernels. GPU time of one
// step of M = 1, 2, 4, 8 consecutive rows on one slot, against a production token, at depths 128, 512 and 2048;
// median of 7, arms interleaved. Extra-row cost = (T(M)/T(token) - 1) / (M - 1).
//
//	GOINFER_METAL_MC3=1 [GOINFER_METAL_MC3_MODEL=…] go test -tags goinfer_testhooks -count=1 -run '^TestMC3Verify_rowCost$' -v ./metal/
func TestMC3Verify_rowCost(t *testing.T) {
	_, r := mc3RealResident(t, 2, 4096) // the step needs slot buffers, which a one-slot resident does not keep
	seed := uint32(13131)
	rnd := func() int { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return int(seed % 20000) }
	med := func(xs []float64) float64 { v := append([]float64(nil), xs...); sort.Float64s(v); return v[len(v)/2] }
	for _, D := range []int{128, 512, 2048} {
		ids := make([]int, D)
		for i := range ids {
			ids[i] = rnd()
		}
		mc3Fill(t, r, 0, ids)
		arms := []int{0, 1, 2, 4, 8} // 0 = production token
		per := map[int][]float64{}
		for rep := range 9 {
			for k := range arms {
				M := arms[(k+rep)%len(arms)]
				if M == 0 {
					r.ForwardEmb(mc3Emb(r, rnd()), D)
				} else {
					seqs := make([]batchSeq, M)
					for i := range seqs {
						seqs[i] = batchSeq{slot: 0, pos: D + i, emb: mc3Emb(r, rnd())}
					}
					mc3Step(t, r, seqs)
				}
				if rep >= 2 {
					per[M] = append(per[M], (r.gpuEnd-r.gpuStart)*1e3)
				}
			}
		}
		tok := med(per[0])
		line := fmt.Sprintf("depth %d: production token %.3f ms", D, tok)
		for _, M := range arms[1:] {
			tm := med(per[M])
			extra := "—"
			if M > 1 {
				extra = fmt.Sprintf("%.3f", (tm/tok-1)/float64(M-1))
			}
			line += fmt.Sprintf("  | M=%d %.3f ms (%.2fx a token, extra row %s)", M, tm, tm/tok, extra)
		}
		fmt.Fprintf(os.Stderr, "[mc3-verify-cost] %s\n", line)
	}
}
