package decoder

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/linalg"
)

// TestCPUBatchS0_matmulScaling is an EXPLORATORY probe for MC3c step 2's follow-on (a tuned small-M amd64 kernel for
// the batched CPU decode step; docs/tasks/task-concurrency-2026-09.md). It is not a gate. On a real checkpoint's layer-0
// weights it times the production matmul (decoder.matmul, which picks the kernel and layout) at M = 1, 2, 3, 4 and 8
// rows, and reports each against M separate M = 1 calls — the cost the batched step would pay with no amortisation.
// The ratio t(M) / t(1) is what batching buys: near 1 means the weights are read once for all M rows; near M means
// nothing is shared.
//
// GOINFER_CPUBATCH_S0_MODEL=<checkpoint> (int4). Reps interleave the M values, and the median is reported. Progress
// goes to stderr as it runs (run with -v).
func TestCPUBatchS0_matmulScaling(t *testing.T) {
	path := os.Getenv("GOINFER_CPUBATCH_S0_MODEL")
	if path == "" {
		t.Skip("exploratory: set GOINFER_CPUBATCH_S0_MODEL=<checkpoint>")
	}
	m, err := Load(path, Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	lw := &m.w.Layers[0]
	lm := &m.w.LMHead
	if lm.Rows() == 0 {
		lm = &m.w.Embed
	}
	shapes := []struct {
		name string
		w    *linalg.WeightMat
	}{
		{"q", &lw.QProj}, {"k", &lw.KProj}, {"v", &lw.VProj}, {"o", &lw.OProj},
		{"gate", &lw.GateProj}, {"up", &lw.UpProj}, {"down", &lw.DownProj}, {"lm_head", lm},
	}
	Ms := []int{1, 2, 3, 4, 8}
	reps := 7
	rng := rand.New(rand.NewSource(1))
	start := time.Now()
	for si, sh := range shapes {
		K, N := sh.w.Cols(), sh.w.Rows()
		a := make([]float32, 8*K)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
		}
		dst := make([]float32, 8*N)
		times := map[int][]float64{}
		iters := max(3, int(2e9/float64(N*K))) // ~2 G weight-bytes-equivalent per timing
		for r := range reps {
			for i := range Ms {
				M := Ms[(i+r)%len(Ms)]                    // rotate the order each rep
				matmul(m.be, sh.w, a[:M*K], dst[:M*N], M) // warm
				t0 := time.Now()
				for range iters {
					matmul(m.be, sh.w, a[:M*K], dst[:M*N], M)
				}
				times[M] = append(times[M], float64(time.Since(t0).Nanoseconds())/float64(iters))
			}
		}
		med := func(v []float64) float64 { s := append([]float64(nil), v...); sort.Float64s(s); return s[len(s)/2] }
		t1 := med(times[1])
		line := fmt.Sprintf("%-8s N=%-6d K=%-6d layout=%-10s t(1)=%8.0f us", sh.name, N, K, sh.w.Int4Layout(), t1/1e3)
		for _, M := range Ms[1:] {
			line += fmt.Sprintf("  t(%d)/t(1)=%.2f", M, med(times[M])/t1)
		}
		fmt.Fprintf(os.Stderr, "[s0 %6.1fs %d/%d] %s\n", time.Since(start).Seconds(), si+1, len(shapes), line)
		t.Log(line)
	}
}

// TestCPUBatchS1_fusedVsUnfused is MC3c step 2 S1's pre-registered measurement (docs/tasks/task-concurrency-2026-09.md,
// "Step 2 follow-on S1"): the batched CPU step with q‖k‖v and gate‖up fused into one batched W4A8 call each
// (cpuBatchFusedW4A8 on) against the per-projection calls (off), in one process, on the same caches.
//
// 8 caches are prefilled to the depth once. For each rep and each B in {2, 4, 8}, both arms run 16 steps from the same
// rewound caches, in an order that alternates by rep. Aggregate = B·16 / elapsed. The metric is fused ÷ unfused, paired
// per rep, median of 5. Every step's logits must be bit-identical between the arms, or it fails: the fused kernel's
// contract, checked here on the real model rather than assumed.
//
// GOINFER_CPUBATCH_S1_MODEL=<checkpoint> (int4), GOINFER_CPUBATCH_S1_DEPTH (default 128). Progress goes to stderr
// per rep (run with -v).
func TestCPUBatchS1_fusedVsUnfused(t *testing.T) {
	path := os.Getenv("GOINFER_CPUBATCH_S1_MODEL")
	if path == "" {
		t.Skip("measurement: set GOINFER_CPUBATCH_S1_MODEL=<checkpoint>")
	}
	depth := 128
	if v := os.Getenv("GOINFER_CPUBATCH_S1_DEPTH"); v != "" {
		fmt.Sscan(v, &depth)
	}
	const steps, reps, maxB = 16, 5, 8
	Bs := []int{2, 4, 8}
	m, err := Load(path, Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	if err := m.cpuBatchModelEligible(); err != nil {
		t.Skipf("%v", err)
	}
	orig := cpuBatchFusedW4A8
	t.Cleanup(func() { cpuBatchFusedW4A8 = orig })
	vocab := m.w.arch.VocabSize
	start := time.Now()
	caches := make([]*KVCache, maxB)
	for b := range maxB {
		prompt := make([]int, depth)
		for i := range prompt {
			prompt[i] = (i*37 + b*11 + 3) % vocab
		}
		caches[b] = m.NewCache(depth + steps + 8)
		if _, err := m.prefillLogits(context.Background(), prompt, caches[b]); err != nil {
			t.Fatalf("prefill: %v", err)
		}
	}
	fmt.Fprintf(os.Stderr, "[s1 %6.1fs] %s depth %d: %d caches prefilled\n", time.Since(start).Seconds(), filepath.Base(path), depth, maxB)
	// run steps B-wide from depth with fusion set to fused; returns tok/s and every step's logits.
	run := func(B int, fused bool) (float64, [][][]float32) {
		cpuBatchFusedW4A8 = fused
		ids := make([]int, B)
		for b := range B {
			caches[b].TruncateTo(depth)
			ids[b] = (b*53 + 7) % vocab
		}
		var outs [][][]float32
		t0 := time.Now()
		for range steps {
			lg, err := m.decodeMultiStep(ids, caches[:B])
			if err != nil {
				t.Fatalf("B=%d fused=%v: %v", B, fused, err)
			}
			for b := range B {
				ids[b] = argmaxF32(lg[b])
			}
			outs = append(outs, lg)
		}
		return float64(B*steps) / time.Since(t0).Seconds(), outs
	}
	ratios := map[int][]float64{}
	for r := range reps {
		line := fmt.Sprintf("rep %d/%d:", r+1, reps)
		for _, B := range Bs {
			var f, u float64
			var fo, uo [][][]float32
			if r%2 == 0 {
				f, fo = run(B, true)
				u, uo = run(B, false)
			} else {
				u, uo = run(B, false)
				f, fo = run(B, true)
			}
			for s := range fo {
				for b := range fo[s] {
					for j := range fo[s][b] {
						if math.Float32bits(fo[s][b][j]) != math.Float32bits(uo[s][b][j]) {
							t.Fatalf("B=%d step %d seq %d logit %d: fused %v, unfused %v — not bit-identical", B, s, b, j, fo[s][b][j], uo[s][b][j])
						}
					}
				}
			}
			ratios[B] = append(ratios[B], f/u)
			line += fmt.Sprintf("  B=%d fused %.2f unfused %.2f tok/s = %.3fx", B, f, u, f/u)
		}
		fmt.Fprintf(os.Stderr, "[s1 %6.1fs] %s\n", time.Since(start).Seconds(), line)
	}
	for _, B := range Bs {
		s := append([]float64(nil), ratios[B]...)
		sort.Float64s(s)
		msg := fmt.Sprintf("S1 METRIC depth %d B=%d fused/unfused median %.3fx (per-rep sorted %.3f)", depth, B, s[len(s)/2], s)
		fmt.Fprintf(os.Stderr, "[s1 %6.1fs] %s\n", time.Since(start).Seconds(), msg)
		t.Log(msg)
	}
}

// TestCPUBatchS0b_fusedShapesAndWidth is an EXPLORATORY probe (not a gate) for what is left after MC3c step 2 S1, on a
// real checkpoint's layer 0:
//   - E1: the shapes the batched step now issues — q‖k‖v and gate‖up as one linalg.MatmulBTW4A8Batch each, o and down
//     alone — timed at M = 1, 2, 4, 8 against M = 1 (t(M)/t(1): 1 = weights read once for every row);
//   - E2: the same shapes at M = 4 (and M = 1) across fan-out widths and serial. Width is numerically inert
//     (parallel matmuls partition output columns), so a width that wins is a free lever for the step's Workspace.
//
// GOINFER_CPUBATCH_S0_MODEL=<checkpoint> (int4). Medians of 7 reps with rotated order; progress to stderr.
func TestCPUBatchS0b_fusedShapesAndWidth(t *testing.T) {
	path := os.Getenv("GOINFER_CPUBATCH_S0_MODEL")
	if path == "" {
		t.Skip("exploratory: set GOINFER_CPUBATCH_S0_MODEL=<checkpoint>")
	}
	m, err := Load(path, Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	lw := &m.w.Layers[0]
	type shape struct {
		name string
		ws   []*linalg.WeightMat
	}
	shapes := []shape{
		{"qkv", []*linalg.WeightMat{&lw.QProj, &lw.KProj, &lw.VProj}},
		{"o", []*linalg.WeightMat{&lw.OProj}},
		{"gate‖up", []*linalg.WeightMat{&lw.GateProj, &lw.UpProj}},
		{"down", []*linalg.WeightMat{&lw.DownProj}},
	}
	rng := rand.New(rand.NewSource(1))
	start := time.Now()
	med := func(v []float64) float64 { s := append([]float64(nil), v...); sort.Float64s(s); return s[len(s)/2] }
	// timeOp runs one fused call of shape sh at M rows with the given width (0 = default) or serial, and returns ns/op.
	timeOp := func(sh shape, M, width int, serial bool, a []float32, dsts [][]float32, iters int) float64 {
		var ws linalg.Workspace
		ws.SetThreshold(int4ParThreshold)
		if serial {
			ws.SetThreshold(1 << 62)
		}
		ws.SetWorkers(width)
		ops := make([]linalg.W4A8Op, len(sh.ws))
		dd := make([][]float32, len(sh.ws))
		for i, w := range sh.ws {
			dd[i] = dsts[i][:M*w.Rows()]
		}
		group, ag, ok := w4a8FusedOps(ops, sh.ws, dd)
		if !ok {
			t.Fatalf("%s: not fusable", sh.name)
		}
		K := sh.ws[0].Cols()
		matmulW4A8Batch(m.be, &ws, a[:M*K], M, K, group, ops, ag) // warm
		t0 := time.Now()
		for range iters {
			matmulW4A8Batch(m.be, &ws, a[:M*K], M, K, group, ops, ag)
		}
		return float64(time.Since(t0).Nanoseconds()) / float64(iters)
	}
	const reps = 7
	for si, sh := range shapes {
		K := sh.ws[0].Cols()
		totalN := 0
		dsts := make([][]float32, len(sh.ws))
		for i, w := range sh.ws {
			totalN += w.Rows()
			dsts[i] = make([]float32, 8*w.Rows())
		}
		a := make([]float32, 8*K)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
		}
		iters := max(3, int(2e9/float64(totalN*K)))
		// E1: M scaling at the default width.
		Ms := []int{1, 2, 4, 8}
		e1 := map[int][]float64{}
		for r := range reps {
			for i := range Ms {
				M := Ms[(i+r)%len(Ms)]
				e1[M] = append(e1[M], timeOp(sh, M, 0, false, a, dsts, iters))
			}
		}
		t1 := med(e1[1])
		line := fmt.Sprintf("E1 %-8s N=%-6d K=%-6d t(1)=%7.0f us", sh.name, totalN, K, t1/1e3)
		for _, M := range Ms[1:] {
			line += fmt.Sprintf("  t(%d)/t(1)=%.2f", M, med(e1[M])/t1)
		}
		fmt.Fprintf(os.Stderr, "[s0b %6.1fs %d/%d] %s\n", time.Since(start).Seconds(), si+1, len(shapes), line)
		t.Log(line)
		// E2: width sweep at M = 4 and M = 1 (0 = default width; -1 = serial).
		widths := []int{0, 4, 6, 8, 12, 16, -1}
		for _, M := range []int{4, 1} {
			e2 := map[int][]float64{}
			for r := range reps {
				for i := range widths {
					w := widths[(i+r)%len(widths)]
					e2[w] = append(e2[w], timeOp(sh, M, max(w, 0), w < 0, a, dsts, iters))
				}
			}
			base := med(e2[0])
			line := fmt.Sprintf("E2 %-8s M=%d default %7.0f us:", sh.name, M, base/1e3)
			for _, w := range widths[1:] {
				name := fmt.Sprintf("w%d", w)
				if w < 0 {
					name = "serial"
				}
				line += fmt.Sprintf("  %s %.3fx", name, base/med(e2[w]))
			}
			fmt.Fprintf(os.Stderr, "[s0b %6.1fs %d/%d] %s   (x = speed-up over default)\n", time.Since(start).Seconds(), si+1, len(shapes), line)
			t.Log(line)
		}
	}
}

// TestCPUBatchS0c_widthAA re-runs S0b's width sweep with an A/A control, because S0b's E2 was confounded: its
// "default" and "w16" arms are the SAME configuration (the default width is GOMAXPROCS = 16 on nobara) and read 1.41x
// apart, so position in the rotation (after a long serial arm the workers are parked) mattered more than width. Here
// there is no serial arm, the default is measured twice (A and A'), every timed block is preceded by an untimed warm
// block, and blocks are longer. A width result counts only if A/A' agree within a few percent.
func TestCPUBatchS0c_widthAA(t *testing.T) {
	path := os.Getenv("GOINFER_CPUBATCH_S0_MODEL")
	if path == "" {
		t.Skip("exploratory: set GOINFER_CPUBATCH_S0_MODEL=<checkpoint>")
	}
	m, err := Load(path, Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	lw := &m.w.Layers[0]
	shapes := []struct {
		name string
		ws   []*linalg.WeightMat
	}{
		{"qkv", []*linalg.WeightMat{&lw.QProj, &lw.KProj, &lw.VProj}},
		{"o", []*linalg.WeightMat{&lw.OProj}},
		{"down", []*linalg.WeightMat{&lw.DownProj}},
	}
	type arm struct {
		name  string
		width int
	}
	arms := []arm{{"A", 0}, {"A'", 0}, {"w8", 8}, {"w12", 12}}
	rng := rand.New(rand.NewSource(1))
	start := time.Now()
	med := func(v []float64) float64 { s := append([]float64(nil), v...); sort.Float64s(s); return s[len(s)/2] }
	const reps = 9
	for _, M := range []int{4, 1} {
		for si, sh := range shapes {
			K := sh.ws[0].Cols()
			totalN := 0
			dd := make([][]float32, len(sh.ws))
			for i, w := range sh.ws {
				totalN += w.Rows()
				dd[i] = make([]float32, M*w.Rows())
			}
			a := make([]float32, M*K)
			for i := range a {
				a[i] = float32(rng.NormFloat64())
			}
			ops := make([]linalg.W4A8Op, len(sh.ws))
			group, ag, ok := w4a8FusedOps(ops, sh.ws, dd)
			if !ok {
				t.Fatalf("%s: not fusable", sh.name)
			}
			iters := max(20, int(8e9/float64(totalN*K)))
			block := func(width int) float64 {
				var ws linalg.Workspace
				ws.SetThreshold(int4ParThreshold)
				ws.SetWorkers(width)
				for range iters / 4 { // untimed warm block: wake the workers, settle the caches
					matmulW4A8Batch(m.be, &ws, a, M, K, group, ops, ag)
				}
				t0 := time.Now()
				for range iters {
					matmulW4A8Batch(m.be, &ws, a, M, K, group, ops, ag)
				}
				return float64(time.Since(t0).Nanoseconds()) / float64(iters)
			}
			times := map[string][]float64{}
			for r := range reps {
				for i := range arms {
					ar := arms[(i+r)%len(arms)]
					times[ar.name] = append(times[ar.name], block(ar.width))
				}
			}
			base := med(times["A"])
			line := fmt.Sprintf("M=%d %-5s A %6.0f us:", M, sh.name, base/1e3)
			for _, ar := range arms[1:] {
				line += fmt.Sprintf("  %s %.3fx", ar.name, base/med(times[ar.name]))
			}
			fmt.Fprintf(os.Stderr, "[s0c %6.1fs %d/%d] %s   (x = speed-up over A; A' is the control)\n", time.Since(start).Seconds(), si+1, len(shapes), line)
			t.Log(line)
		}
	}
}
