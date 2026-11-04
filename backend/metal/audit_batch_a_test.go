//go:build darwin

package metal

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime/debug"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// Phase 2, Batch A of docs/tasks/task-metal-audit-2026-10.md: the timed probes of docs/audit-metal-2026-09-30.md §10
// that run at night. Each prints a "[t1.x] RESULT" line carrying the reading its pre-registration names; the decision
// rules are in the task doc, committed before the run. GOINFER_METAL_AUDIT_A=1 enables them (they are timings, so never
// by day except as a labelled smoke run), GOINFER_AUDIT_MODEL picks the checkpoint, and GOINFER_AUDIT_REPS overrides
// the repetitions (a smoke run sets 1).

func auditA(t *testing.T) {
	t.Helper()
	if os.Getenv("GOINFER_METAL_AUDIT_A") != "1" {
		t.Skip("set GOINFER_METAL_AUDIT_A=1: a timed Batch A probe (night-only)")
	}
}

func auditReps(def int) int {
	if n, err := strconv.Atoi(os.Getenv("GOINFER_AUDIT_REPS")); err == nil && n > 0 {
		return n
	}
	return def
}

// auditLoad loads GOINFER_AUDIT_MODEL (else ~/models/def) at int4 with `slots` KV slots at a ctx-token resident
// context and builds its resident; both close when the test ends.
func auditLoad(t *testing.T, def string, slots, ctx int, knobs *decoder.Knobs) (string, *metalResident) {
	t.Helper()
	path := os.Getenv("GOINFER_AUDIT_MODEL")
	if path == "" {
		home, _ := os.UserHomeDir()
		path = filepath.Join(home, "models", def)
	}
	if strings.HasPrefix(path, "/Volumes/") || strings.HasPrefix(path, "/srv/models") {
		t.Fatalf("%s is on the archive, not the bench set (CLAUDE.md): a timing from it measures the disk", path)
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s: %v", path, err)
	}
	m, err := decoder.Load(path, decoder.Options{Quant: "int4", ResidentContext: ctx, ResidentKVSlots: slots, Knobs: knobs})
	if err != nil {
		t.Fatalf("load %s: %v", path, err)
	}
	r, err := buildResident(m)
	if err != nil {
		m.Close()
		t.Fatalf("build resident: %v", err)
	}
	t.Cleanup(func() {
		r.Close()
		m.Close()
		debug.FreeOSMemory()
	})
	return filepath.Base(path), &metalResident{r: r, hidden: r.H}
}

func auditHB(tag string, t0 time.Time, format string, a ...any) {
	fmt.Fprintf(os.Stderr, "[%s %6.1fs] %s\n", tag, time.Since(t0).Seconds(), fmt.Sprintf(format, a...))
}

func auditMedian(xs []float64) float64 {
	if len(xs) == 0 {
		return math.NaN()
	}
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	if len(s)%2 == 1 {
		return s[len(s)/2]
	}
	return (s[len(s)/2-1] + s[len(s)/2]) / 2
}

func auditEmbs(r *resident, n, salt int) [][]float32 {
	e := make([][]float32, n)
	for i := range e {
		e[i] = mc3Emb(r, 1000+(i*37+salt)%15000)
	}
	return e
}

// TestAuditT11_passCost (T1.1, A-P03): one PrefillLast pass's wall time at startPos 64, 2048 and 8000 for C = 16, 32,
// 48 and 64 tokens, the twelve cells interleaved rep by rep in a rotating order. The reading is the paired delta
// (startPos 2048) - (startPos 64) at C = 32, median over the reps.
func TestAuditT11_passCost(t *testing.T) {
	auditA(t)
	name, a := auditLoad(t, "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", 1, 8320, nil)
	t0 := time.Now()
	embs := auditEmbs(a.r, 8064, 0)
	if _, err := a.PrefillLast(context.Background(), embs, 0); err != nil {
		t.Fatalf("fill 8064 positions: %v", err)
	}
	type cell struct{ at, C int }
	var cells []cell
	for _, at := range []int{64, 2048, 8000} {
		for _, C := range []int{16, 32, 48, 64} {
			cells = append(cells, cell{at, C})
		}
	}
	reps := auditReps(7)
	wall := map[cell][]float64{} // PrefillLast records no GPU timestamps, so wall time only
	for rep := range reps {
		for k := range cells {
			c := cells[(k+rep)%len(cells)]
			st := time.Now()
			if _, err := a.PrefillLast(context.Background(), embs[c.at:c.at+c.C], c.at); err != nil {
				t.Fatalf("pass at %d, C=%d: %v", c.at, c.C, err)
			}
			wall[c] = append(wall[c], time.Since(st).Seconds()*1e3)
		}
		auditHB("t1.1", t0, "rep %d/%d done", rep+1, reps)
	}
	for _, at := range []int{64, 2048, 8000} {
		line := fmt.Sprintf("startPos %d:", at)
		for _, C := range []int{16, 32, 48, 64} {
			c := cell{at, C}
			line += fmt.Sprintf("  C=%d %.1f ms", C, auditMedian(wall[c]))
		}
		auditHB("t1.1", t0, "%s", line)
	}
	var d []float64
	for i := range wall[cell{2048, 32}] {
		d = append(d, wall[cell{2048, 32}][i]-wall[cell{64, 32}][i])
	}
	delta := auditMedian(d)
	verdict := "AMBIGUOUS (10-25 ms): parked"
	switch {
	case delta <= 10:
		verdict = "CLOSE A-P03 (<= 10 ms): the premise is gone"
	case delta >= 25:
		verdict = "BUILD the BQ=16 steel variant (>= 25 ms)"
	}
	auditHB("t1.1", t0, "RESULT %s: paired delta (startPos 2048 - 64) at C=32, median of %d: %.1f ms -> %s (pairs %s)",
		name, reps, delta, verdict, auditFmt(d))
}

// TestAuditT12_blkBelowFloor (T1.2, B-P03): per-token GPU time of decode below attention_fa's floor, the legacy
// per-head kernel against the block kernel (attnFAFloorOverride forces each), at 256 ... 1535 keys; each arm runs 20
// tokens from the same depth, arms interleaved and alternated rep by rep. The reading is legacy / blk per depth.
func TestAuditT12_blkBelowFloor(t *testing.T) {
	auditA(t)
	name, a := auditLoad(t, "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", 1, 4096, nil)
	r, t0 := a.r, time.Now()
	if r.attnFANKV == 0 || r.attnFABlkSplit == 0 {
		t.Skipf("%s: the block kernel is not selected here (attnFANKV %d, attnFABlkSplit %d)", name, r.attnFANKV, r.attnFABlkSplit)
	}
	depths := []int{256, 384, 512, 768, 1024, 1280, 1535}
	const tokens = 20
	embs := auditEmbs(r, 1535+tokens, 7)
	if _, err := a.PrefillLast(context.Background(), embs[:1535], 0); err != nil {
		t.Fatalf("fill: %v", err)
	}
	reps := auditReps(5)
	run := func(D, floor int) float64 {
		r.attnFAFloorOverride = floor
		var ks []float64
		for i := range tokens {
			r.ForwardEmb(embs[D+i], D+i)
			ks = append(ks, (r.gpuEnd-r.gpuStart)*1e3) // the token's GPU execution (kernStart/End is the CPU's scheduling window)
		}
		return auditMedian(ks)
	}
	ratios := map[int]float64{}
	for _, D := range depths {
		var leg, blk []float64
		for rep := range reps {
			if rep%2 == 0 {
				leg, blk = append(leg, run(D, 1<<30)), append(blk, run(D, 1))
			} else {
				blk, leg = append(blk, run(D, 1)), append(leg, run(D, 1<<30))
			}
		}
		r.attnFAFloorOverride = 0
		ratios[D] = auditMedian(leg) / auditMedian(blk)
		auditHB("t1.2", t0, "%s %d keys: legacy %.3f ms, blk %.3f ms per token (medians of %d x %d), legacy/blk %.3f",
			name, D, auditMedian(leg), auditMedian(blk), reps, tokens, ratios[D])
	}
	first := 0
	for _, D := range depths {
		if ratios[D] >= 1.10 {
			first = D
			break
		}
	}
	verdict := fmt.Sprintf("blk %.3fx legacy at 768 keys: below 1.05, the floor stays (on this model)", ratios[768])
	if ratios[768] >= 1.05 {
		verdict = fmt.Sprintf("blk %.3fx legacy at 768 keys (>= 1.05); first depth with >= 1.10: %d (0 = none)", ratios[768], first)
	}
	auditHB("t1.2", t0, "RESULT %s: %s", name, verdict)
}

// TestAuditT13_kernelGap (T1.3, C-B01): the GPU-idle gap between consecutive decode command buffers over 200 greedy
// tokens through the production path (Forward, the pipelined executor, host argmax and embed between tokens). The gap is
// GPUStartTime(t+1) - GPUEndTime(t): the audit wrote kernStart/kernEnd, but Metal's kernelStart/EndTime are the CPU's
// scheduling window for a command buffer, not its execution; that gap is printed too, for reference.
func TestAuditT13_kernelGap(t *testing.T) {
	auditA(t)
	name, a := auditLoad(t, "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", 1, 4096, nil)
	r, t0 := a.r, time.Now()
	prompt := auditEmbs(r, 64, 3)
	if _, err := a.PrefillLast(context.Background(), prompt, 0); err != nil {
		t.Fatalf("prefill: %v", err)
	}
	n := auditReps(200)
	emb := mc3Emb(r, 4242)
	var starts, ends, kStarts, kEnds []float64
	for i := range n {
		lg, err := a.Forward(emb, 64+i)
		if err != nil {
			t.Fatalf("token %d: %v", i, err)
		}
		starts, ends = append(starts, r.gpuStart), append(ends, r.gpuEnd)
		kStarts, kEnds = append(kStarts, r.kernStart), append(kEnds, r.kernEnd)
		emb = mc3Emb(r, argmaxF32(lg))
	}
	var gaps, kGaps []float64
	for i := 1; i < len(starts); i++ {
		gaps = append(gaps, (starts[i]-ends[i-1])*1e3)
		kGaps = append(kGaps, (kStarts[i]-kEnds[i-1])*1e3)
	}
	s := append([]float64(nil), gaps...)
	sort.Float64s(s)
	pct := func(p float64) float64 {
		if len(s) == 0 {
			return math.NaN()
		}
		return s[int(p*float64(len(s)-1))]
	}
	med := auditMedian(gaps)
	verdict := "C-B01 stands (median >= 0.15 ms)"
	if med < 0.15 {
		verdict = "KILLS C-B01 (median < 0.15 ms)"
	}
	auditHB("t1.3", t0, "RESULT %s: GPUStart(t+1) - GPUEnd(t) over %d gaps: median %.3f ms (p10 %.3f, p90 %.3f, max %.3f) -> %s; "+
		"scheduling-window gap kernStart(t+1) - kernEnd(t), for reference: median %.3f ms",
		name, len(gaps), med, pct(0.1), pct(0.9), pct(1), verdict, auditMedian(kGaps))
}

// TestAuditT14_saRows4PerByte (T1.4, B-P04): gemv_w4a8_sa_rows4 alone, at about 15 MB of weights for K = 1536 (48
// groups: the last trip half-idle), 2048 and 3072 (no tail), GPU time per weight byte; the K arms interleave rep by
// rep, and each command buffer rotates through enough weight copies to defeat the system-level cache.
func TestAuditT14_saRows4PerByte(t *testing.T) {
	auditA(t)
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	defer d.ReleaseObjects()
	defer d.ReleaseAll()
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	p, err := d.NewComputePipeline(lib, "gemv_w4a8_sa_rows4")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	cq, t0 := d.NewCommandQueue(), time.Now()
	seed := uint32(20261001)
	rnd := func() uint32 { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return seed }
	type arm struct {
		K, N, copies, bytes int
		ws, ss              []Buffer
		aq, asc, out, uK    Buffer
	}
	var arms []*arm
	for _, K := range []int{1536, 2048, 3072} {
		N := int(math.Round(15e6/float64(K/2+K/16)/32)) * 32 // rows tile by 8*R = 32
		bytes := N*K/2 + N*(K/32)*2
		a := &arm{K: K, N: N, bytes: bytes, copies: min(64, max(1, int(math.Ceil(float64(256<<20)/float64(bytes)))))}
		a.aq = d.NewBufferBytes(K)
		av := a.aq.Int8s()[:K] // one view; each Int8s/U32s call is an objc message
		for i := range av {
			av[i] = int8(rnd()%255) - 127
		}
		a.asc, a.out, a.uK = NewBufferFloats(d, []float32{0.0123}), d.NewBufferLen(N), NewBufferU32(d, uint32(K))
		for range a.copies {
			nw := N * K / 8
			wb := d.NewBufferLen(nw)
			wv := wb.U32s()[:nw]
			for i := range wv {
				wv[i] = rnd()
			}
			sc := make([]uint16, N*(K/32))
			for i := range sc {
				sc[i] = f32ToF16(float32(rnd()%1000+1) * 1e-5)
			}
			a.ws, a.ss = append(a.ws, wb), append(a.ss, NewBufferU16s(d, sc))
		}
		arms = append(arms, a)
	}
	reps := auditReps(7)
	nsPerMB := map[int][]float64{}
	for rep := range reps + 1 { // rep 0 warms up and is dropped
		for k := range arms {
			a := arms[(k+rep)%len(arms)]
			per := max(8, 2*a.copies)
			e := cq.Begin()
			for i := range per {
				e.DispatchTG(p, a.N*32/4, 256, a.K*2, a.ws[i%a.copies], a.ss[i%a.copies], a.aq, a.asc, a.out, a.uK)
			}
			e.End()
			if err := e.Err(); err != nil {
				t.Fatalf("K=%d: %v", a.K, err)
			}
			if rep > 0 {
				nsPerMB[a.K] = append(nsPerMB[a.K], (e.GPUEnd()-e.GPUStart())*1e9/float64(per)/(float64(a.bytes)/1e6))
			}
		}
	}
	for _, a := range arms {
		auditHB("t1.4", t0, "K=%d N=%d (%.1f MB, %d copies): %.1f ns per MB of weights (median of %d)",
			a.K, a.N, float64(a.bytes)/1e6, a.copies, auditMedian(nsPerMB[a.K]), reps)
	}
	tail, ctl := auditMedian(nsPerMB[1536])/auditMedian(nsPerMB[2048]), auditMedian(nsPerMB[3072])/auditMedian(nsPerMB[2048])
	verdict := "AMBIGUOUS: parked"
	switch {
	case tail >= 1.10 && ctl >= 0.95 && ctl <= 1.05:
		verdict = "CONFIRMS the idle tail"
	case tail < 1.05:
		verdict = "KILLS the idle tail as a lever (< 1.05)"
	}
	auditHB("t1.4", t0, "RESULT time per byte K=1536 / K=2048 = %.3f; control K=3072 / K=2048 = %.3f -> %s", tail, ctl, verdict)
}

// TestAuditT110_promptInRowsTiming (T1.10's timing half, E-P01): a fresh K-token prompt's wall time three ways: the
// sequential single-token loop, the batched step in 8-row pieces (bit-identical to the loop, by
// TestMC3Step_promptInRowsBitIdentical), and the batched prefill pass with its floor off. K = 8, 16, 32 and 64; arms
// interleaved in a rotating order rep by rep, each writing positions 0..K-1 of slot 0.
func TestAuditT110_promptInRowsTiming(t *testing.T) {
	auditA(t)
	name, a := auditLoad(t, "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", 2, 1024,
		&decoder.Knobs{"GOINFER_METAL_FAST_PREFILL_FLOOR": "0"})
	r, t0 := a.r, time.Now()
	if r.batch == nil {
		t.Skipf("%s builds no batched step (batchIneligible: %q)", name, r.batchIneligible())
	}
	reps := auditReps(7)
	arms := []string{"sequential", "step", "pass"}
	ms := map[string][]float64{}
	key := func(arm string, K int) string { return fmt.Sprintf("%s/%d", arm, K) }
	for _, K := range []int{8, 16, 32, 64} {
		embs := auditEmbs(r, K, K)
		for rep := range reps {
			for k := range arms {
				arm := arms[(k+rep)%len(arms)]
				if err := r.useKVSlot(0); err != nil {
					t.Fatal(err)
				}
				st := time.Now()
				switch arm {
				case "sequential":
					for i, e := range embs {
						r.ForwardEmb(e, i)
					}
				case "step":
					for c := 0; c < K; c += 8 {
						var seqs []batchSeq
						for i := c; i < min(c+8, K); i++ {
							seqs = append(seqs, batchSeq{slot: 0, pos: i, emb: embs[i]})
						}
						if _, _, err := r.forwardMulti(seqs); err != nil {
							t.Fatalf("step at %d: %v", c, err)
						}
					}
				case "pass":
					if _, err := a.PrefillLast(context.Background(), embs, 0); err != nil {
						t.Fatalf("pass K=%d: %v", K, err)
					}
				}
				ms[key(arm, K)] = append(ms[key(arm, K)], time.Since(st).Seconds()*1e3)
			}
		}
		auditHB("t1.10", t0, "%s K=%d: sequential %.1f ms, step %.1f ms, pass %.1f ms (medians of %d); step/sequential %.3f",
			name, K, auditMedian(ms[key("sequential", K)]), auditMedian(ms[key("step", K)]), auditMedian(ms[key("pass", K)]), reps,
			auditMedian(ms[key("step", K)])/auditMedian(ms[key("sequential", K)]))
	}
	ratio := auditMedian(ms[key("step", 32)]) / auditMedian(ms[key("sequential", 32)])
	verdict := "E-P01 proceeds (step <= 0.6x sequential at K=32)"
	if ratio > 0.6 {
		verdict = "KILLS E-P01 (step > 0.6x sequential at K=32)"
	}
	auditHB("t1.10", t0, "RESULT %s: step / sequential at K=32 = %.3f -> %s; step / pass at K=16, 32, 64 = %.3f, %.3f, %.3f",
		name, ratio, verdict,
		auditMedian(ms[key("step", 16)])/auditMedian(ms[key("pass", 16)]),
		auditMedian(ms[key("step", 32)])/auditMedian(ms[key("pass", 32)]),
		auditMedian(ms[key("step", 64)])/auditMedian(ms[key("pass", 64)]))
}

func auditFmt(xs []float64) string {
	var b strings.Builder
	for i, x := range xs {
		if i > 0 {
			b.WriteString(" ")
		}
		fmt.Fprintf(&b, "%.1f", x)
	}
	return b.String()
}
