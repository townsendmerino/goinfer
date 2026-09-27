//go:build darwin && goinfer_testhooks

package metal

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestR18InSequence is R18's grading instrument (docs/tasks/red-october.md R18; graded 2026-09-26, then wired). On a real
// checkpoint it times the production decode token against the SHIPPED kernels (resident.gemvRows zeroed, so every GEMV
// takes its one-row-per-simdgroup kernel), plus any prototype arms, through resident.gemvRows and the four rows-kernel
// pipelines, and measures:
//
//  1. precondition 1, bit-identity: a teacher-forced sequence of decode tokens after a prefill to each depth, through
//     the production executor (ForwardEmbPipe), every logit compared bit for bit with the shipped kernels';
//  2. the registered metric: in-sequence int4-GEMV work per token — TestMetalDecodeDecomp's no-op method, per
//     category (qkv, o, gate/up, down), full − full-with-that-category-no-op'd at the arm's own grid, summed —
//     shipped ÷ arm per paired rep, median over reps. Each category's difference is the median over matched
//     step pairs (a full token, then the no-op'd one, adjacent), and arm order rotates rep by rep;
//  3. precondition 2's full-token times, and precondition 3's full token after 2 s idle.
//
// A prototype arm (GOINFER_METAL_R18_CANDS) is "<F><qkvR><oR><guR><downR>": F is i (step 0, integer math), f (step 1,
// shift-free f32) or h (R18b: masked, half-staged); each digit is rows per simdgroup for that GEMV, and downR 0 keeps the shipped coal down projection
// (else gemv_w4a8_resid_st<R>). "i1110" is the harness control: the prototype template at the shipped grid, which must
// time like the shipped arm. The graded candidate was "i2244", which is what production now runs.
//
//	GOINFER_METAL_R18_SEQ=1 go test -tags goinfer_testhooks -count=1 -timeout 60m -v -run '^TestR18InSequence$' ./metal/
//
// Env: GOINFER_METAL_R18_MODEL (default the 1.5B q4_k_m), _DEPTHS (default 128), _CANDS (prototype arms, default none),
// _REPS (default 5), _TOKENS (decode steps per measurement, default 20), _IDTOKENS (identity sequence length, default
// 16), _IDLE (after-idle samples per arm, default 3; 0 skips).
func TestR18InSequence(t *testing.T) {
	if os.Getenv("GOINFER_METAL_R18_SEQ") != "1" {
		t.Skip("set GOINFER_METAL_R18_SEQ=1 (loads a real checkpoint; minutes of GPU time)")
	}
	home, _ := os.UserHomeDir()
	path := os.Getenv("GOINFER_METAL_R18_MODEL")
	if path == "" {
		path = filepath.Join(home, "models", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
	if strings.HasPrefix(path, "/Volumes/") || strings.HasPrefix(path, "/srv/models") {
		t.Fatalf("%s is on the archive, not the bench set (CLAUDE.md): a timing from it measures the disk", path)
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s: %v", path, err)
	}
	ints := func(env string, def []int) []int {
		v := os.Getenv(env)
		if v == "" {
			return def
		}
		var out []int
		for _, f := range strings.Split(v, ",") {
			n, err := strconv.Atoi(strings.TrimSpace(f))
			if err != nil {
				t.Fatalf("%s: %v", env, err)
			}
			out = append(out, n)
		}
		return out
	}
	depths := ints("GOINFER_METAL_R18_DEPTHS", []int{128})
	reps := ints("GOINFER_METAL_R18_REPS", []int{5})[0]
	steps := ints("GOINFER_METAL_R18_TOKENS", []int{20})[0]
	idTokens := ints("GOINFER_METAL_R18_IDTOKENS", []int{16})[0]
	idle := ints("GOINFER_METAL_R18_IDLE", []int{3})[0]
	var candSpecs []string
	if v := os.Getenv("GOINFER_METAL_R18_CANDS"); v != "" {
		candSpecs = strings.Split(v, ",")
	}
	t0 := time.Now()
	hb := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[r18-seq %6.1fs] %s\n", time.Since(t0).Seconds(), fmt.Sprintf(format, a...))
	}

	m, err := decoder.Load(path, decoder.Options{Quant: "int4", ResidentContext: metalCtxCapDefault})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("build resident: %v", err)
	}
	defer r.Close()
	if r.moe != nil || r.g4moe != nil || r.sandwich || r.postOnly || r.parallelBlock || r.kvI8 || r.layerNorm ||
		r.decodeLaneW4F16 || r.nonGatedMLP || r.outBias || r.loraLayers != nil {
		t.Skip("R18 covers the plain dense W4A8 decode path only")
	}
	for _, L := range r.layers {
		if L.qGate || L.delta != nil {
			t.Skip("R18 covers the plain dense W4A8 decode path only")
		}
	}
	for _, D := range depths {
		if D+idTokens > r.ctxCap {
			t.Fatalf("depth %d + %d identity tokens does not fit the resident context %d", D, idTokens, r.ctxCap)
		}
	}
	g0 := r.layers[0].geom
	qkvRows := r.nH*g0.hd + 2*g0.kvDim
	for _, L := range r.layers {
		if L.geom.hd != g0.hd || L.geom.kvDim != g0.kvDim {
			t.Skip("R18's divisibility check assumes uniform layer geometry")
		}
	}

	var noop Pipeline
	pipes := map[string]Pipeline{}
	func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		pool := NewARPool()
		defer pool.Drain()
		lib, err := r.d.CompileLibrary(allKernels+"\n"+r18Kernels+"\nkernel void r18_noop() {}\n", MSL3_1)
		if err != nil {
			t.Fatalf("compile: %v", err)
		}
		get := func(name string) Pipeline {
			if p, ok := pipes[name]; ok {
				return p
			}
			p, err := r.d.NewComputePipeline(lib, name)
			if err != nil {
				t.Fatalf("pipeline %s: %v", name, err)
			}
			pipes[name] = p
			return p
		}
		noop = get("r18_noop")
		for _, c := range candSpecs {
			if len(c) != 5 || (c[0] != 'i' && c[0] != 'f' && c[0] != 'h') {
				t.Fatalf("candidate %q: want <i|f|h><qkvR><oR><guR><downR>", c)
			}
			for _, tag := range []string{c[0:1] + c[1:2], c[0:1] + c[2:3], c[0:1] + c[3:4]} {
				get("r18_sa_" + tag)
				get("r18_sa_bias_" + tag)
				get("r18_sa_resid_" + tag)
			}
			if c[4] != '0' {
				get("gemv_w4a8_resid_st" + c[4:5])
			}
		}
	}()

	// An arm is what runs at the four rows-kernel slots plus resident.gemvRows. The baseline, "shipped", zeroes
	// gemvRows, so every GEMV takes its shipped kernel; "production" is what buildResident selected.
	type arm struct {
		name             string
		qkv, o, gu, down Pipeline
		rows             struct{ qkv, o, gu, down int }
	}
	prod := arm{name: "production", qkv: r.pSABiasRows, o: r.pSAResidRows, gu: r.pSARows, down: r.pGemvResidStaged, rows: r.gemvRows}
	shippedPipes := [4]Pipeline{r.pSABias, r.pSAResid, r.pSA, r.pGemvResid}
	arms := []arm{{name: "shipped"}, prod}
	for _, c := range candSpecs {
		d := func(i int) int { return int(c[i] - '0') }
		a := arm{name: c, qkv: pipes["r18_sa_bias_"+c[0:1]+c[1:2]], o: pipes["r18_sa_resid_"+c[0:1]+c[2:3]],
			gu: pipes["r18_sa_"+c[0:1]+c[3:4]]}
		a.rows.qkv, a.rows.o, a.rows.gu, a.rows.down = d(1), d(2), d(3), d(4)
		if a.rows.down > 0 {
			a.down = pipes["gemv_w4a8_resid_st"+c[4:5]]
		}
		// TG 256 = 8 simdgroups x R rows: every GEMV's row count must divide (the prototypes carry no bounds guard)
		for _, chk := range []struct{ rows, R int }{{qkvRows, a.rows.qkv}, {r.H, a.rows.o}, {2 * r.I, a.rows.gu}, {r.H, a.rows.down}} {
			if chk.R > 0 && chk.rows%(8*chk.R) != 0 {
				t.Fatalf("candidate %s: %d rows do not divide into threadgroups of 8x%d", c, chk.rows, chk.R)
			}
		}
		arms = append(arms, a)
	}
	// category order: qkv, o, gate/up, down
	catNames := []string{"qkv", "o", "gate/up", "down"}
	set := func(a arm, noopCat int) (restore func()) {
		r.pSABiasRows, r.pSAResidRows, r.pSARows, r.pGemvResidStaged = a.qkv, a.o, a.gu, a.down
		r.gemvRows = a.rows
		// a no-op'd category no-ops both of its kernels, whichever the arm dispatches
		slots := [4][2]*Pipeline{{&r.pSABias, &r.pSABiasRows}, {&r.pSAResid, &r.pSAResidRows}, {&r.pSA, &r.pSARows},
			{&r.pGemvResid, &r.pGemvResidStaged}}
		for c := range slots {
			if noopCat == c || noopCat == 4 {
				*slots[c][0], *slots[c][1] = noop, noop
			}
		}
		r.stopExec() // drop any command buffer pre-encoded under the previous arm (see runR2GateCell)
		return func() {
			r.pSABias, r.pSAResid, r.pSA, r.pGemvResid = shippedPipes[0], shippedPipes[1], shippedPipes[2], shippedPipes[3]
			r.pSABiasRows, r.pSAResidRows, r.pSARows, r.pGemvResidStaged = prod.qkv, prod.o, prod.gu, prod.down
			r.gemvRows = prod.rows
			r.stopExec()
		}
	}
	hb("loaded %s: H=%d I=%d nL=%d nH=%d hd=%d kvDim=%d V=%d, ctx %d, attention_fa %v; qkv rows %d; production gemvRows %+v; prototype arms %v",
		filepath.Base(path), r.H, r.I, r.nL, r.nH, g0.hd, g0.kvDim, r.V, r.ctxCap, r.decodeAttnFA, qkvRows, r.gemvRows, candSpecs)

	vocab := r.V
	for _, D := range depths {
		embs := make([][]float32, D)
		for i := range embs {
			embs[i] = m.EmbedResidentForTest((i*131 + 7) % vocab)
		}
		prefill := func() {
			r.PrefillLast(embs, 0)
			if err := r.takeExecErr(); err != nil {
				t.Fatalf("prefill to %d: %v", D, err)
			}
		}

		// 1. bit-identity through the production executor, teacher-forced
		seq := make([]int, idTokens)
		for i := range seq {
			seq[i] = (D*131 + 7 + i*977) % vocab
		}
		var refLg [][]float32
		for ai, a := range arms {
			prefill() // with the shipped kernels, so every arm decodes from the same KV
			restore := set(a, -1)
			var lg [][]float32
			for i, tok := range seq {
				out := r.ForwardEmbPipe(m.EmbedResidentForTest(tok), D+i)
				if err := r.takeExecErr(); err != nil {
					t.Fatalf("%s decode at %d: %v", a.name, D+i, err)
				}
				lg = append(lg, append([]float32(nil), out...))
			}
			restore()
			if ai == 0 {
				refLg = lg
				continue
			}
			worst, badPos := 0, 0
			for i := range lg {
				diff := 0
				for j := range lg[i] {
					if math.Float32bits(lg[i][j]) != math.Float32bits(refLg[i][j]) {
						diff++
					}
				}
				if diff > 0 {
					badPos++
				}
				worst = max(worst, diff)
			}
			hb("depth %d identity: %s vs shipped over %d teacher-forced positions (%d..%d, executor): %d positions differ, worst %d of %d logits",
				D, a.name, len(seq), D, D+len(seq)-1, badPos, worst, vocab)
			if badPos > 0 {
				t.Errorf("depth %d: candidate %s is not bit-identical (%d of %d positions differ) — not an R18 candidate", D, a.name, badPos, len(seq))
			}
		}

		// 2. timing: decode at position D over D+1 keys, repeated (the same slot is rewritten)
		prefill()
		tok := (D*131 + 7) % vocab
		step := func() float64 {
			r.Forward(tok, D)
			if err := r.takeExecErr(); err != nil {
				t.Fatalf("decode at %d: %v", D, err)
			}
			return (r.gpuEnd - r.gpuStart) * 1e3
		}
		// pair: `steps` matched pairs, each a full token then the same token with category c no-op'd (4 = all four),
		// adjacent in time, so a GPU clock change between measurements cannot land on one side of the difference —
		// measured 2026-09-26 on the 1.5B, 20-step blocks per side let one block shift wholesale (a rep's summed work
		// read 4.2 ms against 8-9 in the others). Returns the median full token and the median per-pair difference.
		pair := func(a arm, c int) (full, work float64) {
			fs, ds := make([]float64, steps), make([]float64, steps)
			for i := range fs {
				restore := set(a, -1)
				fs[i] = step()
				restore()
				restore = set(a, c)
				ds[i] = fs[i] - step()
				restore()
			}
			return median(fs), median(ds)
		}
		for i := 0; i < 5; i++ {
			step()
		}
		type res struct {
			full, all4, work []float64
			cat              [4][]float64
		}
		rs := make([]res, len(arms))
		for rep := 0; rep < reps; rep++ {
			line := fmt.Sprintf("depth %d rep %d/%d:", D, rep+1, reps)
			for k := range arms {
				ai := (k + rep) % len(arms)
				a := arms[ai]
				w, fulls := 0.0, []float64{}
				for c := 0; c < 4; c++ {
					f, v := pair(a, c)
					rs[ai].cat[c] = append(rs[ai].cat[c], v)
					w += v
					fulls = append(fulls, f)
				}
				f, v := pair(a, 4)
				rs[ai].full = append(rs[ai].full, median(append(fulls, f)))
				rs[ai].all4 = append(rs[ai].all4, v)
				rs[ai].work = append(rs[ai].work, w)
			}
			for ai, a := range arms {
				line += fmt.Sprintf("  %s full %.3f work %.3f", a.name, rs[ai].full[rep], rs[ai].work[rep])
				if ai > 0 {
					line += fmt.Sprintf(" (%.3fx)", rs[0].work[rep]/rs[ai].work[rep])
				}
			}
			hb("%s", line)
		}

		// 3. the full token right after 2 s idle, per arm, alternating
		idleMs := make([][]float64, len(arms))
		for s := 0; s < idle; s++ {
			for ai, a := range arms {
				restore := set(a, -1)
				step() // the swap's first token (encode-side state), then idle
				time.Sleep(2 * time.Second)
				idleMs[ai] = append(idleMs[ai], step())
				restore()
			}
		}

		hb("depth %d — int4-GEMV work per token (qkv+o+gate/up+down, each the median of %d step-paired full − that-category-no-op'd, summed), median of %d paired reps:", D, steps, reps)
		for ai, a := range arms {
			ratios := make([]float64, reps)
			for i := range ratios {
				ratios[i] = rs[0].work[i] / rs[ai].work[i]
			}
			sort.Float64s(ratios)
			cats := ""
			for c := 0; c < 4; c++ {
				cats += fmt.Sprintf(" %s %.3f", catNames[c], median(rs[ai].cat[c]))
				if ai > 0 {
					cats += fmt.Sprintf(" (%.2fx)", median(rs[0].cat[c])/median(rs[ai].cat[c]))
				}
			}
			line := fmt.Sprintf("  %-8s work %.3f ms [%s ] all-4-no-op'd %.3f; full token %.3f ms", a.name, median(rs[ai].work), cats,
				median(rs[ai].all4), median(rs[ai].full))
			if ai > 0 {
				line += fmt.Sprintf(" (%+.3f); METRIC shipped/arm = %.3fx (per-rep sorted %v)", median(rs[ai].full)-median(rs[0].full),
					ratios[len(ratios)/2], fmtRatios(ratios))
			}
			if idle > 0 {
				line += fmt.Sprintf("; after 2 s idle %v", fmtRatios(idleMs[ai]))
			}
			hb("%s", line)
			if ai > 1 { // a prototype arm: R18b's metric is production (arm 1) over it, paired per rep
				pr := make([]float64, reps)
				for i := range pr {
					pr[i] = rs[1].work[i] / rs[ai].work[i]
				}
				sort.Float64s(pr)
				hb("  %-8s METRIC production/arm = %.3fx (per-rep sorted %v); full token %+.3f ms vs production", a.name,
					pr[len(pr)/2], fmtRatios(pr), median(rs[ai].full)-median(rs[1].full))
			}
		}
	}
}

func fmtRatios(xs []float64) string {
	parts := make([]string, len(xs))
	for i, x := range xs {
		parts[i] = strconv.FormatFloat(x, 'f', 3, 64)
	}
	return "[" + strings.Join(parts, " ") + "]"
}
