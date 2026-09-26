package decoder

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/linalg"
)

// decodeMultiStep is MC2's prototype (docs/tasks/task-concurrency-2026-09.md), TEST-ONLY: ONE forward carrying B
// independent sequences, each with its own KV cache at its own position, one token each. The projections, o-proj, MLP
// and LM head run as M = B matmuls through the kernels forwardN uses (bit-identical to M = 1 row for row, by
// forwardN's own contract). Attention runs per sequence over its own cache, exactly as causalAttention's default
// (f32 KV, append-forever) case does at decode. Plain dense families only — see mc2Eligible.
func (m *Model) decodeMultiStep(ids []int, caches []*KVCache) ([][]float32, error) {
	if err := m.mc2Eligible(caches); err != nil {
		return nil, err
	}
	arch := m.w.arch
	be := m.be
	B := len(ids)
	hidden, nKV, hd := arch.HiddenDim, arch.NumKVHeads, arch.HeadDim
	maxQDim, kvDim, inter := arch.maxHeads()*hd, nKV*hd, arch.IntermediateDim
	row := func(b []float32, i, w int) []float32 { return b[i*w : i*w+w] }

	h := make([]float32, B*hidden)
	for b, id := range ids {
		m.w.Embed.Row(id, row(h, b, hidden))
		if arch.EmbedScale != 0 && arch.EmbedScale != 1 {
			s := float32(arch.EmbedScale)
			for j := range row(h, b, hidden) {
				h[b*hidden+j] *= s
			}
		}
	}
	norm := make([]float32, B*hidden)
	q, k, v := make([]float32, B*maxQDim), make([]float32, B*kvDim), make([]float32, B*kvDim)
	ctx, att := make([]float32, B*maxQDim), make([]float32, B*hidden)
	gate, up, mlpOut := make([]float32, B*inter), make([]float32, B*inter), make([]float32, B*hidden)
	pos := make([]int, B)
	for b, c := range caches {
		pos[b] = c.Pos()
	}
	var ws linalg.Workspace
	ws.SetThreshold(DefaultDecodeParallelThreshold)
	var qkvOps [3]linalg.W8A8Op
	var guOps [2]linalg.W8A8Op

	for l := 0; l < arch.NumLayers; l++ {
		lw := &m.w.Layers[l]
		global := arch.isGlobalLayer(l)
		nH := arch.headsAt(l)
		qDim := nH * hd
		q, ctx := q[:B*qDim], ctx[:B*qDim]
		for b := range B {
			normalizeInto(arch, row(norm, b, hidden), row(h, b, hidden), lw.PreAttnNorm, lw.PreAttnNormBias, hidden)
		}
		if isW8A8(&lw.QProj) && isW8A8(&lw.KProj) && isW8A8(&lw.VProj) {
			qkvOps[0] = linalg.W8A8Op{BQ: wmInt8(&lw.QProj), Scales: wmScales(&lw.QProj), Dst: q, N: lw.QProj.Rows()}
			qkvOps[1] = linalg.W8A8Op{BQ: wmInt8(&lw.KProj), Scales: wmScales(&lw.KProj), Dst: k, N: lw.KProj.Rows()}
			qkvOps[2] = linalg.W8A8Op{BQ: wmInt8(&lw.VProj), Scales: wmScales(&lw.VProj), Dst: v, N: lw.VProj.Rows()}
			matmulW8A8Batch(be, &ws, norm, B, lw.QProj.Cols(), qkvOps[:], lw.QProj.ActQuantGroup())
		} else {
			matmul(be, &lw.QProj, norm, q, B)
			matmul(be, &lw.KProj, norm, k, B)
			matmul(be, &lw.VProj, norm, v, B)
		}
		invFreq, ms := arch.ropeInvFreq(l), arch.ropeMscale(l)
		noPE := arch.isNoPELayer(l)
		for b, c := range caches {
			qb, kb, vb, cb := row(q, b, qDim), row(k, b, kvDim), row(v, b, kvDim), row(ctx, b, qDim)
			if arch.QKVBias {
				addBias(qb, lw.QBias)
				addBias(kb, lw.KBias)
				addBias(vb, lw.VBias)
			}
			if arch.QKNorm {
				if arch.QKNormWhole {
					rmsNorm(qb, lw.QNorm, 1, nH*hd, arch.NormEps, arch.RMSAddOne)
					rmsNorm(kb, lw.KNorm, 1, nKV*hd, arch.NormEps, arch.RMSAddOne)
				} else {
					rmsNorm(qb, lw.QNorm, nH, hd, arch.NormEps, arch.RMSAddOne)
					rmsNorm(kb, lw.KNorm, nKV, hd, arch.NormEps, arch.RMSAddOne)
				}
			}
			if !noPE {
				ropeAt(qb, nH, hd, pos[b], invFreq, ms, arch.MRopeSection, c.mropePos, c.mropeDelta, arch.ropeInterleave, arch.MRopeInterleaved)
				ropeAt(kb, nKV, hd, pos[b], invFreq, ms, arch.MRopeSection, c.mropePos, c.mropeDelta, arch.ropeInterleave, arch.MRopeInterleaved)
			}
			if arch.AttnTempBeta != 0 {
				scale := float32(1 + arch.AttnTempBeta*math.Log1p(math.Floor(float64(pos[b])/arch.AttnTempOrigMaxPos)))
				for j := range qb {
					qb[j] *= scale
				}
			}
			// causalAttention's default case, verbatim: f32 global, append-forever, acc64.
			c.Append(l, kb, vb)
			nKeys := c.storedRows(l, kvDim)
			pool := c.scr.headWorkerPool(nH, 1, nKeys, hd, false, true)
			attendBatchedHeads(qb, cb, c.Keys(l), c.Vals(l), 0, c, l, pos[b], 1, global, arch, true, pool)
		}
		matmul(be, &lw.OProj, ctx, att, B)
		if arch.OutBias {
			for b := range B {
				addBias(row(att, b, hidden), lw.OBias)
			}
		}
		addResidual(h, att)
		for b := range B {
			normalizeInto(arch, row(norm, b, hidden), row(h, b, hidden), lw.PreMLPNorm, lw.PreMLPNormBias, hidden)
		}
		if isW8A8(&lw.GateProj) && isW8A8(&lw.UpProj) {
			guOps[0] = linalg.W8A8Op{BQ: wmInt8(&lw.GateProj), Scales: wmScales(&lw.GateProj), Dst: gate, N: lw.GateProj.Rows()}
			guOps[1] = linalg.W8A8Op{BQ: wmInt8(&lw.UpProj), Scales: wmScales(&lw.UpProj), Dst: up, N: lw.UpProj.Rows()}
			matmulW8A8Batch(be, &ws, norm, B, lw.GateProj.Cols(), guOps[:], lw.GateProj.ActQuantGroup())
		} else {
			matmul(be, &lw.GateProj, norm, gate, B)
			matmul(be, &lw.UpProj, norm, up, B)
		}
		switch arch.Act { // forwardN's activation step, verbatim
		case ActGeluTanh:
			if len(gate) < activationFanoutThreshold {
				geglu(gate, up)
			} else {
				parallelElementwise(len(gate), func(lo, hi int) {
					for j := lo; j < hi; j++ {
						gate[j] = geluTanh(gate[j]) * up[j]
					}
				})
			}
		case ActSiLU:
			if len(gate) < activationFanoutThreshold {
				swiglu(gate, up)
			} else {
				parallelElementwise(len(gate), func(lo, hi int) {
					for j := lo; j < hi; j++ {
						gate[j] = silu(gate[j]) * up[j]
					}
				})
			}
		case ActGelu:
			if len(gate) < activationFanoutThreshold {
				gegluExact(gate, up)
			} else {
				parallelElementwise(len(gate), func(lo, hi int) {
					for j := lo; j < hi; j++ {
						gate[j] = geluErf(gate[j]) * up[j]
					}
				})
			}
		default:
			return nil, errNotImplemented
		}
		matmul(be, &lw.DownProj, gate, mlpOut, B)
		addResidual(h, mlpOut)
	}
	for b := range B {
		normalize(arch, row(h, b, hidden), m.w.FinalNorm, m.w.FinalNormBias, hidden)
	}
	flat := m.lmHeadN(h, B)
	out := make([][]float32, B)
	for b := range B {
		out[b] = flat[b*arch.VocabSize : (b+1)*arch.VocabSize]
	}
	return out, nil
}

// mc2Eligible is decodeMultiStep's scope: the generic forward (no family-specific runLayers), the plain pre-norm
// placement on every layer, dense MLP, no attention output gate, no learned positions, no dense weight streaming, and
// caches holding f32 append-forever KV with no adapter, tree mask or manual position.
func (m *Model) mc2Eligible(caches []*KVCache) error {
	a := m.w.arch
	if _, own := a.ownForward(); own {
		return fmt.Errorf("mc2: %s has its own forward", a.Name)
	}
	if a.MoE != nil || a.LearnedPosEmbed || a.hasAttnOutputGate() || m.layerPager != nil {
		return fmt.Errorf("mc2: %s is not a plain dense family", a.Name)
	}
	for l := 0; l < a.NumLayers; l++ {
		if p := a.normPlacementAt(l); p == NormSandwich4 || p == NormPostOnly || p == NormParallel {
			return fmt.Errorf("mc2: layer %d norm placement %v", l, p)
		}
	}
	for i, c := range caches {
		if c.scr == nil || c.localAny || c.quant == kvI8 || c.lora != nil || c.treeMask != nil || c.manualPos {
			return fmt.Errorf("mc2: cache %d is not a plain f32 append-forever cache", i)
		}
	}
	return nil
}

// TestMC2_decodeMultiStepBitIdentical is MC2's identity gate: B sequences stepped together through decodeMultiStep
// emit logits bit-identical to the production single-token forward over copies of the same caches, teacher-forced,
// at different depths per sequence. It runs on the committed llama-tiny fixture (CI), and on
// GOINFER_MC2_MODEL when set (int4 and int8int8).
func TestMC2_decodeMultiStepBitIdentical(t *testing.T) {
	type cfg struct{ path, quant string }
	cfgs := []cfg{{"../testdata/llama-tiny", ""}}
	if p := os.Getenv("GOINFER_MC2_MODEL"); p != "" {
		cfgs = append(cfgs, cfg{p, "int4"}, cfg{p, "int8int8"})
	}
	for _, c := range cfgs {
		t.Run(filepath.Base(c.path)+"/"+c.quant, func(t *testing.T) {
			if _, err := os.Stat(c.path); err != nil {
				t.Skipf("no checkpoint at %s: %v", c.path, err)
			}
			m, err := Load(c.path, Options{Quant: c.quant})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			if err := m.mc2Eligible(nil); err != nil {
				t.Skipf("%v", err)
			}
			vocab := m.w.arch.VocabSize
			ctx := context.Background()
			depths := []int{5, 23, 40, 11}
			multi, ref := make([]*KVCache, len(depths)), make([]*KVCache, len(depths))
			ids := make([]int, len(depths))
			for b, d := range depths {
				prompt := make([]int, d)
				for i := range prompt {
					prompt[i] = (i*37 + b*11 + 3) % vocab
				}
				for _, cp := range []**KVCache{&multi[b], &ref[b]} {
					*cp = m.NewCache(d + 64)
					if _, err := m.prefillLogits(ctx, prompt, *cp); err != nil {
						t.Fatalf("prefill: %v", err)
					}
				}
				ids[b] = (b*53 + 7) % vocab
			}
			for step := 0; step < 12; step++ {
				got, err := m.decodeMultiStep(ids, multi)
				if err != nil {
					t.Fatalf("step %d: %v", step, err)
				}
				for b := range ids {
					want, err := m.forward(ids[b], ref[b])
					if err != nil {
						t.Fatalf("step %d seq %d: forward: %v", step, b, err)
					}
					for j := range want {
						if math.Float32bits(got[b][j]) != math.Float32bits(want[j]) {
							t.Fatalf("step %d seq %d (pos %d): logit %d = %v batched vs %v alone — not bit-identical",
								step, b, ref[b].Pos()-1, j, got[b][j], want[j])
						}
					}
					ids[b] = argmaxF32(want) // teacher-forced: both arms take the same next id
				}
			}
			t.Logf("%s %q: %d sequences x 12 steps bit-identical to the single-token forward", filepath.Base(c.path), c.quant, len(depths))
		})
	}
}

func argmaxF32(v []float32) int {
	best := 0
	for i, x := range v {
		if x > v[best] {
			best = i
		}
	}
	return best
}

// TestMC2_batchedDecodeThroughput is MC2's measurement (docs/tasks/task-concurrency-2026-09.md): aggregate decode tok/s
// on CPU for B sequences through decodeMultiStep (B = 1, 2, 4, 8), against the production single-token forward taking
// the B sequences one at a time ("serial", today's behaviour) and against J8's cell, N independent decode workers on
// the same model (N = 2, 4 concurrent forwards, each on its own cache). Every arm decodes S steps per sequence from a
// cache prefilled to the same depth; arms are interleaved rep by rep, and the arm order rotates.
//
//	GOINFER_MC2=1 GOINFER_MC2_MODEL=~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf \
//	  go test -count=1 -timeout 60m -run '^TestMC2_batchedDecodeThroughput$' -v ./decoder/
//
// Env: GOINFER_MC2_QUANT (default int4), GOINFER_MC2_DEPTH (default 128), GOINFER_MC2_STEPS (default 16),
// GOINFER_MC2_REPS (default 5).
func TestMC2_batchedDecodeThroughput(t *testing.T) {
	if os.Getenv("GOINFER_MC2") != "1" {
		t.Skip("set GOINFER_MC2=1 and GOINFER_MC2_MODEL (loads a real checkpoint; minutes of CPU time)")
	}
	path := os.Getenv("GOINFER_MC2_MODEL")
	if strings.HasPrefix(path, "/Volumes/") || strings.HasPrefix(path, "/srv/models") {
		t.Fatalf("%s is on the archive, not the bench set (CLAUDE.md)", path)
	}
	envInt := func(k string, def int) int {
		if v, err := strconv.Atoi(os.Getenv(k)); err == nil && v > 0 {
			return v
		}
		return def
	}
	quant := os.Getenv("GOINFER_MC2_QUANT")
	if quant == "" {
		quant = "int4"
	}
	depth, steps, reps := envInt("GOINFER_MC2_DEPTH", 128), envInt("GOINFER_MC2_STEPS", 16), envInt("GOINFER_MC2_REPS", 5)
	t0 := time.Now()
	hb := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[mc2 %6.1fs] %s\n", time.Since(t0).Seconds(), fmt.Sprintf(format, a...))
	}
	m, err := Load(path, Options{Quant: quant})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if err := m.mc2Eligible(nil); err != nil {
		t.Skipf("%v", err)
	}
	vocab := m.w.arch.VocabSize
	ctx := context.Background()
	const maxB = 8
	// maxB caches prefilled to depth once; each arm rewinds them to depth before running (positional KV: truncation
	// is exact), so every arm decodes over identical history.
	caches := make([]*KVCache, maxB)
	for b := range caches {
		prompt := make([]int, depth)
		for i := range prompt {
			prompt[i] = (i*37 + b*11 + 3) % vocab
		}
		caches[b] = m.NewCache(depth + steps + 8)
		if _, err := m.prefillLogits(ctx, prompt, caches[b]); err != nil {
			t.Fatalf("prefill: %v", err)
		}
	}
	rewind := func(n int) {
		for b := range n {
			caches[b].TruncateTo(depth)
		}
	}
	startIDs := func(n int) []int {
		ids := make([]int, n)
		for b := range ids {
			ids[b] = (b*53 + 7) % vocab
		}
		return ids
	}
	type arm struct {
		name string
		n    int // sequences
		run  func() error
	}
	batched := func(B int) func() error {
		return func() error {
			ids := startIDs(B)
			for range steps {
				lg, err := m.decodeMultiStep(ids, caches[:B])
				if err != nil {
					return err
				}
				for b := range ids {
					ids[b] = argmaxF32(lg[b])
				}
			}
			return nil
		}
	}
	serial := func(B int) func() error { // today: one generation at a time, each sequence's steps in turn
		return func() error {
			ids := startIDs(B)
			for range steps {
				for b := range ids {
					lg, err := m.forward(ids[b], caches[b])
					if err != nil {
						return err
					}
					ids[b] = argmaxF32(lg)
				}
			}
			return nil
		}
	}
	workers := func(N int) func() error { // J8: N concurrent decode workers, each on its own cache
		return func() error {
			ids := startIDs(N)
			var wg sync.WaitGroup
			errs := make([]error, N)
			for b := range N {
				wg.Add(1)
				go func(b int) {
					defer wg.Done()
					id := ids[b]
					for range steps {
						lg, err := m.forward(id, caches[b])
						if err != nil {
							errs[b] = err
							return
						}
						id = argmaxF32(lg)
					}
				}(b)
			}
			wg.Wait()
			for _, e := range errs {
				if e != nil {
					return e
				}
			}
			return nil
		}
	}
	arms := []arm{
		{"serial x1", 1, serial(1)}, {"serial x4", 4, serial(4)},
		{"batched B=1", 1, batched(1)}, {"batched B=2", 2, batched(2)}, {"batched B=4", 4, batched(4)}, {"batched B=8", 8, batched(8)},
		{"J8 workers N=2", 2, workers(2)}, {"J8 workers N=4", 4, workers(4)},
	}
	hb("loaded %s (%s): depth %d, %d steps per sequence, %d reps, %d arms", filepath.Base(path), quant, depth, steps, reps, len(arms))
	for _, a := range arms { // warm every arm once
		rewind(a.n)
		if err := a.run(); err != nil {
			t.Fatalf("%s: %v", a.name, err)
		}
	}
	rates := make([][]float64, len(arms))
	for rep := range reps {
		line := fmt.Sprintf("rep %d/%d:", rep+1, reps)
		for k := range arms {
			ai := (k + rep) % len(arms)
			a := arms[ai]
			rewind(a.n)
			st := time.Now()
			if err := a.run(); err != nil {
				t.Fatalf("%s: %v", a.name, err)
			}
			r := float64(a.n*steps) / time.Since(st).Seconds()
			rates[ai] = append(rates[ai], r)
		}
		for ai, a := range arms {
			line += fmt.Sprintf("  %s %.1f", a.name, rates[ai][rep])
		}
		hb("%s tok/s", line)
	}
	med := func(xs []float64) float64 {
		s := append([]float64(nil), xs...)
		sort.Float64s(s)
		return s[len(s)/2]
	}
	base := med(rates[0])
	hb("aggregate decode tok/s, median of %d reps (%s %s, depth %d):", reps, filepath.Base(path), quant, depth)
	for ai, a := range arms {
		hb("  %-16s %7.1f tok/s  %.3fx serial x1", a.name, med(rates[ai]), med(rates[ai])/base)
	}
	// The registered MC2 metric: batched B=4 against B=1 (the production single-token forward, one sequence).
	ratios := make([]float64, reps)
	for i := range ratios {
		ratios[i] = rates[4][i] / rates[0][i]
	}
	sort.Float64s(ratios)
	hb("MC2 METRIC batched B=4 / serial x1 = %.3fx (per-rep sorted %v); J8 N=4 / serial x1 = %.3fx",
		ratios[len(ratios)/2], ratios, med(rates[7])/base)
}
