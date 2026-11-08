//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestKV16GateVsReference is the registered fidelity gate of docs/tasks/task-cuda-f16-kv-2026-10.md: the f32 cache (exact) and the f16 cache, both
// scored against the CPU f32/f64 reference of held-out prompt set B, with the flash-decode lane's criteria (a) / (b) / (c) and its preconditions.
// The arms load one after the other (a 7B's two weight copies do not fit the card together); the overflow survey runs inside the f32 arm, on the
// real K/V of every gate prompt, before any score is read.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_PREFILL_GATE_PROMPTS=b GOINFER_CUDA_FLASH_DECODE=16 \
//	  go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestKV16GateVsReference -v -timeout 3h
//
// Cells: D7 K=8000 (decision), S (1.5B) K=3900 (confirmation); GOINFER_KV16_GATE_CELLS="D7:8000,S:3900" overrides. A cell may name how
// many of the set's prompts it reads, "NAME:K:n" (the first n). KV4b's per-group cells are TestKV4bGate_* (kv4b_gate_test.go).
func TestKV16GateVsReference(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (real checkpoints; needs the set-B reference files)")
	}
	if s, _ := strconv.Atoi(os.Getenv("GOINFER_CUDA_FLASH_DECODE")); s < 1 {
		t.Skip("set GOINFER_CUDA_FLASH_DECODE=16: both arms run the production decode lane")
	}
	label, promptFiles := decoder.PrefillGatePromptSet()
	if label != "b" {
		t.Fatalf("the registered gate runs on held-out set B; GOINFER_PREFILL_GATE_PROMPTS=%q selects set %q", os.Getenv("GOINFER_PREFILL_GATE_PROMPTS"), label)
	}
	home, _ := os.UserHomeDir()
	refDir := filepath.Join(home, "goinfer-logs", "prefill-ref-b")
	cells := os.Getenv("GOINFER_KV16_GATE_CELLS")
	if cells == "" {
		cells = "D7:8000,S:3900"
	}
	models := map[string]struct {
		env, def string
		decision bool
		opts     decoder.Options // what the group's build needs beyond the cell's own Backend, Quant, KVPrecision and ResidentContext
	}{
		"D7": {"GOINFER_CUDA_GATE_MODEL_D7", "$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf", true, decoder.Options{}},
		"S":  {"GOINFER_CUDA_GATE_MODEL", "$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", false, decoder.Options{}},
	}
	for cell := range strings.SplitSeq(cells, ",") {
		parts := strings.Split(strings.TrimSpace(cell), ":")
		if len(parts) != 2 && len(parts) != 3 {
			t.Fatalf("bad cell %q", cell)
		}
		name := parts[0]
		K, _ := strconv.Atoi(parts[1])
		files := promptFiles
		if len(parts) == 3 {
			n, _ := strconv.Atoi(parts[2])
			if n < 2 || n > len(promptFiles) {
				t.Fatalf("bad cell %q: n must be 2 to %d", cell, len(promptFiles))
			}
			files = promptFiles[:n]
		}
		mc, ok := models[name]
		if !ok || K < 1 {
			t.Fatalf("bad cell %q", cell)
		}
		t.Run(fmt.Sprintf("%s-K%d", name, K), func(t *testing.T) {
			path := os.Getenv(mc.env)
			if path == "" {
				path = os.ExpandEnv(mc.def)
			}
			for _, bad := range []string{"/srv/models", "/Volumes/"} {
				if strings.HasPrefix(path, bad) {
					t.Fatalf("%s is the archive, not a bench surface (CLAUDE.md)", path)
				}
			}
			if _, err := os.Stat(path); err != nil {
				t.Skipf("no fixture at %s", path)
			}
			for pi := range files {
				if _, err := os.Stat(filepath.Join(refDir, fmt.Sprintf("%s-K%d-p%d.bin", name, K, pi))); err != nil {
					t.Skipf("%s K=%d: reference missing for prompt %d — run Phase A first", name, K, pi)
				}
			}
			runKV16GateCell(t, path, name, K, files, refDir, mc.decision, mc.opts)
		})
	}
}

// kv16ArmResult is one arm's teacher-forced rows for every prompt, plus what its resident said about itself.
type kv16ArmResult struct {
	rows       [][][]float32 // [prompt][row][vocab]
	allocated  int64         // the K/V cache bytes the load allocated (kvAllocSeenForTest)
	kvPool     int64         // the KV pools' and null blocks' bytes, from the layers that size them, block tables included
	tables     int64         // the block tables' share of kvPool (allocated does not count them)
	fit        string        // the build's own fit decision for the arm: its context and KV pool, as the plan sized them
	faLaunches int
	f16        bool
	i8         bool        // the arm's resident holds int8
	aa         [][]float32 // a second run of prompt 0 (the arm asked for it), for the A/A precondition
	maxK, maxV []float32   // per layer, over every prompt (the f32 arm only)
	layers     int
	faLayers   int    // the layers whose attended span reaches the lane's floor at this depth (a sliding window under it never does)
	mlaOnly    bool   // every layer that keeps K/V is MLA's latent, which the flash-decode lane never reads
	faModule   string // the build of decode_fa the lane's kernels were compiled from (cudaResident.faModule)
	kvDim, nKV int    // the KV-keeping layers' K row width and KV heads, when every such layer shares them (uniform)
	uniform    bool
}

func kv16RunArm(t *testing.T, path, kv string, K int, prompts [][]int, refTokens [][]int, survey, aa bool, tag string, opts decoder.Options) kv16ArmResult {
	t.Helper()
	opts.Backend, opts.Quant, opts.KVPrecision, opts.ResidentContext = "cuda", "int4", kv, K+128
	var mu sync.Mutex
	seen := map[*cudaResident]int64{}
	kvAllocSeenForTest = func(r *cudaResident, b int) { mu.Lock(); seen[r] += int64(b); mu.Unlock() }
	m, err := decoder.Load(path, opts)
	kvAllocSeenForTest = nil
	if err != nil {
		t.Fatalf("load (-kv %s): %v", kv, err)
	}
	defer m.Close()
	rf, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatalf("-kv %s: no CUDA resident: %s", kv, m.ResidentDecline())
	}
	if rf.faSplit < 1 || rf.faCombine == (Pipeline{}) {
		t.Fatalf("-kv %s: the flash-decode lane is not loaded (faSplit=%d)", kv, rf.faSplit)
	}
	res := kv16ArmResult{f16: rf.kvF16, i8: rf.kvI8, allocated: seen[rf], kvPool: kvBytesForCap(rf.ctxCap, rf.layers) + kvPoolNullBytes(rf.layers), layers: len(rf.layers),
		tables: int64(kvGroups * 4 * kvBlocksFor(rf.ctxCap))}
	// The build's own fit check sized this arm (BuildResident refuses a load that does not fit, and the load above would have failed): record
	// what it chose, so the cell's record shows each arm passed it.
	res.fit = fmt.Sprintf("-kv %s: resident, context %d positions, %d KV slot(s), KV pools %d B", kv, rf.ctxCap, rf.KVSlots(), res.kvPool)
	res.mlaOnly, res.faModule, res.uniform = true, rf.faModule, true
	for l := range rf.layers {
		if !rf.layers[l].noKV() && !rf.layers[l].isMLA() {
			res.mlaOnly = false
			if res.kvDim == 0 {
				res.kvDim, res.nKV = rf.layers[l].kvDim, rf.layers[l].nKV
			} else if rf.layers[l].kvDim != res.kvDim || rf.layers[l].nKV != res.nKV {
				res.uniform = false
			}
		}
		if w := int(rf.layers[l].window); !rf.layers[l].noKV() && !rf.layers[l].isMLA() && rf.faEligible(l) && (w == 0 || w >= rf.faMinKeys) {
			res.faLayers++
		}
	}
	res.maxK, res.maxV = make([]float32, len(rf.layers)), make([]float32, len(rf.layers))
	ctx := context.Background()
	t0 := time.Now()
	one := func(pi int, doSurvey bool) [][]float32 {
		ids := prompts[pi]
		embs := make([][]float32, K)
		for i, id := range ids[:K] {
			embs[i] = m.EmbedResidentForTest(id)
		}
		seed, err := rf.PrefillLast(ctx, embs, 0)
		if err != nil {
			t.Fatalf("PrefillLast (-kv %s) prompt %d: %v", kv, pi, err)
		}
		if doSurvey {
			for l := range rf.layers {
				if rf.layers[l].noKV() || rf.layers[l].kvShared { // a recurrent layer keeps no rows; a shared one reads another layer's
					continue
				}
				var kb, vb []float32
				if e := rf.do(func() error {
					if e := rf.stream.Sync(); e != nil {
						return e
					}
					var e error
					kb, vb, e = rf.kvRowsForTest(rf.kvSlot, l, 0, K)
					return e
				}); e != nil {
					t.Fatalf("survey download layer %d: %v", l, e)
				}
				for _, v := range kb {
					res.maxK[l] = max(res.maxK[l], float32(math.Abs(float64(v))))
				}
				for _, v := range vb {
					res.maxV[l] = max(res.maxV[l], float32(math.Abs(float64(v))))
				}
			}
		}
		rows := make([][]float32, len(refTokens[pi]))
		rows[0] = append([]float32(nil), seed...)
		pos := K - 1
		for i := 1; i < len(rows); i++ {
			pos++
			lg, err := rf.Forward(m.EmbedResidentForTest(refTokens[pi][i-1]), pos)
			if err != nil {
				t.Fatalf("Forward (-kv %s) prompt %d pos %d: %v", kv, pi, pos, err)
			}
			rows[i] = append([]float32(nil), lg...)
		}
		return rows
	}
	for pi := range prompts {
		res.rows = append(res.rows, one(pi, survey))
		fmt.Printf("[kv16-gate] %s arm -kv %s prompt %d/%d done, elapsed %s\n", tag, kv, pi+1, len(prompts), time.Since(t0).Round(time.Second))
		if pi == 0 && aa {
			res.aa = one(0, false)
			fmt.Printf("[kv16-gate] %s arm -kv %s A/A rerun of prompt 1 done, elapsed %s\n", tag, kv, time.Since(t0).Round(time.Second))
		}
	}
	res.faLaunches = rf.faLaunches
	return res
}

func runKV16GateCell(t *testing.T, path, model string, K int, promptFiles []string, refDir string, decision bool, opts decoder.Options) {
	t.Helper()
	// A checkpoint directory carries tokenizer.json, as the CPU reference read it (decoder's loadRefTokenizer); a GGUF carries its own.
	var tk *tokenizer.Tokenizer
	var err error
	if st, e := os.Stat(path); e == nil && st.IsDir() {
		tk, err = tokenizer.Load(filepath.Join(path, "tokenizer.json"))
	} else {
		tk, err = tokenizer.LoadGGUF(path)
	}
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	prompts := make([][]int, len(promptFiles))
	refTokens := make([][]int, len(promptFiles))
	refLogits := make([][][]float32, len(promptFiles))
	for pi, f := range promptFiles {
		prompts[pi] = decoder.PrefillGateProseIDsForTest(t, tk, f, K)
		ref := filepath.Join(refDir, fmt.Sprintf("%s-K%d-p%d.bin", model, K, pi))
		if err := decoder.PrefillRefCurrentForTest(ref, path, prompts[pi][:K]); err != nil { // the arms read the first K, as the generator keyed them
			t.Fatalf("reference %d: %v", pi, err)
		}
		_, rt, rl, err := decoder.ReadPrefillReferenceForTest(ref)
		if err != nil {
			t.Fatalf("read reference %d: %v", pi, err)
		}
		refTokens[pi], refLogits[pi] = rt, rl
	}
	contN := len(refTokens[0])

	e := kv16RunArm(t, path, "f32", K, prompts, refTokens, true, false, model, opts)
	f := kv16RunArm(t, path, "f16", K, prompts, refTokens, false, true, model, opts)

	// PRECONDITION 4: the overflow survey, printed before any score.
	worst := float32(0)
	for l := range e.maxK {
		worst = max(worst, e.maxK[l], e.maxV[l])
	}
	fmt.Printf("=== KV16 OVERFLOW SURVEY %s K=%d (real K/V of %d gate prompts, %d layers; limit 32752 = f16 max / 2): max|K| %v\n    max|V| %v\n    worst %.3f\n",
		model, K, len(prompts), e.layers, e.maxK, e.maxV, worst)
	if worst >= 65504/2 {
		t.Errorf("PRECONDITION 4 (overflow): max|K or V| = %v is within 2x of f16's range; this family is refused f16 whatever the gate says", worst)
	}
	// PRECONDITION 1: vacuity.
	if !f.f16 || e.f16 {
		t.Fatalf("PRECONDITION 1: arms are not f32/f16 (f32 arm kvF16=%v, f16 arm kvF16=%v)", e.f16, f.f16)
	}
	if f.allocated <= 0 || e.allocated != 2*f.allocated || e.allocated-f.allocated != e.kvPool-f.kvPool {
		t.Fatalf("PRECONDITION 1: K/V cache allocations f32 %d B, f16 %d B, priced difference %d B: the f16 pools are not what was allocated",
			e.allocated, f.allocated, e.kvPool-f.kvPool)
	}
	// An MLA model has no layer the lane takes, so its vacuity rests on the rows below differing; any other model with none would
	// decode its f16 cache through no lane at all, and the gate would grade nothing of it.
	switch want := f.faLayers * (contN - 1) * len(prompts); {
	case f.faLayers == 0 && !f.mlaOnly:
		t.Fatalf("PRECONDITION 1 (vacuity): no layer takes the flash-decode lane at K=%d, so the f16 arm reads its cache through no lane", K)
	case f.faLaunches < want:
		t.Fatalf("PRECONDITION 1 (vacuity): the f16 arm launched fa_partial %d times, expected >= %d", f.faLaunches, want)
	}
	// PRECONDITION 2: A/A.
	for i := range f.aa {
		if !sameLogits(f.aa[i], f.rows[0][i]) {
			t.Fatalf("PRECONDITION 2: A/A FAILED at row %d: the f16 arm is not deterministic", i)
		}
	}
	t.Logf("%s K=%d: A/A bit-identical over %d rows", model, K, len(f.aa))

	var sumEA, sumFA, sumEKL, sumFKL, worstEGap, worstFGap float64
	var eHF, fHF, wins, diffPositions, seedDiff int
	var sumDelta, sumDelta2 float64
	n := len(prompts)
	for pi := range prompts {
		nd := 0
		for i := 1; i < contN; i++ {
			if !sameLogits(e.rows[pi][i], f.rows[pi][i]) {
				nd++
			}
		}
		if nd == 0 {
			t.Fatalf("PRECONDITION 1 (vacuity): prompt %d's f16 decode rows are bit-identical to f32's at every position", pi)
		}
		diffPositions += nd
		if !sameLogits(e.rows[pi][0], f.rows[pi][0]) {
			seedDiff++
		}
		ea, _ := decoder.TeacherForcedTop1AgreementForTest(e.rows[pi], refTokens[pi])
		fa, _ := decoder.TeacherForcedTop1AgreementForTest(f.rows[pi], refTokens[pi])
		es := scoreVsumArm(refLogits[pi], e.rows[pi])
		fs := scoreVsumArm(refLogits[pi], f.rows[pi])
		sumEA += ea
		sumFA += fa
		sumEKL += es.kl
		sumFKL += fs.kl
		eHF += es.hardFail
		fHF += fs.hardFail
		worstEGap = max(worstEGap, es.worstGap)
		worstFGap = max(worstFGap, fs.worstGap)
		if fa >= ea {
			wins++
		}
		d := fs.kl - es.kl
		sumDelta += d
		sumDelta2 += d * d
		fmt.Printf("[kv16-gate] %s K=%d prompt %2d/%2d f32(agree=%.1f%% HF=%d KL=%.5f) f16(agree=%.1f%% HF=%d KL=%.5f) diff(agree=%+.2fpt KL=%+.5f) rowsDiffering=%d/%d\n",
			model, K, pi+1, n, ea*100, es.hardFail, es.kl, fa*100, fs.hardFail, fs.kl, (fa-ea)*100, d, nd, contN-1)
	}
	mEA, mFA := sumEA/float64(n)*100, sumFA/float64(n)*100
	mEKL, mFKL := sumEKL/float64(n), sumFKL/float64(n)
	meanD := sumDelta / float64(n)
	se := math.Sqrt(math.Max(sumDelta2/float64(n)-meanD*meanD, 0) / float64(n-1))
	critA := float64(fHF) <= float64(eHF)+2*math.Sqrt(float64(eHF))
	critB := mFA >= mEA-1.0 && wins*2 >= n
	critC := mFKL <= 1.1*mEKL
	parked := (critB && mFA < mEA-0.8) || (critC && mFKL > 1.05*mEKL)
	verdict := "DOES NOT PASS"
	switch {
	case critA && critB && critC && parked:
		verdict = "AMBIGUOUS — PARKED"
	case critA && critB && critC:
		verdict = "PASSES"
	}
	kind := "confirmation"
	if decision {
		kind = "DECISION"
	}
	flag := ""
	if critC && se > 0 && meanD > 2*se {
		flag = fmt.Sprintf(" (KL paired delta %+.6f is %.1f s.e. above zero, s.e. %.6f)", meanD, meanD/se, se)
	}
	fmt.Printf("=== KV16 GATE %s K=%d (%s cell, set b, n=%d x %d positions): f32(meanAgree=%.2f%% HF=%d worstGap=%.3f%% meanKL=%.6f) f16(meanAgree=%.2f%% HF=%d worstGap=%.3f%% meanKL=%.6f) "+
		"f16>=f32 on %d/%d prompts, rowsDiffering=%d, seedRowsDiffering=%d/%d; critA=%v critB=%v critC=%v KLratio=%.4f%s => %s\n",
		model, K, kind, n, contN, mEA, eHF, worstEGap*100, mEKL, mFA, fHF, worstFGap*100, mFKL, wins, n, diffPositions, seedDiff, n, critA, critB, critC, mFKL/mEKL, flag, verdict)
	if decision && verdict != "PASSES" {
		t.Errorf("%s K=%d is the DECISION cell and it did not pass (%s): f16 stays opt-in, which is a RESULT", model, K, verdict)
	}
}

// TestKV8GateVsReference is S4's G4 for the CUDA int8 cache (docs/measurements/cuda-kv-i8-2026-11/s4-registration.md): three arms, f32 (exact),
// f16 and i8, loaded one after another, teacher-forced and scored against the CPU f32/f64 reference of held-out prompt set B, with the f16 gate's
// criteria (a) / (b) / (c), int8 against f32, and the same three numbers for int8 against f16 beside them. G4 is a published reading, not a
// refusal: the verdict is printed, and only a failed precondition fails the test.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_PREFILL_GATE_PROMPTS=b GOINFER_CUDA_FLASH_DECODE=16 \
//	  go test -tags 'cuda goinfer_testhooks' ./backend/cuda/ -run TestKV8GateVsReference -v -timeout 2h
//
// Cells: D7 K=8000 (decision), S (1.5B) K=3900 (confirmation); GOINFER_KV8_GATE_CELLS="S:3900:2" and the like override, as the f16 gate's do.
func TestKV8GateVsReference(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (real checkpoints; needs the set-B reference files)")
	}
	if s, _ := strconv.Atoi(os.Getenv("GOINFER_CUDA_FLASH_DECODE")); s < 1 {
		t.Skip("set GOINFER_CUDA_FLASH_DECODE=16: every arm runs the production decode lane")
	}
	label, promptFiles := decoder.PrefillGatePromptSet()
	if label != "b" {
		t.Fatalf("the registered gate runs on held-out set B; GOINFER_PREFILL_GATE_PROMPTS=%q selects set %q", os.Getenv("GOINFER_PREFILL_GATE_PROMPTS"), label)
	}
	home, _ := os.UserHomeDir()
	refDir := filepath.Join(home, "goinfer-logs", "prefill-ref-b")
	cells := os.Getenv("GOINFER_KV8_GATE_CELLS")
	if cells == "" {
		cells = "S:3900,D7:8000"
	}
	// PRECONDITION 4: the exact-input G2 at this binary's commit, before any cell: the int8 readers this gate scores must match the CPU's on
	// one cache. It runs here, in the same binary, so the gate cannot be read against readers that were never checked.
	if !t.Run("precondition4-exact-input-G2", TestKV8_G2exact_readersMatchTheCPUOnOneCache) {
		t.Fatal("PRECONDITION 4 (exact-input G2) failed: no cell is read")
	}
	models := map[string]struct {
		def      string
		decision bool
	}{
		"D7": {"$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf", true},
		"S":  {"$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", false},
	}
	for cell := range strings.SplitSeq(cells, ",") {
		parts := strings.Split(strings.TrimSpace(cell), ":")
		if len(parts) != 2 && len(parts) != 3 {
			t.Fatalf("bad cell %q", cell)
		}
		name := parts[0]
		K, _ := strconv.Atoi(parts[1])
		files := promptFiles
		if len(parts) == 3 {
			n, _ := strconv.Atoi(parts[2])
			if n < 2 || n > len(promptFiles) {
				t.Fatalf("bad cell %q: n must be 2 to %d", cell, len(promptFiles))
			}
			files = promptFiles[:n]
		}
		mc, ok := models[name]
		if !ok || K < 1 {
			t.Fatalf("bad cell %q", cell)
		}
		t.Run(fmt.Sprintf("%s-K%d", name, K), func(t *testing.T) {
			path := os.ExpandEnv(mc.def)
			if _, err := os.Stat(path); err != nil {
				t.Fatalf("no checkpoint at %s: a registered cell must run, not skip", path)
			}
			for pi := range files {
				if _, err := os.Stat(filepath.Join(refDir, fmt.Sprintf("%s-K%d-p%d.bin", name, K, pi))); err != nil {
					t.Fatalf("%s K=%d: reference missing for prompt %d", name, K, pi)
				}
			}
			runKV8GateCell(t, path, name, K, files, refDir, mc.decision)
		})
	}
}

// kv8GateRead is one candidate arm scored against a base arm on a cell: the f16 gate's three criteria and the numbers behind them.
type kv8GateRead struct {
	baseAgree, candAgree, baseKL, candKL float64 // means; agreement in percent
	baseHF, candHF, wins, n              int
	critA, critB, critC, parked          bool
	klDelta, klSE                        float64
}

func (g kv8GateRead) verdict() string {
	switch {
	case g.critA && g.critB && g.critC && g.parked:
		return "AMBIGUOUS — PARKED"
	case g.critA && g.critB && g.critC:
		return "PASSES"
	}
	return "DOES NOT PASS"
}

// kv8Score scores cand against base over every prompt (agreement against the reference tokens, KL and hard flips against the reference rows).
func kv8Score(base, cand kv16ArmResult, refTokens [][]int, refLogits [][][]float32) kv8GateRead {
	var g kv8GateRead
	g.n = len(refTokens)
	var sumD, sumD2 float64
	for pi := range refTokens {
		ba, _ := decoder.TeacherForcedTop1AgreementForTest(base.rows[pi], refTokens[pi])
		ca, _ := decoder.TeacherForcedTop1AgreementForTest(cand.rows[pi], refTokens[pi])
		bs, cs := scoreVsumArm(refLogits[pi], base.rows[pi]), scoreVsumArm(refLogits[pi], cand.rows[pi])
		g.baseAgree += ba * 100 / float64(g.n)
		g.candAgree += ca * 100 / float64(g.n)
		g.baseKL += bs.kl / float64(g.n)
		g.candKL += cs.kl / float64(g.n)
		g.baseHF += bs.hardFail
		g.candHF += cs.hardFail
		if ca >= ba {
			g.wins++
		}
		d := cs.kl - bs.kl
		sumD += d
		sumD2 += d * d
	}
	g.klDelta = sumD / float64(g.n)
	g.klSE = math.Sqrt(math.Max(sumD2/float64(g.n)-g.klDelta*g.klDelta, 0) / float64(g.n-1))
	g.critA = float64(g.candHF) <= float64(g.baseHF)+2*math.Sqrt(float64(g.baseHF))
	g.critB = g.candAgree >= g.baseAgree-1.0 && g.wins*2 >= g.n
	g.critC = g.candKL <= 1.1*g.baseKL
	g.parked = (g.critB && g.candAgree < g.baseAgree-0.8) || (g.critC && g.candKL > 1.05*g.baseKL)
	return g
}

func runKV8GateCell(t *testing.T, path, model string, K int, promptFiles []string, refDir string, decision bool) {
	t.Helper()
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	prompts := make([][]int, len(promptFiles))
	refTokens := make([][]int, len(promptFiles))
	refLogits := make([][][]float32, len(promptFiles))
	for pi, f := range promptFiles {
		prompts[pi] = decoder.PrefillGateProseIDsForTest(t, tk, f, K)
		ref := filepath.Join(refDir, fmt.Sprintf("%s-K%d-p%d.bin", model, K, pi))
		if err := decoder.PrefillRefCurrentForTest(ref, path, prompts[pi][:K]); err != nil {
			t.Fatalf("reference %d: %v", pi, err)
		}
		_, rt, rl, err := decoder.ReadPrefillReferenceForTest(ref)
		if err != nil {
			t.Fatalf("read reference %d: %v", pi, err)
		}
		refTokens[pi], refLogits[pi] = rt, rl
	}
	contN := len(refTokens[0])
	e := kv16RunArm(t, path, "f32", K, prompts, refTokens, false, false, model, decoder.Options{})
	h := kv16RunArm(t, path, "f16", K, prompts, refTokens, false, false, model, decoder.Options{})
	q := kv16RunArm(t, path, "i8", K, prompts, refTokens, false, true, model, decoder.Options{})

	// PRECONDITION 5: every arm was sized by the build's own fit check (a refused load fails kv16RunArm before here).
	for _, a := range []kv16ArmResult{e, h, q} {
		fmt.Printf("=== KV8 GATE %s K=%d precondition 5 (fit): %s\n", model, K, a.fit)
	}
	// PRECONDITION 1: vacuity. Each arm holds exactly its precision: all six flags.
	if e.f16 || e.i8 || !h.f16 || h.i8 || q.f16 || !q.i8 {
		t.Fatalf("PRECONDITION 1: arms are not f32/f16/i8 (f16 flags %v %v %v; int8 flags %v %v %v)", e.f16, h.f16, q.f16, e.i8, h.i8, q.i8)
	}
	// Each arm allocated exactly the pools it prices (its tables aside, which allocated does not count); int8's are under half f32's and over
	// a quarter (1 B an element plus the scales), f16's exactly half.
	for _, a := range []struct {
		name string
		r    kv16ArmResult
	}{{"f32", e}, {"f16", h}, {"i8", q}} {
		if a.r.allocated <= 0 || a.r.allocated != a.r.kvPool-a.r.tables {
			t.Fatalf("PRECONDITION 1: the %s arm allocated %d B of K/V; its pools are priced %d B (%d B with %d B of tables)", a.name, a.r.allocated, a.r.kvPool-a.r.tables, a.r.kvPool, a.r.tables)
		}
	}
	// f16 is exactly half of f32. An int8 row is kvDim bytes plus one f32 scale per KV head (kvRowBytes) against f32's 4*kvDim, so on a model
	// whose KV layers share one geometry int8 is exactly f32/4 + f32*nKV/kvDim: 4*i8*kvDim == f32*kvDim + 4*f32*nKV.
	if !e.uniform || e.kvDim <= 0 {
		t.Fatalf("PRECONDITION 1: the model's KV layers do not share one geometry, so the int8 overhead has no single exact figure")
	}
	if 2*h.allocated != e.allocated || 4*q.allocated*int64(e.kvDim) != e.allocated*int64(e.kvDim)+4*e.allocated*int64(e.nKV) {
		t.Fatalf("PRECONDITION 1: K/V allocations f32 %d B, f16 %d B, i8 %d B: f16 is not half, or i8 is not f32/4 plus one f32 scale per KV head and row (want %d B)",
			e.allocated, h.allocated, q.allocated, e.allocated/4+e.allocated*int64(e.nKV)/int64(e.kvDim))
	}
	// Every arm reads its cache through the production flash-decode lane, compiled from its own precision's build of decode_fa.
	for _, a := range []struct {
		name, module string
		r            kv16ArmResult
	}{{"f32", "decode_fa", e}, {"f16", "decode_fa_kv16", h}, {"i8", "decode_fa_kv8", q}} {
		if a.r.faModule != a.module {
			t.Fatalf("PRECONDITION 1 (vacuity): the %s arm's flash-decode lane was compiled from %q, not %q", a.name, a.r.faModule, a.module)
		}
		if want := a.r.faLayers * (contN - 1) * len(prompts); a.r.faLayers == 0 || a.r.faLaunches < want {
			t.Fatalf("PRECONDITION 1 (vacuity): the %s arm launched fa_partial %d times over %d lane layers, expected >= %d", a.name, a.r.faLaunches, a.r.faLayers, want)
		}
	}
	diff, seedDiff := 0, 0
	for pi := range prompts {
		nd := 0
		for i := 1; i < contN; i++ {
			if !sameLogits(e.rows[pi][i], q.rows[pi][i]) {
				nd++
			}
		}
		if nd == 0 {
			t.Fatalf("PRECONDITION 1 (vacuity): prompt %d's int8 decode rows are bit-identical to f32's at every position", pi)
		}
		diff += nd
		if !sameLogits(e.rows[pi][0], q.rows[pi][0]) {
			seedDiff++
		}
	}
	// PRECONDITION 2: A/A.
	for i := range q.aa {
		if !sameLogits(q.aa[i], q.rows[0][i]) {
			t.Fatalf("PRECONDITION 2: A/A FAILED at row %d: the int8 arm is not deterministic", i)
		}
	}
	kind := "confirmation"
	if decision {
		kind = "DECISION"
	}
	for _, c := range []struct {
		name       string
		base, cand kv16ArmResult
	}{{"i8 vs f32 (G4)", e, q}, {"i8 vs f16 (reported)", h, q}, {"f16 vs f32 (control)", e, h}} {
		g := kv8Score(c.base, c.cand, refTokens, refLogits)
		flag := ""
		if g.critC && g.klSE > 0 && g.klDelta > 2*g.klSE {
			flag = fmt.Sprintf(" (KL paired delta %+.6f is %.1f s.e. above zero, s.e. %.6f)", g.klDelta, g.klDelta/g.klSE, g.klSE)
		}
		fmt.Printf("=== KV8 GATE %s K=%d (%s cell, set b, n=%d x %d positions) %s: base(meanAgree=%.2f%% HF=%d meanKL=%.6f) cand(meanAgree=%.2f%% HF=%d meanKL=%.6f) "+
			"cand>=base on %d/%d prompts; critA=%v critB=%v critC=%v KLratio=%.4f%s => %s\n",
			model, K, kind, g.n, contN, c.name, g.baseAgree, g.baseHF, g.baseKL, g.candAgree, g.candHF, g.candKL, g.wins, g.n, g.critA, g.critB, g.critC,
			g.candKL/g.baseKL, flag, g.verdict())
	}
	fmt.Printf("=== KV8 GATE %s K=%d preconditions: int8 allocated %d B (f32 %d B), A/A bit-identical over %d rows, int8 rows differing from f32 %d, seed rows differing %d/%d\n",
		model, K, q.allocated, e.allocated, len(q.aa), diff, seedDiff, len(prompts))
}
