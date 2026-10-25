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
	kvBytes    int           // layer-0 K pool bytes
	faLaunches int
	f16        bool
	aa         [][]float32 // a second run of prompt 0 (the f16 arm only), for the A/A precondition
	maxK, maxV []float32   // per layer, over every prompt (the f32 arm only)
	layers     int
	faLayers   int  // the layers whose attended span reaches the lane's floor at this depth (a sliding window under it never does)
	mlaOnly    bool // every layer that keeps K/V is MLA's latent, which the flash-decode lane never reads
}

func kv16RunArm(t *testing.T, path, kv string, K int, prompts [][]int, refTokens [][]int, survey, aa bool, tag string, opts decoder.Options) kv16ArmResult {
	t.Helper()
	opts.Backend, opts.Quant, opts.KVPrecision, opts.ResidentContext = "cuda", "int4", kv, K+128
	m, err := decoder.Load(path, opts)
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
	kl := 0 // the first layer that keeps K/V (a hybrid's layer 0 may be recurrent)
	for kl < len(rf.layers)-1 && (rf.layers[kl].noKV() || rf.layers[kl].kvShared) {
		kl++
	}
	res := kv16ArmResult{f16: rf.kvF16, kvBytes: rf.kc[kl].Len() * map[bool]int{true: 2, false: 4}[rf.layers[kl].kvHalf], layers: len(rf.layers)}
	res.mlaOnly = true
	for l := range rf.layers {
		if !rf.layers[l].noKV() && !rf.layers[l].isMLA() {
			res.mlaOnly = false
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
		_, rt, rl, err := decoder.ReadPrefillReferenceForTest(filepath.Join(refDir, fmt.Sprintf("%s-K%d-p%d.bin", model, K, pi)))
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
	if f.kvBytes*2 != e.kvBytes {
		t.Fatalf("PRECONDITION 1: the f16 arm's K buffer is %d B against the f32 arm's %d B, not half", f.kvBytes, e.kvBytes)
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
