//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestFlashDecodeOracleRealKV is precondition 4 of docs/measurements/attn-decode-fa-fidelity-PREREGISTERED.md, the new
// kernel's defect detector (S=1 identity is unavailable because online softmax is a different reduction): on the REAL
// K/V of a held-out prompt prefilled to K=8000, at EVERY layer, the lane's attention output must be no further from an
// f64 recompute than the exact path's own output is (median and max over heads, within 1.0x of exact's error plus an
// absolute allowance of 1e-7 of max|ref|).
//
// The K/V are real (downloaded from the resident's own per-layer cache after a real prefill); the QUERIES are not — the
// per-layer roped query at a decode step is not exposed without hooking the forward, so each head's q is a real key row
// of that layer, drawn at a head-dependent position. That gives realistic magnitudes and outlier dimensions but a
// different score distribution from a true query; the record states this.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_PREFILL_GATE_PROMPTS=b GOINFER_CUDA_FLASH_DECODE=16 GOINFER_CUDA_FLASH_DECODE_MIN_KEYS=0 \
//	  go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestFlashDecodeOracleRealKV -v -timeout 1h
func TestFlashDecodeOracleRealKV(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (D7 checkpoint, K=8000 prefill)")
	}
	if s, _ := strconv.Atoi(os.Getenv("GOINFER_CUDA_FLASH_DECODE")); s < 1 {
		t.Skip("set GOINFER_CUDA_FLASH_DECODE=16")
	}
	const K = 8000
	path := os.Getenv("GOINFER_CUDA_GATE_MODEL_D7")
	if path == "" {
		path = os.ExpandEnv("$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4", ResidentContext: K + 128})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	rf, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok || rf.faSplit < 1 {
		t.Fatalf("resident/lane not active (ok=%v faSplit=%d)", ok, func() int {
			if rf != nil {
				return rf.faSplit
			}
			return -1
		}())
	}
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	label, files := decoder.PrefillGatePromptSet()
	ids := decoder.PrefillGateProseIDsForTest(t, tk, files[0], K)[:K]
	embs := make([][]float32, K)
	for i, id := range ids {
		embs[i] = m.EmbedResidentForTest(id)
	}
	if _, err := rf.PrefillLast(context.Background(), embs, 0); err != nil {
		t.Fatalf("prefill: %v", err)
	}
	t.Logf("prompt set %q file %s prefilled to K=%d; S=%d", label, filepath.Base(files[0]), K, rf.faSplit)

	pos := K - 1
	failures := 0
	var lastMedLane, lastMedExact float64
	for l := range rf.layers {
		Ly := &rf.layers[l]
		if !rf.faEligible(l) {
			t.Logf("layer %d: lane declines (not eligible)", l)
			continue
		}
		hd, nKV, nH := Ly.hd, Ly.nKV, rf.nH
		kvDim := nKV * hd
		nKeys := pos + 1
		winStart := 0
		if Ly.window > 0 && nKeys > int(Ly.window) {
			winStart = nKeys - int(Ly.window)
		}
		k := make([]float32, nKeys*kvDim)
		v := make([]float32, nKeys*kvDim)
		exact := make([]float32, nH*hd)
		lane := make([]float32, nH*hd)
		q := make([]float32, nH*hd)
		G := nH / nKV
		err := rf.do(func() error {
			if e := rf.stream.Sync(); e != nil {
				return e
			}
			if e := gpu.Download(rf.kc[l], k); e != nil {
				return e
			}
			if e := gpu.Download(rf.vc[l], v); e != nil {
				return e
			}
			for h := 0; h < nH; h++ {
				p := (h*997 + 13) % nKeys
				copy(q[h*hd:(h+1)*hd], k[p*kvDim+(h/G)*hd:p*kvDim+(h/G)*hd+hd])
			}
			if e := gpu.Upload(rf.qB, q); e != nil {
				return e
			}
			if e := rf.splitKVAttnDecode(l, pos); e != nil {
				return e
			}
			if e := rf.stream.Sync(); e != nil {
				return e
			}
			if e := gpu.Download(rf.cctx, exact); e != nil {
				return e
			}
			if e := rf.flashDecodeAttn(l, pos); e != nil {
				return e
			}
			if e := rf.stream.Sync(); e != nil {
				return e
			}
			return gpu.Download(rf.cctx, lane)
		})
		if err != nil {
			t.Fatalf("layer %d: %v", l, err)
		}
		ref := flashRef(q, k, v, nH, nKV, hd, winStart, nKeys, rf.attnScale)
		var refMax float64
		for _, x := range ref {
			refMax = math.Max(refMax, math.Abs(x))
		}
		perHead := func(got []float32) (med, mx float64) {
			es := make([]float64, nH)
			for h := 0; h < nH; h++ {
				for i := 0; i < hd; i++ {
					es[h] = math.Max(es[h], math.Abs(float64(got[h*hd+i])-ref[h*hd+i]))
				}
				es[h] /= math.Max(refMax, 1e-12)
			}
			sort.Float64s(es)
			return es[nH/2], es[nH-1]
		}
		eMed, eMax := perHead(exact)
		lMed, lMax := perHead(lane)
		const allowance = 1e-7
		bad := lMed > eMed+allowance || lMax > eMax+allowance
		if bad {
			failures++
		}
		lastMedLane, lastMedExact = lMed, eMed
		t.Logf("layer %2d: exact err median %.3g max %.3g | lane err median %.3g max %.3g | %s", l, eMed, eMax, lMed, lMax, map[bool]string{true: "LANE WORSE", false: "ok"}[bad])
	}
	_ = lastMedLane
	_ = lastMedExact
	if failures > 0 {
		t.Errorf("precondition 4 FAILED at %d layers: the lane is further from the f64 recompute than the exact path", failures)
	}
}

// TestFlashDecodeGateVsReference is the registered fidelity gate for the lane (attn-decode-fa-fidelity-PREREGISTERED.md):
// exact vs lane, both scored against the CPU f32/f64 reference of held-out prompt set B, criteria (a) amended / (b) / (c),
// preconditions 1-3 (vacuity by launch count, A/A determinism, seed row identical).
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_PREFILL_GATE_PROMPTS=b GOINFER_CUDA_FLASH_DECODE=16 GOINFER_CUDA_FLASH_DECODE_MIN_KEYS=0 \
//	  go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestFlashDecodeGateVsReference -v -timeout 3h
//
// Cells: D7 K=8000 (decision), S (1.5B) K=3900 (confirmation); override with GOINFER_FA_GATE_CELLS="D7:8000,S:3900".
func TestFlashDecodeGateVsReference(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (real checkpoints; needs Phase A's set-B reference files)")
	}
	nSplit, _ := strconv.Atoi(os.Getenv("GOINFER_CUDA_FLASH_DECODE"))
	if nSplit < 1 {
		t.Skip("set GOINFER_CUDA_FLASH_DECODE=16 (the registered S)")
	}
	label, promptFiles := decoder.PrefillGatePromptSet()
	if label != "b" {
		t.Fatalf("the registered gate runs on held-out set B; GOINFER_PREFILL_GATE_PROMPTS=%q selects set %q", os.Getenv("GOINFER_PREFILL_GATE_PROMPTS"), label)
	}
	home, _ := os.UserHomeDir()
	refDir := filepath.Join(home, "goinfer-logs", "prefill-ref-b")
	cells := os.Getenv("GOINFER_FA_GATE_CELLS")
	if cells == "" {
		cells = "D7:8000,S:3900"
	}
	models := map[string]struct {
		env, def string
		decision bool
	}{
		"D7": {"GOINFER_CUDA_GATE_MODEL_D7", "$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf", true},
		"S":  {"GOINFER_CUDA_GATE_MODEL", "$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", false},
	}
	for _, cell := range strings.Split(cells, ",") {
		parts := strings.Split(strings.TrimSpace(cell), ":")
		if len(parts) != 2 {
			t.Fatalf("bad cell %q", cell)
		}
		name, K := parts[0], 0
		K, _ = strconv.Atoi(parts[1])
		mc, ok := models[name]
		if !ok || K < 1 {
			t.Fatalf("bad cell %q", cell)
		}
		t.Run(fmt.Sprintf("%s-K%d", name, K), func(t *testing.T) {
			path := os.Getenv(mc.env)
			if path == "" {
				path = os.ExpandEnv(mc.def)
			}
			if _, err := os.Stat(path); err != nil {
				t.Skipf("no fixture at %s", path)
			}
			m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4", ResidentContext: K + 128})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer m.Close()
			rf, ok := m.ResidentForwardForTest().(*cudaResident)
			if !ok {
				t.Fatal("cuda resident did not engage")
			}
			if rf.faSplit != nSplit || rf.faCombine == (Pipeline{}) || !rf.splitkvAttn {
				t.Fatalf("lane not loaded (faSplit=%d want %d)", rf.faSplit, nSplit)
			}
			rf.skMinKeys, rf.faMinKeys = 0, 0
			tk, err := tokenizer.LoadGGUF(path)
			if err != nil {
				t.Fatalf("tokenizer: %v", err)
			}
			prompts := make([][]int, 0, len(promptFiles))
			for _, f := range promptFiles {
				prompts = append(prompts, decoder.PrefillGateProseIDsForTest(t, tk, f, K))
			}
			runFlashGateCell(t, rf, m, name, K, nSplit, prompts, refDir, mc.decision)
		})
	}
}

func laneArm(t *testing.T, rf *cudaResident, m *decoder.Model, ids []int, K int, refTokens []int, split int) [][]float32 {
	t.Helper()
	ctx := context.Background()
	embs := make([][]float32, K)
	for i, id := range ids {
		embs[i] = m.EmbedResidentForTest(id)
	}
	rf.faSplit, rf.skVsumSplit = split, 0
	rf.faLaunches = 0
	seed, err := rf.PrefillLast(ctx, embs, 0)
	if err != nil {
		t.Fatalf("PrefillLast (lane split=%d): %v", split, err)
	}
	out := make([][]float32, len(refTokens))
	out[0] = append([]float32(nil), seed...)
	pos := K - 1
	for i := 1; i < len(refTokens); i++ {
		pos++
		lg, err := rf.Forward(m.EmbedResidentForTest(refTokens[i-1]), pos)
		if err != nil {
			t.Fatalf("teacher-forced Forward pos=%d (lane split=%d): %v", pos, split, err)
		}
		out[i] = append([]float32(nil), lg...)
	}
	return out
}

func runFlashGateCell(t *testing.T, rf *cudaResident, m *decoder.Model, model string, K, nSplit int,
	prompts [][]int, refDir string, decision bool) {
	t.Helper()
	for pi := range prompts {
		p := filepath.Join(refDir, fmt.Sprintf("%s-K%d-p%d.bin", model, K, pi))
		if _, err := os.Stat(p); err != nil {
			t.Skipf("%s K=%d: reference missing (%s) — run Phase A first", model, K, p)
		}
	}
	var (
		sumEA, sumFA, sumEKL, sumFKL float64
		eHF, fHF, n, wins            int
		worstEGap, worstFGap         float64
		diffPositions, contN         int
	)
	layers := len(rf.layers)
	t0 := time.Now()
	for pi, ids := range prompts {
		_, refTokens, refLogits, err := decoder.ReadPrefillReferenceForTest(filepath.Join(refDir, fmt.Sprintf("%s-K%d-p%d.bin", model, K, pi)))
		if err != nil {
			t.Fatalf("read reference: %v", err)
		}
		contN = len(refTokens)
		exactCont := laneArm(t, rf, m, ids[:K], K, refTokens, 0)
		if rf.faLaunches != 0 {
			t.Fatalf("PRECONDITION 1: the exact arm launched the lane %d times", rf.faLaunches)
		}
		laneCont := laneArm(t, rf, m, ids[:K], K, refTokens, nSplit)
		// PRECONDITION 1: the lane must have run, by launch count (one fa_partial per eligible layer per decode row).
		if want := layers * (contN - 1); rf.faLaunches < want {
			t.Fatalf("PRECONDITION 1 (vacuity): the lane arm launched fa_partial %d times, expected >= %d (%d layers x %d decode rows)", rf.faLaunches, want, layers, contN-1)
		}
		// PRECONDITION 3a: the prefill seed row is identical across arms (the lane touches M=1 decode attention only).
		if !sameLogits(exactCont[0], laneCont[0]) {
			t.Fatalf("PRECONDITION 3: prefill seed row differs between arms on prompt %d", pi)
		}
		nd := 0
		for i := 1; i < len(laneCont); i++ {
			if !sameLogits(exactCont[i], laneCont[i]) {
				nd++
			}
		}
		diffPositions += nd
		// PRECONDITION 2 (A/A determinism), first prompt only.
		if pi == 0 {
			again := laneArm(t, rf, m, ids[:K], K, refTokens, nSplit)
			for i := range again {
				if !sameLogits(again[i], laneCont[i]) {
					t.Fatalf("PRECONDITION 2: A/A FAILED at row %d — the lane is not deterministic", i)
				}
			}
			t.Logf("%s K=%d: A/A bit-identical over %d rows", model, K, len(again))
		}
		ea, _ := decoder.TeacherForcedTop1AgreementForTest(exactCont, refTokens)
		fa, _ := decoder.TeacherForcedTop1AgreementForTest(laneCont, refTokens)
		e := scoreVsumArm(refLogits, exactCont)
		s := scoreVsumArm(refLogits, laneCont)
		n++
		sumEA += ea
		sumFA += fa
		sumEKL += e.kl
		sumFKL += s.kl
		eHF += e.hardFail
		fHF += s.hardFail
		worstEGap = max(worstEGap, e.worstGap)
		worstFGap = max(worstFGap, s.worstGap)
		if fa >= ea {
			wins++
		}
		fmt.Printf("[fa-gate] %s K=%d prompt %2d/%2d exact(agree=%.1f%% HF=%d/%d KL=%.5f) lane(agree=%.1f%% HF=%d/%d KL=%.5f) diff(agree=%+.2fpt KL=%+.5f) rowsDiffering=%d/%d elapsed=%s\n",
			model, K, pi+1, len(prompts), ea*100, e.hardFail, contN, e.kl, fa*100, s.hardFail, contN, s.kl, (fa-ea)*100, s.kl-e.kl, nd, contN-1, time.Since(t0).Round(time.Second))
	}
	if diffPositions == 0 {
		t.Fatalf("%s K=%d: VACUOUS — both arms produced bit-identical decode logits everywhere", model, K)
	}
	mEA, mFA := sumEA/float64(n)*100, sumFA/float64(n)*100
	mEKL, mFKL := sumEKL/float64(n), sumFKL/float64(n)
	critAStrict := fHF <= eHF
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
	fmt.Printf("=== FLASH-DECODE GATE %s K=%d (%s cell, set b, n=%d x %d positions, S=%d): exact(meanAgree=%.2f%% HF=%d/%d worstGap=%.3f%% meanKL=%.6f) lane(meanAgree=%.2f%% HF=%d/%d worstGap=%.3f%% meanKL=%.6f) pairedWins=%d/%d rowsDiffering=%d critA(amended)=%v [strict %v] critB=%v critC=%v KLratio=%.4f — %s ===\n",
		model, K, kind, n, contN, nSplit, mEA, eHF, n*contN, worstEGap*100, mEKL, mFA, fHF, n*contN, worstFGap*100, mFKL, wins, n, diffPositions, critA, critAStrict, critB, critC, mFKL/mEKL, verdict)
	if decision && verdict != "PASSES" {
		t.Errorf("%s K=%d is the DECISION cell and it did not pass (%s) — the lane stays opt-in, which is a RESULT", model, K, verdict)
	}
}
