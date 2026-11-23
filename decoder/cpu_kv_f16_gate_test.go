//go:build goinfer_testhooks

package decoder

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

	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestCPUKVF16GateVsReference is perf finding 12's fidelity gate (docs/measurements/cpu-kv-f16-2026-11-23/README.md): a CPU load at the
// model's default quant with an f32 KV cache (auto's) and with an f16 one, each teacher-forced on held-out set B's reference tokens and
// scored against the CPU f32 reference with the KV gates' criteria. Cells from CPU_KV_GATE_CELLS (default "P3M:3900,Q05:3900").
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_PREFILL_GATE_PROMPTS=b go test -tags goinfer_testhooks -run '^TestCPUKVF16GateVsReference$' -v -timeout 2h ./decoder/
func TestCPUKVF16GateVsReference(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (real checkpoints and set B's reference files)")
	}
	label, promptFiles := PrefillGatePromptSet()
	if label != "b" {
		t.Fatalf("the registered gate runs on held-out set B; GOINFER_PREFILL_GATE_PROMPTS selects set %q", label)
	}
	home, _ := os.UserHomeDir()
	refDir := filepath.Join(home, "goinfer-logs", "prefill-ref-b")
	cells := os.Getenv("CPU_KV_GATE_CELLS")
	if cells == "" {
		cells = "P3M:3900,Q05:3900"
	}
	models := map[string]struct{ path, quant string }{
		"P3M": {"$HOME/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf", "q4k"},
		"Q05": {"$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf", "int4"},
	}
	for cell := range strings.SplitSeq(cells, ",") {
		parts := strings.Split(strings.TrimSpace(cell), ":")
		K, _ := strconv.Atoi(parts[len(parts)-1])
		mc, ok := models[parts[0]]
		if len(parts) != 2 || !ok || K < 1 {
			t.Fatalf("bad cell %q", cell)
		}
		name := parts[0]
		t.Run(fmt.Sprintf("%s-K%d", name, K), func(t *testing.T) {
			path := os.ExpandEnv(mc.path)
			if _, err := os.Stat(path); err != nil {
				t.Skipf("no checkpoint at %s", path)
			}
			for pi := range promptFiles {
				if _, err := os.Stat(filepath.Join(refDir, fmt.Sprintf("%s-K%d-p%d.bin", name, K, pi))); err != nil {
					t.Fatalf("%s K=%d: reference p%d missing: build it with TestPrefillGateReference first", name, K, pi)
				}
			}
			tk, err := tokenizer.LoadGGUF(path)
			if err != nil {
				t.Fatal(err)
			}
			load := func(kv string) *Model {
				m, err := Load(path, Options{Backend: "cpu", Quant: mc.quant, KVQuant: kv})
				if err != nil {
					t.Fatalf("load (kv %q): %v", kv, err)
				}
				return m
			}
			f32, f16 := load(""), load("f16")
			defer f32.Close()
			defer f16.Close()
			cpuKVGateCell(t, name, K, tk, promptFiles, refDir, f32, f16)
		})
	}
}

// cpuKVArm prefills ids through the production CPU prefill and teacher-forces refTokens: row 0 is the prompt's last logits, row i the
// logits after feeding refTokens[i-1]. It returns the rows and the cache, whose precision and bytes the caller checks.
func cpuKVArm(t *testing.T, m *Model, ids, refTokens []int) ([][]float32, *KVCache) {
	t.Helper()
	cache := m.NewCache(len(ids) + len(refTokens) + 8)
	seed, err := m.PrefillLogitsForTest(context.Background(), ids, cache)
	if err != nil {
		t.Fatalf("prefill: %v", err)
	}
	out := [][]float32{append([]float32(nil), seed...)}
	for i := 1; i < len(refTokens); i++ {
		lg, err := m.forward(refTokens[i-1], cache)
		if err != nil {
			t.Fatalf("teacher-forced forward %d: %v", i, err)
		}
		out = append(out, append([]float32(nil), lg...))
	}
	return out, cache
}

// cpuKVBytes is the K/V bytes a cache holds: f32 rows at 4 bytes an element, f16 rows at 2.
func cpuKVBytes(c *KVCache) int64 {
	var n int64
	for l := range c.keys {
		n += 4 * int64(len(c.keys[l])+len(c.vals[l]))
	}
	for l := range c.keysH {
		n += 2 * int64(len(c.keysH[l])+len(c.valsH[l]))
	}
	return n
}

func cpuKVGateCell(t *testing.T, name string, K int, tk *tokenizer.Tokenizer, promptFiles []string, refDir string, f32, f16 *Model) {
	t.Helper()
	var sumEA, sumFA, sumEKL, sumFKL float64
	var eHF, fHF, n, wins, diffPrompts int
	t0 := time.Now()
	for pi, f := range promptFiles {
		_, refTokens, refLogits, err := ReadPrefillReferenceForTest(filepath.Join(refDir, fmt.Sprintf("%s-K%d-p%d.bin", name, K, pi)))
		if err != nil {
			t.Fatalf("read reference: %v", err)
		}
		ids := PrefillGateProseIDsForTest(t, tk, f, K)[:K]
		exact, ce := cpuKVArm(t, f32, ids, refTokens)
		half, ch := cpuKVArm(t, f16, ids, refTokens)
		// PRECONDITION 1 (vacuity): the candidate's cache holds f16 rows and the control's f32, and their decode rows differ.
		if ch.quant != kvqF16 || ce.quant != kvqF32 {
			t.Fatalf("PRECONDITION 1: caches hold %v (f16 arm) and %v (f32 arm); want f16 and f32", ch.quant, ce.quant)
		}
		differ := false
		for i := 1; i < len(half); i++ {
			for j := range half[i] {
				if half[i][j] != exact[i][j] {
					differ = true
					break
				}
			}
		}
		if differ {
			diffPrompts++
		}
		if pi == 0 {
			again, _ := cpuKVArm(t, f16, ids, refTokens) // PRECONDITION 2: A/A on the first prompt
			for i := range again {
				for j := range again[i] {
					if again[i][j] != half[i][j] {
						t.Fatalf("PRECONDITION 2: A/A differs at row %d", i)
					}
				}
			}
			fmt.Printf("[cpu-kv-gate] %s K=%d: A/A bit-identical over %d rows; KV bytes at depth %d: f32 %d, f16 %d\n", name, K, len(again),
				ce.Pos(), cpuKVBytes(ce), cpuKVBytes(ch))
		}
		ea, _ := TeacherForcedTop1AgreementForTest(exact, refTokens)
		fa, _ := TeacherForcedTop1AgreementForTest(half, refTokens)
		var eKL, fKL float64
		var ehf, fhf int
		for i := range exact {
			if _, _, hf := NearTieArgmaxForTest(refLogits[i], exact[i]); hf {
				ehf++
			}
			if _, _, hf := NearTieArgmaxForTest(refLogits[i], half[i]); hf {
				fhf++
			}
			eKL += KLDivergenceForTest(refLogits[i], exact[i])
			fKL += KLDivergenceForTest(refLogits[i], half[i])
		}
		eKL /= float64(len(exact))
		fKL /= float64(len(half))
		n++
		sumEA, sumFA, sumEKL, sumFKL = sumEA+ea, sumFA+fa, sumEKL+eKL, sumFKL+fKL
		eHF, fHF = eHF+ehf, fHF+fhf
		if fa >= ea {
			wins++
		}
		fmt.Printf("[cpu-kv-gate] %s K=%d prompt %2d/%2d f32(agree=%.1f%% HF=%d KL=%.5f) f16(agree=%.1f%% HF=%d KL=%.5f) diff(agree=%+.2fpt KL=%+.5f) differ=%v elapsed=%s\n",
			name, K, pi+1, len(promptFiles), ea*100, ehf, eKL, fa*100, fhf, fKL, (fa-ea)*100, fKL-eKL, differ, time.Since(t0).Round(time.Second))
	}
	if diffPrompts < n {
		t.Fatalf("PRECONDITION 1 (vacuity): the f16 arm's rows differ from f32's on %d of %d prompts", diffPrompts, n)
	}
	mEA, mFA, mEKL, mFKL := sumEA/float64(n)*100, sumFA/float64(n)*100, sumEKL/float64(n), sumFKL/float64(n)
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
	fmt.Printf("=== CPU KV F16 GATE %s K=%d (set b, n=%d): f32(meanAgree=%.2f%% HF=%d meanKL=%.6f) f16(meanAgree=%.2f%% HF=%d meanKL=%.6f) pairedWins=%d/%d critA=%v [strict %v] critB=%v critC=%v KLratio=%.4f — %s ===\n",
		name, K, n, mEA, eHF, mEKL, mFA, fHF, mFKL, wins, n, critA, fHF <= eHF, critB, critC, mFKL/mEKL, verdict)
	if verdict != "PASSES" {
		t.Errorf("%s K=%d: %s — auto stays f32 for this family, which is a RESULT", name, K, verdict)
	}
}
