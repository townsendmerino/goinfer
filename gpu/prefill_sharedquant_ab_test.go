//go:build gpu && goinfer_testhooks

package gpu

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestPrefill_sharedQuantAB times R-25 (docs/tasks/task-recompute-audit.md): the WebGPU batched prefill with one
// quantization per projection (prefillQuantPerProj, as it was) against sharedQ's one per input (q/k/v of xn, gate/up of
// xn2), on one loaded model (GOINFER_RESIDENT_GGUF, else the 1.5B), at P = 128 and 512 (GOINFER_R25_PS), whole
// PrefillLast wall time, arms alternated rep by rep after a warm-up, last-row logits compared every run. By day,
// in-process, exploratory.
//
//	GOINFER_R25_AB=1 go test -tags 'gpu goinfer_testhooks' -count=1 -run '^TestPrefill_sharedQuantAB$' -v ./gpu/
func TestPrefill_sharedQuantAB(t *testing.T) {
	if os.Getenv("GOINFER_R25_AB") != "1" {
		t.Skip("set GOINFER_R25_AB=1 (times a real checkpoint)")
	}
	home, _ := os.UserHomeDir()
	path := os.Getenv("GOINFER_RESIDENT_GGUF")
	if path == "" {
		path = filepath.Join(home, "models", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no model at %s: %v", path, err)
	}
	m, err := decoder.Load(path, decoder.Options{Backend: "webgpu", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	if !m.ResidentActive() {
		t.Skip("model not GPU-resident")
	}
	pf, ok := m.ResidentForwardForTest().(decoder.Prefiller)
	if !ok {
		t.Skip("resident forward does not implement Prefiller")
	}
	defer func() { prefillQuantPerProj = false }()
	hidden, _, _, _, _, _, _ := m.Dims()
	Ps := []int{128, 512}
	if v := os.Getenv("GOINFER_R25_PS"); v != "" {
		Ps = nil
		var n int
		for _, f := range strings.Split(v, ",") {
			if _, e := fmt.Sscan(f, &n); e == nil && n > 0 {
				Ps = append(Ps, n)
			}
		}
	}
	const reps = 7
	t0 := time.Now()
	for _, P := range Ps {
		rng := rand.New(rand.NewSource(int64(P)))
		embs := make([][]float32, P)
		for i := range embs {
			e := make([]float32, hidden)
			for j := range e {
				e[j] = float32(rng.NormFloat64()) * 0.5
			}
			embs[i] = e
		}
		ms := map[bool][]float64{}
		var ref []float32
		for rep := range reps + 1 {
			order := []bool{true, false}
			if rep%2 == 1 {
				order = []bool{false, true}
			}
			for _, perProj := range order {
				prefillQuantPerProj = perProj
				st := time.Now()
				lg, err := pf.PrefillLast(context.Background(), embs, 0)
				if err != nil {
					t.Fatal(err)
				}
				dt := float64(time.Since(st).Microseconds()) / 1e3
				if ref == nil {
					ref = append([]float32(nil), lg...)
				}
				for j := range ref {
					if math.Float32bits(ref[j]) != math.Float32bits(lg[j]) {
						t.Fatalf("P=%d rep %d: logit %d differs between the arms", P, rep, j)
					}
				}
				if rep > 0 {
					ms[perProj] = append(ms[perProj], dt)
				}
			}
		}
		med := func(v []float64) float64 { s := append([]float64(nil), v...); sort.Float64s(s); return s[len(s)/2] }
		ratio := make([]float64, reps)
		above := 0
		per := ""
		for i := range ratio {
			ratio[i] = ms[true][i] / ms[false][i]
			if ratio[i] > 1 {
				above++
			}
			per += fmt.Sprintf(" %.3f", ratio[i])
		}
		fmt.Fprintf(os.Stderr, "[r25 %6.1fs] P=%d: per-projection %.2f ms, shared %.2f ms (medians of %d); RESULT per-projection / shared median %.3f, %d of %d reps above 1 (per rep%s)\n",
			time.Since(t0).Seconds(), P, med(ms[true]), med(ms[false]), reps, med(ratio), above, reps, per)
	}
}
