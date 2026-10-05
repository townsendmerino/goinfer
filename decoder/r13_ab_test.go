package decoder

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"
)

// TestR13_cpuAB times R-13 (docs/tasks/task-recompute-audit.md) on the CPU: each input quantized once for the W4A8
// projections that share it (the default) against every matmul quantizing its own (w4a8PreOff), on one loaded model
// per path in GOINFER_R13_MODELS (default the 1.5B GGUF and the Qwen1.5-MoE 4-layer slice), int4, one activation scale
// per row. Decode: a fresh cache per arm, 8 warm-up tokens, then the median of 32 timed decode steps. Batched prefill
// (dense models only): a 256-token forwardN. Arms alternated rep by rep after a warm-up rep; the logits are compared
// every rep. By day, in-process, exploratory.
//
//	GOINFER_R13_AB=1 go test -count=1 -run '^TestR13_cpuAB$' -v ./decoder/
func TestR13_cpuAB(t *testing.T) {
	if os.Getenv("GOINFER_R13_AB") != "1" {
		t.Skip("set GOINFER_R13_AB=1 (times real checkpoints)")
	}
	home, _ := os.UserHomeDir()
	paths := []string{filepath.Join(home, "models", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"), filepath.Join(home, "models", "qwen15-moe-a27b-l4slice")}
	if v := os.Getenv("GOINFER_R13_MODELS"); v != "" {
		paths = strings.Split(v, ",")
	}
	defer func() { w4a8PreOff = false }()
	const reps, warm, steps, prefillN = 7, 8, 32, 256
	t0 := time.Now()
	hb := func(f string, a ...any) {
		fmt.Fprintf(os.Stderr, "[r13 %6.1fs] %s\n", time.Since(t0).Seconds(), fmt.Sprintf(f, a...))
	}
	med := func(v []float64) float64 { s := append([]float64(nil), v...); sort.Float64s(s); return s[len(s)/2] }
	for _, path := range paths {
		if strings.HasPrefix(path, "/Volumes/") || strings.HasPrefix(path, "/srv/models") {
			t.Fatalf("%s is on the archive, not the bench set (CLAUDE.md)", path)
		}
		if _, err := os.Stat(path); err != nil {
			hb("%s: missing, skipped", path)
			continue
		}
		m, err := Load(path, Options{Quant: "int4"})
		if err != nil {
			t.Fatalf("load %s: %v", path, err)
		}
		name := filepath.Base(path)
		moe := m.w.arch.MoE != nil
		hb("%s: loaded (%d layers, MoE %v)", name, m.w.arch.NumLayers, moe)
		V := m.w.arch.VocabSize
		ids := make([]int, prefillN)
		for i := range ids {
			ids[i] = (i*7919 + 11) % V
		}
		decode := func(off bool) ([]float32, float64) {
			w4a8PreOff = off
			cache := m.NewCache(warm + steps + 4)
			var last []float32
			var ms []float64
			for i := range warm + steps {
				st := time.Now()
				lg, err := m.forward(ids[i], cache)
				if err != nil {
					t.Fatal(err)
				}
				if i >= warm {
					ms = append(ms, float64(time.Since(st).Microseconds())/1e3)
				}
				last = lg
			}
			return append([]float32(nil), last...), med(ms)
		}
		prefill := func(off bool) ([]float32, float64) {
			w4a8PreOff = off
			st := time.Now()
			out, err := m.forwardN(context.Background(), ids, m.NewCache(prefillN+4))
			if err != nil {
				t.Fatal(err)
			}
			return append([]float32(nil), out[len(out)-1]...), float64(time.Since(st).Microseconds()) / 1e3
		}
		type phase struct {
			name string
			f    func(bool) ([]float32, float64)
		}
		phases := []phase{{"decode ms a token", decode}}
		if !moe {
			phases = append(phases, phase{"prefill 256 ms", prefill})
		}
		for _, ph := range phases {
			times := map[bool][]float64{}
			var ref []float32
			for rep := range reps + 1 {
				order := []bool{true, false}
				if rep%2 == 1 {
					order = []bool{false, true}
				}
				for _, off := range order {
					lg, ms := ph.f(off)
					if ref == nil {
						ref = lg
					}
					for j := range ref {
						if math.Float32bits(ref[j]) != math.Float32bits(lg[j]) {
							t.Fatalf("%s %s rep %d: logit %d differs between the arms", name, ph.name, rep, j)
						}
					}
					if rep > 0 {
						times[off] = append(times[off], ms)
					}
				}
			}
			ratio := make([]float64, reps)
			above, per := 0, ""
			for i := range ratio {
				ratio[i] = times[true][i] / times[false][i]
				if ratio[i] > 1 {
					above++
				}
				per += fmt.Sprintf(" %.3f", ratio[i])
			}
			hb("RESULT %s %s: off %.3f, on %.3f (medians of %d); off / on median %.3f, %d of %d reps above 1 (per rep%s)",
				name, ph.name, med(times[true]), med(times[false]), reps, med(ratio), above, reps, per)
		}
		m.Close()
	}
}
