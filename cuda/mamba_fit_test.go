//go:build cuda && goinfer_testhooks

package cuda

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// OP2b on CUDA, gate F (docs/tasks/task-forward-architecture-2026-10.md §4.7): the resident's own fit plan for the real Nemotron-H checkpoints, read once and
// alone before any tier 1 run, against the predictions committed before a resident existed. A refused load is VOID for its cell, not a FAIL, and no guard is
// bypassed. Reports the free VRAM before and after, the planned and final context, the KV slots and bytes, and the Mamba state, for a default load and for the
// context the prediction names.
//
//	GOINFER_HEAVY_TESTS=1 NEMOTRON_FIT_BUNDLE=<path to a CUDA int4 .giw> go test -count=1 -timeout 20m -tags 'cuda goinfer_testhooks' -run '^TestNemotronResidentCUDA_fitPlan$' -v ./cuda/
func TestNemotronResidentCUDA_fitPlan(t *testing.T) {
	requireHeavyModel(t)
	requireCUDADevice(t)
	defer nemotronFeaturesForTest()()
	path := os.Getenv("NEMOTRON_FIT_BUNDLE")
	if path == "" {
		t.Skip("NEMOTRON_FIT_BUNDLE names the CUDA int4 bundle to plan")
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("VOID: %v", err)
	}
	for _, ctx := range []int{0, 16384, 65536, 131072} {
		freeBefore, _ := decoder.FreeBytesFor("cuda")
		t0 := time.Now()
		m, err := decoder.Load(path, decoder.Options{Backend: "cuda", ResidentContext: ctx})
		if err != nil {
			t.Fatalf("load %s (ctx %d): %v", filepath.Base(path), ctx, err)
		}
		took := time.Since(t0)
		freeAfter, _ := decoder.FreeBytesFor("cuda")
		r, ok := m.ResidentForwardForTest().(*cudaResident)
		if !ok {
			t.Logf("F ctx request %d: NOT RESIDENT (%s), load %.1fs, free VRAM before %.0f MiB: decline: %s", ctx, m.DecodePath(), took.Seconds(), float64(freeBefore)/(1<<20), m.ResidentDecline())
			m.Close()
			continue
		}
		nMamba := 0
		for l := range r.layers {
			if r.layers[l].mixer == decoder.MixerMamba {
				nMamba++
			}
		}
		kv := kvBytesForCap(r.ctxCap, r.layers)
		t.Logf("F ctx request %d: RESIDENT (%s), load %.1fs; planned ctx %d, final ctx %d, %d KV slot(s), KV %.1f MiB a slot; Mamba state %.1f MiB (%d layers); "+
			"free VRAM %.0f MiB before, %.0f MiB after (the load took %.0f MiB)", ctx, m.DecodePath(), took.Seconds(), r.ctxPlanned, r.ctxCap, r.KVSlots(), float64(kv)/(1<<20),
			float64(mambaStateBytes(r.mamba, nMamba))/(1<<20), nMamba, float64(freeBefore)/(1<<20), float64(freeAfter)/(1<<20), float64(freeBefore-freeAfter)/(1<<20))
		m.Close()
	}
}

// nemotronCell is one tier 1 run: a checkpoint resident on CUDA against the CPU model it was built from, at int4.
type nemotronCell struct {
	name, path string
	control    bool // the dense control: below 0.95 the session is VOID
	ctx        int  // an explicit resident context for the CUDA side; 0 leaves the plan's choice
	embedInt4  bool // the token table at int4 on both sides, as serve loads it by default and as a bundle with e4h is baked (the 9B: at int8 its head does not fit beside the rest)
}

// nemotronTier1Reading is a cell's lockstep reading: the worst logit cosine over positions from 2, the argmax-equal count, NaN rows, and the CUDA side's
// mean seconds a step after the first (a recorded rate, not graded) with the first step's time.
type nemotronTier1Reading struct {
	g2rStats
	firstStep, meanStep time.Duration
	decodePath          string
}

// nemotronLockstep is g2rLockstep with the CUDA step times kept: 32 steps of greedy lockstep off the CPU, a fixed prompt then the CPU's argmax. The CUDA side
// must be resident (a silent CPU fallback passes every bar), and an all-zero logit row on either side voids the reading.
func nemotronLockstep(t *testing.T, path string, steps int, embedInt4 bool, ctx int) nemotronTier1Reading {
	t.Helper()
	seed := g2rSeed(t, path)
	mg, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4", EmbedInt4: embedInt4, ResidentContext: ctx})
	if err != nil {
		t.Fatalf("load (cuda) %s: %v", path, err)
	}
	defer mg.Close()
	rf := mg.ResidentForwardForTest()
	if rf == nil {
		t.Fatalf("VOID: the cuda resident declined %s (%s): %s", path, mg.DecodePath(), mg.ResidentDecline())
	}
	mc, err := decoder.Load(path, decoder.Options{Quant: "int4", EmbedInt4: embedInt4})
	if err != nil {
		t.Fatalf("load (cpu) %s: %v", path, err)
	}
	defer mc.Close()
	cache := mc.NewCache(steps + 2)
	rf.Reset()
	out := nemotronTier1Reading{decodePath: mg.DecodePath()}
	out.steps, out.minCos = steps, 1
	var total time.Duration
	tok := seed[0]
	for i := range steps {
		cl, err := mc.ForwardForTest(tok, cache)
		if err != nil {
			t.Fatalf("cpu forward: %v", err)
		}
		t0 := time.Now()
		gl, err := rf.Forward(mg.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("cuda forward: %v", err)
		}
		if d := time.Since(t0); i == 0 {
			out.firstStep = d
		} else {
			total += d
		}
		if allZero(cl) || allZero(gl) {
			t.Fatalf("VOID: step %d has an all-zero logit row (cpu %v, cuda %v)", i, allZero(cl), allZero(gl))
		}
		for _, v := range gl {
			if v != v {
				out.nan++
				break
			}
		}
		if i >= 2 {
			c, _ := cosF32(cl, gl)
			out.minCos = min(out.minCos, c)
		}
		ca, ga := argmaxF(cl), argmaxF(gl)
		if ca == ga {
			out.exact++
		}
		if i+1 < len(seed) {
			tok = seed[i+1]
		} else {
			tok = ca
		}
	}
	out.meanStep = total / time.Duration(steps-1)
	return out
}

// TestNemotronResidentCUDA_tier1 is OP2b's tier 1 on CUDA (docs/tasks/task-forward-architecture-2026-10.md §4.7): whole real Nemotron-H checkpoints resident on
// the card against the CPU model at the same int4, 32 steps of greedy lockstep, G2-R's bars (min logit cosine >= 0.88 from position 2, argmax equal on at
// least 50% of steps, no NaN). The dense Qwen2.5-Coder 1.5B runs first through the same harness; under min cosine 0.95 the session is VOID. A refused load is VOID
// for its cell. NEMOTRON_TIER1 names the cells, comma separated: 4b (the 4B whole), 9bslice (layers 0, 1 and 14 of the 9B), 9b (the 9B whole).
//
//	GOINFER_HEAVY_TESTS=1 NEMOTRON_TIER1=4b go test -count=1 -timeout 40m -tags 'cuda goinfer_testhooks' -run '^TestNemotronResidentCUDA_tier1$' -v ./cuda/
func TestNemotronResidentCUDA_tier1(t *testing.T) {
	requireHeavyModel(t)
	requireCUDADevice(t)
	defer nemotronFeaturesForTest()()
	home := os.ExpandEnv("$HOME/models")
	all := map[string]nemotronCell{
		"4b":      {name: "Nemotron-H 4B whole", path: home + "/nemotron-h-4b-instruct-128k"},
		"9bslice": {name: "Nemotron-Nano-9B-v2 slice 0,1,14", path: home + "/nemotron/nemotron-nano-9b-v2-Q8_0.slice-0-1-14.gguf"},
		"9b":      {name: "Nemotron-Nano-9B-v2 whole (token table at int4)", path: home + "/nemotron/nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q8_0.gguf", embedInt4: true, ctx: 4096},
	}
	cells := []nemotronCell{{name: "control: dense Qwen2.5-Coder 1.5B", path: home + "/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", control: true}}
	for _, k := range strings.Split(os.Getenv("NEMOTRON_TIER1"), ",") {
		c, ok := all[strings.TrimSpace(k)]
		if !ok {
			t.Fatalf("NEMOTRON_TIER1 names an unknown cell %q (4b, 9bslice, 9b)", k)
		}
		cells = append(cells, c)
	}
	const steps, bar, ctrlBar = 32, 0.88, 0.95
	for _, c := range cells {
		if _, err := os.Stat(c.path); err != nil {
			t.Fatalf("VOID: %s is missing", c.path)
		}
	}
	for _, c := range cells {
		t0 := time.Now()
		r := nemotronLockstep(t, c.path, steps, c.embedInt4, c.ctx)
		pass := r.minCos >= bar && 2*r.exact >= steps && r.nan == 0
		if c.control {
			pass = r.minCos >= ctrlBar && r.nan == 0
		}
		t.Logf("%s (%s): min cosine %.6f, argmax %d/%d, NaN rows %d; CUDA first step %.2f s, then %.1f ms a step (%.1f tok/s, recorded, not graded); cell %.0f s -> %s",
			c.name, r.decodePath, r.minCos, r.exact, r.steps, r.nan, r.firstStep.Seconds(), float64(r.meanStep.Microseconds())/1000, 1/r.meanStep.Seconds(),
			time.Since(t0).Seconds(), map[bool]string{true: "PASS", false: "FAIL"}[pass])
		if c.control && !pass {
			t.Fatalf("VOID: the dense control read min cosine %.6f (< %.2f): the session proves nothing", r.minCos, ctrlBar)
		}
		if !pass {
			t.Errorf("%s is outside the bars (min cosine >= %.2f, argmax >= 50%%)", c.name, bar)
		}
	}
}
