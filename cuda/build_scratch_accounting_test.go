//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
)

// The build-scratch / margin accounting gates, registered in docs/tasks/task-multimodal-support-2026-10.md ("Build-scratch / margin accounting on CUDA", 2026-10-08)
// before the code. The finding: the 384 MiB margin is not what falls short. The driver rounds every buffer of 2 MiB or more up to a 2 MiB multiple, Plan prices the
// requested bytes, and the plan therefore asks checkKVFits for a context that has already spent the rounding. The build prices it (packedAllocSlack) into the plan.

var accountingModels = []string{
	"qwen2.5-coder-0.5b-instruct-q4_k_m.gguf", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", "qwen2.5-7b-instruct-q4_k_m.gguf",
	"gemma-3-4b-it", "qwen25vl-3b-instruct", "gemma-4-e2b-gguf/gemma-4-E2B_q4_0-it.gguf",
}

// buildRow is one measured build: what the device gave up before the first checkKVFits probe against what Plan and the slack priced, the plan's context against
// the one checkKVFits left standing, and what was allocated after the probe (beyond the KV) and on first use.
type buildRow struct {
	name                                 string
	preKV, plan, slack, kv, postKV, lazy int64
	slots, ctxPlanned, ctxFinal          int
}

func (b buildRow) residual() int64 { return b.preKV - b.plan - b.slack }

func (b buildRow) log(t *testing.T, tag string) {
	t.Logf("[margin accounting%s] %-42s device before KV %5.0f MiB | Plan %5.0f + slack %4.0f -> residual %4.0f MiB | ctx planned %5d final %5d (%d slots, KV %5.0f MiB) | post-KV scratch %4.0f, first-use %4.0f MiB, against the %0.f MiB margin",
		tag, b.name, mb(b.preKV), mb(b.plan), mb(b.slack), mb(b.residual()), b.ctxPlanned, b.ctxFinal, b.slots, mb(b.kv), mb(b.postKV), mb(b.lazy), mb(ctxCapMarginBytes))
}

// measureBuild loads rel on the card at the defaults (int4, unpinned context), reading the free VRAM around the build and one 520-token prefill + decode. slackOff
// plans without the slack (the planted defect). ok=false when the model is absent or did not build a CUDA resident.
func measureBuild(t *testing.T, rel string, slackOff bool) (row buildRow, ok bool) {
	t.Helper()
	home, _ := os.UserHomeDir()
	p := filepath.Join(home, "models", rel)
	if _, err := os.Stat(p); err != nil {
		t.Logf("no model at %s", p)
		return row, false
	}
	dev, err := gc.GetDevice(0)
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	probe, err := dev.Primary()
	if err != nil {
		t.Skipf("primary ctx: %v", err)
	}
	realFree := func() int64 {
		f, _, e := probe.MemInfo()
		if e != nil {
			t.Fatalf("MemInfo: %v", e)
		}
		return int64(f)
	}
	orig, origOff := cudaFreeVRAM, allocSlackOffForTest
	defer func() { cudaFreeVRAM, allocSlackOffForTest = orig, origOff }()
	allocSlackOffForTest = slackOff
	var atKV int64
	calls := 0
	free0 := realFree()
	cudaFreeVRAM = func(r *cudaResident) (uint64, error) {
		f, _, e := r.dev.Context().MemInfo()
		calls++
		if calls == 1 { // the build's own probe; a later call is after the KV exists
			atKV = int64(f)
		}
		return f, e
	}
	m, err := decoder.Load(p, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load %s: %v", rel, err)
	}
	defer m.Close()
	r, isRes := m.ResidentForwardForTest().(*cudaResident)
	if !isRes || atKV == 0 {
		t.Logf("%s: not CUDA-resident: %s", rel, m.ResidentDecline())
		return row, false
	}
	freeBuilt := realFree()
	ids := make([]int, 520)
	for i := range ids {
		ids[i] = 100 + i%900
	}
	ch, g := m.Generate(context.Background(), ids, 8, decoder.SamplingParams{})
	for range ch {
	}
	if err := g.Err(); err != nil {
		t.Fatalf("generate %s: %v", rel, err)
	}
	freeAfter := realFree()
	row = buildRow{
		name: filepath.Base(rel), preKV: free0 - atKV, plan: m.ResidentDenseWeightBytesFor("cuda"), slack: r.allocSlackBytes,
		kv: int64(r.kvSlotsN) * kvBytesForCap(r.ctxCap, r.layers), slots: r.kvSlotsN, ctxPlanned: r.ctxPlanned, ctxFinal: r.ctxCap,
		lazy: freeBuilt - freeAfter,
	}
	row.postKV = atKV - row.kv - freeBuilt
	return row, true
}

// TestBuildScratchAccounting is G-M1 (and the record G-M2 reads): per bench model, the residual between what the device gave up before the KV and what Plan plus
// the slack priced must be small, and what the build allocates after the probe must fit the margin. Heavy.
func TestBuildScratchAccounting(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("heavy: set GOINFER_HEAVY_TESTS=1")
	}
	requireCUDADevice(t)
	// The first load in a process also pays the context and module cost (~386 MiB on the 0.5B) that a serve process has paid before it plans: a warm-up, not a row.
	measureBuild(t, accountingModels[0], false)
	bands := map[string][2]int64{ // residual in MiB: [-8, +64] on the Qwen text models, [-8, +128] on the others
		"qwen2.5-coder-0.5b-instruct-q4_k_m.gguf": {-8, 64}, "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf": {-8, 64}, "qwen2.5-7b-instruct-q4_k_m.gguf": {-8, 64},
	}
	for _, rel := range accountingModels {
		t.Run(filepath.Base(rel), func(t *testing.T) {
			row, ok := measureBuild(t, rel, false)
			if !ok {
				t.Skip("model absent or not CUDA-resident")
			}
			row.log(t, "")
			lo, hi := int64(-8), int64(128)
			if b, has := bands[rel]; has {
				lo, hi = b[0], b[1]
			}
			if res := row.residual() >> 20; res < lo || res > hi {
				t.Errorf("G-M1: residual %d MiB outside [%d, %d]: the build puts %d MiB on the device before the KV, Plan %d + slack %d: a cost the rounding rule does not know",
					res, lo, hi, row.preKV>>20, row.plan>>20, row.slack>>20)
			}
			if row.postKV+row.lazy > ctxCapMarginBytes {
				t.Errorf("post-KV scratch %d + first-use %d MiB exceeds the %d MiB margin", row.postKV>>20, row.lazy>>20, ctxCapMarginBytes>>20)
			}
		})
	}
}

// TestBuildScratchAccounting_plantedDefect plans WITHOUT the slack (the plan as it was) and shows what each half of the change buys, against the same build with it:
//   - the 7B: without the slack the device is further from Plan than the whole margin (425 MiB against 384, the bound the old TestResidentDenseBytes enforced), with
//     it the residual is under 64 MiB;
//   - every model: the residual with the slack is smaller than without it;
//   - Gemma 3 4B, the one bench model whose plan trims at the build: the trim (planned minus final context) is smaller with the slack than without it.
//
// The registered G-M2 said the 7B and Gemma 3 4B "trimmed before"; the 7B did not (it plans 16384 and keeps 16384 either way), so the 7B's evidence is the
// residual, not a trim. Heavy.
func TestBuildScratchAccounting_plantedDefect(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("heavy: set GOINFER_HEAVY_TESTS=1")
	}
	requireCUDADevice(t)
	measureBuild(t, accountingModels[0], false)
	for _, rel := range accountingModels[1:] {
		with, ok := measureBuild(t, rel, false)
		if !ok {
			continue
		}
		without, _ := measureBuild(t, rel, true)
		with.log(t, " slack on ")
		without.log(t, " slack off")
		if with.residual() >= without.residual() {
			t.Errorf("%s: residual with the slack (%d MiB) is not below the one without it (%d MiB): the slack priced nothing", with.name, with.residual()>>20, without.residual()>>20)
		}
		switch rel {
		case "qwen2.5-7b-instruct-q4_k_m.gguf":
			if without.residual() < ctxCapMarginBytes || with.residual() >= 64<<20 {
				t.Errorf("7B: residual %d MiB without the slack (want >= the %d MiB margin) and %d MiB with it (want < 64)", without.residual()>>20, ctxCapMarginBytes>>20, with.residual()>>20)
			}
		case "gemma-3-4b-it":
			if with.ctxPlanned-with.ctxFinal >= without.ctxPlanned-without.ctxFinal {
				t.Errorf("Gemma 3 4B: the trim at the build is %d positions with the slack and %d without it: the plan did not move toward the build",
					with.ctxPlanned-with.ctxFinal, without.ctxPlanned-without.ctxFinal)
			}
		}
	}
}
