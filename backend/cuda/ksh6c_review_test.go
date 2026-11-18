//go:build cuda && goinfer_testhooks

package cuda

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/decoder/fit"
	"github.com/townsendmerino/goinfer/internal/app/fitcmd"
)

// TestKSH6C_marginAtTheReturnedContext is H6C-G1c: every
// return of resolveCtxCapFit, the shortcuts and the floor included, gives the margin at the context it returns, whatever context
// the plan priced.
func TestKSH6C_marginAtTheReturnedContext(t *testing.T) {
	named := func(c int) int64 { return int64(c) * 1000 } // a margin that names its context
	for _, c := range []struct {
		name              string
		request, modelCtx int
		pinned            bool
		slots             int
	}{
		{"a genuine pin", 8192, 32768, true, 1},
		{"the model window at the floor", 0, cudaCtxCapDefault, false, 1},
		{"the plan, one slot", 0, 1 << 20, false, 1},
		{"the plan, four slots", 0, 1 << 20, false, 4},
	} {
		t.Run(c.name, func(t *testing.T) {
			opts := decoder.Options{Quant: "int4"}
			if c.pinned {
				opts.ResidentContext = c.request
			}
			m, err := decoder.Load("../../testdata/llama-tiny", opts)
			if err != nil {
				t.Fatalf("Load: %v", err)
			}
			defer m.Close()
			ctx, mb := resolveCtxCapFit(m, c.request, c.modelCtx, c.slots, named)
			if mb != named(ctx) {
				t.Errorf("returned context %d with the margin at %d positions", ctx, mb/1000)
			}
		})
	}
}

// TestKSH6C_planSeesTheMoEPool is H6C-S6: qwen35-tiny (DeltaNet and MoE), its window raised so the plan runs, unpinned, the startup
// self-test skipped. The free VRAM the plan reads must be at most 64 MiB above the fit check's: the weights and the pre-KV buffers,
// not moe_route's pool, which openForPlan pays first. KSH6C_S6_PLANT=1 skips that warm-up, leaving the pool to payRoutePool between
// the two readings; it must run in its own process, since the context keeps the pool once it is paid.
func TestKSH6C_planSeesTheMoEPool(t *testing.T) {
	const base = "../../testdata/qwen35-tiny"
	requireDeviceAndFixture(t, base)
	if os.Getenv("KSH6C_S6") == "" {
		t.Skip("set KSH6C_S6=1 (it needs a fresh process: the context keeps the pool once paid)")
	}
	plant := os.Getenv("KSH6C_S6_PLANT") != ""
	prePlanPoolOffForTest = plant
	defer func() { prePlanPoolOffForTest = false }()
	decoder.SkipSelfTests()
	dir := kv3LongContextCopy(t, base, 32768)
	var planFree int64
	planFreeSeenForTest = func(f int64) { planFree = f }
	fitByR := map[*cudaResident]int64{}
	saved := cudaFreeVRAM
	cudaFreeVRAM = func(r *cudaResident) (uint64, error) {
		f, e := saved(r)
		if _, seen := fitByR[r]; !seen {
			fitByR[r] = int64(f)
		}
		return f, e
	}
	m, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4"})
	planFreeSeenForTest, cudaFreeVRAM = nil, saved
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	r, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok || !r.moe {
		t.Fatalf("not a MoE CUDA resident: %s", m.ResidentDecline())
	}
	if planFree == 0 {
		t.Fatal("the plan did not read free VRAM: the window did not put the candidate above the floor")
	}
	mib := func(b int64) float64 { return float64(b) / (1 << 20) }
	gap := planFree - fitByR[r]
	fmt.Fprintf(os.Stderr, "[H6C-S6] plant %v: the plan read %.1f MiB free, the fit check %.1f MiB: %.1f MiB between them; context %d (planned %d)\n",
		plant, mib(planFree), mib(fitByR[r]), mib(gap), r.ctxCap, r.ctxPlanned)
	if plant {
		if gap < 130<<20 {
			t.Errorf("plant: with the warm-up before the plan skipped, the plan read only %.1f MiB more than the fit check (want >= 130): the gate cannot see it", mib(gap))
		}
		return
	}
	if gap > 64<<20 {
		t.Errorf("the plan read %.1f MiB more than the fit check (want <= 64): it priced a pool the build then took", mib(gap))
	}
}

// TestFitCUDALine_isTheBuildsPlan: `fit`'s CUDA line (fitcmd.PlanCUDA, the function Run prints it from) is the resident build's own
// plan. llama-tiny, with its window raised to 32768 so an unpinned context goes through the plan (ctxPlanRange), is planned by fit
// and then built as serve loads it (four KV slots by default), both against one free-VRAM figure, chosen so that the context
// margin and the KV rounding decide where the context lands. fit's context, KV slot count and KV bytes must equal the built
// resident's planned context (kvShareCtx), its slots and kvChargedBytes. A pinned -ctx, and a pinned -ctx with -kv-f16 (the build
// at f16), are held the same way.
func TestFitCUDALine_isTheBuildsPlan(t *testing.T) {
	requireDeviceAndFixture(t, "../../testdata/llama-tiny")
	dir := wideLlamaTiny(t)

	// A free figure at which the unpinned four-slot plan lands inside (cudaCtxCapDefault, fitDefaultCtx): the smallest at which the
	// build's own plan reaches 8192 positions.
	probe, err := decoder.Load(dir, decoder.Options{Backend: "cpu", Quant: "int4", ResidentKVSlots: 4, ResidentKVSlotsDefault: true,
		ResidentWindowedKV: true})
	if err != nil {
		t.Fatal(err)
	}
	planned := func(free int64) int {
		p, _, err := (&cudaBackend{}).deriveMemory(probe, free, 0)
		if err != nil || p.Decline != "" {
			return 0
		}
		return p.CtxPlanned
	}
	lo, hi := int64(0), int64(8<<30)
	for hi-lo > 1<<10 {
		if mid := lo + (hi-lo)/2; planned(mid) >= 8192 {
			hi = mid
		} else {
			lo = mid
		}
	}
	probe.Close()
	free := hi
	decoder.RegisterMemoryProbe("cuda", func() (int64, bool) { return free, true })
	t.Cleanup(func() { decoder.RegisterMemoryProbe("cuda", freeVRAM) })

	for _, c := range []struct {
		name  string
		ctx   int // 0 is unpinned
		kvF16 bool
	}{
		{"unpinned", 0, false},
		{"pinned", 6144, false},
		{"pinned, -kv-f16", 6144, true},
	} {
		t.Run(c.name, func(t *testing.T) {
			a := fitcmd.Args{Ctx: 16384, Quant: "int4", KVF16: c.kvF16}
			opts := decoder.Options{Backend: "cuda", Quant: "int4", ResidentKVSlots: 4, ResidentKVSlotsDefault: true, ResidentWindowedKV: true}
			if c.ctx > 0 {
				a.Ctx, a.CtxPinned, opts.ResidentContext = c.ctx, true, c.ctx
			}
			if c.kvF16 {
				opts.KVPrecision = "f16"
			}
			line, err := fitcmd.PlanCUDA(dir, a)
			if err != nil {
				t.Fatal(err)
			}
			m, err := decoder.Load(dir, opts)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			r, ok := m.ResidentForwardForTest().(*cudaResident)
			if !ok {
				t.Fatalf("not resident on cuda: %s", m.DecodePath())
			}
			if c.ctx == 0 && (r.kvShareCtx <= cudaCtxCapDefault || r.kvShareCtx >= fitDefaultCtx) {
				t.Fatalf("the build planned %d positions at %d bytes free; the case needs a context inside (%d, %d), where the margin decides it",
					r.kvShareCtx, free, cudaCtxCapDefault, fitDefaultCtx)
			}
			if c.kvF16 != r.kvF16 && c.kvF16 {
				t.Fatalf("-kv-f16: the build holds f32 KV; the case needs an f16 build")
			}
			want := kvChargedBytes(r.kvShareCtx, r.kvSlotsN, r.layers)
			if line.Plan.Placement == fit.PlacementDecline || line.Plan.Ctx != r.kvShareCtx || line.KVSlots != r.kvSlotsN || line.Plan.KVBytes != want {
				t.Errorf("fit's CUDA line: %s, context %d, %d KV slots, KV %d bytes; the build: context %d, %d slots, kvChargedBytes %d\n  fit: %s",
					line.Plan.Placement, line.Plan.Ctx, line.KVSlots, line.Plan.KVBytes, r.kvShareCtx, r.kvSlotsN, want, line.Plan.Reason)
			}
		})
	}
}

// wideLlamaTiny is a copy of testdata/llama-tiny whose declared window is 32768, so a load's context is planned rather than held at
// the model's 128 positions.
func wideLlamaTiny(t *testing.T) string {
	t.Helper()
	src, dir := "../../testdata/llama-tiny", t.TempDir()
	for _, f := range []string{"model.safetensors", "generation_config.json"} {
		b, err := os.ReadFile(filepath.Join(src, f))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, f), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	b, err := os.ReadFile(filepath.Join(src, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	var cfg map[string]any
	if err := json.Unmarshal(b, &cfg); err != nil {
		t.Fatal(err)
	}
	cfg["max_position_embeddings"] = 32768
	if b, err = json.Marshal(cfg); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), b, 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}
