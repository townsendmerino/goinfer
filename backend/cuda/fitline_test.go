//go:build cuda && goinfer_testhooks

package cuda

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/decoder/fit"
	"github.com/townsendmerino/goinfer/internal/app/fitcmd"
)

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
