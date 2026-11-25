//go:build gpu && goinfer_testhooks

package webgpu

import (
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/backend/webgpu/internal/kernels"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
)

// TestSetAdapter_partialBindErrorRestoresSteps pins that a SetAdapter that fails partway through
// its conversion loop restores r.steps to r.baseSteps' length. SetAdapter clears and releases
// r.loraLayers unconditionally at the top of the call, so a mid-loop error that left r.steps (the
// flat, Go-side dispatch-step list Run walks) pointing at the previously bound adapter, whose
// bind groups were just released, would make a later Run() dispatch against freed WebGPU
// resources.
//
// This backend has no buffer ledger to assert a leak count directly (unlike Metal's
// d.LedgerLen(); metal's TestSetAdapter_partialBindErrorReleasesBuffers is the leak gate), so
// this test pins the more severe, directly observable half: r.steps must not stay at a stale
// adapter's larger spliced-in step count. Background:
// docs/code-notes/gpu.md#TestSetAdapter_partialBindErrorRestoresSteps.
func TestSetAdapter_partialBindErrorRestoresSteps(t *testing.T) {
	const ckpt = "../../testdata/llama-tiny"
	adapterDir := buildLlamaTinyLoRAFixtureGPU(t)

	m, err := decoder.Load(ckpt, decoder.Options{Backend: "webgpu", Quant: "int4"})
	if err != nil {
		t.Skipf("no webgpu device: %v", err)
	}
	defer m.Close()
	rf := m.ResidentForwardForTest()
	if rf == nil {
		t.Fatalf("llama-tiny did not go resident: decline=%s", m.ResidentDecline())
	}
	ra := decoder.SubsystemsOf(rf).Runtime
	ok := ra.AdapterAvailable()
	if !ok {
		t.Fatalf("webgpu resident runner does not implement decoder.ResidentRuntimeSubsystem")
	}
	rd, ok := rf.(*residentDecoder)
	if !ok {
		t.Fatalf("resident runner is %T, not *residentDecoder", rf)
	}
	r := rd.runner
	if err := m.LoadAdapter("a", adapterDir); err != nil {
		t.Fatalf("LoadAdapter: %v", err)
	}
	layers := m.ResidentAdapterLayersForTest("a")
	if len(layers) < 2 {
		t.Fatalf("llama-tiny adapter has %d layers, need >= 2 to exercise a PARTIAL bind", len(layers))
	}

	baseLen := len((*kernels.DecodeRunnerBaseSteps(r)))

	// First bind: a real, fully valid adapter across every layer — r.steps must grow past
	// r.baseSteps (the LoRA down/up dispatch pairs spliced in), giving a non-trivial "stale"
	// step count to detect if a later failed rebind leaves it untouched.
	if err := ra.SetAdapter(layers); err != nil {
		t.Fatalf("SetAdapter (valid): %v", err)
	}
	boundLen := len((*kernels.DecodeRunnerSteps(r)))
	if boundLen <= baseLen {
		t.Fatalf("test setup: bound step count %d did not grow past base %d — the fixture isn't "+
			"exercising a real splice, this test proves nothing", boundLen, baseLen)
	}

	// Second bind: layer 1's Q rank is invalid, but layer 0 converts cleanly first — this
	// exercises the actual partial-progress path (mk succeeds several times before failing),
	// not a first-projection failure.
	broken := make([]decoder.ResidentAdapterLayer, len(layers))
	copy(broken, layers)
	badQ := *layers[1].Q
	badQ.R = 0 // R<=0: mk refuses
	broken[1].Q = &badQ

	if err := ra.SetAdapter(broken); err == nil {
		t.Fatal("SetAdapter accepted an invalid rank on layer 1 — test setup is wrong")
	}

	if (*kernels.DecodeRunnerLoraLayers(r)) != nil {
		t.Error("r.loraLayers not nil after a failed rebind — a caller checking it would think an adapter is bound")
	}
	if got := len((*kernels.DecodeRunnerSteps(r))); got != baseLen {
		t.Errorf("r.steps has %d entries after a failed rebind, want %d (r.baseSteps, no adapter) — "+
			"still reflects the PREVIOUS successful bind's %d steps, whose bind groups were already "+
			"released at the top of this SetAdapter call: a Run() now dispatches against freed "+
			"WebGPU resources (N-82)", got, baseLen, boundLen)
	}

	// The resident must still be usable after a rejected bind — no adapter, but not broken.
	if err := ra.SetAdapter(nil); err != nil {
		t.Errorf("SetAdapter(nil) after a failed bind: %v", err)
	}
}

// TestBuildResident_kvAllocFailureReleasesWhatItLeased fails a resident build at one layer's V allocation, after that layer's
// K was created, and holds the build to its unwind: once Load has fallen back, no buffer on the KV line, and no buffer the build
// leased on any line, is still open on the backend's Context. Each KV site is covered: the generic attention layer at every KV
// precision (an int8 cache is two KV-line buffers, the codes then their scales, so its cases fail layer 1's V codes and,
// separately, its V scales), Nemotron-H's attention layer and Qwen3.5's gated attention layer.
func TestBuildResident_kvAllocFailureReleasesWhatItLeased(t *testing.T) {
	for _, tc := range []struct {
		ckpt, quant, kv string
		failAt          int    // the KV-line allocation refused, counting from 1
		decline         string // the site's error
	}{
		{"llama-tiny", "int8int8", "", 4, "residency KV alloc (layer 1)"},
		{"llama-tiny", "int8int8", "f16", 4, "residency KV alloc (layer 1)"},
		{"llama-tiny", "int8int8", "i8", 7, "residency KV alloc (layer 1)"},
		{"llama-tiny", "int8int8", "i8", 8, "residency KV alloc (layer 1)"},
		{"nemotron-tiny", "int4", "", 2, "nemotron KV alloc"},
		{"qwen35-tiny", "int4", "", 2, "qwen35 KV alloc"},
	} {
		kv, ckpt := tc.kv, "../../testdata/"+tc.ckpt
		t.Run(fmt.Sprintf("%s/kv=%s/alloc%d", tc.ckpt, kv, tc.failAt), func(t *testing.T) {
			b, err := newWebGPUBackend("kv-alloc-failure test")
			if err != nil {
				t.Skipf("no webgpu device: %v", err)
			}
			defer b.ctx.Close()
			before := b.ctx.Ledger().Usage()
			kvAllocs := 0
			b.ctx.SetCreateFailForTest(func(line string, _ uint64) error {
				if line != mem.LineKV {
					return nil
				}
				if kvAllocs++; kvAllocs == tc.failAt {
					return errors.New("planted: KV allocation refused")
				}
				return nil
			})
			reg := fmt.Sprintf("webgpu-kv-alloc-failure-%s-%s-%d", tc.ckpt, kv, time.Now().UnixNano())
			decoder.RegisterBackend(reg, func() (decoder.Backend, error) { return b, nil })
			m, err := decoder.Load(ckpt, decoder.Options{Backend: reg, Quant: tc.quant, KVPrecision: kv})
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			if !strings.Contains(m.ResidentDecline(), tc.decline) {
				t.Fatalf("the plant did not fail the build at the KV allocation it targets (%q): decline %q", tc.decline, m.ResidentDecline())
			}
			after := b.ctx.Ledger().Usage()
			if n := after.ByLine[mem.LineKV]; n != 0 {
				t.Errorf("%d bytes still leased on the KV line after the failed build", n)
			}
			if after.Leases != before.Leases {
				t.Errorf("%d leases open after the failed build (%d before it): by line %v", after.Leases, before.Leases, after.ByLine)
			}
		})
	}
}
