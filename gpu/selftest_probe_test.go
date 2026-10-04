//go:build gpu && goinfer_testhooks

package gpu

import (
	"bytes"
	"regexp"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// H2 on WebGPU (docs/tasks/task-hardware-coverage-2026-10.md): the resident self-test on a real adapter. A green probe on healthy code, and the same probe going RED when the resident's inputs are
// wrong, which is what makes the green mean something. Skips on a software renderer (lavapipe on a CI runner): its limits and math are not a GPU's, and the margins below are for a real adapter.

func requireRealAdapter(t *testing.T) {
	t.Helper()
	c, err := New()
	if err != nil {
		t.Skip("no WebGPU adapter: " + err.Error())
	}
	defer c.Close()
	if isSoftwareAdapter(c) {
		t.Skip("software WebGPU adapter: the probe's margins are for a real GPU")
	}
}

func TestResidentSelfTest_passesOnHealthyWebGPU(t *testing.T) {
	requireRealAdapter(t)
	for _, quant := range []string{"int4", "int8int8"} {
		t0 := time.Now()
		r := decoder.ProbeBackendForTest("webgpu", quant)
		t.Logf("%s: %s (wall %v)", quant, r.Summary(), time.Since(t0))
		if r.Status != decoder.SelfTestPass {
			t.Errorf("%s: status %s on healthy code: %s", quant, r.Status, r.Summary())
		}
	}
}

// Each mutation changes ONE input the resident consumes, on the GPU copy only. If the probe stays green for any of them, it checks less than it claims.
func TestResidentSelfTest_goesRedOnWrongResidentInputsWebGPU(t *testing.T) {
	requireRealAdapter(t)
	muts := []struct{ name, re, to string }{
		{"rms epsilon", `"rms_norm_eps":\s*[0-9.eE+-]+`, `"rms_norm_eps": 0.5`},
		{"rope base", `"rope_theta":\s*[0-9.eE+-]+`, `"rope_theta": 3.0`},
	}
	for _, mu := range muts {
		t.Run(mu.name, func(t *testing.T) {
			re := regexp.MustCompile(mu.re)
			changed := 0
			restore := decoder.SetProbeGPUConfigMutatorForTest(func(fixture string, cfg []byte) []byte {
				if fixture != "qwen35vl-tiny" {
					return cfg
				}
				out := re.ReplaceAll(cfg, []byte(mu.to))
				if !bytes.Equal(out, cfg) {
					changed++
				}
				return out
			})
			defer restore()
			r := decoder.ProbeBackendForTest("webgpu", "int4")
			t.Logf("%s", r.Summary())
			if changed == 0 {
				t.Fatalf("the mutation %q matched nothing in the fixture's config, so this proves nothing", mu.name)
			}
			if r.Status != decoder.SelfTestDeclined {
				t.Errorf("a wrong %s on the GPU copy left the probe at %s; it must decline", mu.name, r.Status)
			}
		})
	}
}
