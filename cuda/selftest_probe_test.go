//go:build cuda && goinfer_testhooks

package cuda

import (
	"bytes"
	"regexp"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// H2: the resident self-test on a real device (docs/tasks/task-hardware-coverage-2026-10.md). A green probe on healthy code, and the same probe going RED when the resident's inputs are
// wrong, which is what makes the green mean something.

func TestResidentSelfTest_passesOnHealthyCUDA(t *testing.T) {
	skipWithoutCUDA(t)
	// A model holding the context, as the real model does when the probe runs after its resident is built.
	hold, err := decoder.Load("../testdata/llama-tiny", decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatal(err)
	}
	defer hold.Close()
	for _, quant := range []string{"int4", "int8int8"} {
		t0 := time.Now()
		r := decoder.ProbeBackendForTest("cuda", quant)
		t.Logf("%s: %s (wall %v)", quant, r.Summary(), time.Since(t0))
		if r.Status != decoder.SelfTestPass {
			t.Errorf("%s: status %s on healthy code: %s", quant, r.Status, r.Summary())
		}
	}
}

// Each mutation changes ONE input the resident consumes, on the GPU copy only. If the probe stays green for any of them, it checks less than it claims.
func TestResidentSelfTest_goesRedOnWrongResidentInputs(t *testing.T) {
	skipWithoutCUDA(t)
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
			r := decoder.ProbeBackendForTest("cuda", "int4")
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

func skipWithoutCUDA(t *testing.T) {
	t.Helper()
	m, err := decoder.Load("../testdata/llama-tiny", decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if _, ok := m.ResidentForwardForTest().(*cudaResident); !ok {
		t.Skip("no CUDA device, or the tiny model did not go resident here")
	}
}
