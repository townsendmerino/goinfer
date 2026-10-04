//go:build darwin && goinfer_testhooks

package metal

import (
	"bytes"
	"regexp"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// H2 on Metal (docs/tasks/task-hardware-coverage-2026-10.md): the resident self-test on a real Apple GPU. A green probe
// on healthy code, and the same probe going RED when the resident's inputs are wrong, which is what makes the green mean
// something. Metal re-quantizes some loads to int4 on the GPU (docs/tasks/task-metal-int8-2026-10.md); a fixture whose
// resident computes another precision than the load quant is not compared (decoder's residentPrecision), so the probe
// never judges an int4 resident against an int8int8 CPU reference.

// probeQuants are the quants the Metal probe runs: int4 (every fixture goes resident at int4) and int8int8 (native on
// the dense fixtures, llama-tiny and gemma3-vl-tiny; Metal re-quantizes the DeltaNet ones, qwen35vl-tiny and qwen35-tiny,
// to int4, and the probe skips those at int8int8). int4mix is not probed: it is GGUF-only and every fixture is
// safetensors, so none loads (a real int4mix model's probe records "skipped", nothing checked).
var probeQuants = []string{"int4", "int8int8"}

// reQuantized are the fixtures Metal re-quantizes to int4 at each quant, which the probe must name as not probed.
var reQuantized = map[string][]string{"int8int8": {"qwen35vl-tiny", "qwen35-tiny"}}

// probeFixtures are the decoder's self-test fixtures (decoder/selftestdata, byte-identical copies of these).
var probeFixtures = []string{"qwen35vl-tiny", "llama-tiny", "gemma3-vl-tiny", "qwen35-tiny"}

func requireMetal(t *testing.T) {
	t.Helper()
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skip("no Metal device: " + err.Error())
	}
	d.ReleaseObjects()
}

// TestResidentSelfTest_fixturePathsOnMetal logs, for every probe fixture and quant, the decode path Metal gives it, so a
// reader sees which fixtures the probe compares at each quant and which ones Metal re-quantizes or keeps off the GPU.
func TestResidentSelfTest_fixturePathsOnMetal(t *testing.T) {
	requireMetal(t)
	for _, quant := range []string{"int4", "int8int8", "int4mix"} {
		for _, fx := range probeFixtures {
			m, err := decoder.Load("../testdata/"+fx, decoder.Options{Backend: "metal", Quant: quant})
			if err != nil {
				t.Logf("%-8s %-15s load: %v", quant, fx, err)
				continue
			}
			t.Logf("%-8s %-15s %s", quant, fx, m.DecodePath())
			m.Close()
		}
	}
}

func TestResidentSelfTest_passesOnHealthyMetal(t *testing.T) {
	requireMetal(t)
	// A model holding the device, as the real model does when the probe runs after its resident is built.
	hold, err := decoder.Load("../testdata/llama-tiny", decoder.Options{Backend: "metal", Quant: "int4"})
	if err != nil {
		t.Fatal(err)
	}
	defer hold.Close()
	for _, quant := range probeQuants {
		t0 := time.Now()
		r := decoder.ProbeBackendForTest("metal", quant)
		t.Logf("%s: %s (wall %v)", quant, r.Summary(), time.Since(t0))
		if r.Status != decoder.SelfTestPass {
			t.Errorf("%s: status %s on healthy code: %s", quant, r.Status, r.Summary())
		}
		for _, fx := range reQuantized[quant] {
			if !bytes.Contains([]byte(r.Note), []byte(fx+": re-quantized to int4 on metal")) {
				t.Errorf("%s: %s is re-quantized on Metal, and the note does not say it was not probed: %s", quant, fx, r.Note)
			}
		}
	}
}

// Each mutation changes ONE input the resident consumes, on the GPU copy of one fixture only. The test also requires that
// the regex changed the config and that the mutated fixture really went resident on Metal: a first CUDA version passed
// vacuously because its mutated fixture (Phi-3) never went resident and the other fixtures carried the pass.
func TestResidentSelfTest_goesRedOnWrongResidentInputs(t *testing.T) {
	requireMetal(t)
	const target = "qwen35vl-tiny"
	if m, err := decoder.Load("../testdata/"+target, decoder.Options{Backend: "metal", Quant: "int4"}); err != nil {
		t.Fatalf("load %s: %v", target, err)
	} else {
		ok := m.ResidentForwardForTest() != nil
		dp := m.DecodePath()
		m.Close()
		if !ok {
			t.Fatalf("%s does not go resident on Metal at int4 (%s): mutating it would prove nothing", target, dp)
		}
	}
	muts := []struct{ name, re, to string }{
		{"rms epsilon", `"rms_norm_eps":\s*[0-9.eE+-]+`, `"rms_norm_eps": 0.5`},
		{"rope base", `"rope_theta":\s*[0-9.eE+-]+`, `"rope_theta": 3.0`},
	}
	for _, mu := range muts {
		t.Run(mu.name, func(t *testing.T) {
			re := regexp.MustCompile(mu.re)
			changed := 0
			restore := decoder.SetProbeGPUConfigMutatorForTest(func(fixture string, cfg []byte) []byte {
				if fixture != target {
					return cfg
				}
				out := re.ReplaceAll(cfg, []byte(mu.to))
				if !bytes.Equal(out, cfg) {
					changed++
				}
				return out
			})
			defer restore()
			r := decoder.ProbeBackendForTest("metal", "int4")
			t.Logf("%s", r.Summary())
			if changed == 0 {
				t.Fatalf("the mutation %q matched nothing in %s's config, so this proves nothing", mu.name, target)
			}
			if r.Status != decoder.SelfTestDeclined {
				t.Errorf("a wrong %s on the GPU copy left the probe at %s; it must decline", mu.name, r.Status)
			}
			named := false
			for _, mm := range r.Mismatches {
				if bytes.Contains([]byte(mm), []byte(target)) {
					named = true
				}
			}
			if !named {
				t.Errorf("the decline does not name the mutated fixture %s: %v", target, r.Mismatches)
			}
		})
	}
}
