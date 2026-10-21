//go:build darwin && goinfer_testhooks

package metal

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// G2-R amendment 3 (docs/tasks/task-forward-architecture-2026-10.md §4.6, "G2-R amendment 3"): the Metal process loads
// Metal .giw bundles, a refused load voids the run, and every load is fit-checked by day before the job is queued.

// g2rSmall is the Metal precondition's model: its .gguf loads on Metal directly.
func g2rSmall() string { return os.ExpandEnv("$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf") }

// g2rMetalPath is the Metal .giw for a checkpoint at a quant: $GOINFER_G2R_GIW_DIR/<checkpoint base>.<quant>.metal.giw.
// The int8int8 ones are hard links to the CPU bundles, which are the same bytes. A missing bundle voids the run.
func g2rMetalPath(t *testing.T, gguf, quant string) string {
	t.Helper()
	dir := os.Getenv("GOINFER_G2R_GIW_DIR")
	if dir == "" {
		t.Fatalf("VOID: GOINFER_G2R_GIW_DIR is not set: the Metal arms load .giw bundles (amendment 3)")
	}
	p := filepath.Join(dir, strings.TrimSuffix(filepath.Base(gguf), ".gguf")+"."+quant+".metal.giw")
	if _, err := os.Stat(p); err != nil {
		t.Fatalf("VOID: the Metal bundle for %s at %s is missing: %v", filepath.Base(gguf), quant, err)
	}
	return p
}

// g2rLoad loads load on backend ("" is the CPU). A load the fit guard refuses, or one that fails, measures nothing, so it
// voids the run rather than failing G2-R.
func g2rLoad(t *testing.T, load, backend, quant string) *decoder.Model {
	t.Helper()
	m, err := decoder.Load(load, decoder.Options{Backend: backend, Quant: quant})
	if err != nil {
		t.Fatalf("VOID: the %s load of %s at %s was refused or failed: %v", g2rBackendName(backend), filepath.Base(load), quant, err)
	}
	return m
}

func g2rBackendName(b string) string {
	if b == "" {
		return "cpu"
	}
	return b
}

// g2rResident is m's Metal resident; a resident Metal's memory guard declined voids the run, and any other decline
// fails it (admission said the model fits).
func g2rResident(t *testing.T, m *decoder.Model, what string) decoder.ResidentForward {
	t.Helper()
	rf := m.ResidentForwardForTest()
	if rf == nil {
		if why := m.ResidentDecline(); strings.Contains(why, residentMemoryDeclinePrefix) {
			t.Fatalf("VOID: Metal's memory guard declined the resident for %s: %s", what, why)
		}
		t.Fatalf("metal resident DECLINED for %s (%s): admission says it should be admitted", what, m.ResidentDecline())
	}
	return rf
}

// g2rMetalRun runs load on Metal over inputs, or, when inputs is nil, over seed and the resident's own greedy
// continuation to steps tokens. It returns the inputs fed and the logits at each step.
func g2rMetalRun(t *testing.T, load, quant string, seed, inputs []int, steps int) ([]int, [][]float32) {
	t.Helper()
	m := g2rLoad(t, load, "metal", quant)
	defer m.Close()
	rf := g2rResident(t, m, filepath.Base(load))
	if inputs != nil {
		steps = len(inputs)
	}
	var fed []int
	var out [][]float32
	var tok int
	if inputs != nil {
		tok = inputs[0]
	} else {
		tok = seed[0]
	}
	for i := range steps {
		l, err := rf.Forward(m.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("gpu forward: %v", err)
		}
		fed = append(fed, tok)
		out = append(out, append([]float32(nil), l...))
		switch {
		case inputs != nil && i+1 < len(inputs):
			tok = inputs[i+1]
		case inputs == nil && i+1 < len(seed):
			tok = seed[i+1]
		default:
			tok = argmaxF(l)
		}
	}
	return fed, out
}

// g2rMetalPrecondition is amendment 3's precondition: on Metal, the 0.5B through its .giw gives the logits its direct
// .gguf load gives, bit for bit, at every step, at int8int8 and int4, fed the .gguf route's greedy tokens. A difference
// voids the run.
func g2rMetalPrecondition(t *testing.T) {
	t.Helper()
	small := g2rSmall()
	seed := seedPrompt(t, small, probeText)
	for _, q := range []string{"int8int8", "int4"} {
		fed, direct := g2rMetalRun(t, small, q, seed, nil, 24)
		_, viaGIW := g2rMetalRun(t, g2rMetalPath(t, small, q), q, nil, fed, 0)
		for i := range direct {
			for j := range direct[i] {
				if math.Float32bits(direct[i][j]) != math.Float32bits(viaGIW[i][j]) {
					t.Fatalf("VOID (amendment 3's precondition): the 0.5B on Metal at %s through its .giw differs from its direct .gguf load at step %d, logit %d", q, i, j)
				}
			}
		}
		t.Logf("Metal precondition: the 0.5B at %s gives the same logits through its .giw and its .gguf, every step, bit for bit", q)
	}
}

// TestG2R_fitCheck loads, one at a time and alone, every model the two processes load, on its backend and at its quant,
// and reports each one admitted or refused (CLAUDE.md, "Testing tiers": fit-check every night load by day). A Metal
// load also builds its resident, so Metal's memory guard is checked too. It fails when any load is refused; the night
// job is queued only after it passes.
func TestG2R_fitCheck(t *testing.T) {
	requireHeavyModel(t)
	if os.Getenv("GOINFER_G2R_FITCHECK") != "1" {
		t.Skip("set GOINFER_G2R_FITCHECK=1 and GOINFER_G2R_GIW_DIR: G2-R's by-day fit check")
	}
	g2, g3, dense := g2rPaths()
	small := g2rSmall()
	type load struct{ path, backend, quant string }
	var loads []load
	// The CPU process: the precondition's two routes, then the arms' bundles.
	for _, q := range []string{"int8int8", "int4"} {
		loads = append(loads, load{dense, "", q}, load{g2rCPUPath(t, dense, q), "", q}, load{g2rCPUPath(t, g2, q), "", q})
	}
	loads = append(loads, load{g2rCPUPath(t, g3, "int8int8"), "", "int8int8"})
	// The Metal process: the precondition's two routes, then the arms' bundles.
	for _, q := range []string{"int8int8", "int4"} {
		loads = append(loads, load{small, "metal", q}, load{g2rMetalPath(t, small, q), "metal", q}, load{g2rMetalPath(t, g2, q), "metal", q})
	}
	loads = append(loads, load{g2rMetalPath(t, dense, "int8int8"), "metal", "int8int8"}, load{g2rMetalPath(t, g3, "int8int8"), "metal", "int8int8"})
	refused := 0
	for i, l := range loads {
		what := fmt.Sprintf("%2d/%d %-5s %-8s %s", i+1, len(loads), g2rBackendName(l.backend), l.quant, filepath.Base(l.path))
		m, err := decoder.Load(l.path, decoder.Options{Backend: l.backend, Quant: l.quant})
		if err != nil {
			refused++
			fmt.Fprintf(os.Stderr, "[g2r-fit] %s: REFUSED: %v\n", what, err)
			continue
		}
		verdict := "admitted"
		if l.backend == "metal" && m.ResidentForwardForTest() == nil {
			refused++
			verdict = "REFUSED (resident): " + m.ResidentDecline()
		}
		m.Close()
		fmt.Fprintf(os.Stderr, "[g2r-fit] %s: %s; %.1f GiB available after close\n", what, verdict, float64(decoder.HostRAMAvailableBytes())/(1<<30))
	}
	if refused > 0 {
		t.Fatalf("%d of %d loads refused by day: do not queue the night job", refused, len(loads))
	}
}
