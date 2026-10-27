//go:build cuda && goinfer_testhooks

package cuda

import (
	"errors"
	"flag"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/sb2census"
)

// Layer B-cuda of the admission and routing census (docs/tasks/task-subsystems-2026-10.md, "SB2 step 2"): CUDA's own static
// admission, the declines buildResident makes from the model alone, on layer D's fixtures and option cells. A fake backend named
// cuda takes the decoder's real admission; where that admits, it runs buildResident in plan-only mode (every static decline, then
// a return before anything is allocated), records the answer and declines, so nothing is uploaded and no device is needed. A cell
// the decoder declines is layer D's line (decoder's TestAdmissionCensus) and is not repeated here.

var cudaCensusUpdate = flag.Bool("sb2census.update", false, "rewrite this architecture's testdata/sb2_cuda_census.<goarch>.txt.gz")

// censusBackend is the fake: named cuda, it records the plan-only build of what the decoder admitted and declines.
type censusBackend struct {
	decoder.Backend
	mu      sync.Mutex
	reached bool
	answer  string
}

func (b *censusBackend) Name() string { return "cuda" }

// SelfTestEligible keeps the decoder's startup self-test (CUDA kernels on the device) out of the census: that is a device decision.
func (b *censusBackend) SelfTestEligible() (bool, string) { return false, "admission census" }

func (b *censusBackend) BuildResident(m *decoder.Model) (decoder.ResidentForward, bool, error) {
	_, _, err := (&cudaBackend{}).buildResident(m, true)
	var s planOnlySlack
	ans := "plan returned no answer"
	switch {
	case errors.As(err, &s):
		ans = fmt.Sprintf("plan slack=%d", int64(s))
	case err != nil:
		ans = "decline " + err.Error()
	}
	b.mu.Lock()
	b.reached, b.answer = true, ans
	b.mu.Unlock()
	return nil, false, decoder.DeclineResident("admission census: plan only")
}

func (b *censusBackend) take() (bool, string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	r, a := b.reached, b.answer
	b.reached, b.answer = false, ""
	return r, a
}

// buildCUDACensus returns one line per cell that reached CUDA's build. Every cell layer D built resident must be among them.
func buildCUDACensus(t *testing.T) []string {
	t.Helper()
	t.Setenv("GOINFER_SSM_RESIDENT", "")
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "")
	// The fit guard's host reading is fixed, so no reading of this machine decides a line; the fake's registered name has no
	// memory probe or build demand of its own, as in layer D.
	t.Cleanup(decoder.SetHostRAMAvailableForTest(48 << 30))
	be := &censusBackend{}
	const reg = "sb2-census-cuda"
	decoder.RegisterBackend(reg, func() (decoder.Backend, error) {
		cpu, err := decoder.NewBackend("cpu")
		if err != nil {
			return nil, err
		}
		be.Backend = cpu
		return be, nil
	})
	t.Cleanup(func() { decoder.UnregisterBackendForTest(reg) })
	d, err := sb2census.ReadLayerD("..", "cuda")
	if err != nil {
		t.Fatal(err)
	}
	if len(d.Fixtures) < 20 {
		t.Fatalf("only %d fixtures in layer D's golden: a census that read little covers little", len(d.Fixtures))
	}
	var lines []string
	for _, f := range d.Fixtures {
		path, err := sb2census.Path("..", f)
		if err != nil {
			t.Fatal(err)
		}
		for _, c := range sb2census.Cells {
			o := decoder.Options{Backend: reg, Quant: c.Quant, ActQuantGroup: c.ActGroup, KVPrecision: c.KVPrec, KVQuant: c.KVQuant}
			if c.SSM != "" {
				o.Knobs = &decoder.Knobs{"GOINFER_SSM_RESIDENT": c.SSM}
			}
			m, err := decoder.Load(path, o)
			reached, ans := be.take()
			if err == nil {
				m.Close()
			}
			// A reached cell is recorded whatever the load did next: the fake's decline moves the KV to the CPU cache, so a KV
			// refusal after the build is the fake's, not the model's.
			if reached {
				lines = append(lines, fmt.Sprintf("B %s %s cuda %q", f, c.Name, sb2census.Clean(ans)))
			} else if d.Resident[f+" "+c.Name] {
				t.Errorf("%s %s: layer D builds it resident on cuda, but CUDA's build was not reached", f, c.Name)
			}
		}
	}
	return lines
}

// TestBackendAdmissionCensus_cuda holds layer B-cuda to testdata/sb2_cuda_census.<goarch>.txt.gz, line for line, built twice.
func TestBackendAdmissionCensus_cuda(t *testing.T) {
	start := time.Now()
	a, b := buildCUDACensus(t), buildCUDACensus(t)
	if strings.Join(a, "\n") != strings.Join(b, "\n") {
		t.Fatalf("two builds in one process differ")
	}
	t.Logf("%d census lines in %s", len(a), time.Since(start).Round(time.Millisecond))
	golden := sb2census.Golden("testdata", "sb2_cuda_census")
	if *cudaCensusUpdate {
		if err := sb2census.Write(golden, a); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote %s", golden)
		return
	}
	diffs, err := sb2census.Diff(golden, a, 4)
	if err != nil {
		for _, l := range a {
			fmt.Println("SB2CUDACENSUS " + l)
		}
		t.Fatalf("%v: the lines above (prefixed SB2CUDACENSUS) are this architecture's census", err)
	}
	if len(diffs) > 0 {
		t.Errorf("the cuda admission census differs from %s in %d cell(s):\n%s", golden, len(diffs), strings.Join(diffs[:min(len(diffs), 30)], "\n"))
	}
}

// TestBackendAdmissionCensus_cuda_plant shows the census can go red: with the per-32 MoE decline planted off, every MoE cell at
// per-32 activations that reaches CUDA's build must differ from the golden, and no other cell may.
func TestBackendAdmissionCensus_cuda_plant(t *testing.T) {
	golden := sb2census.Golden("testdata", "sb2_cuda_census")
	clean := buildCUDACensus(t)
	const reason = "per-32 activations: the MoE expert/router kernels"
	wantRed := 0
	for _, l := range clean {
		if strings.Contains(l, reason) {
			wantRed++
		}
	}
	if wantRed == 0 {
		t.Fatal("no cell takes the planted decline: the plant has nothing to change")
	}
	defer func(off bool) { moeActG32DeclineOffForTest = off }(moeActG32DeclineOffForTest)
	moeActG32DeclineOffForTest = true
	diffs, err := sb2census.Diff(golden, buildCUDACensus(t), 4)
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range diffs {
		if !strings.Contains(d, reason) {
			t.Errorf("planted MoE per-32 decline off: a cell outside the plant changed: %s", d)
		}
	}
	if len(diffs) != wantRed {
		t.Fatalf("planted MoE per-32 decline off: %d cell(s) differ from the golden, want %d", len(diffs), wantRed)
	}
	t.Logf("planted MoE per-32 decline off: %d of %d cells red, as expected", len(diffs), len(clean))
}
