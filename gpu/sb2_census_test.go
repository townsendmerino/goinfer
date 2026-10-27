//go:build gpu && goinfer_testhooks

package gpu

import (
	"flag"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/sb2census"
)

// Layer B-webgpu of the admission and routing census (docs/tasks/task-subsystems-2026-10.md, "SB2 step 2"): WebGPU's own static
// admission, staticResidentDecline, on layer D's fixtures and option cells. A fake backend named webgpu takes the decoder's real
// admission; where that admits, it records staticResidentDecline's answer and declines, so no device is opened. A cell the decoder
// declines is layer D's line (decoder's TestAdmissionCensus) and is not repeated here.

var webgpuCensusUpdate = flag.Bool("sb2census.update", false, "rewrite this architecture's testdata/sb2_webgpu_census.<goarch>.txt.gz")

// censusBackend is the fake: named webgpu, it records the static admission of what the decoder admitted and declines.
type censusBackend struct {
	decoder.Backend
	mu      sync.Mutex
	reached bool
	answer  string
}

func (b *censusBackend) Name() string { return "webgpu" }

// SelfTestEligible keeps the decoder's startup self-test (WebGPU kernels on the device) out of the census: that is a device decision.
func (b *censusBackend) SelfTestEligible() (bool, string) { return false, "admission census" }

func (b *censusBackend) BuildResident(m *decoder.Model) (decoder.ResidentForward, bool, error) {
	ans := "static ok"
	if err := staticResidentDecline(m); err != nil {
		ans = "decline " + err.Error()
	}
	b.mu.Lock()
	b.reached, b.answer = true, ans
	b.mu.Unlock()
	return nil, false, decoder.DeclineResident("admission census: static gates only")
}

func (b *censusBackend) take() (bool, string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	r, a := b.reached, b.answer
	b.reached, b.answer = false, ""
	return r, a
}

// buildWebGPUCensus returns one line per cell that reached WebGPU's build. Every cell layer D built resident must be among them.
func buildWebGPUCensus(t *testing.T) []string {
	t.Helper()
	t.Setenv("GOINFER_SSM_RESIDENT", "")
	// The fit guard's host reading is fixed, so no reading of this machine decides a line; the fake's registered name has no
	// memory probe or build demand of its own, as in layer D.
	t.Cleanup(decoder.SetHostRAMAvailableForTest(48 << 30))
	be := &censusBackend{}
	const reg = "sb2-census-webgpu"
	decoder.RegisterBackend(reg, func() (decoder.Backend, error) {
		cpu, err := decoder.NewBackend("cpu")
		if err != nil {
			return nil, err
		}
		be.Backend = cpu
		return be, nil
	})
	d, err := sb2census.ReadLayerD("..", "webgpu")
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
				lines = append(lines, fmt.Sprintf("B %s %s webgpu %q", f, c.Name, sb2census.Clean(ans)))
			} else if d.Resident[f+" "+c.Name] {
				t.Errorf("%s %s: layer D builds it resident on webgpu, but WebGPU's build was not reached", f, c.Name)
			}
		}
	}
	return lines
}

// TestBackendAdmissionCensus_webgpu holds layer B-webgpu to testdata/sb2_webgpu_census.<goarch>.txt.gz, line for line, built twice.
func TestBackendAdmissionCensus_webgpu(t *testing.T) {
	start := time.Now()
	a, b := buildWebGPUCensus(t), buildWebGPUCensus(t)
	if strings.Join(a, "\n") != strings.Join(b, "\n") {
		t.Fatalf("two builds in one process differ")
	}
	t.Logf("%d census lines in %s", len(a), time.Since(start).Round(time.Millisecond))
	golden := sb2census.Golden("testdata", "sb2_webgpu_census")
	if *webgpuCensusUpdate {
		if err := sb2census.Write(golden, a); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote %s", golden)
		return
	}
	diffs, err := sb2census.Diff(golden, a, 4)
	if err != nil {
		for _, l := range a {
			fmt.Println("SB2WEBGPUCENSUS " + l)
		}
		t.Fatalf("%v: the lines above (prefixed SB2WEBGPUCENSUS) are this architecture's census", err)
	}
	if len(diffs) > 0 {
		t.Errorf("the webgpu admission census differs from %s in %d cell(s):\n%s", golden, len(diffs), strings.Join(diffs[:min(len(diffs), 30)], "\n"))
	}
}

// TestBackendAdmissionCensus_webgpu_plant shows the census can go red: with the quantized-KV layout decline planted off, every
// cell that takes it must differ from the golden, and no other cell may.
func TestBackendAdmissionCensus_webgpu_plant(t *testing.T) {
	golden := sb2census.Golden("testdata", "sb2_webgpu_census")
	clean := buildWebGPUCensus(t)
	const reason = "KV on webgpu (only the generic GQA path does)"
	wantRed := 0
	for _, l := range clean {
		if strings.Contains(l, reason) {
			wantRed++
		}
	}
	if wantRed == 0 {
		t.Fatal("no cell takes the planted decline: the plant has nothing to change")
	}
	defer func(off bool) { quantKVDeclineOffForTest = off }(quantKVDeclineOffForTest)
	quantKVDeclineOffForTest = true
	diffs, err := sb2census.Diff(golden, buildWebGPUCensus(t), 4)
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range diffs {
		if !strings.Contains(d, reason) {
			t.Errorf("planted quantized-KV decline off: a cell outside the plant changed: %s", d)
		}
	}
	if len(diffs) != wantRed {
		t.Fatalf("planted quantized-KV decline off: %d cell(s) differ from the golden, want %d", len(diffs), wantRed)
	}
	t.Logf("planted quantized-KV decline off: %d of %d cells red, as expected", len(diffs), len(clean))
}
