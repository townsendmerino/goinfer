//go:build gpu

package gpu

import (
	"fmt"
	"github.com/townsendmerino/goinfer/gpu/internal/kernels"
	"math/rand"
	"os"
	"reflect"
	"slices"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/oliverbestmann/webgpu/wgpu"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestDecode_dispatchProfile answers which of the ~13 dispatches per layer pins the decode
// critical path: the question every fusion proposal has to clear first, and one this repo has
// mispredicted before (link-counting, and a qk-norm fold the dependent-fold heuristic predicted
// as a win).
//
// Method: ablation, not attribution. For each distinct compute pipeline the resident plan uses,
// re-record the whole token plan R times into ONE pass with that pipeline's dispatches omitted,
// submit once, poll once (blocking), and difference against the unablated plan. The delta / R is
// what deleting that class would save, barrier effects included, which is exactly what a fusion
// buys. Per-kernel timing would not answer it: a dispatch that fully overlaps its neighbours
// costs nothing to run and nothing to remove.
//
// Values go garbage after the first repetition (residual epilogues accumulate R times). Both arms
// are equally garbage and NVIDIA f32 has no denormal/NaN timing cliff to bias the comparison.
// This is a timing harness; TestResidentForwardN_parity is the correctness gate.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags gpu -run TestDecode_dispatchProfile -v ./gpu/
func TestDecode_dispatchProfile(t *testing.T) {
	requireHeavyModel(t)
	if testing.Short() {
		t.Skip("dispatch profile")
	}
	path := os.Getenv("GOINFER_DECODE_GGUF")
	if path == "" {
		path = os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("model not found: %s (set GOINFER_DECODE_GGUF)", path)
	}
	newOrSkipHW(t).Close() // real-HW gate

	// Load CPU-side, then build residency through the backend directly: the profiler
	// needs the *DecodeRunner itself, which decoder.Load's internal wiring does not expose.
	// GOINFER_DECODE_POS lists the depths profiled (default 64,512); the resident context is sized to the deepest.
	positions := []int{64, 512}
	if v := os.Getenv("GOINFER_DECODE_POS"); v != "" {
		positions = nil
		for _, f := range strings.Split(v, ",") {
			n, err := strconv.Atoi(f)
			if err != nil || n < 1 {
				t.Fatalf("GOINFER_DECODE_POS %q: %v", v, err)
			}
			positions = append(positions, n)
		}
	}
	m, err := decoder.Load(path, decoder.Options{Quant: "int8int8", ResidentContext: slices.Max(positions) + 64})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()

	b, err := newWebGPUBackend("dispatch-profile")
	if err != nil {
		t.Skipf("no webgpu: %v", err)
	}
	defer b.Close()

	rf, ok, err := b.BuildResident(m)
	if err != nil && !decoder.IsResidentDecline(err) {
		t.Fatalf("BuildResident: %v", err)
	}
	if !ok {
		t.Skip("model did not go GPU-resident — the profile needs the resident plan")
	}
	// Close the RESIDENT model, not just the backend: its weight buffers are caller-owned and
	// b.Close() does not free them, so unreleased they stay live for the rest of the process and
	// starve later tests of the GPU.
	defer func() { _ = rf.Close() }()
	rd, isRD := rf.(*residentDecoder)
	if !isRD {
		t.Fatalf("BuildResident returned %T, want *residentDecoder", rf)
	}
	r := rd.runner
	c := (*kernels.DecodeRunnerC(r))

	hidden, nLayers, nH, nKV, hd, inter, vocab := m.Dims()
	names := pipelineFieldNames(c)

	// Step census by pipeline: how many dispatches/token each class contributes.
	type class struct {
		pl    *wgpu.ComputePipeline
		name  string
		count int
	}
	byPL := map[*wgpu.ComputePipeline]*class{}
	var order []*class
	for _, s := range *kernels.DecodeRunnerSteps(r) {
		cl, seen := byPL[s.Pl]
		if !seen {
			nm := names[reflect.ValueOf(s.Pl).Pointer()]
			if nm == "" {
				nm = fmt.Sprintf("pipeline@%p", s.Pl)
			}
			cl = &class{pl: s.Pl, name: nm}
			byPL[s.Pl] = cl
			order = append(order, cl)
		}
		cl.count++
	}
	t.Logf("resident plan: %d dispatches/token across %d pipeline classes", len((*kernels.DecodeRunnerSteps(r))), len(order))
	t.Logf("geometry: hidden=%d layers=%d nH=%d nKV=%d headDim=%d inter=%d vocab=%d  (kvDim=%d, KV bytes/pos/layer=%d)",
		hidden, nLayers, nH, nKV, hd, inter, vocab, nKV*hd, 2*nKV*hd*4)

	// Warm the pipelines and fill the KV cache, so attention is profiled against a
	// realistic nKeys rather than the near-empty cache of pos 0 (where QK^T and the
	// softmax are trivial and the class would read as free).
	rng := rand.New(rand.NewSource(7))
	x := make([]float32, hidden)
	for i := range x {
		x[i] = float32(rng.NormFloat64()) * 0.02
	}

	// timePlan records the plan R times into one pass (one Submit, one blocking Poll),
	// optionally omitting every dispatch of class skip. Min over reps trims scheduler noise.
	const R, reps = 8, 5
	timePlan := func(skip *wgpu.ComputePipeline) time.Duration {
		best := time.Hour
		for range reps {
			enc, err := (*kernels.ContextDevice(c)).TryCreateCommandEncoder(nil)
			if err != nil {
				t.Fatalf("encoder: %v", err)
			}
			pass := enc.BeginComputePass(nil)
			for range R {
				for _, s := range *kernels.DecodeRunnerSteps(r) {
					if s.Pl == skip {
						continue
					}
					pass.SetPipeline(s.Pl)
					pass.SetBindGroup(0, s.Bg, nil)
					pass.DispatchWorkgroups(s.Gx, s.Gy, 1)
				}
			}
			pass.TryEnd()
			pass.Release()
			cmd, err := enc.TryFinish(nil)
			if err != nil {
				t.Fatalf("finish: %v", err)
			}
			t0 := time.Now()
			(*kernels.ContextQueue(c)).Submit(cmd)
			(*kernels.ContextDevice(c)).Poll(true, nil)
			d := time.Since(t0)
			cmd.Release()
			enc.Release()
			if d < best {
				best = d
			}
		}
		return best
	}

	filled := -1
	for _, pos := range positions {
		for p := filled + 1; p <= pos; p++ {
			if _, err := r.Run(x, p, p); err != nil {
				t.Fatalf("warm Run(pos=%d): %v", p, err)
			}
		}
		filled = pos
		full := timePlan(nil)
		perTok := full / R
		t.Logf("")
		t.Logf("=== pos=%d — full plan %.3f ms/token (%d dispatches) ===", pos, float64(perTok.Microseconds())/1000, len((*kernels.DecodeRunnerSteps(r))))

		type row struct {
			name    string
			count   int
			savedMS float64
			pct     float64
		}
		var rows []row
		var sum float64
		for _, cl := range order {
			ab := timePlan(cl.pl)
			saved := float64((full - ab).Microseconds()) / 1000 / float64(R)
			pct := saved / (float64(perTok.Microseconds()) / 1000) * 100
			rows = append(rows, row{cl.name, cl.count, saved, pct})
			sum += saved
		}
		sort.Slice(rows, func(i, j int) bool { return rows[i].savedMS > rows[j].savedMS })
		t.Logf("%-26s %8s %12s %8s", "pipeline class", "n/token", "ablation ms", "% token")
		for _, rw := range rows {
			t.Logf("%-26s %8d %12.3f %7.1f%%", rw.name, rw.count, rw.savedMS, rw.pct)
		}
		t.Logf("%-26s %8d %12.3f %7.1f%%  (sum of parts; >100%% ⇒ overlap, <100%% ⇒ unattributed)",
			"TOTAL", len((*kernels.DecodeRunnerSteps(r))), sum, sum/(float64(perTok.Microseconds())/1000)*100)
	}
}

// pipelineFieldNames maps every non-nil *wgpu.ComputePipeline field of the Context
// to its Go field name, so the profile prints "attnPipeline" instead of an address.
// Uses reflect.Value.Pointer (legal on unexported fields; Interface() would not be)
// so it stays correct as pipelines are added — a hand-maintained list would drift,
// which is the same failure mode Context.releases was rewritten to avoid.
func pipelineFieldNames(c *Context) map[uintptr]string {
	out := map[uintptr]string{}
	v := reflect.ValueOf(c).Elem()
	tt := v.Type()
	want := reflect.TypeFor[*wgpu.ComputePipeline]()
	for i := range tt.NumField() {
		f := v.Field(i)
		if f.Kind() == reflect.Pointer && f.Type() == want && !f.IsNil() {
			out[f.Pointer()] = tt.Field(i).Name
		}
	}
	return out
}
