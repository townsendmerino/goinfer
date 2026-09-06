//go:build gpu

package gpu

import (
	"fmt"
	"math/rand"
	"os"
	"reflect"
	"sort"
	"testing"
	"time"

	"github.com/cogentcore/webgpu/wgpu"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestDecode_dispatchProfile answers "which of the ~13 dispatches/layer actually
// pins the decode critical path NOW" — the question every fusion proposal has to
// clear first, and the one this repo has already mispredicted twice (§0.0's
// link-counting missed by ~3×; the Increment-3 qk-norm fold shipped a −2.7%
// regression that the dependent-fold heuristic predicted as a win).
//
// Method: ABLATION, not attribution. For each distinct compute pipeline the
// resident plan uses, re-record the whole token plan R times into ONE pass with
// that pipeline's dispatches OMITTED, submit once, poll once (blocking → GPU
// done), and difference against the unablated plan. The delta / R is what
// deleting that class would save — barrier effects included, which is exactly the
// quantity a fusion buys. Attribution by per-kernel timing would NOT answer this:
// a dispatch that fully overlaps its neighbours costs wall-clock nothing to run
// and nothing to remove.
//
// Values go garbage after the first repetition (residual epilogues accumulate R
// times). That is deliberate and harmless — both arms are equally garbage, and on
// NVIDIA f32 there is no denormal/NaN timing cliff to bias the comparison. This
// is a timing harness; TestResidentForwardN_parity is the correctness gate.
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
	m, err := decoder.Load(path, decoder.Options{Quant: "int8int8"})
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
	if err != nil {
		t.Fatalf("BuildResident: %v", err)
	}
	if !ok {
		t.Skip("model did not go GPU-resident — the profile needs the resident plan")
	}
	// Close the RESIDENT model, not just the backend: its weight buffers are caller-owned and
	// b.Close() does not free them. Unreleased, this profiler held ~2.4 GB for the rest of the
	// process — the first of the two leaks that were emptying the GPU mid-suite.
	defer func() { _ = rf.Close() }()
	rd, isRD := rf.(*residentDecoder)
	if !isRD {
		t.Fatalf("BuildResident returned %T, want *residentDecoder", rf)
	}
	r := rd.runner
	c := r.c

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
	for _, s := range r.steps {
		cl, seen := byPL[s.pl]
		if !seen {
			nm := names[reflect.ValueOf(s.pl).Pointer()]
			if nm == "" {
				nm = fmt.Sprintf("pipeline@%p", s.pl)
			}
			cl = &class{pl: s.pl, name: nm}
			byPL[s.pl] = cl
			order = append(order, cl)
		}
		cl.count++
	}
	t.Logf("resident plan: %d dispatches/token across %d pipeline classes", len(r.steps), len(order))
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
	fillTo := func(pos int) {
		for p := 0; p <= pos; p++ {
			if _, err := r.Run(x, p); err != nil {
				t.Fatalf("warm Run(pos=%d): %v", p, err)
			}
		}
	}

	// timePlan records the plan R times into one pass (one Submit, one blocking Poll),
	// optionally omitting every dispatch of class skip. Min over reps trims scheduler noise.
	const R, reps = 8, 5
	timePlan := func(skip *wgpu.ComputePipeline) time.Duration {
		best := time.Hour
		for range reps {
			enc, err := c.device.CreateCommandEncoder(nil)
			if err != nil {
				t.Fatalf("encoder: %v", err)
			}
			pass := enc.BeginComputePass(nil)
			for range R {
				for _, s := range r.steps {
					if s.pl == skip {
						continue
					}
					pass.SetPipeline(s.pl)
					pass.SetBindGroup(0, s.bg, nil)
					pass.DispatchWorkgroups(s.gx, s.gy, 1)
				}
			}
			pass.End()
			pass.Release()
			cmd, err := enc.Finish(nil)
			if err != nil {
				t.Fatalf("finish: %v", err)
			}
			t0 := time.Now()
			c.queue.Submit(cmd)
			c.device.Poll(true, nil)
			d := time.Since(t0)
			cmd.Release()
			enc.Release()
			if d < best {
				best = d
			}
		}
		return best
	}

	for _, pos := range []int{64, 512} {
		fillTo(pos)
		full := timePlan(nil)
		perTok := full / R
		t.Logf("")
		t.Logf("=== pos=%d — full plan %.3f ms/token (%d dispatches) ===", pos, float64(perTok.Microseconds())/1000, len(r.steps))

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
			"TOTAL", len(r.steps), sum, sum/(float64(perTok.Microseconds())/1000)*100)
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
