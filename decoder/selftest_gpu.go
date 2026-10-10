package decoder

import (
	"context"
	"embed"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// The resident self-test (docs/tasks/task-hardware-coverage-2026-10.md, H2): for a GPU backend, run tiny committed checkpoints through the RESIDENT path and through the CPU path already in this
// binary, and hold the resident logits to the CPU's with the numeric bars the resident-vs-CPU parity gates already use. It exercises the kernels the real model is about to use (the same resident
// build, not reimplementations), on a fixed deterministic input, and it runs once per backend and quant per process, after the real model's resident is built (so the device context is already held and
// the cost is the tiny models' own, not a context creation). On a mismatch the real model's resident is dropped and the model continues on the CPU path with the reason on its decode path,
// the existing decline route, instead of producing wrong numbers silently.
//
// The fixtures are four tiny random-weight checkpoints, chosen for the kernels they reach: qwen35vl-tiny (a 512-position
// window, so attention runs past 256 keys, beyond both softmax reduction widths, beside DeltaNet layers), llama-tiny (the
// plain dense path), gemma3-vl-tiny (GELU-tanh and QK-norm: the sink-token GELU NaN class) and qwen35-tiny (the DeltaNet
// hybrid with a MoE FFN). A fixture that declines to go resident on a backend checks nothing there, so it is not
// chosen for that backend. They are embedded copies of testdata/ ones, and a test asserts they stay byte-identical.

//go:embed selftestdata
var selfTestFS embed.FS

// Bars: the cosine and relative-L2 the shipped resident-vs-CPU parity gates hold (cuda/cohere_resident_parity_test.go). The probe invents no tighter bound, because a tighter one would decline healthy
// machines, and no looser one, because that would pass a real regression. Non-finite logits fail outright.
const (
	probeCosineFloor = 0.995
	probeRelL2Ceil   = 0.10
)

var selfTestFixtures = []struct {
	name      string
	positions int // at most the checkpoint's own context window (a test asserts it)
}{
	{"qwen35vl-tiny", 264}, // Qwen3.5 text hybrid: full attention past 256 keys, DeltaNet layers
	{"llama-tiny", 100},    // plain dense: RoPE, GQA, SwiGLU, the most common family
	{"gemma3-vl-tiny", 48}, // GELU-tanh and QK-norm
	{"qwen35-tiny", 48},    // DeltaNet + MoE FFN
}

var (
	gpuProbeMu    sync.Mutex
	gpuProbeCache = map[string]SelfTestResult{}
)

// probeMeasured lists the backends whose healthy-hardware margins against the bars were measured, each on ONE device
// (figures: docs/tasks/task-hardware-coverage-2026-10.md). A backend not on it is NOT probed (its result is simply
// absent from the report) rather than probed against bars nobody has seen it clear. WebGPU's margins come from one
// NVIDIA adapter over Vulkan, so AMD, Intel and DX12 adapters are probed against those bars unmeasured, a risk of a
// false decline. Metal's come from one M1 Pro; a fixture Metal re-quantizes to int4 is not compared
// (residentPrecision). Add a backend here with its margins.
var probeMeasured = map[string]bool{"cuda": true, "webgpu": true, "metal": true}

// gpuProbeAfterResident is called by withResidency once a resident is attached. It returns the (cached) result for this model's backend and quant, and whether a probe applies at all.
func gpuProbeAfterResident(m *Model) (SelfTestResult, bool) {
	backend := probeBackendName(m.be)
	if !probeMeasured[backend] || m.noSelfTest || SelfTestsSkipped() {
		return SelfTestResult{}, false
	}
	if why, skip := probeIneligible(m.be); skip {
		r := SelfTestResult{Backend: backend, Status: SelfTestSkipped, Note: why}
		RecordSelfTest(r)
		return r, true
	}
	return probeBackendCached(backend, m.Quant()), true
}

// probeBackendName is the registry name a Backend was registered under: WebGPU's Backend names itself "webgpu:vulkan" (the API it resolved to), and the probe loads fixtures by registry name.
func probeBackendName(be Backend) string { return strings.SplitN(be.Name(), ":", 2)[0] }

// selfTestEligibility is implemented by a Backend that knows it should not be probed on the device it resolved to. WebGPU declines on a software adapter (lavapipe, llvmpipe, SwiftShader): its limits
// and its math are not a GPU's, so the probe's bars, measured on a real adapter, would judge it by the wrong standard, and a CI runner or a headless box would pay the cost for nothing.
type selfTestEligibility interface {
	SelfTestEligible() (ok bool, why string)
}

// probeIneligible reports whether be asked not to be probed, and why.
func probeIneligible(be Backend) (why string, skip bool) {
	if e, ok := be.(selfTestEligibility); ok {
		if yes, w := e.SelfTestEligible(); !yes {
			return w, true
		}
	}
	return "", false
}

func probeBackendCached(backend, quant string) SelfTestResult {
	key := backend + "/" + quant
	gpuProbeMu.Lock()
	defer gpuProbeMu.Unlock()
	if r, ok := gpuProbeCache[key]; ok {
		return r
	}
	r := probeBackend(backend, quant)
	gpuProbeCache[key] = r
	RecordSelfTest(r)
	return r
}

// probeStandalone runs the probe for a backend with no real model loaded, for `check --hardware`: it creates the device context itself, so it costs more than the in-load probe, and says so.
func probeStandalone(backend string) SelfTestResult {
	if SelfTestsSkipped() {
		return SelfTestResult{Backend: backend, Status: SelfTestSkipped}
	}
	if !probeMeasured[backend] {
		r := SelfTestResult{Backend: backend, Status: SelfTestSkipped, Note: "no self-test for this backend yet: its kernels are not checked"}
		RecordSelfTest(r)
		return r
	}
	if be, err := NewBackend(backend); err == nil {
		why, skip := probeIneligible(be)
		_ = be.Close()
		if skip {
			r := SelfTestResult{Backend: backend, Status: SelfTestSkipped, Note: why}
			RecordSelfTest(r)
			return r
		}
	}
	return probeBackendCached(backend, "int4")
}

func probeBackend(backend, quant string) SelfTestResult {
	start := time.Now()
	res := SelfTestResult{Backend: backend, Status: SelfTestPass}
	ran := 0
	var notes []string
	worst := probeWorst{cos: 1}
	for _, fx := range selfTestFixtures {
		mm, skip, err := probeFixture(backend, quant, fx.name, fx.positions, &worst)
		if err != nil {
			res.Mismatches = append(res.Mismatches, fmt.Sprintf("%s: %v", fx.name, err))
			continue
		}
		if skip != "" {
			notes = append(notes, fx.name+": "+skip)
			continue
		}
		ran++
		res.Mismatches = append(res.Mismatches, mm...)
	}
	res.Elapsed = time.Since(start)
	if ran > 0 { // the margin a healthy run has against the bars, so a report shows how close to declining a machine is
		notes = append([]string{fmt.Sprintf("worst cosine %.5f (floor %.3f), worst relative L2 %.4f (ceiling %.2f) over %d fixtures", worst.cos, probeCosineFloor, worst.rel, probeRelL2Ceil, ran)}, notes...)
	}
	res.Note = strings.Join(notes, "; ") // a fixture that could not be compared is named even when the others passed
	switch {
	case len(res.Mismatches) > 0:
		res.Status = SelfTestDeclined
	case ran == 0:
		res.Status = SelfTestSkipped // no fixture could go resident on this backend at this quant: nothing was checked, and the report says so
	}
	return res
}

// probeFixture runs one fixture on the backend and on the CPU. skip is a reason the fixture could not be compared (it did not go resident here), which is not a failure; a non-nil err is a failure to
// run the probe at all.
func probeFixture(backend, quant, name string, positions int, worst *probeWorst) (mismatches []string, skip string, err error) {
	dir, err := os.MkdirTemp("", "goinfer-selftest-*")
	if err != nil {
		return nil, "cannot create a temp directory: " + err.Error(), nil
	}
	defer os.RemoveAll(dir)
	if err := writeSelfTestFixture(dir, name, nil); err != nil {
		return nil, "cannot write a temp file: " + err.Error(), nil
	}
	gdir := dir
	if probeGPUConfigMutator != nil { // test seam: the GPU model reads a deliberately wrong config, the CPU reference the true one
		gdir = filepath.Join(dir, "gpu")
		if err := writeSelfTestFixture(gdir, name, probeGPUConfigMutator); err != nil {
			return nil, "", err
		}
	}
	gm, err := Load(gdir, Options{Backend: backend, Quant: quant, ResidentContext: positions + 16, ResidentKVSlots: 1, noSelfTest: true})
	if err != nil {
		return nil, "cannot load on " + backend + ": " + err.Error(), nil
	}
	defer gm.Close()
	rf := gm.resident
	if rf == nil {
		return nil, "did not go resident on " + backend + ": " + gm.resDecline, nil
	}
	// A resident that computes another precision than the load quant (Metal re-quantizes int8, int4mix and the
	// int8int8 models its native int8 path does not cover to int4) cannot be judged against the CPU at the load quant:
	// that would compare int4 with int8int8 and decline a healthy device. Nor against the CPU at int4, which quantizes
	// from the original weights, not from the int8 ones the resident re-quantized. So it is not compared, and says so.
	if got := residentPrecision(backend, quant, rf); got != quant {
		return nil, fmt.Sprintf("re-quantized to %s on %s (loaded at %s); not probed", got, backend, quant), nil
	}
	cm, err := Load(dir, Options{Backend: "cpu", Quant: quant, noSelfTest: true})
	if err != nil {
		return nil, "", fmt.Errorf("load on cpu: %w", err)
	}
	defer cm.Close()
	_, _, _, _, _, _, vocab := cm.Dims()
	prompt := make([]int, positions)
	for i := range prompt {
		prompt[i] = (i*37 + 3) % vocab
	}

	// Decode: every position through the sequential resident path against the CPU forward.
	rf.Reset()
	cache := cm.NewCache(positions)
	worstCos, worstRel, worstAt := 1.0, 0.0, 0
	for i, tok := range prompt {
		lc, err := cm.forward(tok, cache)
		if err != nil {
			return nil, "", fmt.Errorf("cpu forward at %d: %w", i, err)
		}
		lr, err := rf.Forward(gm.embedResident(tok), i)
		if err != nil {
			return []string{fmt.Sprintf("%s decode: the resident forward failed at position %d: %v", name, i, err)}, "", nil
		}
		if bad := nonFinite(lr); bad {
			return []string{fmt.Sprintf("%s decode: non-finite logits at position %d", name, i)}, "", nil
		}
		cos, rel := compareLogits(lr, lc)
		worst.add(cos, rel)
		if cos < worstCos {
			worstCos, worstAt = cos, i
		}
		worstRel = math.Max(worstRel, rel)
	}
	if worstCos < probeCosineFloor || worstRel > probeRelL2Ceil {
		mismatches = append(mismatches, fmt.Sprintf("%s decode, %d positions: worst cosine %.4f (floor %.3f, at position %d), worst relative L2 %.3f (ceiling %.2f)", name, positions, worstCos, probeCosineFloor, worstAt, worstRel, probeRelL2Ceil))
	}

	// Batched prefill, when the backend has one: a real prompt takes this path. A decline is not a failure (the sequential path is then what runs, and was just checked).
	if pf, ok := rf.(Prefiller); ok {
		rf.Reset()
		embs := make([][]float32, positions)
		for i, tok := range prompt {
			embs[i] = gm.embedResident(tok)
		}
		got, perr := pf.PrefillLast(context.Background(), embs, 0)
		if perr == nil {
			want, err := cm.prefillLogits(context.Background(), prompt, cm.NewCache(positions))
			if err != nil {
				return mismatches, "", fmt.Errorf("cpu prefill: %w", err)
			}
			if nonFinite(got) {
				mismatches = append(mismatches, name+" batched prefill: non-finite logits")
			} else {
				cos, rel := compareLogits(got, want)
				worst.add(cos, rel)
				if cos < probeCosineFloor || rel > probeRelL2Ceil {
					mismatches = append(mismatches, fmt.Sprintf("%s batched prefill: cosine %.4f (floor %.3f), relative L2 %.3f (ceiling %.2f)", name, cos, probeCosineFloor, rel, probeRelL2Ceil))
				}
			}
		}
	}
	return mismatches, "", nil
}

// residentPrecision is the quant a resident actually computes at: what it reports (ResidentQuantReporter), else the int4
// residentQuantLabel names for a re-quantizing (backend, quant), else the load quant.
func residentPrecision(backend, quant string, rf ResidentForward) string {
	if qr, ok := rf.(ResidentQuantReporter); ok && qr.ResidentQuant() != "" {
		return qr.ResidentQuant()
	}
	if residentQuantLabel(backend, quant) != quant {
		return "int4"
	}
	return quant
}

// probeGPUConfigMutator, when set by a test (SetProbeGPUConfigMutatorForTest), rewrites config.json for the GPU copy of each fixture only: the mutation proof that the probe can go red.
var probeGPUConfigMutator func(name string, cfg []byte) []byte

// writeSelfTestFixture writes the embedded fixture into dir; mutate, if non-nil, rewrites config.json on the way.
func writeSelfTestFixture(dir, name string, mutate func(string, []byte) []byte) error {
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	ents, err := selfTestFS.ReadDir("selftestdata/" + name)
	if err != nil {
		return err
	}
	for _, e := range ents {
		b, err := selfTestFS.ReadFile("selftestdata/" + name + "/" + e.Name())
		if err != nil {
			return err
		}
		if mutate != nil && e.Name() == "config.json" {
			b = mutate(name, b)
		}
		if err := os.WriteFile(filepath.Join(dir, e.Name()), b, 0o600); err != nil {
			return err
		}
	}
	return nil
}

// probeWorst is the worst cosine and relative L2 seen across every comparison of a probe.
type probeWorst struct{ cos, rel float64 }

func (w *probeWorst) add(cos, rel float64) {
	w.cos = math.Min(w.cos, cos)
	w.rel = math.Max(w.rel, rel)
}

func nonFinite(x []float32) bool {
	for _, v := range x {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return true
		}
	}
	return false
}

// compareLogits is the cosine and the relative L2 of got against want. A cosine is blind to a uniform scale, which is why the relative L2 sits beside it (a missing logit scale is exactly one).
func compareLogits(got, want []float32) (cos, rel float64) {
	if len(got) != len(want) || len(want) == 0 {
		return -1, math.Inf(1)
	}
	var dot, ng, nw, d float64
	for i := range want {
		g, w := float64(got[i]), float64(want[i])
		dot += g * w
		ng += g * g
		nw += w * w
		e := g - w
		d += e * e
	}
	if ng == 0 || nw == 0 {
		return -1, math.Inf(1) // an all-zero logit vector is a defect, not a match
	}
	return dot / (math.Sqrt(ng) * math.Sqrt(nw)), math.Sqrt(d / nw)
}

// describeProbe is the one-line form a decline carries on the model's decode path.
func describeProbe(r SelfTestResult) string {
	return "startup self-test declined this backend: " + strings.Join(r.Mismatches, "; ")
}

// resetSelfTestCaches forgets the per-(backend, quant) probe results and the recorded GPU ones, for tests that need a first-use probe.
func resetSelfTestCaches() {
	gpuProbeMu.Lock()
	gpuProbeCache = map[string]SelfTestResult{}
	gpuProbeMu.Unlock()
	selfTestMu.Lock()
	for k := range selfTestResults {
		if k != "cpu" { // the CPU test runs once per process and is not part of the probe
			delete(selfTestResults, k)
		}
	}
	selfTestMu.Unlock()
}
