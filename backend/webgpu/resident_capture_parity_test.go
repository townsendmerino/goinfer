//go:build gpu && goinfer_testhooks

package webgpu

import (
	"cmp"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/backend/webgpu/internal/kernels"
	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/testgate"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestResidentCaptureParityWebGPU runs the resident Forward against the CPU ForwardForTest at the same quant (an 8-token
// arbitrary prompt, then a greedy continuation) through the PRODUCTION load path, decoder.Load with Backend:"webgpu", on ANY
// checkpoint (GOINFER_PARITY_CKPT: a safetensors dir or a .gguf). With GOINFER_GPU_CAPTURE=1 it differences the two forwards
// PER SUBLAYER PER LAYER (attention context, attention contribution, everything after it, and the layer's output residual: the
// runner's capture buffers vs decoder.ForwardLayerCaptureLogitsForTest), so a divergence is localised in one run instead of
// argued from the final logits. On a Gemma 4 E-model the CPU side adds the PLE contribution and the layer scalar, so every
// column compares the same quantity on both sides.
// It prints and does not assert a bar (the only assertion is that the cosine is not NaN): what a cosine means for a given
// checkpoint is what TestCPUQuantSensitivity measures for it.
//
// GOINFER_PARITY_README=n makes the prompt README.md's first n tokens under the checkpoint's tokenizer, at a resident context
// of 512: TestPairwiseRoPERealWebGPU's prompt and context.
// GOINFER_PARITY_QUANT: int4 (default), int8, int8int8, or native (CPU f32 vs the resident W8A8-at-upload path, NOT a
// same-quant comparison). GOINFER_INT4_F16_SCALES=1 makes both sides carry the f16-rounded int4 group scales the GPU stores,
// the only single-variable form of the int4 comparison. Written against scripts/mk_phi3_synth.py's phi3-mini-shaped
// synthetic checkpoint; runs on a real one too.
func TestResidentCaptureParityWebGPU(t *testing.T) {
	dir := os.Getenv("GOINFER_PARITY_CKPT")
	if dir == "" {
		t.Skip("set GOINFER_PARITY_CKPT to a phi3-shaped safetensors dir")
	}
	if c, err := New(); err != nil {
		t.Skipf("no webgpu device: %v", err)
	} else {
		c.Close()
	}
	prompt := []int{1, 7, 42, 100, 5, 200, 13, 88}
	if n, _ := strconv.Atoi(os.Getenv("GOINFER_PARITY_README")); n > 0 {
		prompt = readmePrompt(t, dir, n)
	}
	quant := os.Getenv("GOINFER_PARITY_QUANT") // "" ⇒ int4 (the doc's case); "native" for f32-vs-W8A8
	if quant == "" {
		quant = "int4"
	}
	cpuQuant, gpuQuant := quant, quant
	if quant == "native" { // Options.Quant "" is the f32 load; the resident path then W8A8-quantizes at upload
		cpuQuant, gpuQuant = "", ""
	}

	gpuOpts := decoder.Options{Backend: "webgpu", Quant: gpuQuant}
	if os.Getenv("GOINFER_PARITY_README") != "" {
		gpuOpts.ResidentContext = 512 // TestPairwiseRoPERealWebGPU's, so a 7B fits an 8 GB card as it does there
	}
	mg, err := decoder.Load(dir, gpuOpts)
	if err != nil {
		t.Fatalf("load webgpu: %v", err)
	}
	defer mg.Close()
	rf := mg.ResidentForwardForTest()
	if rf == nil {
		t.Fatalf("did not go resident — decode path %q; decline: %s", mg.DecodePath(), mg.ResidentDecline())
	}
	t.Logf("resident decode path: %s", mg.DecodePath())

	mcpu, err := decoder.Load(dir, decoder.Options{Quant: cpuQuant})
	if err != nil {
		t.Fatalf("load cpu: %v", err)
	}
	defer mcpu.Close()
	t.Logf("cpu decode path: %s", mcpu.DecodePath())
	for i := range prompt { // a tiny checkpoint's vocabulary can be smaller than the prompt's ids
		prompt[i] %= mcpu.Shape().Vocab
	}

	cache := mcpu.NewCache(len(prompt) + 16)
	minCos := 1.0
	var lastCPU, lastGPU []float32
	// GOINFER_GPU_CAPTURE=1: per-layer, per-sublayer differencing (CLAUDE.md: "prefer
	// differencing per layer over reasoning from final logits") — the resident runner's capture
	// buffers vs decoder.ForwardLayerCaptureLogitsForTest, from the SAME token on each side.
	capture := os.Getenv("GOINFER_GPU_CAPTURE") != ""
	runner := rf.(*residentDecoder).runner
	for i, tok := range prompt {
		var cpuL []float32
		var cAttn, cMLP, cCtx, cPLE, cOut [][]float32
		var cScalar []float32
		if capture {
			var lc decoder.LayerCaptureForTest
			cpuL, lc, err = mcpu.ForwardLayerCaptureLogitsForTest(tok, cache)
			cAttn, cMLP, cCtx, cPLE, cOut, cScalar = lc.Attn, lc.MLP, lc.Ctx, lc.PLE, lc.Out, lc.Scalar
		} else {
			cpuL, err = mcpu.ForwardForTest(tok, cache)
		}
		if err != nil {
			t.Fatalf("cpu pos %d: %v", i, err)
		}
		x0 := mg.EmbedResidentForTest(tok)
		gpuL, err := rf.Forward(x0, i)
		if err != nil {
			t.Fatalf("resident forward pos %d: %v", i, err)
		}
		cos, maxAbs := cosine(cpuL, gpuL)
		if cos < minCos {
			minCos = cos
		}
		t.Logf("  prompt pos %2d tok %4d: cosine %.6f maxAbs %.3e argmax cpu=%d gpu=%d", i, tok, cos, maxAbs, argmaxF(cpuL), argmaxF(gpuL))
		if capture {
			gCtx, gAttn, gMLP, err := runner.ReadCapture()
			if err != nil {
				t.Fatalf("ReadCapture pos %d: %v", i, err)
			}
			// Rebuild the CPU layer from the GPU's own previous output and the CPU's contributions: h_in(0) = the embedding
			// (identical on both sides), h_attn(l) = h_in(l) + attn(l), h_out(l) = (h_attn(l) + mlp(l) + ple(l)) × scalar(l)
			// (PLE and the scalar on a Gemma 4 E-model only), with h_in(l+1) = the GPU's h_out(l). The GPU captures the
			// residual itself; its contributions are the differences, so "post-attn" on both sides is everything after the
			// attention add. The contribution and "out" columns show layer l's own error; "chain" (the CPU's own layer output
			// against the GPU's) shows the error inherited up to and through layer l.
			hIn := slices.Clone(x0[:len(gAttn[0])]) // the residual; an E-model row carries its PLE inputs past it
			sub := func(a, b []float32) []float32 {
				d := make([]float32, len(a))
				for j := range d {
					d[j] = a[j] - b[j]
				}
				return d
			}
			add := func(a, b []float32) []float32 {
				d := make([]float32, len(a))
				for j := range d {
					d[j] = a[j] + b[j]
				}
				return d
			}
			for l := range gMLP {
				hAttnCPU := add(hIn, cAttn[l])
				hOutCPU := add(hAttnCPU, cMLP[l])
				if cPLE != nil {
					hOutCPU = add(hOutCPU, cPLE[l])
				}
				if sc := cScalar[l]; sc != 0 {
					for j := range hOutCPU {
						hOutCPU[j] *= sc
					}
				}
				postCPU, postGPU := sub(hOutCPU, hAttnCPU), sub(gMLP[l], gAttn[l])
				rel := func(ref, got []float32) float64 { // ||got-ref|| / ||ref||
					var d, n float64
					for j := range ref {
						e := float64(got[j]) - float64(ref[j])
						d += e * e
						n += float64(ref[j]) * float64(ref[j])
					}
					return math.Sqrt(d) / (math.Sqrt(n) + 1e-30)
				}
				cCos, _ := cosine(cCtx[l], gCtx[l])
				aCos, _ := cosine(cAttn[l], sub(gAttn[l], hIn))
				hACos, _ := cosine(hAttnCPU, gAttn[l])
				mCos, _ := cosine(postCPU, postGPU)
				hOCos, _ := cosine(hOutCPU, gMLP[l])
				chCos, _ := cosine(cOut[l], gMLP[l])
				t.Logf("      L%02d ctx cos %.8f rel %.2e | attn-contrib cos %.8f rel %.2e (resid %.8f) | post-attn cos %.8f rel %.2e | out cos %.8f rel %.2e | chain cos %.8f rel %.2e",
					l, cCos, rel(cCtx[l], gCtx[l]), aCos, rel(cAttn[l], sub(gAttn[l], hIn)), hACos, mCos, rel(postCPU, postGPU), hOCos, rel(hOutCPU, gMLP[l]), chCos, rel(cOut[l], gMLP[l]))
				hIn = gMLP[l] // the next layer starts from the GPU's own output, so its columns isolate its own error
			}
		}
		lastCPU = append(lastCPU[:0], cpuL...)
		lastGPU = append(lastGPU[:0], gpuL...)
	}
	// Greedy continuation, each side fed its OWN argmax (the doc's protocol), plus the
	// teacher-forced cosine (both fed the CPU's token) so a first flip does not hide what follows.
	cpuTok, gpuTok := argmaxF(lastCPU), argmaxF(lastGPU)
	pos := len(prompt)
	for step := range 6 {
		cpuL, err := mcpu.ForwardForTest(cpuTok, cache)
		if err != nil {
			t.Fatalf("cpu gen %d: %v", step, err)
		}
		gpuL, err := rf.Forward(mg.EmbedResidentForTest(gpuTok), pos)
		if err != nil {
			t.Fatalf("gpu gen %d: %v", step, err)
		}
		cos, _ := cosine(cpuL, gpuL)
		t.Logf("  greedy step %d: fed cpu=%d gpu=%d → cosine %.6f next cpu=%d gpu=%d", step, cpuTok, gpuTok, cos, argmaxF(cpuL), argmaxF(gpuL))
		cpuTok, gpuTok = argmaxF(cpuL), argmaxF(gpuL)
		pos++
	}
	t.Logf("phi3-synth (%s): min prompt cosine %.6f", quant, minCos)
	if math.IsNaN(minCos) {
		t.Errorf("NaN cosine")
	}
}

// readmePrompt is README.md's first n tokens under dir's tokenizer: TestPairwiseRoPERealWebGPU's prompt.
func readmePrompt(t *testing.T, dir string, n int) []int {
	t.Helper()
	readme, err := os.ReadFile("../../README.md")
	if err != nil {
		t.Fatal(err)
	}
	tk, err := tokenizer.Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	ids, err := tk.Encode(string(readme), false)
	if err != nil || len(ids) < n {
		t.Fatalf("encode README: %d ids, %v", len(ids), err)
	}
	return ids[:n]
}

// TestResidentLayerIsolatedWebGPU differences each layer ALONE, resident against CPU: at every position the resident runs
// first, and the CPU then runs each layer l on the resident's own input to it (KVCache.SetLayerInputsForTest: the embedding
// for layer 0, the resident's layer l-1 output after), so its K/V rows are computed from the same inputs too. A layer's
// columns are then its own rounding: attention context, attention contribution, the rest of the layer, and its output. The
// logits row is the CPU head on the resident's last-layer input. It prints and asserts only that no reading is NaN.
//
//	GOINFER_PARITY_CKPT=<dir> GOINFER_PARITY_README=20 [GOINFER_PARITY_QUANT=int4] go test -tags 'gpu goinfer_testhooks' \
//	  ./backend/webgpu -run TestResidentLayerIsolatedWebGPU -v
func TestResidentLayerIsolatedWebGPU(t *testing.T) {
	dir := os.Getenv("GOINFER_PARITY_CKPT")
	if dir == "" {
		t.Skip("set GOINFER_PARITY_CKPT to a checkpoint directory")
	}
	if c, err := New(); err != nil {
		t.Skipf("no webgpu device: %v", err)
	} else {
		c.Close()
	}
	quant := cmp.Or(os.Getenv("GOINFER_PARITY_QUANT"), "int4")
	prompt := []int{1, 7, 42, 100, 5, 200, 13, 88}
	if n, _ := strconv.Atoi(os.Getenv("GOINFER_PARITY_README")); n > 0 {
		prompt = readmePrompt(t, dir, n)
	}
	t.Setenv("GOINFER_GPU_CAPTURE", "1")
	mg, err := decoder.Load(dir, decoder.Options{Backend: "webgpu", Quant: quant, ResidentContext: 512})
	if err != nil {
		t.Fatalf("load webgpu: %v", err)
	}
	defer mg.Close()
	rd, ok := mg.ResidentForwardForTest().(*residentDecoder)
	if !ok {
		t.Fatalf("did not go resident: %s", mg.ResidentDecline())
	}
	mc, err := decoder.Load(dir, decoder.Options{Quant: quant})
	if err != nil {
		t.Fatalf("load cpu: %v", err)
	}
	defer mc.Close()
	for i := range prompt {
		prompt[i] %= mc.Shape().Vocab
	}
	cache := mc.NewCache(len(prompt) + 1)
	rel := func(ref, got []float32) float64 { // ||got-ref|| / ||ref||; a length mismatch reads as the worst value
		if len(ref) != len(got) {
			return math.Inf(1)
		}
		var d, n float64
		for j := range ref {
			e := float64(got[j]) - float64(ref[j])
			d, n = d+e*e, n+float64(ref[j])*float64(ref[j])
		}
		return math.Sqrt(d) / (math.Sqrt(n) + 1e-30)
	}
	diff := func(a, b []float32) []float32 {
		d := make([]float32, len(a))
		for j := range d {
			d[j] = a[j] - b[j]
		}
		return d
	}
	nan := false
	for i, tok := range prompt {
		x0 := mg.EmbedResidentForTest(tok)
		gl, err := rd.Forward(x0, i)
		if err != nil {
			t.Fatalf("resident pos %d: %v", i, err)
		}
		gCtx, gAttn, gOut, err := rd.runner.ReadCapture()
		if err != nil {
			t.Fatalf("ReadCapture pos %d: %v", i, err)
		}
		hidden := len(gAttn[0])
		in := make([][]float32, len(gOut))
		for l := 1; l < len(gOut); l++ {
			in[l] = gOut[l-1]
		}
		cache.SetLayerInputsForTest(in)
		cl, lc, err := mc.ForwardLayerCaptureLogitsForTest(tok, cache)
		cache.SetLayerInputsForTest(nil)
		if err != nil {
			t.Fatalf("cpu pos %d: %v", i, err)
		}
		cos, _ := cosine(cl, gl)
		nan = nan || math.IsNaN(cos)
		t.Logf("  pos %2d tok %6d: logits cosine %.6f (CPU head on the resident's last-layer input) argmax cpu=%d gpu=%d", i, tok, cos, argmaxF(cl), argmaxF(gl))
		for l := range gOut {
			hIn := x0[:hidden]
			if l > 0 {
				hIn = gOut[l-1]
			}
			gA, cRest, gRest := diff(gAttn[l], hIn), diff(diff(lc.Out[l], hIn), lc.Attn[l]), diff(gOut[l], gAttn[l])
			rc, ra, rr, ro := rel(lc.Ctx[l], gCtx[l]), rel(lc.Attn[l], gA), rel(cRest, gRest), rel(diff(lc.Out[l], hIn), diff(gOut[l], hIn))
			nan = nan || math.IsNaN(rc+ra+rr+ro)
			t.Logf("      L%02d ctx rel %.3e | attn rel %.3e | rest rel %.3e | layer rel %.3e", l, rc, ra, rr, ro)
		}
	}
	if nan {
		t.Error("a reading is NaN")
	}
}

func actG32RequireAdapter(t *testing.T) {
	t.Helper()
	c, err := New()
	if err != nil {
		t.Skipf("no WebGPU adapter: %v", err)
	}
	c.Close()
}

func actG32Cosine(a, b []float32) float64 {
	var ab, aa, bb float64
	for i := range a {
		ab += float64(a[i]) * float64(b[i])
		aa += float64(a[i]) * float64(a[i])
		bb += float64(b[i]) * float64(b[i])
	}
	return ab / math.Sqrt(aa*bb)
}

// actG32ResidentLogits loads path on webgpu at int8int8 with per-32 activations, checks the decode path is resident, and
// returns each position's logits from the resident forward.
func actG32ResidentLogits(t *testing.T, path string, ids []int, opts decoder.Options) [][]float32 {
	t.Helper()
	opts.Backend, opts.Quant, opts.ActQuantGroup = "webgpu", "int8int8", 32
	m, err := decoder.Load(path, opts)
	if err != nil {
		t.Fatalf("webgpu load: %v", err)
	}
	defer m.Close()
	rf := m.ResidentForwardForTest()
	if dp := m.DecodePath(); rf == nil || !strings.HasPrefix(dp, "webgpu") || !strings.Contains(dp, "-resident") {
		t.Fatalf("decode path %q, want a webgpu resident; decline: %s", m.DecodePath(), m.ResidentDecline())
	}
	if rd, ok := rf.(*residentDecoder); !ok || !rd.rm.ActG32 {
		t.Fatalf("the resident (%T) was not built with the per-32 twins", rf)
	}
	rf.Reset()
	out := make([][]float32, len(ids))
	for p, id := range ids {
		lg, err := rf.Forward(m.EmbedResidentForTest(id), p)
		if err != nil {
			t.Fatalf("resident forward pos %d: %v", p, err)
		}
		out[p] = slices.Clone(lg)
	}
	return out
}

// actG32CPULogits is the CPU forward's logits at each position under opts.
func actG32CPULogits(t *testing.T, path string, ids []int, opts decoder.Options) [][]float32 {
	t.Helper()
	m, err := decoder.Load(path, opts)
	if err != nil {
		t.Fatalf("cpu load %+v: %v", opts, err)
	}
	defer m.Close()
	c := m.NewCache(len(ids) + 1)
	out := make([][]float32, len(ids))
	for p, id := range ids {
		lg, err := m.ForwardForTest(id, c)
		if err != nil {
			t.Fatal(err)
		}
		out[p] = slices.Clone(lg)
	}
	return out
}

// actG32Agreement is the sorted per-position cosines of a against b.
func actG32Agreement(a, b [][]float32) []float64 {
	cs := make([]float64, len(a))
	for p := range a {
		cs[p] = actG32Cosine(a[p], b[p])
	}
	slices.Sort(cs)
	return cs
}

// TestActG32_phi3TinyResidentMatchesCPU: the synthetic two-layer Phi-3 at int8int8 with per-32 activations runs resident on
// WebGPU, past the activation-hazard gate, and its logits at every position match the CPU per-32 path (the same algorithm,
// differing by kernel accumulation order) at median cosine >= 0.9999 and minimum >= 0.999. The planted per-vector scale
// index (kernels.ActG32PerVectorScaleForTest) misses the median bar, and an int4 load with per-32 activations declines naming
// the option (on the synthetic llama, which has no activation hazard to refuse it first).
func TestActG32_phi3TinyResidentMatchesCPU(t *testing.T) {
	actG32RequireAdapter(t)
	path := filepath.Join(t.TempDir(), "phi3-tiny.gguf")
	if err := decoder.WritePhi3Q4KTinyGGUFForTest(path); err != nil {
		t.Fatal(err)
	}
	ids := make([]int, 24)
	for p := range ids {
		ids[p] = (7 + 13*p) % 64
	}
	want := actG32CPULogits(t, path, ids, decoder.Options{Quant: "int8int8", ActQuantGroup: 32})
	cs := actG32Agreement(actG32ResidentLogits(t, path, ids, decoder.Options{}), want)
	t.Logf("webgpu per-32 vs cpu per-32: %d positions, cosine min %.7f median %.7f", len(cs), cs[0], cs[len(cs)/2])
	if cs[len(cs)/2] < 0.9999 || cs[0] < 0.999 {
		t.Errorf("median cosine %.7f (want >= 0.9999), min %.7f (want >= 0.999)", cs[len(cs)/2], cs[0])
	}

	testgate.SetForTest(t, &kernels.ActG32PerVectorScaleForTest, true)
	planted := actG32Agreement(actG32ResidentLogits(t, path, ids, decoder.Options{}), want)
	testgate.SetForTest(t, &kernels.ActG32PerVectorScaleForTest, false)
	t.Logf("plant (group 0's scale for every group): cosine min %.7f median %.7f", planted[0], planted[len(planted)/2])
	if planted[len(planted)/2] >= 0.9999 {
		t.Errorf("plant: median cosine %.7f still passes the 0.9999 bar", planted[len(planted)/2])
	}

	// int4 with per-32 activations, on a family the activation-hazard gate does not refuse first.
	llama := filepath.Join(t.TempDir(), "llama-tiny.gguf")
	if err := decoder.WriteLlamaQ4KTinyGGUFForTest(llama); err != nil {
		t.Fatal(err)
	}
	m, err := decoder.Load(llama, decoder.Options{Backend: "webgpu", Quant: "int4", ActQuantGroup: 32})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if m.ResidentActive() || !strings.Contains(m.ResidentDecline(), "ActQuantGroup 32") {
		t.Errorf("int4 with per-32 activations: resident %v, decline %q; want a decline naming ActQuantGroup 32", m.ResidentActive(), m.ResidentDecline())
	}
}

// TestActG32_phi3ResidentMatchesCPU is CUDA's TestActGroup_phi3ResidentMatchesCPU on WebGPU: the real Phi-3-mini at
// int8int8 with per-32 activations, resident, over a ~130-token chat prompt of filler (the regime that broke per-vector
// scales). The bars are CUDA's: median cosine against the CPU per-32 path >= 0.9995, and p10 quality no worse than the CPU
// per-32 path's by more than 0.01. Quality is read against goinfer's weight-only int8 (f32 activations, which hold on
// Phi-3; decoder.ActivationQuantHazard) rather than CUDA's f32 reference, so the three loads fit a 16 GB host in turn.
//
//	GOINFER_PHI3_MINI_GGUF=~/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf \
//	  go test -tags 'gpu goinfer_testhooks' ./backend/webgpu -run TestActG32_phi3ResidentMatchesCPU -v
func TestActG32_phi3ResidentMatchesCPU(t *testing.T) {
	path := os.Getenv("GOINFER_PHI3_MINI_GGUF")
	if path == "" {
		t.Skip("set GOINFER_PHI3_MINI_GGUF")
	}
	actG32RequireAdapter(t)
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatal(err)
	}
	tmpl, err := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has})
	if err != nil {
		t.Fatal(err)
	}
	text := strings.Repeat("the ", 110) + "\n\nThe text above is only filler. Now write a long, detailed essay about the history of the printing press."
	ids, err := tk.EncodeSegments(tmpl.RenderSegments("", []chat.Turn{{Role: "user", Content: text}}), false)
	if err != nil {
		t.Fatal(err)
	}
	gl := actG32ResidentLogits(t, path, ids, decoder.Options{ResidentContext: 512})
	cpuG := actG32CPULogits(t, path, ids, decoder.Options{Quant: "int8int8", ActQuantGroup: 32})
	ref := actG32CPULogits(t, path, ids, decoder.Options{Quant: "int8"})
	report := func(name string, cs []float64) {
		t.Logf("%s: %d positions, cosine min %.6f p10 %.6f median %.6f", name, len(cs), cs[0], cs[len(cs)/10], cs[len(cs)/2])
	}
	agree, gpuQ, cpuQ := actG32Agreement(gl, cpuG), actG32Agreement(gl, ref), actG32Agreement(cpuG, ref)
	report("webgpu per-32 vs cpu per-32", agree)
	report("webgpu per-32 vs cpu weight-only int8", gpuQ)
	report("cpu per-32 vs cpu weight-only int8 (same prompt, the baseline)", cpuQ)
	if agree[len(agree)/2] < 0.9995 {
		t.Errorf("WebGPU per-32 vs CPU per-32: median cosine %.6f, want >= 0.9995", agree[len(agree)/2])
	}
	if gpuQ[len(gpuQ)/10] < cpuQ[len(cpuQ)/10]-0.01 {
		t.Errorf("WebGPU per-32 quality p10 %.4f is worse than the CPU per-32 path's %.4f on the same prompt", gpuQ[len(gpuQ)/10], cpuQ[len(cpuQ)/10])
	}
}
