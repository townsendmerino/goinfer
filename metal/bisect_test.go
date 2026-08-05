//go:build darwin

package metal

import (
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestGemmaBisect_PerLayer walks the residual stream layer by layer, Metal vs CPU, to locate
// WHERE Metal's Gemma-only parity residual enters.
//
// What is already known, so this test does not re-litigate it:
//   - int4 costs gemma3 0.99→0.92 of logit cosine and the control ~nothing (quantbar_test.go).
//     That part is the quantization class at gemma's shape and is not a bug.
//   - Metal still adds a further -0.104 on gemma and NOTHING on the control. That part is a bug.
//   - It is not the weights: the double quantization measured free (decoder/requant_test.go).
//   - It is not multi-key attention at hd=256: the shipped kernel is exact there, including
//     sharp softmax, outlier V, sink, and window engaged (attn_shape_test.go).
//
// So a Gemma-specific op in the compute path is wrong, and the residual's SHAPE says which kind:
// 9 gaps >3% with a worst near-tie of 40.8% is not a per-layer precision delta compounding over
// 34 layers (that would be a smooth droop). It is an op failing on particular positions.
//
// Three deliberate choices, each one a lesson already paid for:
//
//  1. Reference is CPU-INT4, not CPU-int8. Metal's weights are now measured equivalent to the
//     decoder's int4, so int4 is the like-for-like reference; int8 would fold the (large, real,
//     not-a-bug) quantization cost into every number and hide the -0.104 underneath it.
//  2. NORM is reported beside cosine at every layer. The sink cost a week because a cosine was
//     read off a near-zero vector. A collapsing norm and a collapsing cosine are different bugs.
//  3. The probe is an ORDINARY token at pos>0. At pos 0 attention output is v0 exactly and RoPE
//     is the identity — the two ops most under suspicion do not even run, which is precisely why
//     the earlier pos-0 analysis saw nothing.
//
// Each layer is tagged local/global (gemma's 5:1). If the jump tracks the global layers, the
// carrier is attention/window/per-layer-RoPE; if it is uniform, it is a per-layer op.
func TestGemmaBisect_PerLayer(t *testing.T) {
	requireHeavyModel(t)
	if testing.Short() {
		t.Skip("loads real models")
	}
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	for _, tc := range []struct {
		what string
		path string
	}{
		{"gemma3-4b", os.ExpandEnv("$HOME/models/gemma-3-4b-it-Q4_K_M.gguf")},
		{"control qwen2.5-1.5b", os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")},
	} {
		t.Run(tc.what, func(t *testing.T) {
			if _, err := os.Stat(tc.path); err != nil {
				t.Skipf("no checkpoint at %s", tc.path)
			}
			bisectModel(t, tc.path)
		})
	}
}

func bisectModel(t *testing.T, path string) {
	t.Helper()
	seed := seedPrompt(t, path, probeText)

	// Metal consumes an int8 load (BuildResident requires it) and re-quantizes internally;
	// the CPU reference is int4, the measured-equivalent twin of what Metal ends up holding.
	m8, err := decoder.Load(path, decoder.Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load int8: %v", err)
	}
	r, err := buildResident(m8)
	if err != nil {
		t.Fatalf("BuildResident: %v", err)
	}
	defer r.Close()
	m4, err := decoder.Load(path, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load int4: %v", err)
	}
	_, nL, _, nKV, hd, _, _ := m4.Dims()
	cache := decoder.NewKVCache(nL, nKV, hd, 0, 1024)

	// Walk both sides over the seed so the KV caches hold real history, then probe the LAST
	// seed token — an ordinary word at pos>0, where multi-key attention and RoPE actually run.
	layers := make([]int, nL)
	for i := range layers {
		layers[i] = i
	}
	var hidden [][]float32
	for i := 0; i < len(seed); i++ {
		var cerr error
		_, hidden, cerr = m4.ForwardCapture(seed[i], cache, layers)
		if cerr != nil {
			t.Skipf("ForwardCapture: %v", cerr)
		}
		r.forwardTrunkForTest(m8.EmbedResidentForTest(seed[i]), i, nL) // keep Metal's KV in step
	}
	pos := len(seed) - 1
	t.Logf("probe: pos=%d (last seed token), %d layers", pos, nL)

	// Fork 1 setup: the target sign-flipped channels (bisect prompt). Track, per layer depth,
	// Metal's signed value vs CPU-int4's on each — the first layer where a target flips sign names
	// where the bug ENTERS, and that layer is where the attention-vs-MLP split then runs.
	targets := []int{1698, 1723, 227}
	flippedAt := map[int]int{}
	// Re-run the probe position at each truncation depth. Metal's KV for pos already holds this
	// token's K/V from the walk above, so re-encoding it is idempotent.
	for l := 1; l <= nL; l++ {
		got := r.forwardTrunkForTest(m8.EmbedResidentForTest(seed[pos]), pos, l)
		ref := hidden[l-1]
		cos, gn, rn := cosNorm(got, ref)
		tag := "global"
		if m4.LayerIsLocalResident(l - 1) {
			tag = "local "
		}
		flips := ""
		for _, ch := range targets {
			g, c := got[ch], ref[ch]
			flip := (g < 0) != (c < 0) && absf(g) > 1 && absf(c) > 1
			mark := " "
			if flip {
				mark = "✗"
				if _, seen := flippedAt[ch]; !seen {
					flippedAt[ch] = l - 1
				}
			}
			flips += fmt.Sprintf("  d%d[m=%+8.1f c=%+8.1f]%s", ch, g, c, mark)
		}
		t.Logf("layer %2d [%s]: cosine=%.6f |metal|=%9.3f |cpu-int4|=%9.3f%s",
			l-1, tag, cos, gn, rn, flips)
	}
	for _, ch := range targets {
		if l, ok := flippedAt[ch]; ok {
			t.Logf("channel %d FIRST sign-flips at layer %d (Fork 1 splits attention vs MLP there)", ch, l)
		} else {
			t.Logf("channel %d never sign-flips across the trunk — the flip is in final-norm/head only", ch)
		}
	}

	// Cross-backend reconciliation with the CUDA box. Three references at each target channel +
	// neighbors, at the pre-final-norm tap (pos 5):
	//   metal      — Metal resident (int4 weight × int8 activation, W4A8)
	//   cpu-int4   — decoder Quant:int4 == MatmulBTW4A8, ALSO int8 activation → shares the crush
	//   cpu-int8w  — decoder Quant:int8 (weight-only) == MatmulBTQ8, int8 weight × f32 ACTIVATION
	//                → NO activation crush, so its SIGN is the ground truth on crushed channels.
	// The crux the CUDA box surfaced: our two "int4" references disagreed because BOTH quantize
	// activations to int8 and round the near-zero crushed channels differently. cpu-int8w removes
	// the activation quant, so whichever of metal/cpu-int4 disagrees with cpu-int8w's SIGN is the
	// one that flipped. (Off-by-one is ruled out: the 443 spike sits at index 443 on both boxes.)
	m8w, e8w := decoder.Load(path, decoder.Options{Quant: "int8"})
	if e8w != nil {
		t.Fatalf("load int8-weight-only reference: %v", e8w)
	}
	c8w := decoder.NewKVCache(nL, nKV, hd, 0, 1024)
	var trueHidden []float32
	for i := 0; i < len(seed); i++ {
		_, h, cerr := m8w.ForwardCapture(seed[i], c8w, []int{nL - 1})
		if cerr != nil {
			t.Skipf("ForwardCapture (int8w): %v", cerr)
		}
		trueHidden = h[0]
	}
	full := r.forwardTrunkForTest(m8.EmbedResidentForTest(seed[pos]), pos, nL)
	cpuFull := hidden[nL-1]
	t.Logf("NEIGHBOR DUMP (pre-final-norm, pos %d): cpu-int8w = f32-activation ground truth", pos)
	for _, center := range []int{443, 1698, 1723, 227} {
		for ch := center - 2; ch <= center+2; ch++ {
			if ch >= 0 && ch < len(full) {
				flag := ""
				if (full[ch] < 0) != (trueHidden[ch] < 0) && absf(trueHidden[ch]) > 1 {
					flag = "  METAL flips vs truth"
				}
				t.Logf("  ch %4d: metal=%+10.2f  cpu-int4=%+10.2f  cpu-int8w(TRUTH)=%+10.2f%s",
					ch, full[ch], cpuFull[ch], trueHidden[ch], flag)
			}
		}
	}
}

func absf(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

// TestGemmaBisect_Head splits the ONE step the per-layer bisect leaves whole: final-norm → LM
// head. The per-layer walk showed gemma's trunk lands at cosine ~0.981 (control ~0.993) yet the
// logits collapse to 0.818 (control 0.990) — so almost the entire Gemma-only residual enters
// HERE, not in the 34 layers. This test says which of the two sub-steps:
//
//	act:    Metal's head-input activation (r.aq * r.aSc, the int8-quantized final-norm output)
//	        vs the CPU's f32 final-norm output — isolates the QUANTIZATION of the head input.
//	logits: Metal's logits vs CPU's — the same on top of the head MATMUL.
//
// If act collapses and logits track it, the final-norm output is being crushed by int8 absmax
// quantization (an outlier in the normed vector spending the whole -127..127 range). If act is
// clean but logits collapse, the head matmul itself is wrong. Reported with norms, per the sink
// lesson.
func TestGemmaBisect_Head(t *testing.T) {
	requireHeavyModel(t)
	if testing.Short() {
		t.Skip("loads real models")
	}
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	for _, tc := range []struct {
		what string
		path string
	}{
		{"gemma3-4b", os.ExpandEnv("$HOME/models/gemma-3-4b-it-Q4_K_M.gguf")},
		{"control qwen2.5-1.5b", os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")},
	} {
		t.Run(tc.what, func(t *testing.T) {
			if _, err := os.Stat(tc.path); err != nil {
				t.Skipf("no checkpoint at %s", tc.path)
			}
			seed := seedPrompt(t, tc.path, probeText)
			m8, err := decoder.Load(tc.path, decoder.Options{Quant: "int8int8"})
			if err != nil {
				t.Fatalf("load int8: %v", err)
			}
			r, err := buildResident(m8)
			if err != nil {
				t.Fatalf("BuildResident: %v", err)
			}
			defer r.Close()
			m4, err := decoder.Load(tc.path, decoder.Options{Quant: "int4"})
			if err != nil {
				t.Fatalf("load int4: %v", err)
			}
			_, nL, _, nKV, hd, _, _ := m4.Dims()
			cache := decoder.NewKVCache(nL, nKV, hd, 0, 1024)
			layers := []int{nL - 1}
			var cpuLogits []float32
			var lastHidden []float32
			for i := 0; i < len(seed); i++ {
				lg, hid, cerr := m4.ForwardCapture(seed[i], cache, layers)
				if cerr != nil {
					t.Skipf("ForwardCapture: %v", cerr)
				}
				cpuLogits, lastHidden = lg, hid[0]
				r.forwardTrunkForTest(m8.EmbedResidentForTest(seed[i]), i, nL)
			}
			pos := len(seed) - 1
			// Metal's OWN residual entering the final norm (pre-norm), and its post-norm output.
			metalPre := r.forwardTrunkForTest(m8.EmbedResidentForTest(seed[pos]), pos, nL)
			gpuAct, gpuLogits := r.forwardHeadForTest(m8.EmbedResidentForTest(seed[pos]), pos)
			// The decisive isolation: apply the CPU's final-norm to METAL's own pre-norm vector.
			// If the kernel is correct this equals gpuAct (up to int8); if it diverges, the final-
			// norm KERNEL is wrong, independent of any trunk drift.
			cpuNormOfMetal := m4.FinalNormForTest(metalPre)
			ck, _, _ := cosNorm(gpuAct, cpuNormOfMetal)
			t.Logf("%s KERNEL ISOLATION — Metal final-norm out vs CPU norm of Metal's OWN input: cosine=%.6f", tc.what, ck)

			// Where does metalPre diverge from CPU's hidden, and does it land on the dims the
			// final norm blows up? Report the few worst-|Δ| dims with their pre-norm values and
			// the final-norm weight there — the massive-activation signature.
			topDivergentDims(t, tc.what, metalPre, lastHidden, m4.FinalNormForTest)

			cpuAct := m4.FinalNormForTest(lastHidden)

			// Quantize the CPU's f32 final-norm output to int8 (absmax/127) ourselves. If Metal's
			// dequantized activation matches THIS, the norm math agrees and the whole loss is the
			// int8 crush of a high-dynamic-range vector — a precision choice, not a kernel bug. If
			// Metal disagrees even with the CPU's own int8 round, Metal's norm is computing
			// something different.
			cpuActQ := quantDequantI8(cpuAct)
			cq, _, _ := cosNorm(cpuActQ, cpuAct)
			cm, _, _ := cosNorm(gpuAct, cpuActQ)
			t.Logf("%s CPU int8-round of act vs CPU f32: cosine=%.6f | Metal act vs CPU-int8-round: cosine=%.6f",
				tc.what, cq, cm)

			ca, gaN, caN := cosNorm(gpuAct, cpuAct)
			cl, glN, clN := cosNorm(gpuLogits, cpuLogits)
			// Head-input dynamic range: what int8 absmax quantization has to cope with. A big
			// ratio of max|x| to mean|x| is exactly what crushes the rest of the vector.
			mx, mean := dynRange(cpuAct)
			t.Logf("%s head-input dynamic range: max/mean=%.1f (max=%.3f, mean=%.4f)", tc.what, mx/mean, mx, mean)
			t.Logf("%s FINAL-NORM act: cosine=%.6f |metal|=%.4f |cpu|=%.4f", tc.what, ca, gaN, caN)
			t.Logf("%s LOGITS:         cosine=%.6f |metal|=%.4f |cpu|=%.4f", tc.what, cl, glN, clN)
		})
	}
}

// topDivergentDims reports the dims where Metal's pre-final-norm residual diverges most from the
// CPU's, each with the effective final-norm amplification (1+w) recovered by norming a one-hot.
// The hypothesis under test: the worst dims are Gemma's massive-activation channels, where a
// small trunk error is multiplied by a large norm weight and a huge activation magnitude.
func topDivergentDims(t *testing.T, what string, metal, cpu []float32, norm func([]float32) []float32) {
	t.Helper()
	H := len(cpu)
	sqrtH := math.Sqrt(float64(H))
	type dim struct {
		i               int
		m, c, amp, cmag float64
	}
	dims := make([]dim, H)
	for i := 0; i < H; i++ {
		oneHot := make([]float32, H)
		oneHot[i] = 100
		amp := float64(norm(oneHot)[i]) / (100 / sqrtH * sqrtH) // = (1+w_i)
		dims[i] = dim{i, float64(metal[i]), float64(cpu[i]), amp, math.Abs(float64(cpu[i]))}
	}
	// Rank by the amplified error: |metal-cpu| * (1+w) — the term that actually reaches the head.
	ampErr := func(d dim) float64 { return math.Abs(d.m-d.c) * d.amp }
	sort.Slice(dims, func(a, b int) bool { return ampErr(dims[a]) > ampErr(dims[b]) })
	t.Logf("%s top divergent dims (by amplified error):", what)
	for k := 0; k < 6 && k < len(dims); k++ {
		d := dims[k]
		t.Logf("  dim %4d: metalPre=%10.2f cpuPre=%10.2f  Δ=%9.3f  (1+w)=%7.2f  ampErr=%9.2f",
			d.i, d.m, d.c, d.m-d.c, d.amp, ampErr(d))
	}
	// ALSO rank by raw MAGNITUDE of the pre-final-norm residual — a model property that must be
	// identical across backends at the SAME tap point. This is the cross-check the CUDA box asked
	// for: top-magnitude channels (and their scale) reveal whether two backends tap the same place.
	// (Distinct from the amplified-error ranking above, which is where Metal DIVERGES, not where
	// the activation is largest — the two lists legitimately differ.)
	byMag := append([]dim(nil), dims...)
	sort.Slice(byMag, func(a, b int) bool { return math.Abs(byMag[a].c) > math.Abs(byMag[b].c) })
	t.Logf("%s top-MAGNITUDE dims (|cpu-int4 pre-final-norm residual|, model property):", what)
	for k := 0; k < 6 && k < len(byMag); k++ {
		d := byMag[k]
		t.Logf("  dim %4d: |cpuPre|=%10.2f  cpuPre=%10.2f  metalPre=%10.2f", d.i, math.Abs(d.c), d.c, d.m)
	}
}

// quantDequantI8 round-trips x through symmetric per-vector int8 (scale=absmax/127), exactly as
// quant_vec/rmsnorm_quant do on the GPU — the CPU model of what the head input suffers.
func quantDequantI8(x []float32) []float32 {
	var mx float32
	for _, v := range x {
		if a := float32(math.Abs(float64(v))); a > mx {
			mx = a
		}
	}
	sc := mx / 127
	if sc == 0 {
		sc = 1
	}
	out := make([]float32, len(x))
	for i, v := range x {
		q := math.Round(float64(v / sc))
		if q > 127 {
			q = 127
		} else if q < -127 {
			q = -127
		}
		out[i] = float32(q) * sc
	}
	return out
}

func dynRange(x []float32) (mx, mean float64) {
	var sum float64
	for _, v := range x {
		a := math.Abs(float64(v))
		if a > mx {
			mx = a
		}
		sum += a
	}
	return mx, sum/float64(len(x)) + 1e-30
}

// TestGemmaTraceDims follows Gemma's massive-activation dims down the layer stack to find WHERE
// Metal clobbers them. The head bisect showed the final-norm amplifies a handful of outlier dims
// (1698/1730/2482/1723/227) that Metal has zeroed or sign-flipped; an all-dims cosine can't see 6
// bad channels in 2560, so this prints those channels explicitly at every layer, Metal vs CPU.
func TestGemmaTraceDims(t *testing.T) {
	requireHeavyModel(t)
	if testing.Short() {
		t.Skip("loads real models")
	}
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	path := os.ExpandEnv("$HOME/models/gemma-3-4b-it-Q4_K_M.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s", path)
	}
	seed := seedPrompt(t, path, probeText)
	m8, err := decoder.Load(path, decoder.Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load int8: %v", err)
	}
	r, err := buildResident(m8)
	if err != nil {
		t.Fatalf("BuildResident: %v", err)
	}
	defer r.Close()
	m4, err := decoder.Load(path, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load int4: %v", err)
	}
	_, nL, _, nKV, hd, _, _ := m4.Dims()
	cache := decoder.NewKVCache(nL, nKV, hd, 0, 1024)
	layers := make([]int, nL)
	for i := range layers {
		layers[i] = i
	}
	// CPU-int8 as a THIRD reference. Metal's weights are int8 re-quantized to int4; CPU-int4 is
	// f32→int4. If the outlier dims diverge because of int8 ACTIVATION quantization (which both
	// GPU and CPU-int4 do), then CPU-int4 and CPU-int8 diverge too and it is a quantization
	// property, not a Metal bug. If CPU-int4 tracks CPU-int8 on these dims while Metal alone
	// diverges, the fault is in Metal's compute.
	cache8 := decoder.NewKVCache(nL, nKV, hd, 0, 1024)
	var hidden, hidden8 [][]float32
	for i := 0; i < len(seed); i++ {
		_, hidden, _ = m4.ForwardCapture(seed[i], cache, layers)
		_, hidden8, _ = m8.ForwardCapture(seed[i], cache8, layers)
		r.forwardTrunkForTest(m8.EmbedResidentForTest(seed[i]), i, nL)
	}
	pos := len(seed) - 1
	watch := []int{1698, 2482, 1723, 227}
	for l := 1; l <= nL; l++ {
		got := r.forwardTrunkForTest(m8.EmbedResidentForTest(seed[pos]), pos, l)
		ref, ref8 := hidden[l-1], hidden8[l-1]
		line := ""
		for _, w := range watch {
			line += "  d" + itoa(w) + ":M=" + f1(got[w]) + "/i4=" + f1(ref[w]) + "/i8=" + f1(ref8[w])
		}
		t.Logf("layer %2d:%s", l-1, line)
	}
}

func itoa(i int) string { return strconv.Itoa(i) }
func f1(f float32) string {
	return strconv.FormatFloat(float64(f), 'f', 1, 64)
}

func cosNorm(a, b []float32) (cos, na, nb float64) {
	var dot float64
	for i := range b {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-30), math.Sqrt(na), math.Sqrt(nb)
}
