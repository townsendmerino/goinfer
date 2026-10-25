//go:build cuda && goinfer_testhooks

package cuda

import (
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// kso1cKL is KL(softmax(p) ‖ softmax(q)) in nats, in float64.
func kso1cKL(p, q []float32) float64 {
	lse := func(x []float32) float64 {
		m := math.Inf(-1)
		for _, v := range x {
			m = math.Max(m, float64(v))
		}
		s := 0.0
		for _, v := range x {
			s += math.Exp(float64(v) - m)
		}
		return m + math.Log(s)
	}
	lp, lq := lse(p), lse(q)
	kl := 0.0
	for i := range p {
		a := float64(p[i]) - lp
		kl += math.Exp(a) * (a - (float64(q[i]) - lq))
	}
	return kl
}

// TestKSO1C_towardTheFile is O1C-G6 (docs/tasks/task-kernel-survey-2026-10.md, "KS-O1 on CUDA: gates"): on the 0.5B q4_k_m and Phi-3
// mini q4, one CUDA q4k resident holding both arms (kquantKeepInt8ForTest) and the CPU at f32 on the same GGUF (the file exactly).
// Over G3's 8 prompts, the CPU's own greedy continuation of 32 tokens is teacher-forced through each arm, and every position's
// KL(CPU f32 ‖ arm) is taken. PASS per model iff the native arm's mean KL is at most the int8 arm's; up to 1.10× is ambiguous.
//
//	GOINFER_HEAVY_TESTS=1 go test -count=1 -timeout 40m -tags 'cuda goinfer_testhooks' -run '^TestKSO1C_towardTheFile$' -v ./cuda/
func TestKSO1C_towardTheFile(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1")
	}
	for _, name := range kso1cModels {
		path := modelPath(name)
		tk, err := tokenizer.LoadGGUF(path)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		tmpl, err := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has})
		if err != nil {
			t.Fatal(err)
		}
		mg, err := kso1cLoadBothArms(path, decoder.Options{Backend: "cuda", Quant: "q4k", ResidentContext: 512})
		if err != nil {
			t.Fatalf("%s: cuda load: %v", name, err)
		}
		r, ok := mg.ResidentForwardForTest().(*cudaResident)
		if !ok {
			mg.Close()
			t.Fatalf("%s: no CUDA resident: %s", name, mg.ResidentDecline())
		}
		use, nNative := kso1cArms(t, r, name)
		mf, err := decoder.Load(path, decoder.Options{Backend: "cpu", Quant: "f32", ResidentContext: 512})
		if err != nil {
			mg.Close()
			t.Fatalf("%s: cpu f32 load: %v", name, err)
		}
		var sum [2]float64
		lower, positions := 0, 0
		for pi, text := range g3Prompts {
			ids, err := tk.Encode(tmpl.Render("", []chat.Turn{{Role: "user", Content: text}}), false)
			if err != nil {
				t.Fatal(err)
			}
			n := len(ids) + g3NewTokens
			cache := mf.NewCache(n)
			seq := append([]int(nil), ids...)
			var ref [][]float32
			for i := 0; i < n-1; i++ {
				l, err := mf.ForwardForTest(seq[i], cache)
				if err != nil {
					t.Fatal(err)
				}
				ref = append(ref, append([]float32(nil), l...))
				if i >= len(ids)-1 {
					seq = append(seq, argmaxF(l))
				}
			}
			var arm [2][]float64 // 0 int8, 1 native
			for a, native := range []bool{false, true} {
				use(native)
				r.Reset()
				for i := 0; i < n-1; i++ {
					l, err := r.Forward(mg.EmbedResidentForTest(seq[i]), i)
					if err != nil {
						t.Fatal(err)
					}
					arm[a] = append(arm[a], kso1cKL(ref[i], l))
				}
			}
			var ps [2]float64
			for i := range arm[0] {
				ps[0] += arm[0][i]
				ps[1] += arm[1][i]
				if arm[1][i] < arm[0][i] {
					lower++
				}
			}
			sum[0] += ps[0]
			sum[1] += ps[1]
			positions += n - 1
			fmt.Fprintf(os.Stderr, "[O1C-G6] %s prompt %d: %d positions, mean KL int8 %.5f, native %.5f\n", name, pi+1, n-1, ps[0]/float64(n-1), ps[1]/float64(n-1))
		}
		use(true)
		mf.Close()
		mg.Close()
		mi, mn := sum[0]/float64(positions), sum[1]/float64(positions)
		ratio := mn / mi
		verdict := "FAIL"
		switch {
		case ratio <= 1:
			verdict = "PASS"
		case ratio <= 1.10:
			verdict = "AMBIGUOUS (parked)"
		}
		fmt.Fprintf(os.Stderr, "[O1C-G6] %s: %d weights native; over %d positions mean KL to the CPU f32 int8 %.5f, native %.5f, ratio %.4f; native lower at %d positions: %s\n",
			name, nNative, positions, mi, mn, ratio, lower, verdict)
		if verdict != "PASS" {
			t.Errorf("O1C-G6 %s: %s (ratio %.4f)", name, verdict, ratio)
		}
	}
}

// TestKSO1C_realPhi3NonInferiority is O1C-G5, the CUDA G-Q2 (docs/tasks/task-kernel-survey-2026-10.md, "KS-O1 on CUDA: gates"): in one
// process, g3Model on Qwen2.5-Coder-1.5B (the validated reference, as G3c runs it), then g3Run on Phi-3 mini q4 with the CUDA resident
// on the native lane at --quant q4k against the CPU at q4k on the same file. PASS iff Phi-3's teacher-forced agreement is at least the
// reference's - 2.0 points and its free-run passes at least the reference's - 1; 2.0-4.0 points below is ambiguous (parked).
//
//	GOINFER_HEAVY_TESTS=1 go test -count=1 -timeout 40m -tags 'cuda goinfer_testhooks' -run '^TestKSO1C_realPhi3NonInferiority$' -v ./cuda/
func TestKSO1C_realPhi3NonInferiority(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1")
	}
	logf := func(f string, a ...any) { fmt.Fprintf(os.Stderr, "[O1C-G5] "+f+"\n", a...) }
	rp, ra, rn := g3Model(t, modelPath("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"), "reference Qwen2.5-Coder-1.5B", logf)
	path := modelPath("phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf")
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatal(err)
	}
	tmpl, err := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has})
	if err != nil {
		t.Fatal(err)
	}
	opts := decoder.Options{Quant: "q4k", ResidentContext: 512}
	gopts := opts
	gopts.Backend = "cuda"
	mg, err := decoder.Load(path, gopts)
	if err != nil {
		t.Fatal(err)
	}
	defer mg.Close()
	r, ok := mg.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatalf("Phi-3: no CUDA resident: %s", mg.ResidentDecline())
	}
	nNative := 0
	for _, w := range append([]cudaWQ{r.lmW}, func() (ws []cudaWQ) {
		for _, Ly := range r.layers {
			ws = append(ws, Ly.q, Ly.k, Ly.v, Ly.o, Ly.g, Ly.u, Ly.d)
		}
		return
	}()...) {
		if w.kind == "q5k" || w.kind == "q6k" {
			nNative++
		}
	}
	if nNative == 0 {
		t.Fatal("Phi-3: the resident uploaded no weight natively")
	}
	mc, err := decoder.Load(path, opts)
	if err != nil {
		t.Fatal(err)
	}
	defer mc.Close()
	logf("Phi-3 mini: CUDA q4k resident with %d native weights; the CPU at q4k on the same file", nNative)
	pp, pa, pn := g3Run(t, tk, tmpl, r, mg, mc, logf)
	ref, pct := 100*float64(ra)/float64(rn), 100*float64(pa)/float64(pn)
	delta := pct - ref
	logf("O1C-G5: Phi-3 %.2f%% (%d/%d), %d/%d prompts; reference %.2f%% (%d/%d), %d/%d prompts; delta %+.2f points", pct, pa, pn, pp, len(g3Prompts), ref, ra, rn, rp, len(g3Prompts), delta)
	switch {
	case delta >= -2.0 && pp >= rp-1:
		logf("O1C-G5 PASS")
	case delta >= -4.0 && pp >= rp-1:
		t.Errorf("O1C-G5 AMBIGUOUS (parked): delta %+.2f points", delta)
	default:
		t.Errorf("O1C-G5 FAIL: delta %+.2f points, free-run %d vs reference %d", delta, pp, rp)
	}
}

// TestKSO1C_fillerDiagnostic is LQ2 and LQ3 (docs/tasks/task-kernel-survey-2026-10.md, "#230's review, item 1"), a reading, not a
// gate: one Phi-3 resident holding both arms (kso1cLoadBothArms) and the CPU at q4k, each against the CPU at f32, over one prompt.
// The prompt is TestQ4K_phi3ResidentMatchesCPU's filler at GOINFER_KSO1C_DIAG_REPS repeats (110 by default), or, with
// GOINFER_KSO1C_DIAG_DOC set, that file's first GOINFER_KSO1C_DIAG_TOKENS tokens (900) inside "Summarize this document:". Per arm
// it prints p10 and min cosine and mean KL over all positions and over positions 256 and later; the int8 and native arms also run
// with one group native at a time (Q5_K qkv, Q6_K ffn_down, the head). It asserts only that every arm ran.
func TestKSO1C_fillerDiagnostic(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1")
	}
	path := modelPath("phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf")
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatal(err)
	}
	tmpl, err := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has})
	if err != nil {
		t.Fatal(err)
	}
	envInt := func(name string, def int) int {
		if v, err := strconv.Atoi(os.Getenv(name)); err == nil {
			return v
		}
		return def
	}
	reps, label := envInt("GOINFER_KSO1C_DIAG_REPS", 110), ""
	text := strings.Repeat("the ", reps) + "\n\nThe text above is only filler. Now write a long, detailed essay about the history of the printing press."
	label = fmt.Sprintf("filler %d repeats", reps)
	if doc := os.Getenv("GOINFER_KSO1C_DIAG_DOC"); doc != "" {
		b, err := os.ReadFile(doc)
		if err != nil {
			t.Fatal(err)
		}
		dids, err := tk.Encode(string(b), false)
		if err != nil {
			t.Fatal(err)
		}
		n := min(len(dids), envInt("GOINFER_KSO1C_DIAG_TOKENS", 900))
		body, err := tk.Decode(dids[:n])
		if err != nil {
			t.Fatal(err)
		}
		text, label = "Summarize this document:\n\n"+body, fmt.Sprintf("%s, first %d tokens", doc, n)
	}
	ids, err := tk.EncodeSegments(tmpl.RenderSegments("", []chat.Turn{{Role: "user", Content: text}}), false)
	if err != nil {
		t.Fatal(err)
	}
	ctx := envInt("GOINFER_KSO1C_DIAG_CTX", 512)
	for ctx < len(ids)+1 {
		ctx *= 2
	}
	mg, err := kso1cLoadBothArms(path, decoder.Options{Backend: "cuda", Quant: "q4k", ResidentContext: ctx})
	if err != nil {
		t.Fatal(err)
	}
	defer mg.Close()
	r, ok := mg.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatalf("no CUDA resident: %s", mg.ResidentDecline())
	}
	fmt.Fprintf(os.Stderr, "[diag] %s: %d positions; resident context %d, layer 0 KV half %v\n", label, len(ids), r.ctxCap, r.layers[0].kvHalf)
	use, _ := kso1cArms(t, r, "Phi-3")
	cpu := func(opts decoder.Options) [][]float32 {
		m, err := decoder.Load(path, opts)
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		c := m.NewCache(len(ids) + 1)
		var out [][]float32
		for _, id := range ids {
			l, err := m.ForwardForTest(id, c)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, append([]float32(nil), l...))
		}
		return out
	}
	ref := cpu(decoder.Options{Backend: "cpu"})
	cos := func(a, b []float32) float64 {
		var d, na, nb float64
		for i := range a {
			d += float64(a[i]) * float64(b[i])
			na += float64(a[i]) * float64(a[i])
			nb += float64(b[i]) * float64(b[i])
		}
		return d / math.Sqrt(na*nb)
	}
	report := func(name string, rows [][]float32) {
		for _, from := range []int{0, 256} {
			if from >= len(ids) {
				continue
			}
			var cs []float64
			kl := 0.0
			for p := from; p < len(ids); p++ {
				cs = append(cs, cos(rows[p], ref[p]))
				kl += kso1cKL(ref[p], rows[p])
			}
			sort.Float64s(cs)
			fmt.Fprintf(os.Stderr, "[diag] %s | %-26s positions %4d-%4d: cosine min %.6f p10 %.6f median %.6f; mean KL to f32 %.5f\n",
				label, name, from, len(ids)-1, cs[0], cs[len(cs)/10], cs[len(cs)/2], kl/float64(len(cs)))
		}
	}
	report("cpu q4k", cpu(decoder.Options{Backend: "cpu", Quant: "q4k"}))
	vocabRows := r.lmW.N
	arms := []struct {
		name string
		pick func(w cudaWQ) bool // which native weights are bound; nil binds none
	}{
		{"int8 arm", nil},
		{"native arm", func(cudaWQ) bool { return true }},
		{"only Q5_K qkv native", func(w cudaWQ) bool { return w.kind == "q5k" }},
		{"only Q6_K ffn_down native", func(w cudaWQ) bool { return w.kind == "q6k" && w.N != vocabRows }},
		{"only the Q6_K head native", func(w cudaWQ) bool { return w.kind == "q6k" && w.N == vocabRows }},
	}
	slots := []*cudaWQ{&r.lmW}
	for l := range r.layers {
		Ly := &r.layers[l]
		slots = append(slots, &Ly.q, &Ly.k, &Ly.v, &Ly.o, &Ly.g, &Ly.u, &Ly.d)
	}
	ran := 0
	for _, a := range arms {
		use(false)
		for _, p := range r.kquantAlt {
			if a.pick == nil || !a.pick(p[0]) {
				continue
			}
			for _, s := range slots {
				if *s == p[1] {
					*s = p[0]
				}
			}
		}
		r.Reset()
		var rows [][]float32
		for p, id := range ids {
			l, err := r.Forward(mg.EmbedResidentForTest(id), p)
			if err != nil {
				t.Fatal(err)
			}
			rows = append(rows, append([]float32(nil), l...))
		}
		report(a.name, rows)
		ran++
	}
	use(true)
	if ran != len(arms) {
		t.Fatalf("%d of %d arms ran", ran, len(arms))
	}
}
