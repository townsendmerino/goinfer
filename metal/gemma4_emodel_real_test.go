//go:build darwin && goinfer_testhooks

package metal

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// g3Prompts is G3's prompt set, fixed in docs/tasks/task-multimodal-support-2026-10.md before any run.
var g3Prompts = []string{
	"Explain why the sky is blue in two sentences.",
	"Write a haiku about autumn leaves.",
	"What is 17 multiplied by 23? Show your work.",
	`Translate "Where is the train station?" into French and German.`,
	"List three differences between Python and Go.",
	"Summarize the plot of Romeo and Juliet in one paragraph.",
	"Write a Go function that reverses a string.",
	"What are the main causes of inflation?",
}

const g3NewTokens = 32

// TestGemma4EModel_realE2BText is S1's G3: the real Gemma 4 E2B (int4 from one GGUF) decoding resident on Metal
// against the CPU, on the eight pre-registered prompts. PASS: every prompt's 32 greedy tokens identical or first
// diverging at a near-tie (< 3%), and teacher-forced argmax agreement >= 99% pooled; 97-99% is parked for the owner.
//
//	GOINFER_HEAVY_TESTS=1 go test -count=1 -tags goinfer_testhooks -run '^TestGemma4EModel_realE2BText$' -v ./metal/
func TestGemma4EModel_realE2BText(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1 (loads the real E2B GGUF twice)")
	}
	home, _ := os.UserHomeDir()
	path := filepath.Join(home, "models", "gemma-4-e2b-gguf", "gemma-4-E2B_q4_0-it.gguf")
	if p := os.Getenv("GOINFER_GEMMA4_E2B_GGUF"); p != "" {
		path = p
	}
	if strings.HasPrefix(path, "/Volumes/") || strings.HasPrefix(path, "/srv/models") {
		t.Fatalf("%s is on the archive, not the bench set (CLAUDE.md)", path)
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no E2B GGUF: %v", err)
	}
	t0 := time.Now()
	logf := func(format string, a ...any) { // streams under -v (t.Logf is held until the test returns)
		fmt.Fprintf(os.Stderr, "[G3 %6.1fs] %s\n", time.Since(t0).Seconds(), fmt.Sprintf(format, a...))
	}
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatal(err)
	}
	tmpl, err := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has})
	if err != nil {
		t.Fatal(err)
	}
	// Each side loads its own pre-built sidecar of that GGUF (file-backed weights, so the load-time fit guard prices
	// only KV + scratch), at a pinned 512-token context: G3's longest sequence is well under it.
	dir, base := filepath.Dir(path), strings.TrimSuffix(filepath.Base(path), ".gguf")
	metalGiw := filepath.Join(dir, base+".int4.metal.giw")
	for _, p := range []string{metalGiw} {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("no sidecar %s (load the GGUF once on each backend to build it): %v", p, err)
		}
	}
	opts := decoder.Options{Quant: "int4", ResidentContext: 512}
	mg, err := decoder.Load(metalGiw, opts)
	if err != nil {
		t.Fatalf("load (metal): %v", err)
	}
	defer mg.Close()
	if why := residentMemoryDecline(mg); why != "" {
		t.Fatalf("Metal's fit guard declines E2B: %s", why)
	}
	if missing := mg.MissingResidentFeatures(decoder.ResidentBackendFeatures("metal")); len(missing) > 0 {
		t.Fatalf("metal declines E2B: missing %v", missing)
	}
	r, err := buildResident(mg)
	if err != nil {
		t.Fatalf("buildResident: %v", err)
	}
	defer r.Close()
	shared := 0
	for _, L := range r.layers {
		if L.kvShared {
			shared++
		}
	}
	logf("E2B resident on Metal: template %s, P=%d, %d KV-shared layers, FFN %d / %d", tmpl.Name(), r.pleP, shared, r.layers[0].ffnI, r.layers[len(r.layers)-1].ffnI)
	// G3 amendment (after run 1, docs/tasks/task-multimodal-support-2026-10.md): the CPU loads Metal's sidecar too, so
	// both sides run the same int8-pinned embedding/LM-head/PLE tables. Run 1 loaded the CPU's own e4h sidecar (those
	// tables at int4), which compared two quantizations rather than two engines on one set of weights.
	mc, err := decoder.Load(metalGiw, opts)
	if err != nil {
		t.Fatalf("load (cpu): %v", err)
	}
	defer mc.Close()
	logf("both loads done")

	passPrompts, agree, positions := g3Run(t, tk, tmpl, r, mg, mc, logf)
	rate := float64(agree) / float64(positions)
	logf("G3: %d/%d prompts pass the free-run rule; teacher-forced agreement %d/%d = %.2f%%", passPrompts, len(g3Prompts), agree, positions, rate*100)
	switch {
	case passPrompts < len(g3Prompts) || rate < 0.97:
		t.Errorf("G3 FAIL: %d/%d prompts pass, agreement %.2f%%", passPrompts, len(g3Prompts), rate*100)
	case rate < 0.99:
		t.Errorf("G3 AMBIGUOUS (parked for the owner): agreement %.2f%% in 97-99%%", rate*100)
	default:
		t.Logf("G3 PASS: %d/%d prompts, agreement %.2f%%", passPrompts, len(g3Prompts), rate*100)
	}
}

// TestGemma4EModel_realE2BTableControl is G3's control arm (registered with the amendment after run 1; reported, not
// graded): the same teacher-forced comparison, CPU against CPU, Metal's sidecar (int8-pinned embedding/LM-head/PLE
// tables) against the CPU's e4h sidecar (those tables at int4). It measures how much disagreement the table precision
// alone produces — the confound in G3 run 1.
//
//	GOINFER_HEAVY_TESTS=1 go test -count=1 -tags goinfer_testhooks -run '^TestGemma4EModel_realE2BTableControl$' -v ./metal/
func TestGemma4EModel_realE2BTableControl(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1")
	}
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, "models", "gemma-4-e2b-gguf")
	base := "gemma-4-E2B_q4_0-it"
	a8, a4 := filepath.Join(dir, base+".int4.metal.giw"), filepath.Join(dir, base+".int4.e4h.cpu-arm64.giw")
	for _, p := range []string{a8, a4} {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("no sidecar %s: %v", p, err)
		}
	}
	tk, err := tokenizer.LoadGGUF(filepath.Join(dir, base+".gguf"))
	if err != nil {
		t.Fatal(err)
	}
	tmpl, err := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has})
	if err != nil {
		t.Fatal(err)
	}
	opts := decoder.Options{Quant: "int4", ResidentContext: 512}
	m8, err := decoder.Load(a8, opts)
	if err != nil {
		t.Fatal(err)
	}
	defer m8.Close()
	opts.Backend = "cpu" // the e4h sidecar is the repacked cpu-arm64 layout
	m4, err := decoder.Load(a4, opts)
	if err != nil {
		t.Fatal(err)
	}
	defer m4.Close()
	agree, positions := 0, 0
	for pi, text := range g3Prompts {
		ids, err := tk.Encode(tmpl.Render("", []chat.Turn{{Role: "user", Content: text}}), false)
		if err != nil {
			t.Fatal(err)
		}
		n := len(ids) + g3NewTokens
		c8, c4 := m8.NewCache(n), m4.NewCache(n)
		seq := append([]int(nil), ids...)
		pa := 0
		for i := 0; i < n-1; i++ {
			l8, err := m8.ForwardForTest(seq[i], c8)
			if err != nil {
				t.Fatal(err)
			}
			a := argmaxF(l8)
			l4, err := m4.ForwardForTest(seq[i], c4)
			if err != nil {
				t.Fatal(err)
			}
			if argmaxF(l4) == a {
				pa++
			}
			if i >= len(ids)-1 {
				seq = append(seq, a)
			}
		}
		agree += pa
		positions += n - 1
		fmt.Fprintf(os.Stderr, "[G3 control] prompt %d: CPU int8 tables vs CPU int4 (e4h) tables, teacher-forced %d/%d\n", pi+1, pa, n-1)
	}
	t.Logf("G3 control: CPU int8-pinned tables vs CPU e4h tables, teacher-forced agreement %d/%d = %.2f%% (reported, not graded)", agree, positions, 100*float64(agree)/float64(positions))
}

// TestGemma4EModel_realE2BLocalize is exploratory, not a gate: G3's largest teacher-forced disagreement (prompt 6,
// position 18, CPU gap 7.63% in run 2) localized per layer. Both sides replay the prompt to that position, then the
// residual stream after every layer is compared. Smooth decay with depth reads as accumulated quantization noise; a
// step at one layer (the first KV-shared layer, 15, or a PLE-heavy one) reads as a defect.
//
//	GOINFER_HEAVY_TESTS=1 go test -count=1 -tags goinfer_testhooks -run '^TestGemma4EModel_realE2BLocalize$' -v ./metal/
func TestGemma4EModel_realE2BLocalize(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1")
	}
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, "models", "gemma-4-e2b-gguf")
	base := "gemma-4-E2B_q4_0-it"
	giw := filepath.Join(dir, base+".int4.metal.giw")
	if _, err := os.Stat(giw); err != nil {
		t.Skipf("no sidecar: %v", err)
	}
	tk, err := tokenizer.LoadGGUF(filepath.Join(dir, base+".gguf"))
	if err != nil {
		t.Fatal(err)
	}
	tmpl, err := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has})
	if err != nil {
		t.Fatal(err)
	}
	opts := decoder.Options{Quant: "int4", ResidentContext: 512}
	mg, err := decoder.Load(giw, opts)
	if err != nil {
		t.Fatal(err)
	}
	defer mg.Close()
	r, err := buildResident(mg)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	mc, err := decoder.Load(giw, opts)
	if err != nil {
		t.Fatal(err)
	}
	defer mc.Close()
	for _, c := range []struct{ prompt, pos int }{{6, 18}, {3, 46}, {1, 5}} {
		ids, err := tk.Encode(tmpl.Render("", []chat.Turn{{Role: "user", Content: g3Prompts[c.prompt-1]}}), false)
		if err != nil {
			t.Fatal(err)
		}
		// Teacher-forced sequence: the prompt, then the CPU's greedy tokens, up to c.pos.
		cc := mc.NewCache(c.pos + 1)
		seq := append([]int(nil), ids...)
		for i := 0; i < c.pos; i++ {
			l, err := mc.ForwardForTest(seq[i], cc)
			if err != nil {
				t.Fatal(err)
			}
			if i >= len(ids)-1 {
				seq = append(seq, argmaxF(l))
			}
			r.ForwardEmb(mg.EmbedResidentForTest(seq[i]), i)
		}
		decoder.SetGemma4HiddenCaptureForTest(true)
		if _, err := mc.ForwardForTest(seq[c.pos], cc); err != nil {
			t.Fatal(err)
		}
		cpuH := decoder.Gemma4HiddenCaptureForTest()
		decoder.SetGemma4HiddenCaptureForTest(false)
		emb := mg.EmbedResidentForTest(seq[c.pos])
		var line strings.Builder
		for n := 1; n <= len(r.layers); n++ {
			cs, _ := cosMaxAbs(cpuH[n], r.forwardTrunkForTest(emb, c.pos, n))
			fmt.Fprintf(&line, " L%d:%.5f", n-1, cs)
		}
		t.Logf("prompt %d pos %d (token %d) per-layer residual cosine Metal vs CPU:%s", c.prompt, c.pos, seq[c.pos], line.String())
	}
}

// g3Run is G3's procedure (docs/tasks/task-multimodal-support-2026-10.md) over g3Prompts: free-running greedy on each
// side, then the CPU's sequence teacher-forced through Metal. Shared by G3 and its calibration run.
func g3Run(t *testing.T, tk *tokenizer.Tokenizer, tmpl *chat.Template, r *resident, mg, mc *decoder.Model, logf func(string, ...any)) (passPrompts, agree, positions int) {
	t.Helper()
	dec := func(ids []int) string {
		s, err := tk.Decode(ids)
		if err != nil {
			return fmt.Sprintf("<decode: %v>", err)
		}
		return s
	}
	argmaxTie := func(cpu []float32, ga int) (int, float64) {
		ca := argmaxF(cpu)
		return ca, (float64(cpu[ca]) - float64(cpu[ga])) / (math.Abs(float64(cpu[ca])) + 1e-30)
	}
	passPrompts, agree, positions = 0, 0, 0
	for pi, text := range g3Prompts {
		ids, err := tk.Encode(tmpl.Render("", []chat.Turn{{Role: "user", Content: text}}), false)
		if err != nil {
			t.Fatal(err)
		}
		n := len(ids) + g3NewTokens
		// CPU free run: prefill token by token, then 32 greedy tokens.
		cc := mc.NewCache(n)
		cpuSeq := append([]int(nil), ids...)
		var cpuL []float32
		for i := 0; i < n-1; i++ {
			if cpuL, err = mc.ForwardForTest(cpuSeq[i], cc); err != nil {
				t.Fatal(err)
			}
			if i >= len(ids)-1 {
				cpuSeq = append(cpuSeq, argmaxF(cpuL))
			}
		}
		// Metal free run, same shape.
		gpuSeq := append([]int(nil), ids...)
		for i := 0; i < n-1; i++ {
			l := r.ForwardEmb(mg.EmbedResidentForTest(gpuSeq[i]), i)
			if i >= len(ids)-1 {
				gpuSeq = append(gpuSeq, argmaxF(l))
			}
		}
		// Teacher-forced: the CPU's sequence through Metal again, compared with the CPU's logits position by position.
		cc = mc.NewCache(n)
		first, firstTie, promptAgree := -1, 0.0, 0
		for i := 0; i < n-1; i++ {
			cl, err := mc.ForwardForTest(cpuSeq[i], cc)
			if err != nil {
				t.Fatal(err)
			}
			cl = append([]float32(nil), cl...)
			gl := r.ForwardEmb(mg.EmbedResidentForTest(cpuSeq[i]), i)
			ca, gap := argmaxTie(cl, argmaxF(gl))
			if ca == argmaxF(gl) {
				promptAgree++
			} else {
				where := "prompt"
				if i >= len(ids)-1 {
					where = "generated"
				}
				logf("    disagree: prompt %d pos %d (%s) CPU gap %.2f%%", pi+1, i, where, gap*100)
			}
			if i >= len(ids)-1 && first < 0 && gpuSeq[i+1] != cpuSeq[i+1] {
				// The free runs share history up to here, so the CPU's logits at i are what Metal saw too.
				first = i + 1 - len(ids)
				_, firstTie = argmaxTie(cl, gpuSeq[i+1])
			}
		}
		agree += promptAgree
		positions += n - 1
		ok := first < 0 || firstTie < 0.03
		if ok {
			passPrompts++
		}
		verdict := "identical"
		if first >= 0 {
			verdict = fmt.Sprintf("first divergence at generated token %d, CPU gap %.2f%%", first, firstTie*100)
		}
		logf("prompt %d (%d prompt tokens): %s; teacher-forced %d/%d; pass=%v\n    cpu:   %q\n    metal: %q", pi+1, len(ids), verdict, promptAgree, n-1, ok,
			dec(cpuSeq[len(ids):]), dec(gpuSeq[len(ids):]))
	}
	return passPrompts, agree, positions
}

// TestG3Calibration_qwen15b is exploratory, not a gate: G3's exact procedure (g3Run, the same eight prompts, both sides
// on one Metal sidecar, int4) on a model whose Metal resident path is already validated, Qwen2.5-Coder-1.5B. It says
// what teacher-forced agreement a known-good Metal path reaches on this instrument, which G3's 99%/97% bands were set
// without.
//
//	GOINFER_HEAVY_TESTS=1 go test -count=1 -tags goinfer_testhooks -run '^TestG3Calibration_qwen15b$' -v ./metal/
func TestG3Calibration_qwen15b(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1")
	}
	home, _ := os.UserHomeDir()
	base := filepath.Join(home, "models", "qwen2.5-coder-1.5b-instruct-q4_k_m")
	giw := base + ".int4.metal.giw"
	if _, err := os.Stat(giw); err != nil {
		t.Skipf("no sidecar: %v", err)
	}
	t0 := time.Now()
	logf := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[G3cal %6.1fs] %s\n", time.Since(t0).Seconds(), fmt.Sprintf(format, a...))
	}
	tk, err := tokenizer.LoadGGUF(base + ".gguf")
	if err != nil {
		t.Fatal(err)
	}
	tmpl, err := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has})
	if err != nil {
		t.Fatal(err)
	}
	opts := decoder.Options{Quant: "int4", ResidentContext: 512}
	mg, err := decoder.Load(giw, opts)
	if err != nil {
		t.Fatal(err)
	}
	defer mg.Close()
	r, err := buildResident(mg)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	mc, err := decoder.Load(giw, opts)
	if err != nil {
		t.Fatal(err)
	}
	defer mc.Close()
	pass, agree, n := g3Run(t, tk, tmpl, r, mg, mc, logf)
	logf("calibration (Qwen2.5-Coder-1.5B, template %s): %d/%d prompts pass the free-run rule; teacher-forced agreement %d/%d = %.2f%% (reported, not graded)",
		tmpl.Name(), pass, len(g3Prompts), agree, n, 100*float64(agree)/float64(n))
}

// g3Model loads one Metal sidecar twice (Metal resident + CPU), at G3's pinned context, and runs g3Run on it.
func g3Model(t *testing.T, giw, tokGGUF, label string, logf func(string, ...any)) (pass, agree, n int) {
	t.Helper()
	if strings.HasPrefix(giw, "/Volumes/") || strings.HasPrefix(giw, "/srv/models") {
		t.Fatalf("%s is on the archive, not the bench set (CLAUDE.md)", giw)
	}
	if _, err := os.Stat(giw); err != nil {
		t.Skipf("no sidecar %s: %v", giw, err)
	}
	tk, err := tokenizer.LoadGGUF(tokGGUF)
	if err != nil {
		t.Fatal(err)
	}
	tmpl, err := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has})
	if err != nil {
		t.Fatal(err)
	}
	opts := decoder.Options{Quant: "int4", ResidentContext: 512}
	mg, err := decoder.Load(giw, opts)
	if err != nil {
		t.Fatal(err)
	}
	defer mg.Close()
	if why := residentMemoryDecline(mg); why != "" {
		t.Fatalf("%s: Metal's fit guard declines it: %s", label, why)
	}
	r, err := buildResident(mg)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	mc, err := decoder.Load(giw, opts)
	if err != nil {
		t.Fatal(err)
	}
	defer mc.Close()
	logf("%s: loaded (template %s)", label, tmpl.Name())
	pass, agree, n = g3Run(t, tk, tmpl, r, mg, mc, logf)
	logf("%s: %d/%d prompts pass the free-run rule; teacher-forced agreement %d/%d = %.2f%%", label, pass, len(g3Prompts), agree, n, 100*float64(agree)/float64(n))
	return pass, agree, n
}

// TestGemma4EModel_realE2BNonInferiority is G3 as re-registered (docs/tasks/task-multimodal-support-2026-10.md, owner
// decision 2026-10-06): in one process, g3Run on Qwen2.5-Coder-1.5B (whose Metal path is validated: the reference),
// then on E2B. PASS: E2B's teacher-forced agreement >= the reference's - 2.0 points and its free-run passes >= the
// reference's - 1; 2.0-4.0 points below is ambiguous (parked); worse, or free-run passes 2+ short, fails.
//
//	GOINFER_HEAVY_TESTS=1 go test -count=1 -timeout 12m -tags goinfer_testhooks -run '^TestGemma4EModel_realE2BNonInferiority$' -v ./metal/
func TestGemma4EModel_realE2BNonInferiority(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1")
	}
	home, _ := os.UserHomeDir()
	t0 := time.Now()
	logf := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[G3ni %6.1fs] %s\n", time.Since(t0).Seconds(), fmt.Sprintf(format, a...))
	}
	qb := filepath.Join(home, "models", "qwen2.5-coder-1.5b-instruct-q4_k_m")
	rp, ra, rn := g3Model(t, qb+".int4.metal.giw", qb+".gguf", "reference Qwen2.5-Coder-1.5B", logf)
	eb := filepath.Join(home, "models", "gemma-4-e2b-gguf", "gemma-4-E2B_q4_0-it")
	ep, ea, en := g3Model(t, eb+".int4.metal.giw", eb+".gguf", "E2B", logf)
	refPct, e2bPct := 100*float64(ra)/float64(rn), 100*float64(ea)/float64(en)
	delta := e2bPct - refPct
	logf("G3 non-inferiority: E2B %.2f%% (%d/%d prompts) vs reference %.2f%% (%d/%d prompts): delta %+.2f points", e2bPct, ep, len(g3Prompts), refPct, rp, len(g3Prompts), delta)
	switch {
	case delta < -4.0 || ep <= rp-2:
		t.Errorf("G3 FAIL: delta %+.2f points, free-run %d vs reference %d", delta, ep, rp)
	case delta < -2.0:
		t.Errorf("G3 AMBIGUOUS (parked for the owner): delta %+.2f points", delta)
	default:
		t.Logf("G3 PASS: delta %+.2f points (margin -2.0), free-run %d vs reference %d", delta, ep, rp)
	}
}
