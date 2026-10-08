//go:build cuda && goinfer_testhooks

package cuda

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// g3Prompts is G3c's prompt set: the eight prompts Metal's G3 fixed in docs/tasks/task-multimodal-support-2026-10.md before any run, unchanged.
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

// g3Run is G3c's procedure (docs/tasks/task-multimodal-support-2026-10.md, "S1 on CUDA"; Metal's G3 procedure unchanged) over g3Prompts: free-running
// greedy on each side for 32 tokens (a first divergence at a near-tie, under 3% of the CPU's |top-1|, still passes the prompt), then the CPU's sequence
// teacher-forced through the CUDA resident and compared position by position.
func g3Run(t *testing.T, tk *tokenizer.Tokenizer, tmpl *chat.Template, r *cudaResident, mg, mc *decoder.Model, logf func(string, ...any)) (passPrompts, agree, positions int) {
	t.Helper()
	dec := func(ids []int) string {
		s, err := tk.Decode(ids)
		if err != nil {
			return fmt.Sprintf("<decode: %v>", err)
		}
		return s
	}
	fwd := func(id, pos int) []float32 {
		l, err := r.Forward(mg.EmbedResidentForTest(id), pos)
		if err != nil {
			t.Fatalf("cuda forward at pos %d: %v", pos, err)
		}
		return append([]float32(nil), l...)
	}
	argmaxTie := func(cpu []float32, ga int) (int, float64) {
		ca := argmaxF(cpu)
		return ca, (float64(cpu[ca]) - float64(cpu[ga])) / (math.Abs(float64(cpu[ca])) + 1e-30)
	}
	for pi, text := range g3Prompts {
		ids, err := tk.Encode(tmpl.Render("", []chat.Turn{{Role: "user", Content: text}}), false)
		if err != nil {
			t.Fatal(err)
		}
		n := len(ids) + g3NewTokens
		// CPU free run: prefill token by token, then 32 greedy tokens.
		cc := mc.NewCache(n)
		cpuSeq := append([]int(nil), ids...)
		for i := 0; i < n-1; i++ {
			cpuL, err := mc.ForwardForTest(cpuSeq[i], cc)
			if err != nil {
				t.Fatal(err)
			}
			if i >= len(ids)-1 {
				cpuSeq = append(cpuSeq, argmaxF(cpuL))
			}
		}
		// CUDA free run, same shape.
		gpuSeq := append([]int(nil), ids...)
		for i := 0; i < n-1; i++ {
			l := fwd(gpuSeq[i], i)
			if i >= len(ids)-1 {
				gpuSeq = append(gpuSeq, argmaxF(l))
			}
		}
		// Teacher-forced: the CPU's sequence through the resident again, compared with the CPU's logits position by position.
		cc = mc.NewCache(n)
		first, firstTie, promptAgree := -1, 0.0, 0
		for i := 0; i < n-1; i++ {
			cl, err := mc.ForwardForTest(cpuSeq[i], cc)
			if err != nil {
				t.Fatal(err)
			}
			cl = append([]float32(nil), cl...)
			gl := fwd(cpuSeq[i], i)
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
				// The free runs share history up to here, so the CPU's logits at i are what the resident saw too.
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
		logf("prompt %d (%d prompt tokens): %s; teacher-forced %d/%d; pass=%v\n    cpu:  %q\n    cuda: %q", pi+1, len(ids), verdict, promptAgree, n-1, ok,
			dec(cpuSeq[len(ids):]), dec(gpuSeq[len(ids):]))
	}
	return passPrompts, agree, positions
}

// g3Model loads one GGUF (or, for the E4B, one Hugging Face directory) as a CUDA resident and on the CPU, both int4 at G3's pinned 512-token context, and runs g3Run. Table precision is the
// confound Metal's G3 run 1 fell into (its CPU arm loaded a sidecar whose embedding/LM-head/PLE tables were int4 against Metal's int8), so both
// sides load the same GGUF with the same Options.
func g3Model(t *testing.T, gguf, label string, logf func(string, ...any)) (pass, agree, n int) {
	t.Helper()
	if strings.HasPrefix(gguf, "/srv/models") || strings.HasPrefix(gguf, "/Volumes/") {
		t.Fatalf("%s is on the archive, not the bench set (CLAUDE.md)", gguf)
	}
	fi, err := os.Stat(gguf)
	if err != nil {
		t.Skipf("no checkpoint %s: %v", gguf, err)
	}
	var tk *tokenizer.Tokenizer
	if fi.IsDir() { // a Hugging Face directory (the E4B): the tokenizer, and the chat template with it, from its tokenizer.json
		tk, err = tokenizer.Load(filepath.Join(gguf, "tokenizer.json"))
	} else {
		tk, err = tokenizer.LoadGGUF(gguf)
	}
	if err != nil {
		t.Fatal(err)
	}
	tmpl, err := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has})
	if err != nil {
		t.Fatal(err)
	}
	opts := decoder.Options{Quant: "int4", ResidentContext: 512}
	gopts := opts
	gopts.Backend = "cuda"
	mg, err := decoder.Load(gguf, gopts)
	if err != nil {
		t.Fatalf("%s: load (cuda): %v", label, err)
	}
	defer mg.Close()
	r, ok := mg.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatalf("%s: no CUDA resident: %s", label, mg.ResidentDecline())
	}
	// The CPU loads the SAME file with the SAME options, so both sides hold the same quantization of every table. (G3c run 1, 2026-10-07, let the CPU
	// read the CUDA e4h sidecar while the CUDA side loaded the GGUF with Options.EmbedInt4 unset: an int4 head against an int8 pin, the confound
	// Metal's G3 run 1 fell into. It was caught in the log before anything was recorded and is superseded; see the task doc.)
	mc, err := decoder.Load(gguf, opts)
	if err != nil {
		t.Fatalf("%s: load (cpu): %v", label, err)
	}
	defer mc.Close()
	logf("%s: CUDA resident built (template %s, P=%d, hidden %d, %d layers); both sides loaded %s with %+v (EmbedInt4=%v on both)", label, tmpl.Name(), r.pleP, r.hidden, r.nLayers, gguf, opts, opts.EmbedInt4)
	pass, agree, n = g3Run(t, tk, tmpl, r, mg, mc, logf)
	logf("%s: %d/%d prompts pass the free-run rule; teacher-forced agreement %d/%d = %.2f%%", label, pass, len(g3Prompts), agree, n, 100*float64(agree)/float64(n))
	return pass, agree, n
}

// TestGemma4EModel_realE2BNonInferiority is G3c (docs/tasks/task-multimodal-support-2026-10.md, "S1 on CUDA", registered 2026-10-07 before any CUDA
// run, Metal's re-registered G3 rule unchanged): in one process, g3Run on Qwen2.5-Coder-1.5B on CUDA (the validated reference), then on E2B.
// PASS: E2B's teacher-forced agreement >= the reference's - 2.0 points and its free-run passes >= the reference's - 1; 2.0-4.0 points below is
// ambiguous (parked for the owner); worse, or free-run passes 2+ short, fails.
//
//	GOINFER_HEAVY_TESTS=1 go test -count=1 -timeout 20m -tags 'cuda goinfer_testhooks' -run '^TestGemma4EModel_realE2BNonInferiority$' -v ./cuda/
func TestGemma4EModel_realE2BNonInferiority(t *testing.T) {
	requireHeavyModel(t)
	home, _ := os.UserHomeDir()
	t0 := time.Now()
	logf := func(format string, a ...any) { // streams under -v (t.Logf is held until the test returns)
		fmt.Fprintf(os.Stderr, "[G3c %6.1fs] %s\n", time.Since(t0).Seconds(), fmt.Sprintf(format, a...))
	}
	rp, ra, rn := g3Model(t, filepath.Join(home, "models", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"), "reference Qwen2.5-Coder-1.5B", logf)
	ep, ea, en := g3Model(t, filepath.Join(home, "models", "gemma-4-e2b-gguf", "gemma-4-E2B_q4_0-it.gguf"), "E2B", logf)
	refPct, e2bPct := 100*float64(ra)/float64(rn), 100*float64(ea)/float64(en)
	delta := e2bPct - refPct
	logf("G3c non-inferiority: E2B %.2f%% (%d/%d prompts) vs reference %.2f%% (%d/%d prompts): delta %+.2f points", e2bPct, ep, len(g3Prompts), refPct, rp, len(g3Prompts), delta)
	switch {
	case delta < -4.0 || ep <= rp-2:
		t.Errorf("G3c FAIL: delta %+.2f points, free-run %d vs reference %d", delta, ep, rp)
	case delta < -2.0:
		t.Errorf("G3c AMBIGUOUS (parked for the owner): delta %+.2f points", delta)
	default:
		t.Logf("G3c PASS: delta %+.2f points (margin -2.0), free-run %d vs reference %d", delta, ep, rp)
	}
}

// TestGemma4EModel_realE2BPLEHostCost is S1's speed record on CUDA (docs/tasks/task-multimodal-support-2026-10.md, "S1 on CUDA"): the milliseconds
// embedResidentInto spends per token on E2B, i.e. the embedding row plus the PLE inputs the CPU computes before every resident step. A record, not a
// gate; timed, so it runs on the night queue under the timing lock.
//
//	GOINFER_HEAVY_TESTS=1 go test -c -tags 'cuda goinfer_testhooks' -o cuda.test ./cuda/ && GOINFER_HEAVY_TESTS=1 ./cuda.test -test.run '^TestGemma4EModel_realE2BPLEHostCost$' -test.v
func TestGemma4EModel_realE2BPLEHostCost(t *testing.T) {
	requireHeavyModel(t)
	home, _ := os.UserHomeDir()
	gguf := filepath.Join(home, "models", "gemma-4-e2b-gguf", "gemma-4-E2B_q4_0-it.gguf")
	if _, err := os.Stat(gguf); err != nil {
		t.Skipf("no E2B GGUF: %v", err)
	}
	m, err := decoder.Load(gguf, decoder.Options{Backend: "cuda", Quant: "int4", ResidentContext: 512})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	const warm, n = 32, 512
	ids := make([]int, warm+n)
	for i := range ids {
		ids[i] = 1000 + (i*7919)%200000 // spread over the vocabulary, deterministic
	}
	for _, id := range ids[:warm] {
		m.EmbedResidentForTest(id)
	}
	per := make([]float64, n)
	for i, id := range ids[warm:] {
		t0 := time.Now()
		m.EmbedResidentForTest(id)
		per[i] = float64(time.Since(t0).Microseconds()) / 1000
	}
	sum := 0.0
	for _, v := range per {
		sum += v
	}
	sorted := append([]float64(nil), per...)
	sort.Float64s(sorted)
	t.Logf("E2B embedding row + PLE inputs on the host: mean %.3f ms/token, median %.3f, p90 %.3f over %d tokens (row length %d)",
		sum/float64(n), sorted[n/2], sorted[n*9/10], n, len(m.EmbedResidentForTest(ids[0])))
}

// TestGemma4EModel_realE4BNonInferiority is G-E4B-C1 (docs/tasks/task-multimodal-support-2026-10.md, "S6 on nobara", registered 2026-10-08 before any run): G3c's procedure and rule, unchanged, on Gemma 4 E4B loaded from its
// safetensors directory by both sides.
//
//	GOINFER_HEAVY_TESTS=1 go test -count=1 -timeout 40m -tags 'cuda goinfer_testhooks' -run '^TestGemma4EModel_realE4BNonInferiority$' -v ./cuda/
func TestGemma4EModel_realE4BNonInferiority(t *testing.T) {
	requireHeavyModel(t)
	home, _ := os.UserHomeDir()
	t0 := time.Now()
	logf := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[G-E4B-C1 %6.1fs] %s\n", time.Since(t0).Seconds(), fmt.Sprintf(format, a...))
	}
	rp, ra, rn := g3Model(t, filepath.Join(home, "models", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"), "reference Qwen2.5-Coder-1.5B", logf)
	ep, ea, en := g3Model(t, filepath.Join(home, "models", "gemma-4-E4B-it"), "E4B", logf)
	refPct, e4bPct := 100*float64(ra)/float64(rn), 100*float64(ea)/float64(en)
	delta := e4bPct - refPct
	logf("G-E4B-C1 non-inferiority: E4B %.2f%% (%d/%d prompts) vs reference %.2f%% (%d/%d prompts): delta %+.2f points", e4bPct, ep, len(g3Prompts), refPct, rp, len(g3Prompts), delta)
	switch {
	case delta < -4.0 || ep <= rp-2:
		t.Errorf("G-E4B-C1 FAIL: delta %+.2f points, free-run %d vs reference %d", delta, ep, rp)
	case delta < -2.0:
		t.Errorf("G-E4B-C1 AMBIGUOUS (parked for the owner): delta %+.2f points", delta)
	default:
		t.Logf("G-E4B-C1 PASS: delta %+.2f points (margin -2.0), free-run %d vs reference %d", delta, ep, rp)
	}
}

// TestGemma4EModel_realE4BAnchorDump is an EXPLORATORY dump, not a gate. G-E4B-C1 failed (CUDA against the CPU: 87.59% teacher-forced, 3/8 free-run passes, against the E2B's 94.48% and 7/8), and a G3 comparison
// of two int4 implementations cannot say which of them, if either, is wrong. This writes, for the CPU's own greedy sequence on each of G3's prompts, both arms' top-8 (id, logit) at every position, so that an HF
// float32 forward over the SAME ids (scripts/anchor_e4b_hf.py) can say, at every position where the arms disagree, which one HF sides with. Output: $E4B_ANCHOR_DIR (default ~/goinfer-logs/e4b-anchor/dump.json).
func TestGemma4EModel_realE4BAnchorDump(t *testing.T) {
	requireHeavyModel(t)
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, "models", "gemma-4-E4B-it")
	if _, err := os.Stat(dir); err != nil {
		t.Skipf("no %s", dir)
	}
	outDir := os.Getenv("E4B_ANCHOR_DIR")
	if outDir == "" {
		outDir = filepath.Join(home, "goinfer-logs", "e4b-anchor")
	}
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		t.Fatal(err)
	}
	tk, err := tokenizer.Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	tmpl, err := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has})
	if err != nil {
		t.Fatal(err)
	}
	opts := decoder.Options{Quant: "int4", ResidentContext: 512}
	gopts := opts
	gopts.Backend = "cuda"
	mg, err := decoder.Load(dir, gopts)
	if err != nil {
		t.Fatal(err)
	}
	defer mg.Close()
	r, ok := mg.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatalf("no CUDA resident: %s", mg.ResidentDecline())
	}
	mc, err := decoder.Load(dir, opts)
	if err != nil {
		t.Fatal(err)
	}
	defer mc.Close()
	type top struct {
		IDs    []int     `json:"ids"`
		Logits []float32 `json:"logits"`
	}
	topK := func(l []float32, k int) top {
		idx := make([]int, len(l))
		for i := range idx {
			idx[i] = i
		}
		sort.Slice(idx, func(a, b int) bool { return l[idx[a]] > l[idx[b]] })
		o := top{IDs: idx[:k], Logits: make([]float32, k)}
		for i, id := range o.IDs {
			o.Logits[i] = l[id]
		}
		return o
	}
	type promptDump struct {
		Prompt    string `json:"prompt"`
		PromptLen int    `json:"prompt_len"`
		Ids       []int  `json:"ids"`
		CPU       []top  `json:"cpu"`
		CUDA      []top  `json:"cuda"`
	}
	var all []promptDump
	t0 := time.Now()
	for pi, text := range g3Prompts {
		ids, err := tk.Encode(tmpl.Render("", []chat.Turn{{Role: "user", Content: text}}), false)
		if err != nil {
			t.Fatal(err)
		}
		n := len(ids) + g3NewTokens
		cc := mc.NewCache(n)
		seq := append([]int(nil), ids...)
		var cpuTop []top
		for i := 0; i < n-1; i++ {
			l, err := mc.ForwardForTest(seq[i], cc)
			if err != nil {
				t.Fatal(err)
			}
			cpuTop = append(cpuTop, topK(l, 8))
			if i >= len(ids)-1 {
				seq = append(seq, argmaxF(l))
			}
		}
		r.Reset()
		var gpuTop []top
		for i := 0; i < n-1; i++ {
			l, err := r.Forward(mg.EmbedResidentForTest(seq[i]), i)
			if err != nil {
				t.Fatal(err)
			}
			gpuTop = append(gpuTop, topK(l, 8))
		}
		all = append(all, promptDump{Prompt: text, PromptLen: len(ids), Ids: seq, CPU: cpuTop, CUDA: gpuTop})
		fmt.Fprintf(os.Stderr, "[E4B anchor %6.1fs] prompt %d/%d dumped (%d positions)\n", time.Since(t0).Seconds(), pi+1, len(g3Prompts), n-1)
	}
	raw, err := json.Marshal(all)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(outDir, "dump.json"), raw, 0o644); err != nil {
		t.Fatal(err)
	}
}

// TestGemma4EModel_realE4BF32Dump is an EXPLORATORY follow-up to TestGemma4EModel_realE4BAnchorDump, not a gate. The anchor read both int4 arms about 77% from HF float32 (CPU 77.70%, CUDA 77.01%) against 87.59% from each other, which a shared
// deviation or plain int4 sensitivity could both produce. This removes quantization: goinfer's CPU forward in float32 (Options.Quant "f32") teacher-forced over the SAME ids, written in the dump's shape so scripts/anchor_e4b_hf.py can
// compare it with HF float32. If the implementation is right, float32 against float32 is near-identical; if it is not, the gap is the bug.
func TestGemma4EModel_realE4BF32Dump(t *testing.T) {
	requireHeavyModel(t)
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, "models", "gemma-4-E4B-it")
	outDir := os.Getenv("E4B_ANCHOR_DIR")
	if outDir == "" {
		outDir = filepath.Join(home, "goinfer-logs", "e4b-anchor")
	}
	raw, err := os.ReadFile(filepath.Join(outDir, "dump.json"))
	if err != nil {
		t.Skipf("no int4 dump to take the ids from: %v", err)
	}
	type top struct {
		IDs    []int     `json:"ids"`
		Logits []float32 `json:"logits"`
	}
	type promptDump struct {
		Prompt    string `json:"prompt"`
		PromptLen int    `json:"prompt_len"`
		Ids       []int  `json:"ids"`
		CPU       []top  `json:"cpu"`
		CUDA      []top  `json:"cuda"`
	}
	var in []promptDump
	if err := json.Unmarshal(raw, &in); err != nil {
		t.Fatal(err)
	}
	mc, err := decoder.Load(dir, decoder.Options{Quant: "f32", ResidentContext: 512})
	if err != nil {
		t.Fatal(err)
	}
	defer mc.Close()
	topK := func(l []float32, k int) top {
		idx := make([]int, len(l))
		for i := range idx {
			idx[i] = i
		}
		sort.Slice(idx, func(a, b int) bool { return l[idx[a]] > l[idx[b]] })
		o := top{IDs: idx[:k], Logits: make([]float32, k)}
		for i, id := range o.IDs {
			o.Logits[i] = l[id]
		}
		return o
	}
	t0 := time.Now()
	for pi := range in {
		d := &in[pi]
		cc := mc.NewCache(len(d.Ids))
		var tops []top
		for i := 0; i < len(d.Ids)-1; i++ {
			l, err := mc.ForwardForTest(d.Ids[i], cc)
			if err != nil {
				t.Fatal(err)
			}
			tops = append(tops, topK(l, 8))
		}
		d.CPU, d.CUDA = tops, tops // the script reads both fields; here both are goinfer's float32 CPU
		fmt.Fprintf(os.Stderr, "[E4B f32 %6.1fs] prompt %d/%d (%d positions)\n", time.Since(t0).Seconds(), pi+1, len(in), len(tops))
	}
	out, err := json.Marshal(in)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(outDir, "dump-f32.json"), out, 0o644); err != nil {
		t.Fatal(err)
	}
}

// TestGemma4EModel_realE4BQATNonInferiority is G-E4B-C1b (docs/tasks/task-multimodal-support-2026-10.md, "S6 on nobara", re-registered 2026-10-08 before any run on this file): G3c, unchanged in procedure, rule and reference, on Google's
// quantization-aware-trained E4B GGUF, the equivalent of the E2B file G3c used. The plain bf16 checkpoint (TestGemma4EModel_realE4BNonInferiority) failed for the checkpoint's sake, not the implementation's.
//
//	GOINFER_HEAVY_TESTS=1 go test -count=1 -timeout 40m -tags 'cuda goinfer_testhooks' -run '^TestGemma4EModel_realE4BQATNonInferiority$' -v ./cuda/
func TestGemma4EModel_realE4BQATNonInferiority(t *testing.T) {
	requireHeavyModel(t)
	home, _ := os.UserHomeDir()
	t0 := time.Now()
	logf := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[G-E4B-C1b %6.1fs] %s\n", time.Since(t0).Seconds(), fmt.Sprintf(format, a...))
	}
	rp, ra, rn := g3Model(t, filepath.Join(home, "models", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"), "reference Qwen2.5-Coder-1.5B", logf)
	ep, ea, en := g3Model(t, filepath.Join(home, "models", "gemma-4-e4b-gguf", "gemma-4-E4B_q4_0-it.gguf"), "E4B (QAT q4_0)", logf)
	refPct, e4bPct := 100*float64(ra)/float64(rn), 100*float64(ea)/float64(en)
	delta := e4bPct - refPct
	logf("G-E4B-C1b non-inferiority: E4B %.2f%% (%d/%d prompts) vs reference %.2f%% (%d/%d prompts): delta %+.2f points", e4bPct, ep, len(g3Prompts), refPct, rp, len(g3Prompts), delta)
	switch {
	case delta < -4.0 || ep <= rp-2:
		t.Errorf("G-E4B-C1b FAIL: delta %+.2f points, free-run %d vs reference %d", delta, ep, rp)
	case delta < -2.0:
		t.Errorf("G-E4B-C1b AMBIGUOUS (parked for the owner): delta %+.2f points", delta)
	default:
		t.Logf("G-E4B-C1b PASS: delta %+.2f points (margin -2.0), free-run %d vs reference %d", delta, ep, rp)
	}
}
