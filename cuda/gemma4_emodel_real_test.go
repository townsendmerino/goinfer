//go:build cuda && goinfer_testhooks

package cuda

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

// g3Model loads one GGUF as a CUDA resident and on the CPU, both int4 at G3's pinned 512-token context, and runs g3Run. Table precision is the
// confound Metal's G3 run 1 fell into (its CPU arm loaded a sidecar whose embedding/LM-head/PLE tables were int4 against Metal's int8), so it is
// applied up front: the CPU side loads the CUDA side's own sidecar when the CPU can read it, and the log names every file each side read.
func g3Model(t *testing.T, gguf, label string, logf func(string, ...any)) (pass, agree, n int) {
	t.Helper()
	if strings.HasPrefix(gguf, "/srv/models") || strings.HasPrefix(gguf, "/Volumes/") {
		t.Fatalf("%s is on the archive, not the bench set (CLAUDE.md)", gguf)
	}
	if _, err := os.Stat(gguf); err != nil {
		t.Skipf("no GGUF %s: %v", gguf, err)
	}
	tk, err := tokenizer.LoadGGUF(gguf)
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
	base := strings.TrimSuffix(gguf, ".gguf")
	sidecars, _ := filepath.Glob(base + ".int4*.giw")
	logf("%s: CUDA resident built (template %s, P=%d, hidden %d, %d layers); sidecars on disk next to the GGUF: %v", label, tmpl.Name(), r.pleP, r.hidden, r.nLayers, sidecars)
	cpuSrc := gguf
	for _, s := range sidecars {
		if strings.Contains(s, ".cuda.giw") {
			cpuSrc = s
		}
	}
	mc, err := decoder.Load(cpuSrc, opts)
	if err != nil {
		logf("%s: the CPU could not read %s (%v); loading the GGUF instead", label, cpuSrc, err)
		cpuSrc = gguf
		if mc, err = decoder.Load(cpuSrc, opts); err != nil {
			t.Fatalf("%s: load (cpu): %v", label, err)
		}
	}
	defer mc.Close()
	logf("%s: CPU side loaded from %s", label, cpuSrc)
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
