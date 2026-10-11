//go:build gpu && goinfer_testhooks

package gpu

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

// g3Prompts is G3's prompt set, fixed in docs/tasks/task-multimodal-support-2026-10.md (S1) before any run, and the
// same eight on every backend.
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

// g3RunWebGPU is G3's procedure over g3Prompts: free-running greedy on each side, then the CPU's sequence
// teacher-forced through the WebGPU resident.
func g3RunWebGPU(t *testing.T, tk *tokenizer.Tokenizer, tmpl *chat.Template, rf decoder.ResidentForward, mg, mc *decoder.Model, logf func(string, ...any)) (passPrompts, agree, positions int) {
	t.Helper()
	fwd := func(row []float32, pos int) []float32 {
		l, err := rf.Forward(row, pos)
		if err != nil {
			t.Fatalf("webgpu pos %d: %v", pos, err)
		}
		return append([]float32(nil), l...)
	}
	dec := func(ids []int) string {
		s, err := tk.Decode(ids)
		if err != nil {
			return fmt.Sprintf("<decode: %v>", err)
		}
		return s
	}
	argmaxTie := func(cpu []float32, ga int) (int, float64) {
		ca := argmax(cpu)
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
				cpuSeq = append(cpuSeq, argmax(cpuL))
			}
		}
		// WebGPU free run, same shape.
		gpuSeq := append([]int(nil), ids...)
		for i := 0; i < n-1; i++ {
			l := fwd(mg.EmbedResidentForTest(gpuSeq[i]), i)
			if i >= len(ids)-1 {
				gpuSeq = append(gpuSeq, argmax(l))
			}
		}
		// Teacher-forced: the CPU's sequence through WebGPU again, compared with the CPU's logits position by position.
		cc = mc.NewCache(n)
		first, firstTie, promptAgree := -1, 0.0, 0
		for i := 0; i < n-1; i++ {
			cl, err := mc.ForwardForTest(cpuSeq[i], cc)
			if err != nil {
				t.Fatal(err)
			}
			cl = append([]float32(nil), cl...)
			gl := fwd(mg.EmbedResidentForTest(cpuSeq[i]), i)
			ca, gap := argmaxTie(cl, argmax(gl))
			if ca == argmax(gl) {
				promptAgree++
			} else {
				where := "prompt"
				if i >= len(ids)-1 {
					where = "generated"
				}
				logf("    disagree: prompt %d pos %d (%s) CPU gap %.2f%%", pi+1, i, where, gap*100)
			}
			if i >= len(ids)-1 && first < 0 && gpuSeq[i+1] != cpuSeq[i+1] {
				// The free runs share history up to here, so the CPU's logits at i are what WebGPU saw too.
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
		logf("prompt %d (%d prompt tokens): %s; teacher-forced %d/%d; pass=%v\n    cpu:   %q\n    webgpu: %q", pi+1, len(ids), verdict, promptAgree, n-1, ok,
			dec(cpuSeq[len(ids):]), dec(gpuSeq[len(ids):]))
	}
	return passPrompts, agree, positions
}

// g3ModelWebGPU loads one sidecar as a WebGPU resident and on the CPU, both int4 at G3's pinned 512-token context, and
// runs g3RunWebGPU on it. The tokenizer is a .gguf or a Hugging Face directory.
func g3ModelWebGPU(t *testing.T, giw, tok, label string, logf func(string, ...any)) (pass, agree, n int) {
	t.Helper()
	if strings.HasPrefix(giw, "/Volumes/") || strings.HasPrefix(giw, "/srv/models") {
		t.Fatalf("%s is on the archive, not the bench set (CLAUDE.md)", giw)
	}
	if _, err := os.Stat(giw); err != nil {
		t.Skipf("no sidecar %s: %v", giw, err)
	}
	var tk *tokenizer.Tokenizer
	var err error
	if strings.HasSuffix(tok, ".gguf") {
		tk, err = tokenizer.LoadGGUF(tok)
	} else {
		tk, err = tokenizer.Load(filepath.Join(tok, "tokenizer.json"))
	}
	if err != nil {
		t.Fatal(err)
	}
	tmpl, err := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has})
	if err != nil {
		t.Fatal(err)
	}
	mg, err := decoder.Load(giw, decoder.Options{Backend: "webgpu", Quant: "int4", ResidentContext: 512})
	if err != nil {
		t.Fatal(err)
	}
	defer mg.Close()
	rf := mg.ResidentForwardForTest()
	if rf == nil {
		t.Fatalf("%s: webgpu did not build a resident (%s)", label, mg.DecodePath())
	}
	mc, err := decoder.Load(giw, decoder.Options{Quant: "int4", ResidentContext: 512})
	if err != nil {
		t.Fatal(err)
	}
	defer mc.Close()
	logf("%s: loaded (template %s, %s)", label, tmpl.Name(), mg.DecodePath())
	pass, agree, n = g3RunWebGPU(t, tk, tmpl, rf, mg, mc, logf)
	logf("%s: %d/%d prompts pass the free-run rule; teacher-forced agreement %d/%d = %.2f%%", label, pass, len(g3Prompts), agree, n, 100*float64(agree)/float64(n))
	return pass, agree, n
}

// TestGemma4EModel_webgpuRealNonInferiority is G-S12.1d (docs/tasks/task-multimodal-support-2026-10.md, S12): S1's G3
// rule, unchanged, on WebGPU. In one process the reference Qwen2.5-Coder-1.5B runs WebGPU against the CPU, then the
// Gemma 4 E-model named by GOINFER_G3_EMODEL (e2b, the default, or e4b; the quantization-aware GGUF's sidecar). PASS
// at >= the reference's - 2.0 points with free-run passes >= the reference's - 1; 2.0-4.0 points below is parked.
// The sidecars are GOINFER_G3_REF_GIW and GOINFER_G3_EMODEL_GIW, by default the Metal-target ones beside each GGUF,
// whose int4 bytes are the layout WebGPU reads.
//
//	GOINFER_HEAVY_TESTS=1 go test -count=1 -timeout 30m -tags 'gpu goinfer_testhooks' -run '^TestGemma4EModel_webgpuRealNonInferiority$' -v ./gpu/
func TestGemma4EModel_webgpuRealNonInferiority(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1")
	}
	if _, err := New(); err != nil {
		t.Skipf("no WebGPU adapter: %v", err)
	}
	home, _ := os.UserHomeDir()
	t0 := time.Now()
	which := os.Getenv("GOINFER_G3_EMODEL")
	if which == "" {
		which = "e2b"
	}
	logf := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[G-S12.1d %s %6.1fs] %s\n", which, time.Since(t0).Seconds(), fmt.Sprintf(format, a...))
	}
	eb := map[string]string{
		"e2b": filepath.Join(home, "models", "gemma-4-e2b-gguf", "gemma-4-E2B_q4_0-it"),
		"e4b": filepath.Join(home, "models", "gemma-4-e4b-gguf", "gemma-4-E4B_q4_0-it"),
	}[which]
	if eb == "" {
		t.Fatalf("GOINFER_G3_EMODEL=%q: want e2b or e4b", which)
	}
	qb := filepath.Join(home, "models", "qwen2.5-coder-1.5b-instruct-q4_k_m")
	refGIW, eGIW := os.Getenv("GOINFER_G3_REF_GIW"), os.Getenv("GOINFER_G3_EMODEL_GIW")
	if refGIW == "" {
		refGIW = qb + ".int4.metal.giw"
	}
	if eGIW == "" {
		eGIW = eb + ".int4.metal.giw"
	}
	rp, ra, rn := g3ModelWebGPU(t, refGIW, qb+".gguf", "reference Qwen2.5-Coder-1.5B", logf)
	ep, ea, en := g3ModelWebGPU(t, eGIW, eb+".gguf", strings.ToUpper(which), logf)
	refPct, pct := 100*float64(ra)/float64(rn), 100*float64(ea)/float64(en)
	delta := pct - refPct
	logf("non-inferiority: %s %.2f%% (%d/%d prompts) vs reference %.2f%% (%d/%d prompts): delta %+.2f points", strings.ToUpper(which), pct, ep, len(g3Prompts), refPct, rp, len(g3Prompts), delta)
	switch {
	case delta < -4.0 || ep <= rp-2:
		t.Errorf("G-S12.1d FAIL: delta %+.2f points, free-run %d vs reference %d", delta, ep, rp)
	case delta < -2.0:
		t.Errorf("G-S12.1d AMBIGUOUS (parked for the owner): delta %+.2f points", delta)
	default:
		t.Logf("G-S12.1d PASS: delta %+.2f points (margin -2.0), free-run %d vs reference %d", delta, ep, rp)
	}
}
