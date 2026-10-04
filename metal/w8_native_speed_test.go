//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestW8Native_S_decodeSpeed is gate S of docs/tasks/task-metal-int8-2026-10.md, a night-queue measurement: an
// in-process, interleaved A/B of decode speed for an int8int8 model in four arms. They are Metal on the native int8
// path (precise math since 2026-10-04), the same with fast math kept (reported: the precise-math decision's price),
// Metal re-quantized to int4 (what -backend metal ran before), and the CPU (what -backend auto gives an int8 model). Each sample loads one arm, prefills a deterministic prompt of the given depth through Generate, and times
// the greedy decode steps after the first token, as `fit -measure` does, so prefill is excluded. The arm order rotates
// every repetition. Samples go to stderr as they finish and, with GOINFER_W8_GATE_OUT set, to that JSONL file.
// The 7B joins with GOINFER_W8_GATE_7B=1 when the fit guard admits it; nothing bypasses the guard.
// GOINFER_W8_GATE_S_SMOKE=1 shrinks it to one sample per arm on the 0.5B at depth 128, to check the harness by day.
func TestW8Native_S_decodeSpeed(t *testing.T) {
	requireHeavyModel(t)
	if os.Getenv("GOINFER_W8_GATE_S") == "" {
		t.Skip("gate S is a night-queue measurement (docs/tasks/task-metal-int8-2026-10.md): set GOINFER_W8_GATE_S=1")
	}
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	models := []string{
		os.ExpandEnv("$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"),
		os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"),
	}
	if os.Getenv("GOINFER_W8_GATE_7B") != "" {
		models = append(models, os.ExpandEnv("$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf"))
	}
	depths := []int{128, 2048}
	reps := 7
	const decode = 64
	if os.Getenv("GOINFER_W8_GATE_S_SMOKE") != "" { // a daytime check that the harness runs; its numbers are not a result
		models, depths, reps = models[:1], []int{128}, 1
	}
	arms := []w8Arm{{"metal-int8-native", "metal", true, false}, {"metal-int8-native-fastmath", "metal", true, true},
		{"metal-int8-requant", "metal", false, false}, {"cpu-int8int8", "cpu", false, false}}

	var sink *json.Encoder
	if p := os.Getenv("GOINFER_W8_GATE_OUT"); p != "" {
		f, err := os.Create(p)
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		sink = json.NewEncoder(f)
	}
	total, done, start := len(models)*len(depths)*reps*len(arms), 0, time.Now()
	type cell struct {
		model string
		depth int
	}
	rates := map[cell]map[string][]float64{}
	for _, path := range models {
		if _, err := os.Stat(path); err != nil {
			t.Logf("no checkpoint at %s: its %d samples are not taken", path, len(depths)*reps*len(arms))
			total -= len(depths) * reps * len(arms)
			continue
		}
		name := strings.TrimSuffix(path[strings.LastIndex(path, "/")+1:], ".gguf")
		tk, err := tokenizer.LoadGGUF(path)
		if err != nil {
			t.Fatalf("tokenizer %s: %v", path, err)
		}
		for _, depth := range depths {
			c := cell{name, depth}
			prompt := w8SpeedPrompt(t, tk, depth)
			rates[c] = map[string][]float64{}
			for rep := range reps {
				for j := range arms {
					a := arms[(j+rep)%len(arms)]
					s := sampleW8Decode(t, path, a, prompt, decode)
					done++
					if s.Rate > 0 {
						rates[c][a.name] = append(rates[c][a.name], s.Rate)
					}
					fmt.Fprintf(os.Stderr, "[S] %d/%d %s depth=%d rep=%d %s: %.1f tok/s over %d steps (%s)%s elapsed=%s\n",
						done, total, name, depth, rep, a.name, s.Rate, s.Steps, s.Path, s.noteSuffix(), time.Since(start).Round(time.Second))
					if sink != nil {
						s.Model, s.Depth, s.Rep, s.Arm = name, depth, rep, a.name
						if err := sink.Encode(s); err != nil {
							t.Fatal(err)
						}
					}
				}
			}
		}
	}
	for c, byArm := range rates {
		native := median(byArm["metal-int8-native"])
		t.Logf("S %s depth %d: native fast math %.1f tok/s, precise/fast %.3f (the precise-math decision's price, reported)",
			c.model, c.depth, median(byArm["metal-int8-native-fastmath"]), native/median(byArm["metal-int8-native-fastmath"]))
		t.Logf("S %s depth %d: median tok/s native %.1f, requant %.1f, cpu %.1f; native/requant %.3f, native/cpu %.3f (n = %d/%d/%d)",
			c.model, c.depth, native, median(byArm["metal-int8-requant"]), median(byArm["cpu-int8int8"]),
			native/median(byArm["metal-int8-requant"]), native/median(byArm["cpu-int8int8"]),
			len(byArm["metal-int8-native"]), len(byArm["metal-int8-requant"]), len(byArm["cpu-int8int8"]))
	}
}

type w8Arm struct {
	name, backend string
	native        bool // the Metal native int8 path; false on Metal is the int4 re-quant
	fastMath      bool // the native path with fast math kept (w8FastMath): reported, prices the precise-math default
}

// w8Sample is one decode-speed sample; Rate is 0 when the arm could not be measured (Note says why).
type w8Sample struct {
	Model string  `json:"model"`
	Depth int     `json:"depth"`
	Rep   int     `json:"rep"`
	Arm   string  `json:"arm"`
	Rate  float64 `json:"tok_per_s"`
	Steps int     `json:"decode_steps"`
	Path  string  `json:"decode_path"`
	Note  string  `json:"note,omitempty"`
}

func (s w8Sample) noteSuffix() string {
	if s.Note == "" {
		return ""
	}
	return " — " + s.Note
}

// w8SpeedPrompt is depth tokens of short Go functions in the model's own tokenization: a pattern a coder model
// continues rather than ends, so the timed steps are not cut short by EOS. A pseudo-random id sequence (fit
// -measure's probe) ended at once here: the harness's first daytime run timed 0 steps in every arm.
func w8SpeedPrompt(t *testing.T, tk *tokenizer.Tokenizer, depth int) []int {
	t.Helper()
	var b strings.Builder
	for i := 0; ; i++ {
		fmt.Fprintf(&b, "func f%d(x int) int {\n\treturn x*%d + %d\n}\n\n", i, i%7+2, i%13)
		if b.Len() > 8*depth {
			break
		}
	}
	ids, err := tk.Encode(b.String(), false)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if len(ids) < depth {
		t.Fatalf("prompt is %d tokens, want %d", len(ids), depth)
	}
	return ids[:depth]
}

// sampleW8Decode loads path for arm, checks the arm took the path it names, and times n greedy steps after
// prompt.
func sampleW8Decode(t *testing.T, path string, a w8Arm, prompt []int, n int) w8Sample {
	t.Helper()
	prev, prevFast := nativeInt8, w8FastMath
	nativeInt8, w8FastMath = a.native, a.fastMath
	defer func() { nativeInt8, w8FastMath = prev, prevFast }()
	m, err := decoder.Load(path, decoder.Options{Backend: a.backend, Quant: "int8int8", ResidentContext: len(prompt) + n + 64})
	if errors.Is(err, decoder.ErrWontFitResident) {
		return w8Sample{Note: "the fit guard refused the load: " + err.Error()}
	}
	if err != nil {
		t.Fatalf("load %s (%s): %v", path, a.name, err)
	}
	defer m.Close()
	s := w8Sample{Path: m.DecodePath()}
	if a.backend == "metal" {
		ra, ok := m.ResidentForwardForTest().(*metalResident)
		if !ok {
			s.Note = "the metal resident declined: " + m.ResidentDecline()
			return s
		}
		if ra.r.w8 != a.native {
			t.Fatalf("%s: native int8 path = %v, want %v (%s)", a.name, ra.r.w8, a.native, s.Path)
		}
	} else if !strings.HasPrefix(s.Path, "cpu") {
		t.Fatalf("%s: decode path %q, want the CPU", a.name, s.Path)
	}
	out, gen := m.Generate(context.Background(), prompt, n, decoder.SamplingParams{Temperature: 0})
	got := 0
	var first time.Time
	for range out {
		got++
		if got == 1 {
			first = time.Now()
		}
	}
	end := time.Now()
	if err := gen.Err(); err != nil {
		t.Fatalf("%s: generate: %v", a.name, err)
	}
	if got < 9 {
		s.Note = fmt.Sprintf("only %d tokens before EOS", got)
		return s
	}
	s.Steps = got - 1
	s.Rate = float64(s.Steps) / end.Sub(first).Seconds()
	return s
}
