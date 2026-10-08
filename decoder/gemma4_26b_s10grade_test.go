package decoder

import (
	"encoding/binary"
	"encoding/json"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestGemma4_26B_s10Grade is the grading half of S1.0's 26B re-check (docs/tasks/task-multimodal-support-2026-10.md),
// run on nobara: the CPU 26B forward over the sequences the Mac's Metal dump (metal TestGemma4_26B_s10Dump) recorded,
// position by position, against both Metal arms. PASS (S1.0's rule, registered before the dump ran): with the fixes in,
// neither the mean Metal-vs-CPU logit cosine nor the argmax agreement is lower than with them dropped.
//
// NEVER on the 16 GB Mac: it is a CPU forward of the 26B (the M26 rule). GOINFER_S10_DUMP_DIR is the Mac's dump
// (seqs.json, logits-fixed.f32, logits-drop-both.f32); GOINFER_S10_GIW the SAME .int4.metal.giw the Mac loaded.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_S10_DUMP_DIR=<dir> GOINFER_S10_GIW=<giw> go test -count=1 -timeout 60m \
//	    -run '^TestGemma4_26B_s10Grade$' -v ./decoder/
func TestGemma4_26B_s10Grade(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1")
	}
	dir, giw := os.Getenv("GOINFER_S10_DUMP_DIR"), os.Getenv("GOINFER_S10_GIW")
	if dir == "" || giw == "" {
		t.Skip("set GOINFER_S10_DUMP_DIR and GOINFER_S10_GIW")
	}
	if host, _ := os.Hostname(); strings.Contains(strings.ToLower(host), "macbook") {
		t.Fatal("this is a CPU forward of the 26B: never on the 16 GB Mac (the M26 rule)")
	}
	raw, err := os.ReadFile(filepath.Join(dir, "seqs.json"))
	if err != nil {
		t.Fatal(err)
	}
	var s struct {
		Seqs [][]int `json:"seqs"`
	}
	if err := json.Unmarshal(raw, &s); err != nil {
		t.Fatal(err)
	}
	arms := []string{"fixed", "drop-both"}
	if _, err := os.Stat(filepath.Join(dir, "logits-cuda.f32")); err == nil {
		arms = append(arms, "cuda") // nobara's CUDA resident over the same sequences and .giw (cuda TestGemma4_26B_s10DumpCUDA): the third arm that separates a Metal-specific divergence from a shared one
	}
	files := map[string]*os.File{}
	for _, a := range arms {
		f, err := os.Open(filepath.Join(dir, "logits-"+a+".f32"))
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		files[a] = f
	}
	m, err := Load(giw, Options{Quant: "int4"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	V := m.w.arch.VocabSize
	buf := make([]byte, 4*V)
	read := func(f *os.File) []float32 {
		if _, err := io.ReadFull(f, buf); err != nil {
			t.Fatalf("dump shorter than the sequences: %v", err)
		}
		out := make([]float32, V)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[4*i:]))
		}
		return out
	}
	sumCos := map[string]float64{}
	agree := map[string]int{}
	nearG1, nearG3, nearAny := map[string]int{}, map[string]int{}, map[string]int{}
	n := 0
	for pi, seq := range s.Seqs {
		c := m.NewCache(len(seq))
		for i := 0; i < len(seq)-1; i++ {
			cpu, err := m.forward(seq[i], c)
			if err != nil {
				t.Fatal(err)
			}
			for _, a := range arms {
				mt := read(files[a])
				var dot, na, nb float64
				for j := range cpu {
					dot += float64(cpu[j]) * float64(mt[j])
					na += float64(cpu[j]) * float64(cpu[j])
					nb += float64(mt[j]) * float64(mt[j])
				}
				sumCos[a] += dot / math.Sqrt(na*nb)
				t.Logf("POSCOS %-9s prompt %d pos %2d: cosine %.6f", a, pi+1, i, dot/math.Sqrt(na*nb))
				if argmax(cpu) == argmax(mt) {
					agree[a]++
				} else {
					// Per-position margins for the disagreement (the owner's follow-up of 2026-10-07: is the 26B's low argmax agreement near-ties?).
					// G1c's rule: benign iff the CPU's top-1 and the index Metal chose are within 3% in the CPU's logits (relative to the CPU's top-1).
					// The G3 rule is on probability: p(Metal's choice) >= half p(CPU's top-1), the softmax taken over the CPU's logits.
					ct, mtop := argmax(cpu), argmax(mt)
					gap := (float64(cpu[ct]) - float64(cpu[mtop])) / (math.Abs(float64(cpu[ct])) + 1e-30)
					pr := math.Exp(float64(cpu[mtop]) - float64(cpu[ct]))
					sec := math.Inf(-1)
					for j := range cpu {
						if j != ct && float64(cpu[j]) > sec {
							sec = float64(cpu[j])
						}
					}
					tie1, tie2 := gap <= 0.03, pr >= 0.5
					if tie1 || tie2 {
						nearAny[a]++
					}
					if tie1 {
						nearG1[a]++
					}
					if tie2 {
						nearG3[a]++
					}
					t.Logf("DISAGREE %-9s prompt %d pos %2d: cpu top %d (%.3f) vs metal %d (cpu logit %.3f): gap %.2f%% p-ratio %.3f cpu top1-top2 margin %.3f | G1 near-tie %v, G3 near-tie %v",
						a, pi+1, i, ct, cpu[ct], mtop, cpu[mtop], gap*100, pr, float64(cpu[ct])-sec, tie1, tie2)
				}
			}
			n++
		}
		t.Logf("prompt %d: %d positions", pi+1, len(seq)-1)
	}
	for _, a := range arms {
		t.Logf("%-9s mean Metal-vs-CPU cosine %.6f, argmax agreement %d/%d", a, sumCos[a]/float64(n), agree[a], n)
		t.Logf("%-9s of the %d disagreements: %d are G1 near-ties (within 3%% of the CPU's top logit), %d are G3 near-ties (p >= half); hard (neither): %d", a, n-agree[a], nearG1[a], nearG3[a], n-agree[a]-nearAny[a])
	}
	if sumCos["fixed"] < sumCos["drop-both"] || agree["fixed"] < agree["drop-both"] {
		t.Errorf("S1.0 26B re-check FAIL: a metric moved the wrong way with the fixes in")
	} else {
		t.Logf("S1.0 26B re-check PASS")
	}
}
