//go:build darwin

package metal

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/embeddinggemma2"
)

const eg2TinyDir = "../testdata/embeddinggemma2-tiny"

func eg2Cos(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / math.Sqrt(na*nb)
}

func eg2Pool(h []float32, T, d int) []float32 {
	out := make([]float32, d)
	for t := range T {
		for j := range d {
			out[j] += h[t*d+j] / float32(T)
		}
	}
	return out
}

// TestEG2Metal_matchesCPU is Phase M's Gate 1 (docs/tasks/task-embeddinggemma2.md): on the tiny fixture, the Metal
// accelerator's forward matches the CPU forward (itself gated against transformers, Gate 1 of the CPU path) to cosine
// 0.9999 on every layer's input, on the projected last hidden state and on the pooled embedding. Cases: the golden's
// four (1, 2, 9 and 17 tokens; sliding radius 3 and full layers of different head dims and KV-head counts) and two
// random inputs of 300 and 700 tokens, each at attention blocks of 256 (the default), 4 and 5 query rows, so blocks
// cut through the radius-3 window and leave partial last blocks. The tiny fixture's head dims are 16 and 32; the real
// checkpoint's 256 and 512 are Gate 2's.
func TestEG2Metal_matchesCPU(t *testing.T) {
	m, err := embeddinggemma2.Load(eg2TinyDir)
	if err != nil {
		t.Fatalf("load the tiny fixture: %v", err)
	}
	acc, err := m.NewAccelerator("metal")
	if err != nil {
		t.Fatal(err)
	}
	defer acc.Close()
	raw, err := os.ReadFile(eg2TinyDir + "/golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		Cases []struct {
			IDs []int `json:"ids"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	var inputs [][]int
	for _, c := range g.Cases {
		inputs = append(inputs, c.IDs)
	}
	for _, n := range []int{300, 700} {
		ids := make([]int, n)
		for i := range ids {
			ids[i] = (i*7919 + 13) % m.Config().VocabSize
		}
		inputs = append(inputs, ids)
	}
	E := m.Dim()
	defer func(b int) { eg2Block = b }(eg2Block)
	for _, blk := range []int{256, 4, 5} {
		eg2Block = blk
		for ci, ids := range inputs {
			cpuLast, cpuLayers, err := m.ForwardCPU(ids, true)
			if err != nil {
				t.Fatal(err)
			}
			gpuLast, gpuLayers, err := acc.Forward(ids, true)
			if err != nil {
				t.Fatalf("case %d: metal forward: %v", ci, err)
			}
			if len(gpuLayers) != len(cpuLayers) {
				t.Fatalf("case %d: %d layer states, CPU %d", ci, len(gpuLayers), len(cpuLayers))
			}
			worst, worstAt := 1.0, -1
			for li := range cpuLayers {
				if c := eg2Cos(gpuLayers[li], cpuLayers[li]); c < worst {
					worst, worstAt = c, li
				}
			}
			lc := eg2Cos(gpuLast, cpuLast)
			ec := eg2Cos(eg2Pool(gpuLast, len(ids), E), eg2Pool(cpuLast, len(ids), E))
			t.Logf("block %d, case %d (%d tokens): worst layer-input cosine %.9f (index %d), last hidden %.9f, embedding %.9f", blk, ci, len(ids), worst, worstAt, lc, ec)
			if worst < 0.9999 || lc < 0.9999 || ec < 0.9999 {
				t.Errorf("block %d, case %d (%d tokens): worst layer %.9f at index %d, last hidden %.9f, embedding %.9f (bar 0.9999)", blk, ci, len(ids), worst, worstAt, lc, ec)
			}
		}
	}
}
