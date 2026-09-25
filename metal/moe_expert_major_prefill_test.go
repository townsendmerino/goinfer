//go:build darwin

package metal

import (
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

func getEmbs(r *resident, prompt []int) [][]float32 {
	embs := make([][]float32, len(prompt))
	for i, id := range prompt {
		e := make([]float32, r.H)
		r.embed.Row(id, e)
		if r.embedScale > 1 {
			for j := range e {
				e[j] *= r.embedScale
			}
		}
		embs[i] = e
	}
	return embs
}

// TestMoEExpertMajor_ParityVsRowByRow verifies that the expert-major batched prefill
// produces near-identical logits and matching argmax compared to the row-by-row fallback.
func TestMoEExpertMajor_ParityVsRowByRow(t *testing.T) {
	ckpt := "../testdata/mixtral-tiny"
	if _, err := os.Stat(ckpt + "/config.json"); err != nil {
		t.Skipf("no fixture (%s/config.json)", ckpt)
	}
	m, err := decoder.Load(ckpt, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("resident: %v", err)
	}
	if !r.prefillOK || r.moe == nil {
		t.Fatal("fixture should admit MoE batched prefill")
	}

	promptLengths := []int{8, 16, 32, 64}
	for _, M := range promptLengths {
		t.Run(fmt.Sprintf("M=%d", M), func(t *testing.T) {
			prompt := make([]int, M)
			for i := range prompt {
				prompt[i] = (i*7 + 13) % 250
			}

			// Reference: sequential Forward loop
			var seq []float32
			for i, id := range prompt {
				seq = r.Forward(id, i)
			}
			seqArg := argmaxF(seq)

			// 1. Run with expert-major prefill (GOINFER_MOE_EXPERT_MAJOR=1)
			setResidentKnob(t, r, "GOINFER_MOE_EXPERT_MAJOR", "1")
			embs1 := getEmbs(r, prompt)
			outMajor := r.PrefillLast(embs1, 0)

			// 2. Run with row-by-row prefill (GOINFER_MOE_EXPERT_MAJOR=0)
			setResidentKnob(t, r, "GOINFER_MOE_EXPERT_MAJOR", "0")
			embs2 := getEmbs(r, prompt)
			outRow := r.PrefillLast(embs2, 0)

			// Check parity
			argMajor := argmaxF(outMajor)
			argRow := argmaxF(outRow)
			cosMajorSeq := cosF(seq, outMajor)
			cosRowSeq := cosF(seq, outRow)
			cosMajorRow := cosF(outMajor, outRow)

			t.Logf("M=%d: seqArg=%d argMajor=%d argRow=%d | cos(major,seq)=%.5f cos(row,seq)=%.5f cos(major,row)=%.5f",
				M, seqArg, argMajor, argRow, cosMajorSeq, cosRowSeq, cosMajorRow)

			if cosMajorSeq < 0.95 {
				t.Errorf("cos(major,seq) %.5f < 0.95", cosMajorSeq)
			}
			if argMajor != seqArg {
				t.Errorf("argmax mismatch: major=%d seq=%d", argMajor, seqArg)
			}
		})
	}
}

// BenchmarkMoEExpertMajor_VsRowByRow benchmarks the speedup of expert-major prefill
// against row-by-row prefill across prompt lengths M in [16, 32, 64].
func BenchmarkMoEExpertMajor_VsRowByRow(b *testing.B) {
	ckpt := "../testdata/mixtral-tiny"
	if _, err := os.Stat(ckpt + "/config.json"); err != nil {
		b.Skipf("no fixture (%s/config.json)", ckpt)
	}
	m, err := decoder.Load(ckpt, decoder.Options{Quant: "int4"})
	if err != nil {
		b.Fatalf("load: %v", err)
	}
	r, err := buildResident(m)
	if err != nil {
		b.Fatalf("resident: %v", err)
	}

	lengths := []int{16, 32, 64}
	for _, M := range lengths {
		prompt := make([]int, M)
		for i := range prompt {
			prompt[i] = (i*11 + 5) % 250
		}
		embs := getEmbs(r, prompt)

		b.Run(fmt.Sprintf("RowByRow_M=%d", M), func(b *testing.B) {
			setResidentKnob(b, r, "GOINFER_MOE_EXPERT_MAJOR", "0")
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				r.PrefillLast(embs, 0)
			}
		})

		b.Run(fmt.Sprintf("ExpertMajor_M=%d", M), func(b *testing.B) {
			setResidentKnob(b, r, "GOINFER_MOE_EXPERT_MAJOR", "1")
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				r.PrefillLast(embs, 0)
			}
		})
	}
}

// TestMoEPrefillSpeedupDirect compares latency of expert-major vs row-by-row in a single test run.
func TestMoEPrefillSpeedupDirect(t *testing.T) {
	ckpt := "../testdata/mixtral-tiny"
	if _, err := os.Stat(ckpt + "/config.json"); err != nil {
		t.Skipf("no fixture (%s/config.json)", ckpt)
	}
	m, err := decoder.Load(ckpt, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("resident: %v", err)
	}

	for _, M := range []int{16, 32, 64} {
		prompt := make([]int, M)
		for i := range prompt {
			prompt[i] = (i*7 + 3) % 250
		}
		embs := getEmbs(r, prompt)

		// Warmup
		setResidentKnob(t, r, "GOINFER_MOE_EXPERT_MAJOR", "0")
		r.PrefillLast(embs, 0)
		setResidentKnob(t, r, "GOINFER_MOE_EXPERT_MAJOR", "1")
		r.PrefillLast(embs, 0)

		const iters = 10

		setResidentKnob(t, r, "GOINFER_MOE_EXPERT_MAJOR", "0")
		t0 := time.Now()
		for range iters {
			r.PrefillLast(embs, 0)
		}
		dtRow := time.Since(t0)

		setResidentKnob(t, r, "GOINFER_MOE_EXPERT_MAJOR", "1")
		t1 := time.Now()
		for range iters {
			r.PrefillLast(embs, 0)
		}
		dtMajor := time.Since(t1)

		speedup := float64(dtRow) / float64(dtMajor)
		t.Logf("M=%d: RowByRow=%s (%.2f ms/run), ExpertMajor=%s (%.2f ms/run) => %.2fx speedup",
			M, dtRow, float64(dtRow.Microseconds())/iters/1000.0,
			dtMajor, float64(dtMajor.Microseconds())/iters/1000.0, speedup)
	}
}
