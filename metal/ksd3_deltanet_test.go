//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// ksd3Run is a 40-token batched prefill of ckpt on the Metal resident at int4, then 8 greedy decode steps: the prefill's last
// logits and every decode step's. old takes the device-memory scan kernels (dnSeqOldForTest). It returns nil, with the
// reason, when the resident's prefill is not batched, since the scan kernels then never run.
func ksd3Run(t *testing.T, ckpt string, old bool) ([][]float32, string) {
	t.Helper()
	prev := dnSeqOldForTest
	dnSeqOldForTest = old
	defer func() { dnSeqOldForTest = prev }()
	m, err := decoder.Load(ckpt, decoder.Options{Backend: "metal", Quant: "int4"})
	if err != nil {
		t.Fatalf("load %s: %v", ckpt, err)
	}
	defer m.Close()
	decoder.SetKnobEnvForTest(t, m, "GOINFER_METAL_FAST_PREFILL_FLOOR", "0")
	rf := m.ResidentForwardForTest()
	if rf == nil {
		t.Fatalf("%s did not go resident: %s", ckpt, m.ResidentDecline())
	}
	if batched, why := m.PrefillPath(); !batched {
		return nil, why
	}
	pf, ok := rf.(decoder.Prefiller)
	if !ok {
		t.Fatalf("%s: the resident has no batched prefill", ckpt)
	}
	_, _, _, _, _, _, vocab := m.Dims()
	const n = 40
	embs := make([][]float32, n)
	for i := range embs {
		embs[i] = m.EmbedResidentForTest((i*37 + 11) % vocab)
	}
	rf.Reset()
	lg, err := pf.PrefillLast(context.Background(), embs, 0)
	if err != nil {
		t.Fatalf("%s: prefill: %v", ckpt, err)
	}
	out := [][]float32{append([]float32(nil), lg...)}
	for i := range 8 {
		best := 0
		for j := range lg {
			if lg[j] > lg[best] {
				best = j
			}
		}
		if lg, err = rf.Forward(m.EmbedResidentForTest(best), n+i); err != nil {
			t.Fatalf("%s: decode %d: %v", ckpt, i, err)
		}
		out = append(out, append([]float32(nil), lg...))
	}
	return out, ""
}

// TestKSD3_deltaNetModelsBitIdentical is D3-G2 (docs/tasks/task-kernel-survey-2026-10.md, "KS-D3 gates"): on every
// Gated-DeltaNet fixture, the resident's batched prefill and 8 decode steps after give the same logits, bit for bit, with the
// on-chip scan kernels as with the device-memory ones. A fixture whose Metal prefill is not batched never runs the scan kernels
// and is named, not compared (TestDeltaNetSeqKernels_matchDecodeBitwise covers its geometry).
func TestKSD3_deltaNetModelsBitIdentical(t *testing.T) {
	ran := 0
	for _, ckpt := range []string{"../testdata/qwen35-tiny", "../decoder/testdata/qwen3_5-tiny", "../decoder/testdata/qwen3_5_moe-tiny", "../testdata/olmo_hybrid-tiny"} {
		if _, err := os.Stat(ckpt + "/model.safetensors"); err != nil {
			t.Logf("%s: no fixture", ckpt)
			continue
		}
		want, why := ksd3Run(t, ckpt, true)
		if want == nil {
			fmt.Fprintf(os.Stderr, "[D3-G2] %s: not compared, its Metal prefill is not batched (%s)\n", ckpt, why)
			continue
		}
		got, _ := ksd3Run(t, ckpt, false)
		d := 0
		for s := range want {
			for j := range want[s] {
				if math.Float32bits(want[s][j]) != math.Float32bits(got[s][j]) {
					d++
				}
			}
		}
		fmt.Fprintf(os.Stderr, "[D3-G2] %s: %d of %d logits differ (prefill and 8 decode steps)\n", ckpt, d, len(want)*len(want[0]))
		if d != 0 {
			t.Errorf("%s: %d logits differ between the scan forms", ckpt, d)
		}
		ran++
	}
	if ran == 0 {
		t.Fatal("no fixture ran")
	}
}
