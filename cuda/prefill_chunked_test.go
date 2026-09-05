//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestPrefillChunked_bitIdentical pins the property chunking rests on: splitting a prompt into
// several batched passes over the positional KV computes the SAME numbers as one pass of the whole
// prompt. Pass k's rows attend the keys passes 0…k-1 committed, at the same absolute positions, with
// the same per-row kernels — so the last row's logits must be bit-identical, not merely close.
//
// It is checked through PrefillLast at two chunk widths (one that divides the prompt evenly and one
// that leaves a short final pass) against the unchunked reference, and then eight greedy decode steps
// are compared as well: identical logits with a divergent continuation would mean the chunked run
// left the KV in a different state than the reference, which the seed logits alone cannot see.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' -run TestPrefillChunked -v ./cuda/
func TestPrefillChunked_bitIdentical(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a 1.5B model)")
	}
	path := modelPath("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	if _, err := gc.GetDevice(0); err != nil {
		t.Skipf("no device: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cuda): %v", err)
	}
	defer mc.Close()
	rf, ok := mc.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatal("resident is not *cudaResident")
	}
	if batched, why := rf.PrefillPath(); !batched {
		t.Skipf("batched prefill declined for this fixture: %s", why)
	}

	const M = 1024
	embs := make([][]float32, M)
	var s uint32 = 987654321
	for i := range embs {
		row := make([]float32, rf.hidden)
		for j := range row {
			s = s*1664525 + 1013904223
			row[j] = float32(int32(s>>16)%2001-1000) / 10000
		}
		embs[i] = row
	}

	// run prefills the prompt at the given chunk width and returns the seed logits plus the ids of
	// eight greedy decode steps taken from the resulting cache state.
	run := func(chunk string) ([]float32, []int) {
		t.Helper()
		t.Setenv("GOINFER_PREFILL_CHUNK", chunk)
		rf.Reset()
		start := time.Now()
		lg, e := rf.PrefillLast(context.Background(), embs, 0)
		if e != nil {
			t.Fatalf("PrefillLast(chunk=%s): %v", chunk, e)
		}
		fmt.Fprintf(os.Stderr, "[chunked] chunk=%-6s M=%d prefill %s\n", chunk, M, time.Since(start).Round(time.Millisecond))
		ids := make([]int, 0, 8)
		cur := lg
		for k := 0; k < 8; k++ {
			best := 0
			for i, v := range cur {
				if v > cur[best] {
					best = i
				}
			}
			ids = append(ids, best)
			next, e := rf.Forward(mc.EmbedResidentForTest(best), M+k)
			if e != nil {
				t.Fatalf("decode step %d (chunk=%s): %v", k, chunk, e)
			}
			cur = next
		}
		return lg, ids
	}

	// The reference: a width past M, so prefillChunked takes the single-pass branch.
	refLogits, refIDs := run("4096")
	for _, chunk := range []string{"256", "300"} {
		gotLogits, gotIDs := run(chunk)
		if len(gotLogits) != len(refLogits) {
			t.Fatalf("chunk=%s: logits len %d, want %d", chunk, len(gotLogits), len(refLogits))
		}
		diff := 0
		first := -1
		for i := range refLogits {
			if gotLogits[i] != refLogits[i] {
				diff++
				if first < 0 {
					first = i
				}
			}
		}
		// Print the width compared, not just the verdict: "0 differ" over an empty or truncated
		// logit row is the shape a vacuous gate takes, and it reads identically to a real pass.
		fmt.Fprintf(os.Stderr, "[chunked] chunk=%-6s %d/%d seed logits differ from the single-pass reference\n",
			chunk, diff, len(refLogits))
		if diff != 0 {
			t.Errorf("chunk=%s: %d/%d seed logits differ from the single-pass reference (first at %d: %v vs %v) — "+
				"chunked prefill must be bit-identical, not merely close",
				chunk, diff, len(refLogits), first, gotLogits[first], refLogits[first])
		}
		for i := range refIDs {
			if gotIDs[i] != refIDs[i] {
				t.Errorf("chunk=%s: greedy continuation diverges at step %d (%v vs %v) — the chunked run left "+
					"a different KV state even though the seed logits matched", chunk, i, gotIDs, refIDs)
				break
			}
		}
	}
}
