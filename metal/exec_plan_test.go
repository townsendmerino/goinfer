//go:build darwin

package metal

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestExecutorAttnPlan pins the pipelined executor's attention plan to each job's OWN key count.
//
// execLoop encodes token t+1's command buffer while token t runs on the GPU. What that encode bakes in
// includes the attention plan: whether a layer dispatches attention_fa (canUseAttnFA's depth gate at
// attnFADepthFloor) and attention_fa's split grid (attnFASplitFor). Those decisions used to read the
// resident's key count at encode time — the PREVIOUS job's — so (found 2026-09-25, R17:
// docs/measurements/metal-decode-attn-r17-2026-09-25.md):
//   - in steady decode the plan lagged one token (the first step at 1536 keys ran the shipped kernel);
//   - the first decode step of a new request ran the plan of wherever the previous request stopped —
//     attention_fa below its floor after a long request, with a grid sized for the old depth and a split
//     uniform sized for the new one; the shipped kernel after a short request;
//   - after ForwardBatch (which zeroes the depth reading) the next step declined attention_fa at any depth.
//
// The oracle is the synchronous path (ForwardEmb / PrefillLast), which always sets the position before it
// encodes. Every sequence below runs through the production adapter (metalResident.Forward -> the executor)
// and must be BIT-IDENTICAL to the same sequence run synchronously, with nothing flushed in between — the
// executor's state carries over across "requests" exactly as it does in a server.
//
// Fixture: testdata/llama-attnfa-tiny (dense GQA, head dim 128 — the one committed fixture attention_fa
// engages on; see snapshot_golden_test.go). A few seconds; no heavy assets.
func TestExecutorAttnPlan(t *testing.T) {
	dir := "../testdata/llama-attnfa-tiny"
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
		t.Skipf("fixture not present: %v", err)
	}
	m, err := decoder.Load(dir, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("resident: %v", err)
	}
	defer r.Close()
	a := &metalResident{r: r, hidden: r.H}

	// Non-vacuity: attention_fa must be on by default, engage at 1536 keys and decline at 1535 on this
	// fixture, and change the numerics there — otherwise a lagging plan would be invisible to this test.
	if !r.decodeAttnFA {
		t.Fatalf("attention_fa is off by default — this test cannot see the attention plan")
	}
	r.setPos(attnFADepthFloor - 1)
	engaged := r.canUseAttnFA(0)
	r.setPos(attnFADepthFloor - 2)
	declined := !r.canUseAttnFA(0)
	if !engaged || !declined {
		t.Fatalf("fixture no longer straddles the floor: engages at %d keys=%v, declines at %d=%v",
			attnFADepthFloor, engaged, attnFADepthFloor-1, declined)
	}

	copyRow := func(v []float32) []float32 { return append([]float32(nil), v...) }
	ids := []int{1, 7, 42, 100, 5, 200, 13, 88, 3, 71, 9, 17, 60, 200, 33, 2}
	emb := func(pos int) []float32 {
		e := make([]float32, r.H)
		prodEmbedRow(r, ids[pos%len(ids)], e)
		return e
	}
	embs := func(from, to int) [][]float32 {
		var out [][]float32
		for p := from; p < to; p++ {
			out = append(out, emb(p))
		}
		return out
	}
	// A "step" list describes one request: an optional batched prefill of [0, prefill), then decode steps.
	type request struct {
		prefill   int // 0 = no PrefillLast; the request decodes from pos 0
		decodeTo  int // decode positions [prefill, decodeTo)
		recordMin int // record logits from this position on
		batched   int // >0: before decoding, run this many positions through ForwardBatch (no executor)
	}
	run := func(q request, pipe bool) [][]float32 {
		var out [][]float32
		start := 0
		if q.prefill > 0 {
			var lg []float32
			if pipe {
				var err error
				lg, err = a.PrefillLast(context.Background(), embs(0, q.prefill), 0)
				if err != nil {
					t.Skipf("batched prefill not available on this fixture: %v", err)
				}
			} else {
				lg = r.PrefillLast(embs(0, q.prefill), 0)
			}
			if q.prefill-1 >= q.recordMin {
				out = append(out, copyRow(lg))
			}
			start = q.prefill
		}
		if q.batched > 0 {
			lgs, err := r.ForwardBatch(embs(start, start+q.batched), start)
			if err != nil {
				t.Fatalf("ForwardBatch: %v", err)
			}
			for i, lg := range lgs {
				if start+i >= q.recordMin {
					out = append(out, copyRow(lg))
				}
			}
			start += q.batched
		}
		for p := start; p < q.decodeTo; p++ {
			var lg []float32
			if pipe {
				var err error
				if lg, err = a.Forward(emb(p), p); err != nil {
					t.Fatalf("Forward pos %d: %v", p, err)
				}
			} else {
				lg = r.ForwardEmb(emb(p), p)
			}
			if p >= q.recordMin {
				out = append(out, copyRow(lg))
			}
		}
		if err := r.takeExecErr(); err != nil {
			t.Fatalf("exec: %v", err)
		}
		return out
	}
	same := func(t *testing.T, what string, got, want [][]float32, firstPos int) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s: %d rows vs %d", what, len(got), len(want))
		}
		for i := range want {
			for j := range want[i] {
				if math.Float32bits(got[i][j]) != math.Float32bits(want[i][j]) {
					t.Errorf("%s: first differing row is position %d (of %d..%d) — the executor ran a different attention plan than the synchronous path", what, firstPos+i, firstPos, firstPos+len(want)-1)
					return
				}
			}
		}
	}

	floor := attnFADepthFloor
	long := request{decodeTo: floor + 64, recordMin: floor - 8} // crosses the floor by sequential decode
	longPrefilled := request{prefill: floor + 64, decodeTo: floor + 80, recordMin: floor + 63}
	// Short requests are prefilled, not decoded from pos 0: a stale plan at pos 0 runs over ONE key, where
	// softmax is 1.0 and every kernel agrees bit-for-bit — such a case cannot see the defect. On this fixture
	// attention_fa also happened to match the shipped kernel exactly at 101 and at 1001 keys (measured: the
	// stale plan ran and the output coincided), so those sizes are not used. 400 keys (a stale plan there also
	// has a split grid, 14, wider than the split uniform, 12), 700 and 1400 (attention_fa below its floor,
	// grid and uniform agreeing) each failed on the unfixed executor at their first decode step.
	short400 := request{prefill: 400, decodeTo: 416, recordMin: 399}
	short700 := request{prefill: 700, decodeTo: 716, recordMin: 699}
	short1400 := request{prefill: 1400, decodeTo: 1416, recordMin: 1399}
	batchThenDecode := request{prefill: floor + 16, batched: 4, decodeTo: floor + 32, recordMin: floor + 15}

	syncOf := func(q request) [][]float32 { return run(q, false) }
	want := map[string][][]float32{
		"long": syncOf(long), "longPrefilled": syncOf(longPrefilled), "short400": syncOf(short400),
		"short700": syncOf(short700), "short1400": syncOf(short1400), "batchThenDecode": syncOf(batchThenDecode),
	}
	// Non-vacuity, numerically: at the first engaging position the shipped kernel gives different logits.
	{
		r.decodeAttnFA = false
		off := syncOf(long)
		r.decodeAttnFA = true
		if engagedRow := 8; func() bool {
			for j := range off[engagedRow] {
				if math.Float32bits(off[engagedRow][j]) != math.Float32bits(want["long"][engagedRow][j]) {
					return false
				}
			}
			return true
		}() {
			t.Fatalf("attention_fa and the shipped kernel agree bit-for-bit at %d keys on this fixture — a lagging plan would be invisible", floor)
		}
	}

	// One executor for the whole sequence of requests, never flushed — as in a server. Each request's
	// predecessor leaves the executor holding a buffer encoded for a different attention plan.
	r.stopExec()
	t.Run("floor crossing in steady decode", func(t *testing.T) {
		same(t, "long", run(long, true), want["long"], long.recordMin)
	})
	t.Run("400-key request after a long one", func(t *testing.T) {
		same(t, "short400 after long", run(short400, true), want["short400"], short400.recordMin)
	})
	t.Run("long prefilled request after a short one", func(t *testing.T) {
		same(t, "longPrefilled after short400", run(longPrefilled, true), want["longPrefilled"], longPrefilled.recordMin)
	})
	t.Run("700-key request after a long one", func(t *testing.T) {
		same(t, "short700 after longPrefilled", run(short700, true), want["short700"], short700.recordMin)
	})
	t.Run("long prefilled request again", func(t *testing.T) {
		same(t, "longPrefilled after short700", run(longPrefilled, true), want["longPrefilled"], longPrefilled.recordMin)
	})
	t.Run("1400-key request after a long one", func(t *testing.T) {
		same(t, "short1400 after longPrefilled", run(short1400, true), want["short1400"], short1400.recordMin)
	})
	t.Run("decode after ForwardBatch at depth", func(t *testing.T) {
		same(t, "batchThenDecode", run(batchThenDecode, true), want["batchThenDecode"], batchThenDecode.recordMin)
	})
}
