//go:build goinfer_testhooks

package decoder

import (
	"context"
	"fmt"
	"os"
	"sort"
	"strconv"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/internal/cost"
)

// TestCPUAttnShareProbe is the measure-first gate for the fused f16 CPU attention read (docs/measurements/cpu-kv-f16-2026-11-23/README.md
// § "Profile"): it prefills the bench model to GOINFER_BENCH_DEPTH by the batched path BenchmarkDecodeAtDepth uses, then decodes
// CPU_ATTN_PROBE_TOKENS greedy tokens with the CPU cost counters on and prints each operation's share of the decode wall time. The
// "attention" operation is the per-token attention call: the q·k and p·v loops over the cache, softmax, and on a widened cache the
// widen to scratch. A measurement tool: it skips unless CPU_ATTN_PROBE=1.
//
//	CPU_ATTN_PROBE=1 GOINFER_PREQUANT_GGUF=<gguf> GOINFER_BENCH_QUANT=q4k GOINFER_BENCH_BACKEND=cpu GOINFER_BENCH_DEPTH=3900 \
//	  go test -tags goinfer_testhooks -run '^TestCPUAttnShareProbe$' -v ./decoder/
func TestCPUAttnShareProbe(t *testing.T) {
	if os.Getenv("CPU_ATTN_PROBE") != "1" {
		t.Skip("set CPU_ATTN_PROBE=1 and the BenchmarkDecodeAtDepth environment")
	}
	m, err := loadBenchModel()
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	depth, _ := strconv.Atoi(os.Getenv("GOINFER_BENCH_DEPTH"))
	n, _ := strconv.Atoi(os.Getenv("CPU_ATTN_PROBE_TOKENS"))
	if depth < 1 {
		depth = 3900
	}
	if n < 1 {
		n = 32
	}
	if !m.canBatchN(depth) {
		t.Fatalf("canBatchN(%d) is false: the depth setup would not be the benchmark's", depth)
	}
	tok := 785
	ids := make([]int, depth)
	for i := range ids {
		ids[i] = tok
	}
	cache := m.NewCache(depth + n + 8)
	t0 := time.Now()
	if _, err := m.forwardLayersN(context.Background(), ids, cache, false); err != nil {
		t.Fatalf("prefill: %v", err)
	}
	fmt.Fprintf(os.Stderr, "[attn-share] prefill to %d took %s; KV quant %v\n", depth, time.Since(t0).Round(time.Millisecond), cache.quant)
	sampler := NewSampler(SamplingParams{Temperature: 0})
	logits, err := m.forward(tok, cache)
	if err != nil {
		t.Fatal(err)
	}
	next, _ := sampler.Sample(logits)
	c := cost.New()
	cache.cost, cache.costPhase = c, cost.Decode
	t1 := time.Now()
	for range n {
		logits, err := m.forward(next, cache)
		if err != nil {
			t.Fatal(err)
		}
		next, _ = sampler.Sample(logits)
	}
	wall := time.Since(t1)
	cache.cost = nil
	r := c.Report()
	if r.Decode == nil {
		t.Fatal("VACUOUS: the decode phase counted nothing")
	}
	att, ok := r.Decode.Ops["attention"]
	if !ok || att.Calls == 0 || att.TimeNs == 0 {
		t.Fatalf("VACUOUS: no timed attention operation in the decode report (ops %v)", r.Decode.Ops)
	}
	type row struct {
		name string
		ns   int64
	}
	var rows []row
	var sum int64
	for k, v := range r.Decode.Ops {
		rows = append(rows, row{k, v.TimeNs})
		sum += v.TimeNs
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].ns > rows[j].ns })
	for _, x := range rows {
		fmt.Fprintf(os.Stderr, "[attn-share] %-12s %8.2f ms/token  %5.1f%% of wall\n", x.name, float64(x.ns)/1e6/float64(n), 100*float64(x.ns)/float64(wall.Nanoseconds()))
	}
	share := float64(att.TimeNs) / float64(wall.Nanoseconds())
	fmt.Fprintf(os.Stderr, "=== ATTN SHARE depth=%d tokens=%d: wall %.2f ms/token (%.2f tok/s), ops summed %.1f%% of wall, attention %.2f ms/token = %.1f%% of decode wall; attention bytes read %.0f MB/token ===\n",
		depth, n, float64(wall.Nanoseconds())/1e6/float64(n), float64(n)/wall.Seconds(), 100*float64(sum)/float64(wall.Nanoseconds()),
		float64(att.TimeNs)/1e6/float64(n), 100*share, float64(att.BytesRead)/1e6/float64(n))
}
