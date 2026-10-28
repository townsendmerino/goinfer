package decoder

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/internal/cost"
)

const costFixture = "../testdata/llama-tiny"

// loadCostFixture is llama-tiny at f32, whose weights' bytes are exact by shape. It is tracked, so it must load.
func loadCostFixture(t *testing.T) *Model {
	t.Helper()
	m, err := Load(costFixture, Options{Quant: "f32"})
	if err != nil {
		t.Fatalf("load %s: %v", costFixture, err)
	}
	t.Cleanup(func() { _ = m.Close() })
	return m
}

// decodeCounted runs warm tokens uncounted, then one counted decode token at position len(warm), and returns its logits and report.
func decodeCounted(t *testing.T, m *Model, warm []int, id int, counter *cost.Counter) ([]float32, cost.Report) {
	t.Helper()
	cache := m.NewCache(64)
	for _, w := range warm {
		if _, err := m.forward(w, cache); err != nil {
			t.Fatal(err)
		}
	}
	cache.cost, cache.costPhase = counter, cost.Decode
	lg, err := m.forward(id, cache)
	if err != nil {
		t.Fatal(err)
	}
	return slices.Clone(lg), counter.Report()
}

// layerMatrixBytes is every per-layer matrix's bytes, from the dense enumeration less the model-level tables: an independent sum.
func layerMatrixBytes(m *Model) int64 {
	w := m.w
	return m.ResidentDenseWeightBytes() - wmBytes(&w.Embed) - wmBytes(&w.LMHead) - wmBytes(&w.PosEmbed) -
		wmBytes(&w.PerLayerTokenEmbed) - wmBytes(&w.PerLayerModelProj)
}

// costDecodeErrors grades one decode token's report at position pos (G-SB11-2): calls per kind against the program, matrix bytes
// against the dense enumeration, attention's rows, and the CPU's device counts.
func costDecodeErrors(m *Model, r cost.Report, pos int) []string {
	var errs []string
	d := r.Decode
	if d == nil || r.Prefill != nil {
		return []string{fmt.Sprintf("want a decode phase alone, got %+v", r)}
	}
	if d.Tokens != 1 {
		errs = append(errs, fmt.Sprintf("decode tokens %d, want 1", d.Tokens))
	}
	arch, prog := m.w.arch, m.program()
	want := map[string]int64{costEmbed: 1, costFinalNorm: 1, costLMHead: 1}
	for _, ops := range prog.layers {
		for _, o := range ops {
			want[opCostNames[o.kind]]++
		}
	}
	for name, n := range want {
		if got := d.Ops[name].Calls; got != n {
			errs = append(errs, fmt.Sprintf("%s: %d calls, the program has %d", name, got, n))
		}
	}
	for name := range d.Ops {
		if want[name] == 0 {
			errs = append(errs, fmt.Sprintf("%s counted, but the program has none", name))
		}
	}
	H, hd, kvDim := int64(arch.HiddenDim), int64(arch.HeadDim), int64(arch.NumKVHeads*arch.HeadDim)
	nH := int64(arch.NumHeads)
	act := map[string]int64{"proj_qkv": 4 * H, "proj_o": 4 * nH * hd, "ffn": 4 * H, costLMHead: 4 * H, costEmbed: 0}
	var gotW int64
	for name, a := range act {
		gotW += d.Ops[name].BytesRead - d.Ops[name].Calls*a
	}
	embedRow := wmBytes(&m.w.Embed) / int64(m.w.Embed.Rows())
	head := wmBytes(&m.w.LMHead)
	if prog.head.Tied {
		head = wmBytes(&m.w.Embed)
	}
	if wantW := layerMatrixBytes(m) + head + embedRow; gotW != wantW {
		errs = append(errs, fmt.Sprintf("matrix bytes read %d, the loaded weights hold %d (layers %d + head %d + one embedding row %d)",
			gotW, wantW, layerMatrixBytes(m), head, embedRow))
	}
	at := d.Ops["attention"]
	n := int64(len(prog.layers))
	if wantR := n * (4*nH*hd + 2*int64(pos+1)*4*kvDim); at.BytesRead != wantR {
		errs = append(errs, fmt.Sprintf("attention read %d bytes, want %d (%d layers × q + %d rows of K and V)", at.BytesRead, wantR, n, pos+1))
	}
	if wantW := n * (4*nH*hd + 2*4*kvDim); at.BytesWritten != wantW {
		errs = append(errs, fmt.Sprintf("attention wrote %d bytes, want %d", at.BytesWritten, wantW))
	}
	if d.Totals.Launches != 0 || d.Totals.Syncs != 0 || !slices.Contains(d.Totals.Unmeasured, cost.FieldAllocs) {
		errs = append(errs, fmt.Sprintf("device counts %+v: want 0 launches and syncs, allocations unmeasured", d.Totals))
	}
	return errs
}

// TestCost_cpuDecodeExact (G-SB11-2) and the off/on identity (G-SB11-5): one decode token on llama-tiny, counted exactly; its
// logits equal the uncounted token's bit for bit.
func TestCost_cpuDecodeExact(t *testing.T) {
	m := loadCostFixture(t)
	warm := []int{1, 2, 3}
	on, r := decodeCounted(t, m, warm, 5, cost.New())
	for _, e := range costDecodeErrors(m, r, len(warm)) {
		t.Error(e)
	}
	off, _ := decodeCounted(t, m, warm, 5, nil)
	for i := range on {
		if math.Float32bits(on[i]) != math.Float32bits(off[i]) {
			t.Fatalf("logit %d differs with the counters on: %v vs %v", i, on[i], off[i])
		}
	}
	t.Logf("decode totals %+v", r.Decode.Totals)
}

// TestCost_cpuDecodePlants (G-SB11-4): an extra norm in the program moves norm's calls by exactly one, and a second count of the
// output projection breaks the matrix-byte equality; the grader reports each.
func TestCost_cpuDecodePlants(t *testing.T) {
	m := loadCostFixture(t)
	warm := []int{1, 2, 3}
	base := *m.program()
	planted := base
	planted.layers = slices.Clone(base.layers)
	l0 := slices.Clone(base.layers[0])
	ni := slices.IndexFunc(l0, func(o op) bool { return o.kind == opNorm })
	planted.layers[0] = slices.Insert(l0, ni, l0[ni])
	m.prog = &planted
	_, r := decodeCounted(t, m, warm, 5, cost.New())
	m.prog = nil
	errs := costDecodeErrors(m, r, len(warm))
	if !slices.ContainsFunc(errs, func(e string) bool { return strings.HasPrefix(e, "norm: ") }) {
		t.Errorf("a planted extra norm was not seen: %v", errs)
	}
	if got, want := r.Decode.Ops["norm"].Calls, int64(0); got == want {
		t.Fatal("no norm counted")
	}
	seamPlantForTest = plantCostDoubleCount
	_, r = decodeCounted(t, m, warm, 5, cost.New())
	seamPlantForTest = plantNone
	errs = costDecodeErrors(m, r, len(warm))
	if !slices.ContainsFunc(errs, func(e string) bool { return strings.HasPrefix(e, "matrix bytes read") }) {
		t.Errorf("a planted second count of proj_o was not seen: %v", errs)
	}
}

// TestCost_cpuPrefillExact (G-SB11-3 and G-SB11-5): an N-token prefill through the batched walker counts one call per operation
// per pass, each layer's matrices once, and attention's causal triangle; its rows equal the uncounted prefill's bit for bit.
func TestCost_cpuPrefillExact(t *testing.T) {
	m := loadCostFixture(t)
	ids := []int{1, 2, 3, 4, 5, 6, 7}
	run := func(c *cost.Counter) [][]float32 {
		cache := m.NewCache(64)
		cache.cost, cache.costPhase = c, cost.Prefill
		rows, err := m.forwardN(context.Background(), ids, cache)
		if err != nil {
			t.Fatal(err)
		}
		out := make([][]float32, len(rows))
		for i := range rows {
			out[i] = slices.Clone(rows[i])
		}
		return out
	}
	c := cost.New()
	on, off := run(c), run(nil)
	for i := range on {
		for j := range on[i] {
			if math.Float32bits(on[i][j]) != math.Float32bits(off[i][j]) {
				t.Fatalf("row %d logit %d differs with the counters on", i, j)
			}
		}
	}
	p := c.Report().Prefill
	if p == nil || p.Tokens != int64(len(ids)) {
		t.Fatalf("prefill phase %+v, want %d tokens", p, len(ids))
	}
	arch, prog := m.w.arch, m.program()
	want := map[string]int64{}
	for _, ops := range prog.layers {
		for _, o := range ops {
			want[opCostNames[o.kind]]++
		}
	}
	for name, n := range want {
		if got := p.Ops[name].Calls; got != n {
			t.Errorf("%s: %d calls, want %d (one per operation per pass)", name, got, n)
		}
	}
	K := int64(len(ids))
	H, hd, kvDim, nH := int64(arch.HiddenDim), int64(arch.HeadDim), int64(arch.NumKVHeads*arch.HeadDim), int64(arch.NumHeads)
	act := map[string]int64{"proj_qkv": 4 * H, "proj_o": 4 * nH * hd, "ffn": 4 * H}
	var gotW int64
	for name, a := range act {
		gotW += p.Ops[name].BytesRead - K*p.Ops[name].Calls*a
	}
	if wantW := layerMatrixBytes(m); gotW != wantW {
		t.Errorf("matrix bytes read %d, want each layer's matrices once: %d", gotW, wantW)
	}
	n := int64(len(prog.layers))
	causal := K * (K + 1) / 2
	if got, wantR := p.Ops["attention"].BytesRead, n*(K*4*nH*hd+2*causal*4*kvDim); got != wantR {
		t.Errorf("attention read %d bytes, want %d (the causal triangle, %d rows)", got, wantR, causal)
	}
}

// costOffHook is the off path one operation pays in the walkers: costStart's nil check, then the record's.
//
//go:noinline
func costOffHook(cache *KVCache) bool {
	t0 := costStart(cache)
	return cache.cost != nil && !t0.IsZero()
}

// TestCost_offShare (G-SB11-6): the off hook's cost per operation × the operations per decode token, as a share of a measured
// decode token on Qwen2.5-Coder-0.5B on the CPU, is under 0.5%. Heavy (a real checkpoint from ~/models); run it under the
// timing lock and note the box's power state and load in the record.
func TestCost_offShare(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy: set GOINFER_HEAVY_TESTS=1 (a 0.5B decode on the CPU)")
	}
	home, _ := os.UserHomeDir()
	path := filepath.Join(home, "models", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no %s", path)
	}
	m, err := Load(path, Options{Backend: "cpu", Quant: "int4"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ops := 3 // the embed, the final norm and the head
	for _, l := range m.program().layers {
		ops += len(l)
	}
	cache := m.NewCache(256)
	for i := range 8 { // warm the caches and the scratch
		if _, err := m.forward(100+i, cache); err != nil {
			t.Fatal(err)
		}
	}
	const tokens = 64
	start := time.Now()
	for i := range tokens {
		if _, err := m.forward(200+i, cache); err != nil {
			t.Fatal(err)
		}
	}
	perToken := time.Since(start) / tokens
	hook := testing.Benchmark(func(b *testing.B) {
		c := &KVCache{}
		n := 0
		for b.Loop() {
			if costOffHook(c) {
				n++
			}
		}
		_ = n
	})
	perHook := time.Duration(hook.NsPerOp())
	share := float64(perHook) * float64(ops) / float64(perToken)
	t.Logf("decode %v a token, %d operations a token, the off hook %v an operation: %.4f%% of a token", perToken, ops, perHook, 100*share)
	if share >= 0.005 {
		t.Errorf("the off hook costs %.3f%% of a decode token, at or over the registered 0.5%%", 100*share)
	}
}
