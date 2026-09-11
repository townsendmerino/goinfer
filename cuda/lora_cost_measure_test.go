//go:build cuda && goinfer_testhooks && realckpt

package cuda

// Measures audit P-10 (per-request adapter upload) and P-11 (per-token LoRA launch overhead) on a
// real base at a realistic adapter size. A MEASUREMENT, not a gate: it logs paired, interleaved
// numbers (aikit measuring rule 7 — difference matched observations, report a win count).
//
//	GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks realckpt' ./cuda/ \
//	  -run TestLoRACostMeasure -v -timeout 30m

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

func loraMedian(xs []float64) float64 {
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	return s[len(s)/2]
}

// loraPaired reports the paired mean of (a[i]-b[i]) and how many pairs had a < b.
func loraPaired(a, b []float64) (meanDelta float64, aWins int) {
	for i := range a {
		meanDelta += a[i] - b[i]
		if a[i] < b[i] {
			aWins++
		}
	}
	return meanDelta / float64(len(a)), aWins
}

func TestLoRACostMeasure(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("heavy: set GOINFER_HEAVY_TESTS=1")
	}
	t.Setenv("GOINFER_NO_RESIDENT_REUSE", "1") // isolate P-10: repeated prompts must not hit C-02's prefix reuse
	base := decoder.AssetPathForTest(t, "GOINFER_QWEN3_4B")
	// qwen3-4b geometry (config.json): 36 layers, hidden 2560, 32 q heads x 128, 8 kv heads, inter 9728.
	const L, hidden, qDim, kvDim, inter, rank = 36, 2560, 4096, 1024, 9728, 16
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "adapter_config.json"), []byte(`{"r":16,"lora_alpha":32}`), 0o644); err != nil {
		t.Fatal(err)
	}
	fill := func(n, seed int) []float32 {
		d := make([]float32, n)
		for i := range d {
			d[i] = float32((i*7+seed)%13)*0.0005 - 0.003
		}
		return d
	}
	at := map[string]loraCUDAStTensor{}
	var adapterBytes int64
	for l := range L {
		pfx := "base_model.model.model.layers." + strconv.Itoa(l)
		add := func(mod string, in, out, seed int) {
			at[pfx+mod+".lora_A.weight"] = loraCUDAStTensor{[]int{rank, in}, fill(rank*in, seed)}
			at[pfx+mod+".lora_B.weight"] = loraCUDAStTensor{[]int{out, rank}, fill(out*rank, seed+1)}
			adapterBytes += int64(4 * rank * (in + out))
		}
		add(".self_attn.q_proj", hidden, qDim, 1)
		add(".self_attn.k_proj", hidden, kvDim, 2)
		add(".self_attn.v_proj", hidden, kvDim, 3)
		add(".self_attn.o_proj", qDim, hidden, 4)
		add(".mlp.gate_proj", hidden, inter, 5)
		add(".mlp.up_proj", hidden, inter, 6)
		add(".mlp.down_proj", inter, hidden, 7)
	}
	writeLoraCUDASafetensors(t, filepath.Join(dir, "adapter_model.safetensors"), at)
	at = nil
	fmt.Fprintf(os.Stderr, "[lora-measure] adapter: rank %d, 7 projections x %d layers = %d allocations, %.1f MB per upload\n",
		rank, L, 2*7*L, float64(adapterBytes)/1e6)

	t0 := time.Now()
	m, err := decoder.Load(base, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	fmt.Fprintf(os.Stderr, "[lora-measure] loaded qwen3-4b int4 CUDA-resident in %v\n", time.Since(t0).Round(time.Second))
	rf := m.ResidentForwardForTest()
	cr, ok := rf.(*cudaResident)
	if !ok {
		t.Fatalf("not CUDA-resident: %T (%s)", rf, m.ResidentDecline())
	}
	if err := m.LoadAdapter("x", dir); err != nil {
		t.Fatalf("LoadAdapter: %v", err)
	}
	layers := func() []decoder.ResidentAdapterLayer { return m.ResidentAdapterLayersForTest("x") }
	must := func(err error) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
	}
	uncached := func(on bool) {
		if on {
			os.Setenv("GOINFER_NO_LORA_CACHE", "1")
		} else {
			os.Unsetenv("GOINFER_NO_LORA_CACHE")
		}
	}
	defer uncached(false)
	warm := func() { // (re)populate the single-entry cache, untimed
		uncached(false)
		must(cr.SetAdapter(layers()))
		must(cr.SetAdapter(nil))
	}
	const pairs = 10

	// ---- P-10a: bind + clear cost per request ----
	var bc, bu []float64
	for i := range pairs {
		for _, arm := range [2]bool{i%2 == 1, i%2 == 0} { // alternate which arm leads
			if arm {
				uncached(true)
			} else {
				warm()
			}
			s := time.Now()
			must(cr.SetAdapter(layers()))
			must(cr.SetAdapter(nil))
			ms := float64(time.Since(s).Microseconds()) / 1000
			if arm {
				bu = append(bu, ms)
			} else {
				bc = append(bc, ms)
			}
		}
	}
	d, w := loraPaired(bc, bu)
	fmt.Fprintf(os.Stderr, "[lora-measure] P-10 bind+clear per request: cached median %.3f ms | uncached median %.1f ms | paired delta %.1f ms, cached faster %d/%d\n",
		loraMedian(bc), loraMedian(bu), d, w, pairs)

	// ---- P-10b: time to first token through a real Session with the adapter ----
	prompt := make([]int, 16)
	for i := range prompt {
		prompt[i] = 1000 + i*37
	}
	ttft := func() float64 {
		s := m.NewSession(0)
		must(s.UseAdapter("x"))
		start := time.Now()
		out, g := s.Generate(context.Background(), prompt, 2, decoder.SamplingParams{})
		<-out
		ms := float64(time.Since(start).Microseconds()) / 1000
		for range out {
		}
		must(g.Err())
		return ms
	}
	var tc, tu []float64
	for i := range pairs {
		for _, arm := range [2]bool{i%2 == 1, i%2 == 0} {
			if arm {
				uncached(true)
			} else {
				warm()
			}
			if arm {
				tu = append(tu, ttft())
			} else {
				tc = append(tc, ttft())
			}
		}
	}
	d, w = loraPaired(tc, tu)
	fmt.Fprintf(os.Stderr, "[lora-measure] P-10 TTFT (16-token prompt, adapter session): cached median %.1f ms | uncached median %.1f ms | paired delta %.1f ms, cached faster %d/%d\n",
		loraMedian(tc), loraMedian(tu), d, w, pairs)

	// ---- P-11: per-token Forward with the adapter bound vs unbound, same resident ----
	warm()
	const N = 48
	perTok := func(bound bool) float64 {
		if bound {
			must(cr.SetAdapter(layers()))
		} else {
			must(cr.SetAdapter(nil))
		}
		rf.Reset()
		s := time.Now()
		for p := range N {
			if _, err := rf.Forward(m.EmbedResidentForTest(1000+p), p); err != nil {
				t.Fatalf("forward: %v", err)
			}
		}
		return float64(time.Since(s).Microseconds()) / 1000 / N
	}
	var fb, fn []float64
	for i := range pairs {
		for _, bound := range [2]bool{i%2 == 0, i%2 == 1} {
			if bound {
				fb = append(fb, perTok(true))
			} else {
				fn = append(fn, perTok(false))
			}
		}
	}
	d, w = loraPaired(fb, fn)
	fmt.Fprintf(os.Stderr, "[lora-measure] P-11 per-token Forward: adapter bound median %.2f ms | unbound median %.2f ms | paired overhead %.2f ms (%.1f%% of unbound), bound faster %d/%d\n",
		loraMedian(fb), loraMedian(fn), d, 100*d/loraMedian(fn), w, pairs)
	must(cr.SetAdapter(nil))
	up, hit := cr.LoraCacheStatsForTest()
	fmt.Fprintf(os.Stderr, "[lora-measure] bind counters over the run: uploads=%d hits=%d\n", up, hit)
}
