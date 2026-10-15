package decoder

import (
	"path/filepath"
	"testing"
)

// TestCPUKVPrecision_followsTheFamily: under Options.KVQuant "i8", the CPU cache is int8 only where the family allows it
// (Architecture.kvInt8OK; not an MoE), and every place that prices or names the cache reads the cache, not the option:
// NewCache, CPUKVPrecision, the load-time fit guard (kvPricingFor) and the per-request admission (AdmitPrefillMemory). An
// MoE priced at int8 while it allocates f32 was let in at about a quarter of what it holds.
func TestCPUKVPrecision_followsTheFamily(t *testing.T) {
	for _, c := range []struct{ fixture, want string }{{"llama-tiny", "i8"}, {"mixtral-tiny", "f32"}} {
		t.Run(c.fixture, func(t *testing.T) {
			m, err := Load(filepath.Join("..", "testdata", c.fixture), Options{KVQuant: "i8"})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer m.Close()
			if got := m.CPUKVPrecision(); got != c.want {
				t.Fatalf("CPUKVPrecision = %q, want %q", got, c.want)
			}
			if got := m.NewCache(0).quant == kvI8; got != (c.want == "i8") {
				t.Fatalf("NewCache allocated int8 = %v, want %v", got, c.want == "i8")
			}
			p, ok := kvPricingFor(m.Config(), Options{KVQuant: "i8"})
			if !ok {
				t.Fatal("kvPricingFor priced nothing")
			}
			if p.kvI8 != (c.want == "i8") {
				t.Fatalf("the load-time guard prices int8 = %v; the cache is %s", p.kvI8, c.want)
			}

			// Per-request admission, with the memory left between the int8 price and the f32 price: an int8 cache fits, an
			// f32 one does not.
			cfg := m.Config()
			const prompt, gen = 4000, 8
			kvI8, kvF32 := estimateKVBytes(cfg, prompt+gen, false, true), estimateKVBytes(cfg, prompt+gen, false, false)
			if kvF32 <= kvI8 {
				t.Fatalf("the f32 price (%d) is not above the int8 price (%d): the test cannot tell them apart", kvF32, kvI8)
			}
			left := (kvI8+kvF32)/2 + prefillScratchBytes(cfg, prompt)
			injectHostRAMAvailable(t, int64(float64(left)/prefillAvailFraction)+1)
			err = m.AdmitPrefillMemory(prompt, gen, false)
			switch {
			case c.want == "i8" && err != nil:
				t.Fatalf("an int8 cache that fits was refused: %v", err)
			case c.want == "f32" && err == nil:
				t.Fatal("an f32 cache that does not fit was admitted: the request was priced at int8")
			}
		})
	}
}
