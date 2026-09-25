package serveapp

import "testing"

// --kv is the one KV-precision flag for whichever backend serves: it sets the GPU residency cache
// (Options.KVPrecision) and the CPU cache (Options.KVQuant), which has no f16 form. --kv-quant is a
// deprecated alias that, when given, overrides the CPU cache alone. Same for the per-model keys.
func TestKVFlag_drivesBothCachesUnlessKVQuantOverrides(t *testing.T) {
	str := func(s string) *string { return &s }
	for _, c := range []struct {
		name          string
		spec          modelSpec
		cfg           config
		wantGPU, want string
	}{
		{"default", modelSpec{}, config{kvPrec: "f32"}, "f32", "f32"},
		{"--kv i8 reaches the CPU cache too", modelSpec{}, config{kvPrec: "i8"}, "i8", "i8"},
		{"--kv f16: the CPU cache has no f16", modelSpec{}, config{kvPrec: "f16"}, "f16", "f32"},
		{"deprecated --kv-quant still overrides the CPU cache", modelSpec{}, config{kvPrec: "f16", kvQuant: "i8"}, "f16", "i8"},
		{"--kv-quant f32 beats --kv i8 on CPU", modelSpec{}, config{kvPrec: "i8", kvQuant: "f32"}, "i8", "f32"},
		{"per-model kv= drives both", modelSpec{kvPrec: str("i8")}, config{kvPrec: "f32"}, "i8", "i8"},
		{"per-model kv-quant= overrides", modelSpec{kvPrec: str("i8"), kvQuant: str("f32")}, config{kvPrec: "f32"}, "i8", "f32"},
	} {
		o := c.spec.options(c.cfg)
		if o.KVPrecision != c.wantGPU || o.KVQuant != c.want {
			t.Errorf("%s: KVPrecision=%q KVQuant=%q, want %q / %q", c.name, o.KVPrecision, o.KVQuant, c.wantGPU, c.want)
		}
		if err := o.Validate(); err != nil {
			t.Errorf("%s: resolved options do not validate: %v", c.name, err)
		}
	}
}
