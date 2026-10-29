package serveapp

import (
	"encoding/json"
	"flag"
	"os"
	"reflect"
	"testing"

	"github.com/townsendmerino/goinfer/internal/loadflags"
)

var updateOptionsGolden = flag.Bool("update-options-golden", false, "rewrite testdata/options.golden.json")

type optionsCase struct {
	name  string
	args  []string      // the shared loading flags
	spec  string        // one --model value
	serve func(*config) // serve's own load settings
}

func optionsCases() []optionsCase {
	cpu := []string{"-backend", "cpu"}
	cs := []optionsCase{
		{"bare", cpu, "m=/p", nil},
		{"head", cpu, "m=/p,head=/h", nil},
		{"head quant=", cpu, "m=/p,head=/h,quant=int8", nil},
		{"head global --quant", append([]string{"-quant", "f32"}, cpu...), "m=/p,head=/h", nil},
		{"global --kv i8", append([]string{"-kv", "i8"}, cpu...), "m=/p", nil},
		{"global --kv i8, kv=f16", append([]string{"-kv", "i8"}, cpu...), "m=/p,kv=f16", nil},
		{"global --kv-quant", cpu, "m=/p", func(c *config) { c.kvQuant = "f16" }},
		{"global --kv-quant, kv-quant=i8", cpu, "m=/p,kv-quant=i8", func(c *config) { c.kvQuant = "f16" }},
		{"global --weight-cache", append([]string{"-weight-cache", "4"}, cpu...), "m=/p", nil},
		{"global --ctx, ctx=", append([]string{"-ctx", "8192"}, cpu...), "m=/p,ctx=1024", nil},
		{"kv-sessions", cpu, "m=/p", func(c *config) { c.kvSessions, c.kvSessionsSet = 8, true }},
		{"kv-sessions default", cpu, "m=/p", func(c *config) { c.kvSessions = 4 }},
		{"prefill-chunk", cpu, "m=/p", func(c *config) { c.prefillChunk = 512 }},
		{"kv-share off", cpu, "m=/p", func(c *config) { c.kvShareOff = true }},
		{"kv-host-bytes", cpu, "m=/p", func(c *config) { c.kvHostBytes = 1 << 30 }},
		{"cpu-batch on", cpu, "m=/p", func(c *config) { c.cpuBatch = 1 /* decoder.CPUBatchOn */ }},
		{"metal", []string{"-backend", "metal"}, "m=/p", nil},
		{"metal embed-int4", []string{"-backend", "metal"}, "m=/p,embed-int4", nil},
	}
	for _, kv := range []string{"quant=int8", "lora=/l", "kv=f16", "kv-quant=i8", "stream", "stream=false", "weight-cache=3",
		"embed-int4=false", "embed-int4", "ctx=2048"} {
		cs = append(cs, optionsCase{kv, cpu, "m=/p," + kv, nil})
	}
	return cs
}

// specOptionsJSON is o (a decoder.Options) without LoadAbort, with --moe-pager's registered default, which differs by
// OS, written as such (no case sets it). The file does not import decoder: serve's tests move off it (SB7).
func specOptionsJSON(t *testing.T, o any, pagerDefault string) string {
	t.Helper()
	m := map[string]any{}
	v := reflect.ValueOf(o)
	for i := range v.NumField() {
		if f := v.Type().Field(i); f.IsExported() && f.Name != "LoadAbort" {
			m[f.Name] = v.Field(i).Interface()
		}
	}
	if m["MoEPager"] == pagerDefault {
		m["MoEPager"] = "(this OS's default)"
	}
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

// TestSpecOptions_golden is G-SB10-13 (docs/tasks/task-subsystems-2026-10.md § "Step 3b's gates"): modelSpec.options
// over the case table is the golden recorded before Resolve took serve's overrides.
func TestSpecOptions_golden(t *testing.T) {
	t.Setenv("GOINFER_GGUF_DIRECT", "")
	got := map[string]string{}
	for _, c := range optionsCases() {
		fs := flag.NewFlagSet("serve", flag.ContinueOnError)
		lf := loadflags.Register(fs, loadflags.Serve)
		if err := fs.Parse(c.args); err != nil {
			t.Fatal(err)
		}
		var m modelFlag
		if err := m.Set(c.spec); err != nil {
			t.Fatal(err)
		}
		cfg := config{load: *lf}
		if c.serve != nil {
			c.serve(&cfg)
		}
		got[c.name] = specOptionsJSON(t, m[0].options(cfg), fs.Lookup("moe-pager").DefValue)
	}
	const path = "testdata/options.golden.json"
	if *updateOptionsGolden {
		b, err := json.MarshalIndent(got, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, append(b, '\n'), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var want map[string]string
	if err := json.Unmarshal(raw, &want); err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Errorf("%d cases, golden has %d", len(got), len(want))
	}
	for k, w := range want {
		if got[k] != w {
			t.Errorf("%s:\n got %s\nwant %s", k, got[k], w)
		}
	}
}
