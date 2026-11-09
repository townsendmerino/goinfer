package decoder

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestRopeReread_wrongTypeIsAnError: a field the RoPE setup reads a second time out of rope_parameters (YaRN's mscale pair,
// Ministral 3's attention temperature, Olmo Hybrid's rope_theta, partial_rotary_factor), given a value of the wrong type, is
// the config's error at resolveArchitecture. Read as absent, it would change the numerics: scaling off, NoPE, or no partial
// rotary. Each case changes one field of a committed fixture's config, and the unchanged fixture resolves.
func TestRopeReread_wrongTypeIsAnError(t *testing.T) {
	cases := []struct {
		name, fixture string
		edit          func(rp map[string]any)
		field         string
	}{
		{"ministral3 llama_4_scaling_beta", "../testdata/ministral3-tiny", func(rp map[string]any) { rp["llama_4_scaling_beta"] = "0.2" }, "llama_4_scaling_beta"},
		{"ministral3 mscale_all_dim", "../testdata/ministral3-tiny", func(rp map[string]any) { rp["mscale_all_dim"] = "0.8" }, "mscale_all_dim"},
		{"deepseek yarn mscale_all_dim", "../testdata/deepseek-tiny", func(rp map[string]any) {
			rp["rope_type"], rp["factor"], rp["original_max_position_embeddings"], rp["mscale_all_dim"] = "yarn", 4.0, 8, "1"
		}, "mscale_all_dim"},
		{"olmo_hybrid rope_theta", "../testdata/olmo_hybrid-tiny", func(rp map[string]any) { rp["rope_theta"] = "10000" }, "rope_theta"},
		{"qwen3_5 partial_rotary_factor", "testdata/qwen3_5-tiny", func(rp map[string]any) { rp["partial_rotary_factor"] = "0.25" }, "partial_rotary_factor"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			raw, err := os.ReadFile(filepath.Join(c.fixture, "config.json"))
			if err != nil {
				t.Fatalf("committed fixture: %v", err)
			}
			resolve := func(b []byte) error {
				dir := t.TempDir()
				if err := os.WriteFile(filepath.Join(dir, "config.json"), b, 0o644); err != nil {
					t.Fatal(err)
				}
				cfg, err := loadConfig(os.DirFS(dir), "config.json")
				if err != nil {
					return err
				}
				_, _, err = resolveArchitecture(cfg)
				return err
			}
			if err := resolve(raw); err != nil {
				t.Fatalf("the unchanged fixture does not resolve: %v", err)
			}
			var m map[string]any
			if err := json.Unmarshal(raw, &m); err != nil {
				t.Fatal(err)
			}
			top := m
			if tc, ok := m["text_config"].(map[string]any); ok {
				top = tc
			}
			rp, _ := top["rope_parameters"].(map[string]any)
			if rp == nil {
				t.Fatal("the fixture has no rope_parameters object")
			}
			c.edit(rp)
			b, err := json.Marshal(m)
			if err != nil {
				t.Fatal(err)
			}
			err = resolve(b)
			if err == nil {
				t.Fatalf("a string %s resolves; want the config's error", c.field)
			}
			if !strings.Contains(err.Error(), "rope_parameters") {
				t.Errorf("the error does not name rope_parameters: %v", err)
			}
		})
	}
}
