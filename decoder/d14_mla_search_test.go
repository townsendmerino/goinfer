//go:build realckpt

package decoder

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
)

// TestD14_mlaProjSearch runs the DeepSeek family gate with one MLA projection held f32 and the rest int8, per arm named in
// GOINFER_D14_ARMS ("q,kva,o", "all8" for every projection int8), and logs at each continuation step the logit gap between the
// golden's token and the token taken. D14's search (docs/measurements/mem11-int8-2026-10/mla.md); the verdict is the gate's.
func TestD14_mlaProjSearch(t *testing.T) {
	requireHeavyModel(t)
	arms := os.Getenv("GOINFER_D14_ARMS")
	if arms == "" {
		t.Skip("GOINFER_D14_ARMS unset")
	}
	models := []struct{ env, golden, arch, ref string }{
		{"GOINFER_DEEPSEEK_MOONLIGHT", "../testdata/deepseek_moonlight_golden.json.gz", "deepseek_v3", "HF bf16 (Moonlight-16B-A3B; int8 resident)"},
		{"GOINFER_DEEPSEEK_V2LITE", "../testdata/deepseek_v2lite_golden.json", "deepseek_v2", "HF bf16 (DeepSeek-V2-Lite 15.7B; int8 resident)"},
	}
	only := os.Getenv("GOINFER_D14_MODEL") // "deepseek_v3" or "deepseek_v2"; empty runs both
	for _, arm := range strings.Split(arms, ";") {
		for _, md := range models {
			if only != "" && md.arch != only {
				continue
			}
			t.Run(md.arch+"/"+arm, func(t *testing.T) {
				prev := mlaProjF32ForTest
				mlaProjF32ForTest = map[string]bool{}
				for _, p := range strings.Split(arm, "+") {
					if p != "all8" {
						mlaProjF32ForTest[p] = true
					}
				}
				defer func() { mlaProjF32ForTest = prev }()
				d14Gaps(t, assetPath(t, md.env), md.golden)
				deepseekRealGate(t, assetPath(t, md.env), md.golden, md.arch, md.ref, md.arch == "deepseek_v3")
			})
		}
	}
}

// d14Gaps loads ckpt at int8int8 and, teacher-forcing the golden continuation, logs per step the golden token's logit less the
// best other token's and the resident weight bytes.
func d14Gaps(t *testing.T, ckpt, golden string) {
	raw, err := readGolden(golden)
	if err != nil {
		t.Skipf("golden: %v", err)
	}
	var g struct {
		PromptIDs       []int `json:"prompt_ids"`
		NNew            int   `json:"n_new"`
		ContinuationIDs []int `json:"continuation_ids"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	m, err := Load(ckpt, Options{Quant: "int8int8"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	cache := m.NewCache(len(g.PromptIDs) + g.NNew)
	var logits []float32
	for _, id := range g.PromptIDs {
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatal(err)
		}
	}
	var b strings.Builder
	for i, want := range g.ContinuationIDs {
		// gap: the golden token's logit less the best other token's; positive when greedy takes the golden token.
		other := -1
		for j, x := range logits {
			if j != want && (other < 0 || x > logits[other]) {
				other = j
			}
		}
		fmt.Fprintf(&b, " [%d] want %d gap %+.4f (vs %d)", i, want, logits[want]-logits[other], other)
		if logits, err = m.forward(want, cache); err != nil {
			t.Fatal(err)
		}
	}
	fmt.Fprintf(os.Stderr, "[D14] resident weight bytes %d; teacher-forced gaps:%s\n", m.ResidentWeightBytes(), b.String())
}
