//go:build realckpt

package decoder

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"
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
				prev, prevArmed := mlaProjF32ForTest, mlaInt8ForTest
				mlaProjF32ForTest, mlaInt8ForTest = map[string]bool{}, true
				for _, p := range strings.Split(arm, "+") {
					if p != "all8" {
						mlaProjF32ForTest[p] = true
					}
				}
				defer func() { mlaProjF32ForTest, mlaInt8ForTest = prev, prevArmed }()
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

// TestD14_mlaSpeedAB times whole-token CPU decode on Moonlight at int8int8 with MLA's projections int8 (A) against f32 (B), the
// two loads interleaved A B B A in one process, GOINFER_D14_SPEED_PAIRS pairs of 32 tokens each. A cell started with the 1-minute
// load average above 8.0 is VOID and re-run in place (CLAUDE.md, "Run budget"). D14's speed reading, not a gate.
func TestD14_mlaSpeedAB(t *testing.T) {
	requireHeavyModel(t)
	pairs, _ := strconv.Atoi(os.Getenv("GOINFER_D14_SPEED_PAIRS"))
	if pairs == 0 {
		t.Skip("GOINFER_D14_SPEED_PAIRS unset")
	}
	ckpt := assetPath(t, "GOINFER_DEEPSEEK_MOONLIGHT")
	raw, err := readGolden("../testdata/deepseek_moonlight_golden.json.gz")
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		PromptIDs []int `json:"prompt_ids"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	load := func(int8MLA bool) *Model {
		prev := mlaInt8ForTest
		mlaInt8ForTest = int8MLA
		defer func() { mlaInt8ForTest = prev }()
		m, err := Load(ckpt, Options{Quant: "int8int8", ResidentContext: 256})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { m.Close() })
		fmt.Fprintf(os.Stderr, "[D14-speed] loaded int8 MLA=%v: resident weight bytes %d\n", int8MLA, m.ResidentWeightBytes())
		return m
	}
	arms := map[string]*Model{"A": load(true), "B": load(false)}
	const nTok = 32
	cell := func(arm string) float64 {
		for {
			la, mem := d14HostLoad()
			if la > 8.0 {
				fmt.Fprintf(os.Stderr, "[D14-speed] VOID cell %s: load1 %.2f > 8.0, free %d MB; re-run in 30 s\n", arm, la, mem)
				time.Sleep(30 * time.Second)
				continue
			}
			m := arms[arm]
			cache := m.NewCache(len(g.PromptIDs) + nTok + 1)
			var logits []float32
			for _, id := range g.PromptIDs {
				if logits, err = m.forward(id, cache); err != nil {
					t.Fatal(err)
				}
			}
			start := time.Now()
			for range nTok {
				if logits, err = m.forward(argmax(logits), cache); err != nil {
					t.Fatal(err)
				}
			}
			tps := nTok / time.Since(start).Seconds()
			fmt.Fprintf(os.Stderr, "[D14-speed] cell %s: load1 %.2f free %d MB: %.3f tok/s\n", arm, la, mem, tps)
			return tps
		}
	}
	cell("A") // warm-up, discarded
	cell("B")
	var ratios []float64
	for p := range pairs {
		a1, b1 := cell("A"), cell("B")
		b2, a2 := cell("B"), cell("A")
		r := (b1 + b2) / (a1 + a2)
		ratios = append(ratios, r)
		fmt.Fprintf(os.Stderr, "[D14-speed] pair %d/%d: f32/int8 decode %.4f\n", p+1, pairs, r)
	}
	sort.Float64s(ratios)
	fmt.Fprintf(os.Stderr, "[D14-speed] f32/int8 decode ratios %v, median %.4f\n", ratios, ratios[len(ratios)/2])
}

// d14HostLoad returns the 1-minute load average and MemAvailable in MB.
func d14HostLoad() (float64, int) {
	var la float64
	if b, err := os.ReadFile("/proc/loadavg"); err == nil {
		fmt.Sscan(string(b), &la)
	}
	mem := 0
	if b, err := os.ReadFile("/proc/meminfo"); err == nil {
		for _, ln := range strings.Split(string(b), "\n") {
			if strings.HasPrefix(ln, "MemAvailable:") {
				fmt.Sscan(strings.TrimPrefix(ln, "MemAvailable:"), &mem)
			}
		}
	}
	return la, mem / 1024
}
