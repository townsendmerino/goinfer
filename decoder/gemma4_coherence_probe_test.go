//go:build realckpt

package decoder

import (
	"context"
	"fmt"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestGemma4CoherenceProbe runs the two "free checks" for the int4 investigation
// (docs/task-gemma4-moe.md): is the semi-coherent 4-bit output a GREEDY artifact (sampling
// fixes it), and is it a CHAT-TEMPLATE / off-distribution artifact (a properly rendered
// Gemma-4 turn fixes it)? For the env-configured quant it generates {raw, templated} ×
// {greedy, sampled} from one model load. Set ZZBASE / GOINFER_FAKEQUANT[_ACT] as usual.
func TestGemma4CoherenceProbe(t *testing.T) {
	requireHeavyModel(t)
	dir := modelPath("gemma-4-26b-a4b-it")
	if _, err := os.Stat(dir); err != nil {
		t.Skip("real gemma-4-26b-a4b-it not present")
	}
	base := os.Getenv("ZZBASE")
	if base == "" {
		base = "int4"
	}
	m, err := Load(dir, Options{Quant: base})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	tk, err := tokenizer.Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	cfg := fmt.Sprintf("base=%s scheme=%q act=%q", base, os.Getenv("GOINFER_FAKEQUANT"), os.Getenv("GOINFER_FAKEQUANT_ACT"))

	rawIDs, _ := tk.Encode("The capital of France is", true)
	tplSegs := chat.Gemma4().RenderSegments("", []chat.Turn{{Role: "user", Content: "What is the capital of France, and what is it famous for?"}})
	tplIDs, err := tk.EncodeSegments(tplSegs, false)
	if err != nil {
		t.Fatal(err)
	}
	fmt.Printf("PROMPT-IDS raw=%d tmpl=%d (tmpl first 12: %v)\n", len(rawIDs), len(tplIDs), tplIDs[:min(12, len(tplIDs))])

	greedy := SamplingParams{}
	sampled := SamplingParams{Temperature: 0.7, TopP: 0.95, Seed: 42}
	run := func(label string, ids []int, sp SamplingParams) {
		out, _ := m.Generate(context.Background(), ids, 64, sp)
		g := make([]int, 0, 64)
		for id := range out {
			g = append(g, id)
		}
		txt, _ := tk.Decode(g)
		fmt.Printf("[%s | %-13s] %q\n", cfg, label, txt)
	}
	run("raw+greedy", rawIDs, greedy)
	run("raw+sampled", rawIDs, sampled)
	run("tmpl+greedy", tplIDs, greedy)
	run("tmpl+sampled", tplIDs, sampled)
}
