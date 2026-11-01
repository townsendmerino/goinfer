//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/cost"
)

// cudaGenerate runs a greedy Generate on m with c on the context (nil: off) and returns the emitted tokens.
func cudaGenerate(t *testing.T, m *decoder.Model, prompt []int, n int, c *cost.Counter) []int {
	t.Helper()
	stream, g := m.Generate(cost.WithCounter(context.Background(), c), prompt, n, decoder.SamplingParams{Temperature: 0})
	var out []int
	for id := range stream {
		out = append(out, id)
	}
	if err := g.Err(); err != nil {
		t.Fatal(err)
	}
	return out
}

// TestCost_cudaGenerateOnTheWire (G-SB11b-4): llama-tiny resident on CUDA, int4, the greedy fast paths off so each decode token
// is G-SB11-8's plain Forward. A counted Generate's prefill is one span covering the prompt; its decode covers one token per
// emitted token, and each kernel's launches are G-SB11-8's per-token golden times the tokens; its tokens equal an uncounted
// Generate's. Under MC3 (a resident shared by generations) nothing is counted: the report is empty.
func TestCost_cudaGenerateOnTheWire(t *testing.T) {
	requireCUDADevice(t)
	m, err := decoder.Load(filepath.Join("..", "testdata", "llama-tiny"), decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if _, ok := m.ResidentForwardForTest().(*cudaResident); !ok {
		t.Fatalf("not resident: %s", m.ResidentDecline())
	}
	decoder.SetKnobEnvForTest(t, m, "GOINFER_NO_GREEDY_FASTPATH", "1")
	raw, err := os.ReadFile(filepath.Join("testdata", "sb11_decode_llama_tiny.json"))
	if err != nil {
		t.Fatal(err)
	}
	var perToken sb11Token
	if err := json.Unmarshal(raw, &perToken); err != nil {
		t.Fatal(err)
	}

	prompt := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	off := cudaGenerate(t, m, prompt, 5, nil)
	c := cost.New()
	on := cudaGenerate(t, m, prompt, 5, c)
	if !slices.Equal(on, off) {
		t.Fatalf("tokens %v with the counter on, %v off", on, off)
	}
	r := c.Report()
	if r.Prefill == nil || r.Prefill.Tokens != int64(len(prompt)) || r.Prefill.WallNs <= 0 {
		t.Errorf("prefill phase %+v: want one span of %d tokens with its wall time", r.Prefill, len(prompt))
	}
	if r.Decode == nil {
		t.Fatal("no decode phase")
	}
	n := int64(len(on))
	if r.Decode.Tokens != n || r.Decode.WallNs <= 0 {
		t.Errorf("decode phase: %d tokens over %d ns, want %d tokens (one per emitted token) and its wall time", r.Decode.Tokens, r.Decode.WallNs, n)
	}
	for name, v := range perToken {
		if name == "sync" || name == "alloc" {
			continue
		}
		if got, want := r.Decode.Ops[name].Launches, v[0]*n; got != want {
			t.Errorf("decode %s: %d launches, want the golden's %d per token times %d", name, got, v[0], n)
		}
	}
	if o, ok := r.Decode.Ops["launch_other"]; ok {
		t.Errorf("%d decode launches not made through r.launch", o.Launches)
	}

	// MC3: a resident shared by generations counts nothing for any one of them.
	m2, err := decoder.Load(filepath.Join("..", "testdata", "llama-tiny"), decoder.Options{Backend: "cuda", Quant: "int4", ResidentKVSlots: 2})
	if err != nil {
		t.Fatal(err)
	}
	defer m2.Close()
	if got := m2.EnableResidentConcurrency(2); got != 2 {
		t.Skipf("MC3 did not engage on llama-tiny (EnableResidentConcurrency(2) = %d)", got)
	}
	c2 := cost.New()
	cudaGenerate(t, m2, prompt, 3, c2)
	if r2 := c2.Report(); r2.Prefill != nil || r2.Decode != nil {
		t.Errorf("under MC3 a generation was counted: %+v", r2)
	}
}
