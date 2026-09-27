//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"os"
	"path/filepath"
	"runtime/debug"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestMC3_concurrentMatchesAloneOnMetal is MC3's identity gate through the production Generate path on a real
// checkpoint (docs/tasks/task-concurrency-2026-09.md): 4 conversations × 3 turns × 32 greedy tokens, generating at once
// on one Metal resident with EnableResidentConcurrency(4), each emit exactly the ids — and reuse exactly the prefix —
// they do when the same conversations run one after another on a model without concurrency. Greedy tokens a batched
// step serves continue on the host-argmax path; alone they take the device argmax (ForwardArgmax) — so this also pins
// that production's greedy fast path and the full-logits path agree, which MC3 relies on.
//
//	GOINFER_METAL_MC3=1 go test -tags goinfer_testhooks -count=1 -run '^TestMC3_concurrentMatchesAloneOnMetal$' -v ./metal/
func TestMC3_concurrentMatchesAloneOnMetal(t *testing.T) {
	if os.Getenv("GOINFER_METAL_MC3") != "1" {
		t.Skip("set GOINFER_METAL_MC3=1 (loads a real checkpoint twice)")
	}
	home, _ := os.UserHomeDir()
	path := os.Getenv("GOINFER_METAL_MC3_MODEL")
	if path == "" {
		path = filepath.Join(home, "models", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
	if strings.HasPrefix(path, "/Volumes/") || strings.HasPrefix(path, "/srv/models") {
		t.Fatalf("%s is on the archive, not the bench set (CLAUDE.md)", path)
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s: %v", path, err)
	}
	const nConv, turns, maxTok = 4, 3, 32
	load := func() *decoder.Model {
		m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 1024, ResidentKVSlots: nConv})
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		if !m.ResidentActive() || m.ResidentKVSlots() != nConv {
			m.Close()
			t.Fatalf("resident %v with %d slots, want %d", m.ResidentActive(), m.ResidentKVSlots(), nConv)
		}
		return m
	}
	type turn struct {
		ids    []int
		reused int
	}
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Skipf("tokenizer: %v", err)
	}
	enc := func(s string) []int {
		ids, err := tk.Encode(s, false)
		if err != nil {
			t.Fatalf("encode: %v", err)
		}
		return ids
	}
	asks := []string{"a Python function that returns the n-th Fibonacci number", "a Go function that reverses a slice of ints",
		"a JavaScript function that debounces another function", "a SQL query listing the ten most recent orders"}
	run := func(m *decoder.Model, c int) []turn {
		prompt := enc("<|im_start|>user\nWrite " + asks[c] + ".<|im_end|>\n<|im_start|>assistant\n")
		var out []turn
		for tn := range turns {
			ch, gen := m.Generate(context.Background(), prompt, maxTok, decoder.SamplingParams{})
			var ids []int
			for id := range ch {
				ids = append(ids, id)
			}
			if err := gen.Err(); err != nil {
				t.Errorf("conversation %d turn %d: %v", c, tn, err)
				return out
			}
			out = append(out, turn{ids, gen.PrefillReused})
			prompt = append(append(slices.Clone(prompt), ids...), enc("<|im_end|>\n<|im_start|>user\nNow make it shorter.<|im_end|>\n<|im_start|>assistant\n")...)
		}
		return out
	}
	mAlone := load()
	alone := make([][]turn, nConv)
	for c := range nConv {
		alone[c] = run(mAlone, c)
	}
	mAlone.Close()
	debug.FreeOSMemory()

	m := load()
	defer m.Close()
	if got := m.EnableResidentConcurrency(nConv); got != nConv {
		t.Fatalf("EnableResidentConcurrency(%d) = %d: the resident cannot batch", nConv, got)
	}
	concurrent := make([][]turn, nConv)
	var wg sync.WaitGroup
	for c := range nConv {
		wg.Add(1)
		go func(c int) {
			defer wg.Done()
			concurrent[c] = run(m, c)
		}(c)
	}
	wg.Wait()
	b := m.ResidentForwardForTest().(*metalResident).r.batch
	if b.steps == 0 || b.maxSeqs < 2 {
		t.Errorf("%d batched steps (largest %d sequences): the concurrent tokens never batched", b.steps, b.maxSeqs)
	}
	t.Logf("batched steps %d serving %d tokens, largest %d", b.steps, b.seqs, b.maxSeqs)
	for c := range nConv {
		for tn := range alone[c] {
			if tn >= len(concurrent[c]) {
				t.Fatalf("conversation %d: %d turns concurrently, %d alone", c, len(concurrent[c]), len(alone[c]))
			}
			a, b := alone[c][tn], concurrent[c][tn]
			if !slices.Equal(a.ids, b.ids) {
				t.Errorf("conversation %d turn %d: concurrent %v, alone %v", c, tn, b.ids, a.ids)
			}
			if a.reused != b.reused {
				t.Errorf("conversation %d turn %d: concurrent reused %d, alone %d", c, tn, b.reused, a.reused)
			}
		}
		t.Logf("conversation %d: reused per turn %v; turn 0 %d tokens", c, func() []int {
			var r []int
			for _, x := range concurrent[c] {
				r = append(r, x.reused)
			}
			return r
		}(), len(concurrent[c][0].ids))
	}
}
