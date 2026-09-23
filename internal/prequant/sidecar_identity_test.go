package prequant

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestSidecarLoad_matchesDirectLoad is task-never-swap-2026-09.md S1's own registered rule, the
// "byte-identical" third of the three ("anonymous footprint ≤25%, swap-used delta=0, greedy
// stream byte-identical... ALL THREE or the switch does not ship"). This is the third: a plain
// .gguf loaded direct into the heap and the SAME source loaded through EnsureCachedGIW's sidecar
// must produce the exact same greedy token stream — the two paths quantize the same tensors with
// the same code, so a difference here is a defect in one of them, not a tolerance to widen (the
// brief's own Decision).
//
// No small tokenizer-bearing GGUF ships in testdata/ (the same constraint
// TestTranscode_realGGUFSucceedsAndPublishedBundleLoads already works around — glm-tiny.gguf has
// no embedded tokenizer, so Transcode refuses it before ever reaching weight-streaming), so this
// is heavy-gated like that test rather than inventing a CI-native fixture.
func TestSidecarLoad_matchesDirectLoad(t *testing.T) {
	gguf := realTokenizerGGUF(t)
	dir := t.TempDir()
	src := filepath.Join(dir, filepath.Base(gguf))
	copyFixture(t, gguf, src)

	const quant = "int8int8" // higher precision than int4 — a looser quant is more likely to mask a real divergence, not less
	direct, err := decoder.Load(src, decoder.Options{Quant: quant})
	if err != nil {
		t.Fatalf("direct Load: %v", err)
	}
	defer direct.Close()

	giwPath, err := EnsureCachedGIW(context.Background(), src, quant, "")
	if err != nil {
		t.Fatalf("EnsureCachedGIW: %v", err)
	}
	sidecar, err := decoder.Load(giwPath, decoder.Options{})
	if err != nil {
		t.Fatalf("sidecar Load: %v", err)
	}
	defer sidecar.Close()

	prompts := [][]int{
		{3, 141, 7, 88, 219, 14, 66, 190},
		{1, 2, 3, 4, 5},
		{9999, 1, 4242, 7, 12},
	}
	const maxTokens = 64

	for i, prompt := range prompts {
		directOut := greedyTokens(t, direct, prompt, maxTokens)
		sidecarOut := greedyTokens(t, sidecar, prompt, maxTokens)
		if len(directOut) != len(sidecarOut) {
			t.Errorf("prompt[%d]: direct produced %d tokens, sidecar %d", i, len(directOut), len(sidecarOut))
			continue
		}
		for j := range directOut {
			if directOut[j] != sidecarOut[j] {
				t.Errorf("prompt[%d]: token %d diverges: direct=%d sidecar=%d (direct stream %v, sidecar stream %v)",
					i, j, directOut[j], sidecarOut[j], directOut, sidecarOut)
				break
			}
		}
	}
}

func greedyTokens(t *testing.T, m *decoder.Model, prompt []int, maxTokens int) []int {
	t.Helper()
	out, gen := m.Generate(context.Background(), prompt, maxTokens, decoder.SamplingParams{})
	var ids []int
	for id := range out {
		ids = append(ids, id)
	}
	if err := gen.Err(); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	return ids
}

func init() {
	// Guard the fixture-discovery assumption directly rather than let a missing HOME env
	// silently turn every candidate path into a garbage one that simply never matches.
	if os.Getenv("GOINFER_HEAVY_TESTS") != "" && os.Getenv("HOME") == "" {
		panic("GOINFER_HEAVY_TESTS is set but HOME is empty — realTokenizerGGUF's ~/models candidate would silently never match")
	}
}
