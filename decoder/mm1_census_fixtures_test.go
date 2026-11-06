package decoder

import (
	"os/exec"
	"path"
	"slices"
	"strings"
	"testing"
)

// censusFixtures is what the census reads, as paths from this package: the committed tiny checkpoints, directories with a
// config.json and weights and the tiny .gguf files. A fixture that is only on one box (a gitignored one, a real model) is not
// read, so every box computes the census CI does. TestCensusFixtures_areTheCommittedOnes keeps the list whole; a fixture added
// here moves the golden (-census.update).
var censusFixtures = []string{
	"../internal/vision/testdata/gemma4-vision-tiny",
	"../internal/vision/testdata/glm-ocr-vision-tiny",
	"../internal/vision/testdata/pixtral-vision-tiny",
	"../internal/vision/testdata/qwen25vl-vision-tiny",
	"../internal/vision/testdata/qwen35vl-vision-tiny",
	"../internal/vision/testdata/qwen3vl-vision-tiny",
	"../internal/vision/testdata/siglip-tiny",
	"../internal/vision/testdata/siglip2-naflex-tiny",
	"../testdata/bailing_hybrid-tiny",
	"../testdata/cohere-tiny",
	"../testdata/cohere2-tiny",
	"../testdata/deepseek-tiny",
	"../testdata/embeddinggemma2-tiny",
	"../testdata/gemma1-tiny",
	"../testdata/gemma1-tiny.gguf",
	"../testdata/gemma2-hd-tiny",
	"../testdata/gemma2-hd-tiny.gguf",
	"../testdata/gemma2-tiny",
	"../testdata/gemma3-vl-tiny",
	"../testdata/gemma4-audio-tiny",
	"../testdata/gemma4-moe-kv-tiny",
	"../testdata/gemma4-moe-kv-tiny32",
	"../testdata/gemma4-vision-tiny",
	"../testdata/gguf-split/glm-tiny-00001-of-00004.gguf",
	"../testdata/glm-ocr-tiny",
	"../testdata/glm-ocr-vision-tiny",
	"../testdata/glm-tiny",
	"../testdata/glm-tiny-bias",
	"../testdata/glm-tiny.gguf",
	"../testdata/granite-dense-tiny",
	"../testdata/lfm2-tiny",
	"../testdata/llama-attnfa-tiny",
	"../testdata/llama-tiny",
	"../testdata/llama4-tiny",
	"../testdata/ministral3-tiny",
	"../testdata/mistral-tiny-window",
	"../testdata/mixtral-tiny",
	"../testdata/nemotron-tiny",
	"../testdata/nemotron3nano-tiny",
	"../testdata/olmo3-tiny",
	"../testdata/olmo_hybrid-tiny",
	"../testdata/phi3-tiny",
	"../testdata/qwen25vl-tiny",
	"../testdata/qwen35-tiny",
	"../testdata/qwen35vl-tiny",
	"../testdata/qwen3asr-tiny",
	"../testdata/qwen3moe-tiny",
	"../testdata/qwen3moe-tiny-k3",
	"../testdata/qwen3next-tiny",
	"../testdata/qwen3vl-vision-tiny",
	"../testdata/qwen3vlmoe-tiny",
	"../testdata/siglip-tiny",
	"../testdata/smollm3-tiny",
	"../testdata/voxtral-tiny",
	"../testdata/whisper-tiny-rand",
	"../testdata/whisper-tiny-ts",
	"selftestdata/gemma3-vl-tiny",
	"selftestdata/llama-tiny",
	"selftestdata/qwen35-tiny",
	"selftestdata/qwen35vl-tiny",
	"testdata/gptoss_tiny.gguf",
	"testdata/internlm2-tiny",
	"testdata/internlm3-tiny",
	"testdata/laguna-m1-tiny",
	"testdata/laguna-xs2-tiny",
	"testdata/laguna-xs21-tiny",
	"testdata/qwen3_5-tiny",
	"testdata/qwen3_5-tiny-normw",
	"testdata/qwen3_5_moe-tiny",
	"testdata/spark2-5-tiny",
}

// committedCensusFixtures is every committed file under the census's roots that would be a fixture by censusFixtures' rule.
func committedCensusFixtures(t *testing.T) []string {
	t.Helper()
	roots := []string{"../testdata", "testdata", "selftestdata", "../internal/vision/testdata"}
	out, err := exec.Command("git", append([]string{"ls-files", "--"}, roots...)...).Output()
	if err != nil {
		t.Skipf("git ls-files: %v: this check needs a git checkout", err)
	}
	type entry struct{ config, weights, file bool }
	entries := map[string]*entry{}
	for _, f := range strings.Fields(string(out)) {
		for _, root := range roots {
			rest, ok := strings.CutPrefix(f, root+"/")
			if !ok {
				continue
			}
			name, inside, isDir := strings.Cut(rest, "/")
			p := root + "/" + name
			e := entries[p]
			if e == nil {
				e = &entry{}
				entries[p] = e
			}
			switch {
			case !isDir:
				e.file = true
			case inside == "config.json":
				e.config = true
			case !strings.Contains(inside, "/") && path.Ext(inside) == ".safetensors":
				e.weights = true
			}
		}
	}
	var got []string
	for p, e := range entries {
		if (e.config && e.weights) || (e.file && strings.HasSuffix(p, ".gguf")) {
			got = append(got, p)
		}
	}
	const split = "../testdata/gguf-split/glm-tiny-00001-of-00004.gguf"
	if slices.Contains(strings.Fields(string(out)), split) {
		got = append(got, split)
	}
	slices.Sort(got)
	return got
}

// The census reads a declared list so that a box's extra fixtures cannot move it. A committed fixture left off the list would
// leave the census quietly, and a listed one that is not committed would be red on CI only.
func TestCensusFixtures_areTheCommittedOnes(t *testing.T) {
	committed := committedCensusFixtures(t)
	if len(committed) < 20 {
		t.Fatalf("git lists %d census fixtures: this read the wrong tree", len(committed))
	}
	for _, p := range committed {
		if !slices.Contains(censusFixtures, p) {
			t.Errorf("%s is committed and not in censusFixtures: add it, then -census.update", p)
		}
	}
	for _, p := range censusFixtures {
		if !slices.Contains(committed, p) {
			t.Errorf("censusFixtures lists %s, which is not committed: CI does not have it", p)
		}
	}
}
