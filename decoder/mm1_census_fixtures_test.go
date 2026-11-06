package decoder

import (
	"fmt"
	"os/exec"
	"path"
	"regexp"
	"slices"
	"strconv"
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

// censusFixtureRoots are the directories, from this package, that hold the census's fixtures.
var censusFixtureRoots = []string{"../testdata", "testdata", "selftestdata", "../internal/vision/testdata"}

// ggufShard matches a split GGUF's shard name: its number and the shard count.
var ggufShard = regexp.MustCompile(`-(\d{5})-of-(\d{5})\.gguf$`)

// committedCensusFixtures is every committed fixture under censusFixtureRoots, at any depth, by censusFixtures' rule: a directory
// holding a config.json and a .safetensors file, and a .gguf file, a split one by its first shard. It also returns the tracked files
// that the working tree is missing. It skips without a git checkout.
func committedCensusFixtures(t *testing.T) (fixtures, deleted []string) {
	t.Helper()
	lsFiles := func(flags ...string) []string {
		out, err := exec.Command("git", append(append([]string{"ls-files", "-z"}, flags...), append([]string{"--"}, censusFixtureRoots...)...)...).Output()
		if err != nil {
			t.Skipf("git ls-files: %v: this check needs a git checkout", err)
		}
		return strings.FieldsFunc(string(out), func(r rune) bool { return r == 0 })
	}
	files := lsFiles()
	type dir struct{ config, weights bool }
	dirs := map[string]*dir{}
	for _, f := range files {
		d, base := path.Split(f)
		if dirs[d] == nil {
			dirs[d] = &dir{}
		}
		switch m := ggufShard.FindStringSubmatch(base); {
		case base == "config.json":
			dirs[d].config = true
		case path.Ext(base) == ".safetensors":
			dirs[d].weights = true
		case m != nil && m[1] == "00001":
			fixtures = append(fixtures, f)
			n, _ := strconv.Atoi(m[2])
			for k := 2; k <= n; k++ {
				if shard := strings.TrimSuffix(f, m[0]) + fmt.Sprintf("-%05d-of-%s.gguf", k, m[2]); !slices.Contains(files, shard) {
					t.Errorf("%s is committed and its shard %s is not", f, shard)
				}
			}
		case m == nil && path.Ext(base) == ".gguf":
			fixtures = append(fixtures, f)
		}
	}
	for d, e := range dirs {
		if e.config && e.weights {
			fixtures = append(fixtures, strings.TrimSuffix(d, "/"))
		}
	}
	slices.Sort(fixtures)
	return fixtures, lsFiles("--deleted")
}

// The census reads a declared list so that a box's extra fixtures cannot move it. A committed fixture left off the list would
// leave the census quietly, and a listed one that is not committed would be red on CI only. scripts/touched_scans.py runs the
// census when a file under a fixture whose name has "tiny" in it changes, so every fixture's does.
func TestCensusFixtures_areTheCommittedOnes(t *testing.T) {
	committed, deleted := committedCensusFixtures(t)
	if len(committed) < 20 {
		t.Fatalf("git lists %d census fixtures: this read the wrong tree", len(committed))
	}
	if !slices.IsSorted(censusFixtures) || len(slices.Compact(slices.Clone(censusFixtures))) != len(censusFixtures) {
		t.Errorf("censusFixtures is not sorted, or lists a fixture twice")
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
		if !strings.Contains(path.Base(p), "tiny") {
			t.Errorf("censusFixtures lists %s, whose name has no \"tiny\": touched_scans.py would not run the census when it changes", p)
		}
	}
	for _, f := range deleted {
		if slices.ContainsFunc(censusFixtures, func(p string) bool { return f == p || strings.HasPrefix(f, p+"/") || path.Ext(f) == ".gguf" }) {
			t.Errorf("%s is committed and missing from the working tree, so the census here is not CI's", f)
		}
	}
}
