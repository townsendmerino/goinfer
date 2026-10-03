package decoder

import (
	"bytes"
	"context"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The split fixture is testdata/glm-tiny.gguf cut into four shards by llama.cpp's own llama-gguf-split (see the
// .gitignore exception that commits it), so these gates read the real tool's shard format, not a writer of ours.
const (
	splitSingle = "../testdata/glm-tiny.gguf"
	splitFirst  = "../testdata/gguf-split/glm-tiny-00001-of-00004.gguf"
)

func needSplitFixture(t *testing.T) {
	t.Helper()
	for _, p := range []string{splitSingle, splitFirst} {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("no split fixture: %v", err)
		}
	}
}

// TestSplitGGUF_transcodeMatchesSingleFile is P4's gate (task-checkpoint-fetch-2026-09.md): the .giw weights a split
// GGUF transcodes to are byte-identical to those the single file it was split from transcodes to, at each quant the
// sidecar is built at. The sidecar is what every non-direct load runs from, so this is the whole of what a split
// model changes there.
func TestSplitGGUF_transcodeMatchesSingleFile(t *testing.T) {
	needSplitFixture(t)
	for _, quant := range []string{"int4", "int8int8"} {
		var one, set bytes.Buffer
		if _, err := StreamTranscodeGGUF(context.Background(), splitSingle, &one, quant, false, GIWTargetNone, "glm-tiny.gguf"); err != nil {
			t.Fatalf("%s single file: %v", quant, err)
		}
		if _, err := StreamTranscodeGGUF(context.Background(), splitFirst, &set, quant, false, GIWTargetNone, "glm-tiny.gguf"); err != nil {
			t.Fatalf("%s split set: %v", quant, err)
		}
		if one.Len() == 0 || !bytes.Equal(one.Bytes(), set.Bytes()) {
			t.Fatalf("%s: the split set's .giw weights (%d bytes) differ from the single file's (%d bytes)", quant, set.Len(), one.Len())
		}
		t.Logf("%s: %d bytes of .giw weights, identical", quant, one.Len())
	}
}

// TestSplitGGUF_directLoadMatchesSingleFile: a direct load (-direct-load, no sidecar) of the split set produces the
// single file's logits bit for bit over a short prompt, so the GGUF build path reads every tensor from the right shard.
func TestSplitGGUF_directLoadMatchesSingleFile(t *testing.T) {
	needSplitFixture(t)
	logits := func(path string) [][]float32 {
		m, err := Load(path, Options{Backend: "cpu", Quant: "int8int8"})
		if err != nil {
			t.Fatalf("Load %s: %v", path, err)
		}
		defer m.Close()
		cache := m.NewCache(8)
		var out [][]float32
		for _, id := range []int{1, 7, 42, 3} {
			l, _, err := m.ForwardCapture(id, cache, nil)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, append([]float32(nil), l...))
		}
		return out
	}
	want, got := logits(splitSingle), logits(splitFirst)
	for p := range want {
		for i := range want[p] {
			if math.Float32bits(got[p][i]) != math.Float32bits(want[p][i]) {
				t.Fatalf("position %d logit %d: split %v, single file %v", p, i, got[p][i], want[p][i])
			}
		}
	}
}

// TestGGUFShards_refusals: a model is named by its first shard. A later shard is refused with the first shard's name,
// a set with a shard missing names the missing one, and a set's size is all of its shards.
func TestGGUFShards_refusals(t *testing.T) {
	needSplitFixture(t)
	if _, err := GGUFShards(strings.Replace(splitFirst, "00001-of", "00002-of", 1)); err == nil || !strings.Contains(err.Error(), "glm-tiny-00001-of-00004.gguf") {
		t.Errorf("a later shard: %v, want a refusal naming the first shard", err)
	}
	dir := t.TempDir()
	for _, n := range []string{"00001", "00002", "00004"} {
		b, err := os.ReadFile(strings.Replace(splitFirst, "00001-of", n+"-of", 1))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "glm-tiny-"+n+"-of-00004.gguf"), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := OpenGGUFMmap(filepath.Join(dir, "glm-tiny-00001-of-00004.gguf")); err == nil || !strings.Contains(err.Error(), "shard 3 of 4 (glm-tiny-00003-of-00004.gguf) is missing") {
		t.Errorf("a set missing shard 3: %v, want the missing shard named", err)
	}
	if got, ok := GGUFShards(splitSingle); ok != nil || len(got) != 1 || got[0] != splitSingle {
		t.Errorf("a single file is its own set: %v, %v", got, ok)
	}
	var want int64
	shards, _ := GGUFShards(splitFirst)
	for _, p := range shards {
		fi, _ := os.Stat(p)
		want += fi.Size()
	}
	if n, ok := GGUFFileBytes(splitFirst); !ok || n != want || len(shards) != 4 {
		t.Errorf("set size %d (%v) over %d shards, want %d over 4", n, ok, len(shards), want)
	}
}
