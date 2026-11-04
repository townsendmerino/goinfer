package serveapp

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeMMProj writes a minimal GGUF mmproj header: clip metadata for a tower with the given projector type and output
// width, and the patch-embedding and position tensors' directory entries (no data: the config read never touches it).
func writeMMProj(t *testing.T, path, projector string, outDim uint32) {
	t.Helper()
	var b []byte
	u32 := func(v uint32) { b = binary.LittleEndian.AppendUint32(b, v) }
	u64 := func(v uint64) { b = binary.LittleEndian.AppendUint64(b, v) }
	str := func(s string) { u64(uint64(len(s))); b = append(b, s...) }
	type kv struct {
		k   string
		typ uint32
		put func()
	}
	kvs := []kv{
		{"general.architecture", 8, func() { str("clip") }},
		{"clip.projector_type", 8, func() { str(projector) }},
		{"clip.vision.block_count", 4, func() { u32(2) }},
		{"clip.vision.embedding_length", 4, func() { u32(64) }},
		{"clip.vision.feed_forward_length", 4, func() { u32(128) }},
		{"clip.vision.attention.head_count", 4, func() { u32(4) }},
		{"clip.vision.patch_size", 4, func() { u32(16) }},
		{"clip.vision.spatial_merge_size", 4, func() { u32(2) }},
		{"clip.vision.projection_dim", 4, func() { u32(outDim) }},
		{"clip.vision.attention.layer_norm_epsilon", 6, func() { u32(math.Float32bits(1e-6)) }},
	}
	tensors := []struct {
		name string
		dims []uint64
	}{{"v.patch_embd.weight", []uint64{16, 16, 3, 64}}, {"v.patch_embd.weight.1", []uint64{16, 16, 3, 64}}, {"v.position_embd.weight", []uint64{64, 2304}}}
	b = append(b, "GGUF"...)
	u32(3)
	u64(uint64(len(tensors)))
	u64(uint64(len(kvs)))
	for _, e := range kvs {
		str(e.k)
		u32(e.typ)
		e.put()
	}
	for _, tn := range tensors {
		str(tn.name)
		u32(uint32(len(tn.dims)))
		for _, d := range tn.dims {
			u64(d)
		}
		u32(0) // F32
		u64(0)
	}
	for len(b)%32 != 0 {
		b = append(b, 0)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
}

// TestVisionMMProj_refusals is P8b's F5c (docs/multimodal.md, "Finishing this doc"): a GGUF mmproj on -vision is
// routed to the Qwen3.5+ tower, and refused by name when it cannot serve the model: a model that is not Qwen3.5+,
// another family's projector, a file that is not an mmproj, or a projector whose output width is another model
// size's. A matching one is accepted, the tower is marked as an mmproj, and the family's preprocessing takes the serve
// cap.
func TestVisionMMProj_refusals(t *testing.T) {
	dir := t.TempDir()
	good := filepath.Join(dir, "mmproj-qwen3.5-F16.gguf")
	writeMMProj(t, good, "qwen3vl_merger", 1024)
	gemma := filepath.Join(dir, "mmproj-gemma-3-4b-it-f16.gguf")
	writeMMProj(t, gemma, "gemma3", 2560)
	junk := filepath.Join(dir, "mmproj-junk.gguf")
	if err := os.WriteFile(junk, []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}
	if e := visionPathError(good); e != nil {
		t.Fatalf("the path check refused an mmproj (loadVisionTower routes it): %v", e)
	}
	for _, c := range []struct {
		path, mt string
		hidden   int
		want     string
	}{
		{good, "gemma3", 2560, "Qwen3.5+ models only"},
		{gemma, "qwen3_5", 1024, "projector type"},
		{junk, "qwen3_5", 1024, "mmproj"},
		{good, "qwen3_5", 4096, "another model size"},
	} {
		_, _, err := setupQwen35MMProj(c.path, c.mt, c.hidden, false)
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s with %s/%d: %v, want a refusal containing %q", filepath.Base(c.path), c.mt, c.hidden, err, c.want)
		}
	}
	tower, pp, err := setupQwen35MMProj(good, "qwen3_5_moe", 1024, false)
	if err != nil {
		t.Fatal(err)
	}
	if !tower.mmproj || tower.dir != good {
		t.Errorf("tower %+v, want an mmproj tower at %s", tower, good)
	}
	if limit := qwen3MaxImageTokens * pp.MergeSize * pp.MergeSize * pp.PatchSize * pp.PatchSize; pp.MaxPixels != limit || pp.PatchSize != 16 || pp.MergeSize != 2 {
		t.Errorf("preprocessing %+v: want the family's (patch 16, merge 2) under the serve cap %d", pp, limit)
	}

	// Some other plain file: still told what -vision takes; and a directory is left to the loaders.
	other := filepath.Join(dir, "weights.bin")
	if err := os.WriteFile(other, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if e := visionPathError(other); e == nil || !strings.Contains(e.Error(), "directory with a vision tower") {
		t.Errorf("plain file: %v", e)
	}
	if e := visionPathError(dir); e != nil {
		t.Errorf("a directory was refused by the path check: %v", e)
	}
	if e := visionPathError(filepath.Join(dir, "missing")); e != nil {
		t.Errorf("a missing path is the loaders' to judge: %v", e)
	}
}
