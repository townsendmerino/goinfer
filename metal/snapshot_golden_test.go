package metal

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestMetalSnapshotGolden is the ABSOLUTE STORED REFERENCE the Metal gate suite otherwise lacks.
//
// Every other Metal gate is self-consistent or tolerance-based: `paged ≡ non-paged` compares one
// kernel against itself under different residency (any change to the kernel moves BOTH arms
// identically → passes), and `Metal-vs-CPU` is cosine/tolerance (small movements pass by
// construction, unavoidable given the f16 scale gap). That whole class — a reduction-WIDTH change
// (float sum is non-associative, wired to threadgroup width; see tgReduce* in model.go), a
// fused-kernel rewrite, a different accumulation order, a moved scale-application point — is
// invisible to those gates. Only a reference that does NOT move when the code moves catches it.
//
// This decodes a FIXED token sequence through the Metal resident path on tiny committed models, to
// depths PAST the reduction widths (128 and 256), and byte-compares the logits (sha256) to a committed
// golden. It is self-referential: it detects that something moved, not which side is correct — exactly
// what's missing. It is MACHINE-PINNED (Metal float results are deterministic run-to-run and across
// code versions on a given GPU, but not guaranteed identical across chip families). It WILL go red on
// a legitimate improvement — that's the point; regenerate with the refresh flag after verifying the
// change is intentional (the same goldens-refresh discipline the CUDA track uses):
//
//	GOINFER_UPDATE_GOLDENS=1 go test -run TestMetalSnapshotGolden ./metal/
//
// Regenerate too on a hardware change (different Mac). Runs on every `go test` (tiny committed models,
// no heavy-model dependency). Coverage: mixtral-tiny is full-causal (attention softmax denom over
// >256 keys → the width coupling at multi-iteration depth) + rmsnorm_quant; gemma4-dense-scaled covers
// attention_f32 + rmsnorm_f32 + qk_norm. Union = every pinned-width reduction kernel. See §A2-Metal.
func TestMetalSnapshotGolden(t *testing.T) {
	models := []struct{ dir, quant string }{
		{"../testdata/mixtral-tiny", "int8int8"},    // full-causal: attention denom past width; rmsnorm_quant
		{"../testdata/gemma4-dense-scaled", "int4"}, // sandwich: attention_f32, rmsnorm_f32, qk_norm
	}
	checkpoints := map[int]bool{130: true, 260: true, 320: true} // past 128 and 256
	const maxD = 320
	ids := []int{1, 7, 42, 100, 5, 200, 13, 88, 3, 71, 9, 17, 60, 200, 33, 2} // fixed, arbitrary valid ids

	var got []snapEntry
	for _, mm := range models {
		name := filepath.Base(mm.dir)
		m, err := decoder.Load(mm.dir, decoder.Options{Quant: mm.quant})
		if err != nil {
			t.Fatalf("load %s: %v", mm.dir, err)
		}
		_, _, _, _, _, _, V := m.Dims()
		r, err := BuildResident(m)
		if err != nil {
			t.Fatalf("BuildResident %s: %v", mm.dir, err)
		}
		tok := ids[0]
		for pos := 0; pos <= maxD; pos++ {
			if checkpoints[pos] {
				lg := r.Forward(tok, pos)
				h := sha256.Sum256(f32ToBytes(lg))
				got = append(got, snapEntry{Model: name, Quant: mm.quant, Depth: pos, Argmax: argmaxF(lg), SHA256: hex.EncodeToString(h[:])})
				tok = argmaxF(lg)
			} else if pos+1 < len(ids) {
				r.ForwardArgmax(tok, pos)
				tok = ids[pos+1]
			} else {
				tok = int(r.ForwardArgmax(tok, pos))
			}
			if tok <= 0 || tok >= V {
				tok = 1
			}
		}
	}

	if os.Getenv("GOINFER_UPDATE_GOLDENS") != "" {
		b, _ := json.MarshalIndent(got, "", "  ")
		if err := os.WriteFile(snapGoldenPath, append(b, '\n'), 0o644); err != nil {
			t.Fatalf("write golden: %v", err)
		}
		t.Logf("WROTE %d snapshot entries → %s (regenerated; verify the change was intentional)", len(got), snapGoldenPath)
		return
	}

	wantB, err := os.ReadFile(snapGoldenPath)
	if err != nil {
		t.Fatalf("read golden (%v) — first-time generate with: GOINFER_UPDATE_GOLDENS=1 go test -run TestMetalSnapshotGolden ./metal/", err)
	}
	var want []snapEntry
	if err := json.Unmarshal(wantB, &want); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	if len(want) != len(got) {
		t.Fatalf("golden has %d entries, run produced %d — schema changed; regenerate with GOINFER_UPDATE_GOLDENS=1", len(want), len(got))
	}
	mism := 0
	for i := range got {
		if got[i] != want[i] {
			mism++
			t.Errorf("DRIFT %s q=%s depth=%d: argmax %d→%d  sha %s→%s",
				got[i].Model, got[i].Quant, got[i].Depth, want[i].Argmax, got[i].Argmax, want[i].SHA256[:12], got[i].SHA256[:12])
		}
	}
	if mism > 0 {
		t.Fatalf("Metal snapshot DRIFT: %d/%d checkpoints moved. Something in the Metal decode path changed the bits "+
			"(the gate the cosine/paged gates can't provide). If the change is intentional and verified, regenerate: "+
			"GOINFER_UPDATE_GOLDENS=1 go test -run TestMetalSnapshotGolden ./metal/", mism, len(got))
	}
	t.Logf("Metal snapshot: %d checkpoints byte-identical to golden (mixtral-tiny + gemma4-dense-scaled, depths past 128/256)", len(got))
}

type snapEntry struct {
	Model  string `json:"model"`
	Quant  string `json:"quant"`
	Depth  int    `json:"depth"`
	Argmax int    `json:"argmax"`
	SHA256 string `json:"sha256"`
}

const snapGoldenPath = "../testdata/metal_snapshot_golden.json"

func f32ToBytes(v []float32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
	}
	return b
}
