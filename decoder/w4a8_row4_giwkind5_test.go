package decoder

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/giw"
)

// TestSerializedInt4Weights_kind5RepackedOnly_matchesCanonical is L2's round-trip
// gate (docs/task-int4-layout-2026-09.md): a .giw written via
// SerializeWeightsForTarget(GIWTargetCPUArm64) must bake kind 5 (row4-only, NO
// canonical arrays) for every eligible int4 tensor, round-trip with Int4()'s ok
// FALSE and IsInt4() TRUE for each of them (repacked-only, aikit audit M-22), and
// decode BIT-IDENTICAL to (a) the same model loaded straight from GGUF and (b) a
// plain kind-3 .giw of the same weights — three paths, one answer, the same
// standard TestSerializedInt4Weights_row4Kind_matchesCanonical already holds kind
// 4 to. Also checks the on-disk size story: kind 5 swaps canonical for row4
// bytes (same length per tensor, docs/task-w4a8-neon-bandwidth.md), so a kind-5
// bundle should be close to kind-3's size — NOT ~2x like kind 4, which carries
// both.
func TestSerializedInt4Weights_kind5RepackedOnly_matchesCanonical(t *testing.T) {
	path := prequantGGUF(t)

	mGGUF, err := Load(path, Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load gguf (int4, row4-eligible): %v", err)
	}
	var eligible, int4Count int
	for _, wm := range mGGUF.w.matmulWeights() {
		if _, _, _, ok := wm.Int4(); !ok {
			continue
		}
		int4Count++
		if _, _, ok := wm.Int4Row4(); ok {
			eligible++
		}
	}
	if int4Count == 0 {
		t.Fatal("fixture has no int4 weights at all — test is not exercising anything")
	}
	if eligible == 0 {
		t.Skip("no int4 weight in this fixture is row4-eligible (shape/DotProd) — nothing for kind 5 to carry")
	}

	blob3, err := SerializeWeights(mGGUF.w, "kind5-test-k3")
	if err != nil {
		t.Fatalf("SerializeWeights (kind 3): %v", err)
	}
	blob5, err := SerializeWeightsForTarget(mGGUF.w, "kind5-test-k5", GIWTargetCPUArm64)
	if err != nil {
		t.Fatalf("SerializeWeightsForTarget(cpu-arm64): %v", err)
	}
	// Swapped, not doubled: kind 5 costs at most a rounding error more than kind 3
	// (both hold exactly one int4 representation per tensor), unlike kind 4's ~2x.
	if d := len(blob5) - len(blob3); d < -4096 || d > 4096 {
		t.Errorf("kind-5 bundle (%d bytes) should be within a few bytes of kind-3 (%d bytes) — got delta %d", len(blob5), len(blob3), d)
	}
	t.Logf("kind-3 %.1f MB, kind-5 %.1f MB (delta %+d bytes)", float64(len(blob3))/1e6, float64(len(blob5))/1e6, len(blob5)-len(blob3))

	w3, err := LoadSerializedWeights(blob3)
	if err != nil {
		t.Fatalf("LoadSerializedWeights (kind 3): %v", err)
	}
	w5, err := LoadSerializedWeights(blob5)
	if err != nil {
		t.Fatalf("LoadSerializedWeights (kind 5): %v", err)
	}
	srcWeights := mGGUF.w.matmulWeights()
	gotWeights := w5.matmulWeights()
	if len(srcWeights) != len(gotWeights) {
		t.Fatalf("matmulWeights() count changed across round-trip: src=%d got=%d", len(srcWeights), len(gotWeights))
	}
	var repackedOnly int
	for i, wm := range gotWeights {
		if _, _, _, ok := srcWeights[i].Int4(); !ok {
			continue
		}
		_, _, srcRow4Eligible := srcWeights[i].Int4Row4()
		if !srcRow4Eligible {
			// Not eligible for row4 at all (shape), or excluded by weightMatKind3Only
			// (paged/not-yet-scoped) — either way this tensor must have round-tripped kind
			// 3 (canonical still present).
			if _, _, _, ok := wm.Int4(); !ok {
				t.Fatalf("weight %d: not row4-eligible/not kind-5-scoped, but round-tripped WITHOUT canonical bytes", i)
			}
			continue
		}
		if !wm.IsInt4() {
			t.Fatalf("weight %d: row4-eligible int4 tensor lost IsInt4() across kind-5 round-trip", i)
		}
		if _, _, _, ok := wm.Int4(); ok {
			t.Fatalf("weight %d: row4-eligible int4 tensor round-tripped WITH canonical bytes present — kind 5 must be canonical-absent", i)
		}
		if _, _, ok := wm.Int4Row4(); !ok {
			t.Fatalf("weight %d: kind-5 round-trip has no row4 layout either — lost the tensor", i)
		}
		repackedOnly++
	}
	if repackedOnly != eligible {
		t.Fatalf("kind-5 .giw: %d/%d int4 weights are repacked-only after round-trip, want %d (all row4-eligible ones)", repackedOnly, int4Count, eligible)
	}
	t.Logf(".giw kind-5 round-trip: %d/%d int4 weights repacked-only (canonical absent), as expected", repackedOnly, int4Count)

	mK3, err := NewModel(w3, "cpu")
	if err != nil {
		t.Fatalf("new model from kind-3 weights: %v", err)
	}
	mK5, err := NewModel(w5, "cpu")
	if err != nil {
		t.Fatalf("new model from kind-5 weights: %v", err)
	}

	prompt := []int{1, 2, 3, 4, 5, 6, 7, 8}
	tokGGUF := greedyFirst(t, mGGUF, prompt)
	tokK3 := greedyFirst(t, mK3, prompt)
	tokK5 := greedyFirst(t, mK5, prompt)
	if tokGGUF != tokK3 || tokGGUF != tokK5 {
		t.Fatalf("greedy token differs across dispatch paths: gguf(row4-in-RAM)=%d kind3(canonical-only)=%d kind5(repacked-only-from-disk)=%d",
			tokGGUF, tokK3, tokK5)
	}
	t.Logf("identical greedy token across GGUF/kind-3/kind-5 dispatch: %d", tokGGUF)
}

// TestLoad_kind5UnderBackendNeedingCanonical_declinesLoudlyAtLoad is L2's decline
// gate: a kind-5 .giw (built for a cpu-arm64 target — repacked-only, no canonical
// bytes) loaded with a backend that needs canonical (Backend:"metal") must fail
// AT decoder.Load, with a named, actionable error — never a silent fallback to
// CPU/staged (withResidency's own decline is soft/logged, not fatal, and is NOT
// what protects this case; decoder.Load's own post-load check is).
func TestLoad_kind5UnderBackendNeedingCanonical_declinesLoudlyAtLoad(t *testing.T) {
	path := prequantGGUF(t)

	mGGUF, err := Load(path, Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load gguf (int4, row4-eligible): %v", err)
	}
	var eligible bool
	for _, wm := range mGGUF.w.matmulWeights() {
		if _, _, ok := wm.Int4Row4(); ok {
			eligible = true
			break
		}
	}
	if !eligible {
		t.Skip("no int4 weight in this fixture is row4-eligible — nothing for kind 5 to carry")
	}

	giwPath := filepath.Join(t.TempDir(), "kind5-decline-test.giw")
	blob5, err := SerializeWeightsForTarget(mGGUF.w, "kind5-decline-test", GIWTargetCPUArm64)
	if err != nil {
		t.Fatalf("SerializeWeightsForTarget(cpu-arm64): %v", err)
	}
	if err := os.WriteFile(giwPath, giw.Write(blob5, nil), 0o644); err != nil {
		t.Fatalf("write %s: %v", giwPath, err)
	}

	// Sanity: the same file loads fine under Backend:"cpu" (needs no canonical).
	if mCPU, err := Load(giwPath, Options{Backend: "cpu"}); err != nil {
		t.Fatalf("kind-5 .giw failed to load under Backend:\"cpu\": %v", err)
	} else {
		mCPU.Close()
	}

	_, err = Load(giwPath, Options{Backend: "metal"})
	if err == nil {
		t.Fatal("Load succeeded for a kind-5 (repacked-only) .giw under Backend:\"metal\" — want a loud decline naming the reason")
	}
	got := err.Error()
	for _, want := range []string{"row4-only", "kind 5", "Backend", "\"metal\"", "canonical"} {
		if !strings.Contains(got, want) {
			t.Errorf("decline error %q does not mention %q", got, want)
		}
	}
	t.Logf("decline error: %s", got)
}

// TestGiwReaderWeightMat_kind5DeclinesWhenThisCoreCannotUseRow4Only pins the OTHER
// named decline (docs/task-int4-layout-2026-09.md's L2): a kind-5 tensor whose
// shape this core's Int4Row4Usable rejects (rows not a multiple of 4 — the same
// constraint repackRow4ForEmit enforces at WRITE time) must fail readWeightMat
// with a named error, not a panic. Hand-built bytes, bypassing the normal
// write-time eligibility gate on purpose — the only way to observe this branch on
// a box where write-time and read-time eligibility are otherwise the same check.
func TestGiwReaderWeightMat_kind5DeclinesWhenThisCoreCannotUseRow4Only(t *testing.T) {
	const rows, cols, group = 3, 32, 32 // rows%4 != 0 — Int4Row4Usable must reject this
	wantQ4 := rows * ((cols + 1) / 2)
	wantScales := rows * ((cols + group - 1) / group)

	wr := &giwWriter{}
	wr.raw([]byte{5}) // kind 5
	wr.u32(uint32(rows))
	wr.u32(uint32(cols))
	wr.u32(uint32(group))
	wr.raw([]byte{0}) // w8a8
	wr.f32(make([]float32, wantScales))
	wr.bytesField(make([]byte, wantQ4))

	r := &giwReader{data: wr.buf}
	wm := r.weightMat()
	if r.err == nil {
		t.Fatalf("kind-5 tensor with rows=%d (not a multiple of 4) loaded without error — want a decline", rows)
	}
	if wm.Rows() != 0 {
		t.Errorf("declined weightMat should be the zero value, got Rows()=%d", wm.Rows())
	}
	got := r.err.Error()
	for _, want := range []string{"row4-only", "kind 5", "this core"} {
		if !strings.Contains(got, want) {
			t.Errorf("decline error %q does not mention %q", got, want)
		}
	}
	t.Logf("decline error: %s", got)
}
