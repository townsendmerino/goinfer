package metal

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
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
// ⚠ RE-BAKE OWED (audit G-02, fixed on Linux where this suite cannot run). The checkpoint call now
// drives ForwardEmb with the production-scaled embedding row instead of Forward with a raw one, and
// Forward/ForwardArgmax now apply the arch embed scale. For `gemma4-dense-scaled` (EmbedScale =
// √hidden) that CHANGES the hashed stream — the stored entries pin the pre-fix, non-production
// computation, so this test is EXPECTED TO FAIL until re-baked on the Mac:
//
//	GOINFER_UPDATE_GOLDENS=1 go test -run TestMetalSnapshotGolden ./metal/
//
// `mixtral-tiny` has no embed scale and its entries must NOT move; if they do, something other than
// G-02 changed and the re-bake should be refused pending investigation.
//
// Regenerate too on a hardware change (different Mac). Runs on every `go test` (tiny committed models,
// no heavy-model dependency). Coverage: mixtral-tiny is full-causal (attention softmax denom over
// >256 keys → the width coupling at multi-iteration depth) + rmsnorm_quant; gemma4-dense-scaled covers
// attention_f32 + rmsnorm_f32 + qk_norm. Union = every pinned-width reduction kernel. See §A2-Metal.
// prodEmbedRow fills dst with token id's LAYER-0 INPUT exactly as production builds it:
// decoder.embedResident dequantizes the embedding row and multiplies by the arch's embed scale
// (Gemma's √hidden). Mirroring it here is what makes the golden a reference for the SHIPPED
// computation rather than for an entry point production never calls (audit G-02).
func prodEmbedRow(r *resident, id int, dst []float32) {
	r.embed.Row(id, dst)
	if r.embedScale > 1 {
		for i := range dst {
			dst[i] *= r.embedScale
		}
	}
}

// TestMetalEmbedScale_forwardMatchesForwardEmb is the regression gate for G-02's live-correctness
// half — the one the snapshot golden structurally could not provide, because it drove the buggy
// path on BOTH sides of its own comparison.
//
// Before the fix, Forward(id,pos) fed layer 0 a RAW embedding row while production (ForwardEmb via
// decoder.embedResident) fed it a √hidden-scaled one. On gemma4-dense-scaled those are different
// inputs, so this test fails on the old code and passes on the new — which is the property a
// regression test has to have. It is also the cheapest possible statement of the invariant: the
// two entry points must agree on every admitted family.
func TestMetalEmbedScale_forwardMatchesForwardEmb(t *testing.T) {
	const dir = "../testdata/gemma4-dense-scaled" // the admitted family that HAS an embed scale
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("no scaled-embed fixture at %s: %v", dir, err)
	}
	m, err := decoder.Load(dir, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("buildResident: %v", err)
	}
	if r.embedScale <= 1 {
		t.Fatalf("fixture reports embedScale %v — it cannot exercise the finding", r.embedScale)
	}
	H, _, _, _, _, _, _ := m.Dims()

	const id, pos = 42, 0
	viaID := append([]float32(nil), r.Forward(id, pos)...)
	emb := make([]float32, H)
	prodEmbedRow(r, id, emb)
	viaEmb := r.ForwardEmb(emb, pos)

	if len(viaID) != len(viaEmb) {
		t.Fatalf("logit lengths differ: %d vs %d", len(viaID), len(viaEmb))
	}
	for i := range viaID {
		if viaID[i] != viaEmb[i] {
			t.Fatalf("Forward(id) and ForwardEmb(production row) diverge at logit %d (%v vs %v) — "+
				"the id-taking entry point is skipping the arch embed scale, so every direct caller "+
				"gets wrong logits on this family", i, viaID[i], viaEmb[i])
		}
	}
}

func TestMetalSnapshotGolden(t *testing.T) {
	models := []struct{ dir, quant string }{
		{"../testdata/mixtral-tiny", "int8int8"},    // full-causal: attention denom past width; rmsnorm_quant
		{"../testdata/gemma4-dense-scaled", "int4"}, // sandwich: attention_f32, rmsnorm_f32, qk_norm
	}
	checkpoints := map[int]bool{130: true, 260: true, 320: true} // past 128 and 256
	const maxD = 320
	ids := []int{1, 7, 42, 100, 5, 200, 13, 88, 3, 71, 9, 17, 60, 200, 33, 2} // fixed, arbitrary valid ids

	got := snapGolden{Env: snapEnv{OS: macOSVersion()}}
	for _, mm := range models {
		name := filepath.Base(mm.dir)
		m, err := decoder.Load(mm.dir, decoder.Options{Quant: mm.quant})
		if err != nil {
			t.Fatalf("load %s: %v", mm.dir, err)
		}
		H, _, _, _, _, _, V := m.Dims()
		r, err := buildResident(m)
		if err != nil {
			t.Fatalf("BuildResident %s: %v", mm.dir, err)
		}
		if got.Env.GPU == "" {
			got.Env.GPU = r.d.Name()
		}
		tok := ids[0]
		emb := make([]float32, H)
		for pos := 0; pos <= maxD; pos++ {
			if checkpoints[pos] {
				// Drive the PRODUCTION entry point (audit G-02). decoder.embedResident does the
				// lookup + embed scale and calls ForwardEmb; hashing r.Forward instead pinned a
				// stream production never produces — it skipped the √hidden scale on gemma4, so
				// the suite's one absolute reference hashed a different computation than the
				// shipped path, and any regression at the embed→layer-0 seam was invisible to it.
				prodEmbedRow(r, tok, emb)
				lg := r.ForwardEmb(emb, pos)
				h := sha256.Sum256(f32ToBytes(lg))
				got.Entries = append(got.Entries, snapEntry{Model: name, Quant: mm.quant, Depth: pos, Argmax: argmaxF(lg), SHA256: hex.EncodeToString(h[:])})
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
		t.Logf("WROTE %d entries → %s  (baked on %s / macOS %s; verify the change was intentional)", len(got.Entries), snapGoldenPath, got.Env.GPU, got.Env.OS)
		return
	}

	wantB, err := os.ReadFile(snapGoldenPath)
	if err != nil {
		t.Fatalf("read golden (%v) — first-time generate with: GOINFER_UPDATE_GOLDENS=1 go test -run TestMetalSnapshotGolden ./metal/", err)
	}
	var want snapGolden
	if err := json.Unmarshal(wantB, &want); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	sameEnv := want.Env == got.Env
	if len(want.Entries) != len(got.Entries) {
		t.Fatalf("golden has %d entries, run produced %d — schema changed; regenerate with GOINFER_UPDATE_GOLDENS=1", len(want.Entries), len(got.Entries))
	}
	mism := 0
	for i := range got.Entries {
		if got.Entries[i] != want.Entries[i] {
			mism++
			t.Errorf("DRIFT %s q=%s depth=%d: argmax %d→%d  sha %s→%s",
				got.Entries[i].Model, got.Entries[i].Quant, got.Entries[i].Depth, want.Entries[i].Argmax, got.Entries[i].Argmax, want.Entries[i].SHA256[:12], got.Entries[i].SHA256[:12])
		}
	}
	if mism > 0 {
		// Branch the guidance on env — the difference between "expected on other hardware, do NOT
		// refresh" and "same box, real regression, investigate". This is what stops the reflexive
		// GOINFER_UPDATE_GOLDENS reflex from silently destroying the reference on a new Mac / OS update.
		if !sameEnv {
			t.Fatalf("Metal snapshot DRIFT — but HARDWARE/OS DIFFERS from the golden.\n"+
				"  golden baked on: %s / macOS %s\n  this run:        %s / macOS %s\n"+
				"Across-machine/OS bit-identity is NOT deliverable (MSL is recompiled by your OS Metal toolchain), so this red is EXPECTED on a different Mac or after an OS update — it is NOT a regression. "+
				"Do NOT refresh unless you are intentionally re-baselining the reference ON THIS machine; if so: GOINFER_UPDATE_GOLDENS=1 go test -run TestMetalSnapshotGolden ./metal/",
				want.Env.GPU, want.Env.OS, got.Env.GPU, got.Env.OS)
		}
		t.Fatalf("Metal snapshot DRIFT on the SAME hardware/OS the golden was baked on (%s / macOS %s): %d/%d checkpoints moved — the Metal decode bits changed (width sweep, kernel rewrite, math-mode, or a fast-math shift). "+
			"This is a REAL change the cosine/paged gates can't see. INVESTIGATE before refreshing; regenerate only once verified intentional: GOINFER_UPDATE_GOLDENS=1 go test -run TestMetalSnapshotGolden ./metal/",
			got.Env.GPU, got.Env.OS, mism, len(got.Entries))
	}
	if !sameEnv {
		t.Logf("NOTE: entries match but env metadata differs (golden %s/%s vs run %s/%s) — bits happened to coincide; consider refreshing metadata.", want.Env.GPU, want.Env.OS, got.Env.GPU, got.Env.OS)
	}
	t.Logf("Metal snapshot: %d checkpoints byte-identical to golden on %s / macOS %s (mixtral-tiny + gemma4-dense-scaled, depths past 128/256)", len(got.Entries), got.Env.GPU, got.Env.OS)
}

type snapGolden struct {
	Env     snapEnv     `json:"env"`
	Entries []snapEntry `json:"entries"`
}

type snapEnv struct {
	GPU string `json:"gpu"`
	OS  string `json:"os"`
}

type snapEntry struct {
	Model  string `json:"model"`
	Quant  string `json:"quant"`
	Depth  int    `json:"depth"`
	Argmax int    `json:"argmax"`
	SHA256 string `json:"sha256"`
}

const snapGoldenPath = "../testdata/metal_snapshot_golden.json"

// macOSVersion is the OS toolchain identity that (with the GPU) fixes the Metal bits — a mismatch
// against the golden explains an EXPECTED drift (different Mac / OS update) vs a real regression.
func macOSVersion() string {
	p, err := exec.Command("sw_vers", "-productVersion").Output()
	if err != nil {
		return "unknown"
	}
	b, _ := exec.Command("sw_vers", "-buildVersion").Output()
	return strings.TrimSpace(string(p)) + " (" + strings.TrimSpace(string(b)) + ")"
}

func f32ToBytes(v []float32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
	}
	return b
}
