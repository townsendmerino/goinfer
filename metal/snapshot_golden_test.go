//go:build darwin

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
	"strconv"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// prodEmbedRow fills dst with token id's LAYER-0 INPUT exactly as production builds it: decoder.embedResident dequantizes the
// embedding row and multiplies by the arch's embed scale (Gemma's √hidden). Mirroring it here is what makes the golden a
// reference for the SHIPPED computation rather than for an entry point production never calls (audit G-02).
func prodEmbedRow(r *resident, id int, dst []float32) {
	r.embed.Row(id, dst)
	if r.embedScale > 1 {
		for i := range dst {
			dst[i] *= r.embedScale
		}
	}
}

// TestMetalEmbedScale_forwardMatchesForwardEmb is the regression gate for G-02's live-correctness half, the one the snapshot
// golden structurally could not provide because it drove the buggy path on BOTH sides of its own comparison. Forward(id,pos)
// must feed layer 0 the same √hidden-scaled embedding row as production (ForwardEmb via decoder.embedResident); on
// gemma4-dense-scaled a raw row is a different input, so the test fails on the old code and passes on the new, which is the
// property a regression test has to have. It is the cheapest statement of the invariant: the two entry points must agree on every
// admitted family.
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

// TestMetalSnapshotGolden is the ABSOLUTE STORED REFERENCE the Metal gate suite otherwise lacks.
//
// Every other Metal gate is self-consistent or tolerance-based: `paged ≡ non-paged` compares one kernel against itself under
// different residency (any change to the kernel moves BOTH arms identically), and `Metal-vs-CPU` is cosine/tolerance (small
// movements pass by construction, unavoidable given the f16 scale gap). That class (a reduction-WIDTH change: float sum is
// non-associative and wired to threadgroup width, see tgReduce* in model.go; a fused-kernel rewrite; a different accumulation
// order; a moved scale-application point) is invisible to those gates. Only a reference that does NOT move when the code moves
// catches it.
//
// This decodes a FIXED token sequence through the Metal resident path on tiny models, to depths PAST the reduction widths (128
// and 256), and byte-compares the logits (sha256) to a committed golden. It is self-referential: it detects that something
// moved, not which side is correct. It is MACHINE-PINNED (Metal float results are deterministic run-to-run and across code
// versions on a given GPU, but not guaranteed identical across chip families), so it goes red on a legitimate improvement and on
// a hardware change (different Mac). Regenerate with the refresh flag after verifying the change is intentional:
//
//	GOINFER_UPDATE_GOLDENS=1 go test -run TestMetalSnapshotGolden ./metal/
//
// A red here is a REAL drift and must be read as one: a standing "EXPECTED TO FAIL" note would turn the suite's only absolute gate
// into noise. The checkpoint call drives ForwardEmb with the production-scaled embedding row (prodEmbedRow), not Forward with a
// raw one (audit G-02): that changes `gemma4-dense-scaled`'s stream (EmbedScale = √hidden), while `mixtral-tiny` has no embed
// scale and its entries must NOT move; if they do, something other than G-02 changed and a re-bake should be refused pending
// investigation.
//
// Runs on every `go test` for the two COMMITTED models (mixtral-tiny, llama-attnfa-tiny). gemma4-dense-scaled (449 MB) is not
// committed (over GitHub's practical push limit): it is a local-only fixture regenerated deterministically by
// `scripts/pin_gemma4_dense_scaled.py`, and when it is absent this test skips it and still checks the other two (keyed
// comparison, not positional).
//
// Coverage (by fixture construction: the body hashes logits and asserts no dispatch): mixtral-tiny is full-causal (attention
// softmax denom over >256 keys, the width coupling at multi-iteration depth) + rmsnorm_quant; gemma4-dense-scaled covers
// rmsnorm_f32 + qk_norm; llama-attnfa-tiny covers `attention_fa`, DEFAULT ON past depth 1024 (attnFADepthFloor,
// metal/model.go): the other two fixtures fail canUseAttnFA's head_dim==128 guard (8 and 256) and can never dispatch it. Its
// checkpoints straddle the floor exactly (1022 declines, 1023 engages) so an off-by-one at the boundary is caught. Union =
// every pinned-width reduction kernel plus `attention_fa` this build DISPATCHES. attention_f32 is NOT covered and cannot be:
// model.go hard-wires `r.kvF32 = false` and builds kv_store_f32/attention_f32 only inside `if r.kvF32`, so nothing dispatches
// them (N-28; §A2-Metal).
func TestMetalSnapshotGolden(t *testing.T) {
	models := []struct {
		dir, quant  string
		checkpoints map[int]bool
		maxD        int
	}{
		// full-causal: attention denom past width; rmsnorm_quant
		{"../testdata/mixtral-tiny", "int8int8", map[int]bool{130: true, 260: true, 320: true}, 320},
		// sandwich: rmsnorm_f32, qk_norm (N-13: NOT attention_f32 — N-28 above)
		{"../testdata/gemma4-dense-scaled", "int4", map[int]bool{130: true, 260: true, 320: true}, 320},
		// R2: attention_fa, DEFAULT ON past attnFADepthFloor=1024 (metal/model.go). Every other
		// fixture here has head_dim != 128 (mixtral-tiny: 8, gemma4-dense-scaled: 256), so
		// canUseAttnFA's hd==128 guard declines on both, regardless of depth — neither can ever
		// cover this kernel. llama-attnfa-tiny (scripts/pin_llama_attnfa_tiny.py) is a plain dense
		// GQA Llama shaped to clear every other guard too (no sandwich/postOnly/parallelBlock/
		// attnSink/kvI8/lora/MoE/DeltaNet/qGate/window — see canUseAttnFA). Checkpoints straddle
		// the floor exactly: curNKeys = pos+1, so pos=1022 (curNKeys=1023) is the last declining
		// position and pos=1023 (curNKeys=1024) is the first engaging one. 900 is a shipped-kernel-only
		// control below the floor; 1100 confirms the engaged kernel stays stable past the boundary, not
		// just at it.
		{"../testdata/llama-attnfa-tiny", "int4", map[int]bool{900: true, 1022: true, 1023: true, 1100: true}, 1100},
	}
	ids := []int{1, 7, 42, 100, 5, 200, 13, 88, 3, 71, 9, 17, 60, 200, 33, 2} // fixed, arbitrary valid ids

	got := snapGolden{Env: snapEnv{OS: macOSVersion()}}
	skipped := 0
	for _, mm := range models {
		name := filepath.Base(mm.dir)
		// gemma4-dense-scaled (449 MB) is NOT committed (over GitHub's practical push limit); scripts/pin_gemma4_dense_scaled.py
		// (deterministic, seed 0) regenerates it, and every other consumer of it skips gracefully when it is absent. Skipping one model
		// here must not cost the other (committed, always-available) fixtures' coverage: see the schema-tolerant comparison below, keyed
		// by (Model,Quant,Depth) rather than positional/count equality.
		if _, err := os.Stat(mm.dir + "/model.safetensors"); err != nil {
			t.Logf("skip %s: no fixture (%v) — run scripts/pin_gemma4_dense_scaled.py to regenerate it locally", name, err)
			skipped++
			continue
		}
		m, err := decoder.Load(mm.dir, decoder.Options{Quant: mm.quant})
		if err != nil {
			t.Fatalf("load %s: %v", mm.dir, err)
		}
		defer m.Close()
		H, _, _, _, _, _, V := m.Dims()
		r, err := buildResident(m)
		if err != nil {
			t.Fatalf("BuildResident %s: %v", mm.dir, err)
		}
		defer r.Close() // N-10: exercise the Close path so this test also guards the teardown-leak regressions

		if got.Env.GPU == "" {
			got.Env.GPU = r.d.Name()
		}
		tok := ids[0]
		emb := make([]float32, H)
		for pos := 0; pos <= mm.maxD; pos++ {
			if mm.checkpoints[pos] {
				// Drive the PRODUCTION entry point (audit G-02): decoder.embedResident does the lookup + embed scale and calls ForwardEmb.
				// Hashing r.Forward instead would pin a stream production never produces (it skips the √hidden scale on gemma4) and leave a
				// regression at the embed→layer-0 seam invisible to the suite's one absolute reference.
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
		if skipped > 0 {
			t.Fatalf("refusing to write the golden with %d model(s) skipped (missing fixture) — "+
				"a partial re-bake would silently DROP those models' entries for every machine that "+
				"reads this golden afterward. Get every fixture in `models` present locally first "+
				"(see the skip log above for which is missing and how to regenerate it), then re-run.", skipped)
		}
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

	// Keyed, not positional/count: a model skipped here (fixture missing locally) must not make
	// this test blind to drift in the models that DID run — it just means the golden has entries
	// this run has nothing to compare them against, which is expected and reported, not an error.
	key := func(e snapEntry) string { return e.Model + "|" + e.Quant + "|" + strconv.Itoa(e.Depth) }
	wantByKey := make(map[string]snapEntry, len(want.Entries))
	for _, e := range want.Entries {
		wantByKey[key(e)] = e
	}
	mism, matched, notInGolden := 0, 0, 0
	for _, ge := range got.Entries {
		we, ok := wantByKey[key(ge)]
		if !ok {
			notInGolden++
			t.Errorf("NEW checkpoint %s q=%s depth=%d has no golden entry — schema changed; regenerate with GOINFER_UPDATE_GOLDENS=1", ge.Model, ge.Quant, ge.Depth)
			continue
		}
		matched++
		if ge != we {
			mism++
			t.Errorf("DRIFT %s q=%s depth=%d: argmax %d→%d  sha %s→%s",
				ge.Model, ge.Quant, ge.Depth, we.Argmax, ge.Argmax, we.SHA256[:12], ge.SHA256[:12])
		}
	}
	if skipped > 0 {
		t.Logf("NOTE: %d model(s) skipped (fixture missing locally) — %d golden entries for them were not exercised this run, not a failure", skipped, len(want.Entries)-matched-notInGolden)
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
	t.Logf("Metal snapshot: %d/%d checkpoints byte-identical to golden on %s / macOS %s (%d model(s) run, %d skipped)", matched, len(want.Entries), got.Env.GPU, got.Env.OS, len(models)-skipped, skipped)
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
