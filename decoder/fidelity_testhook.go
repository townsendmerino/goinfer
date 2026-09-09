//go:build goinfer_testhooks

// Code added for docs/task-prefill-gap.md §3's fidelity gate (a backend's fast/batched prefill
// vs its own exact/sequential path) and docs/task-peer-benchmarks.md §4's fidelity column
// (goinfer vs a peer engine) -- both want the same teacher-forced top-1 agreement and KL
// divergence scorer, so it is written once here rather than twice. Test-only hook (B-08): these
// gate correctness/quality, not production inference, so they stay off the public API surface.

package decoder

import (
	"bufio"
	"compress/gzip"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// ReadGoldenJSONForTest reads a real-checkpoint golden fixture, transparently gunzipping if path
// ends in ".gz". Convention (2026-09-08): a real-checkpoint pin script for a FIXED-resolution
// vision family (SigLIP: 896×896 for every image, no choice of a small test photo the way
// Qwen2.5-VL's dynamic resolution allows) writes its golden gzip-compressed — gzip on JSON text
// this repetitive (a `pixel_values` float array dominates the file) routinely gets 5-10×, the
// difference between "fits comfortably in git" and "GitHub warns about it": measured,
// testdata/gemma3_real_golden.json was 52.72 MB uncompressed, over GitHub's 50 MB recommendation.
// Existing small (`.json`, no `.gz`) goldens are NOT force-migrated — this is for new goldens
// where the size actually matters, not a blanket rewrite. Callers keep their own existing
// skip-if-missing handling (os.Open's error, not this function, carries that signal).
func ReadGoldenJSONForTest(path string, v any) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	var r io.Reader = f
	if strings.HasSuffix(path, ".gz") {
		gz, err := gzip.NewReader(f)
		if err != nil {
			return fmt.Errorf("decoder: gunzip %s: %w", path, err)
		}
		defer gz.Close()
		r = gz
	}
	return json.NewDecoder(r).Decode(v)
}

// PrefillLogitsForTest exposes the CPU backend's batched prompt prefill (prefillLogits) — weights
// streamed once and reused across all K positions rather than K separate M=1 passes, ~1.7-2x
// faster than the sequential loop, bit-identical to it (the seed token's math is unchanged; only
// the earlier positions' unused logits are skipped). Used by
// decoder/prefill_ref_gen_test.go so building the §3.1 CPU reference doesn't pay the sequential
// loop's cost twice over (once here, once again on the Metal side, which has no batched CPU
// equivalent to borrow). Caller controls exact vs fast attention via GOINFER_CPU_FAST_ATTENTION
// (t.Setenv("GOINFER_CPU_FAST_ATTENTION", "0") for the exact f64-accumulating kernel the reference
// needs) exactly as it would calling the production path.
func (m *Model) PrefillLogitsForTest(ctx context.Context, prompt []int, cache *KVCache) ([]float32, error) {
	return m.prefillLogits(ctx, prompt, cache)
}

// PrefillLogitsWithAdapterForTest is PrefillLogitsForTest with a compute-time LoRA adapter
// (already loaded via Model.LoadAdapter) bound to a fresh cache before prefilling — the CPU-side
// half of a resident-vs-CPU LoRA numeric parity gate (G3, docs/task-gpu-paths-2026-09.md) driven
// from a package (e.g. metal) that cannot reach KVCache.lora or Model.adapter directly, both
// unexported.
func (m *Model) PrefillLogitsWithAdapterForTest(ctx context.Context, prompt []int, adapterName string) ([]float32, error) {
	cache := m.NewCache(len(prompt))
	cache.lora = m.adapter(adapterName)
	return m.prefillLogits(ctx, prompt, cache)
}

// ResidentAdapterLayersForTest exposes residentAdapterLayers' conversion of a loaded compute-time
// adapter to the exported per-layer shape a resident backend's SetAdapter consumes — the same
// conversion generateInto calls in production, so a backend's own parity test can drive
// ResidentAdapter.SetAdapter directly (bypassing Session/generateInto's session-cache plumbing to
// isolate the backend's numerics) with exactly the deltas a real request would bind.
func (m *Model) ResidentAdapterLayersForTest(adapterName string) []ResidentAdapterLayer {
	return residentAdapterLayers(m.adapter(adapterName))
}

// PrefillLogitsVLForTest exposes prefillLogitsVL — Gemma 3's bidirectional-image-block CPU
// prefill GenerateVL drives — the Gemma-3 twin of PrefillLogitsQwenVLForTest, same reason.
func (m *Model) PrefillLogitsVLForTest(ctx context.Context, ids []int, imageFeats []float32, imgPos, imgLen int, cache *KVCache) ([]float32, error) {
	return m.prefillLogitsVL(ctx, ids, imageFeats, imgPos, imgLen, cache)
}

// PrefillLogitsQwenVLForTest exposes prefillLogitsQwenVL — the bidirectional-image-block CPU
// prefill GenerateQwenVL drives — so a cross-package real-checkpoint gate (gap 0, docs/
// multimodal.md) can build a real image's CPU-computed KVCache directly, without going through
// GenerateQwenVL's channel-only public API (which exposes sampled tokens, not per-step logits —
// unusable for a per-step cosine comparison once a resident hybrid decode's own quantization
// noise, an orthogonal and pre-existing property, would otherwise compound through greedy
// sampling and swamp the signal this gate actually needs).
func (m *Model) PrefillLogitsQwenVLForTest(ctx context.Context, ids []int, imageFeats []float32, imgPos, imgLen int, mropePos [][3]int, cache *KVCache) ([]float32, error) {
	return m.prefillLogitsQwenVL(ctx, ids, imageFeats, imgPos, imgLen, mropePos, cache)
}

// MRopePositionsForTest exposes mropePositions — Qwen2.5-VL's per-token (t,h,w) rotary position
// triples from the image grid — so a cross-package test can build the same cache.mropeDelta the
// production GenerateQwenVL path computes, for the same reason as PrefillLogitsQwenVLForTest.
func MRopePositionsForTest(ids []int, imageToken int, gridTHW [][3]int, merge int) ([][3]int, error) {
	return mropePositions(ids, imageToken, gridTHW, merge)
}

// MRopeDeltaForTest exposes a KVCache's m-RoPE decode-position delta (rope.go's mropeDelta —
// scalar decode past the prefill rotates at seqPos+delta), set by prefillLogitsQwenVL. A
// cross-package resident-decode test needs this to compute the same ropePos GenerateQwenVL's
// production decode loop does (gap 0, docs/multimodal.md).
func (c *KVCache) MRopeDeltaForTest() int { return c.mropeDelta }

// NearTieHardFailPct is the bar NearTieArgmaxForTest hard-fails at -- the same 3% every existing
// near-tie gate in this tree already uses inline (cuda/realforward_test.go's argmaxF comparison,
// gpu/kv_i8_parity_test.go), named here so a new gate cites the rule instead of retyping the
// literal.
const NearTieHardFailPct = 0.03

// NearTieArgmaxForTest reproduces the 3%-near-tie rule cuda/realforward_test.go's argmaxF
// comparison established: comparing two logit vectors' argmax, a flip is a defect only if the
// REFERENCE's own margin between its pick and the candidate's pick exceeds NearTieHardFailPct of
// the reference's logit range -- smaller gaps are quant/reassociation noise, not a real
// preference change. gapPct is always computed (0 when they agree), so a caller can report the
// worst gap seen across a run even on ticks that don't hard-fail.
func NearTieArgmaxForTest(refLogits, candLogits []float32) (agree bool, gapPct float64, hardFail bool) {
	refArg, candArg := argmax(refLogits), argmax(candLogits)
	if refArg == candArg {
		return true, 0, false
	}
	lo, hi := refLogits[0], refLogits[0]
	for _, v := range refLogits {
		if v < lo {
			lo = v
		}
		if v > hi {
			hi = v
		}
	}
	gap := float64(refLogits[refArg]-refLogits[candArg]) / (float64(hi-lo) + 1e-9)
	return false, gap, gap > NearTieHardFailPct
}

// TeacherForcedTop1AgreementForTest measures how faithfully an engine reproduces a reference
// continuation WITHOUT the cascade a free-running greedy comparison carries, where one early
// near-tie flip makes every later token diverge and the score collapses to "how long before the
// first flip" instead of "how good is the engine at each position on its own". candLogits[i] is
// the engine's output at continuation position i when fed the reference's own tokens as context
// through position i-1 (teacher-forced, not autoregressive on the engine's own output);
// refTokens[i] is the token the reference continuation actually placed at position i. Reports
// the fraction of positions where the engine's argmax equals the reference token, and the first
// position that disagrees (-1 if none). Returns 0, -1 if the slices are empty or mismatched in
// length -- a caller error, not a measurement of zero agreement.
func TeacherForcedTop1AgreementForTest(candLogits [][]float32, refTokens []int) (agreementRate float64, firstDivergence int) {
	firstDivergence = -1
	n := len(candLogits)
	if n == 0 || n != len(refTokens) {
		return 0, firstDivergence
	}
	agree := 0
	for i, lg := range candLogits {
		if argmax(lg) == refTokens[i] {
			agree++
		} else if firstDivergence == -1 {
			firstDivergence = i
		}
	}
	return float64(agree) / float64(n), firstDivergence
}

// KLDivergenceForTest computes KL(p || q) in nats between two logit vectors, after converting
// each to a probability distribution the same way sampling does (softmaxStable, temperature 1).
// p is the reference/exact distribution and q the candidate/approximate one, so the result reads
// as "how much information is lost approximating p with q" -- the §3 gate's "reported, not
// gating" KL-vs-exact figure. Terms where p is ~0 are skipped rather than evaluated: the limit of
// p*log(p/q) as p->0 is 0 regardless of q, and evaluating it risks NaN from log(0).
func KLDivergenceForTest(pLogits, qLogits []float32) float64 {
	p := softmaxStable(pLogits, 1)
	q := softmaxStable(qLogits, 1)
	var kl float64
	for i, pi := range p {
		if pi < 1e-12 {
			continue
		}
		kl += pi * math.Log(pi/(q[i]+1e-300))
	}
	return kl
}

// PrefillGateProseFiles are real prose read at run time — not scripts/prompts.json's word-
// repetition filler, which docs/task-prefill-gap.md §0 rules out for anything content-dependent
// ("the fidelity gate (§3) uses prose"). Ten distinct real technical documents from this repo,
// chosen only for being real, sizeable (each encodes to well over 3900 tokens on its own, so no
// prompt needs repeating to reach the deepest K), and stable — not for their content, the same
// reasoning metal/spec_prefill_regression_test.go's readRepoCorpus gives for reading real
// repository source instead of a short hand-written corpus.
//
// Paths are relative to a package directory one level under the repo root (as metal/'s and
// decoder/'s own test packages both are), so the same list resolves identically from either —
// this is shared between metal/prefill_gate_test.go (Metal arms) and
// decoder/prefill_ref_gen_test.go (the CPU f32-activation reference, §3.1) precisely so the two
// runs score the SAME ten prompts.
var PrefillGateProseFiles = []string{
	"../docs/audit-2026-09-02.md",
	"../docs/QUEUE.md",
	"../docs/queue-engineering.md",
	"../docs/ollama-chase.md",
	"../docs/benchmarks.md",
	"../docs/parity-coverage-policy.md",
	"../docs/task-w4a8-neon-bandwidth.md",
	"../docs/legacy-benchmarks.md",
	"../docs/task-zeno-compare.md",
	"../docs/queue-release.md",
}

// PrefillGateProseFilesB is prompt set B (docs/task-prefill-gap.md §4 L1's fresh-prompt decision
// run, 2026-09-09): ten more real repo documents, disjoint from set A above, each verified >3900
// tokens against S's own tokenizer (queue-correctness.md, the brief's own tenth candidate, was
// dropped at 1571 tokens — task-gpu-paths-2026-09.md substitutes, at 48027). LIVE paths, same
// "resolves from decoder/ or metal/" convention as set A — but the gate itself reads the SNAPSHOT
// under testdata/prefill-gate-prose-b/ (see PrefillGatePromptSet), not these live paths, so a run
// stays reproducible as these documents keep changing. Kept here only as the record of what the
// snapshot was populated FROM and when.
var PrefillGateProseFilesB = []string{
	"../docs/task-attention-decode-cost.md",
	"../docs/task-moe-streaming.md",
	"../docs/completed/queue-performance.md",
	"../docs/ARCHITECTURE.md",
	"../docs/how-inference-works.md",
	"../docs/task-peer-benchmarks.md",
	"../docs/task-recompute-audit.md",
	"../docs/task-gpu-paths-2026-09.md",
	"../docs/cuda-backend.md",
	"../docs/completed/audit-2026-08-05.md",
}

// PrefillGatePromptSet selects the SNAPSHOT prompt files for the L1 gate, per
// GOINFER_PREFILL_GATE_PROMPTS ("" or "a" = set A, "b" = set B). Returns the set's label (for
// output-path/log labelling, so set A and set B runs never collide or get confused) and the
// snapshot paths themselves — testdata/prefill-gate-prose-<label>/<basename>, resolved with the
// same "../testdata/..." convention PrefillGateProseFiles already documents, so decoder/'s and
// metal/'s own test packages both resolve it identically.
//
// SNAPSHOTS, NOT THE LIVE PATHS ABOVE. docs/QUEUE.md and docs/benchmarks.md (both in set A) change
// most days; task-gpu-paths-2026-09.md (set B) changed within this very session. A gate whose
// prompt content silently drifts between Phase A (the reference) and a later Phase B re-run, or
// between two Phase B re-runs, is not reproducible — the snapshot is taken once, per set, and the
// gate always reads it, so re-running the gate later scores the SAME prompts even if the source
// docs have since moved on.
func PrefillGatePromptSet() (label string, files []string) {
	label = strings.ToLower(strings.TrimSpace(os.Getenv("GOINFER_PREFILL_GATE_PROMPTS")))
	if label != "b" {
		label = "a"
	}
	return label, PrefillGatePromptSetFor(label)
}

// PrefillGatePromptSetFor returns one NAMED set's snapshot files regardless of
// GOINFER_PREFILL_GATE_PROMPTS — for a caller that needs a specific set explicitly (e.g.
// re-scoring set A's stored results alongside a set-B decision run, task-prefill-gap.md §4 L1)
// rather than "whichever set the environment currently selects". label other than "b" means "a".
func PrefillGatePromptSetFor(label string) (files []string) {
	live := PrefillGateProseFiles
	if label == "b" {
		live = PrefillGateProseFilesB
	} else {
		label = "a"
	}
	files = make([]string, len(live))
	for i, f := range live {
		files[i] = "../testdata/prefill-gate-prose-" + label + "/" + filepath.Base(f)
	}
	return files
}

// PrefillGateProseIDsForTest reads f, encodes it with tk, and returns at least minTokens ids
// (repeating the same real content if one file is somehow too short for a future larger K, rather
// than padding with filler — a repeated real paragraph is still content-dependent, unlike
// scripts/prompts.json's "the the the").
func PrefillGateProseIDsForTest(t *testing.T, tk *tokenizer.Tokenizer, f string, minTokens int) []int {
	t.Helper()
	raw, err := os.ReadFile(f)
	if err != nil {
		t.Fatalf("read prose seed %s: %v", f, err)
	}
	text := string(raw)
	ids, err := tk.Encode(text, true)
	if err != nil {
		t.Fatalf("encode prose seed %s: %v", f, err)
	}
	for len(ids) < minTokens {
		more, err := tk.Encode(text, false)
		if err != nil {
			t.Fatalf("encode prose seed %s: %v", f, err)
		}
		ids = append(ids, more...)
	}
	return ids
}

// WritePrefillReferenceForTest serializes docs/task-prefill-gap.md §3.1's CPU f32-activation
// reference for a later cross-process read (ReadPrefillReferenceForTest) — Phase A
// (decoder/prefill_ref_gen_test.go, its own process, CPU only) writes these; Phase B
// (metal/prefill_gate_ref_test.go) reads them back to score both Metal arms against a reference
// neither of them is. Layout, all little-endian: int32 vocab, int32 continuationN, seedLogits
// [vocab]float32, refTokens [continuationN]int32, then continuationN rows of [vocab]float32
// (refLogits). This is a private, single-machine, single-session scratch format — not versioned
// or exported for reuse beyond this pair, which is why it carries no header/magic beyond its two
// size fields.
func WritePrefillReferenceForTest(path string, seedLogits []float32, refTokens []int, refLogits [][]float32) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	vocab := int32(len(seedLogits))
	n := int32(len(refTokens))
	if err := binary.Write(w, binary.LittleEndian, vocab); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, n); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, seedLogits); err != nil {
		return err
	}
	toks32 := make([]int32, len(refTokens))
	for i, v := range refTokens {
		toks32[i] = int32(v)
	}
	if err := binary.Write(w, binary.LittleEndian, toks32); err != nil {
		return err
	}
	for _, row := range refLogits {
		if int32(len(row)) != vocab {
			return os.ErrInvalid
		}
		if err := binary.Write(w, binary.LittleEndian, row); err != nil {
			return err
		}
	}
	return w.Flush()
}

// ReadPrefillReferenceForTest is WritePrefillReferenceForTest's reader. See that function for the
// layout.
func ReadPrefillReferenceForTest(path string) (seedLogits []float32, refTokens []int, refLogits [][]float32, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, nil, err
	}
	defer f.Close()
	r := bufio.NewReader(f)
	var vocab, n int32
	if err = binary.Read(r, binary.LittleEndian, &vocab); err != nil {
		return nil, nil, nil, err
	}
	if err = binary.Read(r, binary.LittleEndian, &n); err != nil {
		return nil, nil, nil, err
	}
	seedLogits = make([]float32, vocab)
	if err = binary.Read(r, binary.LittleEndian, seedLogits); err != nil {
		return nil, nil, nil, err
	}
	toks32 := make([]int32, n)
	if err = binary.Read(r, binary.LittleEndian, toks32); err != nil {
		return nil, nil, nil, err
	}
	refTokens = make([]int, n)
	for i, v := range toks32 {
		refTokens[i] = int(v)
	}
	refLogits = make([][]float32, n)
	for i := range refLogits {
		row := make([]float32, vocab)
		if err = binary.Read(r, binary.LittleEndian, row); err != nil {
			return nil, nil, nil, err
		}
		refLogits[i] = row
	}
	return seedLogits, refTokens, refLogits, nil
}
