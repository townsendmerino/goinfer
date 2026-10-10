//go:build goinfer_testhooks

// Test-only hooks for the prefill fidelity gate (docs/completed/task-prefill-gap.md §3: a backend's fast/batched prefill
// vs its own exact path) and the peer-benchmark fidelity column (docs/tasks/task-peer-benchmarks.md §4: goinfer vs a
// peer engine). Both want the same teacher-forced top-1 agreement and KL scorer, so it is written once here. They gate
// correctness and quality, not production inference, so they stay off the public API surface.

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

// ReadGoldenJSONForTest decodes a real-checkpoint golden fixture into v, gunzipping transparently when path ends in
// ".gz"; a bare .json is read unchanged. Goldens over about 1 MB are committed as `.json.gz` (CLAUDE.md § Tests). A
// missing file returns os.Open's error, which callers use for skip-if-missing.
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

// PrefillLogitsForTest exposes the CPU backend's batched prompt prefill (prefillLogits): weights stream once across all
// positions, bit-identical to K sequential M=1 passes (only earlier positions' unused logits are skipped). The attention
// kernel follows GOINFER_CPU_FAST_ATTENTION as in production; set it to "0" for the exact f64-accumulating kernel a
// reference needs.
func (m *Model) PrefillLogitsForTest(ctx context.Context, prompt []int, cache *KVCache) ([]float32, error) {
	return m.prefillLogits(ctx, prompt, cache)
}

// PrefillLogitsWithAdapterForTest is PrefillLogitsForTest with the compute-time LoRA adapter adapterName (loaded via
// Model.LoadAdapter) bound to a fresh cache: the CPU half of a resident-vs-CPU LoRA parity gate in a package (metal)
// that cannot reach KVCache.lora or Model.adapter.
func (m *Model) PrefillLogitsWithAdapterForTest(ctx context.Context, prompt []int, adapterName string) ([]float32, error) {
	cache := m.NewCache(len(prompt))
	cache.lora = m.adapter(adapterName)
	return m.prefillLogits(ctx, prompt, cache)
}

// ResidentAdapterLayersForTest exposes residentAdapterLayers, the conversion generateInto applies to a loaded adapter, so
// a backend parity test can call ResidentAdapter.SetAdapter directly with exactly the deltas a request would bind.
func (m *Model) ResidentAdapterLayersForTest(adapterName string) []ResidentAdapterLayer {
	return residentAdapterLayers(m.adapter(adapterName))
}

// PrefillLogitsVLForTest exposes prefillLogitsVL — Gemma 3's bidirectional-image-block CPU
// prefill GenerateVL drives — the Gemma-3 twin of PrefillLogitsQwenVLForTest, same reason.
func (m *Model) PrefillLogitsVLForTest(ctx context.Context, ids []int, imageFeats []float32, imgPos, imgLen int, cache *KVCache) ([]float32, error) {
	return m.prefillLogitsVL(ctx, ids, imageFeats, imgPos, imgLen, cache)
}

// PrefillLogitsVLSpansForTest exposes prefillLogitsVLSpans (S11, several images) for a backend's multi-block gate.
func (m *Model) PrefillLogitsVLSpansForTest(ctx context.Context, ids []int, spans []ImageSpan, imageFeats []float32, cache *KVCache) ([]float32, error) {
	return m.prefillLogitsVLSpans(ctx, ids, spans, imageFeats, cache, false)
}

// PrefillLogitsVLCausalSpansForTest is PrefillLogitsVLSpansForTest with the image tokens causal (Pixtral, S10).
func (m *Model) PrefillLogitsVLCausalSpansForTest(ctx context.Context, ids []int, spans []ImageSpan, imageFeats []float32, cache *KVCache) ([]float32, error) {
	return m.prefillLogitsVLSpans(ctx, ids, spans, imageFeats, cache, true)
}

// PrefillLogitsQwenVLForTest exposes prefillLogitsQwenVL, the bidirectional-image-block CPU prefill GenerateQwenVL
// drives, so a cross-package gate can build a real image's CPU KVCache and read per-step logits (GenerateQwenVL's
// channel API exposes only sampled tokens).
func (m *Model) PrefillLogitsQwenVLForTest(ctx context.Context, ids []int, imageFeats []float32, imgPos, imgLen int, mropePos [][3]int, cache *KVCache) ([]float32, error) {
	return m.prefillLogitsQwenVL(ctx, ids, imageFeats, imgPos, imgLen, mropePos, cache)
}

// PrefillLogitsGemma4VLForTest exposes prefillLogitsGemma4VL, Gemma 4's sequential, causal-only CPU prefill that
// GenerateGemma4VL drives, for the same reason as the hooks above (per-step logits).
func (m *Model) PrefillLogitsGemma4VLForTest(ctx context.Context, ids []int, imageFeats []float32, imgPos, imgLen int, cache *KVCache) ([]float32, error) {
	return m.prefillLogitsGemma4VL(ctx, ids, imageFeats, imgPos, imgLen, cache)
}

// PrefillLogitsGemma4VLBidirectionalForTest exposes prefillLogitsGemma4VLBidirectional, Gemma 4's batched,
// blockwise-masked CPU prefill for use_bidirectional_attention: "vision" checkpoints. It builds the KVCache
// GenerateGemma4VL's resident branch uploads; the twin of PrefillLogitsGemma4VLForTest.
func (m *Model) PrefillLogitsGemma4VLBidirectionalForTest(ctx context.Context, ids []int, imageFeats []float32, imgPos, imgLen int, cache *KVCache) ([]float32, error) {
	return m.prefillLogitsGemma4VLBidirectional(ctx, ids, imageFeats, imgPos, imgLen, cache)
}

// MRopePositionsForTest exposes mropePositions, Qwen2.5-VL's per-token (t,h,w) rotary positions from the image grid, so
// a cross-package test can reproduce GenerateQwenVL's cache.mropeDelta.
func MRopePositionsForTest(ids []int, imageToken int, gridTHW [][3]int, merge int) ([][3]int, error) {
	return mropePositions(ids, imageToken, gridTHW, merge)
}

// MRopeDeltaForTest exposes the KVCache's m-RoPE decode-position delta (set by prefillLogitsQwenVL): scalar decode past
// the prefill rotates at seqPos+delta. A cross-package resident-decode test needs it to reproduce GenerateQwenVL's rope
// position.
func (c *KVCache) MRopeDeltaForTest() int { return c.mropeDelta }

// NearTieHardFailPct is the bar NearTieArgmaxForTest hard-fails at, as a fraction of the reference's logit range; read
// NearTieArgmaxForTest before changing it.
const NearTieHardFailPct = 0.03

// NearTieArgmaxForTest applies the 3% near-tie rule to two logit vectors' argmax: a flip is a defect only if the
// reference's margin between its pick and the candidate's pick exceeds NearTieHardFailPct of the reference's logit
// range; smaller gaps are quantization or reassociation noise. gapPct is always computed (0 on agreement) so a caller
// can report the worst gap of a run.
//
// The 3% is in logit-range units, not probability units: on Qwen's typical range it is about 2 nats, roughly the
// separation between probabilities 0.85 and 0.11. The same threshold is used inline by cuda/realforward_test.go and
// gpu/kv_i8_parity_test.go; changing it means changing all three together.
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

// TeacherForcedTop1AgreementForTest measures how faithfully an engine reproduces a reference continuation without the
// cascade of a free-running greedy comparison, where one early near-tie flip makes every later token diverge.
// candLogits[i] is the engine's output at continuation position i when fed the reference's own tokens through position
// i-1 (teacher-forced); refTokens[i] is the reference's token there. It returns the fraction of positions where the
// engine's argmax equals the reference token and the first position that disagrees (-1 if none).
//
// Empty or length-mismatched input returns 0, 0: a caller error, not a measurement. firstDivergence is 0 there, never
// -1, so a gate testing only firstDivergence == -1 cannot pass on a comparison that never ran.
func TeacherForcedTop1AgreementForTest(candLogits [][]float32, refTokens []int) (agreementRate float64, firstDivergence int) {
	n := len(candLogits)
	if n == 0 || n != len(refTokens) {
		return 0, 0
	}
	firstDivergence = -1
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

// KLDivergenceForTest computes KL(p || q) in nats between two logit vectors, each softmaxed as sampling does
// (softmaxStable, temperature 1); p is the reference distribution, q the candidate. Terms where p is ~0 are skipped: the
// limit of p*log(p/q) as p->0 is 0 regardless of q, and evaluating it risks NaN from log(0).
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

// PrefillGateProseFiles are real prose documents from this repo, read at run time, for content-dependent prefill
// fidelity (not scripts/prompts.json's word-repetition filler, which docs/completed/task-prefill-gap.md §0 rules out
// for that). Each encodes to well over 3900 tokens, so no prompt repeats to reach the deepest K. Paths are relative to a
// package directory one level under the repo root, so the decoder and metal test packages resolve the same list: the
// CPU reference (decoder/prefill_ref_gen_test.go) and the Metal arms (metal/prefill_gate_test.go) score the same ten
// prompts.
var PrefillGateProseFiles = []string{
	"../docs/completed/audit-2026-09-02.md", // moved from ../docs/audit-2026-09-02.md when archived
	"../docs/QUEUE.md",
	"../docs/queue-engineering.md",
	"../docs/ollama-chase.md",
	"../docs/benchmarks.md",
	"../docs/parity-coverage-policy.md",
	"../docs/completed/task-w4a8-neon-bandwidth.md", // moved from ../docs/task-w4a8-neon-bandwidth.md when archived
	"../docs/legacy-benchmarks.md",
	"../docs/completed/task-zeno-compare.md", // moved from ../docs/task-zeno-compare.md when archived
	"../docs/queue-release.md",
}

// PrefillGateProseFilesB is prompt set B: ten more real repo documents, disjoint from set A, each over 3900 tokens, with
// the same path convention. The gate reads the snapshot under testdata/prefill-gate-prose-b/ (see PrefillGatePromptSet),
// not these live paths; this list records what the snapshot was populated from.
var PrefillGateProseFilesB = []string{
	"../docs/completed/task-attention-decode-cost.md", // moved from ../docs/task-attention-decode-cost.md when archived
	"../docs/completed/task-moe-streaming.md",         // moved from ../docs/task-moe-streaming.md via docs/tasks/ to docs/completed/
	"../docs/completed/queue-performance.md",
	"../docs/ARCHITECTURE.md",
	"../docs/how-inference-works.md",
	"../docs/tasks/task-peer-benchmarks.md", // moved from ../docs/task-peer-benchmarks.md, still LIVE
	"../docs/tasks/task-recompute-audit.md", // moved from ../docs/task-recompute-audit.md, still LIVE
	"../docs/tasks/task-gpu-paths-2026-09.md",
	"../docs/cuda-backend.md",
	"../docs/completed/audit-2026-08-05.md",
}

// PrefillGatePromptSet selects the snapshot prompt files for the L1 gate per GOINFER_PREFILL_GATE_PROMPTS ("" or "a" =
// set A, "b" = set B). It returns the set's label, so set A and set B outputs never collide, and the snapshot paths
// testdata/prefill-gate-prose-<label>/<basename>, resolved with the "../testdata/..." convention PrefillGateProseFiles
// documents.
//
// The gate reads snapshots, not the live documents: several change most days, and a gate whose prompt content drifts
// between the reference phase and a later re-run is not reproducible. Each set is snapshotted once.
func PrefillGatePromptSet() (label string, files []string) {
	label = strings.ToLower(strings.TrimSpace(os.Getenv("GOINFER_PREFILL_GATE_PROMPTS")))
	if label != "b" {
		label = "a"
	}
	return label, PrefillGatePromptSetFor(label)
}

// PrefillGatePromptSetFor returns the named set's snapshot files regardless of GOINFER_PREFILL_GATE_PROMPTS, for a
// caller that needs a specific set (for example re-scoring set A's stored results beside a set-B run). A label other
// than "b" means "a".
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

// PrefillGateProseIDsForTest reads f, encodes it with tk, and returns at least minTokens ids, repeating the same real
// content if the file is too short rather than padding with filler (a repeated real paragraph stays content-dependent).
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

// WritePrefillReferenceForTest serializes the CPU f32-activation reference (docs/completed/task-prefill-gap.md §3.1)
// for a cross-process read by ReadPrefillReferenceForTest: the CPU-only phase (decoder/prefill_ref_gen_test.go) writes
// it and the Metal phase (metal/prefill_gate_ref_test.go) scores both arms against it. Layout, little-endian: int32
// vocab, int32 continuationN, seedLogits [vocab]float32, refTokens [continuationN]int32, then continuationN rows of
// [vocab]float32 (refLogits). A private scratch format with no version or magic beyond the two size fields.
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

// W8F3Ref is the CPU f32 reference for the metal W8 F3′ gate (docs/tasks/task-metal-int8-2026-10.md): the 8 prompts,
// the f32 model's greedy continuation (the tokens every arm is teacher-forced on) and its logits at every position.
// TestW8F3Reference_write writes it where the f32 model fits; metal's TestW8Native_F3amended_closerToF32 reads it
// through GOINFER_W8_F3_REF_IN.
type W8F3Ref struct {
	Model, Arch    string
	Prompts, Toks  [][]int
	Logits         [][][]float32 // [prompt][position][vocab]
	PromptLen, Pos int
}

// PrefillLogitsAudioForTest exposes prefillLogitsAudio (Qwen3-ASR's soft-token prefill) for cross-package gates that need the logits and the cache, not GenerateAudio's token channel.
func (m *Model) PrefillLogitsAudioForTest(ctx context.Context, ids []int, feats []float32, pos, n int, cache *KVCache) ([]float32, error) {
	return m.prefillLogitsAudio(ctx, ids, feats, pos, n, cache)
}
