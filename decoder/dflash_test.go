package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"strconv"
	"testing"

	"github.com/townsendmerino/aikit/embed"
	"path/filepath"
	"strings"
)

// P10 kill-gate 1 (docs/spec/08): the Go DFlash trunk must match the upstream reference
// on dumped fixtures BEFORE any acceptance measurement. This is the gate 05 cleared too
// late — a structurally-correct-looking head sat at alpha~0.3 for weeks because nothing
// compared it to the reference at the tensor level.
//
// The fixture is the reference's own output, not a reimplementation of it:
// scripts/pin_dflash_trace.py runs z-lab/Qwen3-4B-DFlash-b16's shipped dflash.py
// (unmodified, MIT) and dumps the drafter's INPUTS (fused_context, block_in) alongside
// every layer output. So this test feeds the reference's inputs to our forward and
// compares layer by layer — a mismatch localizes to a layer instead of surfacing as
// "the logits are wrong".
//
// Fixtures: testdata/dflash_qwen3_4b_golden.json (stats + drafted ids, committed) and
// testdata/dflash_qwen3_4b_ref.safetensors (full f32 tensors, committed by an explicit
// .gitignore exception — it is the reference's output, not weights, and costs ~24 GB of
// downloads to recreate).
//
// TIER: T3-in-practice, NOT a CI gate. The drafter WEIGHTS (GOINFER_DFLASH_F32, 2.1 GB)
// cannot be committed, so this skips without them — labelled per
// docs/parity-coverage-policy.md's rule that a committed golden over an uncommitted
// checkpoint is a T3 wearing a T1's clothes. Regenerate the weights with
// scripts/convert_dflash_f32.py.
//
// THE GATE IS FALSIFIABLE, checked rather than assumed (2026-08-15): causal-instead-of-
// bidirectional block, norming the fused context like the block, roping q at the
// block-local position, and dropping the per-head k_norm are each REJECTED by it. A
// first-run cosine of exactly 1.0 is precisely when that check is worth running.
const (
	dflashRefPath    = "../testdata/dflash_qwen3_4b_ref.safetensors"
	dflashGoldenPath = "../testdata/dflash_qwen3_4b_golden.json"
)

type dflashGolden struct {
	Drafter string `json:"drafter"`
	Target  string `json:"target"`
	Traces  []struct {
		Name           string `json:"name"`
		PromptIDs      []int  `json:"prompt_ids"`
		BlockSize      int    `json:"block_size"`
		TargetLayerIDs []int  `json:"target_layer_ids"`
		MaskTokenID    int    `json:"mask_token_id"`
		AnchorToken    int    `json:"anchor_token"`
		BlockIDs       []int  `json:"block_ids"`
		DraftedIDs     []int  `json:"drafted_ids"`
		Tensors        map[string]struct {
			Shape []int `json:"shape"`
		} `json:"tensors"`
	} `json:"traces"`
}

// TestDFlash_referenceParity is the gate. It runs the trunk on the reference's own
// fused_context/block_in and requires every layer output — and the final trunk output —
// to match to f32 tolerance.
func TestDFlash_referenceParity(t *testing.T) {
	dir := assetPath(t, "GOINFER_DFLASH_F32")
	raw, err := os.ReadFile(dflashGoldenPath)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden at %s — run scripts/pin_dflash_trace.py", dflashGoldenPath)
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g dflashGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	ref, err := embed.OpenSafetensorsMmap(dflashRefPath)
	if err != nil {
		t.Skipf("no reference tensors at %s (%v) — run scripts/pin_dflash_trace.py", dflashRefPath, err)
	}
	defer ref.Close()

	d, err := LoadDFlashDrafter(dir)
	if err != nil {
		t.Fatalf("LoadDFlashDrafter(%s): %v", dir, err)
	}
	defer d.Close()

	// Descriptor: the loader must agree with the reference on the protocol constants.
	tr0 := g.Traces[0]
	if d.BlockSize() != tr0.BlockSize {
		t.Errorf("block_size = %d, want %d", d.BlockSize(), tr0.BlockSize)
	}
	if d.MaskTokenID() != tr0.MaskTokenID {
		t.Errorf("mask_token_id = %d, want %d", d.MaskTokenID(), tr0.MaskTokenID)
	}
	if got := d.TargetLayerIDs(); !intsEqual(got, tr0.TargetLayerIDs) {
		t.Errorf("target_layer_ids = %v, want %v", got, tr0.TargetLayerIDs)
	}

	be := &cpuBackend{}
	for _, tr := range g.Traces {
		t.Run(tr.Name, func(t *testing.T) {
			ctxLen := tr.Tensors["fused_context"].Shape[0]
			fused := readRows(t, ref, tr.Name+"/fused_context", ctxLen, d.hidden)
			blockIn := readRows(t, ref, tr.Name+"/block_in", tr.BlockSize, d.hidden)

			// Run the trunk layer by layer so a mismatch names its layer. Mirrors
			// DraftBlockCtx, which is what production calls — including the context
			// cache, so the cached path is what the parity gate actually proves.
			cctx := d.NewContext()
			d.ExtendContext(be, cctx, fused)
			h := make([][]float32, len(blockIn))
			for i, row := range blockIn {
				h[i] = append([]float32(nil), row...)
			}
			for li := range d.layers {
				d.layer(be, &d.layers[li], cctx.k[li], cctx.v[li], h)
				want := readRows(t, ref, tr.Name+"/layer_out."+strconv.Itoa(li), tr.BlockSize, d.hidden)
				cos, maxAbs := compareRows(h, want)
				t.Logf("layer_out.%d: cosine %.8f maxAbs %.3e", li, cos, maxAbs)
				if cos < 1-1e-5 {
					t.Fatalf("layer %d diverges: cosine %.8f (maxAbs %.3e) — later layers are noise, fix this one",
						li, cos, maxAbs)
				}
			}

			// Whole-trunk path, including the final norm.
			out, err := d.DraftBlock(be, fused, blockIn)
			if err != nil {
				t.Fatalf("DraftBlock: %v", err)
			}
			want := readRows(t, ref, tr.Name+"/trunk_out", tr.BlockSize, d.hidden)
			cos, maxAbs := compareRows(out, want)
			t.Logf("trunk_out: cosine %.8f maxAbs %.3e (ctx=%d block=%d)", cos, maxAbs, ctxLen, tr.BlockSize)
			if cos < 1-1e-5 {
				t.Errorf("trunk output cosine %.8f < %.8f (maxAbs %.3e)", cos, 1-1e-5, maxAbs)
			}
		})
	}
}

// readRows loads a [rows, cols] f32 tensor from the reference dump as row slices.
func readRows(t *testing.T, st *embed.SafetensorsFile, name string, rows, cols int) [][]float32 {
	t.Helper()
	flat, err := st.TensorF32(name, rows, cols)
	if err != nil {
		t.Fatalf("reference tensor %q: %v", name, err)
	}
	out := make([][]float32, rows)
	for i := range rows {
		out[i] = flat[i*cols : (i+1)*cols]
	}
	return out
}

// compareRows returns the cosine over all elements and the largest absolute difference.
func compareRows(got, want [][]float32) (cos, maxAbs float64) {
	var dot, na, nb float64
	for i := range got {
		for j := range got[i] {
			a, b := float64(got[i][j]), float64(want[i][j])
			dot += a * b
			na += a * a
			nb += b * b
			if d := math.Abs(a - b); d > maxAbs {
				maxAbs = d
			}
		}
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb)), maxAbs
}

// TestDFlash_targetEndToEnd is gate 1's second half: the LOGIT comparison. The trunk gate
// above feeds the reference's own inputs, which proves the trunk in isolation but says
// nothing about the two ends DFlash borrows from the target — the embedding that builds
// the block and the LM head that turns trunk output into draft tokens. This runs the whole
// path on goinfer's own Qwen3-4B: ForwardCapture -> fuse -> trunk -> target lm_head, and
// requires the drafted token ids to match the reference exactly.
//
// It is the first test where OUR target's hidden states (not HF's) drive the drafter, so
// it also exercises the capture seam's layer convention: target_layer_ids name layer
// OUTPUTS, and ForwardCapture's `layers` uses the same after-layer-l indexing.
func TestDFlash_targetEndToEnd(t *testing.T) {
	requireHeavyModel(t)
	ddir := assetPath(t, "GOINFER_DFLASH_F32")
	tdir := assetPath(t, "GOINFER_QWEN3_4B")
	raw, err := os.ReadFile(dflashGoldenPath)
	if err != nil {
		t.Skipf("no golden (%v) — run scripts/pin_dflash_trace.py", err)
	}
	var g dflashGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}

	d, err := LoadDFlashDrafter(ddir)
	if err != nil {
		t.Fatalf("LoadDFlashDrafter: %v", err)
	}
	defer d.Close()
	m, err := Load(tdir, Options{}) // f32 target — the reference dump is f32
	if err != nil {
		t.Fatalf("Load(%s): %v", tdir, err)
	}
	defer m.Close()
	if m.w.arch.HiddenDim != d.hidden {
		t.Fatalf("target hidden %d != drafter hidden %d", m.w.arch.HiddenDim, d.hidden)
	}

	for _, tr := range g.Traces {
		t.Run(tr.Name, func(t *testing.T) {
			// 1. Capture the target's hidden states at the drafter's tap layers, one
			//    committed position at a time, and concatenate per position.
			cache := m.NewCache(len(tr.PromptIDs) + tr.BlockSize)
			ctxCat := make([][]float32, 0, len(tr.PromptIDs))
			var logits []float32
			for _, id := range tr.PromptIDs {
				lg, hidden, err := m.ForwardCapture(id, cache, d.TargetLayerIDs())
				if err != nil {
					t.Fatalf("ForwardCapture: %v", err)
				}
				row := make([]float32, 0, len(hidden)*d.hidden)
				for _, h := range hidden {
					row = append(row, h...)
				}
				ctxCat = append(ctxCat, row)
				logits = lg
			}

			// 2. The anchor is the target's own greedy next token — the same thing the
			//    reference seeds slot 0 with.
			anchor := argmax(logits)
			if anchor != tr.AnchorToken {
				t.Fatalf("anchor token = %d, want %d (target disagrees with the reference before the drafter runs)",
					anchor, tr.AnchorToken)
			}

			// 3. Fuse, build the block, run the trunk, apply the TARGET's head.
			fused, err := d.FuseContext(m.be, ctxCat)
			if err != nil {
				t.Fatalf("FuseContext: %v", err)
			}
			ids := make([]int, d.BlockSize())
			for i := range ids {
				ids[i] = d.MaskTokenID()
			}
			ids[0] = anchor
			if !intsEqual(ids, tr.BlockIDs) {
				t.Fatalf("block ids = %v, want %v", ids, tr.BlockIDs)
			}
			trunk, err := d.DraftBlock(m.be, fused, m.DrafterEmbedBlock(ids))
			if err != nil {
				t.Fatalf("DraftBlock: %v", err)
			}
			got := make([]int, 0, len(trunk)-1)
			for _, h := range trunk[1:] { // slot 0 is the anchor, never predicted
				got = append(got, argmax(m.DrafterHeadLogits(h)))
			}

			t.Logf("drafted got  = %v", got)
			t.Logf("drafted want = %v", tr.DraftedIDs)
			match := 0
			for i := range got {
				if i < len(tr.DraftedIDs) && got[i] == tr.DraftedIDs[i] {
					match++
				}
			}
			t.Logf("%d/%d drafted ids match the reference", match, len(tr.DraftedIDs))
			if !intsEqual(got, tr.DraftedIDs) {
				t.Errorf("drafted ids differ from the reference (%d/%d match)", match, len(tr.DraftedIDs))
			}
		})
	}
}

// TestDFlash_refusesV2 pins the refusal added after the P15 step-(0)/(2) audit.
//
// It is a CONFIG-ONLY fixture on purpose: the danger this guards is that a DFlash 2 checkpoint
// is field-compatible with the v1 loader, so the check has to fire before any tensor is read.
// Measured on the real incoai/Qwen3.8-27B-DFlash2 (2026-08-20): 76.2% of its tensors are
// v1-shaped and every config key this loader reads is present, so it loaded WITHOUT ERROR and
// silently discarded 914,309,120 bytes of dynamic-conv + candidate-selector weights.
//
// Why that mattered enough to gate: DFlash verify is lossless, so the failure produces correct
// tokens at a worse acceptance rate — slower than v1, with nothing in the logs. A wrong answer
// gets noticed; a silent 20% throughput loss does not.
func TestDFlash_refusesV2(t *testing.T) {
	dir := t.TempDir()
	// The released v2 config's shape, trimmed to what the loader reads. conv_kernel_size and
	// selector_rank are the markers; everything else here is what makes it LOOK like v1.
	cfg := `{
	  "architectures": ["DFlash2DraftModel"],
	  "hidden_size": 5120, "num_hidden_layers": 5, "num_attention_heads": 32,
	  "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 17408,
	  "rms_norm_eps": 1e-06, "vocab_size": 248320,
	  "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
	  "dflash_config": {"block_size": 8, "mask_token_id": 248070,
	    "target_layer_ids": [5,19,33,47,61],
	    "conv_kernel_size": 2, "conv_group_size": 16,
	    "selector_rank": 256, "selector_top_k": 16}
	}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfg), 0o644); err != nil {
		t.Fatal(err)
	}
	// No model.safetensors: the refusal must happen from config alone. If this ever fails with a
	// "read weights" error instead, the check has drifted after the tensor load and a real v2
	// checkpoint would get 3.8 GB of the way in before being told no.
	_, err := LoadDFlashDrafter(dir)
	if err == nil {
		t.Fatal("v1 loader ACCEPTED a DFlash 2 config — it would silently drop the conv and " +
			"selector weights and draft worse than v1, with no error anywhere")
	}
	for _, want := range []string{"DFlash 2", "conv_kernel_size=2", "selector_rank=256", "P15"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("refusal should name %q so the reader knows WHICH checkpoint and where the\n"+
				"work is tracked; got: %v", want, err)
		}
	}
	// And a v1 config (same file minus the two markers) must still get PAST this check — a
	// refusal that also rejects v1 would be a worse bug than the one it fixes.
	v1 := strings.Replace(cfg, `"conv_kernel_size": 2, "conv_group_size": 16,`, "", 1)
	v1 = strings.Replace(v1, `"selector_rank": 256, "selector_top_k": 16`, `"unused": 0`, 1)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(v1), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadDFlashDrafter(dir); err != nil && strings.Contains(err.Error(), "DFlash 2") {
		t.Errorf("a v1 config was refused as v2 — the marker check is too broad: %v", err)
	}
}
