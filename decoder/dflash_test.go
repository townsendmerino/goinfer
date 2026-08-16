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
			// DraftBlock, which is what production calls.
			h := make([][]float32, len(blockIn))
			for i, row := range blockIn {
				h[i] = append([]float32(nil), row...)
			}
			for li := range d.layers {
				d.layer(be, &d.layers[li], fused, h)
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
