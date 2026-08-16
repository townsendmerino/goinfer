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

// P10 kill-gate 1 for DSpark (docs/spec/08): the Go DSpark forward must match DeepSpec's own
// implementation on dumped fixtures before any acceptance claim rests on OUR code. DSpark's
// acceptance is already measured (5.76 / 5.73 / 3.04) but that was measured through DeepSpec's
// loop — this is what lets goinfer claim it.
//
// It doubles as the test of the SHARED-TRUNK claim. `decoder/dspark.go` reuses `blockTrunk`
// rather than reimplementing the forward, on the finding that DeepSpec's `_forward_backbone`
// and z-lab's `DFlashDraftModel.forward` compute the same thing. If that were wrong, the
// per-layer comparison below would diverge at layer 0 — the same code passes the DFlash
// fixture, so a DSpark failure here would localize the difference rather than hide it.
//
// Fixtures: testdata/dspark_qwen3_4b_golden.json + dspark_qwen3_4b_ref.safetensors, from
// scripts/pin_dspark_trace.py. Weights are an asset (GOINFER_DSPARK_F32) — T3-in-practice,
// not CI, exactly as the DFlash gate.
const (
	dsparkRefPath    = "../testdata/dspark_qwen3_4b_ref.safetensors"
	dsparkGoldenPath = "../testdata/dspark_qwen3_4b_golden.json"
)

type dsparkGolden struct {
	Traces []struct {
		Name           string    `json:"name"`
		PromptIDs      []int     `json:"prompt_ids"`
		BlockSize      int       `json:"block_size"`
		TargetLayerIDs []int     `json:"target_layer_ids"`
		MaskTokenID    int       `json:"mask_token_id"`
		LogitsStart    int       `json:"logits_start"`
		AnchorToken    int       `json:"anchor_token"`
		BlockIDs       []int     `json:"block_ids"`
		DraftedIDs     []int     `json:"drafted_ids"`
		Confidence     []float32 `json:"confidence_logits"`
		Tensors        map[string]struct {
			Shape []int `json:"shape"`
		} `json:"tensors"`
	} `json:"traces"`
}

func TestDSpark_referenceParity(t *testing.T) {
	dir := assetPath(t, "GOINFER_DSPARK_F32")
	raw, err := os.ReadFile(dsparkGoldenPath)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden at %s — run scripts/pin_dspark_trace.py", dsparkGoldenPath)
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g dsparkGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	ref, err := embed.OpenSafetensorsMmap(dsparkRefPath)
	if err != nil {
		t.Skipf("no reference tensors (%v) — run scripts/pin_dspark_trace.py", err)
	}
	defer ref.Close()

	d, err := LoadDSparkDrafter(dir)
	if err != nil {
		t.Fatalf("LoadDSparkDrafter(%s): %v", dir, err)
	}
	defer d.Close()

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
	if tr0.LogitsStart != 0 {
		t.Fatalf("fixture says logits_start=%d; this forward assumes 0 (all block positions predict)", tr0.LogitsStart)
	}

	be := &cpuBackend{}
	for _, tr := range g.Traces {
		t.Run(tr.Name, func(t *testing.T) {
			ctxLen := tr.Tensors["fused_context"].Shape[0]
			fused := readRows(t, ref, tr.Name+"/fused_context", ctxLen, d.hidden)
			blockIn := readRows(t, ref, tr.Name+"/block_in", tr.BlockSize, d.hidden)

			// DSpark embeds the block with its OWN table — check that first, since a wrong
			// embedding would make every later tensor wrong for an uninteresting reason.
			gotEmbed := d.EmbedBlock(tr.BlockIDs)
			if cos, maxAbs := compareRows(gotEmbed, blockIn); cos < 1-1e-6 {
				t.Fatalf("own-embedding block_in cosine %.8f (maxAbs %.3e)", cos, maxAbs)
			}

			// Trunk, layer by layer, through the SHARED blockTrunk.
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
					t.Fatalf("layer %d diverges: cosine %.8f — the shared blockTrunk does NOT reproduce DSpark here",
						li, cos)
				}
			}
			trunk, derr := d.DraftBlockCtx(be, cctx, blockIn)
			if derr != nil {
				t.Fatalf("DraftBlockCtx: %v", derr)
			}
			wantTrunk := readRows(t, ref, tr.Name+"/trunk_out", tr.BlockSize, d.hidden)
			cos, maxAbs := compareRows(trunk, wantTrunk)
			t.Logf("trunk_out: cosine %.8f maxAbs %.3e (ctx=%d block=%d)", cos, maxAbs, ctxLen, tr.BlockSize)
			if cos < 1-1e-5 {
				t.Errorf("trunk cosine %.8f < %.8f", cos, 1-1e-5)
			}

			// Markov chain + own LM head: the drafted ids must match exactly.
			ids, corrected := d.SampleBlock(be, trunk, tr.AnchorToken)
			// stored 1-D (one row of the full vocab) to keep the committed fixture small
			flatMk, err := ref.TensorF32(tr.Name+"/markov_logits_pos0", d.vocab)
			if err != nil {
				t.Fatalf("reference markov row: %v", err)
			}
			mcos, mmax := compareRows(corrected[:1], [][]float32{flatMk})
			t.Logf("markov_logits[pos0]: cosine %.8f maxAbs %.3e", mcos, mmax)
			if mcos < 1-1e-5 {
				t.Errorf("markov-corrected logits cosine %.8f < %.8f", mcos, 1-1e-5)
			}
			t.Logf("drafted got  = %v", ids)
			t.Logf("drafted want = %v", tr.DraftedIDs)
			if !intsEqual(ids, tr.DraftedIDs) {
				t.Errorf("drafted ids differ from the reference")
			}

			// Confidence head — the adaptive block-length signal.
			prev := append([]int{tr.AnchorToken}, ids[:len(ids)-1]...)
			conf := d.Confidence(be, trunk, prev)
			if conf == nil {
				t.Fatal("no confidence head loaded, but the checkpoint ships one")
			}
			for i := range conf {
				if d := math.Abs(float64(conf[i] - tr.Confidence[i])); d > 2e-2 {
					t.Errorf("confidence[%d] = %.4f, want %.4f (Δ%.4f)", i, conf[i], tr.Confidence[i], d)
				}
			}
			t.Logf("confidence got  = %v", conf)
			t.Logf("confidence want = %v", tr.Confidence)
		})
	}
}

// TestDSpark_targetEndToEnd is gate 1's second half for DSpark: the whole path on goinfer's
// OWN target, not the reference's tensors. ForwardCapture → FuseContext → shared trunk → its
// own LM head → Markov chain, with the drafted ids required to match the reference exactly.
//
// Simpler than the DFlash equivalent in one respect worth naming: DSpark ships its own
// embedding and LM head, so the only thing it borrows from the target is the hidden states.
// That makes this a clean test of the CAPTURE SEAM — if the 5 taps or the +1 layer-output
// convention were wrong, nothing else here could absorb it.
func TestDSpark_targetEndToEnd(t *testing.T) {
	requireHeavyModel(t)
	ddir := assetPath(t, "GOINFER_DSPARK_F32")
	tdir := assetPath(t, "GOINFER_QWEN3_4B")
	raw, err := os.ReadFile(dsparkGoldenPath)
	if err != nil {
		t.Skipf("no golden (%v) — run scripts/pin_dspark_trace.py", err)
	}
	var g dsparkGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}

	d, err := LoadDSparkDrafter(ddir)
	if err != nil {
		t.Fatalf("LoadDSparkDrafter: %v", err)
	}
	defer d.Close()
	m, err := Load(tdir, Options{}) // f32 — the fixture is f32
	if err != nil {
		t.Fatalf("Load(%s): %v", tdir, err)
	}
	defer m.Close()
	if m.w.arch.HiddenDim != d.hidden {
		t.Fatalf("target hidden %d != drafter hidden %d", m.w.arch.HiddenDim, d.hidden)
	}

	for _, tr := range g.Traces {
		t.Run(tr.Name, func(t *testing.T) {
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
			anchor := argmax(logits)
			if anchor != tr.AnchorToken {
				t.Fatalf("anchor = %d, want %d (the target disagrees before the drafter runs)", anchor, tr.AnchorToken)
			}

			fused, err := d.FuseContext(m.be, ctxCat)
			if err != nil {
				t.Fatalf("FuseContext: %v", err)
			}
			ids := make([]int, d.BlockSize())
			for i := range ids {
				ids[i] = d.MaskTokenID()
			}
			ids[0] = anchor
			trunk, err := d.DraftBlock(m.be, fused, d.EmbedBlock(ids))
			if err != nil {
				t.Fatalf("DraftBlock: %v", err)
			}
			got, _ := d.SampleBlock(m.be, trunk, anchor)

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
