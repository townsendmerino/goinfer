//go:build realckpt

// Laguna DFlash pairing — poolside/Laguna-XS.2-speculator.dflash.
//
// This is the first VENDOR-BLESSED drafter goinfer has paired with (every prior
// DFlash/DSpark pairing was third-party), and it differs structurally from those:
// it ships its OWN embed_tokens and a REDUCED-vocab lm_head (32000 rows against a
// 100352-token target) plus d2t/t2d, where z-lab's drafters borrow the target's.
// Its config is also a fourth dialect — vLLM "speculators" v0.5, with the layer
// geometry under transformer_layer_config and the taps under
// aux_hidden_state_layer_ids.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_LAGUNA_DFLASH=~/models/laguna-xs2-dflash \
//	  go test -tags realckpt ./decoder/ -run TestLagunaDFlash -v
package decoder

import (
	"context"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/tokenizer"
)

func TestLagunaDFlash_load(t *testing.T) {
	requireHeavyModel(t)
	dir := assetPath(t, "GOINFER_LAGUNA_DFLASH")

	d, err := LoadDFlashDrafter(dir)
	if err != nil {
		t.Fatalf("LoadDFlashDrafter(%s): %v", dir, err)
	}
	defer d.Close()

	// Geometry comes from transformer_layer_config, not the top level.
	g := d.DrafterGeometry()
	if g.Layers != 5 || g.Hidden != 2048 || g.NumHeads != 16 || g.NumKVHeads != 8 || g.HeadDim != 128 {
		t.Errorf("geometry = %dL/h%d/%dq/%dkv/hd%d, want 5/2048/16/8/128",
			g.Layers, g.Hidden, g.NumHeads, g.NumKVHeads, g.HeadDim)
	}
	if g.Intermediate != 8192 {
		t.Errorf("intermediate = %d, want 8192", g.Intermediate)
	}
	// Five taps, from aux_hidden_state_layer_ids.
	if ids := d.TargetLayerIDs(); len(ids) != 5 || ids[0] != 1 || ids[4] != 39 {
		t.Errorf("taps = %v, want [1 9 17 36 39]", ids)
	}
	if d.BlockSize() != 8 {
		t.Errorf("block_size = %d, want 8", d.BlockSize())
	}
	// mask_token_id is TOP-LEVEL in this dialect. Getting it wrong is silent and
	// expensive: the drafter still runs and still produces lossless output, just
	// badly — P10 measured a known-good pairing fall from 1.60x to 0.66x on exactly
	// this mistake.
	if d.MaskTokenID() != 12 {
		t.Errorf("mask_token_id = %d, want 12", d.MaskTokenID())
	}
	// The reduced-vocab head and its mapping table.
	if !d.HasOwnHead() {
		t.Fatal("HasOwnHead() = false — this drafter ships lm_head + d2t and must not borrow the target's")
	}
	if d.draftVocab != 32000 {
		t.Errorf("draftVocab = %d, want 32000", d.draftVocab)
	}
	if got := d.lmHead.Rows(); got != 32000 {
		t.Errorf("lm_head rows = %d, want 32000", got)
	}
	if got := d.embed.Rows(); got != 100352 {
		t.Errorf("embed_tokens rows = %d, want 100352 (the TARGET vocab)", got)
	}
	if len(d.d2t) != 32000 {
		t.Errorf("d2t len = %d, want 32000", len(d.d2t))
	}
	// d2t must be a strictly-usable mapping into the target vocab; the loader already
	// range-checks it, so this pins that it is not the identity (which would mean the
	// tensor was misread as zeros and every draft id would be wrong-but-in-range).
	nonZero := 0
	for _, v := range d.d2t {
		if v != 0 {
			nonZero++
		}
	}
	if nonZero == 0 {
		t.Error("d2t is all zeros — read as an identity mapping, which would silently draft wrong ids")
	}
	t.Logf("laguna DFlash: %d layers, taps %v, block %d, mask %d, draft vocab %d (%d non-identity d2t entries)",
		g.Layers, d.TargetLayerIDs(), d.BlockSize(), d.MaskTokenID(), d.draftVocab, nonZero)
}

// TestLagunaDFlash_acceptance measures what the pairing is actually worth: how many
// tokens per round the target ACCEPTS from the drafter.
//
// WHY ACCEPTANCE AND NOT SPEEDUP. Block drafting is lossless by construction — every
// emitted token is one the target's own argmax produced — so no correctness test can
// tell a good drafter from a bad one. Acceptance is the only signal, and P10 learned
// that the hard way twice (a wrong mask token turned 1.60x into 0.66x while output
// stayed perfectly valid; a drafter fed the wrong embeddings would do the same).
//
// This deliberately does NOT report a speedup. Laguna is CPU-only in goinfer today
// (FeatAttnOutputGate makes every resident backend decline it), and P10's own
// kill-gate found the DRAFT, not the verify, was the wall — a CPU draft against a
// CPU target is a different regime from the GPU-resident numbers gate 3 reported.
// Measuring wall-clock here would produce a number that says more about this box
// than about the pairing.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_LAGUNA_XS2=~/models/laguna-xs2 \
//	  GOINFER_LAGUNA_DFLASH=~/models/laguna-xs2-dflash \
//	  go test -tags realckpt ./decoder/ -run TestLagunaDFlash_acceptance -v -timeout 180m
func TestLagunaDFlash_acceptance(t *testing.T) {
	requireHeavyModel(t)
	ckpt := assetPath(t, "GOINFER_LAGUNA_XS2")
	ddir := assetPath(t, "GOINFER_LAGUNA_DFLASH")

	m, err := Load(ckpt, Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("Load target: %v", err)
	}
	defer m.Close()
	d, err := LoadDFlashDrafter(ddir)
	if err != nil {
		t.Fatalf("LoadDFlashDrafter: %v", err)
	}
	defer d.Close()

	if g := d.DrafterGeometry(); g.Hidden != m.w.arch.HiddenDim {
		t.Fatalf("drafter hidden %d != target hidden %d — the trunk consumes the target's "+
			"hidden states directly, with no projection between them", g.Hidden, m.w.arch.HiddenDim)
	}
	for _, l := range d.TargetLayerIDs() {
		if l < 0 || l >= m.w.arch.NumLayers {
			t.Fatalf("tap layer %d outside the target's %d layers", l, m.w.arch.NumLayers)
		}
	}

	const prompt = "〈|EOS|〉<system>\n\nYou are a helpful, conversationally-fluent assistant made by " +
		"Poolside. You are here to be helpful to users through natural language conversations.\n</system>\n" +
		"<user>\nWrite a Python function that reverses a linked list.\n</user>\n<assistant>\n</think>"
	tk, err := tokenizer.Load(ckpt)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	ids, err := tk.Encode(prompt, false)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	const maxNew = 64
	B := d.BlockSize()
	cache := m.NewCache(len(ids) + maxNew + B + 2)
	layers := d.TargetLayerIDs()
	var ctxCat [][]float32
	var logits []float32
	feed := func(id int) {
		lg, hidden, ferr := m.ForwardCapture(id, cache, layers)
		if ferr != nil {
			t.Fatalf("ForwardCapture: %v", ferr)
		}
		row := make([]float32, 0, len(hidden)*d.hidden)
		for _, h := range hidden {
			row = append(row, h...)
		}
		ctxCat = append(ctxCat, row)
		logits = lg
	}
	for _, id := range ids {
		feed(id)
	}

	eos := map[int]bool{}
	for _, e := range m.w.Cfg.EOSIDs() {
		eos[e] = true
	}
	rounds, acceptedTotal, generated := 0, 0, 1
	anchor := argmax(logits)
	emitted := []int{anchor}
	for generated < maxNew {
		fused, ferr := d.FuseContext(m.be, ctxCat)
		if ferr != nil {
			t.Fatalf("FuseContext: %v", ferr)
		}
		blk := make([]int, B)
		for i := range blk {
			blk[i] = d.MaskTokenID()
		}
		blk[0] = anchor
		trunk, derr := d.DraftBlock(m.be, fused, d.EmbedBlock(m, blk))
		if derr != nil {
			t.Fatalf("DraftBlock: %v", derr)
		}
		drafted := d.DraftIDs(m, trunk[1:])

		mark, markCtx := cache.Pos(), len(ctxCat)
		feed(anchor)
		accepted := 0
		next := argmax(logits)
		for i, tok := range drafted {
			if tok != next {
				break
			}
			feed(tok)
			accepted = i + 1
			next = argmax(logits)
		}
		cache.TruncateTo(mark + 1 + accepted)
		ctxCat = ctxCat[:markCtx+1+accepted]

		rounds++
		acceptedTotal += accepted
		emitted = append(emitted, drafted[:accepted]...)
		generated += accepted + 1
		anchor = next
		emitted = append(emitted, anchor)
		if eos[anchor] {
			break
		}
	}

	perRound := float64(acceptedTotal+rounds) / float64(rounds) // accepted drafts + the target's own token
	text, _ := tk.Decode(emitted)
	t.Logf("laguna DFlash acceptance: %d rounds, %d accepted drafts, %.2f tok/round "+
		"(block %d, verify width %d)", rounds, acceptedTotal, perRound, B, B)
	t.Logf("output: %q", text)

	// A pairing that accepts nothing is indistinguishable from a broken one, and both
	// look fine in the output. 1.0 tok/round means every draft was rejected — the
	// target is doing all the work and the drafter is pure overhead.
	if perRound <= 1.0 {
		t.Errorf("%.2f tok/round — no draft was EVER accepted. Losslessness hides this: the "+
			"output is still correct, so only this number can show the pairing is not working",
			perRound)
	}
}

// TestLagunaDFlash_cpuBlockSpec is the pairing end to end: real Laguna-XS.2 target, real
// poolside drafter, driven through the CPU BlockSpec with BATCHED verify.
//
// This is the one that can show a speedup. The acceptance test above verifies
// sequentially, which measures acceptance honestly but performs exactly as many target
// forwards as plain decoding; here the whole drafted block is verified in one pass, so
// accepted drafts actually cost less than the tokens they replace.
//
// It still asserts LOSSLESSNESS first. On a 33B CPU target the timing is noisy and
// machine-specific, but token-identity is exact and is the property that makes the
// speedup meaningful rather than a different (faster, worse) model.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_LAGUNA_XS2=~/models/laguna-xs2 \
//	  GOINFER_LAGUNA_DFLASH=~/models/laguna-xs2-dflash \
//	  go test -tags realckpt ./decoder/ -run TestLagunaDFlash_cpuBlockSpec -v -timeout 180m
func TestLagunaDFlash_cpuBlockSpec(t *testing.T) {
	requireHeavyModel(t)
	ckpt := assetPath(t, "GOINFER_LAGUNA_XS2")
	ddir := assetPath(t, "GOINFER_LAGUNA_DFLASH")

	m, err := Load(ckpt, Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("Load target: %v", err)
	}
	defer m.Close()
	d, err := LoadDFlashDrafter(ddir)
	if err != nil {
		t.Fatalf("LoadDFlashDrafter: %v", err)
	}
	defer d.Close()

	tk, err := tokenizer.Load(ckpt)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	const prompt = "〈|EOS|〉<system>\n\nYou are a helpful, conversationally-fluent assistant made by " +
		"Poolside. You are here to be helpful to users through natural language conversations.\n</system>\n" +
		"<user>\nWrite a Python function that reverses a linked list.\n</user>\n<assistant>\n</think>"
	ids, err := tk.Encode(prompt, false)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	const maxNew = 48
	base := make([]int, 0, maxNew)
	tPlain := time.Now()
	out, _ := m.Generate(context.Background(), ids, maxNew, SamplingParams{})
	for id := range out {
		base = append(base, id)
	}
	plainDur := time.Since(tPlain)
	if len(base) == 0 {
		t.Fatal("plain generation produced nothing")
	}

	spec, err := m.NewCPUBlockSpec(d, len(ids)+maxNew+d.BlockSize()+2)
	if err != nil {
		t.Fatalf("NewCPUBlockSpec: %v", err)
	}
	tSpec := time.Now()
	got, rounds, err := spec.Generate(ids, BlockSpecOptions{MaxTokens: maxNew})
	if err != nil {
		t.Fatalf("BlockSpec.Generate: %v", err)
	}
	specDur := time.Since(tSpec)

	perRound := 0.0
	if rounds > 0 {
		perRound = float64(len(got)) / float64(rounds)
	}
	speedup := float64(plainDur) / float64(specDur) * float64(len(base)) / float64(max(len(got), 1))
	t.Logf("laguna CPU block spec: %d tok in %d rounds (%.2f tok/round) | %v vs plain %v for %d tok | ~%.2fx",
		len(got), rounds, perRound, specDur.Round(time.Millisecond), plainDur.Round(time.Millisecond), len(base), speedup)
	text, _ := tk.Decode(got)
	t.Logf("output: %q", text)

	n := min(len(base), len(got))
	for i := range n {
		if got[i] != base[i] {
			t.Fatalf("DIVERGED at token %d: spec=%d plain=%d — wrong positions or a missed cache "+
				"rollback, not a bad draft (block drafting is lossless by construction)", i, got[i], base[i])
		}
	}
	if rounds > 0 && perRound <= 1.0 {
		t.Errorf("%.2f tok/round — no draft accepted", perRound)
	}
}
