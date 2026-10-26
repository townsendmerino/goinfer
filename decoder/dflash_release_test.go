package decoder

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// writeTinyDFlash writes a BF16 DFlash v1 checkpoint (2 layers, hidden 8) to a temp dir and returns the dir. With ownHead it
// also ships a reduced-vocab head, an embedding and d2t, as poolside's Laguna drafters do.
func writeTinyDFlash(t *testing.T, ownHead bool) string { return writeTinyDFlashAs(t, ownHead, "BF16") }

// writeTinyDFlashAs is writeTinyDFlash with every weight stored as dtype (BF16, F16 or F32); the values, k/16, are exact in each.
func writeTinyDFlashAs(t *testing.T, ownHead bool, dtype string) string {
	t.Helper()
	const h, heads, kv, hd, inter, layers, vocab, draftVocab = 8, 2, 1, 4, 16, 2, 32, 6
	dir := t.TempDir()
	head := ""
	if ownHead {
		head = fmt.Sprintf(`"vocab_size": %d, "draft_vocab_size": %d,`, vocab, draftVocab)
	}
	cfg := fmt.Sprintf(`{"hidden_size": %d, "num_hidden_layers": %d, "num_attention_heads": %d, "num_key_value_heads": %d,
	  "head_dim": %d, "intermediate_size": %d, "rms_norm_eps": 1e-6, "rope_theta": 10000, %s
	  "dflash_config": {"block_size": 4, "mask_token_id": 7, "target_layer_ids": [0, 1]}}`, h, layers, heads, kv, hd, inter, head)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfg), 0o644); err != nil {
		t.Fatal(err)
	}
	seed := 0
	bf := func(shape ...int) stTyped {
		n := 1
		for _, d := range shape {
			n *= d
		}
		var raw []byte
		for range n {
			seed++
			v := float32(seed%13+1) / 16
			switch dtype {
			case "F32":
				raw = binary.LittleEndian.AppendUint32(raw, math.Float32bits(v))
			case "F16":
				raw = binary.LittleEndian.AppendUint16(raw, linalg.F32ToF16(v))
			default:
				raw = binary.LittleEndian.AppendUint16(raw, linalg.F32ToBF16(v))
			}
		}
		return stTyped{dtype, shape, raw}
	}
	ts := map[string]stTyped{"fc.weight": bf(h, 2*h), "hidden_norm.weight": bf(h), "norm.weight": bf(h)}
	if ownHead {
		ts["lm_head.weight"], ts["embed_tokens.weight"] = bf(draftVocab, h), bf(vocab, h)
		d2t := make([]byte, 8*draftVocab)
		for i := range draftVocab {
			binary.LittleEndian.PutUint64(d2t[8*i:], uint64(3*i)) // target id = i + 3i, inside the vocab
		}
		ts["d2t"] = stTyped{"I64", []int{draftVocab}, d2t}
	}
	for i := range layers {
		p := fmt.Sprintf("layers.%d.", i)
		ts[p+"self_attn.q_proj.weight"] = bf(heads*hd, h)
		ts[p+"self_attn.k_proj.weight"] = bf(kv*hd, h)
		ts[p+"self_attn.v_proj.weight"] = bf(kv*hd, h)
		ts[p+"self_attn.o_proj.weight"] = bf(h, heads*hd)
		ts[p+"mlp.gate_proj.weight"] = bf(inter, h)
		ts[p+"mlp.up_proj.weight"] = bf(inter, h)
		ts[p+"mlp.down_proj.weight"] = bf(h, inter)
		ts[p+"self_attn.q_norm.weight"] = bf(hd)
		ts[p+"self_attn.k_norm.weight"] = bf(hd)
		ts[p+"input_layernorm.weight"] = bf(h)
		ts[p+"post_attention_layernorm.weight"] = bf(h)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), buildSafetensors(ts), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

// TestDFlashDrafter_releaseHostWeights: after ReleaseHostWeights the trunk's matrices and norms are gone and the mapping is
// closed, what a BlockSpec reads afterwards (BlockSize, MaskTokenID, TargetLayerIDs, DrafterGeometry) is unchanged, and a
// second release or a Close is harmless.
func TestDFlashDrafter_releaseHostWeights(t *testing.T) {
	d, err := LoadDFlashDrafter(writeTinyDFlash(t, false))
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	if d.HostWeightsReleased() || d.DrafterFC().Rows() == 0 || d.DrafterLayer(1).Down.Rows() == 0 || d.st == nil {
		t.Fatal("the fixture loaded without weights or without its mapping")
	}
	geo, bs, mask, taps := d.DrafterGeometry(), d.BlockSize(), d.MaskTokenID(), d.TargetLayerIDs()
	st := d.st

	if err := d.ReleaseHostWeights(); err != nil {
		t.Fatal(err)
	}
	if !d.HostWeightsReleased() {
		t.Error("HostWeightsReleased is false after the release")
	}
	if _, err := st.Tensor("fc.weight"); err == nil {
		t.Error("the mapping is still open after the release of a drafter with no own head")
	}
	if d.fc.Rows() != 0 || d.hiddenNorm != nil || d.finalNorm != nil {
		t.Error("fc or a trunk norm survived the release")
	}
	for i := range d.layers {
		l := &d.layers[i]
		for _, m := range []*linalg.WeightMat{&l.q, &l.k, &l.v, &l.o, &l.gate, &l.up, &l.down} {
			if m.Rows() != 0 {
				t.Errorf("layer %d: a %d-row matrix survived the release", i, m.Rows())
			}
		}
		if l.qNorm != nil || l.kNorm != nil || l.inputNorm != nil || l.postAttnNorm != nil {
			t.Errorf("layer %d: a norm survived the release", i)
		}
	}
	// the weight accessors refuse rather than hand out the zero matrices
	for name, f := range map[string]func(){
		"DrafterFC":         func() { d.DrafterFC() },
		"DrafterHiddenNorm": func() { d.DrafterHiddenNorm() },
		"DrafterFinalNorm":  func() { d.DrafterFinalNorm() },
		"DrafterLayer":      func() { d.DrafterLayer(0) },
	} {
		wantReleasedPanic(t, name, f)
	}
	if g := d.DrafterGeometry(); !reflect.DeepEqual(g, geo) {
		t.Errorf("geometry changed: %+v, want %+v", g, geo)
	}
	if d.BlockSize() != bs || d.MaskTokenID() != mask || !slices.Equal(d.TargetLayerIDs(), taps) {
		t.Errorf("block %d mask %d taps %v after the release, want %d %d %v", d.BlockSize(), d.MaskTokenID(), d.TargetLayerIDs(), bs, mask, taps)
	}
	if err := d.ReleaseHostWeights(); err != nil {
		t.Errorf("a second release: %v", err)
	}
	if err := d.Close(); err != nil {
		t.Errorf("Close after the release: %v", err)
	}
}

// TestDFlashDrafter_releaseOwnHead: a drafter that ships its own head, embedding and d2t releases those too and closes the
// mapping; HasOwnHead still says which output path it takes; and host drafting on it fails cleanly with ErrDrafterReleased
// instead of reading zero matrices (or, for EmbedBlock, quietly falling back to the target's embedding).
func TestDFlashDrafter_releaseOwnHead(t *testing.T) {
	d, err := LoadDFlashDrafter(writeTinyDFlash(t, true))
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	if !d.HasOwnHead() || d.lmHead.Rows() == 0 || d.embed.Rows() == 0 || len(d.d2t) == 0 {
		t.Fatal("the fixture loaded without its own head, embedding or d2t")
	}
	st := d.st
	if err := d.ReleaseHostWeights(); err != nil {
		t.Fatal(err)
	}
	if _, err := st.Tensor("lm_head.weight"); err == nil {
		t.Error("the mapping is still open after the release of a drafter with its own head")
	}
	if d.lmHead.Rows() != 0 || d.embed.Rows() != 0 || d.d2t != nil {
		t.Error("the own head, embedding or d2t survived the release")
	}
	if !d.HasOwnHead() {
		t.Error("HasOwnHead is false after the release; it names the output path, not whether the weights are held")
	}
	be := &cpuBackend{}
	row := make([]float32, d.hidden)
	if _, err := d.DraftBlock(be, [][]float32{row}, [][]float32{row}); !errors.Is(err, ErrDrafterReleased) {
		t.Errorf("DraftBlock: %v, want ErrDrafterReleased", err)
	}
	if _, err := d.FuseContext(be, [][]float32{make([]float32, 2*d.hidden)}); !errors.Is(err, ErrDrafterReleased) {
		t.Errorf("FuseContext: %v, want ErrDrafterReleased", err)
	}
	ctx := d.NewContext() // reads no weights: the incremental entry points below are each reached on their own
	if _, err := d.DraftBlockCtx(be, ctx, [][]float32{row}); !errors.Is(err, ErrDrafterReleased) {
		t.Errorf("DraftBlockCtx: %v, want ErrDrafterReleased", err)
	}
	for name, f := range map[string]func(){
		"ExtendContext": func() { d.ExtendContext(be, ctx, [][]float32{row}) },
		"EmbedDraft":    func() { d.EmbedDraft(1, row) },
		"EmbedBlock":    func() { d.EmbedBlock(nil, []int{1}) },
		"DraftTokenID":  func() { d.DraftTokenID(be, row) },
		"DraftIDs":      func() { d.DraftIDs(nil, [][]float32{row}) },
	} {
		wantReleasedPanic(t, name, f)
	}
}

// wantReleasedPanic fails unless f panics with ErrDrafterReleased.
func wantReleasedPanic(t *testing.T, name string, f func()) {
	t.Helper()
	defer func() {
		if r := recover(); r != ErrDrafterReleased {
			t.Errorf("%s on a released drafter: recovered %v, want a panic with ErrDrafterReleased", name, r)
		}
	}()
	f()
}
