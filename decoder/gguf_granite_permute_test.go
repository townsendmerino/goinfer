package decoder

import (
	"bytes"
	"encoding/binary"
	"hash/crc32"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/embed"
)

// Audit 2026-09-10 C-05: llama.cpp converts dense Granite (GraniteForCausalLM) through
// GraniteModel(LlamaModel), which inherits undo_permute=True — so every GGUF whose
// general.architecture is "granite" stores attn_q/attn_k rows in llama.cpp's interleaved RoPE
// order. goinfer un-permuted only llama and mellum, so dense Granite loaded exact at position 0
// and rotated the wrong pairs at every position after — fluent, and wrong, with no error.
//
// These gates write a real (data-bearing) GGUF with q/k permuted by a line-for-line port of
// llama.cpp's own permute() (conversion/llama.py), then load it through the real ggufConfig ->
// resolveArchitecture -> buildWeightsFromGGUF path and compare rows. Rows, not logits: the defect
// is a row order, so the row order is the thing to pin.

type ggufDataTensor struct {
	name string
	dims []uint64 // GGUF order: ne0 (innermost, = in-features) first
	data []float32
}

// buildGGUFData is buildGGUF with tensor data: f32 tensors at 32-byte-aligned offsets inside the
// data section, which begins at the 32-byte boundary after the header (embed's default alignment).
func buildGGUFData(kvs []ggufKV, tensors []ggufDataTensor) []byte {
	var hdr bytes.Buffer
	hdr.Write(gU32(0x46554747))
	hdr.Write(gU32(3))
	hdr.Write(gU64(uint64(len(tensors))))
	hdr.Write(gU64(uint64(len(kvs))))
	for _, kv := range kvs {
		hdr.Write(gStr(kv.key))
		hdr.Write(gU32(kv.typ))
		hdr.Write(kv.val)
	}
	var data bytes.Buffer
	for _, t := range tensors {
		for data.Len()%32 != 0 {
			data.WriteByte(0)
		}
		hdr.Write(gStr(t.name))
		hdr.Write(gU32(uint32(len(t.dims))))
		for _, d := range t.dims {
			hdr.Write(gU64(d))
		}
		hdr.Write(gU32(0)) // GGML_TYPE_F32
		hdr.Write(gU64(uint64(data.Len())))
		for _, f := range t.data {
			data.Write(gF32(f))
		}
	}
	for hdr.Len()%32 != 0 {
		hdr.WriteByte(0)
	}
	return append(hdr.Bytes(), data.Bytes()...)
}

// llamaCppPermute ports llama.cpp's conversion/llama.py permute():
//
//	weights.reshape(n_head, 2, rows // n_head // 2, *rest).swapaxes(1, 2).reshape(weights.shape)
//
// i.e. GGUF row (h*hd + 2*j + s) holds HF row (h*hd + s*half + j). w is [rows][in] row-major.
func llamaCppPermute(w []float32, rows, in, nHead int) []float32 {
	hd := rows / nHead
	half := hd / 2
	out := make([]float32, len(w))
	for h := range nHead {
		for s := range 2 {
			for j := range half {
				copy(out[(h*hd+2*j+s)*in:][:in], w[(h*hd+s*half+j)*in:][:in])
			}
		}
	}
	return out
}

func distinct(n, seed int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(seed*1000+i) * 0.001 // every element different, so any misplaced row shows
	}
	return v
}

// tinyNormRopeGGUF writes a 1-layer GQA model (8 hidden, 2 q heads, 1 kv head, head_dim 4) under
// `arch`, with q/k PERMUTED the way llama.cpp writes them. Returns the file and the HF-order q, k, v.
func tinyNormRopeGGUF(arch string) (raw []byte, q, k, v []float32) {
	const hidden, heads, kvHeads, hd, inter, vocab = 8, 2, 1, 4, 16, 16
	qDim, kvDim := heads*hd, kvHeads*hd
	q, k, v = distinct(qDim*hidden, 1), distinct(kvDim*hidden, 2), distinct(kvDim*hidden, 3)
	kvs := []ggufKV{
		kvStr("general.architecture", arch),
		kvU32(arch+".context_length", 64), kvU32(arch+".embedding_length", hidden),
		kvU32(arch+".block_count", 1), kvU32(arch+".attention.head_count", heads),
		kvU32(arch+".attention.head_count_kv", kvHeads), kvU32(arch+".feed_forward_length", inter),
		kvF32(arch+".attention.layer_norm_rms_epsilon", 1e-6), kvF32(arch+".rope.freq_base", 10000),
	}
	if arch == "granite" {
		// The REAL Granite 4.2 multipliers (granite-4.2-3b config.json). residual_scale must be 1.0:
		// every released 4.2 size ships it, and validateGraniteDense rejects anything else because
		// the generic forward has no residual hook. (A first draft used Granite 3.x's 0.22 and the
		// gate "failed" at resolveArchitecture, never reaching a row — a red that proved nothing.)
		kvs = append(kvs, kvF32("granite.attention.scale", 0.015625), kvF32("granite.embedding_scale", 1),
			kvF32("granite.logit_scale", 1), kvF32("granite.residual_scale", 1))
	}
	ones := func(n int) []float32 {
		o := make([]float32, n)
		for i := range o {
			o[i] = 1
		}
		return o
	}
	ts := []ggufDataTensor{
		{"token_embd.weight", []uint64{hidden, vocab}, distinct(vocab*hidden, 4)},
		{"output_norm.weight", []uint64{hidden}, ones(hidden)},
		{"output.weight", []uint64{hidden, vocab}, distinct(vocab*hidden, 5)},
		{"blk.0.attn_norm.weight", []uint64{hidden}, ones(hidden)},
		{"blk.0.attn_q.weight", []uint64{hidden, uint64(qDim)}, llamaCppPermute(q, qDim, hidden, heads)},
		{"blk.0.attn_k.weight", []uint64{hidden, uint64(kvDim)}, llamaCppPermute(k, kvDim, hidden, kvHeads)},
		{"blk.0.attn_v.weight", []uint64{hidden, uint64(kvDim)}, v},
		{"blk.0.attn_output.weight", []uint64{uint64(qDim), hidden}, distinct(hidden*qDim, 6)},
		{"blk.0.ffn_norm.weight", []uint64{hidden}, ones(hidden)},
		{"blk.0.ffn_gate.weight", []uint64{hidden, inter}, distinct(inter*hidden, 7)},
		{"blk.0.ffn_up.weight", []uint64{hidden, inter}, distinct(inter*hidden, 8)},
		{"blk.0.ffn_down.weight", []uint64{inter, hidden}, distinct(hidden*inter, 9)},
	}
	return buildGGUFData(kvs, ts), q, k, v
}

func loadTinyGGUFWeights(t *testing.T, raw []byte, wantArch string) *Weights {
	t.Helper()
	g, err := embed.OpenGGUFBytes(raw)
	if err != nil {
		t.Fatalf("OpenGGUFBytes: %v", err)
	}
	cfg, err := ggufConfig(g)
	if err != nil {
		t.Fatalf("ggufConfig: %v", err)
	}
	arch, _, err := resolveArchitecture(cfg)
	if err != nil {
		t.Fatalf("resolveArchitecture: %v", err)
	}
	if arch.Name != wantArch {
		t.Fatalf("resolved arch %q, want %q", arch.Name, wantArch)
	}
	w, err := buildWeightsFromGGUF(cfg, arch, g, quantNone, false, true, nil, "")
	if err != nil {
		t.Fatalf("buildWeightsFromGGUF: %v", err)
	}
	return w
}

func assertRows(t *testing.T, what string, got interface{ F32() ([]float32, bool) }, want []float32) {
	t.Helper()
	f, ok := got.F32()
	if !ok {
		t.Fatalf("%s: not f32 (quantNone load)", what)
	}
	if len(f) != len(want) {
		t.Fatalf("%s: %d elements, want %d", what, len(f), len(want))
	}
	for i := range f {
		if f[i] != want[i] {
			t.Errorf("%s: element %d = %v, want %v — rows are not in HF rotate_half order, so RoPE "+
				"rotates the wrong pairs at every position past 0", what, i, f[i], want[i])
			return
		}
	}
}

func checkUnpermuted(t *testing.T, arch string) {
	raw, q, k, v := tinyNormRopeGGUF(arch)
	w := loadTinyGGUFWeights(t, raw, arch)
	l := &w.Layers[0]
	assertRows(t, arch+" q_proj", &l.QProj, q)
	assertRows(t, arch+" k_proj (GQA: permuted with n_head_kv)", &l.KProj, k)
	assertRows(t, arch+" v_proj (never permuted — guards an over-applied fix)", &l.VProj, v)
}

// TestGGUF_graniteDenseUnpermutesQK is C-05's gate.
func TestGGUF_graniteDenseUnpermutesQK(t *testing.T) { checkUnpermuted(t, "granite") }

// TestGGUF_llamaUnpermutesQK is the control: the same synthetic writer and permute on the family
// that has always been un-permuted. If THIS fails, the harness is wrong, not the loader.
func TestGGUF_llamaUnpermutesQK(t *testing.T) { checkUnpermuted(t, "llama") }

// giwAtVersion re-stamps a serialized bundle's format version and recomputes its CRC, producing
// exactly what an older writer would have left on disk for the same weights.
func giwAtVersion(t *testing.T, b []byte, v uint32) []byte {
	t.Helper()
	out := append([]byte(nil), b...)
	binary.LittleEndian.PutUint32(out[len(giwMagic):], v)
	binary.LittleEndian.PutUint32(out[len(out)-4:], crc32.ChecksumIEEE(out[:len(out)-4]))
	return out
}

// TestGIW_refusesPreFixDenseGraniteBundle closes C-05's breaking-after-tag half. A dense-granite
// .giw transcoded from a GGUF before the fix is CRC-valid, shape-valid, mtime-fresh and WRONG, so
// every freshness check passed it. Refusing pre-v10 granite bundles is what makes prequant's
// selfCheck (decoder.Load -> LoadSerializedWeights) see it as not loading, and rebuild it.
func TestGIW_refusesPreFixDenseGraniteBundle(t *testing.T) {
	const lastUnfixedVersion = 9 // the last format version written before the C-05 fix
	for _, tc := range []struct {
		arch        string
		wantRefused bool
	}{
		{"granite", true},
		{"llama", false}, // a version gate for one family must not refuse another's bundles
	} {
		raw, _, _, _ := tinyNormRopeGGUF(tc.arch)
		b, err := SerializeWeights(loadTinyGGUFWeights(t, raw, tc.arch), "c05-test")
		if err != nil {
			t.Fatalf("%s: SerializeWeights: %v", tc.arch, err)
		}
		if _, err := LoadSerializedWeights(b); err != nil {
			t.Fatalf("%s: a current-version bundle must load: %v", tc.arch, err)
		}
		_, err = LoadSerializedWeights(giwAtVersion(t, b, lastUnfixedVersion))
		switch {
		case tc.wantRefused && err == nil:
			t.Errorf("%s: a v%d bundle LOADED — a pre-fix sidecar with permuted q/k would be served "+
				"as fresh forever (audit C-05)", tc.arch, lastUnfixedVersion)
		case tc.wantRefused && !strings.Contains(err.Error(), "C-05"):
			t.Errorf("%s: refused, but not for C-05: %v", tc.arch, err)
		case !tc.wantRefused && err != nil:
			t.Errorf("%s: a v%d bundle was refused — the C-05 gate is over-broad: %v", tc.arch, lastUnfixedVersion, err)
		}
	}
}
