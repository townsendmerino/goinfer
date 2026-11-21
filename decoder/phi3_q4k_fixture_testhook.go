//go:build goinfer_testhooks

package decoder

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"strings"
)

// WritePhi3Q4KTinyGGUFForTest writes a synthetic Phi-3 GGUF to path, for the tests keyed on the Phi-3 family and its Q4_K blocks (the
// activation-hazard default, the q4k lane, fit) outside the decoder package: two layers of Phi-3's fused attn_qkv and ffn_up
// (gate‖up) at the smallest Q4_K width, 256, as random Q4_K blocks with small scales, beside f32 embeddings and norms and a 64-token
// list. The bytes are fixed by a seed. It is written, not committed: random weights are no family's fixture.
func WritePhi3Q4KTinyGGUFForTest(path string) error { return writeQ4KTinyGGUF(path, "phi3", false) }

// WritePhi3KMixTinyGGUFForTest is WritePhi3Q4KTinyGGUFForTest with the real Phi-3-mini Q4_K_M file's type mix: attn_qkv stored
// as Q5_K and ffn_down as Q6_K, every other projection Q4_K.
func WritePhi3KMixTinyGGUFForTest(path string) error { return writeQ4KTinyGGUF(path, "phi3", true) }

// WriteLlamaQ4KTinyGGUFForTest is WritePhi3Q4KTinyGGUFForTest for llama, whose q, k, v, gate and up are separate tensors: a family
// with no activation hazard, so an int4 load of it runs resident on Metal.
func WriteLlamaQ4KTinyGGUFForTest(path string) error { return writeQ4KTinyGGUF(path, "llama", false) }

// writeQ4KTinyGGUF writes the two-layer synthetic GGUF for arch, phi3 or llama; kmix stores qkv as Q5_K and down as Q6_K.
func writeQ4KTinyGGUF(path, arch string, kmix bool) error {
	const hidden, heads, hd, inter, vocab, layers = 256, 4, 64, 256, 64, 2
	const tU32, tF32, tStr, tArr = 4, 6, 8, 9
	le := binary.LittleEndian
	var hdr, data bytes.Buffer
	u32 := func(b *bytes.Buffer, v uint32) { _ = binary.Write(b, le, v) }
	u64 := func(b *bytes.Buffer, v uint64) { _ = binary.Write(b, le, v) }
	str := func(b *bytes.Buffer, s string) { u64(b, uint64(len(s))); b.WriteString(s) }
	type kv struct {
		key string
		put func(*bytes.Buffer)
	}
	kvU32 := func(k string, v uint32) kv { return kv{k, func(b *bytes.Buffer) { u32(b, tU32); u32(b, v) }} }
	kvF32 := func(k string, v float32) kv {
		return kv{k, func(b *bytes.Buffer) { u32(b, tF32); u32(b, math.Float32bits(v)) }}
	}
	kvStr := func(k, v string) kv { return kv{k, func(b *bytes.Buffer) { u32(b, tStr); str(b, v) }} }
	toks := make([]string, vocab)
	for i := range toks {
		toks[i] = string(rune('a'+i%26)) + strings.Repeat("x", i/26)
	}
	kvs := []kv{
		kvStr("general.architecture", arch),
		kvU32(arch+".context_length", 256), kvU32(arch+".embedding_length", hidden),
		kvU32(arch+".block_count", layers), kvU32(arch+".attention.head_count", heads),
		kvU32(arch+".attention.head_count_kv", heads), kvU32(arch+".feed_forward_length", inter),
		kvU32(arch+".rope.dimension_count", hd),
		kvF32(arch+".attention.layer_norm_rms_epsilon", 1e-5), kvF32(arch+".rope.freq_base", 10000),
		kvStr("tokenizer.ggml.model", "gpt2"),
		{"tokenizer.ggml.tokens", func(b *bytes.Buffer) {
			u32(b, tArr)
			u32(b, tStr)
			u64(b, uint64(len(toks)))
			for _, s := range toks {
				str(b, s)
			}
		}},
	}
	type tensor struct {
		name string
		dims []uint64 // GGUF order: ne0 (in-features) first
		typ  uint32   // 0 f32, 12 Q4_K, 13 Q5_K, 14 Q6_K
		raw  []byte
	}
	r := rand.New(rand.NewPCG(3, 4))
	f32 := func(name string, dims []uint64, n int, one bool) tensor {
		b := make([]byte, 4*n)
		for i := range n {
			v := float32(0.02 * float64(i%13-6))
			if one {
				v = 1
			}
			le.PutUint32(b[4*i:], math.Float32bits(v))
		}
		return tensor{name, dims, 0, b}
	}
	q4k := func(name string, out, in int) tensor {
		raw := make([]byte, out*(in/256)*144)
		// A super-block: f16 d and dmin, then scales and nibbles. d and dmin are 2^-10 to 2^-7: larger ones (decoder's q4kTestMat
		// draws up to 1) drive two random layers' activations past f16's range in Metal's KV, and the logits to NaN.
		for b := 0; b < len(raw); b += 144 {
			for _, off := range []int{0, 2} {
				le.PutUint16(raw[b+off:], uint16(5+r.IntN(4))<<10|uint16(r.IntN(1024)))
			}
			for i := 4; i < 144; i++ {
				raw[b+i] = byte(r.IntN(256))
			}
		}
		return tensor{name, []uint64{uint64(in), uint64(out)}, 12, raw}
	}
	// q5k is q4k's block with 32 more bytes of fifth bits (176 bytes); q6k's block (210 bytes) puts its f16 d last, here 2^-10 to
	// 2^-8, with int8 sub-scales in [-16, 16] so its weights stay within q4k's range.
	q5k := func(name string, out, in int) tensor {
		raw := make([]byte, out*(in/256)*176)
		for b := 0; b < len(raw); b += 176 {
			for _, off := range []int{0, 2} {
				le.PutUint16(raw[b+off:], uint16(5+r.IntN(4))<<10|uint16(r.IntN(1024)))
			}
			for i := 4; i < 176; i++ {
				raw[b+i] = byte(r.IntN(256))
			}
		}
		return tensor{name, []uint64{uint64(in), uint64(out)}, 13, raw}
	}
	q6k := func(name string, out, in int) tensor {
		raw := make([]byte, out*(in/256)*210)
		for b := 0; b < len(raw); b += 210 {
			for i := range 192 {
				raw[b+i] = byte(r.IntN(256))
			}
			for i := 192; i < 208; i++ {
				raw[b+i] = byte(int8(r.IntN(33) - 16))
			}
			le.PutUint16(raw[b+208:], uint16(5+r.IntN(3))<<10|uint16(r.IntN(1024)))
		}
		return tensor{name, []uint64{uint64(in), uint64(out)}, 14, raw}
	}
	qkvT, downT := q4k, q4k
	if kmix {
		qkvT, downT = q5k, q6k
	}
	ts := []tensor{
		f32("token_embd.weight", []uint64{hidden, vocab}, vocab*hidden, false),
		f32("output_norm.weight", []uint64{hidden}, hidden, true),
		f32("output.weight", []uint64{hidden, vocab}, vocab*hidden, false),
	}
	for l := range layers {
		p := fmt.Sprintf("blk.%d.", l)
		ts = append(ts, f32(p+"attn_norm.weight", []uint64{hidden}, hidden, true))
		if arch == "phi3" {
			ts = append(ts, qkvT(p+"attn_qkv.weight", 3*heads*hd, hidden))
		} else {
			ts = append(ts, q4k(p+"attn_q.weight", heads*hd, hidden), q4k(p+"attn_k.weight", heads*hd, hidden), q4k(p+"attn_v.weight", heads*hd, hidden))
		}
		ts = append(ts, q4k(p+"attn_output.weight", hidden, heads*hd), f32(p+"ffn_norm.weight", []uint64{hidden}, hidden, true))
		if arch == "phi3" {
			ts = append(ts, q4k(p+"ffn_up.weight", 2*inter, hidden))
		} else {
			ts = append(ts, q4k(p+"ffn_gate.weight", inter, hidden), q4k(p+"ffn_up.weight", inter, hidden))
		}
		ts = append(ts, downT(p+"ffn_down.weight", hidden, inter))
	}
	u32(&hdr, 0x46554747) // "GGUF"
	u32(&hdr, 3)
	u64(&hdr, uint64(len(ts)))
	u64(&hdr, uint64(len(kvs)))
	for _, k := range kvs {
		str(&hdr, k.key)
		k.put(&hdr)
	}
	for _, t := range ts {
		for data.Len()%32 != 0 {
			data.WriteByte(0)
		}
		str(&hdr, t.name)
		u32(&hdr, uint32(len(t.dims)))
		for _, d := range t.dims {
			u64(&hdr, d)
		}
		u32(&hdr, t.typ)
		u64(&hdr, uint64(data.Len()))
		data.Write(t.raw)
	}
	for hdr.Len()%32 != 0 {
		hdr.WriteByte(0)
	}
	return os.WriteFile(path, append(hdr.Bytes(), data.Bytes()...), 0o644)
}
