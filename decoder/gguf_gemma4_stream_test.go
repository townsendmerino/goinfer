package decoder

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/aikit/embed"
)

// S2 (task-never-swap-2026-09.md): gemma4 was the last family transcoded by building the whole model
// resident and serializing it once — 34.7 GB RSS for the 26B-A4B, which no 16 GB Mac can build. It now
// streams (build → write → release per layer). These gates pin that the streamed bundle is
// byte-identical to the resident one on synthetic gemma4 GGUFs shaped like the two real layouts:
//
//   - "26b": the 26B-A4B's parallel dense+MoE FFN (router, stacked gate‖up and down experts, the three
//     extra norms) and a global layer with no attn_v (K=V, VFromK);
//   - "e2b": the E-models' Per-Layer Embeddings (model-level per_layer_* inputs in the head, per-layer
//     inp_gate/proj/post_norm) and a KV-shared tail layer, with per-layer FFN widths.
//
// Both carry a sliding/global pattern with different head dims and KV-head counts per type, so every
// per-layer geometry branch in loadG4 is exercised. The resident arm is exactly what
// StreamTranscodeGGUF did for gemma4 before (resolve EOS, build with needCanonical, serialize), so the
// comparison is against the behaviour being replaced. The quant label is excluded by construction: a
// streamed body records "" (the layers do not exist yet when the head is written — see
// writeHeadGlobals' B11 note), the resident one the resolved label.

type g4Variant struct {
	name              string
	moe, ple, kEqV    bool
	sharedKV          int
	layers, ffnGlobal int
}

func tinyGemma4GGUF(v g4Variant) []byte {
	const hidden, heads, hdSWA, hdGlobal, kvSWA, kvGlobal, vocab, ffn = 64, 2, 32, 64, 2, 1, 64, 128
	const nE, topK, moeInter, pleDim = 4, 2, 32, 32
	L := v.layers
	sliding := make([]bool, L)
	ffns := make([]uint32, L)
	kvHeads := make([]uint32, L)
	for i := range L {
		sliding[i] = i != L-1 // last layer global
		ffns[i] = ffn
		kvHeads[i] = kvSWA
		if !sliding[i] {
			ffns[i] = uint32(v.ffnGlobal)
			kvHeads[i] = kvGlobal
		}
	}
	kvs := []ggufKV{
		kvStr("general.architecture", "gemma4"),
		kvU32("gemma4.context_length", 128), kvU32("gemma4.embedding_length", hidden),
		kvU32("gemma4.block_count", uint32(L)), kvU32("gemma4.attention.head_count", heads),
		kvU32Arr("gemma4.attention.head_count_kv", kvHeads),
		kvU32("gemma4.attention.key_length_swa", hdSWA), kvU32("gemma4.attention.key_length", hdGlobal),
		kvU32("gemma4.attention.sliding_window", 16), kvU32("gemma4.attention.shared_kv_layers", uint32(v.sharedKV)),
		kvU32Arr("gemma4.feed_forward_length", ffns),
		kvBoolArr("gemma4.attention.sliding_window_pattern", sliding),
		kvF32("gemma4.attention.layer_norm_rms_epsilon", 1e-6),
		kvF32("gemma4.rope.freq_base", 1e6), kvF32("gemma4.rope.freq_base_swa", 1e4),
		kvF32("gemma4.final_logit_softcapping", 30),
	}
	if v.ple {
		kvs = append(kvs, kvU32("gemma4.embedding_length_per_layer_input", pleDim))
	}
	if v.moe {
		kvs = append(kvs, kvU32("gemma4.expert_count", nE), kvU32("gemma4.expert_used_count", topK),
			kvU32("gemma4.expert_feed_forward_length", moeInter))
	}
	seed := 10
	next := func(n int) []float32 { seed++; return distinct(n, seed) }
	u := func(xs ...int) []uint64 {
		o := make([]uint64, len(xs))
		for i, x := range xs {
			o[i] = uint64(x)
		}
		return o
	}
	ts := []ggufDataTensor{
		{"token_embd.weight", u(hidden, vocab), next(vocab * hidden)},
		{"output_norm.weight", u(hidden), next(hidden)},
	}
	if v.ple {
		tot := L * pleDim
		ts = append(ts,
			ggufDataTensor{"per_layer_token_embd.weight", u(tot, vocab), next(vocab * tot)},
			ggufDataTensor{"per_layer_model_proj.weight", u(hidden, tot), next(tot * hidden)},
			ggufDataTensor{"per_layer_proj_norm.weight", u(pleDim), next(pleDim)})
	}
	firstShared := L - v.sharedKV
	for i := range L {
		p := fmt.Sprintf("blk.%d.", i)
		hd, kvh, f := hdSWA, kvSWA, ffn
		if !sliding[i] {
			hd, kvh, f = hdGlobal, kvGlobal, v.ffnGlobal
		}
		qDim, kvDim := heads*hd, kvh*hd
		add := func(name string, dims []uint64, n int) { ts = append(ts, ggufDataTensor{p + name, dims, next(n)}) }
		add("attn_norm.weight", u(hidden), hidden)
		add("post_attention_norm.weight", u(hidden), hidden)
		add("ffn_norm.weight", u(hidden), hidden)
		add("post_ffw_norm.weight", u(hidden), hidden)
		add("attn_q.weight", u(hidden, qDim), qDim*hidden)
		add("attn_q_norm.weight", u(hd), hd)
		if i < firstShared {
			add("attn_k.weight", u(hidden, kvDim), kvDim*hidden)
			add("attn_k_norm.weight", u(hd), hd)
			if !(v.kEqV && !sliding[i]) { // K=V: the global layer ships no attn_v
				add("attn_v.weight", u(hidden, kvDim), kvDim*hidden)
			}
		}
		add("attn_output.weight", u(qDim, hidden), hidden*qDim)
		add("ffn_gate.weight", u(hidden, f), f*hidden)
		add("ffn_up.weight", u(hidden, f), f*hidden)
		add("ffn_down.weight", u(f, hidden), hidden*f)
		if v.moe {
			add("post_ffw_norm_1.weight", u(hidden), hidden)
			add("pre_ffw_norm_2.weight", u(hidden), hidden)
			add("post_ffw_norm_2.weight", u(hidden), hidden)
			add("ffn_gate_inp.weight", u(hidden, nE), nE*hidden)
			add("ffn_gate_inp.scale", u(hidden), hidden)
			add("ffn_down_exps.scale", u(nE), nE)
			add("ffn_gate_up_exps.weight", u(hidden, 2*moeInter, nE), nE*2*moeInter*hidden)
			add("ffn_down_exps.weight", u(moeInter, hidden, nE), nE*hidden*moeInter)
		}
		if v.ple {
			add("inp_gate.weight", u(hidden, pleDim), pleDim*hidden)
			add("proj.weight", u(pleDim, hidden), hidden*pleDim)
			add("post_norm.weight", u(hidden), hidden)
		}
		add("layer_output_scale.weight", u(1), 1)
	}
	return buildGGUFData(kvs, ts)
}

var gemma4Variants = []g4Variant{
	{name: "26b", moe: true, kEqV: true, layers: 4, ffnGlobal: 128},
	{name: "e2b", ple: true, sharedKV: 1, layers: 4, ffnGlobal: 256},
}

// residentGemma4GIW is StreamTranscodeGGUF's former gemma4 path, verbatim in effect: the same config
// resolution (including the EOS write-back), a resident needCanonical build, one serialize.
func residentGemma4GIW(t *testing.T, path, quant string, target GIWTarget) []byte {
	t.Helper()
	q, err := parseQuant(quant)
	if err != nil {
		t.Fatal(err)
	}
	g, err := embed.OpenGGUFMmap(path)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	cfg, err := ggufConfig(g)
	if err != nil {
		t.Fatalf("ggufConfig: %v", err)
	}
	if raw, jerr := json.Marshal(resolveEOSIDs(filepath.Dir(path), cfg)); jerr == nil {
		cfg.EOSTokenID = raw
	}
	arch, _, err := resolveArchitecture(cfg)
	if err != nil {
		t.Fatalf("resolveArchitecture: %v", err)
	}
	if arch.gemma4 == nil {
		t.Fatalf("resolved %q, not a gemma4 architecture — the fixture is not exercising the gemma4 branch", arch.Name)
	}
	w, err := buildWeightsFromGGUF(cfg, arch, g, q, false, true, false, nil, "", nil)
	if err != nil {
		t.Fatalf("resident build: %v", err)
	}
	var buf bytes.Buffer
	if _, err := SerializeWeightsToForTarget(&buf, w, "g4", target); err != nil {
		t.Fatalf("serialize: %v", err)
	}
	return buf.Bytes()
}

// splitAtLabel returns the body before the quant-label string field and everything after it and the
// v12 header pad that follows it (the pad's length depends on the label's).
func splitAtLabel(t *testing.T, b []byte) (pre, post []byte, label string) {
	t.Helper()
	off := len(giwMagic) + 4 + 4 // magic, version, legacy quant enum
	for range 2 {                // id, then config JSON: u32 length + bytes
		n := int(binary.LittleEndian.Uint32(b[off:]))
		off += 4 + n
	}
	n := int(binary.LittleEndian.Uint32(b[off:]))
	label = string(b[off+4 : off+4+n])
	end := off + 4 + n
	end += (-end) & 15 // alignHead
	return b[:off], b[end:], label
}

func TestGemma4GGUF_streamedMatchesResident(t *testing.T) {
	for _, v := range gemma4Variants {
		path := filepath.Join(t.TempDir(), "gemma4-"+v.name+".gguf")
		if err := os.WriteFile(path, tinyGemma4GGUF(v), 0o644); err != nil {
			t.Fatal(err)
		}
		for _, target := range []GIWTarget{GIWTargetNone, GIWTargetMetal} {
			for _, quant := range []string{"int4", "int8int8", ""} {
				t.Run(fmt.Sprintf("%s/target=%q/quant=%q", v.name, target, quant), func(t *testing.T) {
					resident := residentGemma4GIW(t, path, quant, target)
					var sb bytes.Buffer
					if _, err := StreamTranscodeGGUF(context.Background(), path, &sb, quant, false, target, "g4"); err != nil {
						t.Fatalf("StreamTranscodeGGUF: %v", err)
					}
					streamed := sb.Bytes()
					rPre, rPost, rLabel := splitAtLabel(t, resident)
					sPre, sPost, sLabel := splitAtLabel(t, streamed)
					if sLabel != "" {
						t.Errorf("streamed label %q, want \"\" (no layers exist when the head is written)", sLabel)
					}
					if !bytes.Equal(rPre, sPre) {
						t.Fatalf("bundles differ BEFORE the quant label (%d vs %d B) — the header diverges", len(rPre), len(sPre))
					}
					// The trailing 4 bytes are each body's own CRC, which covers the (different) label.
					rw, sw := rPost[:len(rPost)-4], sPost[:len(sPost)-4]
					if !bytes.Equal(rw, sw) {
						first := -1
						for i := 0; i < len(rw) && i < len(sw); i++ {
							if rw[i] != sw[i] {
								first = i
								break
							}
						}
						t.Fatalf("bundles differ AFTER the label (resident label %q): lengths %d vs %d, first difference at +%d — "+
							"the streamed gemma4 bundle does not carry the same weights", rLabel, len(rw), len(sw), first)
					}
					// And the streamed bundle loads: the reader accepts it and sees the gemma4 structure.
					sw2, err := LoadSerializedWeights(streamed)
					if err != nil {
						t.Fatalf("LoadSerializedWeights(streamed): %v", err)
					}
					if len(sw2.Layers) != v.layers {
						t.Fatalf("streamed bundle loads %d layers, want %d", len(sw2.Layers), v.layers)
					}
					last := &sw2.Layers[v.layers-1]
					switch {
					case v.moe && last.gemma4moe == nil:
						t.Errorf("26b variant: streamed bundle lost the MoE branch")
					case v.kEqV && !last.VFromK:
						t.Errorf("26b variant: streamed bundle lost the K=V flag on the global layer")
					case v.ple && (sw2.PerLayerTokenEmbed.Rows() == 0 || last.PLEGate.Rows() == 0):
						t.Errorf("e2b variant: streamed bundle lost the PLE inputs (model-level rows %d, layer gate rows %d)",
							sw2.PerLayerTokenEmbed.Rows(), last.PLEGate.Rows())
					case v.sharedKV > 0 && !last.KVShared:
						t.Errorf("e2b variant: streamed bundle lost the KV-shared flag")
					}
				})
			}
		}
	}
}

// M-21's cancellation contract holds for the new branch: a pre-cancelled context writes nothing.
func TestGemma4GGUF_streamCancelled(t *testing.T) {
	path := filepath.Join(t.TempDir(), "gemma4.gguf")
	if err := os.WriteFile(path, tinyGemma4GGUF(gemma4Variants[0]), 0o644); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	var body bytes.Buffer
	n, err := StreamTranscodeGGUF(ctx, path, &body, "int4", false, GIWTargetNone, "g4")
	if !errors.Is(err, context.Canceled) || body.Len() != 0 {
		t.Fatalf("pre-cancelled StreamTranscodeGGUF = (%d, %v), wrote %d B; want context.Canceled and 0 B", n, err, body.Len())
	}
}

// kvU32Arr is kvBoolArr for a uint32 array (gemma4's per-layer head_count_kv / feed_forward_length).
func kvU32Arr(k string, vs []uint32) ggufKV {
	b := append(gU32(gtUint32), gU64(uint64(len(vs)))...)
	for _, v := range vs {
		b = append(b, gU32(v)...)
	}
	return ggufKV{k, gtArray, b}
}
