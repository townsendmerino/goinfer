package decoder

import (
	"bytes"
	"context"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"unsafe"

	"github.com/townsendmerino/aikit/linalg"
)

// D8B-G (docs/tasks/task-memory-campaign-2026-10.md, §4 MEM13, "D8b-build gates"): a q4k load's Q4_K tensors as .giw kind 8.

type ggufRawTensor struct {
	name string
	dims []uint64 // GGUF order: ne0 (in-features) first
	typ  uint32   // GGML type: 0 f32, 12 Q4_K
	data []byte
}

// buildGGUFRaw is buildGGUFData with typed raw tensor bytes.
func buildGGUFRaw(kvs []ggufKV, tensors []ggufRawTensor) []byte {
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
		hdr.Write(gU32(t.typ))
		hdr.Write(gU64(uint64(data.Len())))
		data.Write(t.data)
	}
	for hdr.Len()%32 != 0 {
		hdr.WriteByte(0)
	}
	return append(hdr.Bytes(), data.Bytes()...)
}

// q4kTinyGGUF writes a one-layer llama .gguf with a tokenizer whose attention and MLP projections are Q4_K, and returns
// its path.
func q4kTinyGGUF(t *testing.T, dir string) string {
	t.Helper()
	const arch, hidden, heads, kvHeads, hd, inter, vocab = "llama", 256, 4, 2, 64, 512, 64
	qDim, kvDim := heads*hd, kvHeads*hd
	toks := make([]string, vocab)
	for i := range toks {
		toks[i] = string(rune('a'+i%26)) + strings.Repeat("x", i/26)
	}
	kvs := []ggufKV{
		kvStr("general.architecture", arch),
		kvU32(arch+".context_length", 64), kvU32(arch+".embedding_length", hidden),
		kvU32(arch+".block_count", 1), kvU32(arch+".attention.head_count", heads),
		kvU32(arch+".attention.head_count_kv", kvHeads), kvU32(arch+".feed_forward_length", inter),
		kvF32(arch+".attention.layer_norm_rms_epsilon", 1e-6), kvF32(arch+".rope.freq_base", 10000),
		kvStr("tokenizer.ggml.model", "gpt2"), kvStrArr("tokenizer.ggml.tokens", toks),
	}
	r := rand.New(rand.NewPCG(81, 82))
	f32 := func(name string, dims []uint64, n int, one bool) ggufRawTensor {
		b := make([]byte, 0, 4*n)
		for i := range n {
			v := float32(0.02 * float64(i%13-6))
			if one {
				v = 1
			}
			b = append(b, gF32(v)...)
		}
		return ggufRawTensor{name, dims, 0, b}
	}
	q4k := func(name string, out, in int) ggufRawTensor {
		wm := q4kTestMat(t, r, out, in)
		raw, _ := wm.Q4K()
		return ggufRawTensor{name, []uint64{uint64(in), uint64(out)}, 12, raw}
	}
	ts := []ggufRawTensor{
		f32("token_embd.weight", []uint64{hidden, vocab}, vocab*hidden, false),
		f32("output_norm.weight", []uint64{hidden}, hidden, true),
		f32("output.weight", []uint64{hidden, vocab}, vocab*hidden, false),
		f32("blk.0.attn_norm.weight", []uint64{hidden}, hidden, true),
		q4k("blk.0.attn_q.weight", qDim, hidden),
		q4k("blk.0.attn_k.weight", kvDim, hidden),
		q4k("blk.0.attn_v.weight", kvDim, hidden),
		q4k("blk.0.attn_output.weight", hidden, qDim),
		f32("blk.0.ffn_norm.weight", []uint64{hidden}, hidden, true),
		q4k("blk.0.ffn_gate.weight", inter, hidden),
		q4k("blk.0.ffn_up.weight", inter, hidden),
		q4k("blk.0.ffn_down.weight", hidden, inter),
	}
	p := filepath.Join(dir, "q4k-tiny.gguf")
	if err := os.WriteFile(p, buildGGUFRaw(kvs, ts), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// d8bCompare checks that a and b compute the same logits token by token and the same greedy tokens.
func d8bCompare(t *testing.T, a, b *Model) {
	t.Helper()
	prompt := []int{3, 14, 15, 9, 26, 5, 35, 8}
	logits := func(m *Model) []float32 {
		cache := m.NewCache(len(prompt))
		var all []float32
		for _, id := range prompt {
			lg, _, err := m.ForwardCapture(id, cache, nil)
			if err != nil {
				t.Fatal(err)
			}
			all = append(all, lg...)
		}
		return all
	}
	if !slices.Equal(logits(a), logits(b)) {
		t.Error("token-by-token logits differ")
	}
	if ga, gb := greedyTokens(t, a, prompt, 8), greedyTokens(t, b, prompt, 8); !slices.Equal(ga, gb) {
		t.Errorf("greedy tokens differ: %v against %v", ga, gb)
	}
}

// TestD8b_q4kRoundTripsThroughGIW is D8B-G1.
func TestD8b_q4kRoundTripsThroughGIW(t *testing.T) {
	dir := t.TempDir()
	src := q4kTinyGGUF(t, dir)
	direct, err := Load(src, Options{Backend: "cpu", Quant: "q4k", DirectLoad: true})
	if err != nil {
		t.Fatalf("direct q4k load: %v", err)
	}
	defer direct.Close()
	out := filepath.Join(dir, "q4k-tiny.giw")
	if err := TranscodeGIW(context.Background(), src, out, "q4k", false, GIWTargetForBackend("cpu")); err != nil {
		t.Fatalf("TranscodeGIW(q4k): %v", err)
	}
	viaGIW, err := Load(out, Options{Backend: "cpu"}) // no Quant: the bundle's own label stamps per-32 activations
	if err != nil {
		t.Fatalf("Load(%s): %v", out, err)
	}
	defer viaGIW.Close()
	if q := viaGIW.Quant(); q != "q4k" {
		t.Errorf("the bundle reports quant %q, want q4k", q)
	}
	lo, hi := uintptr(unsafe.Pointer(&viaGIW.mmap[0])), uintptr(unsafe.Pointer(&viaGIW.mmap[len(viaGIW.mmap)-1]))
	l := &viaGIW.w.Layers[0]
	n := 0
	for _, m := range []*linalg.WeightMat{&l.QProj, &l.KProj, &l.VProj, &l.OProj, &l.GateProj, &l.UpProj, &l.DownProj} {
		raw, ok := m.Q4K()
		if !ok {
			t.Errorf("a %dx%d projection is %q in the bundle, want q4k (kind 8)", m.Rows(), m.Cols(), m.Kind())
			continue
		}
		if p := uintptr(unsafe.Pointer(&raw[0])); p < lo || p > hi {
			t.Errorf("a %dx%d q4k projection is not aliased from the mapping", m.Rows(), m.Cols())
		}
		n++
	}
	if n != 7 {
		t.Fatalf("%d of 7 projections came back as q4k", n)
	}
	d8bCompare(t, direct, viaGIW)
}

// TestD8b_q4kGGUFTakesItsSidecar is D8B-G2's decoder half: Load's default opens a q4k .gguf's sidecar, and the sidecar
// computes what the direct load does.
func TestD8b_q4kGGUFTakesItsSidecar(t *testing.T) {
	mem1NeedsSidecarPlatform(t)
	src := q4kTinyGGUF(t, t.TempDir())
	sidecarDefaultOn(t)
	m, err := Load(src, Options{Backend: "cpu", Quant: "q4k"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()
	if want := SidecarPath(src, "q4k", "cpu", false); m.srcPath != want || m.mmap == nil {
		t.Fatalf("Load opened %q (mapped %v), want the sidecar %q", m.srcPath, m.mmap != nil, want)
	}
	direct, err := Load(src, Options{Backend: "cpu", Quant: "q4k", DirectLoad: true})
	if err != nil {
		t.Fatal(err)
	}
	defer direct.Close()
	d8bCompare(t, direct, m)
}
