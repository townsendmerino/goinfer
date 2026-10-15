//go:build cuda && goinfer_testhooks

package cuda

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/goinfer/decoder"
)

// KV2's copy gates on CUDA (docs/measurements/kv2-cuda-2026-10-15.md, G2-C): CopyKV moves exactly the rows it names, through
// each slot's own windowed view, and a prefix taken by copy is the prefix computed.
//
//	go test -count=1 -timeout 20m -tags 'cuda goinfer_testhooks' -run '^TestKV2_' -v ./cuda/
//
// GOINFER_HEAVY_TESTS=1 adds the 1.5B.

type kv2Rig struct {
	t     *testing.T
	m     *decoder.Model
	r     *cudaResident
	vocab int
	ctx   context.Context
}

func kv2Load(t *testing.T, path string, opts decoder.Options) *kv2Rig {
	t.Helper()
	opts.Backend, opts.Quant = "cuda", "int4"
	m, err := decoder.Load(path, opts)
	if err != nil {
		t.Fatalf("load %s: %v", path, err)
	}
	t.Cleanup(func() { _ = m.Close() })
	r := mustResident(t, m)
	if want := max(1, opts.ResidentKVSlots); r.KVSlots() != want {
		t.Fatalf("%s: %d KV slots granted, the test needs %d", path, r.KVSlots(), want)
	}
	_, _, _, _, _, _, vocab := m.Dims()
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
	t.Cleanup(cancel)
	return &kv2Rig{t: t, m: m, r: r, vocab: vocab, ctx: ctx}
}

// toks is n token ids from a stream named by seed.
func (g *kv2Rig) toks(n int, seed uint32) []int {
	s := 2463534242 + seed*2654435761
	out := make([]int, n)
	for i := range out {
		s ^= s << 13
		s ^= s >> 17
		s ^= s << 5
		out[i] = int(s % uint32(g.vocab-1))
	}
	return out
}

// fill binds slot and writes ids at positions from, from+1, ...: one batched prefill call when batched, else one decode
// forward per token. It returns the logits of every forward (one entry for a batched call).
func (g *kv2Rig) fill(slot int, ids []int, from int, batched bool) [][]float32 {
	g.t.Helper()
	if err := g.r.UseKVSlot(slot); err != nil {
		g.t.Fatal(err)
	}
	if batched {
		e := make([][]float32, len(ids))
		for i, id := range ids {
			e[i] = append([]float32(nil), g.m.EmbedResidentForTest(id)...)
		}
		lg, err := g.r.PrefillLast(g.ctx, e, from)
		if err != nil {
			g.t.Fatalf("PrefillLast(%d rows at %d) in slot %d: %v", len(ids), from, slot, err)
		}
		return [][]float32{append([]float32(nil), lg...)}
	}
	out := make([][]float32, 0, len(ids))
	for i, id := range ids {
		lg, err := g.r.Forward(g.m.EmbedResidentForTest(id), from+i)
		if err != nil {
			g.t.Fatalf("Forward at %d in slot %d: %v", from+i, slot, err)
		}
		out = append(out, append([]float32(nil), lg...))
	}
	return out
}

// kv2Buf is one K or V buffer of a layer, as kv2Rig.raw reads it.
type kv2Buf struct {
	layer int
	v     bool
	win   bool
	row   int // bytes a position takes
}

// bufs lists every buffer a slot owns: K and V of each layer that does not read another layer's KV, V only where it exists.
func (g *kv2Rig) bufs() []kv2Buf {
	var out []kv2Buf
	for l := range g.r.layers {
		Ly := &g.r.layers[l]
		if Ly.kvShared {
			continue
		}
		elem := 4
		if Ly.kvHalf {
			elem = 2
		}
		for _, v := range []bool{false, true} {
			if v && g.r.kvSlotBufs[0].vc[l].Len() == 0 {
				continue
			}
			out = append(out, kv2Buf{layer: l, v: v, win: Ly.kvWin, row: Ly.kvDim * elem})
		}
	}
	return out
}

// raw is every buffer of slot's KV as bytes, in bufs' order, and the slot's windowed base.
func (g *kv2Rig) raw(slot int) (out [][]byte, base int) {
	g.t.Helper()
	err := g.r.do(func() error {
		if e := g.r.stream.Sync(); e != nil {
			return e
		}
		for _, b := range g.bufs() {
			buf := g.r.kvSlotBufs[slot].kc[b.layer]
			if b.v {
				buf = g.r.kvSlotBufs[slot].vc[b.layer]
			}
			elem := 4
			if g.r.layers[b.layer].kvHalf {
				elem = 2
			}
			p := make([]byte, buf.Len()*elem)
			if e := gpu.Download(buf, p); e != nil {
				return e
			}
			out = append(out, p)
		}
		if g.r.kvWin {
			base = g.r.kvBases[slot]
		}
		return nil
	})
	if err != nil {
		g.t.Fatalf("read slot %d: %v", slot, err)
	}
	return out, base
}

func kv2SameRaw(a, b [][]byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !bytes.Equal(a[i], b[i]) {
			return false
		}
	}
	return true
}

// kv2CheckCopy is G2-C(a)'s rule for one CopyKV(src, dst, start, end) that returned nil: srcB/dstB/byB are slots src, dst
// and a bystander before it, srcA/dstA/byA after.
func (g *kv2Rig) kv2CheckCopy(start, end int, srcB, dstB, byB, srcA, dstA, byA [][]byte, srcBase, dstBaseA int) {
	g.t.Helper()
	if !kv2SameRaw(srcB, srcA) {
		g.t.Errorf("[%d, %d): the source slot changed", start, end)
	}
	if !kv2SameRaw(byB, byA) {
		g.t.Errorf("[%d, %d): a slot outside the copy changed", start, end)
	}
	lo := 0
	if g.r.kvWin {
		lo = max(0, end-(g.r.kvWindow-1))
		if dstBaseA != lo {
			g.t.Errorf("[%d, %d): destination base %d, want %d", start, end, dstBaseA, lo)
		}
	}
	differedBefore := false
	for i, b := range g.bufs() {
		s, d0, d := srcB[i], dstB[i], dstA[i]
		if b.win {
			n := (end - lo) * b.row
			want := s[(lo-srcBase)*b.row : (lo-srcBase)*b.row+n]
			differedBefore = differedBefore || !bytes.Equal(d0[:n], want)
			if !bytes.Equal(d[:n], want) {
				g.t.Errorf("[%d, %d): layer %d %s (windowed): the destination's rows [%d, %d) are not the source's", start, end, b.layer, kv2KV(b.v), lo, end)
			}
			continue
		}
		in0, in1 := start*b.row, end*b.row
		differedBefore = differedBefore || !bytes.Equal(d0[in0:in1], s[in0:in1])
		if !bytes.Equal(d[in0:in1], s[in0:in1]) {
			g.t.Errorf("[%d, %d): layer %d %s: the destination's rows in the range are not the source's", start, end, b.layer, kv2KV(b.v))
		}
		if !bytes.Equal(d[:in0], d0[:in0]) || !bytes.Equal(d[in1:], d0[in1:]) {
			g.t.Errorf("[%d, %d): layer %d %s: the destination changed outside the range", start, end, b.layer, kv2KV(b.v))
		}
	}
	if !differedBefore {
		g.t.Errorf("[%d, %d): the slots already agreed in the range before the copy, so the check could not see a copy that did nothing", start, end)
	}
}

func kv2KV(v bool) string {
	if v {
		return "V"
	}
	return "K"
}

// kv2CopyExact is G2-C(a) on a whole-context fixture: three ranges, the destination refilled before each.
func kv2CopyExact(t *testing.T, g *kv2Rig, batched bool) {
	hi := min(1024, g.r.ctxCap/64*64)
	if hi < 320 {
		t.Fatalf("context %d is too short for the ranges", g.r.ctxCap)
	}
	g.fill(0, g.toks(hi, 1), 0, batched)
	g.fill(2, g.toks(hi, 3), 0, batched)
	for _, rg := range [][2]int{{0, 64}, {0, hi}, {256, hi}} {
		g.fill(1, g.toks(hi, 2), 0, batched)
		srcB, sb := g.raw(0)
		dstB, _ := g.raw(1)
		byB, _ := g.raw(2)
		if err := g.r.CopyKV(0, 1, rg[0], rg[1]); err != nil {
			t.Fatalf("CopyKV [%d, %d): %v", rg[0], rg[1], err)
		}
		srcA, _ := g.raw(0)
		dstA, db := g.raw(1)
		byA, _ := g.raw(2)
		g.kv2CheckCopy(rg[0], rg[1], srcB, dstB, byB, srcA, dstA, byA, sb, db)
	}
	if err := g.r.CopyKV(0, 1, 0, g.r.ctxCap/64*64+64); !kv2IsDecline(err) {
		t.Errorf("a range past the cache: got %v, want a KVCopyDecline", err)
	}
}

func kv2IsDecline(err error) bool {
	var d *decoder.KVCopyDecline
	return errors.As(err, &d)
}

// kv2PrefixIsComputed is G2-C(b): a prompt sharing a 1,000-token preamble (shorter where the context is) with another
// slot's copies the block-aligned rows of it (and, in the second arm, only the rows past the 128 it already holds), prefills the rest, and must equal the same
// prompt prefilled cold in a third slot: every K/V row of the prompt and the last logits.
func kv2PrefixIsComputed(t *testing.T, g *kv2Rig, batched bool) {
	pre := g.toks(min(1000, g.r.ctxCap-64), 11)
	a := append(append([]int(nil), pre...), g.toks(30, 12)...)
	b := append(append([]int(nil), pre...), g.toks(40, 13)...)
	hit := len(pre) / 64 * 64
	if hit < 256 {
		t.Fatalf("context %d is too short for a preamble", g.r.ctxCap)
	}
	g.fill(0, a, 0, batched)
	ref := g.fill(2, b, 0, batched)
	refKV, _ := g.raw(2)
	for _, held := range []int{0, 128} {
		// as long as the hit, so a batched fill runs the same kernel class over the held rows as the cold prefill does
		junk := g.toks(hit, 14)
		copy(junk, pre[:held])
		g.fill(1, junk, 0, batched)
		if err := g.r.CopyKV(0, 1, held, hit); err != nil {
			t.Fatalf("CopyKV [%d, %d): %v", held, hit, err)
		}
		if err := g.r.UseKVSlot(1); err != nil {
			t.Fatal(err)
		}
		got := g.fill(1, b[hit:], hit, batched)
		gotKV, _ := g.raw(1)
		if j, ok := sameBits(got[len(got)-1], ref[len(ref)-1]); !ok {
			t.Errorf("held %d: the last logits differ from the cold prefill's (first at %d)", held, j)
		}
		for i, bf := range g.bufs() {
			n := len(b) * bf.row
			if !bytes.Equal(gotKV[i][:n], refKV[i][:n]) {
				t.Errorf("held %d: layer %d %s: the prompt's rows differ from the cold prefill's", held, bf.layer, kv2KV(bf.v))
			}
		}
	}
}

func TestKV2_copyExact_tiny(t *testing.T) {
	const dir = "../testdata/llama-attnfa-tiny"
	requireDeviceAndFixture(t, dir)
	for _, kv := range []string{"f32", "f16"} {
		t.Run(kv, func(t *testing.T) {
			g := kv2Load(t, dir, decoder.Options{ResidentKVSlots: 3, ResidentContext: 2048, KVPrecision: kv})
			kv2CopyExact(t, g, true)
			kv2PrefixIsComputed(t, g, true)
		})
	}
}

// TestKV2_copyExact_otherLayouts covers the layer kinds the dense fixture has none of: an MLA latent (K only) and layers
// that read another layer's KV (copied with their source).
// Both fixtures declare a short context (64 and 128 positions), so they load from a copy whose config allows 2,048: the
// weights are the fixture's, and RoPE serves any position.
func TestKV2_copyExact_otherLayouts(t *testing.T) {
	for _, fx := range []string{"deepseek-tiny", "gemma4-emodel-tiny"} {
		t.Run(fx, func(t *testing.T) {
			dir := filepath.Join("..", "testdata", fx)
			requireDeviceAndFixture(t, dir)
			g := kv2Load(t, kv2WideFixture(t, dir, 2048), decoder.Options{ResidentKVSlots: 3, ResidentContext: 2048})
			mla, shared := 0, 0
			for l := range g.r.layers {
				if g.r.layers[l].isMLA {
					mla++
				}
				if g.r.layers[l].kvShared {
					shared++
				}
			}
			if (fx == "deepseek-tiny" && mla == 0) || (fx == "gemma4-emodel-tiny" && shared == 0) {
				t.Fatalf("%s: %d MLA layers, %d shared-KV layers: the layout this case covers is absent", fx, mla, shared)
			}
			t.Logf("%d layers: %d MLA latent, %d read another layer's KV; context %d", len(g.r.layers), mla, shared, g.r.ctxCap)
			kv2CopyExact(t, g, false)
			kv2PrefixIsComputed(t, g, false)
		})
	}
}

func TestKV2_copyExact_real1p5B(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads the 1.5B)")
	}
	path := modelPath("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	requireDeviceAndFile(t, path)
	for _, kv := range []string{"f32", "f16"} {
		t.Run(kv, func(t *testing.T) {
			g := kv2Load(t, path, decoder.Options{ResidentKVSlots: 3, ResidentContext: 2048, KVPrecision: kv})
			kv2CopyExact(t, g, true)
			kv2PrefixIsComputed(t, g, true)
		})
	}
}

// TestKV2_copyWindowed is G2-C on the windowed fixtures, with the slack lowered so every slot's base is past 0: the copy
// reads the source through its shifted view, the destination continues from the copied window bit for bit as a slot
// that computed the prompt itself, and a source that has moved past the rows declines with the destination untouched.
func TestKV2_copyWindowed(t *testing.T) { eachWindowFixture(t, kv2CopyWindowed) }

func kv2CopyWindowed(t *testing.T, fx string) {
	requireCUDADevice(t)
	withWindowedKVSlack(t, windowedKVTestSlack)
	g := kv2Load(t, filepath.Join("..", "testdata", fx), decoder.Options{ResidentKVSlots: 3, ResidentWindowedKV: true})
	if !g.r.kvWin {
		t.Fatalf("windowed KV did not engage (ctxCap %d, window %d, slack %d)", g.r.ctxCap, g.r.kvWindow, kvWindowSlack)
	}
	const end = 256
	pre := g.toks(end, 21)
	for _, held := range []int{0, 128} {
		g.fill(0, pre, 0, false)
		junk := g.toks(300, 22)
		copy(junk, pre[:held])
		g.fill(1, junk, 0, false)
		g.fill(2, g.toks(200, 23), 0, false)
		srcB, sb := g.raw(0)
		dstB, _ := g.raw(1)
		byB, _ := g.raw(2)
		if sb == 0 {
			t.Fatal("the source's base is 0: the copy would not read through a shifted view")
		}
		if err := g.r.CopyKV(0, 1, held, end); err != nil {
			t.Fatalf("held %d: CopyKV: %v", held, err)
		}
		srcA, _ := g.raw(0)
		dstA, db := g.raw(1)
		byA, _ := g.raw(2)
		g.kv2CheckCopy(held, end, srcB, dstB, byB, srcA, dstA, byA, sb, db)

		// the destination continues as a slot that computed the prompt itself
		tail := g.toks(200, 24)
		got := g.fill(1, tail, end, false)
		want := g.fill(2, append(append([]int(nil), pre...), tail...), 0, false)[end:]
		for i := range got {
			if j, ok := sameBits(got[i], want[i]); !ok {
				t.Fatalf("held %d: step %d (position %d): logits differ from the slot that computed the prompt (first at %d)", held, i, end+i, j)
			}
		}
	}

	// a source that has moved on past the window the destination needs
	g.fill(0, append(append([]int(nil), pre...), g.toks(100, 25)...), 0, false)
	g.fill(1, g.toks(300, 26), 0, false)
	srcB, sb := g.raw(0)
	dstB, dbB := g.raw(1)
	if sb <= end-(g.r.kvWindow-1) {
		t.Fatalf("the source's base %d still holds the rows from %d: the decline case is not set up", sb, end-(g.r.kvWindow-1))
	}
	if got := g.r.ReusableKVSlot(0, end); got != 0 {
		t.Errorf("ReusableKVSlot(0, %d) = %d with the source's base at %d, want 0", end, got, sb)
	}
	if got := g.r.ReusableKVSlot(0, sb+g.r.kvWindow-1); got != sb+g.r.kvWindow-1 {
		t.Errorf("ReusableKVSlot(0, %d) = %d, want it whole: that window is held", sb+g.r.kvWindow-1, got)
	}
	err := g.r.CopyKV(0, 1, 0, end)
	if !kv2IsDecline(err) {
		t.Fatalf("CopyKV from a source based at %d: got %v, want a KVCopyDecline", sb, err)
	}
	t.Logf("declined: %v", err)
	srcA, _ := g.raw(0)
	dstA, dbA := g.raw(1)
	if !kv2SameRaw(dstB, dstA) || dbA != dbB {
		t.Errorf("a declined copy changed the destination (base %d -> %d)", dbB, dbA)
	}
	if !kv2SameRaw(srcB, srcA) {
		t.Errorf("a declined copy changed the source")
	}
}

// TestKV2_copyArgs: malformed requests are errors, and a resident with one slot or recurrent state declines.
func TestKV2_copyArgs(t *testing.T) {
	const dir = "../testdata/llama-attnfa-tiny"
	requireDeviceAndFixture(t, dir)
	for _, kv := range []string{"f32", "f16"} {
		t.Run(kv, func(t *testing.T) {
			g := kv2Load(t, dir, decoder.Options{ResidentKVSlots: 2, ResidentContext: 1024, KVPrecision: kv})
			for _, c := range [][4]int{{0, 0, 0, 64}, {0, 2, 0, 64}, {-1, 1, 0, 64}, {0, 1, 1, 64}, {0, 1, 0, 63}, {0, 1, 128, 64}} {
				if err := g.r.CopyKV(c[0], c[1], c[2], c[3]); err == nil || kv2IsDecline(err) {
					t.Errorf("CopyKV%v: got %v, want a plain error", c, err)
				}
			}
			if err := g.r.CopyKV(0, 1, 64, 64); err != nil {
				t.Errorf("an empty range: %v", err)
			}
			old := kvCopyF16
			kvCopyF16 = false
			defer func() { kvCopyF16 = old }()
			err := g.r.CopyKV(0, 1, 0, 64)
			if f16 := kv == "f16"; f16 != kv2IsDecline(err) || (!f16 && err != nil) {
				t.Errorf("kvCopyF16 off, %s cache: got %v; an f16 cache must decline and an f32 one copy", kv, err)
			}
		})
	}
}

// kv2WideFixture is fixture dir with every max_position_embeddings in its config set to ctx: a temporary directory whose
// other files link to the fixture's.
func kv2WideFixture(t *testing.T, dir string, ctx int) string {
	t.Helper()
	out := t.TempDir()
	ents, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range ents {
		src, _ := filepath.Abs(filepath.Join(dir, e.Name()))
		if e.Name() != "config.json" {
			if err := os.Symlink(src, filepath.Join(out, e.Name())); err != nil {
				t.Fatal(err)
			}
			continue
		}
		b, err := os.ReadFile(src)
		if err != nil {
			t.Fatal(err)
		}
		var cfg map[string]any
		if err := json.Unmarshal(b, &cfg); err != nil {
			t.Fatal(err)
		}
		var widen func(m map[string]any)
		widen = func(m map[string]any) {
			for k, v := range m {
				if k == "max_position_embeddings" {
					m[k] = ctx
				} else if sub, ok := v.(map[string]any); ok {
					widen(sub)
				}
			}
		}
		widen(cfg)
		if b, err = json.Marshal(cfg); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(out, "config.json"), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return out
}
