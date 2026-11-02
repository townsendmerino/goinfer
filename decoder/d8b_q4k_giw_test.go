package decoder

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"
	"unsafe"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/internal/giw"
	"github.com/townsendmerino/goinfer/internal/mem"
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
	return q4kGGUF(t, dir, 1)
}

// q4kGGUF is q4kTinyGGUF with the given number of layers.
func q4kGGUF(t *testing.T, dir string, layers int) string {
	t.Helper()
	return q4kGGUFSeed(t, dir, layers, 81)
}

// q4kGGUFSeed is q4kGGUF with other weights, at the same path.
func q4kGGUFSeed(t *testing.T, dir string, layers int, seed uint64) string {
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
		kvU32(arch+".block_count", uint32(layers)), kvU32(arch+".attention.head_count", heads),
		kvU32(arch+".attention.head_count_kv", kvHeads), kvU32(arch+".feed_forward_length", inter),
		kvF32(arch+".attention.layer_norm_rms_epsilon", 1e-6), kvF32(arch+".rope.freq_base", 10000),
		kvStr("tokenizer.ggml.model", "gpt2"), kvStrArr("tokenizer.ggml.tokens", toks),
	}
	r := rand.New(rand.NewPCG(seed, seed+1))
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
	}
	for l := range layers {
		b := fmt.Sprintf("blk.%d.", l)
		ts = append(ts,
			f32(b+"attn_norm.weight", []uint64{hidden}, hidden, true),
			q4k(b+"attn_q.weight", qDim, hidden),
			q4k(b+"attn_k.weight", kvDim, hidden),
			q4k(b+"attn_v.weight", kvDim, hidden),
			q4k(b+"attn_output.weight", hidden, qDim),
			f32(b+"ffn_norm.weight", []uint64{hidden}, hidden, true),
			q4k(b+"ffn_gate.weight", inter, hidden),
			q4k(b+"ffn_up.weight", inter, hidden),
			q4k(b+"ffn_down.weight", hidden, inter),
		)
	}
	p := filepath.Join(dir, fmt.Sprintf("q4k-%dlayer.gguf", layers))
	if err := os.WriteFile(p, buildGGUFRaw(kvs, ts), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// d8bLogits is m's logits, token by token, over d8bCompare's prompt.
func d8bLogits(t *testing.T, m *Model) []float32 {
	t.Helper()
	prompt := []int{3, 14, 15, 9, 26, 5, 35, 8}
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

// d8bCompare checks that a and b compute the same logits token by token and the same greedy tokens.
func d8bCompare(t *testing.T, a, b *Model) {
	t.Helper()
	prompt := []int{3, 14, 15, 9, 26, 5, 35, 8}
	if !slices.Equal(d8bLogits(t, a), d8bLogits(t, b)) {
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

// TestStreamedGIW_pagesOnlyTheQuantsTheFitGuardOffers holds mem.QuantPages to what a bundle loaded with StreamWeights does:
// the layer pager is built exactly at the quants the fit guard's retry and refusal offer weight streaming for. The source
// needs more layers than newLayerPager's smallest window, and tensors larger than a page, or that function returns nil at
// any quant. bf16 and f16 have no bundle form.
func TestStreamedGIW_pagesOnlyTheQuantsTheFitGuardOffers(t *testing.T) {
	const budget = 1 << 10 // far below one layer's weights
	dir := t.TempDir()
	src := q4kGGUF(t, dir, 8)
	for _, quant := range []string{"q4k", "f32", "int8", "int8int8", "int4", "int4mix"} {
		t.Run(quant, func(t *testing.T) {
			out := filepath.Join(t.TempDir(), quant+".giw")
			if err := TranscodeGIW(context.Background(), src, out, quant, false, GIWTargetForBackend("cpu")); err != nil {
				t.Fatal(err)
			}
			m, err := Load(out, Options{Backend: "cpu", StreamWeights: true, WeightCacheBytes: budget})
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			if got, want := m.layerPager != nil, mem.QuantPages(quant); got != want {
				t.Errorf("layer pager built = %v, mem.QuantPages = %v", got, want)
			}
			if m.pager != nil {
				t.Error("a dense model got an expert pager")
			}
		})
	}
}

// TestFitRefusal_streamableOnlyWhereItPages: a .gguf the fit guard refuses at a quant weight paging cannot read does not
// offer the weight-streaming retry or recommend the flag; at one it can read, the same file does both, so the result is the
// quant's doing. modelload's reading of the flag is TestLoad_retriesOnlyAStreamableRefusal; the pager's side is
// TestStreamedGIW_pagesOnlyTheQuantsTheFitGuardOffers.
func TestFitRefusal_streamableOnlyWhereItPages(t *testing.T) {
	src := q4kTinyGGUF(t, t.TempDir())
	prev := hostRAMAvailable
	hostRAMAvailable = func() int64 { return 1 }
	resetAvailProbeCache()
	t.Cleanup(func() {
		hostRAMAvailable = prev
		resetAvailProbeCache()
	})
	for quant, streamable := range map[string]bool{"q4k": false, "f32": false, "int8": true, "int8int8": true, "int4": true, "int4mix": true} {
		_, err := Load(src, Options{Backend: "cpu", Quant: quant, DirectLoad: true})
		var fde *FitDeclineError
		if !errors.As(err, &fde) {
			t.Fatalf("%s: %v, want the fit guard's refusal", quant, err)
		}
		if fde.DenseStreamable != streamable {
			t.Errorf("%s: DenseStreamable = %v, want %v", quant, fde.DenseStreamable, streamable)
		}
		if got := strings.Contains(err.Error(), "Re-run goinfer-serve with -stream-weights"); got != streamable {
			t.Errorf("%s: recommends -stream-weights = %v, want %v:\n%v", quant, got, streamable, err)
		}
	}
}

// TestGIW_f32WeightsAreMapped: an f32 bundle's weight matrices alias the mapping, as every other quant's do, so what
// guardGIWFit assumes of a bundle's weights holds at f32 too (docs/queue-engineering.md, B42). The bar on heap growth is the
// registered one. Its output is bit-identical to the direct load and to a kind-1 bundle, which still loads, copied. The
// mapping is read-only, so a forward that wrote through a weight would fault here.
func TestGIW_f32WeightsAreMapped(t *testing.T) {
	dir := t.TempDir()
	src := q4kGGUF(t, dir, 8)
	write := func(name string) string {
		out := filepath.Join(dir, name)
		if err := TranscodeGIW(context.Background(), src, out, "f32", false, GIWTargetForBackend("cpu")); err != nil {
			t.Fatal(err)
		}
		return out
	}
	inMapping := func(m *Model, w *linalg.WeightMat) bool {
		f, ok := w.F32()
		if !ok || len(f) == 0 {
			t.Fatalf("a projection is %q, want f32", w.Kind())
		}
		p := uintptr(unsafe.Pointer(&f[0]))
		return p >= uintptr(unsafe.Pointer(&m.mmap[0])) && p <= uintptr(unsafe.Pointer(&m.mmap[len(m.mmap)-1]))
	}
	projections := func(m *Model) []*linalg.WeightMat {
		var out []*linalg.WeightMat
		for l := range m.w.Layers {
			lw := &m.w.Layers[l]
			out = append(out, &lw.QProj, &lw.KProj, &lw.VProj, &lw.OProj, &lw.GateProj, &lw.UpProj, &lw.DownProj)
		}
		return out
	}

	out := write("f32.giw")
	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)
	m, err := Load(out, Options{Backend: "cpu"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	runtime.GC()
	runtime.ReadMemStats(&after)
	growth := int64(after.HeapAlloc) - int64(before.HeapAlloc)
	assertBundleCost(t, "Load", growth, int64(len(m.mmap)))
	for _, w := range projections(m) {
		if !inMapping(m, w) {
			t.Fatal("an f32 projection is not aliased from the mapping")
		}
	}
	// The embedding tables take the streaming writer's own path (giwWriter.streamTable), and are a real model's largest f32 tensors.
	if !inMapping(m, &m.w.Embed) || !inMapping(m, &m.w.LMHead) {
		t.Error("an f32 embedding table is not aliased from the mapping")
	}

	direct, err := Load(src, Options{Backend: "cpu", Quant: "f32", DirectLoad: true})
	if err != nil {
		t.Fatal(err)
	}
	defer direct.Close()
	d8bCompare(t, direct, m) // reads every weight through a read-only mapping

	forceGIWF32Kind1(t)
	kind1, err := Load(write("kind1.giw"), Options{Backend: "cpu"})
	if err != nil {
		t.Fatal(err)
	}
	defer kind1.Close()
	for _, w := range []*linalg.WeightMat{&kind1.w.Layers[0].QProj, &kind1.w.Embed, &kind1.w.LMHead} {
		if inMapping(kind1, w) {
			t.Error("a kind-1 bundle's f32 matrix is aliased; that encoding is unaligned and copied")
		}
	}
	d8bCompare(t, kind1, m)
}

// assertBundleCost holds a heap cost against a bundle's size, as this platform's mapping decides it: at most the registered 5% where
// a bundle is file-backed, Windows included (giwmap_windows.go), and the whole bundle where the mapping is a heap read.
func assertBundleCost(t *testing.T, what string, cost, bundle int64) {
	t.Helper()
	t.Logf("%s: %d bytes of heap for a %d-byte bundle: %.2f%% (file-backed mapping: %v)", what, cost, bundle, 100*float64(cost)/float64(bundle), giwMapIsFileBacked)
	if giwMapIsFileBacked && float64(cost) > 0.05*float64(bundle) {
		t.Errorf("%s cost %d bytes of heap for a %d-byte bundle, over the registered 5%%", what, cost, bundle)
	}
	if !giwMapIsFileBacked && float64(cost) < 0.9*float64(bundle) {
		t.Errorf("%s cost %d bytes of heap for a %d-byte bundle, where the mapping is a heap read: this platform has a real mapping now", what, cost, bundle)
	}
}

// forceGIWF32Kind1 makes this test's bundles carry kind 1, until t ends. giwF32Kind1ForTest is a package global, so a test that
// calls this must not run under t.Parallel.
func forceGIWF32Kind1(t *testing.T) {
	t.Helper()
	prev := giwF32Kind1ForTest
	giwF32Kind1ForTest = true
	t.Cleanup(func() { giwF32Kind1ForTest = prev })
}

// kind1Sidecar writes src's sidecar at quant in the kind-1 encoding, newer than its source, as a build before kind 10 left it.
func kind1Sidecar(t *testing.T, src, quant string, embedInt4 bool) string {
	t.Helper()
	cache := SidecarPath(src, quant, "cpu", embedInt4)
	t.Run("write a kind-1 sidecar", func(t *testing.T) {
		forceGIWF32Kind1(t)
		if err := TranscodeGIW(context.Background(), src, cache, quant, embedInt4, GIWTargetForBackend("cpu")); err != nil {
			t.Fatal(err)
		}
	})
	future := time.Now().Add(time.Hour)
	if err := os.Chtimes(cache, future, future); err != nil {
		t.Fatal(err)
	}
	return cache
}

// TestSidecar_copiedF32BundleIsRebuilt: a sidecar whose f32 weights are in the copied encoding is not fresh, or it would keep
// loading onto a heap the fit guard treats as file-backed; the rebuild aliases them. -embed-int4 changes nothing at f32: the
// embedding stays f32, which this pins; TestGIWCopies_readsPastAQuantizedEmbedding is the bundle whose first matrix says otherwise.
func TestSidecar_copiedF32BundleIsRebuilt(t *testing.T) {
	for _, embedInt4 := range []bool{false, true} {
		t.Run(fmt.Sprintf("embedInt4=%v", embedInt4), func(t *testing.T) {
			src := q4kGGUF(t, t.TempDir(), 2)
			cache := kind1Sidecar(t, src, "f32", embedInt4)
			t.Run("read the fixture", func(t *testing.T) {
				forceGIWF32Kind1(t) // a load of it is refused otherwise
				m, err := Load(cache, Options{Backend: "cpu"})
				if err != nil {
					t.Fatal(err)
				}
				defer m.Close()
				if got := m.w.Layers[0].QProj.Kind(); got != "f32" {
					t.Errorf("the fixture's projection is %q, want f32", got)
				}
				if got := m.w.Embed.Kind(); got != "f32" {
					t.Errorf("the fixture's embedding is %q with embedInt4=%v, want f32", got, embedInt4)
				}
			})
			if !bundleHeadIsCopied(t, cache) {
				t.Fatal("the fixture is not a copied-f32 bundle")
			}
			got, err := EnsureCachedGIW(context.Background(), src, "f32", "cpu", embedInt4)
			if err != nil {
				t.Fatal(err)
			}
			if got != cache {
				t.Errorf("EnsureCachedGIW = %s, want %s", got, cache)
			}
			if bundleHeadIsCopied(t, got) {
				t.Error("the sidecar was taken as fresh, and still copies its f32 weights")
			}
		})
	}
}

// bundleHeadIsCopied reports whether the bundle at path opens with a kind-1 matrix.
func bundleHeadIsCopied(t *testing.T, path string) bool {
	t.Helper()
	data, err := mapGIW(path)
	if err != nil {
		t.Fatal(err)
	}
	defer unmapGIW(data)
	blob, _, err := giw.Read(data)
	if err != nil {
		t.Fatal(err)
	}
	r, _, _, err := readGIWHeadReader(blob)
	if err != nil {
		t.Fatal(err)
	}
	return blob[r.off] == 1
}

// TestFitQuery_refusesACopiedF32Bundle: fit answers as the load does, rather than saying a bundle fits that the load refuses, reading
// only the bundle's head.
func TestFitQuery_refusesACopiedF32Bundle(t *testing.T) {
	src := q4kGGUF(t, t.TempDir(), 2)
	out := filepath.Join(t.TempDir(), "kind1.giw")
	t.Run("write", func(t *testing.T) {
		forceGIWF32Kind1(t)
		if err := TranscodeGIW(context.Background(), src, out, "f32", false, GIWTargetForBackend("cpu")); err != nil {
			t.Fatal(err)
		}
	})
	if _, err := FitQuery(out, Options{Backend: "cpu"}); !errors.Is(err, errCopiedF32Bundle) {
		t.Errorf("FitQuery: %v, want the copied-f32 refusal", err)
	}
	if err := GIWHeaderCheck(out); !errors.Is(err, errCopiedF32Bundle) {
		t.Errorf("GIWHeaderCheck: %v, want the copied-f32 refusal", err)
	}
}

// TestGIWWeightsQuant: the walk that reads a bundle's quant without copying its weights agrees with a full load, on each encoding
// and on the f32 bundle in both of its encodings; it is what -stream-weights is refused on, before the load.
func TestGIWWeightsQuant(t *testing.T) {
	dir := t.TempDir()
	src := q4kGGUF(t, dir, 2)
	check := func(name, quant, want string) {
		t.Helper()
		out := filepath.Join(dir, name+".giw")
		if err := TranscodeGIW(context.Background(), src, out, quant, false, GIWTargetForBackend("cpu")); err != nil {
			t.Fatal(err)
		}
		got, err := GIWWeightsQuant(out)
		if err != nil || got != want {
			t.Errorf("%s: GIWWeightsQuant = %q, %v; want %q", name, got, err, want)
		}
		m, err := Load(out, Options{Backend: "cpu"})
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		if l := m.w.quantLabel(); l != want {
			t.Errorf("%s: a full load labels it %q, want %q", name, l, want)
		}
	}
	for _, q := range []string{"q4k", "int8", "int8int8", "int4"} {
		check(q, q, q)
	}
	check("f32", "f32", "native")
	t.Run("kind 1", func(t *testing.T) {
		forceGIWF32Kind1(t)
		check("f32kind1", "f32", "native")
	})
	if _, err := GIWWeightsQuant(filepath.Join(dir, "missing.giw")); err == nil {
		t.Error("a missing bundle has no error")
	}
}

// TestGIWWeightsQuant_copiesNoWeights: reading a kind-1 bundle's quant allocates a small share of it, not the bundle: this walk runs
// before a refusal and before a sidecar's rebuild, where a copy is what they exist to avoid.
func TestGIWWeightsQuant_copiesNoWeights(t *testing.T) {
	forceGIWF32Kind1(t)
	dir := t.TempDir()
	out := filepath.Join(dir, "kind1.giw")
	if err := TranscodeGIW(context.Background(), q4kGGUF(t, dir, 8), out, "f32", false, GIWTargetForBackend("cpu")); err != nil {
		t.Fatal(err)
	}
	fi, err := os.Stat(out)
	if err != nil {
		t.Fatal(err)
	}
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	if _, err := GIWWeightsQuant(out); err != nil {
		t.Fatal(err)
	}
	runtime.ReadMemStats(&after)
	assertBundleCost(t, "reading the quant", int64(after.TotalAlloc-before.TotalAlloc), fi.Size())
}

// TestGIWCopiedF32_everyFamily: on every committed family fixture, a kind-1 f32 bundle is refused from its first weight record, by a
// load, by fit and by LoadSerializedWeights, with next to nothing allocated, and the no-copy walk still reads it. Each family's f32
// embedding is what the refusal keys on (opensCopiedF32), so a family this misses fails here.
func TestGIWCopiedF32_everyFamily(t *testing.T) {
	dirs, err := filepath.Glob(filepath.Join("..", "testdata", "*-tiny"))
	if err != nil {
		t.Fatal(err)
	}
	covered := 0
	for _, d := range dirs {
		if _, err := os.Stat(filepath.Join(d, "config.json")); err != nil {
			continue
		}
		out := filepath.Join(t.TempDir(), "m.giw")
		wrote := false
		t.Run("write "+filepath.Base(d), func(t *testing.T) {
			forceGIWF32Kind1(t)
			wrote = TranscodeGIW(context.Background(), d, out, "f32", false, GIWTargetForBackend("cpu")) == nil // else no bundle form at f32
		})
		if !wrote {
			continue
		}
		covered++
		if q, err := GIWWeightsQuant(out); err != nil || q != "native" {
			t.Errorf("%s: GIWWeightsQuant = %q, %v", d, q, err)
		}
		if m, err := Load(out, Options{Backend: "cpu"}); !errors.Is(err, errCopiedF32Bundle) {
			if err == nil {
				m.Close()
			}
			t.Errorf("%s: Load: %v, want the copied-f32 refusal", d, err)
		}
		if err := GIWHeaderCheck(out); !errors.Is(err, errCopiedF32Bundle) {
			t.Errorf("%s: GIWHeaderCheck: %v, want the copied-f32 refusal", d, err)
		}
		raw, err := os.ReadFile(out)
		if err != nil {
			t.Fatal(err)
		}
		blob, _, err := giw.Read(raw)
		if err != nil {
			t.Fatal(err)
		}
		// Refused from the head: a handful of allocations for the head's fields, and no weight.
		allocs := testing.AllocsPerRun(5, func() {
			if _, err := LoadSerializedWeights(blob); !errors.Is(err, errCopiedF32Bundle) {
				t.Errorf("%s: LoadSerializedWeights: %v, want the copied-f32 refusal", d, err)
			}
		})
		if allocs > 16 {
			t.Errorf("%s: %.0f allocations before the refusal: it read past the head", d, allocs)
		}
	}
	if covered < 10 {
		t.Fatalf("only %d family bundles were written: the glob or the fixtures moved", covered)
	}
	t.Logf("%d family bundles", covered)
}

// TestSidecar_copiedRouterStaysFresh is the other side of that rule: an int8 MoE bundle's f32 routers are in the copied encoding
// too, and they are small, so the bundle is not rebuilt for them.
func TestSidecar_copiedRouterStaysFresh(t *testing.T) {
	src := t.TempDir()
	for _, name := range []string{"config.json", "model.safetensors"} {
		b, err := os.ReadFile(filepath.Join("..", "testdata", "mixtral-tiny", name))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(src, name), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	cache := kind1Sidecar(t, src, "int8", false)
	before, err := os.Stat(cache)
	if err != nil {
		t.Fatal(err)
	}
	got, err := EnsureCachedGIW(context.Background(), src, "int8", "cpu", false)
	if err != nil {
		t.Fatal(err)
	}
	after, err := os.Stat(got)
	if err != nil {
		t.Fatal(err)
	}
	if got != cache || !after.ModTime().Equal(before.ModTime()) {
		t.Errorf("an int8 MoE sidecar was rebuilt for its routers: %s, mtime %v -> %v", got, before.ModTime(), after.ModTime())
	}
	m, err := Load(got, Options{Backend: "cpu"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	f, ok := m.w.Layers[0].Router.F32()
	if !ok || len(f) == 0 {
		t.Fatalf("the router is %q, want f32: this fixture no longer tests the rule", m.w.Layers[0].Router.Kind())
	}
	if p := uintptr(unsafe.Pointer(&f[0])); p >= uintptr(unsafe.Pointer(&m.mmap[0])) && p <= uintptr(unsafe.Pointer(&m.mmap[len(m.mmap)-1])) {
		t.Error("the router is aliased, so this bundle is not the kind-1 one this test needs")
	}
}

// TestLoad_refusesCopiedF32Bundle: a bundle named directly whose f32 weights are in the copied encoding is refused, before any of
// them is copied, naming the rebuild; a sidecar is rebuilt instead (TestSidecar_copiedF32BundleIsRebuilt). A kind-10 f32 bundle loads,
// and so does an int8 MoE bundle whose routers alone are kind 1.
func TestLoad_refusesCopiedF32Bundle(t *testing.T) {
	dir := t.TempDir()
	gguf := q4kGGUF(t, dir, 8)
	moe := filepath.Join("..", "testdata", "mixtral-tiny")
	write := func(name, src, quant string, kind1 bool) string {
		out := filepath.Join(dir, name)
		t.Run("write "+name, func(t *testing.T) {
			if kind1 {
				forceGIWF32Kind1(t)
			}
			if err := TranscodeGIW(context.Background(), src, out, quant, false, GIWTargetForBackend("cpu")); err != nil {
				t.Fatal(err)
			}
		})
		return out
	}
	kind1 := write("kind1.giw", gguf, "f32", true)
	fi, err := os.Stat(kind1)
	if err != nil {
		t.Fatal(err)
	}
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	m, err := Load(kind1, Options{Backend: "cpu"})
	runtime.ReadMemStats(&after)
	if err == nil {
		m.Close()
		t.Fatal("a kind-1 f32 bundle loaded")
	}
	if !errors.Is(err, errCopiedF32Bundle) {
		t.Errorf("%v is not the copied-f32 refusal", err)
	}
	var se *SerializeError
	if !errors.As(err, &se) {
		t.Errorf("%v is not a *SerializeError", err)
	}
	for _, want := range []string{"predates the mapped f32 encoding", "go run ./cmd/prequant -quant f32", "kind1.giw"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the refusal does not name %q: %v", want, err)
		}
	}
	// Where the mapping is a heap read, it has read the whole file before anything here runs (B43).
	if alloc := after.TotalAlloc - before.TotalAlloc; giwMapIsFileBacked && float64(alloc) > 0.05*float64(fi.Size()) {
		t.Errorf("%d bytes allocated before the refusal of a %d-byte bundle: it copied the weights first", alloc, fi.Size())
	}
	for _, c := range []struct{ name, path string }{
		{"kind-10 f32", write("kind10.giw", gguf, "f32", false)},
		{"int8 MoE with kind-1 routers", write("moe.giw", moe, "int8", true)},
	} {
		m, err := Load(c.path, Options{Backend: "cpu"})
		if err != nil {
			t.Errorf("%s: %v", c.name, err)
			continue
		}
		m.Close()
	}
}

// TestSidecar_rewriteWhileLoaded: a sidecar rebuilt from a changed source while a model holds it mapped gives the next load the new
// weights, and the loaded model keeps computing what it did. The source's weights change, so a fresh load reading the old bytes, or
// a mapping overwritten in place, each fails. With the replacement refused, as a mapped destination can refuse it on Windows, the
// old file moves aside and the same holds.
func TestSidecar_rewriteWhileLoaded(t *testing.T) {
	for _, refused := range []bool{false, true} {
		t.Run(fmt.Sprintf("replacement refused=%v", refused), func(t *testing.T) {
			dir := t.TempDir()
			src := q4kGGUF(t, dir, 2)
			cache, err := EnsureCachedGIW(context.Background(), src, "int8", "cpu", false)
			if err != nil {
				t.Fatal(err)
			}
			oldBytes, err := os.ReadFile(cache)
			if err != nil {
				t.Fatal(err)
			}
			loaded, err := Load(cache, Options{Backend: "cpu"})
			if err != nil {
				t.Fatal(err)
			}
			defer loaded.Close()
			want := d8bLogits(t, loaded)

			q4kGGUFSeed(t, dir, 2, 99) // new weights at the same path
			future := time.Now().Add(time.Hour)
			if err := os.Chtimes(src, future, future); err != nil { // newer than the sidecar, whatever the clock's grain
				t.Fatal(err)
			}
			if refused {
				prev := renameGIW
				t.Cleanup(func() { renameGIW = prev })
				failed := false
				renameGIW = func(from, to string) error {
					if to == cache && !failed {
						failed = true
						return errors.New("the destination is mapped")
					}
					return prev(from, to)
				}
			}
			got, err := EnsureCachedGIW(context.Background(), src, "int8", "cpu", false)
			if err != nil {
				t.Fatalf("rebuilding a sidecar a model holds mapped: %v", err)
			}
			if got != cache {
				t.Errorf("EnsureCachedGIW = %s, want %s", got, cache)
			}
			newBytes, err := os.ReadFile(cache)
			if err != nil {
				t.Fatal(err)
			}
			if bytes.Equal(oldBytes, newBytes) {
				t.Fatal("the sidecar's bytes did not change")
			}
			fresh, err := Load(cache, Options{Backend: "cpu"})
			if err != nil {
				t.Fatal(err)
			}
			defer fresh.Close()
			if slices.Equal(d8bLogits(t, fresh), want) {
				t.Error("a load after the rebuild computes the old weights")
			}
			if !slices.Equal(d8bLogits(t, loaded), want) {
				t.Error("the loaded model's output changed when its sidecar was replaced")
			}
		})
	}
}

// TestPublishGIW_restoresTheOldBundleOnFailure: when even the move-aside publish cannot complete, the old bundle is back at its name.
// A leftover from an earlier replacement is swept by the next publish, once nothing holds it.
func TestPublishGIW_restoresTheOldBundleOnFailure(t *testing.T) {
	dir := t.TempDir()
	out, tmp := filepath.Join(dir, "m.giw"), filepath.Join(dir, "m.tmp.giw")
	for p, b := range map[string]string{out: "old", tmp: "new"} {
		if err := os.WriteFile(p, []byte(b), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	prev := renameGIW
	t.Cleanup(func() { renameGIW = prev })
	renameGIW = func(from, to string) error {
		if to == out && from == tmp {
			return errors.New("refused")
		}
		return prev(from, to)
	}
	if err := publishGIW(tmp, out); err == nil {
		t.Fatal("a publish that cannot complete reported success")
	}
	if b, err := os.ReadFile(out); err != nil || string(b) != "old" {
		t.Errorf("the old bundle: %q, %v", b, err)
	}
	renameGIW = prev
	stale := out + giwReplacedInfix + "1-1"
	if err := os.WriteFile(stale, []byte("stale"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := publishGIW(tmp, out); err != nil {
		t.Fatal(err)
	}
	if b, err := os.ReadFile(out); err != nil || string(b) != "new" {
		t.Errorf("after a publish: %q, %v", b, err)
	}
	if _, err := os.Stat(stale); !os.IsNotExist(err) {
		t.Errorf("a leftover was not swept: %v", err)
	}
}
