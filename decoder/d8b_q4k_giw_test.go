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
	t.Logf("heap growth %d bytes across Load of a %d-byte f32 bundle: %.2f%%", growth, len(m.mmap), 100*float64(growth)/float64(len(m.mmap)))
	if float64(growth) > 0.05*float64(len(m.mmap)) {
		t.Errorf("heap grew %d bytes across Load of a %d-byte bundle, over the registered 5%%", growth, len(m.mmap))
	}
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
			m, err := Load(cache, Options{Backend: "cpu"})
			if err != nil {
				t.Fatal(err)
			}
			if got := m.w.Layers[0].QProj.Kind(); got != "f32" {
				t.Errorf("the fixture's projection is %q, want f32", got)
			}
			if got := m.w.Embed.Kind(); got != "f32" {
				t.Errorf("the fixture's embedding is %q with embedInt4=%v, want f32", got, embedInt4)
			}
			m.Close()
			if !giwCopiesF32Weights(cache) {
				t.Error("a copied-f32 bundle is not recognized as one")
			}
			got, err := EnsureCachedGIW(context.Background(), src, "f32", "cpu", embedInt4)
			if err != nil {
				t.Fatal(err)
			}
			if got != cache {
				t.Errorf("EnsureCachedGIW = %s, want %s", got, cache)
			}
			if giwCopiesF32Weights(got) {
				t.Error("the sidecar was taken as fresh, and still copies its f32 weights")
			}
		})
	}
}

// TestGIWCopies_readsPastAQuantizedEmbedding: a bundle whose embedding is quantized and whose body is copied f32 counts as copied.
// No writer path produces one today, and a reading of the first matrix alone would pass it. An int8 body with f32 routers does not
// count: TestSidecar_copiedRouterStaysFresh.
func TestGIWCopies_readsPastAQuantizedEmbedding(t *testing.T) {
	forceGIWF32Kind1(t)
	src := q4kGGUF(t, t.TempDir(), 2)
	m, err := Load(src, Options{Backend: "cpu", Quant: "f32", DirectLoad: true})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	m.w.Embed = quantizeWM(m.w.Embed, quantInt8)
	blob, err := SerializeWeights(m.w, "test")
	if err != nil {
		t.Fatal(err)
	}
	if !blobCopiesF32Weights(blob) {
		t.Error("an f32 body behind a quantized embedding is not recognized as copied")
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
	alloc := after.TotalAlloc - before.TotalAlloc
	t.Logf("%d bytes allocated reading the quant of a %d-byte bundle", alloc, fi.Size())
	if float64(alloc) > 0.05*float64(fi.Size()) {
		t.Errorf("reading the quant allocated %d bytes of a %d-byte bundle: it copied the weights", alloc, fi.Size())
	}
}

// TestGIWWalk_readsEveryFamilyBundle holds the walk to the reader's other checks on every committed family fixture: a kind-1 matrix
// it skips must not make a family's own read fail, or giwCopiesF32Weights would answer false for that family's bundles.
func TestGIWWalk_readsEveryFamilyBundle(t *testing.T) {
	forceGIWF32Kind1(t)
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
		if err := TranscodeGIW(context.Background(), d, out, "f32", false, GIWTargetForBackend("cpu")); err != nil {
			continue // a family with no bundle form at f32
		}
		covered++
		if !giwCopiesF32Weights(out) {
			t.Errorf("%s: a kind-1 f32 bundle is not recognized as copied", d)
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
