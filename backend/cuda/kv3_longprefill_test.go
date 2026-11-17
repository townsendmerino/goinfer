//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/runtime/modelload"
	"hash/fnv"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"
)

// KV3's gates for the span bound (docs/measurements/kv3-cuda-2026-10-13.md §2.3): the shared-memory guard holds a layer to the
// exact kernel's bound only when the fused kernel does not serve it, and a batched pass past the bound agrees with the
// per-token path, which is what the decline fell back to.

// TestKV3_shmemGuardEquivalence: for every span at or under the bound the guard answers as it always did, whatever the fused
// predicate says; past the bound it declines a layer exactly when the fused kernel does not serve it. Struct-only, no device.
func TestKV3_shmemGuardEquivalence(t *testing.T) {
	spans := []int{1, 63, 64, 511, 512, 4095, 12159, 12160}
	over := []int{12161, 12288, 16384, 20000, 100000}
	for _, windows := range [][]int32{{0}, {0, 4096}, {128, 0}, {4096}} {
		for _, hd := range []int{4, 64, 96, 128, 256} {
			r := declineFixture(len(windows), "int4")
			for l, w := range windows {
				r.layers[l].window, r.layers[l].hd = w, hd
			}
			for _, fusedHD := range []func(int) bool{nil, func(int) bool { return false }, func(h int) bool { return h == 64 || h == 128 }} {
				for _, n := range spans {
					for _, M := range []int{1, 16, 512} {
						if M > n {
							continue
						}
						start := n - M
						old, now := r.checkPrefillShmem(start, M, nil), r.checkPrefillShmem(start, M, fusedHD)
						if old != nil || now != nil {
							t.Fatalf("windows %v hd %d span %d: guard declined at or under the bound (old %v, new %v)", windows, hd, n, old, now)
						}
					}
				}
				for _, n := range over {
					want := false // does some layer exceed the bound and have no fused kernel?
					for _, w := range windows {
						span := n
						if w > 0 && int(w) < span {
							span = int(w)
						}
						if splitKVRequired(span) && !(fusedHD != nil && fusedHD(hd)) {
							want = true
						}
					}
					if got := r.checkPrefillShmem(n-512, 512, fusedHD) != nil; got != want {
						t.Errorf("windows %v hd %d span %d fused=%v: declined %v, want %v", windows, hd, n, fusedHD != nil, got, want)
					}
				}
			}
		}
	}
}

// kv3LongContextCopy writes a copy of a fixture directory whose config allows maxPos positions, with the weights symlinked.
func kv3LongContextCopy(t *testing.T, src string, maxPos int) string {
	t.Helper()
	dst := t.TempDir()
	cfg, err := os.ReadFile(filepath.Join(src, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	re := regexp.MustCompile(`"max_position_embeddings":\s*\d+`)
	if !re.Match(cfg) {
		t.Fatalf("%s: no max_position_embeddings to raise", src)
	}
	cfg = re.ReplaceAll(cfg, []byte(fmt.Sprintf(`"max_position_embeddings": %d`, maxPos)))
	if err := os.WriteFile(filepath.Join(dst, "config.json"), cfg, 0o644); err != nil {
		t.Fatal(err)
	}
	abs, _ := filepath.Abs(src)
	entries, _ := os.ReadDir(src)
	for _, e := range entries {
		if e.Name() != "config.json" {
			if err := os.Symlink(filepath.Join(abs, e.Name()), filepath.Join(dst, e.Name())); err != nil {
				t.Fatal(err)
			}
		}
	}
	return dst
}

// kv3Score runs one prompt through the new batched pass and through the per-token path and returns the two last-token logit rows.
// perToken is the reference; control replaces the last 8 prompt tokens of the batched arm's input (nil: none).
func kv3Score(t *testing.T, g *kv0Rig, ids []int, controlIDs []int) (batched, ref, ctl []float32, launches int) {
	t.Helper()
	e := g.embs(ids)
	g.poison(len(ids))
	a0, _ := g.launches()
	batched = g.prefill(e, 0) // must not decline: a decline is an error here
	a1, _ := g.launches()
	launches = a1 - a0
	if controlIDs != nil {
		g.poison(len(ids))
		ctl = g.prefill(g.embs(controlIDs), 0)
	}
	// the per-token path: ForwardNoLogits for all but the last row, Forward for the last, as the decoder's own fallback does
	g.poison(len(ids))
	last := time.Now()
	for i := 0; i < len(ids)-1; i++ {
		if err := g.r.ForwardNoLogits(e[i], i); err != nil {
			t.Fatalf("per-token row %d: %v", i, err)
		}
		if time.Since(last) > 30*time.Second {
			fmt.Fprintf(os.Stderr, "[kv3] per-token reference: %d of %d rows\n", i+1, len(ids))
			last = time.Now()
		}
	}
	out, err := g.r.Forward(e[len(ids)-1], len(ids)-1)
	if err != nil {
		t.Fatalf("per-token last row: %v", err)
	}
	ref = append([]float32(nil), out...)
	return batched, ref, ctl, launches
}

func kv3Check(t *testing.T, name string, batched, ref, ctl []float32, launches, layers int, gradeArgmax bool) {
	t.Helper()
	cos := kv0Cosine(batched, ref)
	same := kv0Argmax(batched) == kv0Argmax(ref)
	// the reference's margin: top-1 minus top-2, against 3% of |top-1| (the project's near-tie rule)
	t1, t2 := float32(math.Inf(-1)), float32(math.Inf(-1))
	for _, v := range ref {
		if v > t1 {
			t1, t2 = v, t1
		} else if v > t2 {
			t2 = v
		}
	}
	near := float64(t1-t2) < 0.03*math.Abs(float64(t1))
	fmt.Fprintf(os.Stderr, "[kv3] %-28s cosine %.6f argmax equal %v (reference margin %.4f, near tie %v), fused attention launches %d\n", name, cos, same, t1-t2, near, launches)
	if launches < layers {
		t.Errorf("%s: %d fused attention launches for %d layers: the pass did not run the fused kernel past the bound", name, launches, layers)
	}
	if cos < 0.99 {
		t.Errorf("%s: cosine %.6f is under the registered bar 0.99", name, cos)
	}
	if gradeArgmax && !same && !near {
		t.Errorf("%s: argmax differs from the per-token path and the reference is not a near tie", name)
	}
	if ctl != nil {
		if c := kv0Cosine(ctl, ref); c >= 0.99 {
			t.Errorf("%s: CONTROL cosine %.6f is not under the bar: the comparison cannot see a changed last token, the set is void", name, c)
		} else {
			fmt.Fprintf(os.Stderr, "[kv3] %-28s control (last 8 tokens replaced) cosine %.6f: the comparison sees it\n", name, c)
		}
	}
}

// TestKV3_batchedPrefillPastTheSpanBound_tiny: prompts of 12,400 to 16,384 tokens (past the 12,160-key bound) prefill in
// batched fused passes and agree with the per-token path. Tiny fixture, random tokens; its argmax is not graded.
//
//	go test -count=1 -timeout 30m -tags 'cuda goinfer_testhooks' -run '^TestKV3_batchedPrefillPastTheSpanBound_tiny$' -v ./cuda/
func TestKV3_batchedPrefillPastTheSpanBound_tiny(t *testing.T) {
	const base = "../../testdata/llama-attnfa-tiny"
	requireDeviceAndFixture(t, base)
	dir := kv3LongContextCopy(t, base, 32768)
	for _, kv := range []string{"f32", "f16"} {
		t.Run(kv, func(t *testing.T) {
			g := kv0LoadCtx(t, dir, kv, 17000, kv0RandomTokens)
			for _, n := range []int{12400, 14000, 16384} {
				for seed := 0; seed < 3; seed++ {
					ids := g.tokens(n, seed*7919)
					ctlIDs := append(append([]int(nil), ids[:n-8]...), g.tokens(8, seed*7919+31)...)
					b, ref, ctl, l := kv3Score(t, g, ids, ctlIDs)
					kv3Check(t, fmt.Sprintf("tiny %s n=%d seed=%d", kv, n, seed), b, ref, ctl, l, len(g.r.layers), false)
				}
			}
		})
	}
}

// TestKV3_batchedPrefillPastTheSpanBound_real1p5B: the same on the real Qwen2.5-Coder-1.5B at int4 with this repository's own
// source as the text. KV3_LENGTHS (comma-separated) and KV3_WINDOWS set the set; the default is one 12,500-token prompt, the
// day check. The registered set (12,500, 16,384 and 20,000 tokens, two windows each) runs on the night queue.
func TestKV3_batchedPrefillPastTheSpanBound_real1p5B(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads the 1.5B)")
	}
	path := modelPath("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	requireDeviceAndFile(t, path)
	lengths, windows := []int{12500}, 1
	if v := os.Getenv("KV3_LENGTHS"); v != "" {
		lengths = nil
		for _, f := range regexp.MustCompile(`\d+`).FindAllString(v, -1) {
			var n int
			fmt.Sscan(f, &n)
			lengths = append(lengths, n)
		}
	}
	if v := os.Getenv("KV3_WINDOWS"); v != "" {
		fmt.Sscan(v, &windows)
	}
	for _, kv := range []string{"f16"} {
		t.Run(kv, func(t *testing.T) {
			g := kv0LoadCtx(t, path, kv, 21000, kv0TextTokens(t, path))
			for _, n := range lengths {
				for w := 0; w < windows; w++ {
					ids := g.tokens(n, 40000*(w+1)+n)
					ctlIDs := append(append([]int(nil), ids[:n-8]...), g.tokens(8, 99+w)...)
					b, ref, ctl, l := kv3Score(t, g, ids, ctlIDs)
					kv3Check(t, fmt.Sprintf("1.5B %s n=%d window=%d", kv, n, w), b, ref, ctl, l, len(g.r.layers), true)
				}
			}
		})
	}
}

// TestKV3_prefillScratchBytes_matchesThePass holds PrefillScratchBytes to the buffers prefillCore asks for: the figure admission
// trusts is the sum of each scratch buffer after the allocator's rounding, not an estimate, and it is never under the pass.
func TestKV3_prefillScratchBytes_matchesThePass(t *testing.T) {
	const base = "../../testdata/llama-attnfa-tiny"
	requireDeviceAndFixture(t, base)
	g := kv0LoadCtx(t, base, "f16", 2048, kv0RandomTokens)
	for _, M := range []int{16, 300, 512} {
		var seen int64
		prefillScratchSeenForTest = func(b int64) { seen += b + allocRoundSlack(b) }
		g.prefill(g.embs(g.tokens(M, 5)), 0)
		prefillScratchSeenForTest = nil
		if want := g.r.PrefillScratchBytes(M); want != seen {
			t.Errorf("M=%d: PrefillScratchBytes %d B, the pass allocated %d B", M, want, seen)
		} else {
			fmt.Fprintf(os.Stderr, "[kv3] tiny M=%d: scratch %d B (%.1f MiB) = the pass's allocations\n", M, seen, float64(seen)/(1<<20))
		}
	}
}

// TestKV3_residentReportsItsChunks pins the wiring the decoder reads (the interfaces are matched structurally, so a rename here
// would silently put admission back on the whole-prompt bound): the chunk is the one prefillChunked cuts by, a multiple of the block
// size 64, and the tail minimum is the fused attention's.
func TestKV3_residentReportsItsChunks(t *testing.T) {
	const base = "../../testdata/llama-attnfa-tiny"
	requireDeviceAndFixture(t, base)
	g := kv0LoadCtx(t, base, "f16", 2048, kv0RandomTokens)
	pc := decoder.SubsystemsOf(g.r).Prefill
	if !pc.PrefillChunksAvailable() {
		t.Fatal("cudaResident's prefill subsystem does not report prefill chunks")
	}
	cr := decoder.SubsystemsOf(g.r).Runtime
	if got := cr.ResidentCaps().TailMinRows; got <= 0 || got != g.r.PrefillTailMinRows() {
		t.Fatalf("ResidentCaps().TailMinRows = %d, want PrefillTailMinRows %d (> 0)", got, g.r.PrefillTailMinRows())
	}
	// The fused attention serves this fixture and needs no span scratch, so the row kernels are forced for this one check.
	prefillSplitKVForceForTest = true
	spanScratch := pc.PrefillSpanScratchBytes(512, 512)
	prefillSplitKVForceForTest = false
	if spanScratch <= 0 {
		t.Fatal("cudaResident's prefill subsystem reports no span scratch with the row kernels forced (PrefillSpanScratchBytes(512, 512) <= 0)")
	}
	if c := pc.PrefillChunkRows(); c != g.r.prefillWidth || !slices.Contains(prefillWidths, c) || c%64 != 0 {
		t.Errorf("PrefillChunkRows %d, want the build's chosen width %d, one of %v (each a multiple of the block size 64)", c, g.r.prefillWidth, prefillWidths)
	}
	if got := g.r.PrefillTailMinRows(); got != 16 {
		t.Errorf("PrefillTailMinRows %d, want 16", got)
	}
	if b := pc.PrefillScratchBytes(512); b <= 0 || b >= pc.PrefillScratchBytes(20954) {
		t.Errorf("scratch of one chunk (%d B) must be positive and under a whole 20,954-token prompt's (%d B)", b, pc.PrefillScratchBytes(20954))
	}
}

// TestKV3_admissionPricesTheChunk (gate: a request the old bound refused now answers): on the real 1.5B, with the host's available
// memory set between the whole-prompt price and the resident's per-chunk price of a 20,954-token request (the I7 night's size),
// admission accepts it; at the whole-prompt price it would have refused (the 413 text), and with the room cut under the
// per-chunk price it refuses. No generation: the pass itself is the speed gate's request.
func TestKV3_admissionPricesTheChunk(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads the 1.5B)")
	}
	path := modelPath("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	requireDeviceAndFile(t, path)
	g := kv0LoadCtx(t, path, "f16", 21000, kv0TextTokens(t, path))
	const prompt, maxTok = 20954, 8
	whole := g.m.WholePromptScratchBytesForTest(prompt)
	chunk := g.r.PrefillScratchBytes(g.r.PrefillChunkRows()) + int64(prompt)*4*int64(g.m.BackendView().EmbedLen())
	frac := 0.7 // decoder.prefillAvailFraction: the share of available memory a request may take
	t.Logf("whole-prompt price %.2f GB, per-chunk price %.3f GB (device scratch for %d rows plus the suffix's host embeddings)", float64(whole)/(1<<30), float64(chunk)/(1<<30), g.r.PrefillChunkRows())
	if chunk*4 > whole {
		t.Fatalf("the per-chunk price %d is not far under the whole-prompt price %d: the gate proves nothing", chunk, whole)
	}
	room := int64(float64(whole+chunk) / 2 / frac) // available memory such that remaining sits between the two prices
	restore := decoder.SetHostRAMAvailableForTest(room)
	defer restore()
	if err := g.m.AdmitPrefillMemory(prompt, maxTok, true); err != nil {
		t.Errorf("a request the per-chunk price fits was refused: %v", err)
	}
	restore()
	restore = decoder.SetHostRAMAvailableForTest(int64(float64(chunk) / 2 / frac))
	if err := g.m.AdmitPrefillMemory(prompt, maxTok, true); err == nil {
		t.Error("a request whose chunk does not fit was admitted")
	}
	restore()
}

// TestKV3_chunkerKeepsPassStartsOnTheGrid: at the default chunk, from a start on the 64-row grid, every pass of a prefill starts on
// the grid and the last has at least 16 rows, whatever the prompt's length; a chunk too narrow to give up 64 rows still ends in a
// pass of 16. kv0Passes is the chunker's plan (TestKV0_prefillInvariance holds prefillChunked to it by launch counts). Struct-only.
func TestKV3_chunkerKeepsPassStartsOnTheGrid(t *testing.T) {
	for _, start := range []int{0, 64, 12288} {
		for M := 513; M <= 2200; M++ {
			ps := kv0Passes(start, M, 512)
			rows := 0
			for i, p := range ps {
				if p.start%prefillAlignRows != 0 {
					t.Fatalf("start=%d M=%d: pass %d starts at %d, off the %d-row grid", start, M, i, p.start, prefillAlignRows)
				}
				if p.m < attnFusedMinRows || p.m > 512 {
					t.Fatalf("start=%d M=%d: pass %d has %d rows", start, M, i, p.m)
				}
				rows += p.m
			}
			if rows != M {
				t.Fatalf("start=%d M=%d: the passes cover %d rows", start, M, rows)
			}
		}
	}
	if ps := kv0Passes(0, 1030, 512); len(ps) != 3 || ps[1].m != 448 || ps[2].start != 960 || ps[2].m != 70 {
		t.Errorf("1,030 rows at chunk 512: %+v, want 512 + 448 + 70 with the last pass at 960", ps)
	}
	for M := 65; M <= 400; M++ { // a chunk of 64 cannot give up a whole block
		ps := kv0Passes(0, M, 64)
		if last := ps[len(ps)-1]; last.m < attnFusedMinRows {
			t.Fatalf("chunk 64, M=%d: last pass of %d rows", M, last.m)
		}
	}
	// A chunk under 32 cannot keep 16 rows on both sides of the cut: no pass may be empty, and from 32 up the last pass has 16.
	for chunk := 1; chunk <= 79; chunk++ {
		for M := chunk + 1; M <= 300; M++ {
			ps := kv0Passes(0, M, chunk)
			rows := 0
			for i, p := range ps {
				if p.m < 1 || p.m > chunk {
					t.Fatalf("chunk %d, M=%d: pass %d has %d rows", chunk, M, i, p.m)
				}
				rows += p.m
			}
			if rows != M {
				t.Fatalf("chunk %d, M=%d: the passes cover %d rows", chunk, M, rows)
			}
			if last := ps[len(ps)-1]; chunk >= 2*attnFusedMinRows && last.m < attnFusedMinRows {
				t.Fatalf("chunk %d, M=%d: last pass of %d rows", chunk, M, last.m)
			}
		}
	}
}

// MEM6 on CUDA (docs/measurements/mem6-cuda-2026-11/README.md): a resident prefill reads the suffix's embeddings a pass at a time into one reused slab
// (decoder.ResidentPrefillSubsystem), so the host holds a pass of them, not the suffix. The most embedding rows the decoder holds at once during a prefill of 2,048
// rows is at most one pass (prefillPassRows), and the last logits and 32 decode steps after them are byte-equal to the slice entry point's.
//
//	go test -count=1 -tags 'cuda goinfer_testhooks' -run '^TestMEM6_cudaPrefillHostRows$' -v ./backend/cuda/
func TestMEM6_cudaPrefillHostRows(t *testing.T) {
	requireCUDADevice(t)
	dir := kv3LongContextCopy(t, filepath.Join("../../testdata", "llama-tiny"), 4096)
	m, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4", ResidentContext: 2112})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	a, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatalf("not the CUDA resident: %s", m.ResidentDecline())
	}
	_, _, _, _, _, _, vocab := m.Dims()
	// A tiny fixture runs a prompt this long in one pass; a real model's pass is a few hundred rows, which the learned chunk cap stands in for.
	a.prefillChunkCap.Store(256)
	defer a.prefillChunkCap.Store(0)
	pass := a.prefillPassRows()
	if pass != 256 {
		t.Fatalf("the pass is %d rows, want the 256 the test set", pass)
	}
	held := 0
	decoder.SetEmbedRowsObserverForTest(func(rows int) { held = max(held, rows) })
	defer decoder.SetEmbedRowsObserverForTest(nil)
	hash := func(lg []float32, n int) uint64 {
		h := fnv.New64a()
		put := func(l []float32) {
			for _, x := range l {
				b := math.Float32bits(x)
				h.Write([]byte{byte(b), byte(b >> 8), byte(b >> 16), byte(b >> 24)})
			}
		}
		put(lg)
		next := argmaxF(lg)
		for j := range 32 {
			l, err := a.Forward(m.EmbedResidentForTest(next), n+j)
			if err != nil {
				t.Fatal(err)
			}
			put(l)
			next = argmaxF(l)
		}
		return h.Sum64()
	}
	for _, n := range []int{1024, 2048} {
		ids := make([]int, n)
		for i := range ids {
			ids[i] = (i*37 + 11) % vocab
		}
		// the slice entry point: every row embedded up front, as the decoder did before
		a.Reset()
		emb := make([][]float32, n)
		for i, id := range ids {
			emb[i] = m.EmbedResidentForTest(id)
		}
		lgSlice, err := a.PrefillLast(context.Background(), emb, 0)
		if err != nil {
			t.Fatal(err)
		}
		want := hash(lgSlice, n)
		held = 0
		a.Reset()
		lg, err := m.ResidentPrefillSeedForTest(context.Background(), ids)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("%d rows: at most %d embedding rows held at once (one pass is %d)", n, held, pass)
		// The observer must have fired: a prefill that never called it would read 0 and pass "at most one pass" without reading anything. The
		// prompt is longer than a pass, so the most rows held at once is exactly one pass.
		if held != pass {
			t.Errorf("%d rows: the decoder held at most %d embedding rows at once, want exactly one pass (%d): at most it, and not 0 because the observer never fired", n, held, pass)
		}
		if got := hash(lg, n); got != want {
			t.Errorf("%d rows: the last logits and 32 decode steps differ from the slice entry point's (%#x, want %#x)", n, got, want)
		}
	}
}

// The Go bytes a prompt row adds to a resident prefill stay under one embedding row. With the slab alone the batched pass still copied each pass's
// embeddings into a fresh contiguous host block, so a 20K prompt left its whole size in garbage that the runtime holds resident after the request (MEM0's +218 MiB
// on the 1.5B): the block is one reused buffer now. The slope is read between two lengths so the fixed costs of a prefill drop out.
func TestMEM6_cudaPrefillAllocatesNoRowCopy(t *testing.T) {
	requireCUDADevice(t)
	dir := kv3LongContextCopy(t, filepath.Join("../../testdata", "llama-tiny"), 4096)
	m, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4", ResidentContext: 2112})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	a, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatalf("not the CUDA resident: %s", m.ResidentDecline())
	}
	_, _, _, _, _, _, vocab := m.Dims()
	a.prefillChunkCap.Store(256)
	defer a.prefillChunkCap.Store(0)
	bytesFor := func(n int) uint64 {
		ids := make([]int, n)
		for i := range ids {
			ids[i] = (i*37 + 11) % vocab
		}
		a.Reset()
		if _, err := m.ResidentPrefillSeedForTest(context.Background(), ids); err != nil { // warm: the first pass sizes any reused buffer
			t.Fatal(err)
		}
		a.Reset()
		runtime.GC()
		var s0, s1 runtime.MemStats
		runtime.ReadMemStats(&s0)
		if _, err := m.ResidentPrefillSeedForTest(context.Background(), ids); err != nil {
			t.Fatal(err)
		}
		runtime.ReadMemStats(&s1)
		return s1.TotalAlloc - s0.TotalAlloc
	}
	b1, b2 := bytesFor(1024), bytesFor(2048)
	perRow := float64(int64(b2)-int64(b1)) / 1024
	embRow := float64(4 * m.BackendView().EmbedLen())
	t.Logf("Go bytes allocated by the prefill: %d at 1,024 rows, %d at 2,048; %.0f B per prompt row against %.0f B per embedding row (%.2f)", b1, b2, perRow, embRow, perRow/embRow)
	if perRow > embRow {
		t.Errorf("a prompt row adds %.0f B of Go allocation, over one embedding row (%.0f B): a pass copies its embeddings again", perRow, embRow)
	}
}

// TestKV10_fusedPrefillRule is KV10-a's rule test (docs/measurements/kv10-2026-11/README.md): on a resident whose fused attention module loaded, useAttnFused answers
// yes for exactly the head dims 64 and 128 of every width from 1 to 512, and no under an attention softcap. docs/measurements/kv10-2026-11/fused_census.py applies the
// same rule to the checkpoints' configs.
func TestKV10_fusedPrefillRule(t *testing.T) {
	const base = "../../testdata/llama-tiny"
	requireDeviceAndFixture(t, base)
	g := kv0LoadCtx(t, base, "f16", 128, kv0RandomTokens)
	r := g.r
	if r.bAttnFused64 == (Pipeline{}) || r.bAttnFused128 == (Pipeline{}) || !r.fastAttn {
		t.Fatalf("VOID: the fused modules did not load (64 %v, 128 %v, fastAttn %v), so every answer would be no", r.bAttnFused64 != (Pipeline{}), r.bAttnFused128 != (Pipeline{}), r.fastAttn)
	}
	g.floorKnob("0") // fast kernels at any prompt length, so the floor does not answer for the head dim
	r.passPromptLen = 4096
	var served []int
	for hd := 1; hd <= 512; hd++ {
		if _, ok := r.useAttnFused(hd, attnFusedMinRows); ok {
			served = append(served, hd)
		}
	}
	if !slices.Equal(served, []int{64, 128}) {
		t.Errorf("the fused kernel serves head dims %v, want exactly [64 128]", served)
	}
	r.attnCap = 30
	if _, ok := r.useAttnFused(128, attnFusedMinRows); ok {
		t.Errorf("the fused kernel serves head dim 128 under an attention softcap")
	}
	r.attnCap = 0
	if _, ok := r.useAttnFused(128, attnFusedMinRows-1); ok {
		t.Errorf("the fused kernel serves a pass of %d rows, under attnFusedMinRows", attnFusedMinRows-1)
	}
}

// kv10Beat prints a heartbeat to stderr every 45 seconds while a long step runs (with -v it streams), and returns the function that stops it.
func kv10Beat(label string) func() {
	t0, done := time.Now(), make(chan struct{})
	go func() {
		tk := time.NewTicker(45 * time.Second)
		defer tk.Stop()
		for {
			select {
			case <-tk.C:
				fmt.Fprintf(os.Stderr, "[kv10] %s: %.0f s elapsed\n", label, time.Since(t0).Seconds())
			case <-done:
				return
			}
		}
	}()
	return func() {
		close(done)
		fmt.Fprintf(os.Stderr, "[kv10] %s: done in %.1f s\n", label, time.Since(t0).Seconds())
	}
}

// kv10Tail loads the checkpoint at kv precision with room for depth rows, prefills depth-tail rows, then reads the last tail rows' logits
// (PrefillLastN: row m predicts position start+m+1). A load the fit check refuses skips the test as NOT RUN.
func kv10Tail(t *testing.T, ckpt, kv string, ids []int, tail int) [][]float32 {
	t.Helper()
	depth := len(ids)
	m, err := decoder.Load(ckpt, decoder.Options{Backend: "cuda", Quant: "int4", ResidentContext: depth + 64, KVPrecision: kv})
	if err != nil {
		t.Skipf("NOT RUN: %s at depth %d, kv %s: %v", filepath.Base(ckpt), depth, kv, err)
	}
	defer m.Close()
	r, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Skipf("NOT RUN: %s is not CUDA-resident at depth %d, kv %s: %s", filepath.Base(ckpt), depth, kv, m.ResidentDecline())
	}
	if batched, why := r.PrefillPath(); !batched {
		t.Fatalf("batched prefill declined: %s", why)
	}
	for l := range r.layers {
		if r.layers[l].kvHalf != (kv == "f16") && !r.layers[l].isDeltaNet() && !r.layers[l].noKV() {
			t.Fatalf("layer %d holds f16=%v, kv %s was asked for", l, r.layers[l].kvHalf, kv)
		}
	}
	embs := make([][]float32, depth)
	for i, id := range ids {
		embs[i] = m.EmbedResidentForTest(id)
	}
	stop := kv10Beat(fmt.Sprintf("depth %d kv %s: prefill of %d rows", depth, kv, depth-tail))
	if _, err := r.PrefillLast(context.Background(), embs[:depth-tail], 0); err != nil {
		stop()
		t.Fatalf("PrefillLast(%d rows): %v", depth-tail, err)
	}
	stop()
	outs, err := r.PrefillLastN(embs[depth-tail:], depth-tail)
	if err != nil {
		t.Fatalf("PrefillLastN(%d rows at %d): %v", tail, depth-tail, err)
	}
	out := make([][]float32, len(outs))
	for i := range outs {
		out[i] = append([]float32(nil), outs[i]...)
	}
	return out
}

// kv10Compare is the metric over the tail rows: the min cosine of b against a, the argmax agreement (all rows, and rows where a's top-1 margin is at least
// 0.5), the mean KL(a || b) in nats, and whether any logit is NaN or Inf.
func kv10Compare(a, b [][]float32) (minCos float64, agree, agreeM, rows, rowsM int, meanKL float64, bad bool) {
	minCos = 1
	for i := range a {
		for j := range a[i] {
			if x, y := float64(a[i][j]), float64(b[i][j]); math.IsNaN(x) || math.IsInf(x, 0) || math.IsNaN(y) || math.IsInf(y, 0) {
				bad = true
			}
		}
		if c := kv0Cosine(a[i], b[i]); c < minCos {
			minCos = c
		}
		top, second := float32(math.Inf(-1)), float32(math.Inf(-1))
		for _, v := range a[i] {
			if v > top {
				top, second = v, top
			} else if v > second {
				second = v
			}
		}
		rows++
		same := kv0Argmax(a[i]) == kv0Argmax(b[i])
		if same {
			agree++
		}
		if top-second >= 0.5 {
			rowsM++
			if same {
				agreeM++
			}
		}
		meanKL += kv10KL(a[i], b[i])
	}
	return minCos, agree, agreeM, rows, rowsM, meanKL / float64(len(a)), bad
}

// kv10KL is KL(softmax(p) || softmax(q)) in nats, in float64.
func kv10KL(p, q []float32) float64 {
	lse := func(v []float32) float64 {
		mx := math.Inf(-1)
		for _, x := range v {
			mx = math.Max(mx, float64(x))
		}
		var s float64
		for _, x := range v {
			s += math.Exp(float64(x) - mx)
		}
		return mx + math.Log(s)
	}
	lp, lq := lse(p), lse(q)
	var kl float64
	for i := range p {
		lpi := float64(p[i]) - lp
		kl += math.Exp(lpi) * (lpi - (float64(q[i]) - lq))
	}
	return kl
}

// kv10Text is this repository's source and docs as tokens of the checkpoint's tokenizer, at least n of them (real text, never cycled).
func kv10Text(t *testing.T, ckpt string, n int) []int {
	t.Helper()
	tk, err := modelload.Tokenizer(ckpt)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	var files []string
	for _, g := range []string{"../../decoder/*.go", "../../docs/*.md", "../../backend/cuda/*.go"} {
		m, _ := filepath.Glob(g)
		sort.Strings(m)
		files = append(files, m...)
	}
	var sb strings.Builder
	for _, f := range files {
		if strings.HasSuffix(f, "_test.go") {
			continue
		}
		b, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		sb.Write(b)
		if sb.Len() > 3_000_000 {
			break
		}
	}
	ids, err := tk.Encode(sb.String(), false)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if len(ids) < n {
		t.Fatalf("VOID: the repository's text is %d tokens, the deepest cell needs %d", len(ids), n)
	}
	return ids[:n]
}

// TestKV10_longContextFidelity is KV10-b's harness (docs/measurements/kv10-2026-11/README.md): at each depth, the last 16 rows' logits of the checkpoint with KV f16 against
// KV f32, on this repository's text. GOINFER_KV10_DEPTHS (default "3900,65536,131000"), KV10_CKPT (default ~/models/qwen3.5-0.8b), KV10_OUT (a JSON-lines file).
//
//	GOINFER_HEAVY_TESTS=1 KV10_OUT=<file> go test -tags 'cuda goinfer_testhooks' -run '^TestKV10_longContextFidelity$' -v -timeout 90m ./backend/cuda/
func TestKV10_longContextFidelity(t *testing.T) {
	requireCUDADevice(t)
	requireHeavyModel(t)
	ckpt := os.Getenv("KV10_CKPT")
	if ckpt == "" {
		ckpt = os.ExpandEnv("$HOME/models/qwen3.5-0.8b")
	}
	depths := []int{3900, 65536, 131000}
	if v := os.Getenv("GOINFER_KV10_DEPTHS"); v != "" {
		depths = nil
		for _, f := range strings.Split(v, ",") {
			n, err := strconv.Atoi(strings.TrimSpace(f))
			if err != nil || n < 64 {
				t.Fatalf("GOINFER_KV10_DEPTHS: %q", f)
			}
			depths = append(depths, n)
		}
	}
	const tail = 16
	ids := kv10Text(t, ckpt, slices.Max(depths))
	var out *os.File
	if p := os.Getenv("KV10_OUT"); p != "" {
		f, err := os.OpenFile(p, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		out = f
	}
	var control float64
	for _, depth := range depths {
		t.Run(fmt.Sprintf("depth%d", depth), func(t *testing.T) {
			t0 := time.Now()
			fmt.Fprintf(os.Stderr, "[kv10] depth %d: start %s\n", depth, t0.Format("15:04:05"))
			f32 := kv10Tail(t, ckpt, "f32", ids[:depth], tail)
			f16 := kv10Tail(t, ckpt, "f16", ids[:depth], tail)
			minCos, agree, agreeM, rows, rowsM, kl, bad := kv10Compare(f32, f16)
			verdict := "HOLDS"
			switch {
			case bad || minCos < 0.99:
				verdict = "FAILS"
			case minCos < 0.999 || agreeM < rowsM || (control > 0 && kl > 2*control):
				verdict = "DEGRADES"
			}
			if depth == depths[0] {
				control = kl
			}
			t.Logf("depth %d: min cosine %.6f, argmax agrees on %d of %d rows (%d of %d with a margin of 0.5+), mean KL %.3e nats (control %.3e), NaN/Inf %v: %s; %.0f s",
				depth, minCos, agree, rows, agreeM, rowsM, kl, control, bad, verdict, time.Since(t0).Seconds())
			if out != nil {
				rec, _ := json.Marshal(map[string]any{"depth": depth, "min_cos": minCos, "agree": agree, "rows": rows, "agree_margin": agreeM, "rows_margin": rowsM, "mean_kl": kl,
					"control_kl": control, "nan_inf": bad, "verdict": verdict, "seconds": time.Since(t0).Seconds(), "ckpt": filepath.Base(ckpt)})
				fmt.Fprintf(out, "%s\n", rec)
			}
			if verdict == "FAILS" {
				t.Errorf("depth %d FAILS: min cosine %.6f, NaN/Inf %v", depth, minCos, bad)
			}
		})
	}
}

// TestKV3_gluQuantWritesInPlace is KV3-S1's red-first gate (docs/measurements/kv3-s1-2026-11/README.md): glu_quant_batched leaves its product in the gate
// buffer, so a pass allocates two M x inter float buffers (the gate's and the up's) and no third for the product, and PrefillScratchBytes prices the pass without it.
func TestKV3_gluQuantWritesInPlace(t *testing.T) {
	const base = "../../testdata/llama-tiny" // intermediate width 128 against a hidden width of 64: no other scratch row is that wide
	requireDeviceAndFixture(t, base)
	g := kv0LoadCtx(t, base, "f16", 2048, kv0RandomTokens)
	r := g.r
	maxQDim, maxKvDim := r.prefillMaxGeom()
	for _, d := range []int{r.hidden, maxQDim, maxKvDim, r.hidden / 4, maxQDim / 4} {
		if d == r.inter || d*4 == r.inter*4 {
			t.Fatalf("VOID: the fixture's intermediate width %d equals another scratch row width, so counting M x inter buffers proves nothing", r.inter)
		}
	}
	for _, M := range []int{16, 100} {
		var sizes []int64
		var seen int64
		prefillScratchSeenForTest = func(b int64) { sizes = append(sizes, b); seen += b + allocRoundSlack(b) }
		g.prefill(g.embs(g.tokens(M, 5)), 0)
		prefillScratchSeenForTest = nil
		n := 0
		for _, b := range sizes {
			if b == int64(M)*int64(r.inter)*4 {
				n++
			}
		}
		if n != 2 {
			t.Errorf("M=%d: the pass allocated %d buffers of M x inter floats (%d B), want 2: the gate's and the up's, the product written into the gate's", M, n, int64(M)*int64(r.inter)*4)
		}
		if want := r.PrefillScratchBytes(M); want != seen {
			t.Errorf("M=%d: PrefillScratchBytes %d B, the pass allocated %d B", M, want, seen)
		}
	}
}
