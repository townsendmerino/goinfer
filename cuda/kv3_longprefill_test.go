//go:build cuda && goinfer_testhooks

package cuda

import (
	"fmt"
	"github.com/townsendmerino/goinfer/decoder"
	"math"
	"os"
	"path/filepath"
	"regexp"
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
	const base = "../testdata/llama-attnfa-tiny"
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
	const base = "../testdata/llama-attnfa-tiny"
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
	const base = "../testdata/llama-attnfa-tiny"
	requireDeviceAndFixture(t, base)
	g := kv0LoadCtx(t, base, "f16", 2048, kv0RandomTokens)
	pc, ok := any(g.r).(decoder.ResidentPrefillChunks)
	if !ok {
		t.Fatal("cudaResident does not satisfy decoder.ResidentPrefillChunks")
	}
	if _, ok := any(g.r).(decoder.ResidentPrefillTailMin); !ok {
		t.Fatal("cudaResident does not satisfy decoder.ResidentPrefillTailMin")
	}
	if _, ok := any(g.r).(decoder.ResidentPrefillSpanScratch); !ok {
		t.Fatal("cudaResident does not satisfy decoder.ResidentPrefillSpanScratch")
	}
	if c := pc.PrefillChunkRows(); c != 512 || c%64 != 0 {
		t.Errorf("PrefillChunkRows %d, want the default 512 (a multiple of the block size 64)", c)
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
	chunk := g.r.PrefillScratchBytes(g.r.PrefillChunkRows()) + int64(prompt)*4*int64(g.m.ResidentEmbedLen())
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
