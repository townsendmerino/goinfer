//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"testing"
	"time"

	gc "github.com/eitamring/gocudrv/cuda"
	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// KV0's CUDA invariance gates (docs/measurements/kv0-invariance-2026-10-13.md, the registration and the readings). They
// are the twin of metal's TestMC5_prefillChunkInvariance, with one addition: CUDA picks its kernel class from the PROMPT's
// length and the pass's row count, so a prefill split at an offset can write a row with a different class than the whole
// prompt would. kv0FastPass is that class map, and every arm asserts the outcome it predicts: a pass whose class the map
// gets wrong fails the test instead of being recorded as a surprise.
//
//	GOINFER_HEAVY_TESTS=1 go test -count=1 -timeout 30m -tags 'cuda goinfer_testhooks' -run '^TestKV0_' -v ./cuda/ 2>&1 | tee /tmp/test-kv0-cuda.log
//
// Without GOINFER_HEAVY_TESTS only the tiny fixture runs.

// kv0Ctx is the resident context both fixtures load with: the prompts here are at most 1,100 tokens.
const kv0Ctx = 2048

// kv0FastPass reports whether a pass of m rows starting at start runs the fast (non-exact) kernels. promptLen is the
// whole prompt's length when the pass is one chunk of a multi-chunk prefill, 0 otherwise (prefillCore's passPromptLen =
// max(chunkPromptLen, start+m)). Both levers decline below 16 rows (attnFusedMinRows, gemmMMAMinRows) and below the
// floor.
func kv0FastPass(floor, promptLen, start, m int) bool {
	return m >= attnFusedMinRows && max(promptLen, start+m) >= floor
}

// kv0Pass is one batched pass of a prefill.
type kv0Pass struct{ start, m, promptLen int }

// kv0Passes lists the passes PrefillLast(rows [start, start+M)) makes at chunk width chunk (prefillChunked, including its rule that
// the last pass has at least attnFusedMinRows rows).
func kv0Passes(start, M, chunk int) []kv0Pass {
	if M <= chunk {
		return []kv0Pass{{start, M, 0}}
	}
	var ps []kv0Pass
	for i := 0; i < M; {
		n := min(chunk, M-i)
		if rem := M - (i + n); rem > 0 && rem < attnFusedMinRows {
			n -= prefillTailCut(n, rem) // the chunker's 16-row rule: a prefill's last pass is never shorter than 16 rows
		}
		ps = append(ps, kv0Pass{start + i, n, start + M})
		if n <= 0 { // prefillChunked fails such a pass ("empty prompt"); the plan ends there instead of looping
			return ps
		}
		i += n
	}
	return ps
}

// kv0KV is every layer's K and V rows [0, n), as float32 (one tensor per K and per V), and each tensor's row width.
type kv0KV struct {
	t     [][]float32
	kvDim []int
}

type kv0Diff struct {
	total, differ int
	firstPos      int // lowest position with a differing element, -1 if none
	firstLayer    int
	before, after int // differing elements below / at or above the boundary given to kv0Compare
}

func kv0Compare(want, got kv0KV, boundary int) kv0Diff {
	d := kv0Diff{firstPos: -1}
	for i := range want.t {
		kd := want.kvDim[i]
		for j := range want.t[i] {
			d.total++
			if math.Float32bits(want.t[i][j]) == math.Float32bits(got.t[i][j]) {
				continue
			}
			d.differ++
			if p := j / kd; d.firstPos < 0 || p < d.firstPos {
				d.firstPos, d.firstLayer = p, i/2
			}
			if j/kd < boundary {
				d.before++
			} else {
				d.after++
			}
		}
	}
	return d
}

// kv0CompareRows counts differing elements in positions [lo, hi).
func kv0CompareRows(want, got kv0KV, lo, hi int) (differ, total int) {
	for i := range want.t {
		kd := want.kvDim[i]
		for j := lo * kd; j < hi*kd; j++ {
			total++
			if math.Float32bits(want.t[i][j]) != math.Float32bits(got.t[i][j]) {
				differ++
			}
		}
	}
	return differ, total
}

func kv0LogitsDiff(a, b []float32) (differ int, worst float64) {
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			differ++
			worst = math.Max(worst, math.Abs(float64(a[i])-float64(b[i])))
		}
	}
	return differ, worst
}

func kv0Cosine(a, b []float32) float64 {
	var d, na, nb float64
	for i := range a {
		d += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return d / math.Sqrt(na*nb)
}

func kv0Argmax(a []float32) int {
	best := 0
	for i, v := range a {
		if v > a[best] {
			best = i
		}
	}
	return best
}

// kv0Rig is one loaded model and the means to prefill it and read its KV back.
type kv0Rig struct {
	t      *testing.T
	m      *decoder.Model
	r      *cudaResident
	vocab  int
	tokens func(n, off int) []int // a prompt of n token ids, from stream offset off
	ctx    context.Context
}

func kv0Load(t *testing.T, path, kv string, tokens func(vocab int) func(n, off int) []int) *kv0Rig {
	t.Helper()
	return kv0LoadCtx(t, path, kv, kv0Ctx, tokens)
}

// kv0LoadCtx is kv0Load with the resident context named (KV3's prompts are longer than KV0's).
func kv0LoadCtx(t *testing.T, path, kv string, ctxCap int, tokens func(vocab int) func(n, off int) []int) *kv0Rig {
	t.Helper()
	m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4", ResidentContext: ctxCap, KVPrecision: kv})
	if err != nil {
		t.Fatalf("load %s: %v", path, err)
	}
	t.Cleanup(func() { _ = m.Close() })
	r := mustResident(t, m)
	if batched, why := r.PrefillPath(); !batched {
		t.Fatalf("%s: batched prefill declined: %s", path, why)
	}
	if r.kvWin {
		t.Fatalf("%s: windowed KV is on; this gate reads whole-context caches", path)
	}
	for l := range r.layers {
		if r.layers[l].kvHalf != (kv == "f16") || r.layers[l].isMLA || r.layers[l].isDeltaNet {
			t.Fatalf("%s: layer %d: KV precision asked for %q, layer holds f16=%v (or it is latent or recurrent KV)", path, l, kv, r.layers[l].kvHalf)
		}
	}
	t.Logf("KV precision %s", kv)
	_, _, _, _, _, _, vocab := m.Dims()
	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Minute)
	t.Cleanup(cancel)
	return &kv0Rig{t: t, m: m, r: r, vocab: vocab, tokens: tokens(vocab), ctx: ctx}
}

func (g *kv0Rig) embs(ids []int) [][]float32 {
	out := make([][]float32, len(ids))
	for i, id := range ids {
		out[i] = append([]float32(nil), g.m.EmbedResidentForTest(id)...)
	}
	return out
}

// poison overwrites rows [0, n) of every K and V cache with NaN, so a row an arm fails to write cannot match the row an
// earlier arm left there.
func (g *kv0Rig) poison(n int) {
	g.t.Helper()
	for l := range g.r.layers {
		if g.r.layers[l].kvShared {
			continue
		}
		nan := make([]float32, n*g.r.layers[l].kvDim)
		for i := range nan {
			nan[i] = float32(math.NaN())
		}
		if err := g.r.UploadKV(l, 0, nan, nan); err != nil {
			g.t.Fatalf("poison layer %d: %v", l, err)
		}
	}
}

func (g *kv0Rig) snapshot(n int) kv0KV {
	g.t.Helper()
	var s kv0KV
	err := g.r.do(func() error {
		if e := g.r.stream.Sync(); e != nil {
			return e
		}
		for l := range g.r.layers {
			if g.r.layers[l].kvShared {
				continue
			}
			kd := g.r.layers[l].kvDim
			var k, v []float32
			if g.r.layers[l].kvHalf {
				kh, vh := make([]uint16, n*kd), make([]uint16, n*kd)
				if e := gpu.Download(g.r.kc[l], kh); e != nil {
					return e
				}
				if e := gpu.Download(g.r.vc[l], vh); e != nil {
					return e
				}
				k, v = kv0HalfToF32(kh), kv0HalfToF32(vh)
			} else {
				k, v = make([]float32, n*kd), make([]float32, n*kd)
				if e := gpu.Download(g.r.kc[l], k); e != nil {
					return e
				}
				if e := gpu.Download(g.r.vc[l], v); e != nil {
					return e
				}
			}
			s.t = append(s.t, k, v)
			s.kvDim = append(s.kvDim, kd, kd)
		}
		return nil
	})
	if err != nil {
		g.t.Fatalf("snapshot: %v", err)
	}
	return s
}

// kv0HalfToF32 widens f16 bit patterns exactly, so two f16 caches compare bit for bit through their f32 images (a NaN stays a NaN).
func kv0HalfToF32(h []uint16) []float32 {
	out := make([]float32, len(h))
	for i, x := range h {
		sign, exp, man := uint32(x>>15), uint32(x>>10)&0x1f, uint32(x)&0x3ff
		var b uint32
		switch {
		case exp == 0 && man == 0:
			b = sign << 31
		case exp == 0: // subnormal: normalise
			e := uint32(127 - 15 + 1)
			for man&0x400 == 0 {
				man <<= 1
				e--
			}
			b = sign<<31 | e<<23 | (man&0x3ff)<<13
		case exp == 0x1f:
			b = sign<<31 | 0xff<<23 | man<<13
		default:
			b = sign<<31 | (exp+127-15)<<23 | man<<13
		}
		out[i] = math.Float32frombits(b)
	}
	return out
}

func (g *kv0Rig) prefill(e [][]float32, start int) []float32 {
	g.t.Helper()
	lg, err := g.r.PrefillLast(g.ctx, e, start)
	if err != nil {
		g.t.Fatalf("PrefillLast(%d rows at %d): %v", len(e), start, err)
	}
	return append([]float32(nil), lg...)
}

func (g *kv0Rig) launches() (attn, gemm int) { return g.r.fastAttnLaunches, g.r.fastGemmLaunches }

func (g *kv0Rig) chunk(c string) { decoder.SetKnobEnvForTest(g.t, g.m, "GOINFER_PREFILL_CHUNK", c) }

func (g *kv0Rig) floorKnob(v string) {
	if v == "" {
		decoder.UnsetKnobEnvForTest(g.t, g.m, "GOINFER_CUDA_FAST_PREFILL_FLOOR")
		return
	}
	decoder.SetKnobEnvForTest(g.t, g.m, "GOINFER_CUDA_FAST_PREFILL_FLOOR", v)
}

func kv0RandomTokens(vocab int) func(n, off int) []int {
	return func(n, off int) []int {
		seed := uint32(13579) + uint32(off)*2654435761
		out := make([]int, n)
		for i := range out {
			seed ^= seed << 13
			seed ^= seed >> 17
			seed ^= seed << 5
			out[i] = int(seed % uint32(vocab-1))
		}
		return out
	}
}

// kv0TextTokens tokenizes this repository's own Go sources with the model's tokenizer: natural-looking text, so the logits
// the continuation arm compares are not a flat distribution over garbage.
func kv0TextTokens(t *testing.T, ggufPath string) func(int) func(n, off int) []int {
	tk, err := tokenizer.LoadGGUF(ggufPath)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	files, _ := filepath.Glob("../decoder/*.go")
	sort.Strings(files)
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
		if sb.Len() > 600_000 {
			break
		}
	}
	ids, err := tk.Encode(sb.String(), false)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	return func(int) func(n, off int) []int {
		return func(n, off int) []int {
			off %= max(1, len(ids)-n)
			return append([]int(nil), ids[off:off+n]...)
		}
	}
}

func TestKV0_prefillInvariance_tiny(t *testing.T) {
	const dir = "../testdata/llama-attnfa-tiny"
	requireDeviceAndFixture(t, dir)
	for _, kv := range []string{"f32", "f16"} {
		t.Run(kv, func(t *testing.T) { kv0Invariance(t, kv0Load(t, dir, kv, kv0RandomTokens)) })
	}
}

func TestKV0_prefillInvariance_real1p5B(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads the 1.5B)")
	}
	path := modelPath("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	requireDeviceAndFile(t, path)
	for _, kv := range []string{"f32", "f16"} {
		t.Run(kv, func(t *testing.T) { kv0Invariance(t, kv0Load(t, path, kv, kv0TextTokens(t, path))) })
	}
}

func requireDeviceAndFile(t *testing.T, path string) {
	t.Helper()
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	if _, err := gc.GetDevice(0); err != nil {
		t.Skipf("no device: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s", path)
	}
}

func kv0Invariance(t *testing.T, g *kv0Rig) {
	const N = 1000
	log := func(format string, a ...any) { fmt.Fprintf(os.Stderr, "[kv0] "+format+"\n", a...) }
	start := time.Now()
	floor := fastPrefillFloorFor(g.r.knobValue("GOINFER_CUDA_FAST_PREFILL_FLOOR"))
	if g.r.FastPrefillFloor() == 0 {
		t.Fatalf("the resident has no fast prefill levers: every arm would compare the exact class with itself")
	}
	ids := g.tokens(N, 0)
	e := g.embs(ids)
	log("fixture loaded, KV f16=%v, N=%d floor=%d layers=%d", g.r.layers[0].kvHalf, N, floor, len(g.r.layers))

	// The whole prompt in one pass: the reference, and the per-pass launch counts that prove the fast class ran.
	g.chunk("4096")
	g.poison(N)
	a0, g0 := g.launches()
	lgW := g.prefill(e, 0)
	a1, g1 := g.launches()
	perPassAttn, perPassGemm := a1-a0, g1-g0
	if perPassAttn == 0 || perPassGemm == 0 {
		t.Fatalf("whole prompt of %d tokens launched %d attn_fused and %d gemm_w4a8_mma: the fast class did not run, so no arm can tell the classes apart", N, perPassAttn, perPassGemm)
	}
	W := g.snapshot(N)
	log("whole pass: %d attn_fused, %d gemm_w4a8_mma launches per fast pass", perPassAttn, perPassGemm)

	// The same prompt on the exact class, a prompt below the floor would run.
	g.floorKnob("1000000")
	g.poison(N)
	a0, g0 = g.launches()
	lgE := g.prefill(e, 0)
	a1, g1 = g.launches()
	if a1 != a0 || g1 != g0 {
		t.Fatalf("exact-class reference launched %d attn_fused and %d gemm_w4a8_mma", a1-a0, g1-g0)
	}
	E := g.snapshot(N)
	g.floorKnob("")
	if d := kv0Compare(W, E, N); d.differ == 0 {
		t.Fatalf("fast and exact classes wrote identical KV: the arms below could not see a class mismatch")
	} else {
		ld, _ := kv0LogitsDiff(lgW, lgE)
		log("class sensitivity: fast whole vs exact whole: KV %d of %d elements differ (first at position %d, layer %d); last logits %d of %d differ",
			d.differ, d.total, d.firstPos, d.firstLayer, ld, len(lgW))
	}

	// Run one arm: poison, prefill as the arm says, compare with the whole pass, and check the outcome and the launch
	// counts against the class map.
	arm := func(name string, passes []kv0Pass, run func() []float32) {
		t.Helper()
		g.poison(N)
		a0, g0 := g.launches()
		lg := run()
		a1, g1 := g.launches()
		got := g.snapshot(N)
		d := kv0Compare(W, got, 0)
		ld, worst := kv0LogitsDiff(lgW, lg)
		fast, firstExact := 0, -1
		for _, p := range passes {
			if kv0FastPass(floor, p.promptLen, p.start, p.m) {
				fast++
			} else if firstExact < 0 {
				firstExact = p.start
			}
		}
		wantInvariant := firstExact < 0
		verdict := "INVARIANT"
		if d.differ != 0 || ld != 0 {
			verdict = fmt.Sprintf("NOT INVARIANT, starts at position %d (layer %d)", d.firstPos, d.firstLayer)
		}
		log("%-28s %d passes (%d fast): KV %d of %d elements differ; last logits %d of %d differ (max |diff| %.3g): %s",
			name, len(passes), fast, d.differ, d.total, ld, len(lgW), worst, verdict)
		if (d.differ == 0 && ld == 0) != wantInvariant {
			t.Errorf("%s: observed invariant=%v, the class map predicts %v", name, d.differ == 0 && ld == 0, wantInvariant)
		}
		// No row before the first exact-class pass may differ. The first differing row can sit after it: a row whose attention
		// has a single key (position 0) is the same under both classes, and an f16 store can absorb the rest of a small difference.
		if !wantInvariant && (d.firstPos < firstExact || d.firstPos > firstExact+16) {
			t.Errorf("%s: first differing position %d, the class map predicts it within 16 rows of the first exact-class row %d", name, d.firstPos, firstExact)
		}
		if !wantInvariant && d.firstPos != firstExact {
			log("%-28s   first differing position %d is %d rows after the first exact-class row %d", name, d.firstPos, d.firstPos-firstExact, firstExact)
		}
		if da, dg := a1-a0, g1-g0; da != fast*perPassAttn || dg != fast*perPassGemm {
			t.Errorf("%s: %d attn_fused and %d gemm_w4a8_mma launches, the class map predicts %d and %d (%d fast passes)", name, da, dg, fast*perPassAttn, fast*perPassGemm, fast)
		}
		// An exact-class prefix is the one case where rows can be compared with the exact reference: every row of it read only
		// exact-class history.
		if firstExact == 0 {
			end := N
			for _, p := range passes {
				if kv0FastPass(floor, p.promptLen, p.start, p.m) {
					end = p.start
					break
				}
			}
			if end > 0 && end < N {
				diff, tot := kv0CompareRows(E, got, 0, end)
				log("%-28s   rows [0,%d) against the exact whole pass: %d of %d elements differ", name, end, diff, tot)
				if diff != 0 {
					t.Errorf("%s: the exact-class prefix rows [0,%d) differ from the exact whole pass in %d elements", name, end, diff)
				}
			}
		}
	}

	// Registered arms: whole prompt against chunked, ragged tails included (N=1000 leaves 40, 104, 232 and 488 rows).
	for _, c := range []int{64, 128, 256, 512} {
		cs := fmt.Sprint(c)
		arm("chunk="+cs, kv0Passes(0, N, c), func() []float32 { g.chunk(cs); return g.prefill(e, 0) })
	}
	// Registered arms: prefill [0, s), then [s, n), as a prefix hit does.
	split := func(s int) {
		arm(fmt.Sprintf("split@%d", s), append(kv0Passes(0, s, 4096), kv0Passes(s, N-s, 4096)...), func() []float32 {
			g.chunk("4096")
			g.prefill(e[:s], 0)
			return g.prefill(e[s:], s)
		})
	}
	for _, s := range []int{256, 512, 768} {
		split(s)
	}
	// Probes of the class map's two boundaries: the floor (511 | 512) and the 16-row minimum of the second pass (N-16 | N-15),
	// and a chunk width whose tail is 8 rows.
	for _, s := range []int{511, N - 16, N - 15} {
		split(s)
	}
	arm("chunk=496 (8 left over)", kv0Passes(0, N, 496), func() []float32 { g.chunk("496"); return g.prefill(e, 0) })

	// Control: one token changed at position 700, prefilled in chunks of 256. Nothing may differ before it, and everything
	// that can must from it on.
	ids2 := slices.Clone(ids)
	ids2[700] = (ids2[700] + 1) % (g.vocab - 1)
	e2 := g.embs(ids2)
	g.poison(N)
	g.chunk("256")
	lgC := g.prefill(e2, 0)
	C := g.snapshot(N)
	d := kv0Compare(W, C, 700)
	ld, _ := kv0LogitsDiff(lgW, lgC)
	log("control (token 700 changed): %d KV elements differ before position 700, %d from it on (first at %d); last logits %d of %d differ",
		d.before, d.after, d.firstPos, ld, len(lgW))
	if d.before != 0 || d.after == 0 || d.firstPos != 700 || ld == 0 {
		t.Errorf("control: %d differences before the changed token, %d after, first at %d, %d logits: the comparison is not seeing what it should", d.before, d.after, d.firstPos, ld)
	}
	log("done in %s", time.Since(start).Round(time.Second))
}

// TestKV0_continuationEffect measures §5.6 property 3 of docs/tasks/task-kv-architecture-2026-10.md on CUDA: the K/V rows
// decode writes for a token against the rows a prefill pass writes for the same token, and what that does to the next-token
// logits. Per trial, with the same teacher-forced tokens:
//
//	A: prefill the first np tokens (fast class), then decode d tokens one at a time
//	B: prefill all np+d tokens in one pass (fast class)
//	A': A on the exact class (the floor raised above the prompt)
//	C: prefill all np+d tokens in one pass on the exact class
//
// A against B is the continuation effect a resident reuse hit carries today. A' against C pins its source: the exact
// class is bit-identical to decode, so they must match exactly.
func TestKV0_continuationEffect_tiny(t *testing.T) {
	const dir = "../testdata/llama-attnfa-tiny"
	requireDeviceAndFixture(t, dir)
	for _, kv := range []string{"f32", "f16"} {
		t.Run(kv, func(t *testing.T) { kv0Continuation(t, kv0Load(t, dir, kv, kv0RandomTokens)) })
	}
}

func TestKV0_continuationEffect_real1p5B(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads the 1.5B)")
	}
	path := modelPath("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	requireDeviceAndFile(t, path)
	for _, kv := range []string{"f32", "f16"} {
		t.Run(kv, func(t *testing.T) { kv0Continuation(t, kv0Load(t, path, kv, kv0TextTokens(t, path))) })
	}
}

func kv0Continuation(t *testing.T, g *kv0Rig) {
	const (
		trials = 50
		np     = 600 // above the floor, so the prefill is fast-class
		d      = 32  // decode rows
	)
	log := func(format string, a ...any) { fmt.Fprintf(os.Stderr, "[kv0] "+format+"\n", a...) }
	start, last := time.Now(), time.Now()
	g.chunk("4096")
	type stat struct {
		maxAbsAB, relAB float64 // decode rows vs the fast prefill's rows: max |diff|, rms diff / rms value
		diffAB, totAB   int
		prefixDiffAB    int     // rows [0, np) of A vs B: the same tokens, a 600- and a 632-row pass
		diffAC, totAC   int     // A' vs C: must be 0
		cosAB           float64 // next-token logits, A's last decode step vs B's prefill
		argAB           bool
		cosAC           float64
		argAC           bool
		bitsAB, bitsAC  int // logits that differ bit for bit
		maxValue        float64
		marginB         float64 // B's top-1 minus top-2 logit
	}
	var all []stat
	n := np + d
	for i := 0; i < trials; i++ {
		ids := g.tokens(n, i*997)
		e := g.embs(ids)
		decodeRun := func(floor string) (kv0KV, []float32) {
			g.floorKnob(floor)
			g.poison(n)
			g.prefill(e[:np], 0)
			var lg []float32
			for k := 0; k < d; k++ {
				out, err := g.r.Forward(e[np+k], np+k)
				if err != nil {
					t.Fatalf("trial %d decode step %d: %v", i, k, err)
				}
				lg = append([]float32(nil), out...)
			}
			return g.snapshot(n), lg
		}
		wholeRun := func(floor string) (kv0KV, []float32) {
			g.floorKnob(floor)
			g.poison(n)
			lg := g.prefill(e, 0)
			return g.snapshot(n), lg
		}
		kA, lA := decodeRun("")
		kB, lB := wholeRun("")
		kA2, lA2 := decodeRun("1000000")
		kC, lC := wholeRun("1000000")
		g.floorKnob("")

		var s stat
		var sq, sv float64
		for ti := range kA.t {
			kd := kA.kvDim[ti]
			for j := np * kd; j < n*kd; j++ {
				a, b := float64(kA.t[ti][j]), float64(kB.t[ti][j])
				s.totAB++
				if kA.t[ti][j] != kB.t[ti][j] {
					s.diffAB++
				}
				s.maxAbsAB = math.Max(s.maxAbsAB, math.Abs(a-b))
				s.maxValue = math.Max(s.maxValue, math.Abs(b))
				sq += (a - b) * (a - b)
				sv += b * b
			}
		}
		s.relAB = math.Sqrt(sq / sv)
		s.prefixDiffAB, _ = kv0CompareRows(kB, kA, 0, np)
		s.diffAC, s.totAC = kv0CompareRows(kC, kA2, np, n)
		s.cosAB, s.argAB = kv0Cosine(lA, lB), kv0Argmax(lA) == kv0Argmax(lB)
		s.cosAC, s.argAC = kv0Cosine(lA2, lC), kv0Argmax(lA2) == kv0Argmax(lC)
		s.bitsAB, _ = kv0LogitsDiff(lA, lB)
		s.bitsAC, _ = kv0LogitsDiff(lA2, lC)
		top1, top2 := float32(math.Inf(-1)), float32(math.Inf(-1))
		for _, v := range lB {
			if v > top1 {
				top1, top2 = v, top1
			} else if v > top2 {
				top2 = v
			}
		}
		s.marginB = float64(top1 - top2)
		all = append(all, s)
		if time.Since(last) > 30*time.Second {
			log("continuation: %d of %d trials done, %s elapsed", i+1, trials, time.Since(start).Round(time.Second))
			last = time.Now()
		}
	}
	var maxAbs, maxVal, worstRel, sumRel, minCosAB, sumCosAB, minCosAC float64 = 0, 0, 0, 0, 1, 0, 1
	var diffAB, totAB, prefix, diffAC, argAB, argAC, bitsAB, bitsAC, fullBitMatch int
	margins := make([]float64, 0, trials)
	for _, s := range all {
		maxAbs, maxVal = math.Max(maxAbs, s.maxAbsAB), math.Max(maxVal, s.maxValue)
		worstRel, sumRel = math.Max(worstRel, s.relAB), sumRel+s.relAB
		diffAB, totAB, prefix, diffAC = diffAB+s.diffAB, totAB+s.totAB, prefix+s.prefixDiffAB, diffAC+s.diffAC
		minCosAB, sumCosAB, minCosAC = math.Min(minCosAB, s.cosAB), sumCosAB+s.cosAB, math.Min(minCosAC, s.cosAC)
		bitsAB, bitsAC = bitsAB+s.bitsAB, bitsAC+s.bitsAC
		if s.argAB {
			argAB++
		}
		if s.argAC {
			argAC++
		}
		if s.diffAB == 0 && s.bitsAB == 0 {
			fullBitMatch++
		}
		margins = append(margins, s.marginB)
	}
	sort.Float64s(margins)
	log("continuation, KV f16=%v, %d trials, np=%d d=%d, same teacher-forced tokens:", g.r.layers[0].kvHalf, trials, np, d)
	log("  decode rows vs fast-prefill rows (A vs B): %d of %d elements differ; max |diff| %.3g (largest |value| %.3g); rms diff / rms value mean %.3g worst %.3g",
		diffAB, totAB, maxAbs, maxVal, sumRel/trials, worstRel)
	log("  prompt rows [0,%d) of A vs B (a %d-row pass against the first %d rows of a %d-row pass): %d elements differ", np, np, np, n, prefix)
	log("  next-token logits A vs B: cosine min %.9f mean %.9f; argmax agrees %d of %d; trials with bit-identical logits and rows %d; logits differing bit for bit %d",
		minCosAB, sumCosAB/trials, argAB, trials, fullBitMatch, bitsAB)
	log("  exact class (A' vs C): decode rows differing %d; cosine min %.9f; argmax agrees %d of %d; logits differing bit for bit %d",
		diffAC, minCosAC, argAC, trials, bitsAC)
	log("  B's top-1 minus top-2 logit: min %.3g median %.3g", margins[0], margins[len(margins)/2])
	log("done in %s", time.Since(start).Round(time.Second))
	if diffAC != 0 || bitsAC != 0 {
		t.Errorf("exact-class prefill and decode differ (%d K/V elements, %d logits): the premise that decode equals the exact class does not hold", diffAC, bitsAC)
	}
}
