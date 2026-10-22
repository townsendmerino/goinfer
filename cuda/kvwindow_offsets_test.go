//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// Windowed layers at starts that are not multiples of 64 (docs/measurements/kv1-2026-10-13.md, G2). The KV manager's
// blocks are 64 positions, and a reused prefix ends wherever the prompts diverge, so the prefill that follows a reuse
// starts anywhere. KV0's invariance arms ran on a model with no windowed layer.

// windowOffsetStarts are split points, none a multiple of 64.
var windowOffsetStarts = []int{1, 63, 65, 100, 129, 257, 449}

const windowOffsetN = 480

// TestWindowedKV_offsetsStorage is G2a's storage half: a prompt prefilled in two calls split at each start, on the
// windowed resident and the full-KV one, bit-identical after each call and through the decode steps that follow.
func TestWindowedKV_offsetsStorage(t *testing.T) { eachWindowFixture(t, windowedOffsetsStorage) }

func windowedOffsetsStorage(t *testing.T, fx string) {
	for _, s := range windowOffsetStarts {
		full, win, _, mw := loadWindowedPair(t, fx, 1)
		_, _, _, _, _, _, vocab := mw.Dims()
		rows := make([][]float32, windowOffsetN)
		for i := range rows {
			rows[i] = mw.EmbedResidentForTest((i*37 + 3) % vocab)
		}
		for _, call := range []struct{ lo, hi int }{{0, s}, {s, windowOffsetN}} {
			a, err := full.PrefillLast(context.Background(), rows[call.lo:call.hi], call.lo)
			if err != nil {
				t.Fatalf("split at %d, full prefill of rows [%d, %d): %v", s, call.lo, call.hi, err)
			}
			a = append([]float32(nil), a...)
			b, err := win.PrefillLast(context.Background(), rows[call.lo:call.hi], call.lo)
			if err != nil {
				t.Fatalf("split at %d, windowed prefill of rows [%d, %d): %v", s, call.lo, call.hi, err)
			}
			if j, ok := sameBits(a, b); !ok {
				t.Fatalf("split at %d: after rows [%d, %d) the windowed logits differ from the full-KV ones at index %d (base %d)", s, call.lo, call.hi, j, win.kvBases[win.kvSlot])
			}
		}
		decodeSteps(t, full, win, mw, windowOffsetN, 20)
		if win.kvBases[win.kvSlot] == 0 {
			t.Fatalf("split at %d: the windowed resident never compacted in %d positions; the test did not cross its window", s, windowOffsetN+20)
		}
	}
}

// windowRig loads a model with full-KV storage, so every layer's rows can be read back whole, and requires it to have
// sliding-window layers.
func windowRig(t *testing.T, path string, ctxCap int, tokens func(vocab int) func(n, off int) []int) *kv0Rig {
	t.Helper()
	g := fullKVRig(t, path, ctxCap, tokens)
	windowed := 0
	for l := range g.r.layers {
		if g.r.layers[l].window > 0 {
			windowed++
		}
	}
	if windowed == 0 {
		t.Fatalf("%s: no layer has a sliding window; the test does not exercise the path it names", path)
	}
	t.Logf("%d of %d layers are sliding-window; head dim %d", windowed, len(g.r.layers), g.r.layers[0].hd)
	return g
}

// setWindow gives every layer of a loaded resident a sliding window of w positions. A layer's window is an argument of
// each attention launch, read at launch time, so this takes effect on the next forward; it must be called before the
// first one.
func setWindow(g *kv0Rig, w int) {
	for l := range g.r.layers {
		g.r.layers[l].window = int32(w)
	}
}

func fullKVRig(t *testing.T, path string, ctxCap int, tokens func(vocab int) func(n, off int) []int) *kv0Rig {
	t.Helper()
	m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4", KVPrecision: "f32", ResidentContext: ctxCap})
	if err != nil {
		t.Fatalf("load %s: %v", path, err)
	}
	t.Cleanup(func() { _ = m.Close() })
	r := mustResident(t, m)
	if batched, why := r.PrefillPath(); !batched {
		t.Fatalf("%s: batched prefill declined: %s", path, why)
	}
	for l := range r.layers {
		if r.layers[l].kvWin || r.layers[l].kvHalf {
			t.Fatalf("%s: layer %d holds windowed or f16 storage; this test reads whole f32 caches", path, l)
		}
	}
	_, _, _, _, _, _, vocab := m.Dims()
	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Minute)
	t.Cleanup(cancel)
	return &kv0Rig{t: t, m: m, r: r, vocab: vocab, tokens: tokens(vocab), ctx: ctx}
}

// windowArm is one way of prefilling the same prompt: in two calls split at split, or in one call at chunk width chunk.
type windowArm struct {
	name         string
	split, chunk int
}

type windowReading struct {
	arm      windowArm
	kv       kv0Diff
	logits   int
	fastAttn int
}

// windowInvariance runs every arm against the prompt prefilled in one pass and returns what differed.
func windowInvariance(t *testing.T, g *kv0Rig, n int, arms []windowArm) (whole []float32, ref kv0KV, out []windowReading) {
	t.Helper()
	e := g.embs(g.tokens(n, 0))
	g.poison(n)
	whole = g.prefill(e, 0)
	ref = g.snapshot(n)
	for _, arm := range arms {
		g.poison(n)
		a0, _ := g.launches()
		var lg []float32
		if arm.chunk > 0 {
			g.chunk(fmt.Sprint(arm.chunk))
			lg = g.prefill(e, 0)
			decoder.UnsetKnobEnvForTest(g.t, g.m, "GOINFER_PREFILL_CHUNK")
		} else {
			g.prefill(e[:arm.split], 0)
			lg = g.prefill(e[arm.split:], arm.split)
		}
		a1, _ := g.launches()
		differ, _ := kv0LogitsDiff(whole, lg)
		out = append(out, windowReading{arm: arm, kv: kv0Compare(ref, g.snapshot(n), arm.split+arm.chunk), logits: differ, fastAttn: a1 - a0})
	}
	return whole, ref, out
}

// TestWindowedKV_offsetsInvariance is G2a's invariance half, on the exact class (the only one these fixtures' head
// dimensions can run): a windowed model's prompt split at each start, and in one call at two chunk widths that are not
// multiples of 64, writes the K and V rows and the last logits the whole prompt writes in one pass.
func TestWindowedKV_offsetsInvariance(t *testing.T) { eachWindowFixture(t, windowedOffsetsInvariance) }

func windowedOffsetsInvariance(t *testing.T, fx string) {
	requireCUDADevice(t)
	g := windowRig(t, filepath.Join("..", "testdata", fx), 0, kv0RandomTokens)
	var arms []windowArm
	for _, s := range windowOffsetStarts {
		arms = append(arms, windowArm{name: fmt.Sprintf("split@%d", s), split: s})
	}
	arms = append(arms, windowArm{name: "chunk=50", chunk: 50}, windowArm{name: "chunk=100", chunk: 100})
	whole, ref, got := windowInvariance(t, g, windowOffsetN, arms)
	for _, r := range got {
		if r.fastAttn != 0 {
			t.Errorf("%s: %d attn_fused launches on a fixture whose head dimension cannot run it", r.arm.name, r.fastAttn)
		}
		if r.kv.differ != 0 || r.logits != 0 {
			t.Errorf("%s: NOT invariant: %d of %d K/V elements differ, first at position %d (layer %d); %d logits differ",
				r.arm.name, r.kv.differ, r.kv.total, r.kv.firstPos, r.kv.firstLayer, r.logits)
		}
	}
	// The controls, without which a pass above could be a comparison that reads nothing. One token changed at position
	// 300 must change K/V rows from 300 on and none before. It need not reach the last logits: where every layer is
	// windowed, a token's influence ends a few windows later. So a second token is changed two from the end, and that
	// one must change the logits.
	const changed = 300
	ids := g.tokens(windowOffsetN, 0)
	ids[changed] = (ids[changed] + 1) % (g.vocab - 1)
	g.poison(windowOffsetN)
	g.prefill(g.embs(ids), 0)
	d := kv0Compare(ref, g.snapshot(windowOffsetN), changed)
	if d.before != 0 || d.firstPos != changed || d.after == 0 {
		t.Fatalf("control: a token changed at %d gives %d differing elements before it, the first at %d, %d after: the comparison is not reading the rows", changed, d.before, d.firstPos, d.after)
	}
	ids = g.tokens(windowOffsetN, 0)
	ids[windowOffsetN-2] = (ids[windowOffsetN-2] + 1) % (g.vocab - 1)
	g.poison(windowOffsetN)
	if differ, _ := kv0LogitsDiff(whole, g.prefill(g.embs(ids), 0)); differ == 0 {
		t.Fatalf("control: a token changed two from the end changes no logit: the comparison is not reading the logits")
	}
	t.Logf("%d arms invariant over %d K/V elements; control: first difference at %d, %d elements after it", len(got), got[0].kv.total, d.firstPos, d.after)
}

// G2b: the same question on the fast kernels, which need a head dimension of 64 or 128. No windowed checkpoint that
// fits the card has one, so the window is set on a full-attention model after it loads (setWindow). What was read
// (docs/measurements/kv1-2026-10-13.md, G2b) is pinned here: with a window, a fused prefill pass that starts on a
// multiple of 64 writes the bits the whole prompt writes, and one that starts anywhere else does not, from its first
// rows on; at full attention the start does not matter. The KV manager's class rule rests on it
// (kvmgr.ClassMap.Windowed), so a kernel change that moves either half fails here.
func TestWindowedKV_offsetsFastClass_tiny(t *testing.T) {
	const dir = "../testdata/llama-attnfa-tiny"
	requireDeviceAndFile(t, dir)
	windowedFastClass(t, func() *kv0Rig { return fullKVRig(t, dir, kv0Ctx, kv0RandomTokens) })
}

func TestWindowedKV_offsetsFastClass_real1p5B(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads the 1.5B)")
	}
	home, _ := os.UserHomeDir()
	path := filepath.Join(home, "models", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	requireDeviceAndFile(t, path)
	windowedFastClass(t, func() *kv0Rig { return fullKVRig(t, path, kv0Ctx, kv0TextTokens(t, path)) })
}

func windowedFastClass(t *testing.T, load func() *kv0Rig) {
	const n, window = 1000, 256
	// The control, at full attention: a split at a start that is not a multiple of 64, both calls fast class.
	g := load()
	a0, _ := g.launches()
	_, _, ctl := windowInvariance(t, g, n, []windowArm{{name: "split@600, no window", split: 600}})
	if a1, _ := g.launches(); a1 == a0 {
		t.Fatalf("no attn_fused launch at full attention: the fast class does not run on this model, so nothing here reads it")
	}
	if c := ctl[0]; c.kv.differ != 0 || c.logits != 0 {
		t.Fatalf("control: at full attention a split at 600 is NOT invariant (%d K/V elements from position %d, %d logits); the window arms cannot be read against it", c.kv.differ, c.kv.firstPos, c.logits)
	}
	t.Logf("READING control, full attention, split@600: INVARIANT (%d K/V elements, %d attn_fused launches)", ctl[0].kv.total, ctl[0].fastAttn)

	g = load()
	setWindow(g, window)
	arms := []windowArm{
		{name: "split@576 (a multiple of 64)", split: 576},
		{name: "split@704 (a multiple of 64)", split: 704},
		{name: "split@600", split: 600},
		{name: "split@777", split: 777},
		{name: "chunk=448 (a multiple of 64)", chunk: 448},
		{name: "chunk=500", chunk: 500},
	}
	_, _, got := windowInvariance(t, g, n, arms)
	for _, r := range got {
		start := r.arm.split + r.arm.chunk // where the second pass begins
		invariant := r.kv.differ == 0 && r.logits == 0
		verdict := "INVARIANT"
		if !invariant {
			verdict = fmt.Sprintf("NOT invariant, first differing position %d (layer %d)", r.kv.firstPos, r.kv.firstLayer)
		}
		t.Logf("READING window %d, %s: %s; %d of %d K/V elements differ, %d logits, %d attn_fused launches",
			window, r.arm.name, verdict, r.kv.differ, r.kv.total, r.logits, r.fastAttn)
		if r.fastAttn == 0 {
			t.Errorf("%s: no attn_fused launch: the arm ran the exact class and says nothing about the fused kernel", r.arm.name)
		}
		switch aligned := start%64 == 0; {
		case aligned && !invariant:
			t.Errorf("%s: a pass starting on a multiple of 64 is no longer invariant under a window (first difference at %d)", r.arm.name, r.kv.firstPos)
		case !aligned && invariant:
			t.Errorf("%s: a pass starting off a multiple of 64 is now invariant under a window: the class rule that records it apart (kvmgr.ClassMap.Windowed) can go", r.arm.name)
		case !aligned && (r.kv.before != 0 || r.kv.firstPos < start || r.kv.firstPos >= start+16):
			t.Errorf("%s: the first difference is at %d with %d elements before %d; it should begin with the pass that starts there", r.arm.name, r.kv.firstPos, r.kv.before, start)
		}
	}
}
