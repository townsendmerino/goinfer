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
				t.Fatalf("split at %d: after rows [%d, %d) the windowed logits differ from the full-KV ones at index %d (base %d)", s, call.lo, call.hi, j, win.kvBase)
			}
		}
		decodeSteps(t, full, win, mw, windowOffsetN, 20)
		if win.kvBase == 0 {
			t.Fatalf("split at %d: the windowed resident never compacted in %d positions; the test did not cross its window", s, windowOffsetN+20)
		}
	}
}

// windowRig loads a windowed model with full-KV storage, so every layer's rows can be read back whole.
func windowRig(t *testing.T, path string, ctxCap int, tokens func(vocab int) func(n, off int) []int) *kv0Rig {
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
	windowed := 0
	for l := range r.layers {
		if r.layers[l].kvWin || r.layers[l].kvHalf {
			t.Fatalf("%s: layer %d holds windowed or f16 storage; this test reads whole f32 caches", path, l)
		}
		if r.layers[l].window > 0 {
			windowed++
		}
	}
	if windowed == 0 {
		t.Fatalf("%s: no layer has a sliding window; the test does not exercise the path it names", path)
	}
	t.Logf("%d of %d layers are sliding-window; head dim %d", windowed, len(r.layers), r.layers[0].hd)
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
		out = append(out, windowReading{arm: arm, kv: kv0Compare(ref, g.snapshot(n), arm.split), logits: differ, fastAttn: a1 - a0})
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

// TestWindowedKV_offsetsFastClass_mistral7B is G2b: the same question on the fast kernels, which need a head dimension
// of 64 or 128 and so a real windowed model. It records whether a windowed fused prefill is invariant to a split or a
// chunk width that is not a multiple of 64, past the window. It measures; KV1 does not depend on the answer.
func TestWindowedKV_offsetsFastClass_mistral7B(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads Mistral-7B-Instruct-v0.1)")
	}
	home, _ := os.UserHomeDir()
	path := filepath.Join(home, "models", "mistral-7b-instruct-v0.1.Q4_K_M.gguf")
	requireDeviceAndFile(t, path)
	const n = 5000
	g := windowRig(t, path, 5120, kv0TextTokens(t, path))
	window := int(g.r.layers[0].window)
	if window <= 0 || window >= n-256 {
		t.Fatalf("the model's window is %d; a %d-token prompt does not put enough rows past it", window, n)
	}
	arms := []windowArm{
		{name: "split@4480 (a multiple of 64)", split: 4480},
		{name: "split@4736 (a multiple of 64)", split: 4736},
		{name: "split@4500", split: 4500},
		{name: "split@4737", split: 4737},
		{name: "chunk=448 (a multiple of 64)", chunk: 448},
		{name: "chunk=500", chunk: 500},
	}
	a0, _ := g.launches()
	_, _, got := windowInvariance(t, g, n, arms)
	if a1, _ := g.launches(); a1 == a0 {
		t.Fatalf("no attn_fused launch in any pass: the fast class did not run, so this reads nothing about it")
	}
	for _, r := range got {
		verdict := "INVARIANT"
		if r.kv.differ != 0 || r.logits != 0 {
			verdict = fmt.Sprintf("NOT invariant, first differing position %d (layer %d)", r.kv.firstPos, r.kv.firstLayer)
		}
		t.Logf("READING window %d, %s: %s; %d of %d K/V elements differ, %d logits, %d attn_fused launches",
			window, r.arm.name, verdict, r.kv.differ, r.kv.total, r.logits, r.fastAttn)
		if r.fastAttn == 0 {
			t.Errorf("%s: no attn_fused launch: the arm ran the exact class and says nothing about the fused kernel", r.arm.name)
		}
	}
}
