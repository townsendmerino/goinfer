//go:build gpu && goinfer_testhooks

package gpu

import (
	"context"
	"fmt"
	"github.com/townsendmerino/goinfer/gpu/internal/kernels"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// KV0's WebGPU invariance gates (docs/measurements/kv0-invariance-2026-10-13.md): the twin of metal's
// TestMC5_prefillChunkInvariance and of cuda's TestKV0_prefillInvariance_*. A prompt prefilled whole is compared, K and V
// row by row and the last logits, with the same prompt prefilled in blocks of prefillBlockRows, split into two calls at an
// offset (what a prefix hit does), and with each multi-row int4 kernel pinned in turn. The registered outcome is that every
// arm is invariant: no kernel on this path reads the prompt's length.
//
//	GOINFER_HEAVY_TESTS=1 go test -count=1 -timeout 30m -tags 'gpu goinfer_testhooks' -run '^TestKV0_' -v ./gpu/ 2>&1 | tee /tmp/test-kv0-gpu.log

const kv0Ctx = 2048

type kv0KV struct {
	t     [][]float32
	kvDim []int
}

type kv0Diff struct {
	total, differ, firstPos, firstLayer, before, after int
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

func kv0LogitsDiff(a, b []float32) (differ int, worst float64) {
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			differ++
			worst = math.Max(worst, math.Abs(float64(a[i])-float64(b[i])))
		}
	}
	return differ, worst
}

type kv0Rig struct {
	t      *testing.T
	m      *decoder.Model
	rd     *residentDecoder
	vocab  int
	tokens func(n, off int) []int
	ctx    context.Context
}

func kv0Load(t *testing.T, path string, tokens func(vocab int) func(n, off int) []int) *kv0Rig {
	t.Helper()
	m, err := decoder.Load(path, decoder.Options{Backend: "webgpu", Quant: "int4", ResidentContext: kv0Ctx})
	if err != nil {
		t.Skipf("no WebGPU device or load failed: %v", err)
	}
	t.Cleanup(func() { _ = m.Close() })
	rd, ok := m.ResidentForwardForTest().(*residentDecoder)
	if !ok || rd == nil {
		t.Fatalf("%s: no WebGPU resident (%s)", path, m.DecodePath())
	}
	if rd.rm.KvF16 || rd.rm.KvI8 {
		t.Fatalf("%s: KV is not f32; this gate reads f32 rows", path)
	}
	if rd.rm.SlidingWindow > 0 || rd.rm.Mla != nil || rd.rm.Dnet != nil || rd.rm.Mamba != nil {
		t.Fatalf("%s: windowed, latent or recurrent layers; this gate reads whole-context K/V rows", path)
	}
	_, _, _, _, _, _, vocab := m.Dims()
	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Minute)
	t.Cleanup(cancel)
	t.Logf("backend %q, decode path %s", rd.c.Backend(), m.DecodePath())
	return &kv0Rig{t: t, m: m, rd: rd, vocab: vocab, tokens: tokens(vocab), ctx: ctx}
}

func (g *kv0Rig) layerDim(l int) int {
	if ly := g.rd.rm.Layers[l]; ly.Ghd != 0 {
		return ly.GnKV * ly.Ghd
	}
	return g.rd.nKV * g.rd.hd
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
	for l, ly := range g.rd.rm.Layers {
		if ly.KvShared {
			continue
		}
		nan := make([]float32, n*g.layerDim(l))
		for i := range nan {
			nan[i] = float32(math.NaN())
		}
		if err := g.rd.UploadKV(l, 0, nan, nan); err != nil {
			g.t.Fatalf("poison layer %d: %v", l, err)
		}
	}
}

func (g *kv0Rig) snapshot(n int) kv0KV {
	g.t.Helper()
	var s kv0KV
	for l, ly := range g.rd.rm.Layers {
		if ly.KvShared {
			continue
		}
		kd := g.layerDim(l)
		for _, b := range []*DeviceBuffer{kernels.DeviceBufferLit(ly.KCache, n*kd), kernels.DeviceBufferLit(ly.VCache, n*kd)} {
			f, err := g.rd.c.Readback(b)
			if err != nil {
				g.t.Fatalf("snapshot layer %d: %v", l, err)
			}
			s.t = append(s.t, f)
			s.kvDim = append(s.kvDim, kd)
		}
	}
	return s
}

func (g *kv0Rig) prefill(e [][]float32, start int) []float32 {
	g.t.Helper()
	lg, err := g.rd.PrefillLast(g.ctx, e, start)
	if err != nil {
		g.t.Fatalf("PrefillLast(%d rows at %d): %v", len(e), start, err)
	}
	return append([]float32(nil), lg...)
}

// blockRows sets prefillBlockRows for the rest of the test.
func (g *kv0Rig) blockRows(n int) {
	old := prefillBlockRows
	prefillBlockRows = n
	g.t.Cleanup(func() { prefillBlockRows = old })
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
	if _, err := os.Stat(dir + "/config.json"); err != nil {
		t.Skipf("no fixture at %s", dir)
	}
	kv0Invariance(t, kv0Load(t, dir, kv0RandomTokens))
}

func TestKV0_prefillInvariance_real1p5B(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads the 1.5B)")
	}
	root := os.Getenv("GOINFER_MODELS_DIR")
	if root == "" {
		h, _ := os.UserHomeDir()
		root = filepath.Join(h, "models")
	}
	path := filepath.Join(root, "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s", path)
	}
	kv0Invariance(t, kv0Load(t, path, kv0TextTokens(t, path)))
}

func kv0Invariance(t *testing.T, g *kv0Rig) {
	const N = 1000
	log := func(format string, a ...any) { fmt.Fprintf(os.Stderr, "[kv0] "+format+"\n", a...) }
	start := time.Now()
	ids := g.tokens(N, 0)
	e := g.embs(ids)
	log("fixture loaded, N=%d layers=%d backend=%s", N, len(g.rd.rm.Layers), g.rd.c.Backend())

	g.blockRows(512)
	g.poison(N)
	lgW := g.prefill(e, 0)
	W := g.snapshot(N)

	// Registered outcome: every arm is invariant.
	arm := func(name string, run func() []float32) {
		t.Helper()
		g.poison(N)
		lg := run()
		got := g.snapshot(N)
		d := kv0Compare(W, got, 0)
		ld, worst := kv0LogitsDiff(lgW, lg)
		verdict := "INVARIANT"
		if d.differ != 0 || ld != 0 {
			verdict = fmt.Sprintf("NOT INVARIANT, starts at position %d (layer %d)", d.firstPos, d.firstLayer)
			t.Errorf("%s: %d of %d KV elements and %d logits differ (first at position %d)", name, d.differ, d.total, ld, d.firstPos)
		}
		log("%-28s KV %d of %d elements differ; last logits %d of %d differ (max |diff| %.3g): %s", name, d.differ, d.total, ld, len(lgW), worst, verdict)
	}
	for _, c := range []int{64, 128, 256, 512} {
		arm(fmt.Sprintf("block=%d", c), func() []float32 { g.blockRows(c); return g.prefill(e, 0) })
	}
	g.blockRows(512)
	split := func(s int) {
		arm(fmt.Sprintf("split@%d", s), func() []float32 { g.prefill(e[:s], 0); return g.prefill(e[s:], s) })
	}
	for _, s := range []int{256, 512, 768, 511, N - 16, N - 15} {
		split(s)
	}
	arm("block=496 (8-row tail)", func() []float32 { g.blockRows(496); return g.prefill(e, 0) })
	g.blockRows(512)
	// The two multi-row int4 kernels, pinned, against the row-count choice the whole pass made.
	for _, kind := range []kernels.W4MultiKind{kernels.W4MultiRows, kernels.W4MultiTiled} {
		arm(fmt.Sprintf("w4Multi pinned to kind %d", kind), func() []float32 {
			(*kernels.ContextW4MultiPin(g.rd.c)) = int(kind) + 1
			defer func() { (*kernels.ContextW4MultiPin(g.rd.c)) = 0 }()
			return g.prefill(e, 0)
		})
	}

	// Control: one token changed at position 700.
	ids2 := slices.Clone(ids)
	ids2[700] = (ids2[700] + 1) % (g.vocab - 1)
	g.poison(N)
	g.blockRows(256)
	lgC := g.prefill(g.embs(ids2), 0)
	d := kv0Compare(W, g.snapshot(N), 700)
	ld, _ := kv0LogitsDiff(lgW, lgC)
	log("control (token 700 changed): %d KV elements differ before position 700, %d from it on (first at %d); last logits %d of %d differ",
		d.before, d.after, d.firstPos, ld, len(lgW))
	if d.before != 0 || d.after == 0 || d.firstPos != 700 || ld == 0 {
		t.Errorf("control: %d differences before the changed token, %d after, first at %d, %d logits: the comparison is not seeing what it should", d.before, d.after, d.firstPos, ld)
	}
	log("done in %s", time.Since(start).Round(time.Second))
}
