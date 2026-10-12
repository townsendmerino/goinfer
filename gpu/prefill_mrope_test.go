//go:build gpu && goinfer_testhooks

package gpu

import (
	"context"
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/oliverbestmann/webgpu/wgpu"
	"github.com/townsendmerino/goinfer/decoder"
)

// ropeInPlace runs one rotary dispatch over a copy of vec and returns the rotated copy.
func ropeInPlace(t *testing.T, c *Context, vec []float32, pl *wgpu.ComputePipeline, layout *wgpu.BindGroupLayout, gx, gy uint32, rest ...[]byte) []float32 {
	t.Helper()
	dev := c.device
	var keep []func()
	defer func() {
		for _, f := range keep {
			f()
		}
	}()
	mk := func(contents []byte, usage wgpu.BufferUsage) *wgpu.Buffer {
		b, err := dev.TryCreateBufferInit(&wgpu.BufferInitDescriptor{Contents: contents, Usage: usage})
		if err != nil {
			t.Fatal(err)
		}
		keep = append(keep, b.Release)
		return b
	}
	bufs := []*wgpu.Buffer{mk(wgpu.ToBytes(vec), wgpu.BufferUsageStorage|wgpu.BufferUsageCopySrc)}
	for i, r := range rest {
		usage := wgpu.BufferUsageStorage
		if i == 1 { // the uniform is always the third binding
			usage = wgpu.BufferUsageUniform
		}
		bufs = append(bufs, mk(r, usage))
	}
	es := make([]wgpu.BindGroupEntry, len(bufs))
	for i, b := range bufs {
		es[i] = wgpu.BindGroupEntry{Binding: uint32(i), Buffer: b, Size: b.GetSize()}
	}
	bg, err := dev.TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: layout, Entries: es})
	if err != nil {
		t.Fatal(err)
	}
	keep = append(keep, bg.Release)
	enc, err := dev.TryCreateCommandEncoder(nil)
	if err != nil {
		t.Fatal(err)
	}
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pl)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(gx, gy, 1)
	if err := pass.TryEnd(); err != nil {
		t.Fatal(err)
	}
	pass.Release()
	cmd, err := enc.TryFinish(nil)
	if err != nil {
		t.Fatal(err)
	}
	c.queue.Submit(cmd)
	cmd.Release()
	out, err := c.readbackRaw(bufs[0], len(vec))
	if err != nil {
		t.Fatal(err)
	}
	return out
}

// G-S12.4b-4a-1 (docs/tasks/task-multimodal-support-2026-10.md): the per-row-position rotary kernel against decode's
// single-row rotary kernel run once per axis at that axis's position, each pair taken from the run its axis names.
func TestMRoPERows_matchesSingleRow(t *testing.T) {
	c := newOrSkipHW(t)
	defer c.Close()
	for _, f := range []func() error{c.ensureAttn, c.ensurePrefillBatched, c.ensureMRoPERows} {
		if err := f(); err != nil {
			t.Fatal(err)
		}
	}
	const heads, hd, M = 4, 16, 9
	const half = hd / 2
	rng := rand.New(rand.NewSource(5))
	vec := make([]float32, M*heads*hd)
	for i := range vec {
		vec[i] = float32(rng.NormFloat64())
	}
	inv := make([]float32, half)
	for d := range inv {
		inv[d] = float32(math.Pow(10000, -float64(d)/half))
	}
	mpos := make([]uint32, 3*M)
	for i := range mpos {
		mpos[i] = uint32(rng.Intn(300))
	}
	gx := uint32(heads*half+63) / 64
	for name, axis := range map[string][]uint32{"sectioned": {0, 0, 1, 1, 1, 2, 2, 2}, "interleaved": {0, 1, 2, 0, 1, 2, 0, 1}} {
		got := ropeInPlace(t, c, vec, c.ropeMRoPERows.pl, c.ropeMRoPERows.layout, gx, M,
			wgpu.ToBytes(inv), wgpu.ToBytes([]uint32{heads, hd, half, 0, f32bits(1), 0, 0, 0}), wgpu.ToBytes(mpos), wgpu.ToBytes(axis))
		diff := 0
		for r := range M {
			row := vec[r*heads*hd : (r+1)*heads*hd]
			var byAxis [3][]float32
			for a := range byAxis {
				byAxis[a] = ropeInPlace(t, c, row, c.ropePipeline, c.ropeLayout, gx, 1,
					wgpu.ToBytes(inv), wgpu.ToBytes([]uint32{heads, hd, half, mpos[r*3+a], f32bits(1), 0, 0, 0}))
			}
			for h := range heads {
				for d := range half {
					for _, e := range []int{h*hd + d, h*hd + half + d} {
						if math.Float32bits(got[r*heads*hd+e]) != math.Float32bits(byAxis[axis[d]][e]) {
							diff++
						}
					}
				}
			}
		}
		if diff != 0 {
			t.Errorf("%s axis table: %d of %d rotated values differ in their bits from the single-row kernel's", name, diff, len(vec))
		}
	}
}

// mropeCos is the cosine of two logit vectors and whether b's argmax is a's or a near-tie under a (a gives b's choice at
// least half the probability of its own).
func mropeCos(a, b []float32) (cos float64, argOK bool) {
	var dot, na, nb float64
	for j := range a {
		dot += float64(a[j]) * float64(b[j])
		na += float64(a[j]) * float64(a[j])
		nb += float64(b[j]) * float64(b[j])
	}
	ia, ib := argmax(a), argmax(b)
	return dot / math.Sqrt(na*nb), ia == ib || math.Exp(float64(a[ib]-a[ia])) >= 0.5
}

// G-S12.4b-4a-2: the pass on rows whose three positions are equal and are not their cache slots, against the walk
// through ForwardMRoPE: the last row and 8 decode steps, bit for bit.
func TestPrefillMRoPE_equalPositionsMatchWalk(t *testing.T) {
	if _, err := New(); err != nil {
		t.Skipf("no WebGPU adapter: %v", err)
	}
	mropeEqualPositions(t, filepath.Join("..", "testdata", "qwen25vl-tiny"), decoder.Options{ResidentContext: 96}, 7, 40)
}

// mropeEqualPositions is G-S12.4b-4a-2's reading on one model at each row count.
func mropeEqualPositions(t *testing.T, dir string, opts decoder.Options, counts ...int) {
	t.Helper()
	opts.Backend, opts.Quant = "webgpu", "int4"
	for _, n := range counts {
		m, err := decoder.Load(dir, opts)
		if err != nil {
			t.Fatal(err)
		}
		rd, ok := m.ResidentForwardForTest().(*residentDecoder)
		if !ok {
			t.Fatalf("the resident is a %T", m.ResidentForwardForTest())
		}
		_, _, _, _, _, _, vocab := m.Dims()
		rng := rand.New(rand.NewSource(int64(n)))
		rows := make([][]float32, n)
		mrope := make([][3]int, n)
		for i := range rows {
			rows[i] = m.EmbedResidentForTest(rng.Intn(vocab - 1))
			p := i
			if i >= 5 {
				p += 37 // as text after an image: the rotary position runs ahead of the cache slot
			}
			mrope[i] = [3]int{p, p, p}
		}
		next := mrope[n-1][0] + 1
		read := func(prefill func() ([]float32, error), toks []int) ([][]float32, []int) {
			rd.Reset()
			last, err := prefill()
			if err != nil {
				t.Fatalf("%d rows: %v", n, err)
			}
			out := [][]float32{append([]float32(nil), last...)}
			var chosen []int
			tok := argmax(last)
			for i := range prefillInt4Steps {
				if toks != nil {
					tok = toks[i]
				}
				chosen = append(chosen, tok)
				l, err := rd.ForwardMRoPE(m.EmbedResidentForTest(tok), n+i, next+i)
				if err != nil {
					t.Fatal(err)
				}
				out = append(out, append([]float32(nil), l...))
				tok = argmax(l)
			}
			return out, chosen
		}
		want, toks := read(func() ([]float32, error) {
			var last []float32
			for i, row := range rows {
				l, err := rd.ForwardMRoPE(row, i, mrope[i][0])
				if err != nil {
					return nil, err
				}
				last = l
			}
			return last, nil
		}, nil)
		got, _ := read(func() ([]float32, error) { return rd.prefillBlocks(context.Background(), rows, 0, mrope) }, toks)
		if why := prefillInt4Grade(want, got); why != "" {
			t.Errorf("%d rows: %s", n, why)
		}
		m.Close()
	}
}

// mropePrompt is one image prompt: its tokens, the image token and where its run starts, the patch grid and the merged
// features of the run.
type mropePrompt struct {
	InputIDs      []int     `json:"input_ids"`
	ImageToken    int       `json:"image_token_id"`
	ImageStart    int       `json:"image_token_start"`
	NImageTokens  int       `json:"n_image_tokens"`
	GridTHW       [][3]int  `json:"grid_thw"`
	ImageFeatures []float32 `json:"image_features"`
}

// mropeGolden reads a tiny fixture's image prompt from its golden.
func mropeGolden(t *testing.T, path string) func(*decoder.Model) mropePrompt {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("no golden (%s)", path)
	}
	var g mropePrompt
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return func(*decoder.Model) mropePrompt { return g }
}

// mropeImageReading is G-S12.4b-4a-3's reading on one model: the worst cosine, over the last row and 8 teacher-forced
// decode steps on the resident, between the CPU's prefill uploaded and the device's prefill, for a text prompt (the
// control, where the device's prefill is the walk's bits) and for the golden's image prompt; and whether every argmax
// difference on the image prompt was a near-tie.
func mropeImageReading(t *testing.T, dir string, prompt func(m *decoder.Model) mropePrompt) (control, image float64, argOK bool) {
	t.Helper()
	m, err := decoder.Load(dir, decoder.Options{Backend: "webgpu", Quant: "int4"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	rd, ok := m.ResidentForwardForTest().(*residentDecoder)
	if !ok {
		t.Fatalf("the resident is a %T", m.ResidentForwardForTest())
	}
	if b := rd.c.Backend(); !mropePrefillBackends[b] { // the gate runs the device prefill wherever it is run
		mropePrefillBackends[b] = true
		defer delete(mropePrefillBackends, b)
	}
	ctx := context.Background()
	g := prompt(m)
	n := len(g.InputIDs)
	mropePos, err := decoder.MRopePositionsForTest(g.InputIDs, g.ImageToken, g.GridTHW, 2)
	if err != nil {
		t.Fatal(err)
	}
	// worst is the lowest cosine between the two routes over the last row and the decode steps, the CPU route's argmax
	// choosing the tokens; rope is the first decode step's rotary position.
	worst := func(cpu func(cache *decoder.KVCache) ([]float32, error), dev func() ([]float32, error), rope int) (float64, bool) {
		rd.Reset()
		cache := m.NewCache(n + prefillInt4Steps + 1)
		lc, err := cpu(cache)
		if err != nil {
			t.Fatal(err)
		}
		if err := m.ResidentUploadPrefillForTest(cache); err != nil {
			t.Fatal(err)
		}
		steps := func(first []float32, toks []int) ([][]float32, []int) {
			out := [][]float32{append([]float32(nil), first...)}
			var chosen []int
			tok := argmax(first)
			for i := range prefillInt4Steps {
				if toks != nil {
					tok = toks[i]
				}
				chosen = append(chosen, tok)
				l, err := rd.ForwardMRoPE(m.EmbedResidentForTest(tok), n+i, rope+i)
				if err != nil {
					t.Fatal(err)
				}
				out = append(out, append([]float32(nil), l...))
				tok = argmax(l)
			}
			return out, chosen
		}
		a, toks := steps(lc, nil)
		rd.Reset()
		ld, err := dev()
		if err != nil {
			t.Fatal(err)
		}
		b, _ := steps(ld, toks)
		low, ok := 1.0, true
		for s := range a {
			cos, argOK := mropeCos(a[s], b[s])
			if !(cos >= low) {
				low = cos
			}
			ok = ok && argOK
		}
		return low, ok
	}
	_, _, _, _, _, _, vocab := m.Dims()
	rng := rand.New(rand.NewSource(9))
	text := make([]int, n)
	rows := make([][]float32, n)
	for i := range text {
		for text[i] = rng.Intn(vocab - 1); text[i] == g.ImageToken; {
			text[i] = rng.Intn(vocab - 1)
		}
		rows[i] = m.EmbedResidentForTest(text[i])
	}
	control, _ = worst(
		func(cache *decoder.KVCache) ([]float32, error) { return m.PrefillLogitsForTest(ctx, text, cache) },
		func() ([]float32, error) { return rd.PrefillLast(ctx, rows, 0) }, n)
	image, argOK = worst(
		func(cache *decoder.KVCache) ([]float32, error) {
			return m.PrefillLogitsQwenVLForTest(ctx, g.InputIDs, g.ImageFeatures, g.ImageStart, g.NImageTokens, mropePos, cache)
		},
		func() ([]float32, error) {
			l, _, err := m.ResidentMRoPEPrefillForTest(ctx, rd, g.InputIDs, g.ImageFeatures, g.ImageStart, g.NImageTokens, mropePos)
			return l, err
		}, mropePos[n-1][0]+1)
	return control, image, argOK
}

// mropeImageBar is G-S12.4b-4a-3's bar on the image prompt's worst cosine, from the text control's: the image prompt
// may sit twice as far from 1 as the control does, and 1e-5 further.
func mropeImageBar(control float64) float64 { return 1 - (2*(1-control) + 1e-5) }

// G-S12.4b-4a-3: an image prompt through the device's m-RoPE prefill against the CPU's image prefill, with the text
// control read first, and the planted defects, each of which must fall below the bar.
func TestPrefillMRoPE_imageAgainstCPU(t *testing.T) {
	if _, err := New(); err != nil {
		t.Skipf("no WebGPU adapter: %v", err)
	}
	dir := filepath.Join("..", "testdata", "qwen25vl-tiny")
	golden := mropeGolden(t, filepath.Join("..", "testdata", "qwen25vl_tiny_image_golden.json"))
	control, image, argOK := mropeImageReading(t, dir, golden)
	bar := mropeImageBar(control)
	t.Logf("control (text prompt) worst cosine %.9f; image prompt worst cosine %.9f (bar %.9f); argmax equal or near-tie: %v", control, image, bar, argOK)
	if !(image >= bar) || !argOK {
		t.Errorf("the image prompt's device prefill sits further from the CPU's than the text control allows: %.9f under %.9f, near-ties %v", image, bar, argOK)
	}
	defer func() { prefillDefectForTest = 0 }()
	for d, name := range map[int]string{12: "the axis table rotated by one pair", 13: "each row's positions taken from the next row", 14: "height and width swapped"} {
		prefillDefectForTest = d
		_, bad, badArg := mropeImageReading(t, dir, golden)
		prefillDefectForTest = 0
		if bad >= bar && badArg {
			t.Errorf("%s: the gate stayed green (worst cosine %.9f)", name, bad)
			continue
		}
		t.Logf("%s: red (worst cosine %.9f, near-ties %v)", name, bad, badArg)
	}
}

// G-S12.4b-4a-2 on a real Qwen2.5-VL checkpoint (GOINFER_RESIDENT_GGUF): the equal-position pass against the walk at
// 64 and 300 rows. An image prompt is not read here: against the CPU's prefill, random tokens and random features at
// real size read anywhere from 0.50 to 0.97 with the right positions and can read better with wrong ones, so that
// reading grades nothing. A real image turn is compared through the server instead.
func TestPrefillMRoPE_real(t *testing.T) {
	mropeEqualPositions(t, prefillInt4RealPath(t), decoder.Options{}, 64, 300)
}
