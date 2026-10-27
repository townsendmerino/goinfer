//go:build gpu && goinfer_testhooks

package gpu

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/townsendmerino/goinfer/gpu/internal/kernels"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/oliverbestmann/webgpu/wgpu"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/vision"
	"github.com/townsendmerino/goinfer/multimodal"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// ropeInPlace runs one rotary dispatch over a copy of vec and returns the rotated copy.
func ropeInPlace(t *testing.T, c *Context, vec []float32, pl *wgpu.ComputePipeline, layout *wgpu.BindGroupLayout, gx, gy uint32, rest ...[]byte) []float32 {
	t.Helper()
	dev := (*kernels.ContextDevice(c))
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
	(*kernels.ContextQueue(c)).Submit(cmd)
	cmd.Release()
	out, err := kernels.ContextReadbackRaw(c, bufs[0], len(vec))
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
	for _, f := range []func() error{kernels.ContextEnsureAttnFunc(c), kernels.ContextEnsurePrefillBatchedFunc(c), kernels.ContextEnsureMRoPERowsFunc(c)} {
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
		got := ropeInPlace(t, c, vec, (*kernels.ContextRopeMRoPERows(c)).Pl, (*kernels.ContextRopeMRoPERows(c)).Layout, gx, M,
			wgpu.ToBytes(inv), wgpu.ToBytes([]uint32{heads, hd, half, 0, kernels.F32bits(1), 0, 0, 0}), wgpu.ToBytes(mpos), wgpu.ToBytes(axis))
		diff := 0
		for r := range M {
			row := vec[r*heads*hd : (r+1)*heads*hd]
			var byAxis [3][]float32
			for a := range byAxis {
				byAxis[a] = ropeInPlace(t, c, row, (*kernels.ContextRopePipeline(c)), (*kernels.ContextRopeLayout(c)), gx, 1,
					wgpu.ToBytes(inv), wgpu.ToBytes([]uint32{heads, hd, half, mpos[r*3+a], kernels.F32bits(1), 0, 0, 0}))
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
	ControlIDs    []int     `json:"-"` // the text control's tokens; nil for random ones of the prompt's length
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
	control, _, image, argOK = mropeImageReading4(t, dir, prompt)
	return control, image, argOK
}

// mropeImageReading4 is mropeImageReading with the control's own near-tie verdict: whether the two routes' argmaxes on
// the text prompt were equal or near-ties at every reading.
func mropeImageReading4(t *testing.T, dir string, prompt func(m *decoder.Model) mropePrompt) (control float64, controlArgOK bool, image float64, argOK bool) {
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
			if !argOK {
				ia, ib := argmax(a[s]), argmax(b[s])
				t.Logf("reading %d of %d: the CPU route picks token %d, the device route %d, which the CPU route gives %.3f of its top probability (cosine %.6f)",
					s, len(a), ia, ib, math.Exp(float64(a[s][ib]-a[s][ia])), cos)
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
		if g.ControlIDs != nil {
			text[i] = g.ControlIDs[i%len(g.ControlIDs)]
		}
		rows[i] = m.EmbedResidentForTest(text[i])
	}
	control, controlArgOK = worst(
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
	return control, controlArgOK, image, argOK
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
	defer func() { kernels.PrefillDefectForTest = 0 }()
	for d, name := range map[int]string{12: "the axis table rotated by one pair", 13: "each row's positions taken from the next row", 14: "height and width swapped"} {
		kernels.PrefillDefectForTest = d
		_, bad, badArg := mropeImageReading(t, dir, golden)
		kernels.PrefillDefectForTest = 0
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

// mropeFirstLayerK is the sharpest reading of where the device's m-RoPE prefill put an image prompt's rows: the first
// layer's rotated K, row by row, against the CPU prefill's. A first-layer K is one projection and one rotation from the
// input, so the two routes' quantization has had no depth to drift apart in, and a wrong position turns the row. It
// returns the lowest cosine over the image rows and over the text rows.
func mropeFirstLayerK(t *testing.T, src string, g mropePrompt) (image, text float64) {
	t.Helper()
	m, err := decoder.Load(src, decoder.Options{Backend: "webgpu", Quant: "int4"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	rd, ok := m.ResidentForwardForTest().(*residentDecoder)
	if !ok {
		t.Fatalf("the resident is a %T", m.ResidentForwardForTest())
	}
	if b := rd.c.Backend(); !mropePrefillBackends[b] {
		mropePrefillBackends[b] = true
		defer delete(mropePrefillBackends, b)
	}
	ctx := context.Background()
	n := len(g.InputIDs)
	mropePos, err := decoder.MRopePositionsForTest(g.InputIDs, g.ImageToken, g.GridTHW, 2)
	if err != nil {
		t.Fatal(err)
	}
	cache := m.NewCache(n + 1)
	if _, err := m.PrefillLogitsQwenVLForTest(ctx, g.InputIDs, g.ImageFeatures, g.ImageStart, g.NImageTokens, mropePos, cache); err != nil {
		t.Fatal(err)
	}
	kCPU, _, base := cache.LayerKVForTest(0)
	if base != 0 {
		t.Fatalf("the CPU cache's first layer starts at %d", base)
	}
	if _, _, err := m.ResidentMRoPEPrefillForTest(ctx, rd, g.InputIDs, g.ImageFeatures, g.ImageStart, g.NImageTokens, mropePos); err != nil {
		t.Fatal(err)
	}
	kDev, err := kernels.ContextReadbackRaw(rd.c, rd.rm.Layers[0].KCache, len(kCPU))
	if err != nil {
		t.Fatal(err)
	}
	kvDim := len(kCPU) / n
	image, text = 1, 1
	for r := range n {
		var dot, na, nb float64
		for j := range kvDim {
			a, b := float64(kCPU[r*kvDim+j]), float64(kDev[r*kvDim+j])
			dot, na, nb = dot+a*b, na+a*a, nb+b*b
		}
		cos := dot / math.Sqrt(na*nb)
		if r >= g.ImageStart && r < g.ImageStart+g.NImageTokens {
			if !(cos >= image) {
				image = cos
			}
		} else if !(cos >= text) {
			text = cos
		}
	}
	return image, text
}

// G-S12.4b-4a-3 on a real image turn: Qwen2.5-VL's own prompt for a table image, the tower's features for it, through
// the device's m-RoPE prefill and through the CPU's image prefill. Two readings. The first layer's K, row by row, is
// the one that can see a position at this size, and the planted defects are read on it. The logits after the prefill
// and over 8 decode steps, with a control of English text of the same length, are the registered reading; at this
// size the two routes' quantization keeps them 0.1 or more apart on the control itself, so it bounds little. GOINFER_QWEN25VL_DIR names the checkpoint directory (default ~/models/qwen25vl-3b-instruct); its
// WebGPU sidecar is loaded when it is beside it.
func TestPrefillMRoPE_realImage(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1")
	}
	if _, err := New(); err != nil {
		t.Skipf("no WebGPU adapter: %v", err)
	}
	dir := os.Getenv("GOINFER_QWEN25VL_DIR")
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, "models", "qwen25vl-3b-instruct")
	}
	if _, err := os.Stat(dir); err != nil {
		t.Skipf("no checkpoint at %s", dir)
	}
	tk, err := tokenizer.Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	imgTok, ok := tk.TokenID("<|image_pad|>")
	if !ok {
		t.Fatal("no <|image_pad|> token")
	}
	enc, err := vision.LoadQwenVisionEncoder(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	pp, err := multimodal.LoadQwenPreprocessConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	pp.MaxPixels = min(pp.MaxPixels, 448*448)                                         // 256 merged rows: enough to place every axis, and a few seconds of CPU tower
	data, err := os.ReadFile(filepath.Join("..", "testdata", "glm_ocr", "table.png")) // a table: where a patch sits matters to the reply
	if err != nil {
		t.Fatal(err)
	}
	px, grid, err := multimodal.QwenPreprocess(data, pp)
	if err != nil {
		t.Fatal(err)
	}
	feats, err := enc.Forward(px, [][3]int{grid})
	if err != nil {
		t.Fatal(err)
	}
	enc.Close()
	nImg := grid[0] * grid[1] * grid[2] / 4
	pre, err := tk.Encode("<|im_start|>user\n<|vision_start|>", false)
	if err != nil {
		t.Fatal(err)
	}
	post, err := tk.Encode("<|vision_end|>Transcribe this table.<|im_end|>\n<|im_start|>assistant\n", false)
	if err != nil {
		t.Fatal(err)
	}
	control, err := tk.Encode("<|im_start|>user\nThe harbour was quiet in the early morning, and the boats lay still against the wall while the tide went out. A man walked along the quay with a coil of rope over his shoulder, counting the moorings as he passed them, and a dog followed him at a distance. ", false)
	if err != nil {
		t.Fatal(err)
	}
	g := mropePrompt{ImageToken: imgTok, ImageStart: len(pre), NImageTokens: nImg, GridTHW: [][3]int{grid}, ImageFeatures: feats, ControlIDs: control}
	g.InputIDs = append(g.InputIDs, pre...)
	for range nImg {
		g.InputIDs = append(g.InputIDs, imgTok)
	}
	g.InputIDs = append(g.InputIDs, post...)
	src := dir
	if sc := dir + fmt.Sprintf(".int4.e4h.webgpu.v%d.giw", decoder.GIWFormatVersion); func() bool { _, err := os.Stat(sc); return err == nil }() {
		src = sc
	}
	// The first layer's K, the reading that can see a position: the right wiring, then each planted defect.
	kImage, kText := mropeFirstLayerK(t, src, g)
	const kBar = 0.99999
	fmt.Fprintf(os.Stderr, "REAL IMAGE first-layer K against the CPU's: lowest cosine %.9f over the image rows, %.9f over the text rows (bar %.5f)\n", kImage, kText, kBar)
	if !(kImage >= kBar) || !(kText >= kBar) {
		t.Errorf("the device's first-layer K is not the CPU's: lowest cosine %.9f (image rows), %.9f (text rows), want %.5f", kImage, kText, kBar)
	}
	defer func() { kernels.PrefillDefectForTest = 0 }()
	for _, d := range []int{12, 13, 14} {
		kernels.PrefillDefectForTest = d
		bad, _ := mropeFirstLayerK(t, src, g)
		kernels.PrefillDefectForTest = 0
		fmt.Fprintf(os.Stderr, "REAL IMAGE planted defect %d: first-layer K lowest cosine over the image rows %.9f\n", d, bad)
		if bad >= kBar {
			t.Errorf("planted defect %d leaves the first-layer K at %.9f: this reading cannot see it", d, bad)
		}
	}
	ctl, ctlArgOK, image, argOK := mropeImageReading4(t, src, func(*decoder.Model) mropePrompt { return g })
	bar := mropeImageBar(ctl)
	fmt.Fprintf(os.Stderr, "REAL IMAGE %s (%d image rows, grid %v): control (text prompt) worst cosine %.9f, argmax equal or near-tie: %v; image prompt worst cosine %.9f (bar %.9f), argmax equal or near-tie: %v\n",
		filepath.Base(src), nImg, grid, ctl, ctlArgOK, image, bar, argOK)
	if !(image >= bar) || !argOK {
		t.Errorf("the image prompt's device prefill sits further from the CPU's than the text control allows: %.9f under %.9f, near-ties %v", image, bar, argOK)
	}
}
