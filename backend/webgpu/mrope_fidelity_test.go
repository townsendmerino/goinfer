//go:build gpu && goinfer_testhooks

package webgpu

import (
	"bufio"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"github.com/townsendmerino/goinfer/backend/webgpu/internal/kernels"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/fidelity"
	"github.com/townsendmerino/goinfer/internal/testgate"
	"github.com/townsendmerino/goinfer/internal/vision"
	"github.com/townsendmerino/goinfer/multimodal"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// G-S12.4b-4a-5: which of the two routes an image turn can prefill by
// under WebGPU is nearer the model. Both arms decode on the resident at int4 and are scored, position by position,
// against the float32 CPU forward of the same checkpoint on the same prompt and image features:
//   - exact, the route in use: the CPU's int4 image prefill, uploaded;
//   - candidate: the device's m-RoPE prefill (PrefillMRoPELast).
// A prompt is an image at a pixel budget and a question, in the model's own chat shape. Its positions are the one
// after the prompt and the ones after each of the reference's own greedy tokens, forced on all three.

const mropeFidelityPositions = 32

var mropeFidelityQuestions = []string{
	"Transcribe the text in this image.",
	"What kind of document is this? Answer in one sentence.",
	"List the amounts shown in this document.",
	"Who issued this document, and to whom?",
	"Describe this image.",
}

// mropeFidelityPrompt names one prompt of the set: an image under testdata, the budget cell it is read at and a
// question's index.
type mropeFidelityPrompt struct {
	cell, image string
	question    int
}

func (p mropeFidelityPrompt) id() string {
	return fmt.Sprintf("%s|%s|q%d", p.cell, p.image, p.question)
}

// mropeFidelitySet is the fixed prompt set of a mode. "pilot" sizes the gate and shares no image with it; "gate" and
// "more" are the two registered sets, and "both" is the decision set: the two pooled, "gate" first.
func mropeFidelitySet(mode string) []mropeFidelityPrompt {
	if mode == "both" {
		return append(mropeFidelitySet("gate"), mropeFidelitySet("more")...)
	}
	var ps []mropeFidelityPrompt
	invoices := make([]string, 15)
	for i := range invoices {
		invoices[i] = fmt.Sprintf("glm_ocr/invoices/inv%02d.png", i+1)
	}
	switch mode {
	case "pilot":
		for _, img := range []string{"glm_ocr/formula.png", "glm_ocr/table.png", "glm_ocr/invoice.png"} {
			ps = append(ps, mropeFidelityPrompt{"small", img, 0}, mropeFidelityPrompt{"small", img, 1})
		}
		ps = append(ps, mropeFidelityPrompt{"served", "glm_ocr/table.png", 0}, mropeFidelityPrompt{"served", "glm_ocr/invoice.png", 1})
	case "gate":
		for _, img := range invoices {
			ps = append(ps, mropeFidelityPrompt{"small", img, 0}, mropeFidelityPrompt{"small", img, 1}, mropeFidelityPrompt{"served", img, 2})
		}
		for _, img := range []string{"gemma3_preprocess_image.png", "qwen25vl_preprocess_image.png", "qwen25vl_preprocess_image_resize.png", "qwen35vl_preprocess_image.png"} {
			ps = append(ps, mropeFidelityPrompt{"small", img, 4})
		}
	case "more":
		for _, img := range invoices {
			ps = append(ps, mropeFidelityPrompt{"small", img, 2}, mropeFidelityPrompt{"small", img, 3}, mropeFidelityPrompt{"served", img, 0})
		}
	}
	return ps
}

// mropeFidelityLine is one finished prompt in the live log, which a restarted run reads back and skips.
type mropeFidelityLine struct {
	ID        string              `json:"id"`
	Cell      string              `json:"cell"`
	Rows      int                 `json:"rows"`
	ImageRows int                 `json:"image_rows"`
	Tokens    []int               `json:"reference_tokens"`
	Candidate []fidelity.Position `json:"candidate"`
	Exact     []fidelity.Position `json:"exact"`
}

// mropeScore scores an arm's logits against the reference's at one position.
func mropeScore(ref, arm []float32) fidelity.Position {
	agree, _, hard := decoder.NearTieArgmaxForTest(ref, arm)
	logp := func(l []float32) []float64 {
		mx := float64(l[0])
		for _, v := range l {
			mx = math.Max(mx, float64(v))
		}
		var z float64
		for _, v := range l {
			z += math.Exp(float64(v) - mx)
		}
		out := make([]float64, len(l))
		for i, v := range l {
			out[i] = float64(v) - mx - math.Log(z)
		}
		return out
	}
	lr, la := logp(ref), logp(arm)
	var kl float64
	for i := range lr {
		kl += math.Exp(lr[i]) * (lr[i] - la[i])
	}
	return fidelity.Position{Agree: agree, HardFlip: hard, KL: kl}
}

// TestS12MRoPEFidelity runs one mode of the gate (GOINFER_S12_FIDELITY: plan, pilot, gate, more or both) and writes its
// records under GOINFER_S12_FIDELITY_OUT. "plan" loads no model: it builds the gate's prompts and prints their hash
// and sizes. GOINFER_S12_FIDELITY_DEFECT=1 plants the pass's "each row's positions from the next row" defect in the
// candidate, to show the verdict can go red. It grades by the verdict it prints; a run that completes exits 0.
func TestS12MRoPEFidelity(t *testing.T) {
	mode := os.Getenv("GOINFER_S12_FIDELITY")
	if mode == "" {
		t.Skip("set GOINFER_S12_FIDELITY to plan, pilot, gate, more or both")
	}
	home, _ := os.UserHomeDir()
	dir := os.Getenv("GOINFER_QWEN25VL_DIR")
	if dir == "" {
		dir = filepath.Join(home, "models", "qwen25vl-3b-instruct")
	}
	for _, root := range []string{"/Volumes/", "/srv/models"} {
		if strings.HasPrefix(dir, root) {
			t.Fatalf("%s is the archive, not a place to measure from", dir)
		}
	}
	tk, err := tokenizer.Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	imgTok, ok := tk.TokenID("<|image_pad|>")
	if !ok {
		t.Fatal("no <|image_pad|> token")
	}
	base, err := multimodal.LoadQwenPreprocessConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	began := time.Now()
	say := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "%s [%5.0fs] %s\n", time.Now().Format("15:04:05"), time.Since(began).Seconds(), fmt.Sprintf(format, a...))
	}
	type pixels struct {
		px   []float32
		grid [3]int
	}
	pix := map[string]pixels{} // by cell and image
	pixelsOf := func(p mropeFidelityPrompt) pixels {
		key := p.cell + "|" + p.image
		if v, ok := pix[key]; ok {
			return v
		}
		pp := base
		if p.cell == "small" {
			pp.MaxPixels = min(pp.MaxPixels, 448*448)
		}
		data, err := os.ReadFile(filepath.Join("../../testdata", filepath.FromSlash(p.image)))
		if err != nil {
			t.Fatal(err)
		}
		px, grid, err := multimodal.QwenPreprocess(data, pp)
		if err != nil {
			t.Fatalf("%s: %v", p.image, err)
		}
		pix[key] = pixels{px, grid}
		return pix[key]
	}
	idsOf := func(p mropeFidelityPrompt) (ids []int, start, nImg int) {
		g := pixelsOf(p).grid
		nImg = g[0] * g[1] * g[2] / 4
		pre, err := tk.Encode("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|>", false)
		if err != nil {
			t.Fatal(err)
		}
		post, err := tk.Encode("<|vision_end|>"+mropeFidelityQuestions[p.question]+"<|im_end|>\n<|im_start|>assistant\n", false)
		if err != nil {
			t.Fatal(err)
		}
		ids = append(ids, pre...)
		for range nImg {
			ids = append(ids, imgTok)
		}
		return append(ids, post...), len(pre), nImg
	}
	set := mropeFidelitySet(mode)
	if mode == "plan" {
		set = mropeFidelitySet("gate")
	}
	if len(set) == 0 {
		t.Fatalf("no prompt set for mode %q", mode)
	}
	all := make([][]int, len(set))
	rows := map[string]int{}
	for i, p := range set {
		all[i], _, _ = idsOf(p)
		rows[p.cell] += len(all[i])
	}
	say("mode %s: %d prompts, prompt-set hash %s; rows by cell %v", mode, len(set), fidelity.PromptSetHash(all), rows)
	if mode == "plan" {
		for _, extra := range []string{"pilot", "more", "both"} {
			es := mropeFidelitySet(extra)
			ei := make([][]int, len(es))
			er := 0
			for i, p := range es {
				ei[i], _, _ = idsOf(p)
				er += len(ei[i])
			}
			say("mode %s: %d prompts, prompt-set hash %s; %d rows", extra, len(es), fidelity.PromptSetHash(ei), er)
		}
		return
	}
	if _, err := New(); err != nil {
		t.Fatalf("no WebGPU adapter: %v", err)
	}
	out := os.Getenv("GOINFER_S12_FIDELITY_OUT")
	if out == "" {
		t.Fatal("set GOINFER_S12_FIDELITY_OUT to the directory the records go in")
	}
	if err := os.MkdirAll(out, 0o755); err != nil {
		t.Fatal(err)
	}
	// Finished prompts from an earlier, interrupted run of this mode.
	livePath := filepath.Join(out, "positions-live.jsonl")
	done := map[string]mropeFidelityLine{}
	if f, err := os.Open(livePath); err == nil {
		sc := bufio.NewScanner(f)
		sc.Buffer(make([]byte, 1<<20), 1<<26)
		for sc.Scan() {
			var l mropeFidelityLine
			if json.Unmarshal(sc.Bytes(), &l) == nil && l.ID != "" {
				done[l.ID] = l
			}
		}
		f.Close()
	}
	live, err := os.OpenFile(livePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	defer live.Close()

	enc, err := vision.LoadQwenVisionEncoder(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	defer enc.Close()
	say("loading the float32 reference (CPU) and the int4 resident (WebGPU)")
	mR, err := decoder.Load(dir, decoder.Options{Backend: "cpu", Quant: "f32"})
	if err != nil {
		t.Fatalf("the float32 reference: %v", err)
	}
	defer mR.Close()
	src := dir // the directory's WebGPU sidecar when it is beside it: the same int4 weights, mapped
	if sc := dir + fmt.Sprintf(".int4.e4h.webgpu.v%d.giw", decoder.GIWFormatVersion); func() bool { _, err := os.Stat(sc); return err == nil }() {
		src = sc
	}
	mA, err := decoder.Load(src, decoder.Options{Backend: "webgpu", Quant: "int4", ResidentContext: 4096})
	if err != nil {
		t.Fatalf("the int4 resident: %v", err)
	}
	defer mA.Close()
	rd, ok := mA.ResidentForwardForTest().(*residentDecoder)
	if !ok {
		t.Fatalf("the model is not resident on WebGPU: %s", mA.ResidentDecline())
	}
	backend := rd.c.Backend()
	if !mropePrefillBackends[backend] { // the gate reads the device prefill wherever it is run
		mropePrefillBackends[backend] = true
		defer delete(mropePrefillBackends, backend)
	}
	defect := os.Getenv("GOINFER_S12_FIDELITY_DEFECT") == "1"
	say("backend %q, decode path %s; planted defect in the candidate: %v; %d of %d prompts already in %s", backend, mA.DecodePath(), defect, len(done), len(set), livePath)

	ctx := context.Background()
	feats := map[string][]float32{}
	for i, p := range set {
		if _, ok := done[p.id()]; ok {
			continue
		}
		px := pixelsOf(p)
		key := p.cell + "|" + p.image
		if feats[key] == nil {
			t0 := time.Now()
			if feats[key], err = enc.Forward(px.px, [][3]int{px.grid}); err != nil {
				t.Fatalf("%s: tower: %v", p.id(), err)
			}
			say("tower: %s at %v in %.1f s", key, px.grid, time.Since(t0).Seconds())
		}
		ids, start, nImg := idsOf(p)
		n := len(ids)
		mrope, err := decoder.MRopePositionsForTest(ids, imgTok, [][3]int{px.grid}, 2)
		if err != nil {
			t.Fatal(err)
		}
		delta := mrope[n-1][0] + 1 - n
		// The reference, and the tokens it forces on both arms.
		cR := mR.NewCache(n + mropeFidelityPositions + 1)
		l, err := mR.PrefillLogitsQwenVLForTest(ctx, ids, feats[key], start, nImg, mrope, cR)
		if err != nil {
			t.Fatalf("%s: reference prefill: %v", p.id(), err)
		}
		ref := [][]float32{append([]float32(nil), l...)}
		toks := []int{argmax(l)}
		for k := 1; k < mropeFidelityPositions; k++ {
			if l, err = mR.ForwardForTest(toks[k-1], cR); err != nil {
				t.Fatalf("%s: reference step %d: %v", p.id(), k, err)
			}
			ref = append(ref, append([]float32(nil), l...))
			toks = append(toks, argmax(l))
		}
		arm := func(prefill func() ([]float32, error)) []fidelity.Position {
			rd.Reset()
			l, err := prefill()
			if err != nil {
				t.Fatalf("%s: %v", p.id(), err)
			}
			pos := []fidelity.Position{mropeScore(ref[0], l)}
			for k := 1; k < mropeFidelityPositions; k++ {
				if l, err = rd.ForwardMRoPE(mA.EmbedResidentForTest(toks[k-1]), n+k-1, n+k-1+delta); err != nil {
					t.Fatalf("%s: resident step %d: %v", p.id(), k, err)
				}
				pos = append(pos, mropeScore(ref[k], l))
			}
			return pos
		}
		line := mropeFidelityLine{ID: p.id(), Cell: p.cell, Rows: n, ImageRows: nImg, Tokens: toks}
		line.Exact = arm(func() ([]float32, error) {
			cA := mA.NewCache(n + mropeFidelityPositions + 1)
			l, err := mA.PrefillLogitsQwenVLForTest(ctx, ids, feats[key], start, nImg, mrope, cA)
			if err != nil {
				return nil, err
			}
			l = append([]float32(nil), l...)
			return l, mA.ResidentUploadPrefillForTest(cA)
		})
		if defect {
			kernels.PrefillDefectForTest = 13
		}
		line.Candidate = arm(func() ([]float32, error) {
			l, _, err := mA.ResidentMRoPEPrefillForTest(ctx, rd, ids, feats[key], start, nImg, mrope)
			return l, err
		})
		kernels.PrefillDefectForTest = 0
		b, _ := json.Marshal(line)
		if _, err := live.Write(append(b, '\n')); err != nil {
			t.Fatal(err)
		}
		done[p.id()] = line
		tally := func(ps []fidelity.Position) (agree int, kl float64) {
			for _, q := range ps {
				agree += int(b2i(q.Agree))
				kl += q.KL
			}
			return agree, kl / float64(len(ps))
		}
		ea, ek := tally(line.Exact)
		ca, ck := tally(line.Candidate)
		say("prompt %d of %d %s: %d rows (%d image); agreement exact %d/%d candidate %d/%d; mean KL exact %.4f candidate %.4f",
			i+1, len(set), p.id(), n, nImg, ea, mropeFidelityPositions, ca, mropeFidelityPositions, ek, ck)
	}

	cells := map[string]*fidelity.Cell{}
	var order []string
	for _, p := range set {
		l := done[p.id()]
		if cells[p.cell] == nil {
			cells[p.cell] = &fidelity.Cell{Name: p.cell}
			order = append(order, p.cell)
		}
		cells[p.cell].Prompts = append(cells[p.cell].Prompts, fidelity.Prompt{ID: l.ID, Candidate: l.Candidate, Exact: l.Exact})
	}
	var cs []fidelity.Cell
	for _, name := range order {
		cs = append(cs, *cells[name])
	}
	pf, err := os.Create(filepath.Join(out, "positions.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	if err := fidelity.WritePositions(pf, cs); err != nil {
		t.Fatal(err)
	}
	pf.Close()
	// The registered criteria, as amended on the owner's word after the pilot: agreement within 2 points, KL within 10%.
	crit := fidelity.Criteria{AgreeMarginPts: 2, KLMargin: 0.10, Alpha: 0.05}
	v, err := fidelity.NonInferior(cs, crit)
	if err != nil {
		t.Fatal(err)
	}
	show := func(name string, s fidelity.Summary) string {
		return fmt.Sprintf("%s: %d prompts, %d positions; agreement candidate %.2f%% exact %.2f%% (difference %+.2f pts, lower bound %+.2f); mean KL candidate %.4f exact %.4f (ratio %.3f, upper bound %.3f); hard flips candidate %d exact %d; discordant positions %d",
			name, s.Prompts, s.Positions, s.AgreeCandPct, s.AgreeExactPct, s.AgreeDiffPts, s.AgreeDiffLCB, s.MeanKLCand, s.MeanKLExact, s.KLRatio, s.KLRatioUCB, s.FlipsCand, s.FlipsExact, s.DiscordantPositions)
	}
	var report strings.Builder
	fmt.Fprintf(&report, "G-S12.4b-4a-5 mode %s, backend %s, planted defect %v, prompt-set hash %s\n", mode, backend, defect, fidelity.PromptSetHash(all))
	fmt.Fprintf(&report, "criteria: agreement margin %.1f pts, KL margin %.2f, one-sided alpha %.2f\n", crit.AgreeMarginPts, crit.KLMargin, crit.Alpha)
	for _, name := range order {
		fmt.Fprintln(&report, show("cell "+name, v.PerCell[name]))
	}
	fmt.Fprintln(&report, show("pooled", v.Pooled))
	// The registered bands: PASS; PARKED when no point estimate breaches its margin and a bound does; OFF otherwise.
	verdict := "OFF"
	switch {
	case v.Pass:
		verdict = "PASS"
	case v.FlipsPass && v.Pooled.AgreeDiffPts >= -crit.AgreeMarginPts && v.Pooled.KLRatio <= 1+crit.KLMargin:
		verdict = "PARKED"
	}
	fmt.Fprintf(&report, "VERDICT %s (hard flips pass %v, agreement pass %v, KL pass %v) %s\n", verdict, v.FlipsPass, v.AgreePass, v.KLPass, strings.Join(v.Reasons, "; "))
	fmt.Fprint(os.Stderr, report.String())
	if err := os.WriteFile(filepath.Join(out, "verdict.txt"), []byte(report.String()), 0o644); err != nil {
		t.Fatal(err)
	}
}

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}

// cohPeaked copies a tiny safetensors fixture with every 2-D F32 tensor scaled by factor, as Metal's and CUDA's pairwise
// gates do: the committed fixtures' ~0.02-std weights give nearly uniform attention, where a wrong rotation hides.
func cohPeaked(t *testing.T, src string, factor float32) string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(src, "model.safetensors"))
	if err != nil {
		t.Fatalf("the committed fixture %s: %v", src, err)
	}
	n := binary.LittleEndian.Uint64(raw[:8])
	var hdr map[string]json.RawMessage
	if err := json.Unmarshal(raw[8:8+n], &hdr); err != nil {
		t.Fatal(err)
	}
	data, scaled := raw[8+n:], 0
	for name, h := range hdr {
		if name == "__metadata__" {
			continue
		}
		var e struct {
			Dtype   string   `json:"dtype"`
			Shape   []int    `json:"shape"`
			Offsets [2]int64 `json:"data_offsets"`
		}
		if err := json.Unmarshal(h, &e); err != nil {
			t.Fatal(err)
		}
		if len(e.Shape) != 2 {
			continue
		}
		if e.Dtype != "F32" {
			t.Fatalf("tensor %s is %s: F32 only", name, e.Dtype)
		}
		for off := e.Offsets[0]; off < e.Offsets[1]; off += 4 {
			binary.LittleEndian.PutUint32(data[off:], math.Float32bits(math.Float32frombits(binary.LittleEndian.Uint32(data[off:]))*factor))
		}
		scaled++
	}
	if scaled < 8 {
		t.Fatalf("scaled only %d tensors", scaled)
	}
	dst := t.TempDir()
	if err := os.WriteFile(filepath.Join(dst, "model.safetensors"), raw, 0o644); err != nil {
		t.Fatal(err)
	}
	for _, f := range []string{"config.json", "generation_config.json"} {
		if b, err := os.ReadFile(filepath.Join(src, f)); err == nil {
			if err := os.WriteFile(filepath.Join(dst, f), b, 0o644); err != nil {
				t.Fatal(err)
			}
		}
	}
	return dst
}

// TestCohereResidentParityWebGPU is Metal's G-PR2 on WebGPU: Cohere (cohere-tiny) and Cohere2 (cohere2-tiny: sliding window and
// a NoPE global layer), every 2-D weight scaled by 12.5 so attention is peaked, resident at int8int8 against the CPU int8int8
// over 40 decoded positions. Bars, Metal's: worst cosine >= 0.995 and relL2 <= 0.15. Each plant must miss them: the NeoX
// rotation (kernels.PairwiseNeoXForTest), RMSNorm in LayerNorm's place (kernels.LayerNormAsRMSForTest), and the parallel
// block's MLP fed from the post-attention residual (kernels.ParallelFromResidualForTest). The batched prefill declines.
func TestCohereResidentParityWebGPU(t *testing.T) {
	if c, err := New(); err != nil {
		t.Skipf("no webgpu: %v", err)
	} else {
		c.Close()
	}
	const n, cosBar, relBar = 40, 0.995, 0.15
	for _, fx := range []string{"cohere-tiny", "cohere2-tiny"} {
		t.Run(fx, func(t *testing.T) {
			ckpt := cohPeaked(t, filepath.Join("../../testdata", fx), 12.5)
			cpu, err := decoder.Load(ckpt, decoder.Options{Backend: "cpu", Quant: "int8int8"})
			if err != nil {
				t.Fatal(err)
			}
			defer cpu.Close()
			_, _, _, _, _, _, vocab := cpu.Dims()
			toks := make([]int, n)
			for i := range toks {
				toks[i] = (i*37 + 3) % vocab
			}
			c := cpu.NewCache(n)
			want := make([][]float32, n)
			for i, tok := range toks {
				l, err := cpu.ForwardForTest(tok, c)
				if err != nil {
					t.Fatal(err)
				}
				want[i] = append([]float32(nil), l...)
			}
			run := func(plant *bool) (cosW, relW float64) {
				if plant != nil {
					*plant = true
					defer func() { *plant = false }()
				}
				m, err := decoder.Load(ckpt, decoder.Options{Backend: "webgpu", Quant: "int8int8"})
				if err != nil {
					t.Fatal(err)
				}
				defer m.Close()
				rd, ok := m.ResidentForwardForTest().(*residentDecoder)
				if !ok || !rd.rm.LayerNorm || !rd.rm.Parallel || !rd.rm.Pairwise {
					t.Fatalf("not resident with LayerNorm, the parallel block and pairwise RoPE: %s", m.ResidentDecline())
				}
				rd.Reset()
				cosW, relW = 1, 0
				embs := make([][]float32, n)
				for i, tok := range toks {
					embs[i] = m.EmbedResidentForTest(tok)
					l, err := rd.Forward(embs[i], i)
					if err != nil {
						t.Fatal(err)
					}
					var ab, aa, bb, dd float64
					for j := range l {
						g, r := float64(l[j]), float64(want[i][j])
						ab, aa, bb, dd = ab+g*r, aa+g*g, bb+r*r, dd+(g-r)*(g-r)
					}
					c, r := ab/math.Sqrt(aa*bb), math.Sqrt(dd/bb)
					if math.IsNaN(c) || math.IsNaN(r) { // NaN, or an all-zero row, is the worst reading
						c, r = -2, math.Inf(1)
					}
					cosW, relW = math.Min(cosW, c), math.Max(relW, r)
				}
				rd.Reset()
				if _, err := rd.PrefillLast(context.Background(), embs[:4], 0); err == nil {
					t.Error("the batched prefill ran on a LayerNorm / parallel-block model")
				}
				return cosW, relW
			}
			cw, rw := run(nil)
			t.Logf("COHERE  worst cos %.6f relL2 %.3g", cw, rw)
			if !(cw >= cosBar && rw <= relBar) {
				t.Errorf("diverges from the CPU: worst cosine %.6f (want >= %g), relL2 %.3g (want <= %g)", cw, cosBar, rw, relBar)
			}
			for name, p := range map[string]*bool{"NeoX rotation": &kernels.PairwiseNeoXForTest, "RMSNorm for LayerNorm": &kernels.LayerNormAsRMSForTest,
				"MLP from the post-attention residual": &kernels.ParallelFromResidualForTest} {
				cw, rw := run(p)
				t.Logf("plant %-38s worst cos %.6f relL2 %.3g", name, cw, rw)
				if !(cw < cosBar || rw > relBar) {
					t.Errorf("the gate is BLIND to %s: worst cosine %.6f, relL2 %.3g", name, cw, rw)
				}
			}
		})
	}
}

// TestPairwiseRoPERealWebGPU is Metal's G-PR4 (TestPairwiseRoPERealMetal) on WebGPU: GLM-OCR's text decoder, Command-R7B and
// Aya-expanse-8B from ~/models, resident at int4 against the CPU int4, 64 tokens of README.md and a 16-token greedy
// continuation the CPU chose, teacher-forced. Bars, CUDA's real-checkpoint tier: mean per-position cosine >= 0.99, worst
// >= 0.90, and every continuation argmax difference a near-tie (within 3% of the CPU's logit range). With the NeoX kernels
// bound (kernels.PairwiseNeoXForTest) each model must miss the mean or the worst bar. The CPU reference is computed and its
// model closed before each WebGPU load. GOINFER_HEAVY_TESTS=1; a model missing from ~/models is skipped by name.
func TestPairwiseRoPERealWebGPU(t *testing.T) {
	testgate.RequireHeavy(t)
	if c, err := New(); err != nil {
		t.Skipf("no webgpu: %v", err)
	} else {
		c.Close()
	}
	home, _ := os.UserHomeDir()
	readme, err := os.ReadFile("../../README.md")
	if err != nil {
		t.Fatal(err)
	}
	const nPrompt, nCont = 64, 16
	argmax := func(v []float32) int {
		b := 0
		for i := range v {
			if v[i] > v[b] {
				b = i
			}
		}
		return b
	}
	ran, cohRan := 0, 0
	t.Cleanup(func() { // a heavy run that gates no Cohere checkpoint (LayerNorm, parallel block) is a failure, not a pass
		if (ran == 0 || cohRan == 0) && !t.Failed() {
			t.Errorf("no checkpoint of command-r7b or aya-expanse-8b under %s (%d of glm-ocr ran): the Cohere path was not gated", filepath.Join(home, "models"), ran)
		}
	})
	for _, name := range []string{"glm-ocr", "command-r7b", "aya-expanse-8b"} {
		t.Run(name, func(t *testing.T) {
			dir := filepath.Join(home, "models", name)
			if _, err := os.Stat(filepath.Join(dir, "config.json")); err != nil {
				t.Skipf("no %s", dir)
			}
			ran++
			if name != "glm-ocr" {
				cohRan++
			}
			tk, err := tokenizer.Load(filepath.Join(dir, "tokenizer.json"))
			if err != nil {
				t.Fatal(err)
			}
			ids, err := tk.Encode(string(readme), false)
			if err != nil || len(ids) < nPrompt {
				t.Fatalf("encode README: %d ids, %v", len(ids), err)
			}
			toks := append([]int(nil), ids[:nPrompt]...)
			mc, err := decoder.Load(dir, decoder.Options{Backend: "cpu", Quant: "int4"})
			if err != nil {
				t.Fatalf("load cpu: %v", err)
			}
			c := mc.NewCache(nPrompt + nCont)
			var ref [][]float32
			for i := 0; i < nPrompt+nCont; i++ {
				l, err := mc.ForwardForTest(toks[i], c)
				if err != nil {
					t.Fatal(err)
				}
				ref = append(ref, append([]float32(nil), l...))
				if i >= nPrompt-1 && len(toks) < nPrompt+nCont {
					toks = append(toks, argmax(l))
				}
			}
			mc.Close()
			measure := func() (mean, worst float64, real, ties int) {
				m, err := decoder.Load(dir, decoder.Options{Backend: "webgpu", Quant: "int4", ResidentContext: 512})
				if err != nil {
					t.Fatalf("load webgpu: %v", err)
				}
				defer m.Close()
				rd, ok := m.ResidentForwardForTest().(*residentDecoder)
				if !ok || !rd.rm.Pairwise {
					t.Fatalf("not resident with the pairwise twins: %s", m.ResidentDecline())
				}
				rd.Reset()
				worst = 1
				for i, tok := range toks {
					l, err := rd.Forward(m.EmbedResidentForTest(tok), i)
					if err != nil {
						t.Fatal(err)
					}
					var ab, aa, bb float64
					for j := range l {
						ab, aa, bb = ab+float64(l[j])*float64(ref[i][j]), aa+float64(l[j])*float64(l[j]), bb+float64(ref[i][j])*float64(ref[i][j])
					}
					cs := ab / math.Sqrt(aa*bb)
					if math.IsNaN(cs) { // NaN, or an all-zero row, is the worst reading
						cs = -2
					}
					mean, worst = mean+cs, math.Min(worst, cs)
					if cs < 0.95 {
						t.Logf("  %s position %d (%s): cosine %.6f", name, i, map[bool]string{true: "prompt", false: "continuation"}[i < nPrompt], cs)
					}
					if i >= nPrompt-1 && i < len(toks)-1 {
						if want, got := argmax(ref[i]), argmax(l); want != got {
							lo, hi := ref[i][0], ref[i][0]
							for _, v := range ref[i] {
								lo, hi = min(lo, v), max(hi, v)
							}
							if float64(ref[i][want]-ref[i][got])/float64(hi-lo) <= 0.03 {
								ties++
							} else {
								real++
							}
						}
					}
				}
				return mean / float64(len(toks)), worst, real, ties
			}
			mean, worst, real, ties := measure()
			t.Logf("PAIRWISE %s: %d positions, mean cos %.6f, worst %.6f; continuation flips %d near-tie, %d real", name, len(toks), mean, worst, ties, real)
			if !(mean >= 0.99 && worst >= 0.90 && real == 0) {
				t.Errorf("%s: mean %.6f (>= 0.99), worst %.6f (>= 0.90), %d non-tie continuation flips (0)", name, mean, worst, real)
			}
			kernels.PairwiseNeoXForTest = true
			t.Cleanup(func() { kernels.PairwiseNeoXForTest = false })
			nm, nw, _, _ := measure()
			kernels.PairwiseNeoXForTest = false
			t.Logf("NeoX     %s: mean cos %.6f, worst %.6f", name, nm, nw)
			if !(nm < 0.99 || nw < 0.90) {
				t.Errorf("%s: the gate is BLIND: with the NeoX kernels it reads mean %.6f, worst %.6f", name, nm, nw)
			}
		})
	}
}
