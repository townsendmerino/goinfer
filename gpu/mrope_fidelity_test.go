//go:build gpu && goinfer_testhooks

package gpu

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/fidelity"
	"github.com/townsendmerino/goinfer/multimodal"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// G-S12.4b-4a-5 (docs/tasks/task-multimodal-support-2026-10.md): which of the two routes an image turn can prefill by
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

// mropeFidelitySet is the fixed prompt set of a mode. "pilot" sizes the gate and shares no image with it; "gate" is the
// decision set; "more" is the set a parked verdict may be extended by, registered with the gate.
func mropeFidelitySet(mode string) []mropeFidelityPrompt {
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

// TestS12MRoPEFidelity runs one mode of the gate (GOINFER_S12_FIDELITY: plan, pilot, gate or more) and writes its
// records under GOINFER_S12_FIDELITY_OUT. "plan" loads no model: it builds the gate's prompts and prints their hash
// and sizes. GOINFER_S12_FIDELITY_DEFECT=1 plants the pass's "each row's positions from the next row" defect in the
// candidate, to show the verdict can go red. It grades by the verdict it prints; a run that completes exits 0.
func TestS12MRoPEFidelity(t *testing.T) {
	mode := os.Getenv("GOINFER_S12_FIDELITY")
	if mode == "" {
		t.Skip("set GOINFER_S12_FIDELITY to plan, pilot, gate or more")
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
		data, err := os.ReadFile(filepath.Join("..", "testdata", filepath.FromSlash(p.image)))
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
		for _, extra := range []string{"pilot", "more"} {
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
	if sc := dir + ".int4.e4h.webgpu.v16.giw"; func() bool { _, err := os.Stat(sc); return err == nil }() {
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
			prefillDefectForTest = 13
		}
		line.Candidate = arm(func() ([]float32, error) {
			l, _, err := mA.ResidentMRoPEPrefillForTest(ctx, rd, ids, feats[key], start, nImg, mrope)
			return l, err
		})
		prefillDefectForTest = 0
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
	crit := fidelity.Criteria{AgreeMarginPts: 1, KLMargin: 0.10, Alpha: 0.05}
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
