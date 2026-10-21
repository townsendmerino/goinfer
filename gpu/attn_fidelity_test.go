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

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/fidelity"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// G-S12.4c-6 (docs/tasks/task-multimodal-support-2026-10.md): which attention kernel leaves a wide-head model nearer
// the model itself. Gemma 3 4B decodes on the WebGPU resident at int4 twice, once built with the per-dimension kernel a
// head over 128 used to get (exact) and once with the key-split kernel it gets now (candidate), prefill and decode
// alike. Both are scored, position by position, against the float32 CPU forward of the same checkpoint on the same
// prompt. A prompt's positions are the one after it and the ones after each of the reference's own greedy tokens,
// forced on both arms.

const attnFidelityPositions = 32

var attnFidelityInstructions = []string{
	"Summarize the following text in two sentences.",
	"List three facts stated in the following text.",
	"What is the main point of the following text? Answer in one sentence.",
	"Rewrite the following text for a ten-year-old reader.",
	"Write one question a careful reader would ask about the following text.",
}

// attnFidelityTexts are the gate's passages; the pilot's are attnFidelityPilotTexts and share none of them.
var attnFidelityTexts = []string{
	"The harbor at dawn was already loud. Cranes swung containers from the decks of two freighters onto flatbed trucks, and a pilot boat cut across the channel toward a tanker waiting beyond the breakwater. The port authority had dredged the main berth the previous winter, so ships drawing fourteen meters could now tie up at any tide. Customs officers walked the quay with tablets, checking seals against manifests before the gates opened to the morning traffic.",
	"A sourdough starter is a culture of wild yeast and bacteria kept alive with regular feedings of flour and water. Bakers discard part of it each day and replace what they removed, which keeps the acidity in check. A starter fed with whole rye ferments faster than one fed with white wheat. At room temperature it peaks about six hours after a feeding; in a refrigerator the same rise can take two days, which is why many home bakers store theirs cold between loaves.",
	"The committee met on the third floor to review the transit budget. Ridership on the eastern line had fallen eleven percent in two years, while the cost of maintaining its aging signals had doubled. One member proposed replacing weekend rail service with buses. Another argued that the line's decline followed the closure of a single factory and would recover when the new hospital opened. The vote was postponed until the engineering report on the signals arrived.",
	"Granite forms when magma cools slowly deep underground, which gives its crystals time to grow large enough to see. The pink grains are usually potassium feldspar, the white ones plagioclase, the glassy gray ones quartz and the black flecks biotite or hornblende. Because it resists weathering, granite often stands above softer rock as domes and ridges. Quarry workers once split it by drilling a line of holes and driving in steel wedges, a method still used for monument stone.",
	"When the library moved its catalog online, the reference desk expected fewer questions. The opposite happened. Patrons who had never used the card catalog now searched from home and arrived with longer lists, and the desk spent its days locating volumes shelved in the wrong place decades earlier. The head librarian hired two students to walk the stacks with a scanner. In one summer they found nine hundred misplaced books and a box of unprocessed letters from the eighteen nineties.",
	"A violin's sound begins with the bow, whose rosined hair grips the string and releases it hundreds of times a second. The bridge carries that vibration to the spruce top, and a thin post inside the body passes it to the maple back. Makers tune the two plates by tapping them and shaving wood where the pitch is too high. Small changes matter: a bridge moved one millimeter toward the fingerboard makes the instrument noticeably darker in tone.",
	"The expedition reached the northern camp nine days late. Meltwater had cut the track in three places, and the sledges had to be unloaded and carried across each gap. The cook reported that the flour was sound but a third of the dried fruit had spoiled. The surveyor wanted to press on to the ridge before the weather turned; the doctor wanted two days of rest for the men with frostbitten feet. The leader chose one day, and set the two strongest to repair the runners.",
	"Compilers translate programs in stages. A lexer breaks the source text into tokens, and a parser arranges those tokens into a tree that reflects the grammar. Later passes check types, then rewrite the tree into a simpler intermediate form on which optimizations run. The last stage chooses machine instructions and assigns values to registers. Because each stage has a narrow job, a compiler for a new processor can reuse everything except that final stage.",
}

var attnFidelityPilotTexts = []string{
	"The village market opens before sunrise on Thursdays. Farmers from the upper valley bring cheese, honey and late apples, and a fishmonger drives in from the coast with ice packed around the night's catch. By nine the square is crowded, and by noon the stalls come down so the school buses can turn. The mayor has twice proposed moving the market to the car park by the river, and the stallholders have twice refused.",
	"Tide tables are computed from the positions of the moon and sun and from decades of local measurements. A harbor at the head of a long bay can see its high water an hour after a harbor at the mouth. Spring tides, the largest, come just after the new and full moon; neap tides, the smallest, a week later. Sailors add a correction for wind and air pressure, since a deep storm can raise the sea half a meter above the prediction.",
}

// attnFidelityPrompt names one prompt of a set: its cell (the length band), the passages it joins, in order, and the
// instruction's index.
type attnFidelityPrompt struct {
	cell        string
	texts       []int
	instruction int
	pilot       bool
}

func (p attnFidelityPrompt) id() string {
	kind := "gate"
	if p.pilot {
		kind = "pilot"
	}
	return fmt.Sprintf("%s|%s|t%v|i%d", kind, p.cell, p.texts, p.instruction)
}

func (p attnFidelityPrompt) text() string {
	src := attnFidelityTexts
	if p.pilot {
		src = attnFidelityPilotTexts
	}
	var parts []string
	for _, t := range p.texts {
		parts = append(parts, src[t])
	}
	return "<bos><start_of_turn>user\n" + attnFidelityInstructions[p.instruction] + "\n\n" + strings.Join(parts, "\n\n") + "<end_of_turn>\n<start_of_turn>model\n"
}

// attnFidelitySet is the fixed prompt set of a mode: "pilot" (eight prompts on the pilot's passages, used with the
// planted defect to show the verdict can go red) or "gate", the decision set: 40 short prompts (one passage), 16 medium
// (four passages) and 8 long ones (twelve passages, past the model's 1,024-position sliding window).
func attnFidelitySet(mode string) []attnFidelityPrompt {
	var ps []attnFidelityPrompt
	if mode == "pilot" {
		for t := range attnFidelityPilotTexts {
			for i := 0; i < 3; i++ {
				ps = append(ps, attnFidelityPrompt{cell: "short", texts: []int{t}, instruction: i, pilot: true})
			}
		}
		return append(ps, attnFidelityPrompt{cell: "medium", texts: []int{0, 1, 0, 1}, instruction: 3, pilot: true},
			attnFidelityPrompt{cell: "medium", texts: []int{1, 0, 1, 0}, instruction: 4, pilot: true})
	}
	n := len(attnFidelityTexts)
	for t := 0; t < n; t++ {
		for i := range attnFidelityInstructions {
			ps = append(ps, attnFidelityPrompt{cell: "short", texts: []int{t}, instruction: i})
		}
	}
	rot := func(start, count int) []int {
		out := make([]int, count)
		for k := range out {
			out[k] = (start + k) % n
		}
		return out
	}
	for k := 0; k < 16; k++ {
		ps = append(ps, attnFidelityPrompt{cell: "medium", texts: rot(k%n, 4), instruction: (k + k/n) % len(attnFidelityInstructions)})
	}
	for k := 0; k < 8; k++ {
		ps = append(ps, attnFidelityPrompt{cell: "long", texts: rot(k, 12), instruction: k % len(attnFidelityInstructions)})
	}
	return ps
}

// attnFidelityLine is one finished prompt in the live log, which a restarted run reads back and skips.
type attnFidelityLine struct {
	ID        string              `json:"id"`
	Cell      string              `json:"cell"`
	Rows      int                 `json:"rows"`
	Tokens    []int               `json:"reference_tokens"`
	Differ    int                 `json:"positions_where_the_arms_differ"`
	Candidate []fidelity.Position `json:"candidate"`
	Exact     []fidelity.Position `json:"exact"`
}

// TestS12AttnKernelFidelity runs one mode of the gate (GOINFER_S12_ATTN_FIDELITY: plan, pilot or gate) and writes its
// records under GOINFER_S12_ATTN_FIDELITY_OUT. "plan" loads no model: it builds the prompts and prints their hash and
// sizes. GOINFER_S12_ATTN_FIDELITY_DEFECT=1 plants a defect in the candidate (each decode step is fed the token before
// the one it should be), to show the verdict can go red. It grades by the verdict it prints; a run that completes exits 0.
func TestS12AttnKernelFidelity(t *testing.T) {
	mode := os.Getenv("GOINFER_S12_ATTN_FIDELITY")
	if mode == "" {
		t.Skip("set GOINFER_S12_ATTN_FIDELITY to plan, pilot or gate")
	}
	home, _ := os.UserHomeDir()
	dir := os.Getenv("GOINFER_GEMMA3_DIR")
	if dir == "" {
		dir = filepath.Join(home, "models", "gemma-3-4b-it")
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
	bos, okB := tk.TokenID("<bos>")
	sot, okS := tk.TokenID("<start_of_turn>")
	if !okB || !okS {
		t.Fatal("the tokenizer has no <bos> or <start_of_turn> token")
	}
	began := time.Now()
	say := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "%s [%5.0fs] %s\n", time.Now().Format("15:04:05"), time.Since(began).Seconds(), fmt.Sprintf(format, a...))
	}
	encode := func(set []attnFidelityPrompt) ([][]int, map[string]int) {
		all := make([][]int, len(set))
		rows := map[string]int{}
		for i, p := range set {
			ids, err := tk.Encode(p.text(), false)
			if err != nil {
				t.Fatal(err)
			}
			if len(ids) < 8 || ids[0] != bos || ids[1] != sot {
				t.Fatalf("%s: the prompt does not start <bos><start_of_turn> as tokens: %v", p.id(), ids[:min(len(ids), 4)])
			}
			all[i] = ids
			rows[p.cell] += len(ids)
		}
		return all, rows
	}
	set := attnFidelitySet(mode)
	if mode == "plan" {
		set = attnFidelitySet("gate")
	}
	all, rows := encode(set)
	say("mode %s: %d prompts, prompt-set hash %s; rows by cell %v", mode, len(set), fidelity.PromptSetHash(all), rows)
	if mode == "plan" {
		pa, pr := encode(attnFidelitySet("pilot"))
		say("mode pilot: %d prompts, prompt-set hash %s; rows by cell %v", len(pa), fidelity.PromptSetHash(pa), pr)
		longest := 0
		for _, ids := range all {
			longest = max(longest, len(ids))
		}
		say("the longest prompt is %d rows", longest)
		return
	}
	if mode != "pilot" && mode != "gate" {
		t.Fatalf("GOINFER_S12_ATTN_FIDELITY=%q: want plan, pilot or gate", mode)
	}
	if _, err := New(); err != nil {
		t.Fatalf("no WebGPU adapter: %v", err)
	}
	out := os.Getenv("GOINFER_S12_ATTN_FIDELITY_OUT")
	if out == "" {
		t.Fatal("set GOINFER_S12_ATTN_FIDELITY_OUT to the directory the records go in")
	}
	if err := os.MkdirAll(out, 0o755); err != nil {
		t.Fatal(err)
	}
	livePath := filepath.Join(out, "positions-live.jsonl")
	done := map[string]attnFidelityLine{}
	if f, err := os.Open(livePath); err == nil {
		sc := bufio.NewScanner(f)
		sc.Buffer(make([]byte, 1<<20), 1<<26)
		for sc.Scan() {
			var l attnFidelityLine
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

	say("loading the float32 reference (CPU) and two int4 residents (WebGPU), one a kernel")
	mR, err := decoder.Load(dir, decoder.Options{Backend: "cpu", Quant: "f32", ExactPrefill: true})
	if err != nil {
		t.Fatalf("the float32 reference: %v", err)
	}
	defer mR.Close()
	src := dir // the directory's WebGPU sidecar when it is beside it: the same int4 weights, mapped
	if sc := dir + fmt.Sprintf(".int4.e4h.webgpu.v%d.giw", decoder.GIWFormatVersion); func() bool { _, err := os.Stat(sc); return err == nil }() {
		src = sc
	}
	// A resident's decode kernels are chosen when it is built, so each arm is its own resident, built under its switch.
	keysWas := attnKeysDisabled
	defer func() { attnKeysDisabled = keysWas }()
	build := func(disabled bool) (*decoder.Model, *residentDecoder) {
		attnKeysDisabled = disabled
		m, err := decoder.Load(src, decoder.Options{Backend: "webgpu", Quant: "int4", ResidentContext: 2048})
		if err != nil {
			t.Fatalf("the int4 resident (key-split off: %v): %v", disabled, err)
		}
		rd, ok := m.ResidentForwardForTest().(*residentDecoder)
		if !ok {
			t.Fatalf("the model is not resident on WebGPU (key-split off: %v): %s", disabled, m.ResidentDecline())
		}
		return m, rd
	}
	mOld, rdOld := build(true)
	defer mOld.Close()
	mNew, rdNew := build(false)
	defer mNew.Close()
	if _, _, _, _, hd, _, _ := mNew.Dims(); hd <= attnWG {
		t.Fatalf("this model's heads are %d wide: the two arms would run the same kernel", hd)
	}
	defect := os.Getenv("GOINFER_S12_ATTN_FIDELITY_DEFECT") == "1"
	say("backend %q, decode path %s; planted defect in the candidate: %v; %d of %d prompts already in %s",
		rdNew.c.Backend(), mNew.DecodePath(), defect, len(done), len(set), livePath)

	ctx := context.Background()
	for i, p := range set {
		if _, ok := done[p.id()]; ok {
			continue
		}
		ids := all[i]
		n := len(ids)
		// The reference, and the tokens it forces on both arms.
		cR := mR.NewCache(n + attnFidelityPositions + 1)
		l, err := mR.PrefillLogitsForTest(ctx, ids, cR)
		if err != nil {
			t.Fatalf("%s: reference prefill: %v", p.id(), err)
		}
		ref := [][]float32{append([]float32(nil), l...)}
		toks := []int{argmax(l)}
		for k := 1; k < attnFidelityPositions; k++ {
			if l, err = mR.ForwardForTest(toks[k-1], cR); err != nil {
				t.Fatalf("%s: reference step %d: %v", p.id(), k, err)
			}
			ref = append(ref, append([]float32(nil), l...))
			toks = append(toks, argmax(l))
		}
		arm := func(m *decoder.Model, rd *residentDecoder, disabled, stale bool) ([]fidelity.Position, [][]float32) {
			attnKeysDisabled = disabled // the batched pass reads the switch when it runs
			rd.Reset()
			embs := make([][]float32, n)
			for r, id := range ids {
				embs[r] = append([]float32(nil), m.EmbedResidentForTest(id)...)
			}
			l, err := rd.PrefillLast(ctx, embs, 0)
			if err != nil {
				t.Fatalf("%s: resident prefill (key-split off: %v): %v", p.id(), disabled, err)
			}
			raw := [][]float32{append([]float32(nil), l...)}
			pos := []fidelity.Position{mropeScore(ref[0], l)}
			for k := 1; k < attnFidelityPositions; k++ {
				tok := toks[k-1]
				if stale { // the planted defect: the token one step back
					tok = ids[n-1]
					if k > 1 {
						tok = toks[k-2]
					}
				}
				if l, err = rd.Forward(m.EmbedResidentForTest(tok), n+k-1); err != nil {
					t.Fatalf("%s: resident step %d (key-split off: %v): %v", p.id(), k, disabled, err)
				}
				raw = append(raw, append([]float32(nil), l...))
				pos = append(pos, mropeScore(ref[k], l))
			}
			return pos, raw
		}
		line := attnFidelityLine{ID: p.id(), Cell: p.cell, Rows: n, Tokens: toks}
		var rawOld, rawNew [][]float32
		line.Exact, rawOld = arm(mOld, rdOld, true, false)
		line.Candidate, rawNew = arm(mNew, rdNew, false, defect)
		for k := range rawOld {
			for v := range rawOld[k] {
				if math.Float32bits(rawOld[k][v]) != math.Float32bits(rawNew[k][v]) {
					line.Differ++
					break
				}
			}
		}
		b, _ := json.Marshal(line)
		if _, err := live.Write(append(b, '\n')); err != nil {
			t.Fatal(err)
		}
		done[p.id()] = line
		tally := func(ps []fidelity.Position) (agree int, kl float64) {
			for _, q := range ps {
				agree += b2i(q.Agree)
				kl += q.KL
			}
			return agree, kl / float64(len(ps))
		}
		ea, ek := tally(line.Exact)
		ca, ck := tally(line.Candidate)
		say("prompt %d of %d %s: %d rows; agreement exact %d/%d candidate %d/%d; mean KL exact %.4f candidate %.4f; the arms' logits differ at %d of %d positions",
			i+1, len(set), p.id(), n, ea, attnFidelityPositions, ca, attnFidelityPositions, ek, ck, line.Differ, attnFidelityPositions)
	}

	cells := map[string]*fidelity.Cell{}
	var order []string
	differ, positions := 0, 0
	for _, p := range set {
		l := done[p.id()]
		if cells[p.cell] == nil {
			cells[p.cell] = &fidelity.Cell{Name: p.cell}
			order = append(order, p.cell)
		}
		cells[p.cell].Prompts = append(cells[p.cell].Prompts, fidelity.Prompt{ID: l.ID, Candidate: l.Candidate, Exact: l.Exact})
		differ += l.Differ
		positions += len(l.Exact)
	}
	if differ == 0 {
		t.Fatalf("the two arms wrote the same logits at all %d positions: they ran one kernel, and the gate compared nothing", positions)
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
	fmt.Fprintf(&report, "G-S12.4c-6 mode %s, backend %s, planted defect %v, prompt-set hash %s\n", mode, rdNew.c.Backend(), defect, fidelity.PromptSetHash(all))
	fmt.Fprintf(&report, "exact: the per-dimension kernel; candidate: the key-split kernel; the arms' logits differ at %d of %d positions\n", differ, positions)
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
