//go:build goinfer_testhooks

package decoder

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/internal/fidelity"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestKSO4_gptossFidelity is KS-O4's G-FID (docs/measurements/ks-o4-2026-10/README.md, which registers it): gpt-oss-20b on the CPU,
// the candidate (the batched acc64 attention with its sink term, the window layers on the KV ring) and the exact arm (today's
// attendSinksToken over append-forever rows), both at int4, scored position by position against a weight-only int8 reference on the
// exact arm's attention. Both arms are teacher-forced on the reference's greedy tokens. A measurement tool: it skips unless KSO4_MODEL
// is set.
//
//	KSO4_MODEL=<gpt-oss-20b .gguf> KSO4_MODE=pilot|gate [KSO4_N=12] KSO4_OUT=<dir> go test -run '^TestKSO4_gptossFidelity$' -v -timeout 4h ./decoder/
func TestKSO4_gptossFidelity(t *testing.T) {
	model := os.Getenv("KSO4_MODEL")
	if model == "" {
		t.Skip("set KSO4_MODEL, KSO4_MODE and KSO4_OUT")
	}
	mode, out := os.Getenv("KSO4_MODE"), os.Getenv("KSO4_OUT")
	if mode != "pilot" && mode != "gate" {
		t.Fatalf("KSO4_MODE %q: want pilot or gate", mode)
	}
	if err := os.MkdirAll(out, 0o755); err != nil {
		t.Fatal(err)
	}
	n := 12
	if v := os.Getenv("KSO4_N"); v != "" {
		var err error
		if n, err = strconv.Atoi(v); err != nil || n < 1 {
			t.Fatalf("KSO4_N %q", v)
		}
	}
	start := time.Now()
	say := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "ks-o4 [%6.0fs] %s\n", time.Since(start).Seconds(), fmt.Sprintf(format, a...))
	}

	tk, err := tokenizer.LoadGGUF(model)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	tmpl, err := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has})
	if err != nil {
		t.Fatalf("chat template: %v", err)
	}
	set := ksO4PromptSet(mode, n)
	var prompts [][]int
	for _, p := range set {
		ids, err := tk.EncodeSegments(tmpl.RenderSegments("", []chat.Turn{{Role: "user", Content: p.text}}), false)
		if err != nil {
			t.Fatal(err)
		}
		prompts = append(prompts, ids)
	}
	hash := fidelity.PromptSetHash(prompts)
	lens := map[string][]int{}
	for i, p := range set {
		lens[p.cell] = append(lens[p.cell], len(prompts[i]))
	}
	say("mode %s: %d prompts, prompt-set hash %s, lengths by cell %v", mode, len(set), hash, lens)

	// The reference: int8 weights, f32 activations, today's attention over append-forever rows.
	refLogits := make([][][]float32, len(set))
	refTokens := make([][]int, len(set))
	{
		m := ksO4Load(t, model, "int8", say)
		ksO4SetOrder(m, attnSinksF64)
		for i, p := range prompts {
			refLogits[i], refTokens[i] = ksO4Run(t, m, ksO4AppendForever(m, len(p)+ksO4Positions), p, nil)
			say("reference %d/%d (%s, %d tokens)", i+1, len(set), set[i].cell, len(p))
		}
		m.Close()
		runtime.GC()
	}

	// The two arms on one int4 load, in alternating order per prompt.
	m := ksO4Load(t, model, "int4", say)
	defer m.Close()
	type line struct {
		ID        string              `json:"id"`
		Cell      string              `json:"cell"`
		Candidate []fidelity.Position `json:"candidate"`
		Exact     []fidelity.Position `json:"exact"`
		Direct    []fidelity.Position `json:"direct"` // the candidate against the exact arm itself
	}
	cells := map[string]*fidelity.Cell{}
	var order []string
	var lines []line
	for i, p := range prompts {
		var exL, caL [][]float32
		runExact := func() {
			ksO4SetOrder(m, attnSinksF64)
			exL, _ = ksO4Run(t, m, ksO4AppendForever(m, len(p)+ksO4Positions), p, refTokens[i])
		}
		runCand := func() {
			ksO4SetOrder(m, attnBatchedAcc64)
			caL, _ = ksO4Run(t, m, m.NewCache(len(p)+ksO4Positions), p, refTokens[i])
		}
		if i%2 == 0 {
			runExact()
			runCand()
		} else {
			runCand()
			runExact()
		}
		l := line{ID: set[i].id, Cell: set[i].cell}
		for j := range refLogits[i] {
			l.Candidate = append(l.Candidate, ksO4Score(refLogits[i][j], caL[j]))
			l.Exact = append(l.Exact, ksO4Score(refLogits[i][j], exL[j]))
			l.Direct = append(l.Direct, ksO4Score(exL[j], caL[j]))
		}
		lines = append(lines, l)
		if cells[l.Cell] == nil {
			cells[l.Cell] = &fidelity.Cell{Name: l.Cell}
			order = append(order, l.Cell)
		}
		cells[l.Cell].Prompts = append(cells[l.Cell].Prompts, fidelity.Prompt{ID: l.ID, Candidate: l.Candidate, Exact: l.Exact})
		say("arms %d/%d (%s)", i+1, len(set), set[i].cell)
	}
	ksO4SetOrder(m, attnBatchedAcc64)

	raw, _ := json.MarshalIndent(lines, "", " ")
	if err := os.WriteFile(filepath.Join(out, mode+"-scores.json"), raw, 0o644); err != nil {
		t.Fatal(err)
	}
	var cs []fidelity.Cell
	for _, c := range order {
		cs = append(cs, *cells[c])
	}
	pf, err := os.Create(filepath.Join(out, mode+"-positions.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	if err := fidelity.WritePositions(pf, cs); err != nil {
		t.Fatal(err)
	}
	pf.Close()

	var report strings.Builder
	fmt.Fprintf(&report, "KS-O4 G-FID, mode %s, prompt-set hash %s, %d prompts\n", mode, hash, len(set))
	var dAgree, dPos int
	var dKL, dKLMax float64
	for _, l := range lines {
		for _, d := range l.Direct {
			dPos++
			if d.Agree {
				dAgree++
			}
			dKL += d.KL
			dKLMax = math.Max(dKLMax, d.KL)
		}
	}
	fmt.Fprintf(&report, "beside it, the candidate against the exact arm: top-1 agreement %d/%d (%.2f%%), mean KL(exact||candidate) %.3g, max %.3g\n",
		dAgree, dPos, 100*float64(dAgree)/float64(dPos), dKL/float64(dPos), dKLMax)
	crit := fidelity.Criteria{AgreeMarginPts: 1, KLMargin: 0.10, Alpha: 0.05}
	v, err := fidelity.NonInferior(cs, crit)
	if err != nil {
		t.Fatal(err)
	}
	show := func(name string, s fidelity.Summary) {
		fmt.Fprintf(&report, "%-7s prompts %d positions %d: agreement cand %.2f%% exact %.2f%% (diff %+.2f, LCB %+.2f pts); mean KL cand %.4g exact %.4g (ratio %.3f, UCB %.3f); hard flips cand %d exact %d\n",
			name, s.Prompts, s.Positions, s.AgreeCandPct, s.AgreeExactPct, s.AgreeDiffPts, s.AgreeDiffLCB, s.MeanKLCand, s.MeanKLExact, s.KLRatio, s.KLRatioUCB, s.FlipsCand, s.FlipsExact)
	}
	for _, c := range order {
		show(c, v.PerCell[c])
	}
	show("pooled", v.Pooled)
	verdict := "FAIL"
	switch {
	case v.Pass:
		verdict = "PASS"
	case v.FlipsPass && v.Pooled.AgreeDiffLCB > -2 && v.Pooled.KLRatioUCB <= 1.25:
		verdict = "AMBIGUOUS"
	}
	if mode == "pilot" {
		fmt.Fprintf(&report, "pilot: size N with `python3 scripts/power.py fidelity --from-positions %s` (registered: 10 <= N <= 24 a cell); the pilot grades nothing\n",
			filepath.Join(out, "pilot-positions.jsonl"))
	} else {
		fmt.Fprintf(&report, "VERDICT %s %s\n", verdict, strings.Join(v.Reasons, "; "))
	}
	if err := os.WriteFile(filepath.Join(out, mode+"-report.txt"), []byte(report.String()), 0o644); err != nil {
		t.Fatal(err)
	}
	fmt.Fprint(os.Stderr, report.String())
}

const ksO4Positions = 16

type ksO4Prompt struct{ id, cell, text string }

// ksO4PromptSet is the registered set (the 2026-10-25 10:20 amendment): short cells of a passage's first two sentences and long cells of
// two passages (about 100 and 300 tokens), each with an instruction; the pilot uses passages the gate set never does.
func ksO4PromptSet(mode string, n int) []ksO4Prompt {
	var set []ksO4Prompt
	add := func(cell, body string, idx []int, ins int) {
		set = append(set, ksO4Prompt{id: fmt.Sprintf("%s|%s|t%v|i%d", mode, cell, idx, ins), cell: cell,
			text: ksO4Instructions[ins] + "\n\n" + body})
	}
	short := func(texts []string, i int) string { return ksO4FirstSentences(texts[i], 2) }
	long := func(texts []string, i, j int) string { return texts[i] + "\n\n" + texts[j] }
	if mode == "pilot" {
		add("short", short(ksO4PilotTexts, 0), []int{0}, 0)
		add("short", short(ksO4PilotTexts, 1), []int{1}, 1)
		add("long", long(ksO4PilotTexts, 0, 1), []int{0, 1}, 2)
		add("long", long(ksO4PilotTexts, 1, 0), []int{1, 0}, 3)
		return set
	}
	nt := len(ksO4Texts)
	for k := range n {
		add("short", short(ksO4Texts, k%nt), []int{k % nt}, (k+k/nt)%len(ksO4Instructions))
	}
	for k := range n {
		i, j := k%nt, (k+1+k/nt)%nt
		add("long", long(ksO4Texts, i, j), []int{i, j}, (k+k/nt)%len(ksO4Instructions))
	}
	return set
}

// ksO4FirstSentences is text's first n sentences (split after ". ").
func ksO4FirstSentences(text string, n int) string {
	parts := strings.SplitAfter(text, ". ")
	if len(parts) <= n {
		return text
	}
	return strings.TrimSpace(strings.Join(parts[:n], ""))
}

// ksO4Load loads the checkpoint on the CPU at quant, straight into the heap (no sidecar written).
func ksO4Load(t *testing.T, model, quant string, say func(string, ...any)) *Model {
	t.Helper()
	t0 := time.Now()
	m, err := Load(model, Options{Backend: "cpu", Quant: quant, DirectLoad: true})
	if err != nil {
		t.Fatalf("load %s at %s: %v", model, quant, err)
	}
	if m.w.arch.gptoss == nil {
		t.Fatalf("%s is not gpt-oss", model)
	}
	say("loaded %s at %s in %.0fs", filepath.Base(model), quant, time.Since(t0).Seconds())
	return m
}

// ksO4SetOrder sets the attention order of every layer of m's program: attnSinksF64 is today's gpt-oss path, attnBatchedAcc64 KS-O4's.
func ksO4SetOrder(m *Model, o attnOrder) {
	for _, ops := range m.program().layers {
		for i := range ops {
			if ops[i].kind == opAttention {
				ops[i].attn.Order = o
			}
		}
	}
}

// ksO4AppendForever is the cache attnSinksF64 reads: every layer append-forever, no ring (attendSinksToken reads Keys/Vals).
func ksO4AppendForever(m *Model, capHint int) *KVCache {
	a := m.w.arch
	c := NewKVCache(a.NumLayers, a.NumKVHeads, a.HeadDim, a.SlidingWindow, capHint, a.hasNoAttentionKVAt)
	c.scr = newDecodeScratch(a)
	return c
}

// ksO4Run feeds prompt, then ksO4Positions-1 more tokens: forced ones when given, else its own greedy picks. It returns the logits of
// the ksO4Positions positions (the one after the prompt, then one after each fed token) and the tokens fed after the prompt.
func ksO4Run(t *testing.T, m *Model, cache *KVCache, prompt, forced []int) ([][]float32, []int) {
	t.Helper()
	var logits [][]float32
	var toks []int
	feed := func(id int) []float32 {
		lg, err := m.forward(id, cache)
		if err != nil {
			t.Fatal(err)
		}
		return append([]float32(nil), lg...)
	}
	var last []float32
	for _, id := range prompt {
		last = feed(id)
	}
	logits = append(logits, last)
	for i := range ksO4Positions - 1 {
		next := argmaxF32(last)
		if forced != nil {
			next = forced[i]
		}
		toks = append(toks, next)
		last = feed(next)
		logits = append(logits, last)
	}
	return logits, toks
}

// ksO4Score is one position of arm against ref: top-1 agreement, a hard flip by the near-tie rule, and KL(ref || arm) in float64.
func ksO4Score(ref, arm []float32) fidelity.Position {
	agree, _, hard := NearTieArgmaxForTest(ref, arm)
	lse := func(x []float32) float64 {
		mx := math.Inf(-1)
		for _, v := range x {
			mx = math.Max(mx, float64(v))
		}
		var s float64
		for _, v := range x {
			s += math.Exp(float64(v) - mx)
		}
		return mx + math.Log(s)
	}
	lr, la := lse(ref), lse(arm)
	var kl float64
	for i := range ref {
		lp := float64(ref[i]) - lr
		kl += math.Exp(lp) * (lp - (float64(arm[i]) - la))
	}
	return fidelity.Position{Agree: agree, HardFlip: hard, KL: math.Max(kl, 0)}
}

var ksO4Instructions = []string{
	"Summarize the following text in two sentences.",
	"List three facts stated in the following text.",
	"What is the main point of the following text? Answer in one sentence.",
	"Rewrite the following text for a ten-year-old reader.",
	"Write one question a careful reader would ask about the following text.",
}

// ksO4Texts are the gate's passages, the ones gpu/attn_fidelity_test.go's gate uses; the pilot's share none of them.
var ksO4Texts = []string{
	"The harbor at dawn was already loud. Cranes swung containers from the decks of two freighters onto flatbed trucks, and a pilot boat cut across the channel toward a tanker waiting beyond the breakwater. The port authority had dredged the main berth the previous winter, so ships drawing fourteen meters could now tie up at any tide. Customs officers walked the quay with tablets, checking seals against manifests before the gates opened to the morning traffic.",
	"A sourdough starter is a culture of wild yeast and bacteria kept alive with regular feedings of flour and water. Bakers discard part of it each day and replace what they removed, which keeps the acidity in check. A starter fed with whole rye ferments faster than one fed with white wheat. At room temperature it peaks about six hours after a feeding; in a refrigerator the same rise can take two days, which is why many home bakers store theirs cold between loaves.",
	"The committee met on the third floor to review the transit budget. Ridership on the eastern line had fallen eleven percent in two years, while the cost of maintaining its aging signals had doubled. One member proposed replacing weekend rail service with buses. Another argued that the line's decline followed the closure of a single factory and would recover when the new hospital opened. The vote was postponed until the engineering report on the signals arrived.",
	"Granite forms when magma cools slowly deep underground, which gives its crystals time to grow large enough to see. The pink grains are usually potassium feldspar, the white ones plagioclase, the glassy gray ones quartz and the black flecks biotite or hornblende. Because it resists weathering, granite often stands above softer rock as domes and ridges. Quarry workers once split it by drilling a line of holes and driving in steel wedges, a method still used for monument stone.",
	"When the library moved its catalog online, the reference desk expected fewer questions. The opposite happened. Patrons who had never used the card catalog now searched from home and arrived with longer lists, and the desk spent its days locating volumes shelved in the wrong place decades earlier. The head librarian hired two students to walk the stacks with a scanner. In one summer they found nine hundred misplaced books and a box of unprocessed letters from the eighteen nineties.",
	"A violin's sound begins with the bow, whose rosined hair grips the string and releases it hundreds of times a second. The bridge carries that vibration to the spruce top, and a thin post inside the body passes it to the maple back. Makers tune the two plates by tapping them and shaving wood where the pitch is too high. Small changes matter: a bridge moved one millimeter toward the fingerboard makes the instrument noticeably darker in tone.",
	"The expedition reached the northern camp nine days late. Meltwater had cut the track in three places, and the sledges had to be unloaded and carried across each gap. The cook reported that the flour was sound but a third of the dried fruit had spoiled. The surveyor wanted to press on to the ridge before the weather turned; the doctor wanted two days of rest for the men with frostbitten feet. The leader chose one day, and set the two strongest to repair the runners.",
	"Compilers translate programs in stages. A lexer breaks the source text into tokens, and a parser arranges those tokens into a tree that reflects the grammar. Later passes check types, then rewrite the tree into a simpler intermediate form on which optimizations run. The last stage chooses machine instructions and assigns values to registers. Because each stage has a narrow job, a compiler for a new processor can reuse everything except that final stage.",
}

var ksO4PilotTexts = []string{
	"The village market opens before sunrise on Thursdays. Farmers from the upper valley bring cheese, honey and late apples, and a fishmonger drives in from the coast with ice packed around the night's catch. By nine the square is crowded, and by noon the stalls come down so the school buses can turn. The mayor has twice proposed moving the market to the car park by the river, and the stallholders have twice refused.",
	"Tide tables are computed from the positions of the moon and sun and from decades of local measurements. A harbor at the head of a long bay can see its high water an hour after a harbor at the mouth. Spring tides, the largest, come just after the new and full moon; neap tides, the smallest, a week later. Sailors add a correction for wind and air pressure, since a deep storm can raise the sea half a meter above the prediction.",
}
