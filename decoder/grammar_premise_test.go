//go:build realckpt

// Phase 2's PREMISE TEST. The decision rule is pre-registered in
// docs/measurements/phase2-grammar-premise-PREREGISTERED.md and was committed before this file
// produced a number; read it first, and do not restate its thresholds here (a checklist that
// restates a value maintained elsewhere is a second copy that drifts).
//
// The question: does a MODEL drafter (DFlash) accept BETTER when the target is grammar-masked?
// Not answered by docs/spec/01-grammar-fused.md, which measured the grammar automaton used AS
// the drafter and says so in its own words.
//
// CPU is the right instrument, per dflash_accept_test.go's header: acceptance is a property of
// the drafter's distribution against the target's -- numerics -- and does not depend on the
// backend. Wall-clock does NOT transfer and is not measured here.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags realckpt ./decoder/ -run TestGrammarPremise -v -timeout 2h
package decoder

import (
	"math"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/constrain"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// The constrained-JSON suite. Each prompt asks for an object the schema then PINS, so the
// grammar is doing real work at most positions rather than decorating free text.
var grammarPremiseSuite = []struct {
	name, prompt, schema string
}{
	{
		"weather",
		"Give me the current weather for Paris in celsius, as JSON.",
		`{"type":"object","properties":{"location":{"type":"string"},"unit":{"enum":["celsius","fahrenheit"]},"temperature":{"type":"integer"}},"required":["location","unit","temperature"],"additionalProperties":false}`,
	},
	{
		"person",
		"Describe a fictional person as JSON with their name, age and whether they are active.",
		`{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"},"active":{"type":"boolean"}},"required":["name","age","active"],"additionalProperties":false}`,
	},
	{
		"book",
		"Return a JSON object describing a book: its title, author and year of publication.",
		`{"type":"object","properties":{"title":{"type":"string"},"author":{"type":"string"},"year":{"type":"integer"}},"required":["title","author","year"],"additionalProperties":false}`,
	},
	{
		"config",
		"Return a JSON config with a host string, a port integer, and a tls boolean.",
		`{"type":"object","properties":{"host":{"type":"string"},"port":{"type":"integer"},"tls":{"type":"boolean"}},"required":["host","port","tls"],"additionalProperties":false}`,
	},
	{
		"order",
		"Return a JSON order record with an id integer, a customer string and a total integer.",
		`{"type":"object","properties":{"id":{"type":"integer"},"customer":{"type":"string"},"total":{"type":"integer"}},"required":["id","customer","total"],"additionalProperties":false}`,
	},
	{
		"measurement",
		"Return a JSON measurement with a sensor string, a value integer and an ok boolean.",
		`{"type":"object","properties":{"sensor":{"type":"string"},"value":{"type":"integer"},"ok":{"type":"boolean"}},"required":["sensor","value","ok"],"additionalProperties":false}`,
	},
}

// argmaxMasked applies the grammar state g's mask to a COPY of logits and returns the argmax.
// A copy because the caller's logits are reused across arms and must not be mutated.
func argmaxMasked(mk *constrain.Masker, g constrain.Grammar, logits []float32) int {
	buf := make([]float32, len(logits))
	copy(buf, logits)
	mk.MaskAt(g, buf)
	return argmax(buf)
}

type premiseArm struct {
	maskTarget, maskDrafter bool
}

type premiseResult struct {
	rounds, generated int
	tokPerRound       float64
}

// runPremiseArm is dflashRun's loop with the mask applied at the two points that matter. It is
// a deliberate copy rather than a refactor of dflashRun: that function is a load-bearing gate
// for kill-gate 2 and threading two optional masks through it would change the shipped
// measurement path to serve an experiment.
func runPremiseArm(t *testing.T, m *Model, d *DFlashDrafter, tk *tokenizer.Tokenizer,
	prompt, schema string, maxNew int, arm premiseArm) premiseResult {
	t.Helper()

	var mk *constrain.Masker
	if arm.maskTarget || arm.maskDrafter {
		gr, err := constrain.JSONSchema([]byte(schema))
		if err != nil {
			t.Fatalf("JSONSchema: %v", err)
		}
		mk = constrain.NewMasker(gr, constrain.TokenBytes(m.w.arch.VocabSize, tk.TokenText), m.eosIDs)
	}

	turns := []chat.Turn{{Role: "user", Content: prompt}}
	tmpl, err := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has})
	if err != nil {
		t.Fatalf("chat.Detect: %v", err)
	}
	ids, err := tk.EncodeSegments(tmpl.RenderSegments("", turns), false)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if _, ok := tk.TokenID("<think>"); ok {
		ids = append(ids, noThinkSuffix(t, tk)...)
	}

	B := d.BlockSize()
	vw := verifyWidth()
	cache := m.NewCache(len(ids) + maxNew + B + 2)
	layers := d.TargetLayerIDs()
	var ctxCat [][]float32
	var logits []float32
	feed := func(id int) {
		lg, hidden, e := m.ForwardCapture(id, cache, layers)
		if e != nil {
			t.Fatalf("ForwardCapture: %v", e)
		}
		row := make([]float32, 0, len(hidden)*d.hidden)
		for _, h := range hidden {
			row = append(row, h...)
		}
		ctxCat = append(ctxCat, row)
		logits = lg
	}
	for _, id := range ids {
		feed(id)
	}
	eos := map[int]bool{}
	for _, e := range m.w.Cfg.EOSIDs() {
		eos[e] = true
	}

	pick := func(lg []float32) int {
		if arm.maskTarget && mk != nil {
			return argmaxMasked(mk, mk.GrammarClone(), lg)
		}
		return argmax(lg)
	}

	var res premiseResult
	anchor := pick(logits)
	generated := 1
	for generated < maxNew {
		fused, e := d.FuseContext(m.be, ctxCat)
		if e != nil {
			t.Fatalf("FuseContext: %v", e)
		}
		blk := make([]int, B)
		for i := range blk {
			blk[i] = d.MaskTokenID()
		}
		blk[0] = anchor
		trunk, e := d.DraftBlock(m.be, fused, m.DrafterEmbedBlock(blk))
		if e != nil {
			t.Fatalf("DraftBlock: %v", e)
		}
		// The drafter's proposals. Under maskDrafter the grammar is rolled forward over the
		// drafted prefix with a CLONE, so each proposed position is masked by the state that
		// position would actually be in -- the mask is stateful and a single snapshot would
		// mask position k+1 with position 0's legal set.
		drafted := make([]int, 0, B-1)
		var dg constrain.Grammar
		if arm.maskDrafter && mk != nil {
			dg = mk.GrammarClone()
			dg.Commit(mk.TokenBytes(anchor))
		}
		for _, h := range trunk[1:] {
			lg := m.DrafterHeadLogits(h)
			var id int
			if arm.maskDrafter && mk != nil {
				id = argmaxMasked(mk, dg, lg)
				dg.Commit(mk.TokenBytes(id))
			} else {
				id = argmax(lg)
			}
			drafted = append(drafted, id)
		}
		if vw > 0 && len(drafted) > vw-1 {
			drafted = drafted[:vw-1]
		}

		mark := cache.Pos()
		markCtx := len(ctxCat)
		feed(anchor)
		accepted := 0
		next := pick(logits)
		for i, tok := range drafted {
			if tok != next {
				break
			}
			feed(tok)
			accepted = i + 1
			next = pick(logits)
		}
		cache.TruncateTo(mark + 1 + accepted)
		ctxCat = ctxCat[:markCtx+1+accepted]

		// Advance the REAL grammar over exactly what was committed.
		if mk != nil {
			mk.Commit(anchor)
			for _, tok := range drafted[:accepted] {
				mk.Commit(tok)
			}
		}
		res.rounds++
		generated += accepted + 1
		anchor = next
		if eos[anchor] {
			break
		}
		if mk != nil && mk.CanEnd() && accepted == 0 {
			break // the object closed; further rounds would measure trailing filler
		}
	}
	res.generated = generated
	if res.rounds > 0 {
		res.tokPerRound = float64(generated) / float64(res.rounds)
	}
	return res
}

func TestGrammarPremise_constrainedJSON(t *testing.T) {
	requireHeavyModel(t)
	target := assetPath(t, "GOINFER_QWEN3_4B")
	ddir := assetPath(t, "GOINFER_DFLASH_F32")
	m, err := Load(target, Options{Quant: "int8"})
	if err != nil {
		t.Fatalf("Load target: %v", err)
	}
	defer m.Close()
	d, err := LoadDFlashDrafter(ddir)
	if err != nil {
		t.Fatalf("LoadDFlashDrafter: %v", err)
	}
	defer d.Close()
	tk, err := tokenizer.Load(target)
	if err != nil {
		t.Skipf("tokenizer: %v", err)
	}
	const maxNew = 64

	var b, a1, a2 []float64
	for _, c := range grammarPremiseSuite {
		rb := runPremiseArm(t, m, d, tk, c.prompt, c.schema, maxNew, premiseArm{})
		r1 := runPremiseArm(t, m, d, tk, c.prompt, c.schema, maxNew, premiseArm{maskTarget: true})
		r2 := runPremiseArm(t, m, d, tk, c.prompt, c.schema, maxNew, premiseArm{maskTarget: true, maskDrafter: true})
		b = append(b, rb.tokPerRound)
		a1 = append(a1, r1.tokPerRound)
		a2 = append(a2, r2.tokPerRound)
		t.Logf("%-12s B(unconstrained) %.2f | A1(target-masked) %.2f | A2(both-masked) %.2f  tok/round",
			c.name, rb.tokPerRound, r1.tokPerRound, r2.tokPerRound)
	}

	paired := func(x, base []float64) (delta float64, wins, n int) {
		for i := range x {
			if base[i] > 0 {
				delta += (x[i]/base[i] - 1) * 100
				if x[i] > base[i] {
					wins++
				}
				n++
			}
		}
		if n > 0 {
			delta /= float64(n)
		}
		return
	}
	meanOf := func(xs []float64) float64 {
		s := 0.0
		for _, x := range xs {
			s += x
		}
		return s / math.Max(1, float64(len(xs)))
	}

	d2, w2, n2 := paired(a2, b)
	d1, w1, _ := paired(a1, b)
	t.Logf("=== PREMISE (rule: docs/measurements/phase2-grammar-premise-PREREGISTERED.md) ===")
	t.Logf("   B  unconstrained    mean %.2f tok/round", meanOf(b))
	t.Logf("   A1 target-masked    mean %.2f tok/round   paired %+.1f%%  wins %d/%d  [DIAGNOSTIC]",
		meanOf(a1), d1, w1, n2)
	t.Logf("   A2 both-masked      mean %.2f tok/round   paired %+.1f%%  wins %d/%d  [DECISION ARM]",
		meanOf(a2), d2, w2, n2)
	t.Logf("   -> verdict inputs: delta=%+.1f%%  wins=%d/%d  absolute=%.2f", d2, w2, n2, meanOf(a2))
}
