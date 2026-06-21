package decoder

import (
	"context"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/constrain"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestTreeUpside is the 03 inc-2 (tree verification) go/no-go. A tree only beats the
// linear priority router when two sources DISAGREE at a position and the linear
// router picked the wrong one — i.e. the chosen source's token is rejected but
// another source's token is the model's (masked) pick. Tree verification is the
// project's hardest build (a non-contiguous attention mask across the CPU / resident
// / int8 / ring paths), so measure the upside first.
//
// It replays the drafters over a real constrained greedy generation (agent-loop
// case, where the router already wins 4.25 tok/round) and counts tree-recoverable
// positions. ~0 ⇒ trees add nothing for these greedy sources ⇒ defer.
func TestTreeUpside(t *testing.T) {
	m, err := loadBenchModel()
	if err != nil {
		t.Skipf("no model (%v); set GINFER_PREQUANT_GGUF", err)
	}
	tk, err := tokenizer.LoadGGUF(benchGGUFPath())
	if err != nil {
		t.Skipf("no tokenizer (%v)", err)
	}
	vocab := m.w.arch.VocabSize
	vocabBytes := constrain.TokenBytes(vocab, tk.TokenText)
	encode := func(s string) []int { ids, _ := tk.Encode(s, false); return ids }
	ctx := context.Background()

	schema := `{"type":"object","properties":{"location":{"type":"string"},"unit":{"enum":["celsius","fahrenheit"]}},"required":["location","unit"],"additionalProperties":false}`
	priorJSON := "{\n  \"location\": \"Paris\",\n  \"unit\": \"celsius\"\n}"
	prompt, _ := tk.Encode("<|im_start|>user\nRepeat this JSON exactly:\n"+priorJSON+"<|im_end|>\n<|im_start|>assistant\n", true)
	newMask := func() *constrain.Masker {
		g, _ := constrain.JSONSchema([]byte(schema))
		return constrain.NewMasker(g, vocabBytes, m.eosIDs).StopWhenComplete()
	}

	// The actual constrained greedy output (what the verify must reproduce).
	sp := SamplingParams{Temperature: 0, LogitProcessor: newMask().Process}
	gen := collectTokens(first(m.Generate(ctx, prompt, 64, sp)))

	// Replay grammar + n-gram proposals against each emitted token.
	mask := newMask()
	ngd := &NgramDrafter{}
	nctx := slices.Clone(prompt)
	var total, gProp, gCorrect, nProp, nCorrect, disagree, treeRecoverable int
	var grammarFirstHit, ngramFirstHit int // first-token-correct under each priority order
	for _, actual := range gen {
		total++
		gp, np := -1, -1
		if b := mask.ForcedBytesRun(64); len(b) > 0 {
			if ts := encode(string(b)); len(ts) > 0 {
				gp = ts[0]
			}
		}
		if d := ngd.Draft(nctx, 1); len(d) > 0 { // nctx = prompt+generated so far (excludes actual)
			np = d[0]
		}
		if gp >= 0 {
			gProp++
			if gp == actual {
				gCorrect++
			}
		}
		if np >= 0 {
			nProp++
			if np == actual {
				nCorrect++
			}
		}
		if gp >= 0 && np >= 0 && gp != np {
			disagree++
		}
		// priority router picks grammar if it proposed, else n-gram. A tree recovers a
		// position when the chosen source missed but the OTHER source had the answer.
		if gp >= 0 && gp != actual && np == actual {
			treeRecoverable++ // grammar chosen+wrong, n-gram would've been right
		}
		// Cheap alternative to a tree: source ORDERING. grammar-first vs n-gram-first
		// (each falling back to the other when its first choice abstains).
		gFirst := gp
		if gFirst < 0 {
			gFirst = np
		}
		nFirst := np
		if nFirst < 0 {
			nFirst = gp
		}
		if gFirst == actual {
			grammarFirstHit++
		}
		if nFirst == actual {
			ngramFirstHit++
		}
		mask.Commit(actual)
		nctx = append(nctx, actual)
	}

	t.Logf("agent-loop, %d tokens: grammar proposed %d (correct %d) | n-gram proposed %d (correct %d) | sources disagree %d | TREE-RECOVERABLE %d",
		total, gProp, gCorrect, nProp, nCorrect, disagree, treeRecoverable)
	t.Logf("→ tree upside over the linear priority router: %d / %d positions (%.1f%%)",
		treeRecoverable, total, 100*float64(treeRecoverable)/float64(total))
	t.Logf("→ but ORDERING captures it cheaply: first-token-correct grammar-first %d/%d vs n-gram-first %d/%d — "+
		"the upside is priority-order suboptimality (inc-3 α̂/confidence selection), not a missing tree",
		grammarFirstHit, total, ngramFirstHit, total)
}
