package clef

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// The WHOLE pipeline against the reference: request -> Go encoder -> Go backbone (decoder.PromptHiddenAll on the committed tiny qwen3_5, hidden 64, UNTIED
// lm_head, random final-norm weight) -> Go head (width 32, seeded random weights in every parameter) -> per-option probabilities, compared with the official
// joint_schema_model.py chained as ClefModel.forward chains it, at f32 (scripts/pin_clef_e2e_tiny.py). It exists because the encoder, the backbone seam and the
// head are each gated in isolation, and nothing else runs them together: where the lm_head rows come from, whether the head sees the post-final-norm hidden state,
// and whether the encoder's spans index the rows the backbone returns are only visible end to end.
//
// Everything it reads is committed, so a missing file FAILS (a skip would hide the absence). PROBABILITY BAR, written before the first run: 1e-4 absolute against the
// reference. The head alone matched its reference to 1.2e-7 given identical inputs; the backbone adds its own f32 differences (D11's bar is relative L2 1e-5 on the
// hidden state), which the head's attention and softmax pass through, so the end-to-end bar is looser than the head's 1e-5 by design and the measured value is logged.
const pipelineProbBar = 1e-4

// chunkTokenize is the fixture's stand-in tokenizer: each fragment's UTF-8 bytes in chunks of 4, one id per chunk, v = fold(v*131 + byte) mod 256. Keep in step
// with chunk_ids in scripts/pin_clef_e2e_tiny.py.
func chunkTokenize(text string, _ bool) ([]int, error) {
	b := []byte(text)
	var out []int
	for i := 0; i < len(b); i += 4 {
		v := 0
		for _, c := range b[i:min(i+4, len(b))] {
			v = (v*131 + int(c)) % 256
		}
		out = append(out, v)
	}
	return out, nil
}

type pipelineGolden struct {
	Items []struct {
		Name      string          `json:"name"`
		Request   json.RawMessage `json:"request"`
		InputIDs  []int           `json:"input_ids"`
		Probs     [][]float64     `json:"probs"`
		Questions []struct {
			ID           string   `json:"id"`
			Type         int      `json:"type"`
			OptionIDs    []string `json:"option_ids"`
			QuestionSpan [2]int   `json:"question_span"`
			OptionSpans  [][2]int `json:"option_spans"`
		} `json:"questions"`
	} `json:"items"`
}

func loadTinyPipeline(t *testing.T) (*Model, pipelineGolden) {
	t.Helper()
	raw, err := os.ReadFile("testdata/tiny/golden.json")
	if err != nil {
		t.Fatalf("no tiny golden (committed; a skip would hide this): %v", err)
	}
	var g pipelineGolden
	if err := json.Unmarshal(raw, &g); err != nil || len(g.Items) < 4 {
		t.Fatalf("golden.json: %v (%d items)", err, len(g.Items))
	}
	bb, err := decoder.Load("../../decoder/testdata/qwen3_5-tiny-normw", decoder.Options{})
	if err != nil {
		t.Fatalf("backbone (committed): %v", err)
	}
	t.Cleanup(func() { _ = bb.Close() })
	head, err := LoadHead("testdata/tiny")
	if err != nil {
		t.Fatalf("head (committed): %v", err)
	}
	m, err := NewModel(bb, head, chunkTokenize, 0)
	if err != nil {
		t.Fatal(err)
	}
	return m, g
}

func TestPipeline_tinyMatchesReference(t *testing.T) {
	m, g := loadTinyPipeline(t)
	worst := 0.0
	for _, it := range g.Items {
		// The encoder half, exactly: ids and spans.
		enc, err := Encode(it.Request, chunkTokenize, 0)
		if err != nil {
			t.Fatalf("%s: %v", it.Name, err)
		}
		if fmt.Sprint(enc.InputIDs) != fmt.Sprint(it.InputIDs) {
			t.Fatalf("%s: the encoder's token ids differ from the official encoder's", it.Name)
		}
		for i, q := range it.Questions {
			e := enc.Questions[i]
			if e.ID != q.ID || e.Type != q.Type || e.QuestionSpan != q.QuestionSpan || fmt.Sprint(e.OptionSpans) != fmt.Sprint(q.OptionSpans) || fmt.Sprint(e.OptionIDs) != fmt.Sprint(q.OptionIDs) {
				t.Fatalf("%s question %d: encoder %+v, official %+v", it.Name, i, e, q)
			}
		}
		// The whole pipeline.
		res, err := m.Decide(context.Background(), it.Request)
		if err != nil {
			t.Fatalf("%s: %v", it.Name, err)
		}
		if res.InputTokens != len(it.InputIDs) || len(res.Answers) != len(it.Probs) {
			t.Fatalf("%s: %d tokens and %d answers, want %d and %d", it.Name, res.InputTokens, len(res.Answers), len(it.InputIDs), len(it.Probs))
		}
		for qi, a := range res.Answers {
			if len(a.Probs) != len(it.Probs[qi]) {
				t.Fatalf("%s question %d: %d options, want %d", it.Name, qi, len(a.Probs), len(it.Probs[qi]))
			}
			var sum float64
			for o, p := range a.Probs {
				sum += p
				d := math.Abs(p - it.Probs[qi][o])
				worst = math.Max(worst, d)
				if d > pipelineProbBar {
					t.Errorf("%s question %q option %q: probability %.8f, reference %.8f (diff %.2g > %g)", it.Name, a.ID, a.Options[o], p, it.Probs[qi][o], d, pipelineProbBar)
				}
			}
			if math.Abs(sum-1) > 1e-9 {
				t.Errorf("%s question %q: probabilities sum to %.12f", it.Name, a.ID, sum)
			}
		}
	}
	t.Logf("%d requests (noul, choice, score, three questions at once), whole pipeline against the official reference: worst probability difference %.3g (bar %g)", len(g.Items), worst, pipelineProbBar)
}

// The three requests' answers must actually depend on the request: the same state with a different question, or the same question with a different state, must move
// the probabilities. Without this a pipeline that ignored its input (a constant head output) would match a golden it was fitted to.
func TestPipeline_answersDependOnTheRequest(t *testing.T) {
	m, _ := loadTinyPipeline(t)
	ask := func(state, instr string) []float64 {
		req := fmt.Sprintf(`{"state":%q,"questions":{"q":{"type":"noul","instructions":%q}}}`, state, instr)
		r, err := m.Decide(context.Background(), []byte(req))
		if err != nil {
			t.Fatal(err)
		}
		return r.Answers[0].Probs
	}
	base := ask("Customer asks for a refund.", "Refund?")
	for name, other := range map[string][]float64{"another state": ask("Customer asks to cancel.", "Refund?"), "another question": ask("Customer asks for a refund.", "Escalate?")} {
		if math.Abs(other[0]-base[0]) < 1e-3 {
			t.Errorf("%s did not move the probability (%.6f vs %.6f)", name, other[0], base[0])
		}
	}
}

func TestNewModel_refusesAMismatchedPair(t *testing.T) {
	m, _ := loadTinyPipeline(t)
	head := *m.head
	head.Cfg.HiddenSize = 128
	if _, err := NewModel(m.backbone, &head, chunkTokenize, 0); err == nil {
		t.Error("a head expecting 128 columns was joined to a 64-wide backbone")
	}
	if _, err := NewModel(nil, m.head, chunkTokenize, 0); err == nil {
		t.Error("a nil backbone was accepted")
	}
}
