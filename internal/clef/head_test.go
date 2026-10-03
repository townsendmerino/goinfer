package clef

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// D12's head gate (docs/tasks/task-constrained-confidence.md): given the reference's own inputs, the head's per-option probabilities
// are within 1e-5 absolute of the reference's, at f32.
//
// A GOLDEN is a directory: golden.json (per item: tag, id, input ids, the encoder's questions, the lm_head rows' token ids, and the
// reference's logits and probabilities) plus <tag>.hidden.f32 ([L, 4096] little-endian f32, the backbone's post-final-norm hidden state)
// and <tag>.rows.f32 ([len(row_tokens), 4096] f32, the raw lm_head rows of the tokens the options use).
//
// The head is checked in ISOLATION: the hidden states are inputs, not computed. Two goldens exist or will:
//   - a synthetic one (seeded random hidden states and lm_head rows, run through the reference JointSchemaHead), for development before
//     the backbone's hidden states exist; it is NOT committed (about 8 MB of incompressible floats);
//   - the real one, from the D10 f32 fixture's last_hidden_state for the 3 shortest items (the `d10-clef-f32` night job).
//
// Where to find them: $CLEF_HEAD_GOLDEN (a golden directory), else testdata/decisions/clef/head_golden. The head weights: $CLEF_HEAD (a
// directory with joint_head.safetensors and joint_head_config.json), else ~/models/clef-flash. Either missing FAILS under
// GOINFER_HEAVY_TESTS=1 and otherwise skips; a skip is not a pass.

type headGoldenItem struct {
	Tag       string          `json:"tag"`
	ID        string          `json:"id"`
	N         int             `json:"n"`
	IDs       []int           `json:"ids"`
	RowTokens []int           `json:"row_tokens"`
	Request   json.RawMessage `json:"request"` // present on a multi-question item: the request the official encoder encoded
	Logits    [][]float64     `json:"logits"`
	Probs     [][]float64     `json:"probs"`
	Questions []struct {
		ID          string   `json:"id"`
		Type        int      `json:"type"`
		Span        [2]int   `json:"question_span"`
		OptionSpans [][2]int `json:"option_spans"`
		OptionIDs   []string `json:"option_ids"`
	} `json:"questions"`
}

func needAsset(t *testing.T, what string, dirs ...string) string {
	t.Helper()
	for _, d := range dirs {
		if d == "" {
			continue
		}
		if _, err := os.Stat(d); err == nil {
			return d
		}
	}
	if os.Getenv("GOINFER_HEAVY_TESTS") == "1" {
		t.Fatalf("GOINFER_HEAVY_TESTS=1 but no %s (looked in %v)", what, dirs)
	}
	t.Skipf("no %s (looked in %v)", what, dirs)
	return ""
}

func readF32(t *testing.T, path string, n int) []float32 {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(b) != 4*n {
		t.Fatalf("%s: %d bytes, want %d", path, len(b), 4*n)
	}
	out := make([]float32, n)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out
}

func TestHead_matchesReference(t *testing.T) {
	home, _ := os.UserHomeDir()
	gdir := needAsset(t, "head golden", os.Getenv("CLEF_HEAD_GOLDEN"), "../../testdata/decisions/clef/head_golden")
	hdir := needAsset(t, "Clef-flash head weights", os.Getenv("CLEF_HEAD"), filepath.Join(home, "models", "clef-flash"))
	raw, err := os.ReadFile(filepath.Join(gdir, "golden.json"))
	if err != nil {
		t.Fatal(err)
	}
	var items []headGoldenItem
	if err := json.Unmarshal(raw, &items); err != nil || len(items) == 0 {
		t.Fatalf("golden.json: %v (%d items)", err, len(items))
	}
	head, err := LoadHead(hdir)
	if err != nil {
		t.Fatal(err)
	}
	const hs = 4096
	worstP, worstL := 0.0, 0.0
	for _, it := range items {
		hflat := readF32(t, filepath.Join(gdir, it.Tag+".hidden.f32"), it.N*hs)
		hidden := make([][]float32, it.N)
		for i := range hidden {
			hidden[i] = hflat[i*hs : (i+1)*hs]
		}
		rflat := readF32(t, filepath.Join(gdir, it.Tag+".rows.f32"), len(it.RowTokens)*hs)
		rowOf := map[int][]float32{}
		for i, tok := range it.RowTokens {
			rowOf[tok] = rflat[i*hs : (i+1)*hs]
		}
		var qs []Question
		for _, q := range it.Questions {
			qs = append(qs, Question{ID: q.ID, Type: q.Type, QuestionSpan: q.Span, OptionSpans: q.OptionSpans, OptionIDs: q.OptionIDs})
		}
		got, err := head.Forward(hidden, it.IDs, qs, func(id int) ([]float32, error) {
			r, ok := rowOf[id]
			if !ok {
				t.Fatalf("%s: the head asked for lm_head row %d, which the golden does not carry", it.Tag, id)
			}
			return r, nil
		})
		if err != nil {
			t.Fatal(err)
		}
		for qi := range qs {
			p := Softmax(got[qi])
			if len(p) != len(it.Probs[qi]) {
				t.Fatalf("%s q%d: %d options, want %d", it.Tag, qi, len(p), len(it.Probs[qi]))
			}
			for o := range p {
				dp := math.Abs(p[o] - it.Probs[qi][o])
				dl := math.Abs(float64(got[qi][o]) - it.Logits[qi][o])
				worstP, worstL = math.Max(worstP, dp), math.Max(worstL, dl)
				if dp > 1e-5 {
					t.Errorf("%s q%d option %d: probability %.8f, reference %.8f (diff %.2g > 1e-5)", it.Tag, qi, o, p[o], it.Probs[qi][o], dp)
				}
			}
		}
	}
	t.Logf("%d items, head isolated: worst probability difference %.3g (bar 1e-5), worst logit difference %.3g", len(items), worstP, worstL)
}

func TestLoadHead_refusesWhatItWasNotVerifiedAgainst(t *testing.T) {
	dir := t.TempDir()
	write := func(cfg string) {
		if err := os.WriteFile(filepath.Join(dir, "joint_head_config.json"), []byte(cfg), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	for name, cfg := range map[string]string{
		"unknown key": `{"hidden_size":4096,"width":1024,"routing_layers":2,"layers":4,"heads":16,"feedforward":4096,"dropout":0.1}`,
		"missing key": `{"hidden_size":4096,"width":1024,"routing_layers":2,"layers":4,"heads":16}`,
		"bad heads":   `{"hidden_size":4096,"width":1000,"routing_layers":2,"layers":4,"heads":16,"feedforward":4096}`,
		"zero layers": `{"hidden_size":4096,"width":1024,"routing_layers":0,"layers":4,"heads":16,"feedforward":4096}`,
	} {
		write(cfg)
		if _, err := LoadHead(dir); err == nil {
			t.Errorf("%s: the config was accepted", name)
		}
	}
}

// The fixture's 150 records are all single-question, so the golden carries one multi-question record encoded by the OFFICIAL encoder:
// the Go encoder must reproduce its ids and spans (FIELD numbering, several questions' spans shifted together, a quote and an accented
// letter in an instruction), and the head must match the reference on it (cross-question self-attention, options split per question).
func TestEncode_multiQuestionMatchesReference(t *testing.T) {
	home, _ := os.UserHomeDir()
	gdir := needAsset(t, "head golden", os.Getenv("CLEF_HEAD_GOLDEN"), "../../testdata/decisions/clef/head_golden")
	tk := clefTokenizer(t)
	_ = home
	raw, err := os.ReadFile(filepath.Join(gdir, "golden.json"))
	if err != nil {
		t.Fatal(err)
	}
	var items []headGoldenItem
	if err := json.Unmarshal(raw, &items); err != nil {
		t.Fatal(err)
	}
	n := 0
	for _, it := range items {
		if len(it.Request) == 0 {
			continue
		}
		n++
		got, err := Encode(it.Request, literalTokenizer(tk), 0)
		if err != nil {
			t.Fatal(err)
		}
		if len(got.InputIDs) != len(it.IDs) {
			t.Fatalf("%s: %d tokens, the official encoder made %d", it.Tag, len(got.InputIDs), len(it.IDs))
		}
		for i := range it.IDs {
			if got.InputIDs[i] != it.IDs[i] {
				t.Fatalf("%s: token %d is %d, the official encoder has %d", it.Tag, i, got.InputIDs[i], it.IDs[i])
			}
		}
		if len(got.Questions) != len(it.Questions) {
			t.Fatalf("%s: %d questions, want %d", it.Tag, len(got.Questions), len(it.Questions))
		}
		for i, w := range it.Questions {
			g := got.Questions[i]
			if g.ID != w.ID || g.Type != w.Type || g.QuestionSpan != w.Span || len(g.OptionSpans) != len(w.OptionSpans) {
				t.Errorf("%s question %d: got %+v, want %+v", it.Tag, i, g, w)
				continue
			}
			for j := range g.OptionSpans {
				if g.OptionSpans[j] != w.OptionSpans[j] || g.OptionIDs[j] != w.OptionIDs[j] {
					t.Errorf("%s question %d option %d: got %v %q, want %v %q", it.Tag, i, j, g.OptionSpans[j], g.OptionIDs[j], w.OptionSpans[j], w.OptionIDs[j])
				}
			}
		}
		t.Logf("%s: %d tokens, %d questions: ids and spans identical to the official encoder", it.Tag, len(got.InputIDs), len(got.Questions))
	}
	if n == 0 {
		t.Fatal("the golden carries no multi-question request, so this checked nothing")
	}
}
