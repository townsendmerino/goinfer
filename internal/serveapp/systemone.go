package serveapp

// POST /v1/systemone — TypeSafe's decisions wire shape (D5 of docs/tasks/task-constrained-confidence.md; the request
// and response JSON are recorded verbatim in docs/measurements/decisions-d0-prior-art-2026-09-27.md §3), answered by
// label scoring on the served model (Route A, internal/decide). It exists so jevx and TypeSafe's SDKs (JS, Python,
// Vercel, LangChain), which all take a base-URL override, work against goinfer unchanged.
//
// What it is not: TypeSafe's hosted model or a trained decision head. Its distributions are the served model's own
// probabilities over the options, calibrated only for a kind whose temperature -decisions-calibration supplies (the
// response's goinfer.calibrated says which).

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"

	"github.com/townsendmerino/goinfer/internal/decide"
)

// maxSystemOneQuestions bounds one request: each question is one prefill of the state. TypeSafe documents no limit;
// jevx sends at most 32.
const maxSystemOneQuestions = 256

type systemOneQuestion struct {
	Type         string          `json:"type"`
	Instructions json.RawMessage `json:"instructions"`
	Criteria     json.RawMessage `json:"criteria"`
}

// systemOneItem is one validated question: its name, the decide request it becomes, and what the answer reports
// back (the option names a choice's criteria keyed, a score's level texts).
type systemOneItem struct {
	name  string
	req   decide.Request
	names []string // choice: the criteria keys, aligned with req.Options
}

func (s *server) handleSystemOne(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeErr(w, http.StatusBadRequest, "read body: "+err.Error())
		return
	}
	var req struct {
		Model     string          `json:"model"`
		State     json.RawMessage `json:"state"`
		Questions json.RawMessage `json:"questions"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid JSON body: "+err.Error())
		return
	}
	state := jsonText(req.State)
	if strings.TrimSpace(state) == "" {
		writeErr(w, http.StatusUnprocessableEntity, "state is required")
		return
	}
	names, raws, err := orderedObject(req.Questions)
	if err != nil {
		writeErr(w, http.StatusUnprocessableEntity, "questions must be an object of named questions: "+err.Error())
		return
	}
	if len(names) == 0 || len(names) > maxSystemOneQuestions {
		writeErr(w, http.StatusUnprocessableEntity, fmt.Sprintf("questions must hold 1..%d questions, got %d", maxSystemOneQuestions, len(names)))
		return
	}
	items := make([]systemOneItem, len(names))
	for i, name := range names {
		it, err := systemOneRequest(name, state, raws[i])
		if err != nil {
			writeErr(w, http.StatusUnprocessableEntity, fmt.Sprintf("question %q: %v", name, err))
			return
		}
		items[i] = it
	}
	s.withModel(w, req.Model, func(lm *loadedModel) { s.serveSystemOne(w, r, lm, items) })
}

// systemOneRequest turns one TypeSafe question into a decide request, validating it against decide's own rules.
func systemOneRequest(name, state string, raw json.RawMessage) (systemOneItem, error) {
	var q systemOneQuestion
	if err := json.Unmarshal(raw, &q); err != nil {
		return systemOneItem{}, err
	}
	it := systemOneItem{name: name, req: decide.Request{Kind: q.Type, State: state, Question: jsonText(q.Instructions)}}
	if strings.TrimSpace(it.req.Question) == "" {
		return it, errors.New("instructions is required")
	}
	switch q.Type {
	case decide.KindNoul:
		it.req.Options = []string{"false", "true"}
		if len(q.Criteria) > 0 && string(q.Criteria) != "null" {
			var c map[string]json.RawMessage
			if err := json.Unmarshal(q.Criteria, &c); err != nil {
				return it, errors.New("a noul question's criteria is an optional {\"true\": …, \"false\": …} object")
			}
			it.req.Descriptions = []string{jsonText(c["false"]), jsonText(c["true"])}
		}
	case decide.KindChoice:
		keys, vals, err := orderedObject(q.Criteria)
		if err != nil || len(keys) == 0 {
			return it, errors.New("a choice question's criteria is a required {\"option\": description | null} object")
		}
		if len(keys) > decide.MaxChoiceOptions {
			return it, fmt.Errorf("goinfer's label scoring takes at most %d options (the letters A..P), got %d", decide.MaxChoiceOptions, len(keys))
		}
		it.req.Options, it.names = keys, keys
		it.req.Descriptions = make([]string, len(keys))
		for i, v := range vals {
			it.req.Descriptions[i] = jsonText(v)
		}
	case decide.KindScore:
		var levels []json.RawMessage
		if err := json.Unmarshal(q.Criteria, &levels); err != nil {
			return it, errors.New("a score question's criteria is a required ordered array of level descriptions")
		}
		if len(levels) < 2 || len(levels) > decide.MaxScoreLevels {
			return it, fmt.Errorf("a score takes 2..%d levels, got %d", decide.MaxScoreLevels, len(levels))
		}
		it.req.Options = decide.ScoreOptions(len(levels))
		it.req.Descriptions = make([]string, len(levels))
		for i, l := range levels {
			it.req.Descriptions[i] = jsonText(l)
		}
	default:
		return it, fmt.Errorf("unknown type %q (want noul, choice or score)", q.Type)
	}
	return it, decide.Validate(it.req)
}

func (s *server) serveSystemOne(w http.ResponseWriter, r *http.Request, lm *loadedModel, items []systemOneItem) {
	if lm.adapter != "" {
		// decide prefills through Model.Generate, which does not apply a compute-time adapter: it would answer
		// with the base model under the adapter's name.
		writeErr(w, http.StatusBadRequest, "decisions are not supported on a compute-time adapter entry; ask the base model")
		return
	}
	d, err := lm.getDecider(s.cfg)
	if err != nil {
		writeErr(w, http.StatusBadRequest, "this model cannot answer decisions: "+err.Error())
		return
	}
	textBytes := 0
	for _, it := range items {
		textBytes = max(textBytes, len(it.req.State)+len(it.req.Question)+len(strings.Join(it.req.Options, ""))+len(strings.Join(it.req.Descriptions, "")))
	}
	if err := lm.promptTooLargeForContext(textBytes); err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	if !lm.enter(w, r, admissionRecord{}, s.haltState) {
		return
	}
	defer lm.exit()
	var answers orderedJSON
	calibrated := map[string]bool{}
	inputTokens := 0
	for _, it := range items {
		res, err := d.Decide(r.Context(), it.req)
		if err != nil {
			if r.Context().Err() != nil {
				return // the client left
			}
			writeServerErr(w, fmt.Sprintf("question %q: %v", it.name, err))
			return
		}
		inputTokens += res.PromptTokens * res.Prefills
		calibrated[it.req.Kind] = res.Calibrated
		answers = append(answers, orderedField{it.name, systemOneAnswer(it, res)})
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"model":   lm.name,
		"answers": answers,
		// Label scoring decodes nothing: every prompt token is input, and no token is generated.
		"usage": map[string]int{"input_tokens": inputTokens, "output_tokens": 0},
		"goinfer": map[string]any{"route": "label", "template": d.Template(), "calibrated": calibrated,
			"note": "probabilities are the served model's own over the options; calibrated only where a fitted temperature says so"},
	})
}

// systemOneAnswer is one answer in TypeSafe's shape: noul → its P(true); choice → the chosen option, the
// probabilities by option name and a confidence; score → the expected level, the legend, the probabilities by level
// and a confidence.
func systemOneAnswer(it systemOneItem, res decide.Result) any {
	n := len(res.Distribution)
	switch it.req.Kind {
	case decide.KindNoul:
		return map[string]any{"type": "noul", "noul": res.Distribution[1]}
	case decide.KindChoice:
		probs := orderedJSON{}
		for i, name := range it.names {
			probs = append(probs, orderedField{name, res.Distribution[i]})
		}
		return map[string]any{"type": "choice", "choice": it.names[res.Index], "probabilities": probs,
			"confidence": typesafeConfidence(res.Confidence, n)}
	default: // score
		probs, legend := orderedJSON{}, orderedJSON{}
		for i := range res.Distribution {
			level := it.req.Options[i]
			probs = append(probs, orderedField{level, res.Distribution[i]})
			legend = append(legend, orderedField{level, it.req.Descriptions[i]})
		}
		return map[string]any{"type": "score", "score": *res.ExpectedScore, "legend": legend, "probabilities": probs,
			"confidence": typesafeConfidence(res.Confidence, n)}
	}
}

// typesafeConfidence maps the top probability of n options to [0, 1] by its margin over uniform, (n·p − 1)/(n − 1):
// 0 when the model cannot tell the options apart, 1 when it is certain. TypeSafe does not publish its formula; its
// docs' demo uses this form for three options, (3p − 1)/2. jevx compares it against its min_confidence (0.6).
func typesafeConfidence(pmax float64, n int) float64 {
	if n < 2 {
		return 1
	}
	return math.Max(0, math.Min(1, (float64(n)*pmax-1)/float64(n-1)))
}

// getDecider builds this model's decider once: its label tokens resolved, the -decisions-template, and the
// -decisions-calibration temperatures if any.
func (lm *loadedModel) getDecider(cfg config) (*decide.Decider, error) {
	lm.deciderOnce.Do(func() {
		if lm.model == nil || lm.tk == nil {
			lm.deciderErr = errors.New("not a generative model")
			return
		}
		tmpl := cfg.decisionsTemplate
		if tmpl == "" {
			tmpl = decide.TemplateChat // the flag's default
		}
		var cal *decide.Calibration
		if cfg.decisionsCal != "" {
			if cal, lm.deciderErr = decide.LoadCalibration(cfg.decisionsCal); lm.deciderErr != nil {
				return
			}
		}
		lm.decider, lm.deciderErr = decide.New(decide.NewPlainTokenizer(lm.tk), decide.ModelPrefill(lm.model),
			decide.Options{Template: tmpl, Calibration: cal})
	})
	return lm.decider, lm.deciderErr
}

// jsonText is a TypeSafe text field as the prompt shows it: a JSON string's text, anything else (an object, an
// array) compacted to its JSON, and "" for absent or null. jevx does the same to a JSON state.
func jsonText(raw json.RawMessage) string {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || string(raw) == "null" {
		return ""
	}
	var s string
	if raw[0] == '"' && json.Unmarshal(raw, &s) == nil {
		return s
	}
	var b bytes.Buffer
	if json.Compact(&b, raw) == nil {
		return b.String()
	}
	return string(raw)
}

// orderedObject decodes a JSON object's keys in document order with their raw values. Choice options are labelled
// A, B, C… in the order the client wrote them, which a Go map would lose.
func orderedObject(raw json.RawMessage) ([]string, []json.RawMessage, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	tok, err := dec.Token()
	if err != nil {
		return nil, nil, err
	}
	if d, ok := tok.(json.Delim); !ok || d != '{' {
		return nil, nil, errors.New("not a JSON object")
	}
	var keys []string
	var vals []json.RawMessage
	seen := map[string]bool{}
	for dec.More() {
		t, err := dec.Token()
		if err != nil {
			return nil, nil, err
		}
		k, _ := t.(string)
		if seen[k] {
			return nil, nil, fmt.Errorf("duplicate key %q", k)
		}
		seen[k] = true
		var v json.RawMessage
		if err := dec.Decode(&v); err != nil {
			return nil, nil, err
		}
		keys, vals = append(keys, k), append(vals, v)
	}
	return keys, vals, nil
}

// orderedJSON is a JSON object that keeps its fields in insertion order.
type orderedJSON []orderedField

type orderedField struct {
	key string
	val any
}

func (o orderedJSON) MarshalJSON() ([]byte, error) {
	var b bytes.Buffer
	b.WriteByte('{')
	for i, f := range o {
		if i > 0 {
			b.WriteByte(',')
		}
		k, err := json.Marshal(f.key)
		if err != nil {
			return nil, err
		}
		v, err := json.Marshal(f.val)
		if err != nil {
			return nil, err
		}
		b.Write(k)
		b.WriteByte(':')
		b.Write(v)
	}
	b.WriteByte('}')
	return b.Bytes(), nil
}
