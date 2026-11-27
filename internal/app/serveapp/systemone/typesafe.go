package systemone

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"

	"github.com/townsendmerino/goinfer/internal/decide"
)

// MaxQuestions bounds one /v1/systemone request.
const MaxQuestions = 256

type questionSpec struct {
	Type         string          `json:"type"`
	Instructions json.RawMessage `json:"instructions"`
	Criteria     json.RawMessage `json:"criteria"`
}

// Item is one validated TypeSafe question mapped onto decide.Request.
type Item struct {
	Name  string
	Req   decide.Request
	Names []string // choice: criteria keys aligned with Req.Options
}

// ParseRequest validates a TypeSafe state and named-questions object into decide requests.
func ParseRequest(stateRaw, questionsRaw json.RawMessage) ([]Item, error) {
	state := JSONText(stateRaw)
	if strings.TrimSpace(state) == "" {
		return nil, errors.New("state is required")
	}
	names, raws, err := OrderedObject(questionsRaw)
	if err != nil {
		return nil, fmt.Errorf("questions must be an object of named questions: %w", err)
	}
	if len(names) == 0 || len(names) > MaxQuestions {
		return nil, fmt.Errorf("questions must hold 1..%d questions, got %d", MaxQuestions, len(names))
	}
	items := make([]Item, len(names))
	for i, name := range names {
		it, err := ParseQuestion(name, state, raws[i])
		if err != nil {
			return nil, fmt.Errorf("question %q: %w", name, err)
		}
		items[i] = it
	}
	return items, nil
}

// ParseQuestion validates a single TypeSafe question into an Item.
func ParseQuestion(name, state string, raw json.RawMessage) (Item, error) {
	var q questionSpec
	if err := json.Unmarshal(raw, &q); err != nil {
		return Item{}, err
	}
	it := Item{Name: name, Req: decide.Request{Kind: q.Type, State: state, Question: JSONText(q.Instructions)}}
	if strings.TrimSpace(it.Req.Question) == "" {
		return it, errors.New("instructions is required")
	}
	switch q.Type {
	case decide.KindNoul:
		it.Req.Options = []string{"false", "true"}
		if len(q.Criteria) > 0 && string(q.Criteria) != "null" {
			var c map[string]json.RawMessage
			if err := json.Unmarshal(q.Criteria, &c); err != nil {
				return it, errors.New("a noul question's criteria is an optional {\"true\": …, \"false\": …} object")
			}
			it.Req.Descriptions = []string{JSONText(c["false"]), JSONText(c["true"])}
		}
	case decide.KindChoice:
		keys, vals, err := OrderedObject(q.Criteria)
		if err != nil || len(keys) == 0 {
			return it, errors.New("a choice question's criteria is a required {\"option\": description | null} object")
		}
		if len(keys) > decide.MaxChoiceOptions {
			return it, fmt.Errorf("goinfer's label scoring takes at most %d options (the letters A..P), got %d", decide.MaxChoiceOptions, len(keys))
		}
		it.Req.Options, it.Names = keys, keys
		it.Req.Descriptions = make([]string, len(keys))
		for i, v := range vals {
			it.Req.Descriptions[i] = JSONText(v)
		}
	case decide.KindScore:
		var levels []json.RawMessage
		if err := json.Unmarshal(q.Criteria, &levels); err != nil {
			return it, errors.New("a score question's criteria is a required ordered array of level descriptions")
		}
		if len(levels) < 2 || len(levels) > decide.MaxScoreLevels {
			return it, fmt.Errorf("a score takes 2..%d levels, got %d", decide.MaxScoreLevels, len(levels))
		}
		it.Req.Options = decide.ScoreOptions(len(levels))
		it.Req.Descriptions = make([]string, len(levels))
		for i, l := range levels {
			it.Req.Descriptions[i] = JSONText(l)
		}
	default:
		return it, fmt.Errorf("unknown type %q (want noul, choice or score)", q.Type)
	}
	return it, decide.Validate(it.Req)
}

// MaxTextBytes returns the largest per-question input byte length across items for context-size pre-screening.
func MaxTextBytes(items []Item) int {
	textBytes := 0
	for _, it := range items {
		textBytes = max(textBytes, len(it.Req.State)+len(it.Req.Question)+len(strings.Join(it.Req.Options, ""))+len(strings.Join(it.Req.Descriptions, "")))
	}
	return textBytes
}

// FormatResponse builds the /v1/systemone response payload from decide results.
func FormatResponse(model, route, template string, items []Item, results []decide.Result) map[string]any {
	var answers OrderedJSON
	calibrated := map[string]bool{}
	dropped := []string{}
	inputTokens := 0
	for i, it := range items {
		res := results[i]
		inputTokens += res.PromptTokens * res.Prefills
		calibrated[it.Req.Kind] = res.Calibrated
		if res.DescriptionsDropped {
			dropped = append(dropped, it.Name)
		}
		answers = append(answers, OrderedField{Key: it.Name, Val: Answer(it, res)})
	}
	gi := map[string]any{"route": route, "template": template, "calibrated": calibrated, "note": Note(route)}
	if route == decide.RouteHead {
		gi["descriptions_dropped"] = dropped
	}
	return map[string]any{
		"model":   model,
		"answers": answers,
		"usage":   map[string]int{"input_tokens": inputTokens, "output_tokens": 0},
		"goinfer": gi,
	}
}

// Answer formats one question's decide.Result into TypeSafe's answer shape.
func Answer(it Item, res decide.Result) any {
	n := len(res.Distribution)
	switch it.Req.Kind {
	case decide.KindNoul:
		return map[string]any{"type": "noul", "noul": res.Distribution[1]}
	case decide.KindChoice:
		probs := OrderedJSON{}
		for i, name := range it.Names {
			probs = append(probs, OrderedField{Key: name, Val: res.Distribution[i]})
		}
		return map[string]any{"type": "choice", "choice": it.Names[res.Index], "probabilities": probs,
			"confidence": Confidence(res.Confidence, n)}
	default:
		probs, legend := OrderedJSON{}, OrderedJSON{}
		for i := range res.Distribution {
			level := it.Req.Options[i]
			probs = append(probs, OrderedField{Key: level, Val: res.Distribution[i]})
			legend = append(legend, OrderedField{Key: level, Val: it.Req.Descriptions[i]})
		}
		return map[string]any{"type": "score", "score": *res.ExpectedScore, "legend": legend, "probabilities": probs,
			"confidence": Confidence(res.Confidence, n)}
	}
}

// Confidence maps the top probability of n options to [0, 1] by its margin over uniform, (n*p - 1)/(n - 1).
func Confidence(pmax float64, n int) float64 {
	if n < 2 {
		return 1
	}
	return math.Max(0, math.Min(1, (float64(n)*pmax-1)/float64(n-1)))
}

// Note describes the probability calibration semantics of route.
func Note(route string) string {
	if route == decide.RouteHead {
		return "probabilities are a trained decision head's over the options, calibrated by its own per-kind temperatures"
	}
	return "probabilities are the served model's own over the options; calibrated only where a fitted temperature says so"
}

// JSONText renders a JSON field as prompt text: unquoted string text, compacted JSON for arrays/objects, or "" for null/empty.
func JSONText(raw json.RawMessage) string {
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

// OrderedObject decodes a JSON object's keys and raw values in document order, rejecting duplicate keys.
func OrderedObject(raw json.RawMessage) ([]string, []json.RawMessage, error) {
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

// OrderedJSON is a JSON object slice that marshals fields in insertion order.
type OrderedJSON []OrderedField

// OrderedField is one key-value pair in an OrderedJSON object.
type OrderedField struct {
	Key string
	Val any
}

// MarshalJSON serializes o as a JSON object preserving field order.
func (o OrderedJSON) MarshalJSON() ([]byte, error) {
	var b bytes.Buffer
	b.WriteByte('{')
	for i, f := range o {
		if i > 0 {
			b.WriteByte(',')
		}
		k, err := json.Marshal(f.Key)
		if err != nil {
			return nil, err
		}
		v, err := json.Marshal(f.Val)
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
