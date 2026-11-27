package systemone

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/townsendmerino/goinfer/internal/clef"
	"github.com/townsendmerino/goinfer/internal/decide"
)

const (
	RouteClef      = "clef"
	ClefConfidence = "top_probability"
)

// IsClefDir reports whether path is a Clef model directory carrying joint_head.safetensors.
func IsClefDir(path string) bool {
	st, err := os.Stat(filepath.Join(path, "joint_head.safetensors"))
	return err == nil && !st.IsDir()
}

// ClefQuantRefusal refuses int4 quantization on Clef checkpoints where fidelity degrades against int8int8/f32.
func ClefQuantRefusal(quant string) error {
	if quant == "int4" {
		return fmt.Errorf("a Clef model is not served at int4: it reads clearly worse than int8int8, the default, against the f32 reference (docs/measurements/decisions-d13-clef-fidelity-2026-10-03.md, section 7); use quant=int8int8 or quant=f32")
	}
	return nil
}

// ClefQuestion holds the request-order metadata needed to shape one Clef answer.
type ClefQuestion struct {
	Name   string
	Typ    string
	Keys   []string          // choice: criteria keys in request order
	Levels []json.RawMessage // score: level descriptions
}

// ParseClefQuestions validates the questions object of a Clef /v1/systemone request in document order.
func ParseClefQuestions(rawQuestions json.RawMessage) ([]ClefQuestion, error) {
	names, raws, err := OrderedObject(rawQuestions)
	if err != nil {
		return nil, fmt.Errorf("questions must be an object of named questions: %w", err)
	}
	if len(names) == 0 || len(names) > MaxQuestions {
		return nil, fmt.Errorf("questions must hold 1..%d questions, got %d", MaxQuestions, len(names))
	}
	qs := make([]ClefQuestion, len(names))
	for i, name := range names {
		q := ClefQuestion{Name: name}
		var spec struct {
			Type     string          `json:"type"`
			Criteria json.RawMessage `json:"criteria"`
		}
		if err := json.Unmarshal(raws[i], &spec); err != nil {
			return nil, fmt.Errorf("question %q: %w", name, err)
		}
		q.Typ = spec.Type
		switch spec.Type {
		case decide.KindNoul:
		case decide.KindChoice:
			keys, _, kerr := OrderedObject(spec.Criteria)
			if kerr != nil || len(keys) == 0 {
				return nil, fmt.Errorf("question %q: a choice question's criteria is a required {\"option\": description | null} object", name)
			}
			q.Keys = keys
		case decide.KindScore:
			if err := json.Unmarshal(spec.Criteria, &q.Levels); err != nil || len(q.Levels) == 0 {
				return nil, fmt.Errorf("question %q: a score question's criteria is a required ordered array of level descriptions", name)
			}
		default:
			return nil, fmt.Errorf("question %q: unknown type %q (want noul, choice or score)", name, spec.Type)
		}
		qs[i] = q
	}
	return qs, nil
}

// FormatClefResponse shapes a Clef joint-head result into a /v1/systemone response payload.
func FormatClefResponse(model string, qs []ClefQuestion, res *clef.Result) (map[string]any, error) {
	if len(res.Answers) != len(qs) {
		return nil, fmt.Errorf("the model answered %d questions for %d asked", len(res.Answers), len(qs))
	}
	var answers OrderedJSON
	for i, q := range qs {
		a, err := ClefAnswer(q, res.Answers[i])
		if err != nil {
			return nil, fmt.Errorf("question %q: %w", q.Name, err)
		}
		answers = append(answers, OrderedField{Key: q.Name, Val: a})
	}
	gi := map[string]any{
		"route": RouteClef, "calibrated": false, "confidence": ClefConfidence, "note": ClefNote(),
		"state_tokens_truncated": res.StateTruncated,
	}
	return map[string]any{
		"model":   model,
		"answers": answers,
		"usage":   map[string]int{"input_tokens": res.InputTokens, "output_tokens": 0},
		"goinfer": gi,
	}, nil
}

// ClefNote returns the Clef route's calibration and confidence description.
func ClefNote() string {
	return "Cloudflare's Clef joint decision model: one backbone pass over the whole record answers every question. Probabilities are the model's own, " +
		"uncalibrated by goinfer; confidence is the top probability (the reference's), not the margin over uniform the label route reports; " +
		"probabilities are rounded to four decimals as the reference rounds them. A state longer than the context is cut to its first tokens " +
		"(goinfer.state_tokens_truncated counts what was cut)."
}

// Round4 rounds a probability to 4 decimal places to match Python's round(x, 4) in the Clef reference.
func Round4(x float64) float64 { return math.Round(x*1e4) / 1e4 }

// ClefAnswer formats one Clef answer in the reference's systemone_answer shape.
func ClefAnswer(q ClefQuestion, a clef.Answer) (any, error) {
	byID := make(map[string]float64, len(a.Options))
	for i, id := range a.Options {
		byID[id] = a.Probs[i]
	}
	switch q.Typ {
	case decide.KindNoul:
		p, ok := byID["true"]
		if !ok {
			return nil, errors.New("no probability for the option \"true\"")
		}
		return map[string]any{"type": "noul", "noul": Round4(p)}, nil
	case decide.KindChoice:
		probs := OrderedJSON{}
		best, bestP := "", math.Inf(-1)
		for _, k := range q.Keys {
			p, ok := byID[k]
			if !ok {
				return nil, fmt.Errorf("no probability for the option %q", k)
			}
			if p > bestP {
				best, bestP = k, p
			}
			probs = append(probs, OrderedField{Key: k, Val: Round4(p)})
		}
		return map[string]any{"type": "choice", "choice": best, "confidence": Round4(bestP), "probabilities": probs}, nil
	default:
		probs, legend := OrderedJSON{}, OrderedJSON{}
		expected, top := 0.0, math.Inf(-1)
		for i := range q.Levels {
			level := fmt.Sprint(i)
			p, ok := byID[level]
			if !ok {
				return nil, fmt.Errorf("no probability for the level %q", level)
			}
			expected += float64(i) * p
			top = math.Max(top, p)
			probs = append(probs, OrderedField{Key: level, Val: Round4(p)})
			legend = append(legend, OrderedField{Key: level, Val: json.RawMessage(bytes.TrimSpace(q.Levels[i]))})
		}
		return map[string]any{"type": "score", "score": Round4(expected), "confidence": Round4(top), "legend": legend, "probabilities": probs}, nil
	}
}
