package serveapp

// Route C of POST /v1/systemone (D13 of docs/tasks/task-constrained-confidence.md): a model directory that carries joint_head.safetensors is Cloudflare's Clef
// decision model, and one backbone pass over the whole record answers every question (internal/clef). The request is the same TypeSafe shape; what differs is
// that it is read as the reference's encoder reads it, from the raw body, and answered in the reference's own shape (systemone_answer in the checkpoint's
// joint_schema_model.py, read in full for docs/measurements/decisions-d10-clef-2026-10-02.md):
//
//   - noul: round(P(true), 4);
//   - choice: the argmax over the REQUEST's criteria order (the first maximum wins), the probabilities by option in that order, and confidence = the top probability;
//   - score: the expected level, the legend, the probabilities by level, and confidence = the top probability.
//
// confidence is the top probability on this route (owner decision 2026-10-03), not the label route's margin over uniform, and the response says so. Probabilities
// are rounded to four decimals as the reference rounds them.

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"os"
	"path/filepath"

	"github.com/townsendmerino/goinfer/internal/clef"
	"github.com/townsendmerino/goinfer/internal/decide"
)

const (
	routeClef = "clef"
	// clefConfidence names, in the response, which "confidence" this route reports.
	clefConfidence = "top_probability"
)

// isClefDir reports whether a --model path is a Clef model directory: one that carries the joint head beside the backbone.
func isClefDir(path string) bool {
	st, err := os.Stat(filepath.Join(path, "joint_head.safetensors"))
	return err == nil && !st.IsDir()
}

// clefQuantRefusal is the error a Clef model gets for a quant it is not offered at, or nil. int4 is refused (owner decision 2026-10-03, after D13): on 150
// records it read mean KL 0.051 and top-1 0.840 against the f32 reference on the CPU, and 0.044 and 0.840 on CUDA, where the default int8int8 reads 0.0165
// and 0.907 (decisions-d13-clef-fidelity-2026-10-03.md section 7). Other quants are not refused here: f32 and int8int8 are graded, int4mix is simply not measured.
func clefQuantRefusal(quant string) error {
	if quant == "int4" {
		return fmt.Errorf("a Clef model is not served at int4: D13 measured it at mean KL 0.051 and top-1 agreement 0.840 against the f32 reference (int8int8, the default: 0.0165 and 0.907); use quant=int8int8 or quant=f32")
	}
	return nil
}

// isClefModel reports whether the named entry is a loaded Clef model. It resolves through resolveAndLock like every other route to a model (the guard in
// liveness_test.go allows no other caller of lookupLocked) and releases at once; withModel takes the lock again for the request and rechecks.
func (s *server) isClefModel(name string) bool {
	lm, release := s.resolveAndLock(name)
	if lm == nil {
		return false
	}
	defer release()
	return lm.clef != nil
}

// clefQuestion is what the response shaping needs from one question of the request, in the order sent.
type clefQuestion struct {
	name   string
	typ    string
	keys   []string          // choice: the criteria keys in request order
	levels []json.RawMessage // score: the level descriptions
}

func (s *server) serveClef(w http.ResponseWriter, r *http.Request, lm *loadedModel, body []byte, rawQuestions json.RawMessage) {
	if lm.clef == nil {
		writeErr(w, http.StatusConflict, "this model is not a Clef model; the entry changed while the request was admitted")
		return
	}
	if lm.adapter != "" {
		writeErr(w, http.StatusBadRequest, "decisions are not supported on a compute-time adapter entry; ask the base model")
		return
	}
	names, raws, err := orderedObject(rawQuestions)
	if err != nil {
		writeErr(w, http.StatusUnprocessableEntity, "questions must be an object of named questions: "+err.Error())
		return
	}
	if len(names) == 0 || len(names) > maxSystemOneQuestions {
		writeErr(w, http.StatusUnprocessableEntity, fmt.Sprintf("questions must hold 1..%d questions, got %d", maxSystemOneQuestions, len(names)))
		return
	}
	qs := make([]clefQuestion, len(names))
	for i, name := range names {
		q := clefQuestion{name: name}
		var spec struct {
			Type     string          `json:"type"`
			Criteria json.RawMessage `json:"criteria"`
		}
		if err := json.Unmarshal(raws[i], &spec); err != nil {
			writeErr(w, http.StatusUnprocessableEntity, fmt.Sprintf("question %q: %v", name, err))
			return
		}
		q.typ = spec.Type
		switch spec.Type {
		case decide.KindNoul:
		case decide.KindChoice:
			keys, _, kerr := orderedObject(spec.Criteria)
			if kerr != nil || len(keys) == 0 {
				writeErr(w, http.StatusUnprocessableEntity, fmt.Sprintf("question %q: a choice question's criteria is a required {\"option\": description | null} object", name))
				return
			}
			q.keys = keys
		case decide.KindScore:
			if err := json.Unmarshal(spec.Criteria, &q.levels); err != nil || len(q.levels) == 0 {
				writeErr(w, http.StatusUnprocessableEntity, fmt.Sprintf("question %q: a score question's criteria is a required ordered array of level descriptions", name))
				return
			}
		default:
			writeErr(w, http.StatusUnprocessableEntity, fmt.Sprintf("question %q: unknown type %q (want noul, choice or score)", name, spec.Type))
			return
		}
		qs[i] = q
	}
	if !lm.enter(w, r, admissionRecord{}, s.haltState) {
		return
	}
	defer lm.exit()
	res, err := lm.clef.Decide(r.Context(), body)
	if err != nil {
		if r.Context().Err() != nil {
			return // the client left
		}
		var re *clef.RequestError
		if errors.As(err, &re) {
			writeErr(w, http.StatusUnprocessableEntity, err.Error())
			return
		}
		writeServerErr(w, err.Error())
		return
	}
	if len(res.Answers) != len(qs) {
		writeServerErr(w, fmt.Sprintf("the model answered %d questions for %d asked", len(res.Answers), len(qs)))
		return
	}
	var answers orderedJSON
	for i, q := range qs {
		a, err := clefAnswer(q, res.Answers[i])
		if err != nil {
			writeServerErr(w, fmt.Sprintf("question %q: %v", q.name, err))
			return
		}
		answers = append(answers, orderedField{q.name, a})
	}
	gi := map[string]any{"route": routeClef, "calibrated": false, "confidence": clefConfidence, "note": clefNote(),
		"state_tokens_truncated": res.StateTruncated}
	writeJSON(w, http.StatusOK, map[string]any{
		"model":   lm.name,
		"answers": answers,
		// A decision decodes nothing: every prompt token is input, and no token is generated.
		"usage":   map[string]int{"input_tokens": res.InputTokens, "output_tokens": 0},
		"goinfer": gi,
	})
}

func clefNote() string {
	return "Cloudflare's Clef joint decision model: one backbone pass over the whole record answers every question. Probabilities are the model's own, " +
		"uncalibrated by goinfer; confidence is the top probability (the reference's), not the margin over uniform the label route reports; " +
		"probabilities are rounded to four decimals as the reference rounds them. A state longer than the context is cut to its first tokens " +
		"(goinfer.state_tokens_truncated counts what was cut)."
}

// round4 is Python's round(x, 4) for a probability.
func round4(x float64) float64 { return math.Round(x*1e4) / 1e4 }

// clefAnswer shapes one question's probabilities as the reference's systemone_answer does.
func clefAnswer(q clefQuestion, a clef.Answer) (any, error) {
	byID := make(map[string]float64, len(a.Options))
	for i, id := range a.Options {
		byID[id] = a.Probs[i]
	}
	switch q.typ {
	case decide.KindNoul:
		p, ok := byID["true"]
		if !ok {
			return nil, errors.New("no probability for the option \"true\"")
		}
		return map[string]any{"type": "noul", "noul": round4(p)}, nil
	case decide.KindChoice:
		probs := orderedJSON{}
		best, bestP := "", math.Inf(-1)
		for _, k := range q.keys {
			p, ok := byID[k]
			if !ok {
				return nil, fmt.Errorf("no probability for the option %q", k)
			}
			if p > bestP { // strictly greater: the first maximum in request order wins, as Python's max() does
				best, bestP = k, p
			}
			probs = append(probs, orderedField{k, round4(p)})
		}
		return map[string]any{"type": "choice", "choice": best, "confidence": round4(bestP), "probabilities": probs}, nil
	default: // score
		probs, legend := orderedJSON{}, orderedJSON{}
		expected, top := 0.0, math.Inf(-1)
		for i := range q.levels {
			level := fmt.Sprint(i)
			p, ok := byID[level]
			if !ok {
				return nil, fmt.Errorf("no probability for the level %q", level)
			}
			expected += float64(i) * p
			top = math.Max(top, p)
			probs = append(probs, orderedField{level, round4(p)})
			legend = append(legend, orderedField{level, json.RawMessage(bytes.TrimSpace(q.levels[i]))})
		}
		return map[string]any{"type": "score", "score": round4(expected), "confidence": round4(top), "legend": legend, "probabilities": probs}, nil
	}
}
