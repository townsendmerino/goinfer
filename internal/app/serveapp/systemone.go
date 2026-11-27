package serveapp

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"

	"github.com/townsendmerino/goinfer/internal/app/serveapp/systemone"
	"github.com/townsendmerino/goinfer/internal/decide"
)

type systemOneItem = systemone.Item

var (
	systemOneRequest   = systemone.ParseQuestion
	typesafeConfidence = systemone.Confidence
	jsonText           = systemone.JSONText
	orderedObject      = systemone.OrderedObject
)

func (s *server) handleSystemOne(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeBodyErr(w, "read body", err)
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
	if s.isClefModel(req.Model) {
		s.withModel(w, req.Model, func(lm *loadedModel) { s.serveClef(w, r, lm, body, req.Questions) })
		return
	}
	items, err := systemone.ParseRequest(req.State, req.Questions)
	if err != nil {
		writeErr(w, http.StatusUnprocessableEntity, err.Error())
		return
	}
	s.withModel(w, req.Model, func(lm *loadedModel) { s.serveSystemOne(w, r, lm, items) })
}

func (s *server) serveSystemOne(w http.ResponseWriter, r *http.Request, lm *loadedModel, items []systemOneItem) {
	if lm.adapter != "" {
		writeErr(w, http.StatusBadRequest, "decisions are not supported on a compute-time adapter entry; ask the base model")
		return
	}
	d, err := lm.getDecider(s.cfg)
	if err != nil {
		writeErr(w, http.StatusBadRequest, "this model cannot answer decisions: "+err.Error())
		return
	}
	for _, it := range items {
		if err := d.Validate(it.Req); err != nil {
			writeErr(w, http.StatusUnprocessableEntity, fmt.Sprintf("question %q: %v", it.Name, err))
			return
		}
	}
	if err := lm.promptTooLargeForContext(systemone.MaxTextBytes(items)); err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	if !lm.enter(w, r, admissionRecord{}, s.haltState) {
		return
	}
	defer lm.exit()
	reqs := make([]decide.Request, len(items))
	for i, it := range items {
		reqs[i] = it.Req
	}
	results, err := d.DecideMany(r.Context(), reqs)
	if err != nil {
		if r.Context().Err() != nil {
			return
		}
		writeServerErr(w, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, systemone.FormatResponse(lm.name, d.Route(), d.Template(), items, results))
}

// getDecider builds this model's decider once: its label tokens resolved, the -decisions-template, and the
// -decisions-calibration temperatures if any.
func (lm *loadedModel) getDecider(cfg config) (*decide.Decider, error) {
	lm.decisions.deciderOnce.Do(func() {
		if lm.model == nil || lm.tk == nil {
			lm.decisions.deciderErr = errors.New("not a generative model")
			return
		}
		tmpl := cfg.decisionsTemplate
		if tmpl == "" {
			tmpl = decide.TemplateChat
		}
		var cal *decide.Calibration
		if cfg.decisionsCal != "" {
			if cal, lm.decisions.deciderErr = decide.LoadCalibration(cfg.decisionsCal); lm.decisions.deciderErr != nil {
				return
			}
		}
		if lm.decisions.head != nil {
			lm.decisions.decider, lm.decisions.deciderErr = decide.New(decide.NewPlainTokenizer(lm.tk), nil,
				decide.Options{Template: lm.decisions.head.Template, Head: lm.decisions.head, Hidden: decide.ModelHidden(lm.model), HiddenMany: decide.ModelHiddenMany(lm.model)})
			return
		}
		lm.decisions.decider, lm.decisions.deciderErr = decide.New(decide.NewPlainTokenizer(lm.tk), decide.ModelPrefill(lm.model),
			decide.Options{Template: tmpl, Calibration: cal, PrefillMany: decide.ModelPrefillMany(lm.model)})
	})
	return lm.decisions.decider, lm.decisions.deciderErr
}
