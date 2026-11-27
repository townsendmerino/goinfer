package serveapp

import (
	"encoding/json"
	"errors"
	"net/http"

	"github.com/townsendmerino/goinfer/internal/app/serveapp/systemone"
	"github.com/townsendmerino/goinfer/internal/clef"
)

const (
	routeClef      = systemone.RouteClef
	clefConfidence = systemone.ClefConfidence
)

type clefQuestion = systemone.ClefQuestion

var (
	isClefDir        = systemone.IsClefDir
	clefQuantRefusal = systemone.ClefQuantRefusal
	clefAnswer       = systemone.ClefAnswer
	round4           = systemone.Round4
)

func (s *server) isClefModel(name string) bool {
	lm, release := s.resolveAndLock(name)
	if lm == nil {
		return false
	}
	defer release()
	return lm.decisions.clef != nil
}

func (s *server) serveClef(w http.ResponseWriter, r *http.Request, lm *loadedModel, body []byte, rawQuestions json.RawMessage) {
	if lm.decisions.clef == nil {
		writeErr(w, http.StatusConflict, "this model is not a Clef model; the entry changed while the request was admitted")
		return
	}
	if lm.adapter != "" {
		writeErr(w, http.StatusBadRequest, "decisions are not supported on a compute-time adapter entry; ask the base model")
		return
	}
	qs, err := systemone.ParseClefQuestions(rawQuestions)
	if err != nil {
		writeErr(w, http.StatusUnprocessableEntity, err.Error())
		return
	}
	if !lm.enter(w, r, admissionRecord{}, s.haltState) {
		return
	}
	defer lm.exit()
	res, err := lm.decisions.clef.Decide(r.Context(), body)
	if err != nil {
		if r.Context().Err() != nil {
			return
		}
		if _, ok := errors.AsType[*clef.RequestError](err); ok {
			writeErr(w, http.StatusUnprocessableEntity, err.Error())
			return
		}
		writeServerErr(w, err.Error())
		return
	}
	payload, err := systemone.FormatClefResponse(lm.name, qs, res)
	if err != nil {
		writeServerErr(w, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, payload)
}
