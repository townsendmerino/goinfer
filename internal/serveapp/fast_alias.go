package serveapp

import (
	"encoding/json"
	"net/http"
	"strings"
)

// The fast alias: a served model N is also reachable as "N-fast". Same weights, same chat template, the same KV slot; a request
// under the alias runs with thinking off and nothing else different: its output limit is whatever the request, or the main
// model's own default, gives. It exists for the side requests an agent sends for its own bookkeeping (a session title): they
// carry no use for reasoning, and a reasoning model spent 18 to 41 s of GPU time thinking its way to a title
// (docs/measurements/side-requests-2026-10/). `connect` points an agent's small-model setting at the alias; the main model's
// settings are untouched. There is no output cap on purpose: a summary cut short is worse than a slow one.
//
// The alias is not a served entry: it is not in /v1/models as a model of its own (a client that lists models would read a second
// model, and connect's "the only generative model" default would break). The model's own entry advertises it in fast_model.
// A model actually served under the alias's name wins over it. -fast-alias=false turns the alias off.

const fastSuffix = "-fast"

// fastTarget is the served model name the alias name stands for, ok false when name is not an alias: the feature is off, name
// does not end in the suffix, a model is served under name itself, or the model it names is not served or has no chat template.
func (s *server) fastTarget(name string) (string, bool) {
	if !s.cfg.fastAlias || !strings.HasSuffix(name, fastSuffix) {
		return "", false
	}
	base := strings.TrimSuffix(name, fastSuffix)
	s.regMu.RLock()
	defer s.regMu.RUnlock()
	if _, taken := s.models[name]; taken {
		return "", false
	}
	if lm := s.models[base]; lm != nil && lm.tmpl != nil {
		return base, true
	}
	return "", false
}

// fastAliasOf is the alias a served model advertises, "" when it has none.
func (s *server) fastAliasOf(model string) string {
	if !s.cfg.fastAlias {
		return ""
	}
	if _, ok := s.fastTarget(model + fastSuffix); ok {
		return model + fastSuffix
	}
	return ""
}

// fastRoute resolves a decoded request's model name: when it is an alias the name becomes the served model's and fast reports
// true, and the caller applies the route's rewrite (applyFast). The request log marks the request.
func (s *server) fastRoute(r *http.Request, model *string) (fast bool) {
	base, ok := s.fastTarget(*model)
	if !ok {
		return false
	}
	*model = base
	if tr := traceFrom(r.Context()); tr != nil {
		tr.mu.Lock()
		tr.fast = true
		tr.mu.Unlock()
	}
	return true
}

func (r *chatReq) applyFast() {
	if r.ChatTemplateKwargs == nil {
		r.ChatTemplateKwargs = map[string]json.RawMessage{}
	} else { // the request's own map is not shared with anything else, but do not edit a caller's value in place
		m := make(map[string]json.RawMessage, len(r.ChatTemplateKwargs)+1)
		for k, v := range r.ChatTemplateKwargs {
			m[k] = v
		}
		r.ChatTemplateKwargs = m
	}
	r.ChatTemplateKwargs["enable_thinking"] = json.RawMessage("false")
	r.ThinkingTokenBudget = nil
}

func (r *responseReq) applyFast() {
	r.Reasoning = &struct {
		Effort string `json:"effort"`
	}{Effort: "none"}
}

func (r *anthropicReq) applyFast() {
	r.Thinking = &struct {
		Type         string `json:"type"`
		BudgetTokens int    `json:"budget_tokens"`
	}{Type: "disabled"}
}
