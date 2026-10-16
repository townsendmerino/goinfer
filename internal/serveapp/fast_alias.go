package serveapp

import (
	"encoding/json"
	"net/http"
	"strings"
)

// The fast alias: a served model N is also reachable as "N-fast". Same weights, same chat template, the same KV slot; a request
// under the alias runs with thinking off and its output capped at -fast-max-tokens. It exists for the side requests an agent
// sends for its own bookkeeping (a session title, a one-line summary): they carry no use for reasoning, and a reasoning model
// spent 18 to 41 s of GPU time thinking its way to a title (docs/measurements/side-requests-2026-10/). `connect` points an
// agent's small-model setting at the alias; the main model's settings are untouched.
//
// The alias is not a served entry: it is not in /v1/models as a model of its own (a client that lists models would read a second
// model, and connect's "the only generative model" default would break). The model's own entry advertises it in fast_model and
// fast_max_tokens. A model actually served under the alias's name wins over it. -fast-max-tokens 0 turns the alias off.

const fastSuffix = "-fast"

// fastTarget is the served model name the alias name stands for, ok false when name is not an alias: the feature is off, name
// does not end in the suffix, a model is served under name itself, or the model it names is not served or has no chat template.
func (s *server) fastTarget(name string) (string, bool) {
	if s.cfg.fastMaxTokens <= 0 || !strings.HasSuffix(name, fastSuffix) {
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
	if s.cfg.fastMaxTokens <= 0 {
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

// clampFast lowers *p to at most max, or sets it to max when unset.
func clampFast(p **int, max int) {
	if *p == nil || **p > max {
		v := max
		*p = &v
	}
}

func (r *chatReq) applyFast(max int) {
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
	clampFast(&r.MaxTokens, max)
	if r.MaxCompletionTokens != nil {
		clampFast(&r.MaxCompletionTokens, max)
	}
}

func (r *completionReq) applyFast(max int) {
	clampFast(&r.MaxTokens, max)
	if r.MaxCompletionTokens != nil {
		clampFast(&r.MaxCompletionTokens, max)
	}
}

func (r *responseReq) applyFast(max int) {
	r.Reasoning = &struct {
		Effort string `json:"effort"`
	}{Effort: "none"}
	clampFast(&r.MaxOutputTokens, max)
}

func (r *anthropicReq) applyFast(max int) {
	r.Thinking = &struct {
		Type         string `json:"type"`
		BudgetTokens int    `json:"budget_tokens"`
	}{Type: "disabled"}
	clampFast(&r.MaxTokens, max)
}
