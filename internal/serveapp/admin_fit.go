package serveapp

import (
	"fmt"
	"net/http"
	"runtime/debug"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/modelload"
)

// handleAdminFit is POST /admin/models/fit: what POST /admin/models/load would do with the same body, without loading.
// It resolves the path as the load does (modelload.Plan: the same route choice, without fetching a reference or building a
// .giw) and runs the load's own memory check on it (decoder.FitQuery), priced against the memory available now. When only
// a step it does not take could tell (a reference, a sidecar .giw not built yet), "fits" is null and "unknown" says why.
// docs/server.md, "Admin: would this load fit?".
func (s *server) handleAdminFit(w http.ResponseWriter, r *http.Request) {
	var req adminLoadReq
	if !decodeJSON(w, r, &req) {
		return
	}
	if req.Path == "" {
		writeErr(w, http.StatusBadRequest, "path is required")
		return
	}
	resp, err := s.fitAnswer(req)
	// Reading a .gguf's header parses its whole metadata, tokenizer included, into the heap (tens of MB on a 150k
	// vocabulary). Hand it back before answering, so a caller that measures after the reply finds the footprint it left.
	debug.FreeOSMemory()
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

// fitAnswer is the route's body, or the error the load would return before its memory check.
func (s *server) fitAnswer(req adminLoadReq) (map[string]any, error) {
	c := s.adminLoadConfig(req)
	spec := modelSpec{path: req.Path}
	opts := spec.options(c)
	if err := opts.Validate(); err != nil {
		return nil, fmt.Errorf("--model %q: %w", spec.path, err)
	}
	if c.drafter != "" {
		opts.Drafter = true
	}
	p, err := modelload.Plan(modelload.Request{Spec: spec.path, Opts: opts, DirectLoad: c.load.DirectLoad, ExplicitQuant: spec.explicitQuant(c)})
	if err != nil {
		return nil, err
	}
	resp := map[string]any{"path": p.Source, "load_path": p.LoadPath, "backend": p.Opts.Backend, "quant": p.Opts.Quant,
		"fits": nil, "unknown": p.Unknown}
	if len(p.Notes) > 0 {
		resp["notes"] = p.Notes
	}
	if p.Unknown != "" {
		if p.Build != "" {
			resp["build"] = p.Build
		}
		return resp, nil
	}
	a, err := decoder.FitQuery(p.LoadPath, p.Opts)
	if err != nil {
		return nil, err
	}
	resp["fits"] = a.Fits
	resp["guard"] = a.Guard
	resp["ctx"], resp["ctx_pinned"], resp["ctx_capped"], resp["max_ctx"] = a.Ctx, a.CtxPinned, a.CtxCapped, a.MaxCtx
	resp["need_bytes"], resp["budget_bytes"], resp["available_bytes"] = a.NeedBytes, a.BudgetBytes, a.AvailBytes
	resp["weight_bytes"], resp["kv_bytes"] = a.WeightBytes, a.KVBytes
	resp["decline"] = a.Decline
	if a.Unpriced != "" {
		resp["unpriced"] = a.Unpriced
	}
	// A dense .gguf the fit guard refuses is not refused by the load: it retries with weight streaming, which builds a .giw.
	if !a.Fits && p.RetryStreams && a.Streamable {
		resp["retry"] = "the load retries with -stream-weights (it builds a .giw first) instead of refusing; --fit=off keeps the refusal"
	}
	if a.ResidentCtx > 0 {
		resp["resident_ctx"] = a.ResidentCtx
	}
	if a.DecodePath != "" {
		resp["decode_path"] = a.DecodePath
	} else {
		resp["decode_path"], resp["decode_path_unknown"] = "unknown", a.DecodeUnknown
	}
	return resp, nil
}
