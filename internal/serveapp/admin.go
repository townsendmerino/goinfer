package serveapp

import (
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Dynamic model load/unload, gated behind --allow-admin: loading an attacker-supplied path is RCE-adjacent, so it is off
// by default (403 when off). Unload unpublishes the model immediately, then DRAINS in-flight requests before freeing its
// native memory (purego has no ARC or finalizers, so GC never reclaims it); see handleAdminUnload and
// docs/completed/task-admin-unload-drain.md. The drain snapshots warm KV, and the response is 200 (freed) or 202
// (draining) per the wait.

// adminCancelReq is the body of POST /admin/generations/{id}/cancel. reason is required so a cancelled generation's
// finish_reason and log line always say WHY, not just THAT (docs/tasks/task-halt-2026-09.md: every halt and cancel is
// loud and attributed).
type adminCancelReq struct {
	Reason string `json:"reason"`
}

// handleAdminGenerationsList lists every in-flight generation. s.gens is its own registry, independent of model
// load/unload, so no liveness or regMu interaction is needed.
func (s *server) handleAdminGenerationsList(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"generations": s.gens.list()})
}

// handleAdminGenerationCancel cancels one generation by id. Cancelling an id that has already finished (or never
// existed) is 200 either way, since either way the generation is stopped, which is what the caller asked for;
// found:false tells the two apart for an operator or test that cares.
func (s *server) handleAdminGenerationCancel(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	var req adminCancelReq
	if !decodeJSON(w, r, &req) {
		return
	}
	if req.Reason == "" {
		writeErr(w, http.StatusBadRequest, "reason is required")
		return
	}
	found := s.gens.cancel(id, req.Reason)
	fmt.Fprintf(os.Stderr, "admin: cancel %s (found=%v reason=%q)\n", id, found, req.Reason)
	writeJSON(w, http.StatusOK, map[string]any{"id": id, "found": found})
}

type adminLoadReq struct {
	Name  string `json:"name"` // served id (default: file/dir basename)
	Path  string `json:"path"` // .gguf / .giw / HF dir
	Quant string `json:"quant"`
	Lora  string `json:"lora"`
}

type adminUnloadReq struct {
	Name string `json:"name"`
}

// requireAdmin is the TCP-listener gate for /admin/*. It is chain-level (alongside auth/haltGate/inf in main.go), not a
// handler-internal check, so it can be left off when the same handlers are registered on the admin socket, where the
// socket's file permissions (mode 0600) are the auth and -allow-admin has no meaning to enforce
// (docs/tasks/task-halt-2026-09.md).
func (s *server) requireAdmin(h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if !s.cfg.allowAdmin {
			writeErr(w, http.StatusForbidden, "admin API disabled (start serve with --allow-admin)")
			return
		}
		h(w, r)
	}
}

// registerAdminRoutes registers every /admin/* route (model load/unload, generation list/cancel/status, halt/resume)
// onto mux, each wrapped with wrap. main.go calls it for the TCP mux (wrap = auth+requireAdmin) or, when -admin-socket
// is set, for the admin socket (wrap = identity: the socket's file permissions are the auth); the socket replaces the
// TCP registration rather than adding to it (docs/tasks/task-halt-2026-09.md).
func registerAdminRoutes(mux *http.ServeMux, s *server, textCap int64, wrap func(http.HandlerFunc) http.HandlerFunc) {
	mux.HandleFunc("POST /admin/models/load", wrap(maxBytes(textCap, s.handleAdminLoad)))
	mux.HandleFunc("POST /admin/models/unload", wrap(maxBytes(textCap, s.handleAdminUnload)))
	mux.HandleFunc("GET /admin/generations", wrap(s.handleAdminGenerationsList))
	mux.HandleFunc("POST /admin/generations/{id}/cancel", wrap(maxBytes(textCap, s.handleAdminGenerationCancel)))
	mux.HandleFunc("POST /admin/halt", wrap(maxBytes(textCap, s.handleAdminHalt)))
	mux.HandleFunc("POST /admin/resume", wrap(s.handleAdminResume))
	// GET /admin/status exists because the one-word `serve status` CLI needs a status route on a socket that serves ONLY
	// /admin/*; /health is deliberately not registered there. It is admin-scoped rather than reusing /health so it is
	// reachable wherever /admin/* is.
	mux.HandleFunc("GET /admin/status", wrap(s.handleAdminStatus))
}

// handleAdminStatus is GET /admin/status: the current halt state (mirroring /health's halted/halt_reason/halt_at) plus
// the live generation count, so `serve status` has something to report without /health on the same listener.
func (s *server) handleAdminStatus(w http.ResponseWriter, r *http.Request) {
	hi := s.haltState()
	resp := map[string]any{"halted": hi != nil, "generations_inflight": s.gens.count()}
	if hi != nil {
		resp["halt_reason"] = hi.reason
		resp["halt_trigger"] = hi.trigger
		resp["halt_at"] = hi.at
	}
	// The resident batcher's counters per model running several generations at once (the numbers serve prints at
	// shutdown), so a harness can difference them across one cell rather than across the server's whole life: load,
	// warm-up and first-use pipeline compiles excluded (docs/tasks/task-concurrency-2026-09.md).
	rb := map[string]any{}
	for _, lm := range s.modelList() {
		if lm.model != nil && lm.model.ResidentConcurrency() > 1 {
			rb[lm.name] = lm.model.ResidentBatchStats()
		}
	}
	if len(rb) > 0 {
		resp["resident_batch"] = rb
	}
	writeJSON(w, http.StatusOK, resp)
}

// handleAdminLoad loads a new generative model into the registry.
func (s *server) handleAdminLoad(w http.ResponseWriter, r *http.Request) {
	var req adminLoadReq
	if !decodeJSON(w, r, &req) {
		return
	}
	if req.Path == "" {
		writeErr(w, http.StatusBadRequest, "path is required")
		return
	}
	name := req.Name
	if name == "" {
		name = strings.TrimSuffix(filepath.Base(req.Path), ".gguf")
	}
	s.regMu.RLock()
	_, dup := s.models[name]
	s.regMu.RUnlock()
	if dup {
		writeErr(w, http.StatusConflict, fmt.Sprintf("model %q already loaded", name))
		return
	}
	// Per-request quant/lora override the global defaults; everything else
	// (backend, kv-sessions, session-dir) comes from the server config.
	c := s.cfg
	// The request is the authority for its own load. explicitQuant() drives the .giw baked-quant mismatch check and
	// reads cfg.load.QuantSet, which the CLI set and which says nothing about THIS request. Inheriting it is wrong both
	// ways: with no CLI --quant and an admin request for int8, the bundle's baked int4 would load silently; with a CLI
	// --quant and an admin request naming none, the request would be checked against a quant it never named and
	// rejected. So QuantSet is true only when the request names a quant; otherwise the CLI value stays a default, not an
	// explicit choice to conflict with.
	if req.Quant != "" {
		c.load.Quant, c.load.QuantSet = req.Quant, true
	} else {
		c.load.QuantSet = false
	}
	if req.Lora != "" {
		c.load.LoRA = req.Lora
	}
	lm, err := loadDecoder(r.Context(), modelSpec{name: name, path: req.Path}, c)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	if !s.publishLoaded(lm) {
		writeErr(w, http.StatusConflict, fmt.Sprintf("model %q already loaded", lm.name))
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"id": lm.name, "object": "model", "status": "loaded"})
}

// publishLoaded makes a freshly loaded model routable, or, when another load of the same name published first, closes it
// and reports false. Shared by the admin load and the web UI's load (webui.go) so the close-on-race and the session
// restore live in one place.
func (s *server) publishLoaded(lm *loadedModel) bool {
	if line := lm.setConcurrency(s.cfg); line != "" { // before the model is routable
		fmt.Fprintf(os.Stderr, "%q %s\n", lm.name, line)
	}
	s.regMu.Lock()
	if _, dup := s.models[lm.name]; dup { // raced another load of the same name
		s.regMu.Unlock()
		// The LOSER of the race holds a fully loaded model (resident device memory, the .giw mmap, an uploaded block
		// drafter), and purego installs no finalizers, so nothing reclaims those unless they are closed here. Close
		// here rather than in a defer: it must NOT run on the success path, where the registry now owns the model.
		//
		// Safe to Close unconditionally: loadDecoder always builds a fresh decoder.Load, so this entry shares its
		// weights with no other registry entry, and retainLocked has not run for it.
		lm.model.Close()
		lm.closeEntryNatives()
		return false
	}
	s.models[lm.name] = lm
	s.retainLocked(lm.model) // one more registry entry backed by this *decoder.Model (liveness refs)
	s.regMu.Unlock()
	if s.cfg.sessionDir != "" && s.cfg.kvSessions > 0 {
		// The model is now published, so a request can already acquire it. lm.sessMu guards the sessionLRU
		// (drive/driveVL take it around their own sessions.acquire; docs/tasks/task-work-queue-2026-09.md) and load does
		// not take it, so hold it here to serialize this not-goroutine-safe restore against a handler that reaches the
		// LRU first.
		lm.sessMu.Lock()
		lm.sessions.load(sessionSubdir(s.cfg.sessionDir, lm.fp))
		lm.sessMu.Unlock()
	}
	return true
}

// handleAdminUnload drops a model from the registry and DRAINS before freeing its native memory.
//
// Calling lm.model.Close() straight after the registry delete is a use-after-free: a request past pick() but not yet at
// enter() holds the *lm pointer and touches lm.model in its preamble (tokenize/prepare) with no lock, so a Close there
// frees weights mid-request (on CUDA, a driver SIGSEGV). Instead every in-flight holder takes a per-model liveness RLock
// via withModel (spanning the preamble and the generation), and unload waits that lock out before closing. See
// docs/completed/task-admin-unload-drain.md and the reciprocal note at resident.Close.
//
// Two phases, in unloadByName (shared with the web route). Phase 1 (under regMu): unpublish the entry and decide
// last-ownership, delete-before-decide, so two concurrent sibling unloads cannot both decline (releaseLocked). Phase 2
// (startDrain, detached): drain in-flight holders, checkpoint the settled KV, close the entry's private natives, close
// the shared model iff last owner. The response is a bounded wait: 200 (freed) if the drain completes within
// -unload-drain-wait, else 202 with the drain continuing detached; the model is unroutable immediately either way, and
// /health lists what is still draining. ?wait=false skips straight to 202.
func (s *server) handleAdminUnload(w http.ResponseWriter, r *http.Request) {
	var req adminUnloadReq
	if !decodeJSON(w, r, &req) {
		return
	}
	wait := s.cfg.unloadDrainWait
	if r.URL.Query().Get("wait") == "false" {
		wait = 0
	}
	status, body, ok := s.unloadByName(req.Name, wait)
	if !ok {
		s.modelNotFound(w, req.Name)
		return
	}
	writeJSON(w, status, body)
}

// unloadByName is handleAdminUnload's two phases (see its doc comment for the drain design), shared with the web route
// (webui.go) so a model unloaded from the page is unpublished and drained exactly as one unloaded through /admin: the
// two routes differ only in which names a caller may name, not in what happens once one is accepted.
func (s *server) unloadByName(name string, wait time.Duration) (status int, body map[string]any, ok bool) {
	s.regMu.Lock()
	lm, found := s.models[name]
	if !found {
		s.regMu.Unlock()
		return 0, nil, false
	}
	delete(s.models, name)                // unpublish: no new request can resolve it
	ml, last := s.releaseLocked(lm.model) // decrement refs + last-owner decision (delete-before-decide)
	s.regMu.Unlock()

	// Detached drain-and-close: waits out in-flight holders, checkpoints KV, frees native memory.
	// It owns the free and runs to completion regardless of this request (a disconnected caller
	// must not orphan the model) and regardless of shutdown (bare goroutine, never joined).
	done := s.startDrain(lm, ml, last)

	select {
	case <-done:
		return http.StatusOK, map[string]any{"id": name, "status": "unloaded", "freed": last}, true
	case <-time.After(wait):
		return http.StatusAccepted, map[string]any{
			"id": name, "status": "unloading", "freed": false,
			"note": "native memory is released as in-flight requests finish; poll GET /health (draining) until this model no longer appears",
		}, true
	}
}
