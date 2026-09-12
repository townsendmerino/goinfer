package serveapp

import (
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Dynamic model load/unload (Track B Inc3), mirroring llama.cpp / mistral.rs
// admin conventions but kept small. Gated behind --allow-admin: loading an
// attacker-supplied path is RCE-adjacent, so it is off by default (403 when off).
// Unload unpublishes the model immediately, then DRAINS in-flight requests before
// freeing its native memory (purego has no ARC / finalizers, so GC never reclaims
// it) — see handleAdminUnload and docs/completed/task-admin-unload-drain.md. It snapshots warm
// KV as part of the drain, and reports 200 (freed) or 202 (draining) per the wait.

// adminCancelReq is the body of POST /admin/generations/{id}/cancel (K1,
// docs/task-halt-2026-09.md). reason is required so a cancelled generation's finish_reason and
// log line always say WHY, not just THAT — "every halt/cancel is loud and attributed" is this
// doc's own ground rule.
type adminCancelReq struct {
	Reason string `json:"reason"`
}

// handleAdminGenerationsList lists every in-flight generation (K1). No liveness/regMu
// interaction needed — s.gens is its own registry, independent of model load/unload.
func (s *server) handleAdminGenerationsList(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"generations": s.gens.list()})
}

// handleAdminGenerationCancel cancels one generation by id (K1). Cancelling an id that has
// already finished (or never existed) is reported the same way either finding leaves the
// generation stopped, which is what the caller asked for — 200 either way, with found:false
// distinguishing them for an operator or test that cares.
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

// requireAdmin is the TCP-listener gate for /admin/* — chain-level (alongside auth/haltGate/inf
// in main.go), not a handler-internal check, so it can be left off entirely when registering the
// same handlers on the admin socket (K5, docs/task-halt-2026-09.md): there, the socket's file
// permissions (mode 0600) are the auth, and -allow-admin has no TCP-listener meaning to enforce.
func (s *server) requireAdmin(h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if !s.cfg.allowAdmin {
			writeErr(w, http.StatusForbidden, "admin API disabled (start serve with --allow-admin)")
			return
		}
		h(w, r)
	}
}

// registerAdminRoutes registers every /admin/* route — model load/unload, K1's generation
// list/cancel/status, K2's halt/resume — onto mux, each wrapped with wrap. main.go calls this
// twice: once for the TCP mux (wrap = auth+requireAdmin, gated by -api-key/-allow-admin), and
// once for the admin socket when -admin-socket is set (K5, docs/task-halt-2026-09.md; wrap =
// identity there — the socket's file permissions are the auth, and it replaces the TCP
// registration rather than adding to it, so the two never both run for the same server).
func registerAdminRoutes(mux *http.ServeMux, s *server, textCap int64, wrap func(http.HandlerFunc) http.HandlerFunc) {
	mux.HandleFunc("POST /admin/models/load", wrap(maxBytes(textCap, s.handleAdminLoad)))
	mux.HandleFunc("POST /admin/models/unload", wrap(maxBytes(textCap, s.handleAdminUnload)))
	mux.HandleFunc("GET /admin/generations", wrap(s.handleAdminGenerationsList))
	mux.HandleFunc("POST /admin/generations/{id}/cancel", wrap(maxBytes(textCap, s.handleAdminGenerationCancel)))
	mux.HandleFunc("POST /admin/halt", wrap(maxBytes(textCap, s.handleAdminHalt)))
	mux.HandleFunc("POST /admin/resume", wrap(s.handleAdminResume))
	// GET /admin/status: not asked for outside K5, but K5's one-word CLI needs a "status"
	// subcommand that works over a socket serving ONLY /admin/* (health.go's /health is
	// deliberately not registered there — it's a /v1-shaped route with no admin content). Admin
	// scoped rather than reusing /health so it is reachable wherever /admin/* is.
	mux.HandleFunc("GET /admin/status", wrap(s.handleAdminStatus))
}

// handleAdminStatus is GET /admin/status: the current halt state (mirroring /health's own
// halted/halt_reason/halt_at fields) plus K1's live generation count, so `serve status` (K5) has
// something to report without needing /health on the same listener.
func (s *server) handleAdminStatus(w http.ResponseWriter, r *http.Request) {
	hi := s.haltState()
	resp := map[string]any{"halted": hi != nil, "generations_inflight": s.gens.count()}
	if hi != nil {
		resp["halt_reason"] = hi.reason
		resp["halt_trigger"] = hi.trigger
		resp["halt_at"] = hi.at
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
	// N-19: quantSet, not just quant. explicitQuant() drives the .giw baked-quant mismatch
	// check, and it reads cfg.quantSet — which was inherited from the CLI and said nothing
	// about THIS request. Both directions were wrong:
	//
	//   no CLI --quant + admin asks int8  → quantSet false → no check → the bundle's baked
	//                                       int4 loads silently under an int8 request
	//   CLI --quant given + admin asks nothing → quantSet true → this request is checked
	//                                       against a quant it never named, and is rejected
	//
	// The admin request is the authority for its own load: if it names a quant that is the
	// explicit choice, and if it does not, the CLI value stays as a DEFAULT but is not an
	// explicit choice to conflict with.
	if req.Quant != "" {
		c.quant, c.quantSet = req.Quant, true
	} else {
		c.quantSet = false
	}
	if req.Lora != "" {
		c.lora = req.Lora
	}
	lm, err := loadDecoder(r.Context(), modelSpec{name: name, path: req.Path}, c)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	s.regMu.Lock()
	if _, dup := s.models[lm.name]; dup { // raced another load of the same name
		s.regMu.Unlock()
		// M-24: the LOSER of the race holds a fully loaded model — resident device
		// memory, the .giw mmap, an uploaded block drafter — and refusing to publish it
		// used to just drop the pointer. purego installs no finalizers, so nothing ever
		// reclaims those; that is the whole reason the drain design exists. Close here
		// rather than in a defer: it must NOT run on the success path, where the registry
		// now owns the model.
		//
		// Safe to Close unconditionally: loadDecoder always builds a fresh decoder.Load,
		// so this entry shares its weights with no other registry entry, and retainLocked
		// has not run for it — it was never published.
		lm.model.Close()
		lm.closeEntryNatives()
		writeErr(w, http.StatusConflict, fmt.Sprintf("model %q already loaded", lm.name))
		return
	}
	s.models[lm.name] = lm
	s.retainLocked(lm.model) // one more registry entry backed by this *decoder.Model (liveness refs)
	s.regMu.Unlock()
	if s.cfg.sessionDir != "" && s.cfg.kvSessions > 0 {
		// The model is now published, so a request can already acquire it. lm.mu is the
		// sessionLRU's guard (every handler holds it via enter across sessions.acquire), and
		// load doesn't take it — so hold it here to serialize this not-goroutine-safe restore
		// against a handler that reaches the LRU first, instead of racing the map (M5).
		lm.mu.Lock()
		lm.sessions.load(sessionSubdir(s.cfg.sessionDir, lm.fp))
		lm.mu.Unlock()
	}
	writeJSON(w, http.StatusOK, map[string]any{"id": lm.name, "object": "model", "status": "loaded"})
}

// handleAdminUnload drops a model from the registry and DRAINS before freeing its native memory.
//
// The naive fix — lm.model.Close() straight after the registry delete — is a use-after-free: a
// request past pick() but not yet at enter() holds the *lm pointer and touches lm.model in its
// preamble (tokenize/prepare) with no lock, so a Close there frees weights mid-request (on CUDA, a
// driver SIGSEGV). The safe fix is a DRAIN: every in-flight holder takes a per-model liveness RLock
// via withModel (spanning the preamble and the generation), and unload waits that lock out before
// closing. See docs/completed/task-admin-unload-drain.md and the reciprocal note at resident.Close.
//
// Two phases. Phase 1 (here, under regMu): unpublish the entry and decide last-ownership —
// delete-before-decide, so two concurrent sibling unloads cannot both decline (releaseLocked). Phase
// 2 (startDrain, detached): drain in-flight holders, checkpoint the settled KV, close the entry's
// private natives, close the shared model iff last owner. The response is a bounded wait: 200
// (freed) if the drain completes within -unload-drain-wait, else 202 with the drain continuing
// detached — the model is unroutable immediately either way, and /health lists what is still
// draining. ?wait=false skips straight to 202. (This replaces the old 409-busy, which was only ever
// safe because it never freed anything.)
func (s *server) handleAdminUnload(w http.ResponseWriter, r *http.Request) {
	var req adminUnloadReq
	if !decodeJSON(w, r, &req) {
		return
	}
	s.regMu.Lock()
	lm, ok := s.models[req.Name]
	if !ok {
		s.regMu.Unlock()
		s.modelNotFound(w, req.Name)
		return
	}
	delete(s.models, req.Name)            // unpublish: no new request can resolve it
	ml, last := s.releaseLocked(lm.model) // decrement refs + last-owner decision (delete-before-decide)
	s.regMu.Unlock()

	// Detached drain-and-close: waits out in-flight holders, checkpoints KV, frees native memory.
	// It owns the free and runs to completion regardless of this request (a disconnected admin
	// client must not orphan the model) and regardless of shutdown (bare goroutine, never joined).
	done := s.startDrain(lm, ml, last)

	wait := s.cfg.unloadDrainWait
	if r.URL.Query().Get("wait") == "false" {
		wait = 0
	}
	select {
	case <-done:
		writeJSON(w, http.StatusOK, map[string]any{"id": req.Name, "status": "unloaded", "freed": last})
	case <-time.After(wait):
		writeJSON(w, http.StatusAccepted, map[string]any{
			"id": req.Name, "status": "unloading", "freed": false,
			"note": "native memory is released as in-flight requests finish; poll GET /health (draining) until this model no longer appears",
		})
	}
}
