package serveapp

import (
	"fmt"
	"net/http"
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

type adminLoadReq struct {
	Name  string `json:"name"` // served id (default: file/dir basename)
	Path  string `json:"path"` // .gguf / .giw / HF dir
	Quant string `json:"quant"`
	Lora  string `json:"lora"`
}

type adminUnloadReq struct {
	Name string `json:"name"`
}

func (s *server) adminEnabled(w http.ResponseWriter) bool {
	if !s.cfg.allowAdmin {
		writeErr(w, http.StatusForbidden, "admin API disabled (start serve with --allow-admin)")
		return false
	}
	return true
}

// handleAdminLoad loads a new generative model into the registry.
func (s *server) handleAdminLoad(w http.ResponseWriter, r *http.Request) {
	if !s.adminEnabled(w) {
		return
	}
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
	if req.Quant != "" {
		c.quant = req.Quant
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
	if !s.adminEnabled(w) {
		return
	}
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
