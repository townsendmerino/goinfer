package serveapp

import (
	"fmt"
	"net/http"
	"path/filepath"
	"strings"
)

// Dynamic model load/unload (Track B Inc3), mirroring llama.cpp / mistral.rs
// admin conventions but kept small. Gated behind --allow-admin: loading an
// attacker-supplied path is RCE-adjacent, so it is off by default (403 when off).
// Unload refuses a model that is mid-generation (the model's own mutex, try-lock
// → 409) and snapshots its warm KV first. Go's GC + mmap mean RSS shrinks lazily
// after unload, not immediately.

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
	lm, err := loadDecoder(modelSpec{name: name, path: req.Path}, c)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	s.regMu.Lock()
	if _, dup := s.models[lm.name]; dup { // raced another load of the same name
		s.regMu.Unlock()
		writeErr(w, http.StatusConflict, fmt.Sprintf("model %q already loaded", lm.name))
		return
	}
	s.models[lm.name] = lm
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

// handleAdminUnload drops a model from the registry (refusing if it is busy).
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
	if !lm.mu.TryLock() { // a generation holds the mutex — refuse rather than wait
		s.regMu.Unlock()
		writeErr(w, http.StatusConflict, fmt.Sprintf("model %q is busy (generating); retry", req.Name))
		return
	}
	delete(s.models, req.Name)
	s.regMu.Unlock()
	// Snapshot warm KV before the model goes unreferenced.
	if s.cfg.sessionDir != "" && s.cfg.kvSessions > 0 {
		_ = lm.sessions.save(sessionSubdir(s.cfg.sessionDir, lm.fp))
	}
	lm.mu.Unlock()
	writeJSON(w, http.StatusOK, map[string]any{"id": req.Name, "status": "unloaded"})
}
