package serveapp

import (
	"maps"
	"net/http"
	"sort"
)

// GET /health — a goinfer-native operator surface, deliberately NOT an OpenAI-compatible one.
//
// WHY IT EXISTS SEPARATELY FROM /v1/models. The resolved decode/prefill paths are also attached to
// each /v1/models entry, where they are a vendor extension: extra keys on a schema somebody else
// owns. The Go, Python and JS OpenAI clients ignore unknown keys, but a strictly-typed decoder in
// another language may reject the whole response, and that would be goinfer breaking a client to
// report a diagnostic. /health carries the same three fields on a payload with no compatibility
// contract at all, so an operator or a batch job never has to choose between reading the resolved
// path and holding to the OpenAI schema.
//
// The fields come from server.pathFields — the same function /v1/models uses — so the two surfaces
// cannot drift apart.
func (s *server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	models := []map[string]any{}
	for _, name := range s.servedNames() {
		e := map[string]any{"id": name}
		maps.Copy(e, s.pathFields(name))
		models = append(models, e)
	}
	// draining: models unpublished by an unload but whose native memory is not yet freed (the
	// detached drain is still waiting out in-flight requests). An operator who got a 202 from unload
	// polls here and reloads once their model no longer appears — the signal that memory is reclaimed.
	s.regMu.RLock()
	draining := make([]string, 0, len(s.draining))
	for name := range s.draining {
		draining = append(draining, name)
	}
	s.regMu.RUnlock()
	sort.Strings(draining)

	writeJSON(w, http.StatusOK, map[string]any{
		"status":   "ok",
		"backend":  s.cfg.backend,
		"models":   models,
		"draining": draining,
	})
}
