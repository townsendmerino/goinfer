package serveapp

import (
	"maps"
	"net/http"
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
	writeJSON(w, http.StatusOK, map[string]any{
		"status":  "ok",
		"backend": s.cfg.backend,
		"models":  models,
	})
}
