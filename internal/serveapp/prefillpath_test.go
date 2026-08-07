package serveapp

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// The serve half of the prefill-path report.
//
// WHY THIS EXISTS. Both fast paths a GPU backend advertises — the resident decode runner and the
// batched prefill — fall back SILENTLY when a model doesn't qualify. `--backend cuda --quant
// int8int8` loads clean, reports a GPU, decodes resident, and then takes one forward per prompt
// token because the batched GEMV is int4-only: 9× TTFT on a 300-token prompt, discovered only by
// timing it. serve therefore states the resolved paths at startup, publishes them on /v1/models, and
// -require-backend turns a decline into a startup failure for clients that can't absorb a 9×.
//
// The registry is built directly from decoder.Load rather than newServer: the committed tiny GGUF
// carries no embedded tokenizer (serve's loadDecoder needs one), and neither the /v1/models handler
// nor requireFastPaths touches the tokenizer. That keeps this gate in CI instead of behind a
// GOINFER_SERVE_MODEL download.
func tinyServed(t *testing.T) (*server, *loadedModel) {
	t.Helper()
	p := filepath.Join("..", "..", "testdata", "glm-tiny.gguf")
	if _, err := os.Stat(p); err != nil {
		t.Skipf("no committed tiny fixture at %s", p)
	}
	m, err := decoder.Load(p, decoder.Options{Backend: "cpu", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load tiny fixture: %v", err)
	}
	t.Cleanup(func() { _ = m.Close() })
	lm := &loadedModel{name: "tiny", model: m}
	return &server{models: map[string]*loadedModel{"tiny": lm}}, lm
}

// TestServe_modelsReportsResolvedPaths: /v1/models must carry the paths a model actually got, so a
// batch client can read them instead of inferring them from latency.
func TestServe_modelsReportsResolvedPaths(t *testing.T) {
	srv, lm := tinyServed(t)
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/models", srv.handleModels)
	ts := httptest.NewServer(mux)
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/v1/models")
	if err != nil {
		t.Fatalf("GET /v1/models: %v", err)
	}
	defer resp.Body.Close()
	var ml struct {
		Data []struct {
			ID             string `json:"id"`
			DecodePath     string `json:"decode_path"`
			PrefillPath    string `json:"prefill_path"`
			PrefillBatched *bool  `json:"prefill_batched"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&ml); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(ml.Data) != 1 || ml.Data[0].ID != "tiny" {
		t.Fatalf("/v1/models = %+v, want one entry named tiny", ml.Data)
	}
	e := ml.Data[0]
	if e.DecodePath == "" || e.PrefillPath == "" || e.PrefillBatched == nil {
		t.Fatalf("resolved paths missing from /v1/models: %+v", e)
	}
	// The report must agree with the model itself, not be a re-derivation from the flags: the whole
	// point is that the REQUESTED backend and the RESOLVED path can differ.
	wantBatched, wantWhy := lm.model.PrefillPath()
	if e.DecodePath != lm.model.DecodePath() || e.PrefillPath != wantWhy || *e.PrefillBatched != wantBatched {
		t.Errorf("/v1/models disagrees with the model: got %+v, want decode=%q prefill=%q batched=%v",
			e, lm.model.DecodePath(), wantWhy, wantBatched)
	}
	if !strings.HasPrefix(e.DecodePath, "cpu (") {
		t.Errorf("decode_path = %q, want cpu (<quant>) for a cpu-backend load", e.DecodePath)
	}
}

// TestServe_requireBackendRejectsDecline is the strict-mode gate: a declined prefill must fail the
// LOAD (serve exits before binding a port), not be discovered 14 minutes into a batch run.
func TestServe_requireBackendRejectsDecline(t *testing.T) {
	_, lm := tinyServed(t)
	cfg := config{backend: "cpu", requireBE: true}

	if err := requireFastPaths("tiny", cfg, lm, false, "sequential — batched prefill requires int4 projections"); err == nil {
		t.Fatal("--require-backend accepted a model that declined the batched prefill")
	} else if !strings.Contains(err.Error(), "int4") {
		t.Errorf("startup error drops the reason (an operator can't fix what isn't named): %v", err)
	}
	if err := requireFastPaths("tiny", cfg, lm, true, "batched"); err != nil {
		t.Errorf("--require-backend rejected a healthy cpu model: %v", err)
	}
	// A GPU backend with no resident decode path is the other silent fallback — the whole forward
	// runs staged/CPU while the flags still say cuda.
	if err := requireFastPaths("tiny", config{backend: "cuda", requireBE: true}, lm, true, "batched"); err == nil {
		t.Fatal("--require-backend accepted a cuda model with no resident decode path")
	}
}

// TestServe_healthMirrorsModels: /health is the non-compat surface for the same three fields, for
// clients whose typed OpenAI decoder would reject the /v1/models vendor extension. It must agree
// with /v1/models field for field — a second surface that can disagree is worse than no second
// surface, since an operator would then be reading a number the server doesn't act on.
func TestServe_healthMirrorsModels(t *testing.T) {
	srv, lm := tinyServed(t)
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", srv.handleHealth)
	ts := httptest.NewServer(mux)
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/health")
	if err != nil {
		t.Fatalf("GET /health: %v", err)
	}
	defer resp.Body.Close()
	var h struct {
		Status  string `json:"status"`
		Backend string `json:"backend"`
		Models  []struct {
			ID             string `json:"id"`
			DecodePath     string `json:"decode_path"`
			PrefillPath    string `json:"prefill_path"`
			PrefillBatched *bool  `json:"prefill_batched"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&h); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if h.Status != "ok" || len(h.Models) != 1 {
		t.Fatalf("/health = %+v, want status ok with one model", h)
	}
	wantBatched, wantWhy := lm.model.PrefillPath()
	e := h.Models[0]
	if e.DecodePath != lm.model.DecodePath() || e.PrefillPath != wantWhy || e.PrefillBatched == nil || *e.PrefillBatched != wantBatched {
		t.Errorf("/health disagrees with the model: got %+v, want decode=%q prefill=%q batched=%v",
			e, lm.model.DecodePath(), wantWhy, wantBatched)
	}
}

// TestServe_requireBackendNamesTheResidencyDecline: -require-backend must say WHY residency is
// absent, not only that it is. The three ways `--backend cuda` ends up on CPU — module not built in,
// no usable device, ineligible arch — are fixed by three different actions, and the failure is
// identical from the outside, so a message that doesn't distinguish them costs a debugging session.
func TestServe_requireBackendNamesTheResidencyDecline(t *testing.T) {
	_, lm := tinyServed(t)
	// A cpu-backend load never attempts residency, so the reason names exactly that.
	decline := lm.model.ResidentDecline()
	if decline == "" {
		t.Fatal("no residency decline recorded for a cpu-backend load — the reason is being discarded again")
	}
	err := requireFastPaths("tiny", config{backend: "cuda", requireBE: true}, lm, true, "batched")
	if err == nil {
		t.Fatal("--require-backend accepted a cuda model with no resident decode path")
	}
	if !strings.Contains(err.Error(), decline) {
		t.Errorf("startup error omits the recorded reason %q: %v", decline, err)
	}
}

// TestServe_requireBackendOffIsTheDefault: strict mode is OPT-IN. Without it a decline must stay a
// (now visible) fallback — refusing to start by default would break every existing deployment.
func TestServe_requireBackendOffIsTheDefault(t *testing.T) {
	_, lm := tinyServed(t)
	if got := (config{}).requireBE; got {
		t.Fatal("-require-backend defaults to on")
	}
	// The report is computed either way; only the enforcement is gated.
	if _, why := lm.model.PrefillPath(); why == "" {
		t.Error("no prefill path reported without strict mode")
	}
}
