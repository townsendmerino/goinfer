//go:build goinfer_testhooks

package serveapp

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/giw"
	"github.com/townsendmerino/goinfer/internal/loadflags"
	"github.com/townsendmerino/goinfer/internal/modelload"
)

// TestAdminFit_answerIsWhatTheLoadDoes: POST /admin/models/fit says what the load then does with the same body. Where it
// fits, decoder.Load of the path and options the load resolves succeeds on the decode path the answer named; where it does
// not (available memory set below the model's price), that Load is refused with the answer's decline. The fit call itself
// loads no weights, registers no model and leaves no live heap behind. The load half calls decoder.Load rather than POST /admin/models/load because the
// committed tiny fixtures carry no tokenizer, which the admin load needs and the memory check does not.
func TestAdminFit_answerIsWhatTheLoadDoes(t *testing.T) {
	path, _ := filepath.Abs(filepath.Join("..", "..", "testdata", "glm-tiny.gguf"))
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no committed tiny fixture at %s", path)
	}
	// DirectLoad: a .gguf otherwise loads through a sidecar .giw the test would have to build; Fit false is --fit=off, so a
	// refused dense .gguf is refused rather than retried with weight streaming.
	srv, err := newServer(config{allowAdmin: true, load: loadflags.Flags{Backend: "cpu", Quant: "int8int8", DirectLoad: true}})
	if err != nil {
		t.Fatal(err)
	}
	mux := http.NewServeMux()
	registerAdminRoutes(mux, srv, 1<<20, func(h http.HandlerFunc) http.HandlerFunc { return h })
	ts := httptest.NewServer(mux)
	defer ts.Close()
	post := func(route, body string) (int, string) {
		r, err := http.Post(ts.URL+route, "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		defer r.Body.Close()
		b, _ := io.ReadAll(r.Body)
		return r.StatusCode, string(b)
	}
	type answer struct {
		Fits       *bool  `json:"fits"`
		Unknown    string `json:"unknown"`
		Decline    string `json:"decline"`
		DecodePath string `json:"decode_path"`
		NeedBytes  int64  `json:"need_bytes"`
	}
	fit := func(body string) answer {
		code, b := post("/admin/models/fit", body)
		if code != http.StatusOK {
			t.Fatalf("fit: %d %s", code, b)
		}
		var a answer
		if err := json.Unmarshal([]byte(b), &a); err != nil {
			t.Fatalf("fit body %q: %v", b, err)
		}
		return a
	}

	var ms runtime.MemStats
	live := func() uint64 { runtime.GC(); runtime.ReadMemStats(&ms); return ms.HeapInuse }
	fit(`{"path":"` + path + `"}`) // a first call settles one-time state (the quant byte-cost table)
	heap, allocs := live(), decoder.WeightAllocsForTest()
	a := fit(`{"path":"` + path + `"}`)
	if grew := int64(live()) - int64(heap); grew > 1<<20 {
		t.Errorf("the fit call left %d more bytes of live heap", grew)
	}
	if a.Fits == nil || !*a.Fits || a.Unknown != "" || a.NeedBytes <= 0 {
		t.Fatalf("a load that fits: %+v", a)
	}
	if got := decoder.WeightAllocsForTest() - allocs; got != 0 || len(srv.modelList()) != 0 {
		t.Fatalf("the fit call loaded weights %d time(s) and registered %d model(s)", got, len(srv.modelList()))
	}
	if a.DecodePath != "cpu (int8int8)" {
		t.Errorf("decode_path %q, want cpu (int8int8)", a.DecodePath)
	}
	load := func() (*decoder.Model, error) {
		c := srv.adminLoadConfig(adminLoadReq{Path: path})
		spec := modelSpec{path: path}
		p, err := modelload.Plan(modelload.Request{Spec: path, Opts: spec.options(c), DirectLoad: c.load.DirectLoad, ExplicitQuant: spec.explicitQuant(c)})
		if err != nil {
			t.Fatal(err)
		}
		return decoder.Load(p.LoadPath, p.Opts)
	}
	m, err := load()
	if err != nil {
		t.Fatalf("the answer said it fits; the load: %v", err)
	}
	if got := m.DecodePath(); !strings.HasPrefix(got, a.DecodePath) {
		t.Errorf("loaded on %q, the answer said %q", got, a.DecodePath)
	}
	m.Close()

	restore := decoder.SetHostRAMAvailableForTest(64 << 10)
	defer restore()
	a = fit(`{"path":"` + path + `"}`)
	if a.Fits == nil || *a.Fits || a.Decline == "" {
		t.Fatalf("a load that does not fit: %+v", a)
	}
	if m, err := load(); err == nil || err.Error() != a.Decline {
		if m != nil {
			m.Close()
		}
		t.Errorf("the answer declined with %q; the load returned %v", a.Decline, err)
	}
}

// TestAdminFit_unbuiltSidecarIsUnknown: on the default sidecar route, a .gguf whose .giw is not built yet gets no verdict:
// the load would build the .giw first, so the answer says so and names it.
func TestAdminFit_unbuiltSidecarIsUnknown(t *testing.T) {
	src, _ := filepath.Abs(filepath.Join("..", "..", "testdata", "glm-tiny.gguf"))
	raw, err := os.ReadFile(src)
	if err != nil {
		t.Skipf("no committed tiny fixture: %v", err)
	}
	path := filepath.Join(t.TempDir(), "glm-tiny.gguf") // a fresh copy has no sidecar beside it
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	srv, err := newServer(config{allowAdmin: true, load: loadflags.Flags{Backend: "cpu", Quant: "int8int8"}})
	if err != nil {
		t.Fatal(err)
	}
	got, err := srv.fitAnswer(adminLoadReq{Path: path})
	if err != nil {
		t.Fatal(err)
	}
	if got["fits"] != nil || !strings.HasPrefix(got["unknown"].(string), ".giw not built") || got["build"] == nil {
		t.Fatalf("an unbuilt sidecar: %v", got)
	}
}

// TestAdminFit_builtSidecarIsPricedWithoutLoading: once the sidecar .giw the answer named is built, the answer prices that
// .giw with the load's .giw check, and the call enters decoder.Load zero times (the load's own sidecar freshness check
// loads the bundle; the route must not). decoder.Load of the named file then succeeds, as the answer said.
func TestAdminFit_builtSidecarIsPricedWithoutLoading(t *testing.T) {
	src, _ := filepath.Abs(filepath.Join("..", "..", "testdata", "glm-tiny.gguf"))
	raw, err := os.ReadFile(src)
	if err != nil {
		t.Skipf("no committed tiny fixture: %v", err)
	}
	path := filepath.Join(t.TempDir(), "glm-tiny.gguf")
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	srv, err := newServer(config{allowAdmin: true, load: loadflags.Flags{Backend: "cpu", Quant: "int8int8"}})
	if err != nil {
		t.Fatal(err)
	}
	c := srv.adminLoadConfig(adminLoadReq{Path: path})
	spec := modelSpec{path: path}
	p, err := modelload.Plan(modelload.Request{Spec: path, Opts: spec.options(c), ExplicitQuant: spec.explicitQuant(c)})
	if err != nil || p.Build == "" {
		t.Fatalf("plan before the build: %+v, %v", p, err)
	}
	// The sidecar the load would build, written directly: the committed fixture has no embedded tokenizer, which
	// prequant's transcode needs and the fit check does not read.
	built := p.Build
	src0, err := decoder.Load(path, p.Opts)
	if err != nil {
		t.Fatal(err)
	}
	blob, err := decoder.SerializeWeightsForTarget(src0.Weights(), "fit-test", decoder.GIWTargetForBackend(p.Opts.Backend))
	src0.Close()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(built, giw.Write(blob, []byte("tok")), 0o644); err != nil {
		t.Fatal(err)
	}
	if again, err := modelload.Plan(modelload.Request{Spec: path, Opts: spec.options(c), ExplicitQuant: spec.explicitQuant(c)}); err != nil || again.LoadPath != built {
		t.Fatalf("plan after the build: %+v, %v", again, err)
	}

	loads := decoder.LoadCallsForTest()
	got, err := srv.fitAnswer(adminLoadReq{Path: path})
	if err != nil {
		t.Fatal(err)
	}
	if n := decoder.LoadCallsForTest() - loads; n != 0 {
		t.Fatalf("the fit call entered decoder.Load %d time(s)", n)
	}
	if got["fits"] != true || got["guard"] != "giw" || got["load_path"] != built {
		t.Fatalf("a built sidecar: %v", got)
	}
	m, err := decoder.Load(built, p.Opts)
	if err != nil {
		t.Fatalf("the answer said it fits; the load: %v", err)
	}
	m.Close()
}
