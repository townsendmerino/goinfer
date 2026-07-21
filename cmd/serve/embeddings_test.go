package main

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/encoder"
)

func TestParseEmbedInput(t *testing.T) {
	cases := []struct {
		raw  string
		want []string
		err  bool
	}{
		{`"hello"`, []string{"hello"}, false},
		{`["a","b","c"]`, []string{"a", "b", "c"}, false},
		{`[]`, []string{}, false},
		{`42`, nil, true},
		{`{"x":1}`, nil, true},
	}
	for _, c := range cases {
		got, err := parseEmbedInput(json.RawMessage(c.raw))
		if (err != nil) != c.err {
			t.Errorf("parseEmbedInput(%s) err=%v, want err=%v", c.raw, err, c.err)
			continue
		}
		if !c.err && !equalStrs(got, c.want) {
			t.Errorf("parseEmbedInput(%s) = %v, want %v", c.raw, got, c.want)
		}
	}
}

func TestParseInputType(t *testing.T) {
	cases := []struct {
		in      string
		isQuery bool
		err     bool
	}{
		{"", false, false},
		{"document", false, false},
		{"search_document", false, false},
		{"query", true, false},
		{"search_query", true, false},
		{"bogus", false, true},
	}
	for _, c := range cases {
		got, err := parseInputType(c.in)
		if (err != nil) != c.err || got != c.isQuery {
			t.Errorf("parseInputType(%q) = (%v,%v), want (%v,err=%v)", c.in, got, err, c.isQuery, c.err)
		}
	}
}

// TestResolveDimensions covers both model kinds, because whether a `dimensions` request is valid
// depends entirely on whether the model was trained with Matryoshka Representation Learning.
// Truncating one that wasn't yields a unit-length, plausible, worse-retrieving vector, so it is
// refused; see resolveDimensions. mrlMin mirrors aikit's Truncatable column (0 = not truncatable).
func TestResolveDimensions(t *testing.T) {
	d := func(n int) *int { return &n }
	cases := []struct {
		name   string
		mrlMin int
		in     *int
		want   int
		err    bool
	}{
		// Pass-through is identical for both kinds: nothing is being truncated.
		{"non-MRL unset", 0, nil, 768, false},
		{"non-MRL zero", 0, d(0), 768, false},
		{"non-MRL native width", 0, d(768), 768, false},
		{"MRL unset", 256, nil, 768, false},
		{"MRL native width", 256, d(768), 768, false},

		// The fix: a non-MRL model refuses ANY real truncation.
		{"non-MRL truncation refused", 0, d(256), 0, true},
		{"non-MRL truncation refused (1 dim)", 0, d(1), 0, true},

		// An MRL model truncates down to its documented floor, and no further.
		{"MRL at floor", 256, d(256), 256, false},
		{"MRL above floor", 256, d(512), 512, false},
		{"MRL below floor refused", 256, d(255), 0, true},
		{"MRL far below floor refused", 256, d(64), 0, true},

		// Range checks still apply regardless of MRL.
		{"over native width", 256, d(769), 0, true},
		{"negative", 256, d(-1), 0, true},
		{"over native width, non-MRL", 0, d(769), 0, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			s := &server{embedDim: 768, embedMRLMin: c.mrlMin, embedID: "m"}
			got, err := s.resolveDimensions(c.in)
			if (err != nil) != c.err || (!c.err && got != c.want) {
				t.Errorf("resolveDimensions(mrlMin=%d, %v) = (%d,%v), want (%d,err=%v)",
					c.mrlMin, c.in, got, err, c.want, c.err)
			}
		})
	}
}

func TestWantsBase64(t *testing.T) {
	for _, c := range []struct {
		in   string
		want bool
		err  bool
	}{
		{"", false, false}, {"float", false, false}, {"base64", true, false}, {"xml", false, true},
	} {
		got, err := wantsBase64(c.in)
		if (err != nil) != c.err || got != c.want {
			t.Errorf("wantsBase64(%q) = (%v,%v), want (%v,err=%v)", c.in, got, err, c.want, c.err)
		}
	}
}

func TestPostprocess(t *testing.T) {
	// Full width: output is unit length.
	v := postprocess([]float32{3, 4}, 2)
	if got := norm(v); math.Abs(got-1) > 1e-6 {
		t.Errorf("normalized norm = %v, want 1", got)
	}
	if math.Abs(float64(v[0])-0.6) > 1e-6 || math.Abs(float64(v[1])-0.8) > 1e-6 {
		t.Errorf("normalized = %v, want [0.6 0.8]", v)
	}

	// Truncation renormalizes and does not mutate the original.
	orig := []float32{1, 2, 2, 10}
	cp := append([]float32(nil), orig...)
	got := postprocess(cp, 2)
	if len(got) != 2 {
		t.Fatalf("truncated len = %d, want 2", len(got))
	}
	if n := norm(got); math.Abs(n-1) > 1e-6 {
		t.Errorf("truncated norm = %v, want 1", n)
	}
	// [1,2] renormalized = [1,2]/sqrt(5).
	want0 := float32(1 / math.Sqrt(5))
	if math.Abs(float64(got[0]-want0)) > 1e-6 {
		t.Errorf("truncated[0] = %v, want %v", got[0], want0)
	}

	// Zero vector is left intact (no NaN).
	z := postprocess([]float32{0, 0, 0}, 3)
	for _, x := range z {
		if x != 0 {
			t.Errorf("zero vector changed: %v", z)
		}
	}
}

func TestFloat32sToBase64_roundTrip(t *testing.T) {
	v := []float32{0.6, 0.8, -1.5, 0}
	s := float32sToBase64(v)
	raw, err := base64.StdEncoding.DecodeString(s)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(raw) != 4*len(v) {
		t.Fatalf("bytes = %d, want %d", len(raw), 4*len(v))
	}
	for i, want := range v {
		got := math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		if got != want {
			t.Errorf("v[%d] = %v, want %v", i, got, want)
		}
	}
}

// stubEncoder implements encoder.Encoder with deterministic vectors, so the
// handler's HTTP plumbing is testable without a real model.
type stubEncoder struct{ dim int }

func (e stubEncoder) HiddenDim() int                         { return e.dim }
func (e stubEncoder) Encode(string, bool) ([]float32, error) { return e.vec(), nil }
func (e stubEncoder) vec() []float32 {
	v := make([]float32, e.dim)
	for i := range v {
		v[i] = float32(i + 1)
	}
	return v
}
func (e stubEncoder) EncodeBatch(texts []string, _ []bool, _ int) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i := range texts {
		out[i] = e.vec()
	}
	return out, nil
}

// newEmbedTestServer is a MATRYOSHKA-capable stub (floor 2 of 4), so the existing dimensions
// tests keep exercising real truncation. Non-MRL behavior is covered by newNonMRLEmbedTestServer.
func newEmbedTestServer() *server {
	return &server{embed: stubEncoder{dim: 4}, embedDim: 4, embedID: "stub-embed", embedMRLMin: 2}
}

// newNonMRLEmbedTestServer is the default shape: an embedder never certified for truncation, so
// embedMRLMin is 0 and any dimensions request must be refused.
func newNonMRLEmbedTestServer() *server {
	return &server{embed: stubEncoder{dim: 4}, embedDim: 4, embedID: "stub-nonmrl"}
}

func postEmbed(t *testing.T, s *server, body string) *httptest.ResponseRecorder {
	t.Helper()
	rr := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/v1/embeddings", bytes.NewReader([]byte(body)))
	s.handleEmbeddings(rr, req)
	return rr
}

func TestHandleEmbeddings_floatAndShape(t *testing.T) {
	s := newEmbedTestServer()
	rr := postEmbed(t, s, `{"model":"stub-embed","input":["a","bb"]}`)
	if rr.Code != 200 {
		t.Fatalf("status %d: %s", rr.Code, rr.Body.String())
	}
	var resp struct {
		Object string `json:"object"`
		Model  string `json:"model"`
		Data   []struct {
			Object    string    `json:"object"`
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
		Usage struct {
			PromptTokens int `json:"prompt_tokens"`
			TotalTokens  int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal: %v\n%s", err, rr.Body.String())
	}
	if resp.Object != "list" || resp.Model != "stub-embed" || len(resp.Data) != 2 {
		t.Fatalf("bad envelope: %+v", resp)
	}
	for i, d := range resp.Data {
		if d.Object != "embedding" || d.Index != i {
			t.Errorf("data[%d] object/index = %q/%d", i, d.Object, d.Index)
		}
		if len(d.Embedding) != 4 {
			t.Errorf("data[%d] dim = %d, want 4", i, len(d.Embedding))
		}
		if n := norm32(d.Embedding); math.Abs(n-1) > 1e-5 {
			t.Errorf("data[%d] not unit length: norm=%v", i, n)
		}
	}
}

func TestHandleEmbeddings_dimensionsAndBase64(t *testing.T) {
	s := newEmbedTestServer()
	rr := postEmbed(t, s, `{"input":"x","dimensions":2,"encoding_format":"base64"}`)
	if rr.Code != 200 {
		t.Fatalf("status %d: %s", rr.Code, rr.Body.String())
	}
	var resp struct {
		Data []struct {
			Embedding string `json:"embedding"`
		} `json:"data"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(resp.Data) != 1 {
		t.Fatalf("data len = %d", len(resp.Data))
	}
	raw, err := base64.StdEncoding.DecodeString(resp.Data[0].Embedding)
	if err != nil {
		t.Fatalf("decode base64: %v", err)
	}
	if len(raw) != 2*4 { // dimensions=2 → 2 float32s
		t.Fatalf("base64 decoded to %d bytes, want 8", len(raw))
	}
	vec := []float32{
		math.Float32frombits(binary.LittleEndian.Uint32(raw[0:])),
		math.Float32frombits(binary.LittleEndian.Uint32(raw[4:])),
	}
	if n := norm32(vec); math.Abs(n-1) > 1e-5 {
		t.Errorf("truncated base64 vec not unit length: %v (norm %v)", vec, n)
	}
}

func TestHandleEmbeddings_badRequests(t *testing.T) {
	s := newEmbedTestServer()
	for _, body := range []string{
		`{}`,                                    // missing input
		`{"input":42}`,                          // wrong input type
		`{"input":"x","input_type":"nope"}`,     // bad input_type
		`{"input":"x","dimensions":99}`,         // out-of-range dims
		`{"input":"x","encoding_format":"xml"}`, // bad format
		`not json`,                              // malformed body
	} {
		rr := postEmbed(t, s, body)
		if rr.Code != http.StatusBadRequest {
			t.Errorf("body %s → status %d, want 400", body, rr.Code)
		}
	}
}

// TestServe_embedIntegration runs the real encoder end to end. Gated on
// GOINFER_EMBED_MODEL (a CodeRankEmbed HF dir) so it skips in normal CI.
//
//	GOINFER_EMBED_MODEL=~/models/coderankembed go test ./cmd/serve -run embedIntegration -v
func TestServe_embedIntegration(t *testing.T) {
	dir := os.Getenv("GOINFER_EMBED_MODEL")
	if dir == "" {
		t.Skip("set GOINFER_EMBED_MODEL=<CodeRankEmbed dir> to run the embeddings integration test")
	}
	srv, err := newServer(config{embedPath: dir, embedQuant: "f32"})
	if err != nil {
		t.Fatalf("newServer: %v", err)
	}
	rr := postEmbed(t, srv, `{"input":["package main","func add(a,b int) int { return a+b }"],"input_type":"document"}`)
	if rr.Code != 200 {
		t.Fatalf("status %d: %s", rr.Code, rr.Body.String())
	}
	var resp struct {
		Data []struct {
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
		Usage struct {
			PromptTokens int `json:"prompt_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(resp.Data) != 2 || len(resp.Data[0].Embedding) != srv.embedDim {
		t.Fatalf("got %d vectors of dim %d, want 2 of %d", len(resp.Data), len(resp.Data[0].Embedding), srv.embedDim)
	}
	if resp.Usage.PromptTokens == 0 {
		t.Errorf("prompt_tokens = 0, want > 0")
	}
	if n := norm32(resp.Data[0].Embedding); math.Abs(n-1) > 1e-4 {
		t.Errorf("embedding not unit length: norm %v", n)
	}
	t.Logf("embedded 2 docs, dim %d, prompt_tokens %d", srv.embedDim, resp.Usage.PromptTokens)
}

// --- helpers ---

func norm(v []float32) float64 {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	return math.Sqrt(s)
}
func norm32(v []float32) float64 { return norm(v) }

func equalStrs(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestHandleEmbeddings_dimensionsRequiresMatryoshka is the guard against a silent-wrong: honoring
// `dimensions` for a model not trained with Matryoshka Representation Learning returns a
// unit-length, entirely plausible vector that simply RETRIEVES WORSE. That is measured, not
// theoretical — aikit's TestEmbedderCoverage_matryoshka shows multilingual-e5-base sliced to a
// quarter width dropping paraphrase-pair recall 1.00 → 0.80, while genuine MRL models hold their
// documented floor. Only two of aikit's eight certified embedders qualify.
//
// Both directions are asserted, because a guard that only rejects is as broken as one that only
// accepts: it would refuse the legitimate MRL truncation the parameter exists for.
func TestHandleEmbeddings_dimensionsRequiresMatryoshka(t *testing.T) {
	t.Run("non-MRL model rejects any truncation", func(t *testing.T) {
		s := newNonMRLEmbedTestServer()
		rr := postEmbed(t, s, `{"input":"x","dimensions":2}`)
		if rr.Code != http.StatusBadRequest {
			t.Fatalf("status %d, want 400 — truncating a non-MRL model must be refused, not silently degraded: %s",
				rr.Code, rr.Body.String())
		}
		// The message has to say WHY, or the caller just retries with another width.
		if body := rr.Body.String(); !strings.Contains(body, "Matryoshka") {
			t.Errorf("error does not explain the refusal: %s", body)
		}
	})

	t.Run("non-MRL model still allows native width", func(t *testing.T) {
		s := newNonMRLEmbedTestServer()
		for _, body := range []string{
			`{"input":"x"}`,                // unset
			`{"input":"x","dimensions":0}`, // explicit zero
			`{"input":"x","dimensions":4}`, // exactly the native width — no truncation happens
		} {
			rr := postEmbed(t, s, body)
			if rr.Code != http.StatusOK {
				t.Errorf("body %s → status %d, want 200 (nothing is being truncated): %s", body, rr.Code, rr.Body.String())
			}
		}
	})

	t.Run("MRL model accepts down to its floor and no further", func(t *testing.T) {
		s := newEmbedTestServer() // floor 2 of 4
		rr := postEmbed(t, s, `{"input":"x","dimensions":2}`)
		if rr.Code != http.StatusOK {
			t.Fatalf("status %d at the documented floor, want 200: %s", rr.Code, rr.Body.String())
		}
		var resp struct {
			Data []struct {
				Embedding []float32 `json:"embedding"`
			} `json:"data"`
		}
		if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
			t.Fatalf("decode: %v", err)
		}
		if got := len(resp.Data[0].Embedding); got != 2 {
			t.Errorf("returned %d dims, want 2", got)
		}
		// Below the floor is refused: the model was never certified that short.
		rr = postEmbed(t, s, `{"input":"x","dimensions":1}`)
		if rr.Code != http.StatusBadRequest {
			t.Errorf("status %d below the floor, want 400: %s", rr.Code, rr.Body.String())
		}
	})
}

// TestMatryoshkaFloor_contract pins the shape of the aikit lookup loadEncoder depends on. Without
// it, a change to aikit's key format would make every MRL model resolve to 0 and the server would
// silently start REFUSING legitimate truncation — a quiet capability regression, the mirror of the
// bug this guard was added for. Both call shapes matter: loadEncoder passes a filesystem path,
// while the registry is keyed by HF id.
func TestMatryoshkaFloor_contract(t *testing.T) {
	for _, c := range []struct {
		name string
		want int
		ok   bool
	}{
		{"nomic-ai/nomic-embed-text-v1.5", 64, true},         // HF id
		{"/home/me/models/nomic-embed-text-v1.5", 64, true},  // path, as loadEncoder passes it
		{"/home/me/models/nomic-embed-text-v1.5/", 64, true}, // trailing slash
		{"nomic-ai/nomic-embed-text-v2-moe", 256, true},      //
		{"BAAI/bge-m3", 0, false},                            // certified, but NOT truncatable
		{"/home/me/models/multilingual-e5-base", 0, false},   // the one measured to degrade
		{"some-org/never-heard-of-it", 0, false},             // unknown ⇒ refuse, never guess
	} {
		got, ok := encoder.MatryoshkaFloor(c.name)
		if got != c.want || ok != c.ok {
			t.Errorf("MatryoshkaFloor(%q) = (%d, %v), want (%d, %v)", c.name, got, ok, c.want, c.ok)
		}
	}
}
