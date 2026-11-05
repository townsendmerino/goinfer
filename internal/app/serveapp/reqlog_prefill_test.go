package serveapp

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestReqLog_prefillFallback: a request whose prefill fell back carries prefill_fallback=<path> and its quoted reason on its
// -log-requests line, and the path and reason in its token-id line, through the middleware's own report; a request with
// no fallback has neither, so every other line keeps its shape.
func TestReqLog_prefillFallback(t *testing.T) {
	for _, fall := range []bool{false, true} {
		var buf bytes.Buffer
		tlPath := filepath.Join(t.TempDir(), "ids.jsonl")
		tl, err := openTokenIDLog(tlPath)
		if err != nil {
			t.Fatal(err)
		}
		s := &server{tokenLog: tl}
		h := s.traceRequests(&buf, func(w http.ResponseWriter, r *http.Request) {
			if fall {
				decoder.NotePrefillFallback(r.Context(), "layer-major", "prompt reaches 8192 keys")
			}
			if tr := traceFrom(r.Context()); tr != nil {
				tr.generated(&loadedModel{name: "m"}, genRequest{promptIDs: []int{1, 2, 3}}, nil, []int{4})
			}
		})
		h(httptest.NewRecorder(), httptest.NewRequest("POST", "/v1/messages", nil))
		tl.f.Close()
		line := strings.TrimSpace(buf.String())
		raw, _ := os.ReadFile(tlPath)
		var tok map[string]any
		_ = json.Unmarshal(bytes.TrimSpace(raw), &tok)
		has := strings.HasSuffix(line, ` prefill_fallback=layer-major prefill_fallback_reason="prompt reaches 8192 keys"`)
		if has != fall || strings.Contains(line, "prefill_fallback") != fall {
			t.Errorf("fallback %v: request line %q", fall, line)
		}
		if got := tok["prefill_fallback_reason"]; fall && got != "prompt reaches 8192 keys" || !fall && got != nil {
			t.Errorf("fallback %v: token-id line %v", fall, tok)
		}
	}
}

// TestReqLog_prefillReportWithoutALog: with neither -log-requests nor the token-id log on, a generation's context still
// carries a prefill report, so a backend can announce its fallback once per request.
func TestReqLog_prefillReportWithoutALog(t *testing.T) {
	var first, tracked bool
	h := (&server{}).traceRequests(nil, func(w http.ResponseWriter, r *http.Request) {
		first, tracked = decoder.NotePrefillFallback(r.Context(), "layer-major", "x")
	})
	h(httptest.NewRecorder(), httptest.NewRequest("POST", "/v1/messages", nil))
	if !first || !tracked {
		t.Errorf("first %v, tracked %v: a request with logging off carries no prefill report", first, tracked)
	}
}
