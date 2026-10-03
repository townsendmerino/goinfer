package serveapp

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// An image request used to accept logprobs:true and answer 200 with no logprobs at all: driveVL threw them away. They are now
// returned on the buffered reply (TestGemma4VLReal_E2B_logprobs is the through-the-model proof), and the streamed vision reply,
// which has no logprobs field, refuses the combination the way the text route does rather than dropping them again.
func TestHandleChat_imageStreamLogprobsRejected(t *testing.T) {
	s := &server{models: map[string]*loadedModel{}}
	img := `{"type":"image_url","image_url":{"url":"data:image/png;base64,` + png1x1b64 + `"}}`
	body := func(extra string) string {
		return `{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"what is this"},` + img + `]}],` + extra + `}`
	}
	post := func(b string) *httptest.ResponseRecorder {
		w := httptest.NewRecorder()
		s.handleChat(w, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(b)))
		return w
	}

	w := post(body(`"stream":true,"logprobs":true`))
	if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "logprobs") {
		t.Errorf("image + stream + logprobs: status %d body %s, want a 400 naming logprobs", w.Code, w.Body.String())
	}
	// The rule is narrow: neither half alone is refused on this ground (this empty server then fails later, on the unknown model).
	for name, extra := range map[string]string{"stream without logprobs": `"stream":true`, "logprobs without stream": `"logprobs":true`} {
		if w := post(body(extra)); strings.Contains(w.Body.String(), "logprobs is not supported") {
			t.Errorf("%s was refused for logprobs: %s", name, w.Body.String())
		}
	}
}
