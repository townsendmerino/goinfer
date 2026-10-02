package serveapp

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/loadflags"
)

func dataURI(payload string) string {
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString([]byte(payload))
}

func chatMsgs(t *testing.T, body string) []chatMessage {
	t.Helper()
	var req struct {
		Messages []chatMessage `json:"messages"`
	}
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		t.Fatal(err)
	}
	return req.Messages
}

// R23: the sequence handleChat runs — omit earlier images, then collect what is left — so the vision path's one-image guard sees one image for a chat that
// resends its history, and still sees two for a caller who put two in the same message.
func TestOmitChatHistoryImages(t *testing.T) {
	one, two := dataURI("first"), dataURI("second")
	hist := `{"messages":[
	  {"role":"user","content":[{"type":"text","text":"what is this?"},{"type":"image_url","image_url":{"url":"` + one + `","detail":"low"}}]},
	  {"role":"assistant","content":"a cat"},
	  {"role":"user","content":[{"type":"text","text":"and this?"},{"type":"image_url","image_url":{"url":"` + two + `"}}]}]}`
	msgs := chatMsgs(t, hist)
	if n := omitChatHistoryImages(msgs); n != 1 {
		t.Fatalf("omitted %d, want 1", n)
	}
	imgs, err := chatImages(msgs)
	if err != nil {
		t.Fatal(err)
	}
	if len(imgs) != 1 || string(imgs[0].data) != "second" {
		t.Fatalf("after omission the vision path would see %d image(s) %q, want exactly the newest", len(imgs), imgs)
	}
	if got := msgs[0].text(); !strings.Contains(got, "what is this?") || !strings.Contains(got, olderImageNote) {
		t.Errorf("the earlier turn's text = %q, want its words plus the omission note", got)
	}
	if strings.Contains(string(msgs[0].Content), "image_url") {
		t.Errorf("the earlier image part is still there: %s", msgs[0].Content)
	}
	if string(msgs[1].Content) != `"a cat"` {
		t.Errorf("a plain-string message was touched: %s", msgs[1].Content)
	}

	// Three turns with an image each: two omitted, the newest kept.
	three := chatMsgs(t, `{"messages":[
	  {"role":"user","content":[{"type":"image_url","image_url":{"url":"`+dataURI("a")+`"}}]},
	  {"role":"user","content":[{"type":"image_url","image_url":{"url":"`+dataURI("b")+`"}}]},
	  {"role":"user","content":[{"type":"image_url","image_url":{"url":"`+dataURI("c")+`"}}]}]}`)
	if n := omitChatHistoryImages(three); n != 2 {
		t.Errorf("three turns: omitted %d, want 2", n)
	}
	if imgs, _ := chatImages(three); len(imgs) != 1 || string(imgs[0].data) != "c" {
		t.Errorf("three turns: kept %q", imgs)
	}

	// Two images in the SAME (latest) message are the caller's explicit request: nothing is omitted, and the guard still sees two.
	same := chatMsgs(t, `{"messages":[{"role":"user","content":[
	  {"type":"image_url","image_url":{"url":"`+dataURI("a")+`"}},{"type":"image_url","image_url":{"url":"`+dataURI("b")+`"}}]}]}`)
	if n := omitChatHistoryImages(same); n != 0 {
		t.Errorf("two images in one message: omitted %d, want 0", n)
	}
	if imgs, _ := chatImages(same); len(imgs) != 2 || len(imgs) <= maxImagesPerTurn {
		t.Errorf("two images in one message: %d seen — the one-image 400 would no longer fire", len(imgs))
	}

	// One image, or none: the bytes are left exactly as they came.
	for _, body := range []string{
		`{"messages":[{"role":"user","content":"hi"}]}`,
		`{"messages":[{"role":"user","content":[{"type":"text","text":"x"},{"type":"image_url","image_url":{"url":"` + one + `"}}]}]}`,
	} {
		ms := chatMsgs(t, body)
		before := make([]string, len(ms))
		for i := range ms {
			before[i] = string(ms[i].Content)
		}
		if n := omitChatHistoryImages(ms); n != 0 {
			t.Errorf("%s: omitted %d", body, n)
		}
		for i := range ms {
			if string(ms[i].Content) != before[i] {
				t.Errorf("a message with nothing to omit was rewritten: %s", ms[i].Content)
			}
		}
	}
}

func TestOmitAnthropicHistoryImages(t *testing.T) {
	var req anthropicReq
	body := `{"messages":[
	  {"role":"user","content":[{"type":"text","text":"look"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"` + base64.StdEncoding.EncodeToString([]byte("first")) + `"}}]},
	  {"role":"assistant","content":[{"type":"text","text":"ok"}]},
	  {"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"` + base64.StdEncoding.EncodeToString([]byte("second")) + `"}},{"type":"text","text":"and?"}]}]}`
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		t.Fatal(err)
	}
	if n := omitAnthropicHistoryImages(req.Messages); n != 1 {
		t.Fatalf("omitted %d, want 1", n)
	}
	imgs, err := anthropicImages(&req)
	if err != nil || len(imgs) != 1 || string(imgs[0].data) != "second" {
		t.Fatalf("after omission: %d image(s), err %v", len(imgs), err)
	}
	if !strings.Contains(string(req.Messages[0].Content), olderImageNote) || !strings.Contains(string(req.Messages[0].Content), `"look"`) {
		t.Errorf("the earlier turn lost its words or has no note: %s", req.Messages[0].Content)
	}
}

// Through handleChat on a text-only model (the only vision-less path CI can run): the header names how many earlier images were left out, and a request
// with one image, or none, carries no header.
func TestHandleChat_announcesOmittedHistoryImages(t *testing.T) {
	p := filepath.Join("..", "..", "testdata", "tiny-qwen2-moe")
	if _, err := os.Stat(p); err != nil {
		t.Skipf("no committed tiny checkpoint at %s", p)
	}
	srv, err := newServer(config{models: modelFlag{{name: "tiny", path: p}}, load: loadflags.Flags{Backend: "cpu", Quant: "int8int8"}})
	if err != nil {
		t.Fatal(err)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", srv.handleChat)
	mux.HandleFunc("POST /v1/messages", srv.handleMessages)
	ts := httptest.NewServer(mux)
	defer ts.Close()
	postTo := func(path, body string) *http.Response {
		resp, err := http.Post(ts.URL+path, "application/json", bytes.NewReader([]byte(body)))
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		return resp
	}
	post := func(body string) *http.Response { return postTo("/v1/chat/completions", body) }
	img := func(s string) string {
		return `{"type":"image_url","image_url":{"url":"` + dataURI(s) + `"}}`
	}
	two := post(`{"model":"tiny","max_tokens":2,"messages":[{"role":"user","content":[` + img("a") + `]},{"role":"assistant","content":"x"},{"role":"user","content":[` + img("b") + `]}]}`)
	if got := two.Header.Get(imagesOmittedHeader); got != "1" {
		t.Errorf("%s = %q for a history with two images, want 1", imagesOmittedHeader, got)
	}
	one := post(`{"model":"tiny","max_tokens":2,"messages":[{"role":"user","content":[` + img("a") + `]}]}`)
	if got := one.Header.Get(imagesOmittedHeader); got != "" {
		t.Errorf("%s = %q for a single image, want none", imagesOmittedHeader, got)
	}

	// The Anthropic route announces it the same way.
	ablock := func(s string) string {
		return `{"type":"image","source":{"type":"base64","media_type":"image/png","data":"` + base64.StdEncoding.EncodeToString([]byte(s)) + `"}}`
	}
	am := postTo("/v1/messages", `{"model":"tiny","max_tokens":2,"messages":[{"role":"user","content":[`+ablock("a")+`]},{"role":"assistant","content":[{"type":"text","text":"x"}]},{"role":"user","content":[`+ablock("b")+`]}]}`)
	if got := am.Header.Get(imagesOmittedHeader); got != "1" {
		t.Errorf("/v1/messages: %s = %q for a history with two images, want 1", imagesOmittedHeader, got)
	}
}
