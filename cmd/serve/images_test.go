package main

import (
	"encoding/json"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
)

// 1×1 px PNG, base64.
const png1x1b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

func TestContentPartsText(t *testing.T) {
	if got := contentPartsText(json.RawMessage(`"hi there"`)); got != "hi there" {
		t.Errorf("string: %q", got)
	}
	arr := `[{"type":"text","text":"a "},{"type":"image_url","image_url":{"url":"data:image/png;base64,xx"}},{"type":"text","text":"b"}]`
	if got := contentPartsText(json.RawMessage(arr)); got != "a b" {
		t.Errorf("array text concat: %q", got)
	}
	if got := contentPartsText(nil); got != "" {
		t.Errorf("nil: %q", got)
	}
}

func TestContentPartsImages(t *testing.T) {
	arr := `[{"type":"text","text":"caption"},{"type":"image_url","image_url":{"url":"data:image/png;base64,` + png1x1b64 + `"}}]`
	imgs, err := contentPartsImages(json.RawMessage(arr))
	if err != nil || len(imgs) != 1 || len(imgs[0].data) == 0 {
		t.Fatalf("data-URI image: imgs=%d err=%v", len(imgs), err)
	}
	if imgs[0].mediaType != "image/png" {
		t.Errorf("mediaType = %q", imgs[0].mediaType)
	}
	// Plain string content → no images, no error.
	if imgs, err := contentPartsImages(json.RawMessage(`"just text"`)); err != nil || imgs != nil {
		t.Errorf("string content: imgs=%v err=%v", imgs, err)
	}
	// http(s) URL → rejected (SSRF guard).
	httpArr := `[{"type":"image_url","image_url":{"url":"https://evil.example/x.png"}}]`
	if _, err := contentPartsImages(json.RawMessage(httpArr)); err == nil {
		t.Error("http url image should be rejected")
	}
}

func TestDecodeDataURI(t *testing.T) {
	if _, err := decodeDataURI("data:image/png;base64," + png1x1b64); err != nil {
		t.Errorf("valid data uri: %v", err)
	}
	if _, err := decodeDataURI("https://x/y.png"); err == nil {
		t.Error("non-data uri should error")
	}
	if _, err := decodeDataURI("data:image/png,notbase64"); err == nil {
		t.Error("non-base64 data uri should error")
	}
}

func TestLastUserTurn(t *testing.T) {
	turns := []chat.Turn{{Role: "user", Content: "a"}, {Role: "assistant", Content: "b"}, {Role: "user", Content: "c"}}
	if i := lastUserTurn(turns); i != 2 {
		t.Errorf("lastUserTurn = %d, want 2", i)
	}
	if i := lastUserTurn([]chat.Turn{{Role: "assistant", Content: "x"}}); i != -1 {
		t.Errorf("no user turn should give -1, got %d", i)
	}
}
