package media

import (
	"encoding/base64"
	"encoding/json"
	"net/http"
	"strings"
	"testing"
)

const png1x1b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

func TestContentPartsTextAndImages(t *testing.T) {
	if got := ContentPartsText(json.RawMessage(`"hi there"`)); got != "hi there" {
		t.Errorf("string: %q", got)
	}
	arr := `[{"type":"text","text":"a "},{"type":"image_url","image_url":{"url":"data:image/png;base64,` + png1x1b64 + `"}},{"type":"text","text":"b"}]`
	if got := ContentPartsText(json.RawMessage(arr)); got != "a b" {
		t.Errorf("array text concat: %q", got)
	}
	imgs, err := ContentPartsImages(json.RawMessage(arr))
	if err != nil || len(imgs) != 1 || imgs[0].MediaType != "image/png" || imgs[0].At != 2 || len(imgs[0].Data) == 0 {
		t.Fatalf("data-URI image: %+v err=%v", imgs, err)
	}
	if _, err := ContentPartsImages(json.RawMessage(`[{"type":"image_url","image_url":{"url":"https://evil.example/x.png"}}]`)); err == nil {
		t.Error("http url image must be rejected")
	}
	if _, err := DecodeDataURI("data:image/png,notbase64"); err == nil {
		t.Error("non-base64 data uri must error")
	}
	ref, err := DecodeBase64Image("image/png", png1x1b64)
	if err != nil || ref.MediaType != "image/png" || len(ref.Data) == 0 {
		t.Errorf("DecodeBase64Image: %+v err=%v", ref, err)
	}
}

func TestOmitHistoryMediaAndNoteHeader(t *testing.T) {
	u1 := "data:image/png;base64," + base64.StdEncoding.EncodeToString([]byte("first"))
	u2 := "data:image/png;base64," + base64.StdEncoding.EncodeToString([]byte("second"))
	m0 := json.RawMessage(`[{"type":"text","text":"look"},{"type":"image_url","image_url":{"url":"` + u1 + `"}}]`)
	m1 := json.RawMessage(`"ok"`)
	m2 := json.RawMessage(`[{"type":"image_url","image_url":{"url":"` + u2 + `"}}]`)
	cs := []*json.RawMessage{&m0, &m1, &m2}
	if n := OmitHistoryImages(cs, "image_url"); n != 1 {
		t.Fatalf("omitted %d, want 1", n)
	}
	if !strings.Contains(string(m0), OlderImageNote) || strings.Contains(string(m0), "image_url") {
		t.Errorf("m0 not rewritten: %s", m0)
	}
	if string(m1) != `"ok"` {
		t.Errorf("plain string modified: %s", m1)
	}
	h := http.Header{}
	NoteOmittedImages(h, 0)
	if got := h.Get(ImagesOmittedHeader); got != "" {
		t.Errorf("zero omitted set header %q", got)
	}
	NoteOmittedImages(h, 2)
	if got := h.Get(ImagesOmittedHeader); got != "2" {
		t.Errorf("header = %q, want 2", got)
	}
}

func TestParseEmbedItemsAndBounds(t *testing.T) {
	if err := CheckEmbedInputBounds(make([]string, MaxEmbedInputs)); err != nil {
		t.Errorf("legal batch failed: %v", err)
	}
	if err := CheckEmbedInputBounds(make([]string, MaxEmbedInputs+1)); err == nil {
		t.Error("oversized batch must fail")
	}
	u := "data:image/png;base64," + base64.StdEncoding.EncodeToString([]byte("A"))
	body := `["plain",{"image":"` + u + `","text":"cap"},[{"type":"image_url","image_url":{"url":"` + u + `"}},{"type":"text","text":"tail"}]]`
	items, err := ParseEmbedItems(json.RawMessage(body))
	if err != nil || len(items) != 3 {
		t.Fatalf("ParseEmbedItems: len=%d err=%v", len(items), err)
	}
	imgs, aud := EmbedMedia(items)
	if imgs != 2 || aud != 0 {
		t.Errorf("EmbedMedia = (%d, %d), want (2, 0)", imgs, aud)
	}
	if _, err := ParseEmbedItems(json.RawMessage(`[[{"type":"text","text":"before"},{"type":"image_url","image_url":{"url":"` + u + `"}}]]`)); err == nil {
		t.Error("text before image must be rejected")
	}
}
