package serveapp

import (
	"encoding/json"
	"strings"
	"testing"
)

// G-S11f (S11, docs/tasks/task-multimodal-support-2026-10.md): several images in one message keep their order on every
// API surface. Each parser records where its images sat among the text parts (imageRef.at), and placeImageBlocks puts
// each block back there.

func blocksFor(n int) []string {
	b := make([]string, n)
	for i := range b {
		b[i] = "<IMG" + string(rune('A'+i)) + ">"
	}
	return b
}

func TestTwoImages_partOrderOpenAI(t *testing.T) {
	msgs := chatMsgs(t, `{"messages":[{"role":"user","content":[
	  {"type":"text","text":"Compare "},{"type":"image_url","image_url":{"url":"`+dataURI("a")+`"}},
	  {"type":"text","text":" with "},{"type":"image_url","image_url":{"url":"`+dataURI("b")+`"}},{"type":"text","text":"."}]}]}`)
	imgs, err := chatImages(msgs)
	if err != nil || len(imgs) != 2 {
		t.Fatalf("chatImages: %v, %d images", err, len(imgs))
	}
	text, ordered := chatMediaText(msgs)
	if !ordered || text != "Compare  with ." {
		t.Fatalf("chatMediaText = %q, %v", text, ordered)
	}
	if got := placeImageBlocks(text, text, ordered, imgs, blocksFor(2)); got != "Compare <IMGA> with <IMGB>." {
		t.Errorf("placed %q", got)
	}
	// Adjacent, no text between: two blocks, in order, still two.
	adj := chatMsgs(t, `{"messages":[{"role":"user","content":[{"type":"text","text":"Q"},
	  {"type":"image_url","image_url":{"url":"`+dataURI("a")+`"}},{"type":"image_url","image_url":{"url":"`+dataURI("b")+`"}}]}]}`)
	ai, _ := chatImages(adj)
	at, ao := chatMediaText(adj)
	if got := placeImageBlocks(at, at, ao, ai, blocksFor(2)); got != "Q<IMGA><IMGB>" {
		t.Errorf("adjacent: placed %q", got)
	}
	// The media message is not the last user message: every block leads the last user turn, as one image always did.
	later := chatMsgs(t, `{"messages":[{"role":"user","content":[{"type":"text","text":"see "},
	  {"type":"image_url","image_url":{"url":"`+dataURI("a")+`"}}]},{"role":"assistant","content":"ok"},{"role":"user","content":"and?"}]}`)
	li, _ := chatImages(later)
	lt, lo := chatMediaText(later)
	if lo {
		t.Error("an image in an earlier message reported as the last user message's")
	}
	if got := placeImageBlocks("and?", lt, lo, li, blocksFor(1)); got != "<IMGA>and?" {
		t.Errorf("not the last user message: placed %q", got)
	}
	// A turn merged onto earlier text: the offsets apply from where the media message's text starts.
	if got := placeImageBlocks("earlier\nCompare  with .", text, true, imgs, blocksFor(2)); got != "earlier\nCompare <IMGA> with <IMGB>." {
		t.Errorf("merged turn: placed %q", got)
	}
}

func TestTwoImages_partOrderAnthropic(t *testing.T) {
	var req anthropicReq
	b64 := strings.TrimPrefix(dataURI("a"), "data:image/png;base64,")
	body := `{"messages":[{"role":"user","content":[{"type":"text","text":"Compare "},
	  {"type":"image","source":{"type":"base64","media_type":"image/png","data":"` + b64 + `"}},
	  {"type":"text","text":" with "},
	  {"type":"image","source":{"type":"base64","media_type":"image/png","data":"` + b64 + `"}},{"type":"text","text":"."}]}]}`
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		t.Fatal(err)
	}
	imgs, err := anthropicImages(&req)
	if err != nil || len(imgs) != 2 {
		t.Fatalf("anthropicImages: %v, %d images", err, len(imgs))
	}
	text, ordered := anthropicMediaText(&req)
	if !ordered || text != "Compare  with ." {
		t.Fatalf("anthropicMediaText = %q, %v", text, ordered)
	}
	if got := placeImageBlocks(text, text, ordered, imgs, blocksFor(2)); got != "Compare <IMGA> with <IMGB>." {
		t.Errorf("placed %q", got)
	}
}

func TestTwoImages_partOrderResponses(t *testing.T) {
	msgs, err := responseInputToMessages(json.RawMessage(`[{"role":"user","content":[{"type":"input_text","text":"Compare "},
	  {"type":"input_image","image_url":"` + dataURI("a") + `"},{"type":"input_text","text":" with "},
	  {"type":"input_image","image_url":"` + dataURI("b") + `"},{"type":"input_text","text":"."}]}]`))
	if err != nil {
		t.Fatal(err)
	}
	imgs, err := chatImages(msgs)
	if err != nil || len(imgs) != 2 || string(imgs[1].data) != "b" {
		t.Fatalf("an input_image was dropped: %v, %d images", err, len(imgs))
	}
	text, ordered := chatMediaText(msgs)
	if got := placeImageBlocks(text, text, ordered, imgs, blocksFor(2)); got != "Compare <IMGA> with <IMGB>." {
		t.Errorf("placed %q", got)
	}
	// Text-only content is still the plain string it always was.
	plain, _ := responseInputToMessages(json.RawMessage(`[{"role":"user","content":[{"type":"input_text","text":"hi"}]}]`))
	if string(plain[0].Content) != `"hi"` {
		t.Errorf("text-only content became %s", plain[0].Content)
	}
	if _, err := responseInputToMessages(json.RawMessage(`[{"role":"user","content":[{"type":"input_image","file_id":"f1"}]}]`)); err == nil {
		t.Error("an input_image by file_id was accepted")
	}
}

// The 400s: a ninth image names the cap; GLM-OCR's second image names the family; audio is the only media or none.
func TestTwoImages_refusals(t *testing.T) {
	if m := tooManyImages(maxImagesPerTurn + 1); !strings.Contains(m, "at most 8 images") {
		t.Errorf("the cap message %q does not name the cap", m)
	}
	two := []imageRef{{data: []byte("a")}, {data: []byte("b")}}
	_, err := (&loadedModel{glm: &glmOcrTower{}}).visionPromptN(nil, "", nil, two, false, "")
	if err == nil || !strings.Contains(err.Error(), "GLM-OCR takes one image") {
		t.Errorf("GLM-OCR with two images: %v", err)
	}
	_, err = (&loadedModel{}).visionPromptN(nil, "", nil, []imageRef{{data: []byte("a")}, {data: []byte("w"), audio: true}}, false, "")
	if err == nil || !strings.Contains(err.Error(), "audio clip must be the only media") {
		t.Errorf("an image and a clip: %v", err)
	}
}
