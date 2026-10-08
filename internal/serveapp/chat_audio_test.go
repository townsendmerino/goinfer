package serveapp

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
)

// wavBytes is a RIFF WAV of n silent 16-bit mono samples at rate Hz.
func wavBytes(n, rate int) []byte {
	b := make([]byte, 44+2*n)
	copy(b[0:], "RIFF")
	binary.LittleEndian.PutUint32(b[4:], uint32(36+2*n))
	copy(b[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(b[16:], 16)
	binary.LittleEndian.PutUint16(b[20:], 1) // PCM
	binary.LittleEndian.PutUint16(b[22:], 1) // mono
	binary.LittleEndian.PutUint32(b[24:], uint32(rate))
	binary.LittleEndian.PutUint32(b[28:], uint32(2*rate))
	binary.LittleEndian.PutUint16(b[32:], 2)
	binary.LittleEndian.PutUint16(b[34:], 16)
	copy(b[36:], "data")
	binary.LittleEndian.PutUint32(b[40:], uint32(2*n))
	return b
}

func audioPart(data, format string) string {
	p := map[string]any{"type": "input_audio", "input_audio": map[string]string{"data": data, "format": format}}
	b, _ := json.Marshal(p)
	return string(b)
}

// TestContentPartsImages_inputAudio: an OpenAI input_audio part (S5) is a media item flagged audio, from bare base64 or
// a data: URI; a format other than wav, or no data, is refused by name.
func TestContentPartsImages_inputAudio(t *testing.T) {
	wav := wavBytes(160, 16000)
	b64 := base64.StdEncoding.EncodeToString(wav)
	for name, part := range map[string]string{
		"bare base64": audioPart(b64, "wav"),
		"data URI":    audioPart("data:audio/wav;base64,"+b64, ""),
	} {
		refs, err := contentPartsImages(json.RawMessage(`[` + part + `,{"type":"text","text":"hi"}]`))
		if err != nil || len(refs) != 1 || !refs[0].audio || string(refs[0].data) != string(wav) {
			t.Errorf("%s: %d refs, err %v", name, len(refs), err)
		}
	}
	for name, part := range map[string]string{
		"mp3":     audioPart(b64, "mp3"),
		"no data": audioPart("", "wav"),
		"garbage": audioPart("!!not base64!!", "wav"),
	} {
		if _, err := contentPartsImages(json.RawMessage(`[` + part + `]`)); err == nil || !strings.Contains(err.Error(), "input_audio") {
			t.Errorf("%s: want an input_audio error, got %v", name, err)
		}
	}
}

// TestOmitChatHistory_audio: the generation takes one media span, so an earlier message's clip is replaced by a note
// (as an earlier image is), the newest media item is kept, and an image followed by a clip keeps only the clip.
func TestOmitChatHistory_audio(t *testing.T) {
	a1 := audioPart(base64.StdEncoding.EncodeToString(wavBytes(160, 16000)), "wav")
	a2 := audioPart(base64.StdEncoding.EncodeToString(wavBytes(320, 16000)), "wav")
	img := `{"type":"image_url","image_url":{"url":"data:image/png;base64,` + base64.StdEncoding.EncodeToString([]byte("png")) + `"}}`
	msgs := []chatMessage{
		{Role: "user", Content: json.RawMessage(`[` + a1 + `]`)},
		{Role: "assistant", Content: rawStr("ok")},
		{Role: "user", Content: json.RawMessage(`[` + img + `]`)},
		{Role: "assistant", Content: rawStr("ok")},
		{Role: "user", Content: json.RawMessage(`[` + a2 + `]`)},
	}
	if n := omitChatHistoryImages(msgs); n != 2 {
		t.Fatalf("omitted %d, want 2 (the earlier clip and the image)", n)
	}
	media, err := chatImages(msgs)
	if err != nil || len(media) != 1 || !media[0].audio || len(media[0].data) != len(wavBytes(320, 16000)) {
		t.Fatalf("after omission: %d media items, err %v; want the newest clip only", len(media), err)
	}
	if !strings.Contains(msgs[0].text(), olderAudioNote) || !strings.Contains(msgs[2].text(), olderImageNote) {
		t.Errorf("the notes are missing: %q / %q", msgs[0].text(), msgs[2].text())
	}
}

// TestGemma4AudioPrompt_refusals: a model without an audio tower is told so; a clip over 30 s and a WAV that is not
// 16 kHz mono 16-bit are refused by name, before any tower runs.
func TestGemma4AudioPrompt_refusals(t *testing.T) {
	turns := func() []chat.Turn { return []chat.Turn{{Role: "user", Content: "Transcribe this audio."}} }
	tmpl := chat.Gemma4()
	lm := &loadedModel{tmpl: tmpl}
	clip := imageRef{data: wavBytes(16000, 16000), audio: true}
	if _, err := lm.visionPromptUncached(tmpl, "", turns(), clip); err == nil || !strings.Contains(err.Error(), "no audio tower") {
		t.Errorf("no tower: %v", err)
	}
	lm.gemma4AudioDir = "unused"
	long := imageRef{data: wavBytes(31*16000, 16000), audio: true}
	if _, err := lm.gemma4AudioPrompt(tmpl, "", turns(), 0, long); err == nil || !strings.Contains(err.Error(), "30 s") {
		t.Errorf("31 s clip: %v", err)
	}
	wrong := imageRef{data: wavBytes(16000, 44100), audio: true}
	if _, err := lm.gemma4AudioPrompt(tmpl, "", turns(), 0, wrong); err == nil || !strings.Contains(err.Error(), "16000 Hz") {
		t.Errorf("44.1 kHz clip: %v", err)
	}
}
