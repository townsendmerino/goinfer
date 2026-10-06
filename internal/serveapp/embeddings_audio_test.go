package serveapp

import (
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"strings"
	"testing"
)

// mediaStub embeds images and audio, recording each call.
type mediaStub struct {
	imgStub
}

func (e mediaStub) EmbedAudioTask(samples []float32, text, prompt string) ([]float32, int, error) {
	*e.calls = append(*e.calls, fmt.Sprintf("aud:%d|%s|%s", len(samples), text, prompt))
	return e.vec(), 5, nil
}

func newMediaTestServer() (*server, *[]string) {
	s, calls := newImageTestServer()
	s.embed = mediaStub{imgStub: s.embed.(imgStub)}
	return s, calls
}

// testWAV is a 16-bit PCM WAV of n samples at rate Hz with ch channels.
func testWAV(n, rate, ch int) []byte {
	data := make([]byte, 2*n*ch)
	b := make([]byte, 44, 44+len(data))
	copy(b[0:], "RIFF")
	binary.LittleEndian.PutUint32(b[4:], uint32(36+len(data)))
	copy(b[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(b[16:], 16)
	binary.LittleEndian.PutUint16(b[20:], 1)
	binary.LittleEndian.PutUint16(b[22:], uint16(ch))
	binary.LittleEndian.PutUint32(b[24:], uint32(rate))
	binary.LittleEndian.PutUint32(b[28:], uint32(rate*2*ch))
	binary.LittleEndian.PutUint16(b[32:], uint16(2*ch))
	binary.LittleEndian.PutUint16(b[34:], 16)
	copy(b[36:], "data")
	binary.LittleEndian.PutUint32(b[40:], uint32(len(data)))
	return append(b, data...)
}

func wavB64(n, rate, ch int) string { return base64.StdEncoding.EncodeToString(testWAV(n, rate, ch)) }

// TestEmbeddings_audioShapes: the object form ({"audio","text"}: a data: URI or bare base64 of a WAV) and the OpenAI
// input_audio part (alone, or before text), mixed with images and strings in one request, are each one input in
// order; each clip reaches the audio embedder as its samples, with its text and the request's prompt.
func TestEmbeddings_audioShapes(t *testing.T) {
	s, calls := newMediaTestServer()
	body := `{"input_type":"query","input":[` +
		`"plain text",` +
		`{"audio":"data:audio/wav;base64,` + wavB64(1600, 16000, 1) + `"},` +
		`{"audio":"` + wavB64(800, 16000, 1) + `","text":"a dog"},` +
		`[{"type":"input_audio","input_audio":{"data":"` + wavB64(400, 16000, 1) + `","format":"wav"}},{"type":"text","text":"rain"}],` +
		`{"image":"` + embedDataURI("A") + `"}` +
		`]}`
	rr := postEmbed(t, s, body)
	if rr.Code != 200 {
		t.Fatalf("status %d: %s", rr.Code, rr.Body.String())
	}
	want := "aud:1600||query,aud:800|a dog|query,aud:400|rain|query,img:A||query"
	if got := strings.Join(*calls, ","); got != want {
		t.Errorf("calls %s, want %s", got, want)
	}
}

// TestEmbeddings_audioRefusals: a WAV at another rate, stereo, not a WAV, over 30 s, an image and audio in one input,
// a non-wav input_audio format, text before the audio, too many clips, and audio sent to a model without audio input
// are 400s naming the problem, and the embedder never runs.
func TestEmbeddings_audioRefusals(t *testing.T) {
	cases := []struct{ body, want string }{
		{`{"input":[{"audio":"` + wavB64(1600, 44100, 1) + `"}]}`, "16000 Hz"},
		{`{"input":[{"audio":"` + wavB64(1600, 16000, 2) + `"}]}`, "mono"},
		{`{"input":[{"audio":"` + base64.StdEncoding.EncodeToString([]byte("not a wav at all")) + `"}]}`, "RIFF"},
		{`{"input":[{"audio":"` + wavB64(31*16000, 16000, 1) + `"}]}`, "over the 30 s"},
		{`{"input":[{"audio":"` + wavB64(1600, 16000, 1) + `","image":"` + embedDataURI("A") + `"}]}`, "not both"},
		{`{"input":[{"type":"input_audio","input_audio":{"data":"` + wavB64(1600, 16000, 1) + `","format":"mp3"}}]}`, "only wav"},
		{`{"input":[[{"type":"text","text":"x"},{"type":"input_audio","input_audio":{"data":"` + wavB64(1600, 16000, 1) + `"}}]]}`, "text before the audio"},
	}
	for _, c := range cases {
		s, calls := newMediaTestServer()
		rr := postEmbed(t, s, c.body)
		if rr.Code != 400 || !strings.Contains(rr.Body.String(), c.want) {
			t.Errorf("%s: status %d, body %s (want 400 with %q)", c.body[:50], rr.Code, rr.Body.String(), c.want)
		}
		if len(*calls) != 0 {
			t.Errorf("%s: the embedder ran on a refused request", c.body[:50])
		}
	}
	var many []string
	for range maxEmbedAudio + 1 {
		many = append(many, `{"audio":"`+wavB64(320, 16000, 1)+`"}`)
	}
	s, _ := newMediaTestServer()
	if rr := postEmbed(t, s, `{"input":[`+strings.Join(many, ",")+`]}`); rr.Code != 400 || !strings.Contains(rr.Body.String(), "too many audio clips") {
		t.Errorf("%d clips: status %d, %s", maxEmbedAudio+1, rr.Code, rr.Body.String())
	}
	s, _ = newImageTestServer() // images, no audio
	if rr := postEmbed(t, s, `{"input":[{"audio":"`+wavB64(1600, 16000, 1)+`"}]}`); rr.Code != 400 || !strings.Contains(rr.Body.String(), "takes no audio input") {
		t.Errorf("audio on an image-only embedder: status %d, %s", rr.Code, rr.Body.String())
	}
}
