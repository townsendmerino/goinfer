package serveapp

import (
	"encoding/base64"
	"encoding/json"
	"strings"
	"testing"
)

// imgStub is a taskStub that also embeds images, recording each call.
type imgStub struct {
	taskStub
	calls *[]string
}

func (e imgStub) EmbedImageTask(img []byte, text, prompt string) ([]float32, int, error) {
	*e.calls = append(*e.calls, "img:"+string(img)+"|"+text+"|"+prompt)
	return e.vec(), 7, nil
}

func newImageTestServer() (*server, *[]string) {
	s, _ := newTaskTestServer()
	calls := &[]string{}
	s.embed = imgStub{taskStub: s.embed.(taskStub), calls: calls}
	return s, calls
}

func embedDataURI(b string) string {
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString([]byte(b))
}

// TestEmbeddings_imageShapes: the object form ({"image","text"}, data URI or bare base64), the OpenAI part form (an
// array of parts as one input, or a lone part), and plain strings, mixed in one request, are each one input in order;
// the image goes to the image embedder with its text and the request's prompt.
func TestEmbeddings_imageShapes(t *testing.T) {
	s, calls := newImageTestServer()
	body := `{"input_type":"query","input":[` +
		`"plain text",` +
		`{"image":"` + embedDataURI("A") + `"},` +
		`{"image":"` + base64.StdEncoding.EncodeToString([]byte("B")) + `","text":"caption b"},` +
		`[{"type":"image_url","image_url":{"url":"` + embedDataURI("C") + `"}},{"type":"text","text":"caption c"}],` +
		`{"type":"image_url","image_url":{"url":"` + embedDataURI("D") + `"}}` +
		`]}`
	rr := postEmbed(t, s, body)
	if rr.Code != 200 {
		t.Fatalf("status %d: %s", rr.Code, rr.Body.String())
	}
	var resp struct {
		Data  []struct{ Index int } `json:"data"`
		Usage struct {
			Prompt int `json:"prompt_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if len(resp.Data) != 5 {
		t.Fatalf("%d vectors, want 5", len(resp.Data))
	}
	want := []string{"img:A||query", "img:B|caption b|query", "img:C|caption c|query", "img:D||query"}
	if strings.Join(*calls, ",") != strings.Join(want, ",") {
		t.Errorf("image calls %v, want %v", *calls, want)
	}
	if resp.Usage.Prompt != 4*7+3 { // four images at 7 tokens each, plus the text stub's 3
		t.Errorf("prompt_tokens %d, want %d", resp.Usage.Prompt, 4*7+3)
	}
}

// TestEmbeddings_imageRefusals: text before the image, two images in one input, a remote URL, too many images, and
// an image sent to a model with no image input are 400s naming the problem; a text-only request never reaches the
// image embedder.
func TestEmbeddings_imageRefusals(t *testing.T) {
	cases := []struct{ body, want string }{
		{`{"input":[[{"type":"text","text":"x"},{"type":"image_url","image_url":{"url":"` + embedDataURI("A") + `"}}]]}`, "text before the image"},
		{`{"input":[[{"type":"image_url","image_url":{"url":"` + embedDataURI("A") + `"}},{"type":"image_url","image_url":{"url":"` + embedDataURI("B") + `"}}]]}`, "one image or audio clip per input"},
		{`{"input":[{"type":"image_url","image_url":{"url":"https://example.com/a.png"}}]}`, "data: URI"},
		{`{"input":[{"image":"not base64 !!"}]}`, "neither a data: URI nor valid base64"},
		{`{"input":[{"image":"` + embedDataURI("A") + `","colour":"red"}]}`, "unknown field"},
	}
	for _, c := range cases {
		s, calls := newImageTestServer()
		rr := postEmbed(t, s, c.body)
		if rr.Code != 400 || !strings.Contains(rr.Body.String(), c.want) {
			t.Errorf("%s: status %d, body %s (want 400 with %q)", c.body[:40], rr.Code, rr.Body.String(), c.want)
		}
		if len(*calls) != 0 {
			t.Errorf("%s: the image embedder ran on a refused request", c.body[:40])
		}
	}
	var many []string
	for range maxEmbedImages + 1 {
		many = append(many, `{"image":"`+embedDataURI("A")+`"}`)
	}
	s, _ := newImageTestServer()
	if rr := postEmbed(t, s, `{"input":[`+strings.Join(many, ",")+`]}`); rr.Code != 400 || !strings.Contains(rr.Body.String(), "too many images") {
		t.Errorf("%d images: status %d, %s", maxEmbedImages+1, rr.Code, rr.Body.String())
	}
	if rr := postEmbed(t, newEmbedTestServer(), `{"input":[{"image":"`+embedDataURI("A")+`"}]}`); rr.Code != 400 || !strings.Contains(rr.Body.String(), "takes no image input") {
		t.Errorf("image on a text-only encoder: status %d, %s", rr.Code, rr.Body.String())
	}
	s, calls := newImageTestServer()
	if rr := postEmbed(t, s, `{"input":["a","b"]}`); rr.Code != 200 || len(*calls) != 0 {
		t.Errorf("a text-only request: status %d, image calls %v", rr.Code, *calls)
	}
}
