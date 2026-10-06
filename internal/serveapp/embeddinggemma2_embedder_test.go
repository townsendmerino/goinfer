package serveapp

import (
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/embeddinggemma2"
)

// taskStub is a taskEmbedder with EmbeddingGemma 2's query/document prompts and one more. It records the prompt each
// call was given, so a test can see which one the handler chose.
type taskStub struct {
	stubEncoder
	prompts map[string]string
	last    *string
}

func (e taskStub) PromptNames() []string {
	var n []string
	for k := range e.prompts {
		n = append(n, k)
	}
	sort.Strings(n)
	return n
}

func (e taskStub) PromptText(name string) (string, error) {
	if p, ok := e.prompts[name]; ok {
		return p, nil
	}
	return "", fmt.Errorf("unknown task %q; this model's tasks are %s, or none", name, strings.Join(e.PromptNames(), ", "))
}

func (e taskStub) EncodeTasks(texts []string, prompt string) ([][]float32, []int, error) {
	*e.last = prompt
	vecs := make([][]float32, len(texts))
	counts := make([]int, len(texts))
	for i := range texts {
		vecs[i], counts[i] = e.vec(), 3
	}
	return vecs, counts, nil
}

func newTaskTestServer() (*server, *string) {
	last := new(string)
	*last = "<not called>"
	enc := taskStub{stubEncoder: stubEncoder{dim: 8}, last: last, prompts: map[string]string{
		"query": "task: search result | query: ", "document": "title: none | text: ", "STS": "task: sentence similarity | query: ",
	}}
	return &server{embed: enc, embedDim: 8, embedID: "eg2", embedWidths: []int{8, 4, 2}}, last
}

// TestEmbeddings_taskPromptChoice is Gate 3's prompt rule (docs/tasks/task-embeddinggemma2.md, owner decision
// 2026-10-06): with no task and no input_type the model gets no prompt (sentence-transformers' own default); input_type
// maps onto the model's query and document prompts; task names any of its prompts, and "none" means none. The prompt
// applied is echoed in goinfer_task and the X-Goinfer-Embedding-Task header.
func TestEmbeddings_taskPromptChoice(t *testing.T) {
	for _, c := range []struct {
		body, wantPrompt, wantName string
	}{
		{`{"input":"a"}`, "", "none"},
		{`{"input":"a","input_type":"query"}`, "query", "query"},
		{`{"input":"a","input_type":"document"}`, "document", "document"},
		{`{"input":"a","task":"STS"}`, "STS", "STS"},
		{`{"input":"a","task":"none","input_type":"query"}`, "", "none"},
		{`{"input":"a","task":"STS","input_type":"query"}`, "STS", "STS"},
	} {
		s, last := newTaskTestServer()
		rr := postEmbed(t, s, c.body)
		if rr.Code != 200 {
			t.Fatalf("%s: status %d: %s", c.body, rr.Code, rr.Body.String())
		}
		if *last != c.wantPrompt {
			t.Errorf("%s: encoded under prompt %q, want %q", c.body, *last, c.wantPrompt)
		}
		var resp struct {
			Task struct{ Name, Prompt string } `json:"goinfer_task"`
		}
		if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
			t.Fatal(err)
		}
		if resp.Task.Name != c.wantName || rr.Header().Get("X-Goinfer-Embedding-Task") != c.wantName {
			t.Errorf("%s: echoed task %q (header %q), want %q", c.body, resp.Task.Name, rr.Header().Get("X-Goinfer-Embedding-Task"), c.wantName)
		}
		if wantText, _ := s.embed.(taskEmbedder).PromptText(c.wantPrompt); c.wantPrompt != "" && resp.Task.Prompt != wantText {
			t.Errorf("%s: echoed prompt %q, want %q", c.body, resp.Task.Prompt, wantText)
		}
	}
}

// TestEmbeddings_taskRefusals: an unknown task is a 400 naming the model's tasks; a task sent to an embedder with no
// task prompts is a 400 too, not silently ignored.
func TestEmbeddings_taskRefusals(t *testing.T) {
	s, _ := newTaskTestServer()
	rr := postEmbed(t, s, `{"input":"a","task":"Retrieval"}`)
	if rr.Code != 400 || !strings.Contains(rr.Body.String(), "STS, document, query") {
		t.Errorf("unknown task: status %d, body %s (want 400 naming the tasks)", rr.Code, rr.Body.String())
	}
	rr = postEmbed(t, newEmbedTestServer(), `{"input":"a","task":"query"}`)
	if rr.Code != 400 || !strings.Contains(rr.Body.String(), "no named task prompts") {
		t.Errorf("task on a plain encoder: status %d, body %s (want 400)", rr.Code, rr.Body.String())
	}
	if rr := postEmbed(t, newEmbedTestServer(), `{"input":"a"}`); rr.Code != 200 || strings.Contains(rr.Body.String(), "goinfer_task") {
		t.Errorf("a plain encoder's response: status %d, body %s (want 200 with no goinfer_task)", rr.Code, rr.Body.String())
	}
}

// TestEmbeddings_matryoshkaWidthSet: with a trained width set, dimensions takes exactly those widths and names them
// when refused, and the returned vector has the asked width.
func TestEmbeddings_matryoshkaWidthSet(t *testing.T) {
	s, _ := newTaskTestServer()
	for _, d := range []int{8, 4, 2} {
		rr := postEmbed(t, s, fmt.Sprintf(`{"input":"a","dimensions":%d}`, d))
		var resp struct {
			Data []struct{ Embedding []float32 } `json:"data"`
		}
		if rr.Code != 200 || json.Unmarshal(rr.Body.Bytes(), &resp) != nil || len(resp.Data[0].Embedding) != d {
			t.Errorf("dimensions %d: status %d, body %s", d, rr.Code, rr.Body.String())
		}
	}
	for _, d := range []int{3, 6, 1} {
		rr := postEmbed(t, s, fmt.Sprintf(`{"input":"a","dimensions":%d}`, d))
		if rr.Code != 400 || !strings.Contains(rr.Body.String(), "[8 4 2]") {
			t.Errorf("dimensions %d: status %d, body %s (want 400 naming [8 4 2])", d, rr.Code, rr.Body.String())
		}
	}
}

// TestLoadEmbeddingGemma2_refusesAQuantItCannotHonour: -embed-quant q8 on an EmbeddingGemma 2 checkpoint is refused by
// name before anything loads, not dropped (the M-17 class).
func TestLoadEmbeddingGemma2_refusesAQuantItCannotHonour(t *testing.T) {
	s := &server{}
	err := s.loadEmbeddingGemma2(config{embedPath: "../../testdata/embeddinggemma2-tiny", embedQuant: "q8"})
	if err == nil || !strings.Contains(err.Error(), "f32 only") {
		t.Fatalf("q8: %v, want a refusal naming f32", err)
	}
	if s.embed != nil {
		t.Fatal("an embedder was attached after the refusal")
	}
}

// TestLoadEmbeddingGemma2_refusesAnUnknownResize: an -embed-image-resize the encoder does not have is refused by name
// before anything loads (the tiny fixture has no tokenizer, so reaching the load would fail differently).
func TestLoadEmbeddingGemma2_refusesAnUnknownResize(t *testing.T) {
	s := &server{}
	err := s.loadEmbeddingGemma2(config{embedPath: "../../testdata/embeddinggemma2-tiny", embedResize: "lanczos"})
	if err == nil || !strings.Contains(err.Error(), "-embed-image-resize") || !strings.Contains(err.Error(), "bicubic") {
		t.Fatalf("lanczos: %v, want a refusal naming the flag and the choices", err)
	}
	if s.embed != nil {
		t.Fatal("an embedder was attached after the refusal")
	}
}

// TestIsEmbeddingGemma2: the dispatch in loadEncoder recognises the checkpoint by its config.json, and nothing else.
func TestIsEmbeddingGemma2(t *testing.T) {
	if !isEmbeddingGemma2("../../testdata/embeddinggemma2-tiny") {
		t.Error("the tiny EmbeddingGemma 2 fixture is not recognised")
	}
	if isEmbeddingGemma2(t.TempDir()) {
		t.Error("an empty directory is recognised")
	}
}

// eg2AccelStub stands in for the encoder in chooseEG2Device's test: err, when set, is what UseAccelerator returns.
type eg2AccelStub struct {
	err  error
	used []string
}

func (s *eg2AccelStub) UseAccelerator(name string) (embeddinggemma2.Accelerator, error) {
	s.used = append(s.used, name)
	return nil, s.err
}

// TestChooseEG2Device: serve puts EmbeddingGemma 2 on Metal when its backend resolved to metal, falls back to the CPU
// with the reason when Metal declines (and refuses instead under -require-backend), and runs it on the CPU, saying so,
// for every other backend.
func TestChooseEG2Device(t *testing.T) {
	ok := &eg2AccelStub{}
	if where, err := chooseEG2Device(ok, "metal", false); err != nil || where != "Metal" || len(ok.used) != 1 || ok.used[0] != "metal" {
		t.Errorf("metal: %q %v %v", where, err, ok.used)
	}
	bad := &eg2AccelStub{err: errors.New("no device")}
	if where, err := chooseEG2Device(bad, "metal", false); err != nil || !strings.Contains(where, "Metal declined: no device") {
		t.Errorf("metal declined: %q %v", where, err)
	}
	if _, err := chooseEG2Device(bad, "metal", true); err == nil {
		t.Error("metal declined under -require-backend did not refuse")
	}
	for _, be := range []string{"cpu", ""} {
		s := &eg2AccelStub{}
		if where, err := chooseEG2Device(s, be, true); err != nil || where != "CPU" || len(s.used) != 0 {
			t.Errorf("%q: %q %v %v", be, where, err, s.used)
		}
	}
	s := &eg2AccelStub{}
	if where, err := chooseEG2Device(s, "cuda", false); err != nil || !strings.Contains(where, "Metal only") || len(s.used) != 0 {
		t.Errorf("cuda: %q %v %v", where, err, s.used)
	}
	if _, err := chooseEG2Device(s, "cuda", true); err == nil {
		t.Error("cuda under -require-backend did not refuse")
	}
}
