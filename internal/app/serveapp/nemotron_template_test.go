package serveapp

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"testing"

	"github.com/townsendmerino/goinfer/internal/app/loadflags"
)

// nemotronCheckpoint is tinyChatCheckpoint with the markers in its vocabulary and Nemotron-H 4B's own chat template, read
// from the golden HF rendered, so what is loaded is what the real checkpoint ships.
func nemotronCheckpoint(t *testing.T) string {
	t.Helper()
	dir := tinyChatCheckpoint(t)
	var tk struct {
		Model struct {
			Type   string         `json:"type"`
			Vocab  map[string]int `json:"vocab"`
			Merges []string       `json:"merges"`
		} `json:"model"`
		PreTokenizer map[string]any   `json:"pre_tokenizer"`
		Decoder      map[string]any   `json:"decoder"`
		AddedTokens  []map[string]any `json:"added_tokens"`
	}
	if err := json.Unmarshal(tinyChatTokenizer(t), &tk); err != nil {
		t.Fatal(err)
	}
	for _, s := range []string{"<SPECIAL_10>", "<SPECIAL_11>"} {
		id := len(tk.Model.Vocab)
		tk.Model.Vocab[s] = id
		tk.AddedTokens = append(tk.AddedTokens, map[string]any{"id": id, "content": s, "special": true})
	}
	raw, err := json.Marshal(tk)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), raw, 0o644); err != nil {
		t.Fatal(err)
	}
	gold, err := os.ReadFile(filepath.Join("../../../testdata/chat_goldens/nemotron_h.json"))
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		ChatTemplate string `json:"chat_template"`
	}
	if err := json.Unmarshal(gold, &g); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "chat_template.jinja"), []byte(g.ChatTemplate), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

// TestServe_nemotronRendersItsMarkersAndStops: through the real load path and Chat Completions, a checkpoint carrying
// Nemotron-H's template renders the turn markers as single tokens, and the real generator ends a reply at <SPECIAL_11>.
// The prompt is read through the scripted seam, which does not apply stops; the stop is read on the tiny model, with the
// marker forced by logit_bias.
func TestServe_nemotronRendersItsMarkersAndStops(t *testing.T) {
	srv, err := newServer(config{models: modelFlag{{name: "tiny", path: nemotronCheckpoint(t)}}, load: loadflags.Flags{Backend: "cpu", Quant: "f32"}})
	if err != nil {
		t.Fatal(err)
	}
	lm := srv.models["tiny"]
	if lm.tmpl == nil || lm.tmpl.Name() != "nemotron_h" {
		t.Fatalf("template %v, want nemotron_h: the load fell back to a raw render", lm.tmpl)
	}
	encode := func(s string) []int {
		ids, err := lm.tk.Encode(s, false)
		if err != nil {
			t.Fatal(err)
		}
		return ids
	}
	s10, ok10 := lm.tk.TokenID("<SPECIAL_10>")
	s11, ok11 := lm.tk.TokenID("<SPECIAL_11>")
	if !ok10 || !ok11 {
		t.Fatal("the fixture's vocabulary lost a marker")
	}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", srv.handleChat)
	ts := httptest.NewServer(mux)
	defer ts.Close()
	type reply struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	ask := func(bias map[string]float64) reply {
		t.Helper()
		body, err := json.Marshal(map[string]any{"model": "tiny", "temperature": 0, "max_tokens": 4, "logit_bias": bias,
			"messages": []any{map[string]any{"role": "system", "content": "Sys"}, map[string]any{"role": "user", "content": "Hi"}}})
		if err != nil {
			t.Fatal(err)
		}
		resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", bytes.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		var r reply
		if err := json.NewDecoder(resp.Body).Decode(&r); err != nil {
			t.Fatal(err)
		}
		if resp.StatusCode != http.StatusOK || len(r.Choices) != 1 {
			t.Fatalf("status %d, %d choices", resp.StatusCode, len(r.Choices))
		}
		return r
	}

	// The do-nothing arm: the same forcing on an ordinary token runs to the budget, so the stop below is the marker's doing.
	o := encode("o")[0]
	if r := ask(map[string]float64{strconv.Itoa(o): 100}); r.Choices[0].Message.Content != "oooo" || r.Choices[0].FinishReason != "length" {
		t.Errorf("forced o: %+v, want oooo and length", r.Choices[0])
	}
	if r := ask(map[string]float64{strconv.Itoa(s11): 100}); r.Choices[0].Message.Content != "" || r.Choices[0].FinishReason != "stop" {
		t.Errorf("forced <SPECIAL_11>: %+v, want an empty reply and stop", r.Choices[0])
	}

	lm.scripted = encode("ok")
	var prompt []int
	lm.onScripted = func(p []int) { prompt = slices.Clone(p) }
	ask(nil)
	want := slices.Concat([]int{s10}, encode("System\nSys\n\n"), []int{s11}, encode("User\nHi\n"), []int{s11}, encode("Assistant\n"))
	if !slices.Equal(prompt, want) {
		t.Errorf("prompt ids\n got %v\nwant %v", prompt, want)
	}
}
