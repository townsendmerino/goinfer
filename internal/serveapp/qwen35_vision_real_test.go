//go:build realckpt

package serveapp

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/internal/loadflags"
)

// TestServe_qwen35Image_G4 is P8a's G4 (docs/measurements/p8a-qwen35-vl-2026-09/preregistration.md) on
// the real Qwen3.5-0.8B, through the real serve stack: an OpenAI `image_url` data-URI content part
// -> handleChat -> preprocess -> lazily-loaded tower -> GenerateQwenVL, temperature 0.
//
//	(a) the prompt serve builds (chat template + image block + tokenizer) is EXACTLY the ids HF builds
//	    from Qwen3.5's own template with the empty think block removed (serve_{A,B,C}.json), and the
//	    reply text is HF's 32 greedy tokens over those ids, decoded, for each of the 3 images. The
//	    think block: Qwen3.5's template emits `<think>\n\n</think>\n\n` after the generation prompt
//	    unless enable_thinking is true; goinfer's generic ChatML renderer emits nothing there. That is
//	    a PRE-EXISTING text-path difference and is pinned here as a named 4-token delta, not fixed:
//	    fixing it changes every Qwen3.5 text turn (P8a gate G3 says they must not move);
//	(b) usage.prompt_tokens is that id count, completion_tokens is 32;
//	(c) the same request twice returns identical text (cold determinism).
//
// The end-to-end "reused == cold" cell is deliberately NOT here — see the pre-registration: with CPU
// decode an image turn never populates the resident cache, so it would pass vacuously. The decision
// function that would govern it is gated in decoder (TestResidentReuseLen_recurrentImageClaims).
//
//	GOINFER_HEAVY_TESTS=1 go test -tags realckpt ./internal/serveapp/ -run TestServe_qwen35Image_G4 -v -timeout 30m
func TestServe_qwen35Image_G4(t *testing.T) {
	requireHeavyModel(t)
	home := os.Getenv("HOME")
	ckpt := filepath.Join(home, "models", "qwen3.5-0.8b")
	gdir := filepath.Join(home, "models", "qwen35vl_g2")
	for _, p := range []string{filepath.Join(ckpt, "config.json"), filepath.Join(gdir, "golden_A.json")} {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("asset missing: %s", p)
		}
	}
	srv, err := newServer(config{models: modelFlag{{name: "q35", path: ckpt}}, load: loadflags.Flags{Backend: "cpu"}, kvSessions: 2})
	if err != nil {
		t.Fatalf("newServer: %v", err)
	}
	lm := srv.models["q35"]
	if lm == nil || lm.qwen3 == nil || !lm.visionCapable() {
		t.Fatal("auto-discovery did not attach the Qwen3.5 tower to the served model")
	}
	if lm.qwen3.enc != nil {
		t.Error("the tower loaded at startup; it must load on the first image")
	}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", srv.handleChat)
	ts := httptest.NewServer(mux)
	defer ts.Close()

	for _, k := range []string{"A", "B", "C"} {
		t.Run(k, func(t *testing.T) {
			type gold struct {
				Question string `json:"question"`
				InputIDs []int  `json:"input_ids"`
				HFTokens []int  `json:"hf_tokens"`
				HFText   string `json:"hf_text"`
			}
			read := func(name string) gold {
				var g gold
				raw, err := os.ReadFile(filepath.Join(gdir, name+"_"+k+".json"))
				if err != nil {
					t.Fatal(err)
				}
				if err := json.Unmarshal(raw, &g); err != nil {
					t.Fatal(err)
				}
				return g
			}
			g, def := read("serve"), read("golden")
			// The named delta: HF's default-template ids are serve's ids plus the empty think block
			// (<think> \n\n </think> \n\n = 4 tokens). If this stops holding, the template moved.
			if len(def.InputIDs) != len(g.InputIDs)+4 || !slices.Equal(def.InputIDs[:len(g.InputIDs)], g.InputIDs) {
				t.Fatalf("serve_ ids are not HF's default-template ids minus a 4-token think block (lens %d vs %d)", len(g.InputIDs), len(def.InputIDs))
			}
			png, err := os.ReadFile(filepath.Join(gdir, "img_"+k+".png"))
			if err != nil {
				t.Fatal(err)
			}

			// (a1) the prompt ids, built by serve's own code path.
			vi, err := lm.visionPrompt(lm.tmpl, "", []chat.Turn{{Role: "user", Content: g.Question}}, imageRef{mediaType: "image/png", data: png})
			if err != nil {
				t.Fatalf("visionPrompt: %v", err)
			}
			if !slices.Equal(vi.ids, g.InputIDs) {
				n := min(len(vi.ids), len(g.InputIDs))
				at := n
				for i := range n {
					if vi.ids[i] != g.InputIDs[i] {
						at = i
						break
					}
				}
				t.Fatalf("serve's prompt ids differ from HF's at index %d (lens %d vs %d): serve %v ... HF %v", at, len(vi.ids), len(g.InputIDs),
					vi.ids[max(0, at-2):min(len(vi.ids), at+3)], g.InputIDs[max(0, at-2):min(len(g.InputIDs), at+3)])
			}

			// (a2, b, c) through HTTP.
			body, _ := json.Marshal(map[string]any{
				"model": "q35", "temperature": 0, "max_tokens": len(g.HFTokens),
				"messages": []map[string]any{{"role": "user", "content": []map[string]any{
					{"type": "text", "text": g.Question},
					{"type": "image_url", "image_url": map[string]string{"url": "data:image/png;base64," + base64.StdEncoding.EncodeToString(png)}},
				}}},
			})
			post := func() (string, usage) {
				resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", bytes.NewReader(body))
				if err != nil {
					t.Fatal(err)
				}
				defer resp.Body.Close()
				var out struct {
					Choices []struct {
						Message      struct{ Content string } `json:"message"`
						FinishReason string                   `json:"finish_reason"`
					} `json:"choices"`
					Usage usage `json:"usage"`
				}
				if err := json.NewDecoder(resp.Body).Decode(&out); err != nil || resp.StatusCode != 200 || len(out.Choices) != 1 {
					t.Fatalf("status %d, err %v, choices %d", resp.StatusCode, err, len(out.Choices))
				}
				return out.Choices[0].Message.Content, out.Usage
			}
			text1, u1 := post()
			if text1 != g.HFText {
				t.Errorf("reply text differs from HF's decoded tokens:\n got  %q\n want %q", text1, g.HFText)
			}
			if u1.PromptTokens != len(g.InputIDs) || u1.CompletionTokens != len(g.HFTokens) {
				t.Errorf("usage prompt/completion = %d/%d, want %d/%d", u1.PromptTokens, u1.CompletionTokens, len(g.InputIDs), len(g.HFTokens))
			}
			if text2, _ := post(); text2 != text1 {
				t.Errorf("the same request twice returned different text:\n 1: %q\n 2: %q", text1, text2)
			}
		})
	}
	if lm.qwen3.enc == nil {
		t.Error("the tower never loaded: the image path did not run through it")
	}
}
