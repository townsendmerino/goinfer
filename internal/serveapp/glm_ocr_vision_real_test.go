//go:build realckpt

package serveapp

import (
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
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

// TestServe_glmOcrImage_O3 is gate O3's serve half, on the real GLM-OCR checkpoint at f32 on the CPU backend, through the
// real serve stack: an OpenAI `image_url` data-URI part plus the task prompt -> handleChat -> preprocess -> lazily-loaded
// tower -> GenerateQwenVL, temperature 0. The three images are the procedurally rendered documents of
// scripts/gen_glm_ocr_doc_images.py (not real scans); the references are transformers 5.12.0 f32 greedy
// (scripts/pin_glm_ocr_e2e.py).
//
//	(a) the prompt ids serve builds (the chat.GlmOCR renderer + multimodal.GlmOcrImageBlock + the tokenizer) are EXACTLY the
//	    ids HF's processor builds from the checkpoint's own chat_template.jinja, for each image and task prompt (cheap: the
//	    tower has not run);
//	(b) the reply text is HF's 64 greedy tokens decoded, and usage.prompt_tokens / completion_tokens are the id counts;
//	(c) the tower loaded lazily, on the first image.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags realckpt ./internal/serveapp/ -run 'TestServe_glmOcrImage_O3/invoice' -v -timeout 20m 2>&1 | tee /tmp/glm-ocr-serve-invoice.log
func TestServe_glmOcrImage_O3(t *testing.T) {
	requireHeavyModel(t)
	ckpt := filepath.Join(os.Getenv("HOME"), "models", "glm-ocr")
	if _, err := os.Stat(filepath.Join(ckpt, "config.json")); err != nil {
		t.Skipf("asset missing: %s", ckpt)
	}
	srv, err := newServer(config{models: modelFlag{{name: "glm", path: ckpt}}, load: loadflags.Flags{Backend: "cpu"}, kvSessions: 2})
	if err != nil {
		t.Fatalf("newServer: %v", err)
	}
	lm := srv.models["glm"]
	if lm == nil || lm.glm == nil || !lm.visionCapable() {
		t.Fatal("auto-discovery did not attach the GLM-OCR tower to the served model")
	}
	if lm.tmpl == nil || lm.tmpl.Name() != "glm_ocr" {
		t.Fatalf("chat template %v, want glm_ocr", lm.tmpl)
	}
	if lm.glm.enc != nil {
		t.Error("the tower loaded at startup; it must load on the first image")
	}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", srv.handleChat)
	ts := httptest.NewServer(mux)
	defer ts.Close()

	for _, name := range []string{"invoice", "table", "formula"} {
		t.Run(name, func(t *testing.T) {
			type gold struct {
				ImageSHA string   `json:"image_sha256"`
				Prompt   string   `json:"prompt"`
				InputIDs []int    `json:"input_ids"`
				HFTokens []int    `json:"hf_tokens"`
				HFText   string   `json:"hf_text"`
				Stopped  *int     `json:"hf_stopped_at"`
				NNew     int      `json:"n_new"`
				Grid     [][3]int `json:"grid_thw"`
			}
			var g gold
			f, err := os.Open(filepath.Join("..", "..", "testdata", "glm_ocr", "golden_"+name+".json.gz"))
			if err != nil {
				t.Fatal(err)
			}
			gz, err := gzip.NewReader(f)
			if err != nil {
				t.Fatal(err)
			}
			if err := json.NewDecoder(gz).Decode(&g); err != nil {
				t.Fatal(err)
			}
			f.Close()
			png, err := os.ReadFile(filepath.Join("..", "..", "testdata", "glm_ocr", name+".png"))
			if err != nil {
				t.Fatal(err)
			}
			if sum := sha256.Sum256(png); hex.EncodeToString(sum[:]) != g.ImageSHA {
				t.Fatalf("%s.png does not match the golden's image sha256: regenerated after the reference was taken", name)
			}

			// (a) the prompt ids, built by serve's own code path.
			towerBefore := lm.glm.enc // nil on the first image; a later subtest starts with the tower already loaded
			vi, err := lm.visionPrompt(lm.tmpl, "", []chat.Turn{{Role: "user", Content: g.Prompt}}, imageRef{mediaType: "image/png", data: png})
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
			if vi.grid != g.Grid[0] {
				t.Fatalf("serve's grid %v, HF's processor %v", vi.grid, g.Grid[0])
			}
			t.Logf("prompt ids identical to HF's: %d ids (%d image tokens), grid %v, prompt %q", len(vi.ids), vi.imgLen, vi.grid, g.Prompt)
			if lm.glm.enc != towerBefore {
				t.Error("visionPrompt ran the tower: features must stay lazy")
			}

			// (b) through HTTP.
			body, _ := json.Marshal(map[string]any{
				"model": "glm", "temperature": 0, "max_tokens": g.NNew,
				"messages": []map[string]any{{"role": "user", "content": []map[string]any{
					{"type": "image_url", "image_url": map[string]string{"url": "data:image/png;base64," + base64.StdEncoding.EncodeToString(png)}},
					{"type": "text", "text": g.Prompt},
				}}},
			})
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
			if out.Choices[0].Message.Content != g.HFText {
				t.Errorf("reply text differs from HF's decoded tokens:\n got  %q\n want %q", out.Choices[0].Message.Content, g.HFText)
			}
			if out.Usage.PromptTokens != len(g.InputIDs) || out.Usage.CompletionTokens != len(g.HFTokens) {
				t.Errorf("usage prompt/completion = %d/%d, want %d/%d", out.Usage.PromptTokens, out.Usage.CompletionTokens, len(g.InputIDs), len(g.HFTokens))
			}
			if lm.glm.enc == nil {
				t.Error("the tower never loaded: the image path did not run through it")
			}
		})
	}
}
