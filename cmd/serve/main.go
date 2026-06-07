// Command serve is an OpenAI-compatible HTTP server for a single goinfer model:
// pure stdlib net/http, no dependencies. It speaks /v1/chat/completions,
// /v1/completions, and /v1/models — enough for Open WebUI, LangChain, the OpenAI
// SDKs, and anything else that points at an OpenAI base URL — including
// streaming (SSE) and `response_format: json_schema` constrained decoding (the
// model physically cannot emit non-conforming JSON; see the constrain package).
//
//	go run ./cmd/serve --model ~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
//	# then point a client at http://localhost:8080/v1
//	curl localhost:8080/v1/chat/completions -d '{"model":"local",
//	  "messages":[{"role":"user","content":"hi"}]}'
package main

import (
	"flag"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

func main() {
	var (
		model   = flag.String("model", "", "path to a .gguf file or HF checkpoint dir (required)")
		addr    = flag.String("addr", ":8080", "listen address")
		backend = flag.String("backend", "cpu", "compute backend: cpu | webgpu")
		quant   = flag.String("quant", "int8int8", "weight quant: \"\" | int8 | int8int8 | int4")
		name    = flag.String("served-model-name", "", "model id reported by /v1/models (default: file/dir basename)")
	)
	flag.Parse()
	if *model == "" {
		fmt.Fprintln(os.Stderr, "error: --model is required")
		flag.Usage()
		os.Exit(2)
	}

	srv, err := newServer(*model, *backend, *quant, *name)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", srv.handleChat)
	mux.HandleFunc("POST /v1/completions", srv.handleCompletions)
	mux.HandleFunc("GET /v1/models", srv.handleModels)

	fmt.Fprintf(os.Stderr, "goinfer serving %q (OpenAI-compatible) on %s\n", srv.modelID, *addr)
	if err := http.ListenAndServe(*addr, mux); err != nil {
		fmt.Fprintf(os.Stderr, "server: %v\n", err)
		os.Exit(1)
	}
}

// newServer loads the tokenizer + model once and resolves the chat template.
func newServer(path, backend, quant, name string) (*server, error) {
	loadTok := tokenizer.Load
	if strings.HasSuffix(path, ".gguf") {
		loadTok = tokenizer.LoadGGUF
	}
	tk, err := loadTok(path)
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}
	t0 := time.Now()
	model, err := decoder.Load(path, decoder.Options{Backend: backend, Quant: quant})
	if err != nil {
		return nil, fmt.Errorf("load model: %w", err)
	}
	cfg := model.Config()
	if name == "" {
		name = strings.TrimSuffix(filepath.Base(path), ".gguf")
	}
	s := &server{
		tk: tk, model: model,
		vocab: cfg.VocabSize, eosIDs: cfg.EOSIDs(), modelID: name,
	}
	if tmpl, derr := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has}); derr == nil {
		s.tmpl = tmpl
		for _, str := range tmpl.Stops().Strings {
			if id, ok := tk.TokenID(str); ok {
				s.stopIDs = append(s.stopIDs, id)
			}
		}
	}
	fmt.Fprintf(os.Stderr, "loaded %d-layer model (vocab %d) in %s [chat: %s]\n",
		cfg.NumLayers, cfg.VocabSize, time.Since(t0).Round(time.Millisecond), templateName(s.tmpl))
	return s, nil
}

func templateName(t *chat.Template) string {
	if t == nil {
		return "raw (no template)"
	}
	return t.Name()
}
