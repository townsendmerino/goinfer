package serveapp

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestScriptedServe_forHarness serves the scripted model of empty_answer_test.go on a real port, for driving a real coding
// agent against a reply it cannot get from a real model on demand (an empty answer, a call inside the reasoning). It is a
// measurement tool, not a check: it skips unless SCRIPTED_SERVE_ADDR is set.
//
//	SCRIPTED_SERVE_ADDR=127.0.0.1:18291 SCRIPTED_REPLY='<think>\nAdd returns 5.\n' SCRIPTED_SECONDS=300 \
//	  go test -count=1 -run '^TestScriptedServe_forHarness$' -v ./internal/serveapp/
//
// SCRIPTED_REPLY is a Go-quoted string's body (\n for a newline). Every request is logged to stderr as it finishes: route,
// status, and the empty-answer and from-reasoning extension fields a non-streamed or streamed reply carried.
func TestScriptedServe_forHarness(t *testing.T) {
	addr := os.Getenv("SCRIPTED_SERVE_ADDR")
	if addr == "" {
		t.Skip("set SCRIPTED_SERVE_ADDR, SCRIPTED_REPLY and SCRIPTED_SECONDS")
	}
	reply, err := strconv.Unquote(`"` + os.Getenv("SCRIPTED_REPLY") + `"`)
	if err != nil {
		t.Fatalf("SCRIPTED_REPLY: %v", err)
	}
	secs, _ := strconv.Atoi(os.Getenv("SCRIPTED_SECONDS"))
	if secs <= 0 {
		secs = 300
	}
	tk := byteLevelScriptTokenizer(t)
	ids, err := tk.Encode(reply, false)
	if err != nil {
		t.Fatal(err)
	}
	s := &server{}
	lm := &loadedModel{tk: tk, name: os.Getenv("SCRIPTED_MODEL"), scripted: append([]int{}, ids...)}
	if lm.name == "" {
		lm.name = "scripted"
	}
	lm.tmpl = s.cfg.tuneTemplate(mellum21Template(t))
	s.models = map[string]*loadedModel{lm.name: lm}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/models", s.handleModels)
	mux.HandleFunc("POST /v1/responses", s.handleResponses)
	mux.HandleFunc("POST /v1/chat/completions", s.handleChat)
	mux.HandleFunc("POST /v1/messages", s.handleMessages)
	n := 0
	logged := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rec := &recordingWriter{ResponseWriter: w, status: 200}
		mux.ServeHTTP(rec, r)
		n++
		fmt.Fprintf(os.Stderr, "%s scripted: #%d %s %s -> %d %s\n", time.Now().Format("15:04:05"), n, r.Method, r.URL.Path, rec.status, rec.extensions())
	})
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		t.Fatal(err)
	}
	srv := &http.Server{Handler: logged}
	go srv.Serve(ln)
	fmt.Fprintf(os.Stderr, "scripted: serving %q on %s for %d s, reply %q (%d tokens)\n", lm.name, addr, secs, reply, len(ids))
	time.Sleep(time.Duration(secs) * time.Second)
	srv.Close()
	fmt.Fprintf(os.Stderr, "scripted: %d requests\n", n)
}

// recordingWriter keeps the status and notes the extension fields a reply carried.
type recordingWriter struct {
	http.ResponseWriter
	status int
	body   []byte
}

func (r *recordingWriter) WriteHeader(c int) { r.status = c; r.ResponseWriter.WriteHeader(c) }
func (r *recordingWriter) Write(b []byte) (int, error) {
	if len(r.body) < 1<<20 {
		r.body = append(r.body, b...)
	}
	return r.ResponseWriter.Write(b)
}
func (r *recordingWriter) Flush() {
	if f, ok := r.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func (r *recordingWriter) extensions() string {
	var out string
	for _, key := range []string{`"incomplete_details":`, `"goinfer_empty_answer":`, `"goinfer_tool_calls_from_reasoning":`} {
		if i := bytes.Index(r.body, []byte(key)); i >= 0 {
			end := min(len(r.body), i+len(key)+600)
			var v any
			dec := json.NewDecoder(bytes.NewReader(r.body[i+len(key) : end]))
			if dec.Decode(&v) == nil {
				b, _ := json.Marshal(v)
				out += key + string(b) + " "
			}
		}
	}
	return out
}

// byteLevelScriptTokenizer is scriptTokenizer over all 256 byte-level symbols (GPT-2's byte-to-unicode map), so any prompt a
// real client sends encodes.
func byteLevelScriptTokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	vocab := map[string]int{}
	add := func(tok string) {
		if _, ok := vocab[tok]; !ok {
			vocab[tok] = len(vocab)
		}
	}
	extra := 0
	for b := 0; b < 256; b++ {
		if (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF) {
			add(string(rune(b)))
		} else {
			add(string(rune(256 + extra)))
			extra++
		}
	}
	add("ĊĊ")
	var added []map[string]any
	for _, sp := range []string{"<|im_start|>", "<|im_end|>", "<think>", "</think>"} {
		add(sp)
		added = append(added, map[string]any{"id": vocab[sp], "content": sp, "special": true})
	}
	raw, _ := json.Marshal(map[string]any{
		"model":         map[string]any{"type": "BPE", "vocab": vocab, "merges": []string{"Ċ Ċ"}},
		"added_tokens":  added,
		"pre_tokenizer": map[string]any{"type": "ByteLevel", "add_prefix_space": false},
		"decoder":       map[string]any{"type": "ByteLevel"},
	})
	tk, err := tokenizer.LoadJSONBytes(raw)
	if err != nil {
		t.Fatal(err)
	}
	return tk
}
