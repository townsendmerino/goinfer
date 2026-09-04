package serveapp

import (
	"encoding/json"
	"go/ast"
	"go/parser"
	"go/token"
	"net/http/httptest"
	"strings"
	"testing"
)

// A streaming client has no token count unless the server sends one, and counting SSE chunks is not
// a substitute: streamTokens emits a chunk only when `end > printed`, so a token held back for an
// incomplete UTF-8 rune or a partial stop-string match produces NO chunk, and the token that
// resolves the holdback produces one chunk carrying several tokens' bytes. Chunks <= tokens, always
// in the same direction. bench_peer.py counted chunks and called them tokens.
func TestStreamOptions_includeUsageShape(t *testing.T) {
	got := usageChunk("chatcmpl-x", 1234, "m", usage{PromptTokens: 7, CompletionTokens: 11, TotalTokens: 18})

	if got["object"] != "chat.completion.chunk" {
		t.Errorf("object = %v, want chat.completion.chunk", got["object"])
	}
	// OpenAI's shape: choices is present and EMPTY on the usage chunk, so a client iterating
	// choices sees nothing new and only a usage-aware client reads the counts.
	ch, ok := got["choices"].([]any)
	if !ok || len(ch) != 0 {
		t.Errorf("choices = %#v, want an empty array", got["choices"])
	}
	u, ok := got["usage"].(usage)
	if !ok || u.CompletionTokens != 11 || u.PromptTokens != 7 || u.TotalTokens != 18 {
		t.Errorf("usage = %#v, want 7/11/18", got["usage"])
	}
	// It must survive JSON round-tripping as the client will read it.
	b, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !strings.Contains(string(b), `"completion_tokens":11`) {
		t.Errorf("marshalled chunk lacks completion_tokens: %s", b)
	}
}

// The field must parse from the wire in OpenAI's nested shape, and stay absent (nil) when the
// client does not ask — a non-nil zero value would be indistinguishable from include_usage:false
// only by luck.
func TestStreamOptions_parsing(t *testing.T) {
	var withOpt chatReq
	if err := json.Unmarshal([]byte(`{"model":"m","stream":true,"stream_options":{"include_usage":true}}`), &withOpt); err != nil {
		t.Fatal(err)
	}
	if withOpt.StreamOptions == nil || !withOpt.StreamOptions.IncludeUsage {
		t.Errorf("include_usage did not parse: %#v", withOpt.StreamOptions)
	}
	var without chatReq
	if err := json.Unmarshal([]byte(`{"model":"m","stream":true}`), &without); err != nil {
		t.Fatal(err)
	}
	if without.StreamOptions != nil {
		t.Errorf("absent stream_options should stay nil, got %#v", without.StreamOptions)
	}
}

// M-26: include_usage was honoured on the plain chat stream ONLY. The tool and vision streams
// silently omitted the usage chunk, and /v1/completions did not even parse stream_options.
// Agent harnesses declare tools on every turn and rely on that chunk for context accounting,
// so the one surface that worked was the one they least often use.
//
// THE ANTI-DRIFT GATE, in the shape this audit's "N of M sites" findings keep calling for:
// count the sites rather than list them. Every streaming surface reaches sseDone at its normal
// completion; each such path must send the usage chunk first, and they all now go through the
// one sendUsage helper. Reading the AST means a fifth surface added later cannot slip past by
// being formatted differently.
func TestStreamSurfaces_allSendUsageBeforeDone(t *testing.T) {
	// The files that terminate an OpenAI-shaped SSE stream. anthropic_stream.go is excluded on
	// purpose: it is a different protocol with its own terminator and no stream_options.
	files := []string{"openai.go", "tools.go", "vision_serve.go"}
	fset := token.NewFileSet()
	var withUsage, total int
	for _, f := range files {
		af, err := parser.ParseFile(fset, f, nil, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", f, err)
		}
		ast.Inspect(af, func(n ast.Node) bool {
			blk, ok := n.(*ast.BlockStmt)
			if !ok {
				return true
			}
			for i, st := range blk.List {
				call, ok := callOf(st)
				if !ok || call != "sseDone" {
					continue
				}
				// An sseDone that follows sseErr is an ERROR exit — no usage is owed there,
				// and demanding one would be wrong. Only normal completions count.
				if i > 0 {
					if prev, ok := callOf(blk.List[i-1]); ok && prev == "sseErr" {
						continue
					}
				}
				total++
				for j := i - 1; j >= 0 && j >= i-3; j-- {
					if prev, ok := callOf(blk.List[j]); ok && prev == "sendUsage" {
						withUsage++
						break
					}
				}
			}
			return true
		})
	}
	// chat, /v1/completions, tools, vision — four normal completions across the three files.
	if total != 4 {
		t.Errorf("found %d normal stream completions, want 4 — a surface was added or removed "+
			"and this guard is now counting the wrong thing", total)
	}
	if withUsage != total {
		t.Errorf("%d of %d normal stream completions send the usage chunk; every one must, or "+
			"include_usage is honoured on some surfaces and silently dropped on others (M-26)",
			withUsage, total)
	}
}

// callOf returns the name of the function called by a bare expression statement.
func callOf(st ast.Stmt) (string, bool) {
	es, ok := st.(*ast.ExprStmt)
	if !ok {
		return "", false
	}
	call, ok := es.X.(*ast.CallExpr)
	if !ok {
		return "", false
	}
	id, ok := call.Fun.(*ast.Ident)
	if !ok {
		return "", false
	}
	return id.Name, true
}

// And the behaviour, not only the shape: sendUsage must emit exactly when asked to.
func TestSendUsage_onlyWhenRequested(t *testing.T) {
	for name, tc := range map[string]struct {
		so   *streamOptions
		want bool
	}{
		"absent":            {nil, false},
		"present but false": {&streamOptions{IncludeUsage: false}, false},
		"requested":         {&streamOptions{IncludeUsage: true}, true},
	} {
		t.Run(name, func(t *testing.T) {
			rec := httptest.NewRecorder()
			ss, ok := sseStart(rec)
			if !ok {
				t.Fatal("sseStart")
			}
			sendUsage(ss, tc.so, "id", 1, "m", usage{1, 2, 3})
			got := strings.Contains(rec.Body.String(), `"usage"`)
			if got != tc.want {
				t.Errorf("emitted=%v want=%v; body=%q", got, tc.want, rec.Body.String())
			}
		})
	}
}

// M-25, AT THE CALL SITE. The tokenizer tests prove DecodePiece/DecodeContinuation keep the
// leading space; they cannot prove the streaming loop calls one of them. Measured: reverting
// streamTokens to Decode broke NO test — the component was correct and vouched for behaviour
// the system did not produce, which is the trap CLAUDE.md names. So the assertion belongs here.
//
// streamTokens is shared by chat, /v1/completions and vision, so all three carried the defect
// the audit scoped to /v1/completions. The generated ids must never go through Decode: those
// ids continue the prompt, and Decode applies SentencePiece's sequence-level dummy-prefix strip
// to them — eating the response's leading space on Llama-2/Mistral.
//
// R-08 (audit-2026-09-02 / task-recompute-audit.md) replaced the per-token DecodeContinuation(ids)
// re-decode of the WHOLE generated sequence (O(n^2) in output length) with DecodePiece(id)
// appended incrementally. DecodePiece is the same non-stripping contract DecodeContinuation was
// relied on for here — its own doc comment states it explicitly ("does NOT apply the
// whole-sequence dummy-prefix strip... a caller printing piece-by-piece emitted
// 'Theanswerisfour'" if it did) — so this guard now requires DecodePiece instead of
// DecodeContinuation, and forbids Decode exactly as before.
func TestStreamTokens_decodesAsAContinuation(t *testing.T) {
	fset := token.NewFileSet()
	af, err := parser.ParseFile(fset, "openai.go", nil, 0)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	var fn *ast.FuncDecl
	ast.Inspect(af, func(n ast.Node) bool {
		if d, ok := n.(*ast.FuncDecl); ok && d.Name.Name == "streamTokens" {
			fn = d
		}
		return true
	})
	if fn == nil {
		t.Fatal("streamTokens not found — this guard is watching nothing")
	}
	var piece, whole int
	ast.Inspect(fn, func(n ast.Node) bool {
		sel, ok := n.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		switch sel.Sel.Name {
		case "DecodePiece":
			piece++
		case "Decode":
			whole++
		}
		return true
	})
	if whole != 0 {
		t.Errorf("streamTokens calls Decode %d time(s): the sequence-level dummy-prefix strip "+
			"applies to the generated ids and eats the response's leading space on Llama-2 / "+
			"Mistral (M-25)", whole)
	}
	if piece != 1 {
		t.Errorf("streamTokens has %d DecodePiece call(s), want 1 — the decode site moved and "+
			"this guard is now counting the wrong thing", piece)
	}
}
