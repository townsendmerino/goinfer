package serveapp

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/internal/modelload"
)

// TestKV0_tokenizeBodies turns recorded request bodies into the token ids the server would prefill, for KV0's traffic study
// (docs/measurements/kv0-traffic-2026-10-13.md). It is a measurement tool, not a check: it skips unless KV0_BODIES is set.
//
//	KV0_BODIES=<dir of recorded bodies> KV0_GGUF=<the served model: a .gguf, a .giw or a model directory> KV0_OUT=<ids.jsonl> \
//	  go test -count=1 -run '^TestKV0_tokenizeBodies$' -v ./internal/serveapp/
//
// Each body is rendered by the same calls its route makes (serveChatToolsWith / serveChatText, serveResponsesWith and
// respondTools, serveMessagesWith) with the model's own tokenizer and template and the server's default settings. Only a
// route's rendering is mirrored: a body the route would refuse before rendering is reported as an error line, and the study
// compares each count with the usage the server itself reported, so a rendering that drifted from the route shows there.
func TestKV0_tokenizeBodies(t *testing.T) {
	dir, gguf, out := os.Getenv("KV0_BODIES"), os.Getenv("KV0_GGUF"), os.Getenv("KV0_OUT")
	if dir == "" {
		t.Skip("set KV0_BODIES, KV0_GGUF and KV0_OUT")
	}
	tk, err := modelload.Tokenizer(gguf)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	tmpl, err := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has})
	if err != nil {
		t.Fatalf("chat template: %v", err)
	}
	s := &server{}
	lm := &loadedModel{tk: tk, name: "kv0"}
	lm.tmpl = s.cfg.tuneTemplate(tmpl)

	var files []string
	_ = filepath.WalkDir(dir, func(p string, d os.DirEntry, err error) error {
		if err == nil && !d.IsDir() && strings.HasSuffix(p, ".json") {
			files = append(files, p)
		}
		return nil
	})
	sort.Strings(files)
	of, err := os.Create(out)
	if err != nil {
		t.Fatal(err)
	}
	defer of.Close()
	w := bufio.NewWriter(of)
	defer w.Flush()
	enc := json.NewEncoder(w)
	ok := 0
	for _, f := range files {
		raw, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		var rec struct {
			Path string `json:"path"`
			Body string `json:"body"`
		}
		if err := json.Unmarshal(raw, &rec); err != nil {
			t.Fatalf("%s: %v", f, err)
		}
		route := strings.Split(rec.Path, "?")[0]
		ids, rerr := kv0Render(s, lm, route, []byte(rec.Body))
		line := map[string]any{"file": f, "route": route}
		if rerr != nil {
			line["error"] = rerr.Error()
		} else {
			pieces := make([]string, len(ids))
			for i, id := range ids {
				pieces[i] = lm.tokenText(id)
			}
			line["n"], line["ids"], line["pieces"] = len(ids), ids, pieces
			ok++
		}
		if err := enc.Encode(line); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("%d of %d bodies rendered", ok, len(files))
}

func kv0Render(s *server, lm *loadedModel, route string, body []byte) ([]int, error) {
	ids, _, err := kv0RenderIDs(s, lm, route, body, true)
	return ids, err
}

// kv0RenderIDs is kv0Render with the carried-id outcome of a /v1/responses body (carried_ids.go); useIDs false renders it as
// a request whose blobs carry no ids (the reasoning text still read from them).
func kv0RenderIDs(s *server, lm *loadedModel, route string, body []byte, useIDs bool) ([]int, idsOutcome, error) {
	switch route {
	case "/v1/chat/completions":
		var req chatReq
		if err := json.Unmarshal(body, &req); err != nil {
			return nil, idsOutcome{}, err
		}
		req.Tools, _ = functionTools(req.Tools)
		ts, err := s.resolveThink(req.think())
		if err != nil {
			return nil, idsOutcome{}, err
		}
		tm := lm.templateFor(ts)
		system, turns := messagesToTurns(req.Messages)
		if len(req.Tools) > 0 && toolChoiceMode(req.ToolChoice) != "none" {
			tools := make([]chat.Tool, len(req.Tools))
			for i, t := range req.Tools {
				tools[i] = chat.Tool{Name: t.Function.Name, Description: t.Function.Description, Parameters: t.Function.Parameters}
			}
			if toolsConstrainedFromStart(forcedTool(req.ToolChoice, tools, endsWithToolResult(turns)), toolChoiceMode(req.ToolChoice) == "function", openAIUnionMode(req.ToolChoice), tools) {
				tm = lm.constrainedTemplate(ts)
			}
			ids, err := lm.tk.EncodeSegments(tm.RenderToolsSegments(system, turns, tools), false)
			return ids, idsOutcome{}, err
		}
		if req.sampling.constrainsOutput() {
			tm = lm.constrainedTemplate(ts)
		}
		ids, err := lm.promptForT(tm, system, turns)
		return ids, idsOutcome{}, err
	case "/v1/responses":
		var req responseReq
		if err := json.Unmarshal(body, &req); err != nil {
			return nil, idsOutcome{}, err
		}
		if req.PreviousResponseID != "" {
			return nil, idsOutcome{}, fmt.Errorf("previous_response_id: the study's renderer has no response store")
		}
		req.Tools, _ = functionTools(req.Tools)
		var messages []chatMessage
		if req.Instructions != "" {
			messages = append(messages, chatMessage{Role: "system", Content: rawStr(req.Instructions)})
		}
		in, err := responseInputToMessages(req.Input)
		if err != nil {
			return nil, idsOutcome{}, err
		}
		messages = append(messages, in...)
		ts, err := s.resolveThink(req.thinkRequest())
		if err != nil {
			return nil, idsOutcome{}, err
		}
		tm := lm.templateFor(ts)
		system, turns, carriers := messagesToTurnsCarried(messages)
		ck := s.idsCheckFor(lm, nil, carriers)
		if !useIDs {
			ck = nil
		}
		if len(req.Tools) > 0 && toolChoiceMode(req.ToolChoice) != "none" {
			tools := make([]chat.Tool, len(req.Tools))
			for i, t := range req.Tools {
				tools[i] = chat.Tool{Name: t.Function.Name, Description: t.Function.Description, Parameters: t.Function.Parameters}
			}
			if toolsConstrainedFromStart(forcedTool(req.ToolChoice, tools, endsWithToolResult(turns)), toolChoiceMode(req.ToolChoice) == "function", openAIUnionMode(req.ToolChoice), tools) {
				tm = lm.constrainedTemplate(ts)
			}
			return lm.encodePrompt(tm, system, turns, tools, ck)
		}
		return lm.encodePrompt(tm, system, turns, nil, ck)
	case "/v1/messages":
		var req anthropicReq
		if err := json.Unmarshal(body, &req); err != nil {
			return nil, idsOutcome{}, err
		}
		system, turns, carriers, aerr := anthropicTurnsCarried(&req)
		if aerr != nil {
			return nil, idsOutcome{}, fmt.Errorf("anthropicTurns: %v", aerr)
		}
		ck := s.idsCheckFor(lm, nil, carriers)
		if !useIDs {
			ck = nil
		}
		ts, err := s.resolveThink(req.thinkRequest())
		if err != nil {
			return nil, idsOutcome{}, err
		}
		tm := lm.templateFor(ts)
		var tools []chat.Tool
		if mode, name := anthropicToolMode(req.ToolChoice); len(req.Tools) > 0 && mode != "none" {
			tools = anthropicTools(req.Tools)
			if anthropicToolsConstrainedFromStart(mode, name, tools) {
				tm = lm.constrainedTemplate(ts)
			}
		}
		return lm.encodePrompt(tm, system, turns, tools, ck)
	}
	return nil, idsOutcome{}, fmt.Errorf("route %s is not a generation route", route)
}
