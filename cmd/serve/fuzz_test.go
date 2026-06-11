package main

import (
	"encoding/json"
	"testing"
)

// Track 2.6 (testing campaign): the request JSON for /v1/chat/completions and
// /v1/responses is attacker-supplied. These targets fuzz the model-free request
// shaping — decode + sampling/stop/content/message translation + the
// response_format → grammar dispatch — which must never panic. (The grammar
// COMPILE itself is fuzzed harder in constrain; the end-to-end handler path with
// a loaded model is the maintainer-only soak in Track 5.)

var chatSeeds = []string{
	`{"model":"m","messages":[{"role":"user","content":"hi"}],"temperature":0.5,"stop":["x","y"]}`,
	`{"model":"m","messages":[{"role":"system","content":"s"},{"role":"assistant","tool_calls":[{"id":"1","function":{"name":"f","arguments":"{}"}}]},{"role":"tool","name":"f","tool_call_id":"1","content":"r"}]}`,
	`{"messages":[],"stop":"END","top_k":-1,"max_tokens":-5,"temperature":1e308,"top_p":-2,"seed":-9223372036854775808}`,
	`{"response_format":{"type":"json_schema","json_schema":{"name":"n","schema":{"type":"object","additionalProperties":false,"properties":{"a":{"type":"string"}},"required":["a"]}}}}`,
	`{"response_format":{"type":"json_object"}}`,
	`{"stop":[1,2,3],"messages":[{"role":"user","content":[{"type":"text","text":"x"}]}]}`,
}

// FuzzServeChatRequest decodes arbitrary bytes as a chat request and runs the
// model-free shapers: response_format→grammar, the sampling/stop translation
// (prepare with a zero model, no masker), and message→turn mapping. Bar: never a
// panic, whatever the body.
func FuzzServeChatRequest(f *testing.F) {
	for _, s := range chatSeeds {
		f.Add([]byte(s))
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		var req chatReq
		if json.Unmarshal(data, &req) != nil {
			return // a decode error is the handler's StatusBadRequest path — fine
		}
		// response_format → constraint grammar (the attacker schema dispatch).
		_, _ = grammarFor(req.ResponseFormat)
		// Sampling/stop translation needs no model when there's no response_format
		// (the masker path legitimately requires a loaded tokenizer).
		sm := req.sampling
		sm.ResponseFormat = nil
		lm := &loadedModel{}
		_, _ = lm.prepare(sm, []int{1, 2, 3})
		_, _ = messagesToTurns(req.Messages)
		_ = parseStop(req.sampling.Stop)
	})
}

var messagesSeeds = []string{
	`{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}`,
	`{"max_tokens":1,"system":[{"type":"text","text":"s"}],"messages":[{"role":"user","content":[{"type":"text","text":"x","cache_control":{"type":"ephemeral"}},{"type":"image","source":{}}]}]}`,
	`{"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"f","input":{"a":1}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":[{"type":"text","text":"r"}]}]}]}`,
	`{"messages":[{"role":"user","content":[1,2,3]}],"tool_choice":{"type":"tool","name":"f"},"tools":[{"name":"f","input_schema":{}}]}`,
	`{"stop_sequences":["END"],"temperature":2,"top_k":-1,"top_p":-9,"max_tokens":-3,"messages":[{"role":"x","content":null}]}`,
}

// FuzzServeMessagesRequest decodes arbitrary bytes as an Anthropic /v1/messages
// request and runs the model-free shapers: system+message→turn translation
// (string-or-block content, tool_use/tool_result replay, image rejection,
// adjacent-role merge), tool_choice parsing, and the sampling translation. Bar:
// never a panic, whatever the body.
func FuzzServeMessagesRequest(f *testing.F) {
	for _, s := range messagesSeeds {
		f.Add([]byte(s))
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		var req anthropicReq
		if json.Unmarshal(data, &req) != nil {
			return // a decode error is the handler's invalid_request_error path — fine
		}
		_, _, _ = anthropicTurns(&req)
		_ = anthropicText(req.System)
		mode, name := anthropicToolMode(req.ToolChoice)
		_ = anthropicForcedTool(mode, name, anthropicTools(req.Tools))
		lm := &loadedModel{}
		_, _ = lm.prepare(req.toSampling(), []int{1, 2, 3})
	})
}

var imagePartSeeds = []string{
	`"plain string content"`,
	`[{"type":"text","text":"hi"},{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgo="}}]`,
	`[{"type":"image_url","image_url":{"url":"http://x/y.png"}}]`,
	`[{"type":"image_url","image_url":{"url":"data:;base64,!!!notbase64"}}]`,
	`[{"type":"image_url","image_url":{"url":"data:image/png,nope"}}]`,
	`[{"type":"image_url"}]`,
	`[1,2,3]`,
	`{}`,
}

// FuzzServeImageParts decodes arbitrary bytes as OpenAI message content and runs
// the model-free image-extraction shapers — contentPartsText, contentPartsImages
// (data-URI decode + the SSRF URL reject), decodeDataURI — plus an Anthropic
// image-block decode. Bar: never a panic, whatever the body.
func FuzzServeImageParts(f *testing.F) {
	for _, s := range imagePartSeeds {
		f.Add([]byte(s))
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		raw := json.RawMessage(data)
		_ = contentPartsText(raw)
		_, _ = contentPartsImages(raw)
		_, _ = decodeDataURI(string(data))
		// Anthropic image block (the source is the same base64 path).
		req := &anthropicReq{Messages: []anthropicMessage{{Role: "user", Content: raw}}}
		_, _ = anthropicImages(req)
	})
}

// FuzzServeResponsesInput fuzzes the /v1/responses `input` parser (a string, or
// an array of message items whose content is a string or an array of parts) and
// the content flattener — both pure, both attacker-supplied.
func FuzzServeResponsesInput(f *testing.F) {
	f.Add([]byte(`"just a string"`))
	f.Add([]byte(`[{"role":"user","content":[{"type":"input_text","text":"hi"}]}]`))
	f.Add([]byte(`[{"role":"x","content":"y"},{"content":[{"type":"t","text":"z"}]}]`))
	f.Add([]byte(`{}`))
	f.Fuzz(func(t *testing.T, data []byte) {
		if msgs, err := responseInputToMessages(json.RawMessage(data)); err == nil {
			for _, m := range msgs {
				_ = m.Content
			}
		}
		_ = contentText(json.RawMessage(data))
	})
}
