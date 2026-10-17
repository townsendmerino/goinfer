package sdktest

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	aoption "github.com/anthropics/anthropic-sdk-go/option"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"

	"github.com/townsendmerino/goinfer/server"
)

// tinyTemplate is a plain ChatML template, which goinfer reads as ChatML and can render tools for.
const tinyTemplate = `{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}`

// tinyTokenizer is a byte-level BPE tokenizer.json for the tiny fixture's 256-id vocabulary: printable ASCII, the
// byte-level space and newline, ChatML's three special tokens and one merge. About one token a character.
func tinyTokenizer(t *testing.T) []byte {
	t.Helper()
	vocab := map[string]int{"Ġ": 0, "Ċ": 1}
	for c := '!'; c <= '~'; c++ {
		vocab[string(c)] = len(vocab)
	}
	var added []map[string]any
	for _, s := range []string{"<|endoftext|>", "<|im_start|>", "<|im_end|>"} {
		id := len(vocab)
		vocab[s] = id
		added = append(added, map[string]any{"id": id, "content": s, "special": true})
	}
	vocab["hi"] = len(vocab) // one merge, so the tokenizer loads as an encoder
	b, err := json.Marshal(map[string]any{
		"model":         map[string]any{"type": "BPE", "vocab": vocab, "merges": []string{"h i"}},
		"pre_tokenizer": map[string]any{"type": "ByteLevel", "add_prefix_space": false},
		"decoder":       map[string]any{"type": "ByteLevel"},
		"added_tokens":  added,
	})
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// tinyWindow is the context the tiny model is given here. Its committed config says 128, which a tool definition's
// prompt does not fit; the model has no learned positions, so the window is the config's to state.
const tinyWindow = 2048

// tinyClient serves the committed tiny model in process and returns an HTTP client bound to it. The model's replies are
// noise; the tests check the protocol, and grammar-constrained output where the content matters.
func tinyClient(t *testing.T) *http.Client {
	t.Helper()
	src := filepath.Join("..", "testdata", "llama-tiny")
	dir := t.TempDir()
	for _, name := range []string{"generation_config.json", "model.safetensors"} {
		b, err := os.ReadFile(filepath.Join(src, name))
		if err != nil {
			t.Fatalf("committed fixture %s: %v", name, err)
		}
		if err := os.WriteFile(filepath.Join(dir, name), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	raw, err := os.ReadFile(filepath.Join(src, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	var cfg map[string]any
	if err := json.Unmarshal(raw, &cfg); err != nil {
		t.Fatal(err)
	}
	cfg["max_position_embeddings"] = tinyWindow
	raw, _ = json.Marshal(cfg)
	for name, body := range map[string][]byte{"config.json": raw, "tokenizer.json": tinyTokenizer(t), "chat_template.jinja": []byte(tinyTemplate)} {
		if err := os.WriteFile(filepath.Join(dir, name), body, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	h, closer, err := server.NewHandler("-model", "tiny="+dir, "-backend", "cpu", "-quant", "int8int8")
	if err != nil {
		t.Fatalf("server.NewHandler: %v", err)
	}
	t.Cleanup(func() { closer.Close() })
	return &http.Client{Transport: server.Transport(h)}
}

// constrainedRoom is the output cap for the constrained replies. The grammar allows whitespace between JSON tokens
// and this model's choices are noise, so a reply can spend many tokens before it closes; the requests are greedy, so
// what it spends is the same on every run.
const constrainedRoom = 1024

// okSchema is a schema whose every value is bounded, so a constrained reply closes whatever the model would say.
var okSchema = map[string]any{
	"type":                 "object",
	"properties":           map[string]any{"ok": map[string]any{"type": "boolean"}},
	"required":             []string{"ok"},
	"additionalProperties": false,
}

func wantOK(t *testing.T, what, raw string) {
	t.Helper()
	var v struct {
		OK *bool `json:"ok"`
	}
	if err := json.Unmarshal([]byte(raw), &v); err != nil || v.OK == nil {
		t.Errorf("%s: %q is not the schema's object with a boolean ok (%v)", what, raw, err)
	}
}

func openaiClient(t *testing.T) openai.Client {
	return openai.NewClient(option.WithBaseURL("http://goinfer/v1/"), option.WithAPIKey("unused"),
		option.WithHTTPClient(tinyClient(t)), option.WithMaxRetries(0))
}

// The official OpenAI Go SDK against the in-process server: a chat completion, a stream, a forced tool call, a
// schema-constrained reply, the Responses API, and a context overflow read back as the SDK's typed error.
func TestOpenAISDK(t *testing.T) {
	client := openaiClient(t)
	ctx := context.Background()
	hi := []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hi")}

	t.Run("chat", func(t *testing.T) {
		resp, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{Model: "tiny", Messages: hi, MaxTokens: openai.Int(4), Temperature: openai.Float(0)})
		if err != nil {
			t.Fatal(err)
		}
		if len(resp.Choices) != 1 || resp.Choices[0].Message.Role != "assistant" || resp.Usage.PromptTokens == 0 || resp.Usage.CompletionTokens == 0 {
			t.Errorf("chat completion: %s", resp.RawJSON())
		}
	})
	t.Run("stream", func(t *testing.T) {
		stream := client.Chat.Completions.NewStreaming(ctx, openai.ChatCompletionNewParams{Model: "tiny", Messages: hi, MaxTokens: openai.Int(6), Temperature: openai.Float(0)})
		chunks, finished := 0, false
		for stream.Next() {
			chunks++
			for _, c := range stream.Current().Choices {
				finished = finished || c.FinishReason != ""
			}
		}
		if err := stream.Err(); err != nil || chunks < 2 || !finished {
			t.Errorf("stream: %d chunks, finish reason seen %v, err %v", chunks, finished, err)
		}
	})
	t.Run("forced tool call", func(t *testing.T) {
		resp, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
			Model: "tiny", Messages: hi, MaxTokens: openai.Int(constrainedRoom), Temperature: openai.Float(0),
			Tools: []openai.ChatCompletionToolUnionParam{openai.ChatCompletionFunctionTool(shared.FunctionDefinitionParam{
				Name: "pick", Description: openai.String("pick one"), Parameters: shared.FunctionParameters(okSchema)})},
			ToolChoice: openai.ToolChoiceOptionFunctionToolChoice(openai.ChatCompletionNamedToolChoiceFunctionParam{Name: "pick"}),
		})
		if err != nil {
			t.Fatal(err)
		}
		calls := resp.Choices[0].Message.ToolCalls
		if len(calls) != 1 || calls[0].Function.Name != "pick" || calls[0].ID == "" {
			t.Fatalf("forced tool call: %s", resp.RawJSON())
		}
		wantOK(t, "the tool call's arguments", calls[0].Function.Arguments)
	})
	t.Run("structured output", func(t *testing.T) {
		resp, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
			Model: "tiny", Messages: hi, MaxTokens: openai.Int(constrainedRoom), Temperature: openai.Float(0),
			ResponseFormat: openai.ChatCompletionNewParamsResponseFormatUnion{OfJSONSchema: &shared.ResponseFormatJSONSchemaParam{
				JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{Name: "answer", Schema: okSchema, Strict: openai.Bool(true)}}},
		})
		if err != nil {
			t.Fatal(err)
		}
		wantOK(t, "the reply", resp.Choices[0].Message.Content)
	})
	t.Run("responses", func(t *testing.T) {
		resp, err := client.Responses.New(ctx, responses.ResponseNewParams{Model: "tiny", MaxOutputTokens: openai.Int(4), Temperature: openai.Float(0),
			Input: responses.ResponseNewParamsInputUnion{OfString: openai.String("hi")}})
		if err != nil {
			t.Fatal(err)
		}
		if resp.Status != "completed" && resp.Status != "incomplete" || len(resp.Output) == 0 {
			t.Errorf("responses: %s", resp.RawJSON())
		}
	})
	t.Run("overflow is the SDK's typed error", func(t *testing.T) {
		long := []openai.ChatCompletionMessageParamUnion{openai.UserMessage(strings.Repeat("word ", tinyWindow/4))}
		_, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{Model: "tiny", Messages: long, MaxTokens: openai.Int(4)})
		var apiErr *openai.Error
		if !errors.As(err, &apiErr) || apiErr.StatusCode != 400 || apiErr.Code != "context_length_exceeded" {
			t.Errorf("a conversation past the window: %v; want a 400 with code context_length_exceeded", err)
		}
	})
}

// The official Anthropic Go SDK reads a reused prompt prefix as Usage.CacheReadInputTokens, on a reply and on a stream it
// accumulates, and InputTokens is the rest of the prompt: the two sum to count_tokens.
func TestAnthropicSDK_cacheRead(t *testing.T) {
	client := anthropic.NewClient(aoption.WithBaseURL("http://goinfer"), aoption.WithAPIKey("unused"),
		aoption.WithHTTPClient(tinyClient(t)), aoption.WithMaxRetries(0))
	ctx := context.Background()
	turns := []anthropic.MessageParam{anthropic.NewUserMessage(anthropic.NewTextBlock(strings.Repeat("a shared prefix ", 16)))}
	next := func(prev []anthropic.MessageParam, reply string) []anthropic.MessageParam {
		return append(append([]anthropic.MessageParam{}, prev...),
			anthropic.NewAssistantMessage(anthropic.NewTextBlock(reply)), anthropic.NewUserMessage(anthropic.NewTextBlock("go on")))
	}
	check := func(what string, msgs []anthropic.MessageParam, u anthropic.Usage) {
		t.Helper()
		ct, err := client.Messages.CountTokens(ctx, anthropic.MessageCountTokensParams{Model: "tiny", Messages: msgs})
		if err != nil {
			t.Fatal(err)
		}
		if u.CacheReadInputTokens <= 0 || u.InputTokens+u.CacheReadInputTokens != ct.InputTokens {
			t.Errorf("%s: InputTokens %d, CacheReadInputTokens %d; want a reused prefix, the two summing to count_tokens' %d",
				what, u.InputTokens, u.CacheReadInputTokens, ct.InputTokens)
		}
	}
	if _, err := client.Messages.New(ctx, anthropic.MessageNewParams{Model: "tiny", MaxTokens: 4, Temperature: anthropic.Float(0), Messages: turns}); err != nil {
		t.Fatal(err)
	}
	turns = next(turns, "ok")
	resp, err := client.Messages.New(ctx, anthropic.MessageNewParams{Model: "tiny", MaxTokens: 4, Temperature: anthropic.Float(0), Messages: turns})
	if err != nil {
		t.Fatal(err)
	}
	check("message", turns, resp.Usage)

	turns = next(turns, "more")
	stream := client.Messages.NewStreaming(ctx, anthropic.MessageNewParams{Model: "tiny", MaxTokens: 4, Temperature: anthropic.Float(0), Messages: turns})
	var msg anthropic.Message
	for stream.Next() {
		if err := msg.Accumulate(stream.Current()); err != nil {
			t.Fatalf("accumulate: %v", err)
		}
	}
	if err := stream.Err(); err != nil {
		t.Fatal(err)
	}
	check("stream", turns, msg.Usage)
}

// The official Anthropic Go SDK against the in-process server: a message, a stream, a forced tool use, and a context
// overflow in the words Claude Code matches.
func TestAnthropicSDK(t *testing.T) {
	client := anthropic.NewClient(aoption.WithBaseURL("http://goinfer"), aoption.WithAPIKey("unused"),
		aoption.WithHTTPClient(tinyClient(t)), aoption.WithMaxRetries(0))
	ctx := context.Background()
	hi := []anthropic.MessageParam{anthropic.NewUserMessage(anthropic.NewTextBlock("hi"))}

	t.Run("message", func(t *testing.T) {
		resp, err := client.Messages.New(ctx, anthropic.MessageNewParams{Model: "tiny", MaxTokens: 4, Temperature: anthropic.Float(0), Messages: hi})
		if err != nil {
			t.Fatal(err)
		}
		if resp.Role != "assistant" || resp.StopReason == "" || resp.Usage.InputTokens == 0 || resp.Usage.OutputTokens == 0 {
			t.Errorf("message: %s", resp.RawJSON())
		}
	})
	t.Run("stream", func(t *testing.T) {
		stream := client.Messages.NewStreaming(ctx, anthropic.MessageNewParams{Model: "tiny", MaxTokens: 6, Temperature: anthropic.Float(0), Messages: hi})
		var msg anthropic.Message
		events := 0
		for stream.Next() {
			events++
			if err := msg.Accumulate(stream.Current()); err != nil {
				t.Fatalf("accumulate: %v", err)
			}
		}
		if err := stream.Err(); err != nil || events < 3 || msg.StopReason == "" {
			t.Errorf("stream: %d events, stop reason %q, err %v", events, msg.StopReason, err)
		}
	})
	t.Run("forced tool use", func(t *testing.T) {
		resp, err := client.Messages.New(ctx, anthropic.MessageNewParams{
			Model: "tiny", MaxTokens: constrainedRoom, Temperature: anthropic.Float(0), Messages: hi,
			Tools: []anthropic.ToolUnionParam{anthropic.ToolUnionParamOfTool(anthropic.ToolInputSchemaParam{
				Properties: okSchema["properties"], Required: []string{"ok"}}, "pick")},
			ToolChoice: anthropic.ToolChoiceParamOfTool("pick"),
		})
		if err != nil {
			t.Fatal(err)
		}
		found := false
		for _, block := range resp.Content {
			if block.Type == "tool_use" && block.Name == "pick" && block.ID != "" {
				found = true
				wantOK(t, "the tool use's input", string(block.Input))
			}
		}
		if !found || resp.StopReason != "tool_use" {
			t.Errorf("forced tool use: %s", resp.RawJSON())
		}
	})
	t.Run("overflow in Claude Code's words", func(t *testing.T) {
		long := []anthropic.MessageParam{anthropic.NewUserMessage(anthropic.NewTextBlock(strings.Repeat("word ", tinyWindow/4)))}
		_, err := client.Messages.New(ctx, anthropic.MessageNewParams{Model: "tiny", MaxTokens: 4, Messages: long})
		var apiErr *anthropic.Error
		if !errors.As(err, &apiErr) || apiErr.StatusCode != 400 || !strings.Contains(apiErr.RawJSON(), "prompt is too long: ") {
			t.Errorf("a conversation past the window: %v; want a 400 saying \"prompt is too long: N tokens > M maximum\"", err)
		}
	})
}
