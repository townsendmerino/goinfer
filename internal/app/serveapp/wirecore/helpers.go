package wirecore

import (
	"encoding/json"
	"fmt"
	"strings"
	"unicode/utf8"

	"github.com/townsendmerino/goinfer/internal/app/serveapp/jobs"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/media"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/streamframe"
)

// ChatChunk formats one OpenAI chat.completion.chunk SSE payload.
func ChatChunk(id string, created int64, model string, d Delta, finish *string) map[string]any {
	return map[string]any{
		"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
		"choices": []any{map[string]any{"index": 0, "delta": d, "finish_reason": finish}},
	}
}

// UsageChunk formats the trailing OpenAI chat.completion.chunk carrying usage with empty choices.
func UsageChunk(id string, created int64, model string, u jobs.Usage) map[string]any {
	return map[string]any{
		"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
		"choices": []any{}, "usage": u,
	}
}

// CompletionChunk formats one legacy OpenAI text_completion SSE payload.
func CompletionChunk(id string, created int64, model, text string, finish *string) map[string]any {
	return map[string]any{
		"id": id, "object": "text_completion", "created": created, "model": model,
		"choices": []any{map[string]any{"index": 0, "text": text, "finish_reason": finish}},
	}
}

// FinishChunk formats the final choice chunk of an OpenAI streaming turn, attaching empty-answer or reasoning-call markers when set.
func FinishChunk(id string, created int64, model, finish string, e *streamframe.EmptyAnswer, fromReasoning bool) map[string]any {
	c := ChatChunk(id, created, model, Delta{}, &finish)
	ch := c["choices"].([]any)[0].(map[string]any)
	if e != nil {
		ch["goinfer_empty_answer"] = e
	}
	if fromReasoning {
		ch[streamframe.FromReasoningField] = true
	}
	return c
}

// ParseStop decodes OpenAI's stop field (either a single string or an array of strings).
func ParseStop(raw json.RawMessage) []string {
	if len(raw) == 0 {
		return nil
	}
	var one string
	if json.Unmarshal(raw, &one) == nil {
		if one == "" {
			return nil
		}
		return []string{one}
	}
	var many []string
	_ = json.Unmarshal(raw, &many)
	return many
}

// SinglePromptString decodes /v1/completions prompt (string or single-element []string) and rejects batches or token arrays.
func SinglePromptString(raw json.RawMessage) (string, error) {
	if len(raw) == 0 {
		return "", nil
	}
	var one string
	if json.Unmarshal(raw, &one) == nil {
		return one, nil
	}
	var many []string
	if json.Unmarshal(raw, &many) == nil {
		if len(many) == 1 {
			return many[0], nil
		}
		return "", fmt.Errorf("prompt: batch prompts are not supported (%d entries); send a single string", len(many))
	}
	return "", fmt.Errorf("prompt: unsupported shape — send a string; token-id arrays and batches are not supported")
}

// FirstStop returns the byte offset and matched stop sequence of the earliest non-empty stop string in text.
func FirstStop(text string, stops []string) (int, string, bool) {
	cut, which := -1, ""
	for _, st := range stops {
		if st == "" {
			continue
		}
		if i := strings.Index(text, st); i >= 0 && (cut < 0 || i < cut) {
			cut, which = i, st
		}
	}
	if cut < 0 {
		return 0, "", false
	}
	return cut, which, true
}

// StopTailHold returns the byte length of the longest suffix of text that is a proper prefix of any stop sequence.
func StopTailHold(text string, stops []string) int {
	hold := 0
	for _, st := range stops {
		k := min(len(st)-1, len(text))
		for ; k > hold; k-- {
			if strings.HasSuffix(text, st[:k]) {
				hold = k
				break
			}
		}
	}
	return hold
}

// CompleteUTF8 returns the byte length of the longest prefix of s ending on a complete UTF-8 rune boundary.
func CompleteUTF8(s string) int {
	end := 0
	for i := 0; i < len(s); {
		if !utf8.FullRuneInString(s[i:]) {
			break
		}
		_, size := utf8.DecodeRuneInString(s[i:])
		i += size
		end = i
	}
	return end
}

// HumanBytes formats a byte count as B, KiB, MiB, or GiB.
func HumanBytes(n int64) string {
	switch {
	case n >= 1<<30:
		return fmt.Sprintf("%.1f GiB", float64(n)/(1<<30))
	case n >= 1<<20:
		return fmt.Sprintf("%.1f MiB", float64(n)/(1<<20))
	case n >= 1<<10:
		return fmt.Sprintf("%.1f KiB", float64(n)/(1<<10))
	default:
		return fmt.Sprintf("%d B", n)
	}
}

// OmitChatHistoryImages replaces media parts in all but the newest media-bearing message with omission notes.
func OmitChatHistoryImages(msgs []ChatMessage) int {
	cs := make([]*json.RawMessage, len(msgs))
	for i := range msgs {
		cs[i] = &msgs[i].Content
	}
	return media.OmitHistoryMedia(cs, map[string]string{"image_url": media.OlderImageNote, "input_audio": media.OlderAudioNote})
}
