package serveapp

import (
	"encoding/json"
	"strconv"
)

// Images already in a conversation's history (R23, docs/tasks/task-first-hour.md). The vision path takes one image span per generation
// (decoder.GenerateVL and its Qwen and Gemma 4 variants), and a chat client resends the whole conversation, so its second image turn used to carry
// both images and fail with "v1 supports 1 image per request, got 2". Owner decision 2026-10-01: NO multi-image history. The newest image is kept and
// every image in an EARLIER message is replaced by a visible note, so the model knows something was there rather than hearing nothing, and the client
// is told with a header. Several images inside the one latest message are a different thing — the caller asked for them together — and stay a 400.

// imagesOmittedHeader tells the client how many earlier images this request's answer did not see.
const imagesOmittedHeader = "X-Goinfer-Images-Omitted"

// olderImageNote stands where an omitted image was, in the text the model reads.
const olderImageNote = "[an earlier image in this conversation was omitted: this server keeps only the newest image]"

// omitHistoryImages rewrites msgs in place: every image part in a message BEFORE the last message that carries one becomes a text part holding
// olderImageNote. imageType is the dialect's part type ("image_url" for OpenAI chat, "image" for Anthropic). It returns how many it replaced.
func omitHistoryImages(contents []*json.RawMessage, imageType string) int {
	last := -1
	for i, c := range contents {
		if n, _ := countOrReplaceImages(*c, imageType, false); n > 0 {
			last = i
		}
	}
	omitted := 0
	for i, c := range contents {
		if i >= last {
			break
		}
		if n, rewritten := countOrReplaceImages(*c, imageType, true); n > 0 {
			*c = rewritten
			omitted += n
		}
	}
	return omitted
}

// countOrReplaceImages counts the image parts of one message's content array and, when replace is set, returns the content with each of them swapped for
// a text part. A plain-string content, or one that is not an array of objects, has none. Every other field of every part is preserved as it came.
func countOrReplaceImages(raw json.RawMessage, imageType string, replace bool) (int, json.RawMessage) {
	if len(raw) == 0 || raw[0] != '[' {
		return 0, raw
	}
	var parts []map[string]json.RawMessage
	if json.Unmarshal(raw, &parts) != nil {
		return 0, raw
	}
	n := 0
	for i, p := range parts {
		var t string
		if json.Unmarshal(p["type"], &t) != nil || t != imageType {
			continue
		}
		n++
		if replace {
			text, _ := json.Marshal(olderImageNote)
			parts[i] = map[string]json.RawMessage{"type": json.RawMessage(`"text"`), "text": text}
		}
	}
	if !replace || n == 0 {
		return n, raw
	}
	out, err := json.Marshal(parts)
	if err != nil {
		return 0, raw
	}
	return n, out
}

// omitChatHistoryImages is omitHistoryImages for an OpenAI chat request.
func omitChatHistoryImages(msgs []chatMessage) int {
	cs := make([]*json.RawMessage, len(msgs))
	for i := range msgs {
		cs[i] = &msgs[i].Content
	}
	return omitHistoryImages(cs, "image_url")
}

// omitAnthropicHistoryImages is omitHistoryImages for an Anthropic Messages request.
func omitAnthropicHistoryImages(msgs []anthropicMessage) int {
	cs := make([]*json.RawMessage, len(msgs))
	for i := range msgs {
		cs[i] = &msgs[i].Content
	}
	return omitHistoryImages(cs, "image")
}

// noteOmittedImages sets the response header when any image was omitted.
func noteOmittedImages(h interface{ Set(k, v string) }, omitted int) {
	if omitted > 0 {
		h.Set(imagesOmittedHeader, strconv.Itoa(omitted))
	}
}
