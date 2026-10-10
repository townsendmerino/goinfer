package serveapp

import (
	"encoding/json"
	"strconv"
)

// Images already in a conversation's history (docs/tasks/task-first-hour.md). The vision path takes one image span per
// generation (decoder.GenerateVL and its Qwen and Gemma 4 variants), and a chat client resends the whole conversation,
// so its second image turn would carry both images and fail with "v1 supports 1 image per request, got 2". There is NO
// multi-image history: the newest image is kept and every image in an EARLIER message is replaced by a visible note, so
// the model knows something was there rather than hearing nothing, and the client is told with a header. Several images
// inside the one latest message are a different thing (the caller asked for them together) and are taken, each its own
// block, up to maxImagesPerTurn (docs/tasks/task-multimodal-support-2026-10.md).

// imagesOmittedHeader tells the client how many earlier images this request's answer did not see.
const imagesOmittedHeader = "X-Goinfer-Images-Omitted"

// olderImageNote stands where an omitted image was, in the text the model reads.
const olderImageNote = "[an earlier image in this conversation was omitted: this server keeps only the newest image]"

// olderAudioNote stands where an omitted audio clip was: the generation takes one media span, image or audio, so an
// earlier message's clip is noted the same way, and counted in the same header.
const olderAudioNote = "[an earlier audio clip in this conversation was omitted: this server keeps only the newest image or clip]"

// omitHistoryImages rewrites msgs in place: every image part in a message BEFORE the last message that carries one becomes a text part holding
// olderImageNote. imageType is the dialect's part type ("image_url" for OpenAI chat, "image" for Anthropic). It returns how many it replaced.
func omitHistoryImages(contents []*json.RawMessage, imageType string) int {
	return omitHistoryMedia(contents, map[string]string{imageType: olderImageNote})
}

// omitHistoryMedia is omitHistoryImages over several part types at once (part type -> the note that replaces it): the
// newest message carrying ANY of them keeps its parts, and every such part in an earlier message is replaced.
func omitHistoryMedia(contents []*json.RawMessage, notes map[string]string) int {
	last := -1
	for i, c := range contents {
		if n, _ := countOrReplaceImages(*c, notes, false); n > 0 {
			last = i
		}
	}
	omitted := 0
	for i, c := range contents {
		if i >= last {
			break
		}
		if n, rewritten := countOrReplaceImages(*c, notes, true); n > 0 {
			*c = rewritten
			omitted += n
		}
	}
	return omitted
}

// countOrReplaceImages counts the media parts (the types in notes) of one message's content array and, when replace is set, returns the content with each of
// them swapped for a text part holding its type's note. A plain-string content, or one that is not an array of objects, has none. Every other field of every
// part is preserved as it came.
func countOrReplaceImages(raw json.RawMessage, notes map[string]string, replace bool) (int, json.RawMessage) {
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
		if json.Unmarshal(p["type"], &t) != nil {
			continue
		}
		note, ok := notes[t]
		if !ok {
			continue
		}
		n++
		if replace {
			text, _ := json.Marshal(note)
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
	return omitHistoryMedia(cs, map[string]string{"image_url": olderImageNote, "input_audio": olderAudioNote})
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
