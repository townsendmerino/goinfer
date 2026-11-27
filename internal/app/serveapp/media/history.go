package media

import (
	"encoding/json"
	"strconv"
)

// ImagesOmittedHeader reports how many earlier media items in a conversation
// were replaced with omission notes.
const ImagesOmittedHeader = "X-Goinfer-Images-Omitted"

// OlderImageNote replaces an omitted image in an earlier message's text.
const OlderImageNote = "[an earlier image in this conversation was omitted: this server keeps only the newest image]"

// OlderAudioNote replaces an omitted audio clip in an earlier message's text.
const OlderAudioNote = "[an earlier audio clip in this conversation was omitted: this server keeps only the newest image or clip]"

// OmitHistoryImages rewrites contents in place: every image part of imageType
// in a message before the last message carrying one becomes a text part holding
// OlderImageNote. It returns how many parts were replaced.
func OmitHistoryImages(contents []*json.RawMessage, imageType string) int {
	return OmitHistoryMedia(contents, map[string]string{imageType: OlderImageNote})
}

// OmitHistoryMedia keeps media parts in the newest message carrying any type in
// notes and replaces every matching part in earlier messages with its note.
func OmitHistoryMedia(contents []*json.RawMessage, notes map[string]string) int {
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

// NoteOmittedImages sets ImagesOmittedHeader when omitted is positive.
func NoteOmittedImages(h interface{ Set(k, v string) }, omitted int) {
	if omitted > 0 {
		h.Set(ImagesOmittedHeader, strconv.Itoa(omitted))
	}
}
