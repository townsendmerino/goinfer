package wirecore

import (
	"slices"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/tokenizer"
)

func renderSegments(tm *chat.Template, system string, turns []chat.Turn, tools []chat.Tool) []chat.Segment {
	if len(tools) > 0 {
		return tm.RenderToolsSegments(system, turns, tools)
	}
	return tm.RenderSegments(system, turns)
}

// PrefixBoundary returns how many leading token IDs of ids form the stable system-and-tools prefix before turns[0]'s content.
func PrefixBoundary(tk *tokenizer.Tokenizer, tm *chat.Template, system string, turns []chat.Turn, tools []chat.Tool, ids []int) int {
	if tk == nil || tm == nil || len(turns) == 0 {
		return 0
	}
	a := renderSegments(tm, system, turns, tools)
	changed := slices.Clone(turns)
	changed[0].Content = "\x01" + changed[0].Content
	b := renderSegments(tm, system, changed, tools)
	k := 0
	for k < len(a) && k < len(b) && a[k] == b[k] {
		k++
	}
	if k == 0 || k == len(a) {
		return 0
	}
	pre, err := tk.EncodeSegments(a[:k], false)
	if err != nil || len(pre) == 0 || len(pre) >= len(ids) || !slices.Equal(pre, ids[:len(pre)]) {
		return 0
	}
	return len(pre)
}

// RecurrentKeep returns how many leading token IDs of ids remain identical when turns is extended with a reply and a follow-up user turn.
func RecurrentKeep(tk *tokenizer.Tokenizer, tm *chat.Template, system string, turns []chat.Turn, tools []chat.Tool, ids []int) int {
	if tk == nil || tm == nil || len(turns) == 0 {
		return 0
	}
	segs := renderSegments(tm, system, chat.NextTurns(turns), tools)
	nids, err := tk.EncodeSegments(segs, false)
	if err != nil {
		return 0
	}
	k := 0
	for k < len(ids) && k < len(nids) && ids[k] == nids[k] {
		k++
	}
	if k == len(ids) {
		return 0
	}
	return k
}

// RecurrentMsgBoundary returns how many leading token IDs of ids precede the final turn's content in a multi-turn conversation.
func RecurrentMsgBoundary(tk *tokenizer.Tokenizer, tm *chat.Template, system string, turns []chat.Turn, tools []chat.Tool, ids []int) int {
	if tk == nil || tm == nil || len(turns) < 2 {
		return 0
	}
	a := renderSegments(tm, system, turns, tools)
	changed := slices.Clone(turns)
	last := len(changed) - 1
	changed[last].Content = "\x01" + changed[last].Content
	b := renderSegments(tm, system, changed, tools)
	k := 0
	for k < len(a) && k < len(b) && a[k] == b[k] {
		k++
	}
	if k == 0 || k == len(a) {
		return 0
	}
	pre, err := tk.EncodeSegments(a[:k], false)
	if err != nil || len(pre) == 0 || len(pre) >= len(ids) || !slices.Equal(pre, ids[:len(pre)]) {
		return 0
	}
	return len(pre)
}
