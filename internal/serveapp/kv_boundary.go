package serveapp

import (
	"slices"

	"github.com/townsendmerino/goinfer/chat"
)

// prefixBoundary is how many leading ids of ids, the prompt rendered from (system, turns, tools), come before the first conversation turn's content: the
// stable system-and-tools section, which every session of one client shares and which a prefix snapshot is taken at during the cold prefill
// (docs/tasks/task-kv7-step2-boundary-snapshot-2026-10.md). 0 when there is none to find or the server cannot use one.
//
// It is found without knowing the family's markers: the prompt is rendered again with the first turn's content changed, and the segments that
// render identically are the stable section (the first turn's opening marker is in it; the content that differs is not). The ids of those
// segments are a prefix of ids, because EncodeSegments tokenizes segment by segment, and that is checked rather than assumed.
func (lm *loadedModel) prefixBoundary(tm *chat.Template, system string, turns []chat.Turn, tools []chat.Tool, ids []int) int {
	if lm.model == nil || !lm.model.KVSnapshotsSupported() {
		return 0
	}
	return lm.prefixBoundaryOf(tm, system, turns, tools, ids)
}

// prefixBoundaryOf is prefixBoundary for a server that can use one (the model check is the caller's).
func (lm *loadedModel) prefixBoundaryOf(tm *chat.Template, system string, turns []chat.Turn, tools []chat.Tool, ids []int) int {
	if tm == nil || len(turns) == 0 {
		return 0
	}
	render := func(ts []chat.Turn) []chat.Segment {
		if len(tools) > 0 {
			return tm.RenderToolsSegments(system, ts, tools)
		}
		return tm.RenderSegments(system, ts)
	}
	a := render(turns)
	changed := slices.Clone(turns)
	changed[0].Content = "\x01" + changed[0].Content
	b := render(changed)
	k := 0
	for k < len(a) && k < len(b) && a[k] == b[k] {
		k++
	}
	if k == 0 || k == len(a) {
		return 0
	}
	pre, err := lm.tk.EncodeSegments(a[:k], false)
	if err != nil || len(pre) == 0 || len(pre) >= len(ids) || !slices.Equal(pre, ids[:len(pre)]) {
		return 0
	}
	return len(pre)
}
