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

// recurrentKeep is how many leading ids of ids, the prompt rendered from (system, turns, tools), the next turn's prompt repeats: ids rendered
// again with a reply and a new user message after them (chat.NextTurns) share that many leading ids with ids. A recurrent model's state is
// kept there (KVScope.Keep, docs/queue-engineering.md B35), because a template that rewrites the end of the prompt in its history render
// (Qwen3.5's empty think block) leaves no other point the next turn continues from. 0 for a model whose snapshots do not carry its state,
// when the next prompt repeats all of ids (the slot's own state serves it), or when nothing is shared.
func (lm *loadedModel) recurrentKeep(tm *chat.Template, system string, turns []chat.Turn, tools []chat.Tool, ids []int) int {
	if lm.model == nil || !lm.model.KeepsRecurrentSnapshots() || tm == nil || len(turns) == 0 {
		return 0
	}
	next := chat.NextTurns(turns)
	var segs []chat.Segment
	if len(tools) > 0 {
		segs = tm.RenderToolsSegments(system, next, tools)
	} else {
		segs = tm.RenderSegments(system, next)
	}
	nids, err := lm.tk.EncodeSegments(segs, false)
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
