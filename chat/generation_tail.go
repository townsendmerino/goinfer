package chat

// generationTailProbe is the reply and the next user message GenerationTail renders after the conversation: text no template starts a
// marker with, so the two renders part ways at the reply, never inside it.
const generationTailProbe = "\x00goinfer generation tail probe\x00"

// NextTurns is turns followed by a probe reply and a probe user message: the shape of the next request. Rendering it beside turns, with
// the same system prompt and tools, shows where the next prompt and this one part ways. The user message matters: a template that keeps
// the reasoning of the turns after the last user query (Qwen3.5's tool-loop rule) renders a trailing assistant turn with its think
// block, and drops it once a user turn follows.
func NextTurns(turns []Turn) []Turn {
	return append(append([]Turn(nil), turns...), Turn{Role: "assistant", Content: generationTailProbe}, Turn{Role: "user", Content: generationTailProbe})
}

// GenerationTail is the end of Render(system, turns) that the next turn's prompt does not contain: the part of the generation prompt
// the history render rewrites once the assistant has replied (Qwen3.5 with thinking off: "<think>\n\n</think>\n\n", which its
// history render drops). It is "" for a template whose history keeps the whole generation prompt.
//
// It is found from the template, not per family: the conversation is rendered again with a reply and a new user message after it (the
// next request's shape), and the tail is what the first render has past the two renders' common prefix. A recurrent model's state can
// be kept at the end of the prompt minus this tail and continued by the next turn (docs/queue-engineering.md, B35); callers count it
// in tokens from the two renders' common token prefix, not from this string's length.
func (t *Template) GenerationTail(system string, turns []Turn) string {
	gen := t.Render(system, turns)
	next := t.Render(system, NextTurns(turns))
	i := 0
	for i < len(gen) && i < len(next) && gen[i] == next[i] {
		i++
	}
	return gen[i:]
}
