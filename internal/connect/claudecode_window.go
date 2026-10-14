package connect

import (
	"fmt"
	"strings"
)

// claudeCodeHeadroom is the room Claude Code needs beside its first request and its output limit: it compacts once its
// context plus the output limit reaches the window (CLAUDE_CODE_MAX_OUTPUT_TOKENS in its environment-variable
// reference), so the margin fit calls viable, a quarter of the first request, is counted after both.
func (h harness) claudeCodeHeadroom() int { return h.FirstRequest / 4 }

// claudeCodeRoom is what a window leaves Claude Code to work in before its own auto-compaction fires.
func (h harness) claudeCodeRoom(window int) int { return window - OutputLimit(window) - h.FirstRequest }

// claudeCodeWarnings is what connect adds for Claude Code about the window it is given: a window too small for its
// own compaction to leave working room, and a model id for which Claude Code does not apply
// CLAUDE_CODE_MAX_CONTEXT_TOKENS. Empty for every other agent.
func claudeCodeWarnings(name string, h harness, s settings) string {
	if name != "claude-code" {
		return ""
	}
	var b strings.Builder
	if room := h.claudeCodeRoom(s.Window); room < h.claudeCodeHeadroom() {
		need := s.Window + 1
		for h.claudeCodeRoom(need) < h.claudeCodeHeadroom() {
			need++
		}
		fmt.Fprintf(&b, "  warning: Claude Code compacts on its own when its context plus the output limit (%d) reaches the window; "+
			"beside its first request that leaves %d tokens, under the %d it needs, and it will compact after nearly every step: "+
			"restart goinfer-serve with -ctx %d or more\n", s.Output, max(room, 0), h.claudeCodeHeadroom(), need)
	}
	// Claude Code applies CLAUDE_CODE_MAX_CONTEXT_TOKENS directly only to an id it cannot resolve to a Claude model, and
	// not to a [1m] variant (code.claude.com/docs/en/model-config, "Correct the window for a gateway or custom model ID").
	if id := strings.ToLower(s.Model); strings.Contains(id, "claude-") || strings.Contains(id, "[1m]") {
		fmt.Fprintf(&b, "  warning: Claude Code may read the model id %q as a Claude model, and then CLAUDE_CODE_MAX_CONTEXT_TOKENS "+
			"may not apply: serve the model under another name (-model name=path)\n", s.Model)
	}
	return b.String()
}
