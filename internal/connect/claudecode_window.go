package connect

import (
	"fmt"
	"strings"
)

// claudeCodeOutput is the output cap connect writes for Claude Code (CLAUDE_CODE_MAX_OUTPUT_TOKENS) at every window.
// Claude Code counts the cap against the window before it auto-compacts, so a cap that grew with the window would
// spend the room it is given; a reply longer than the cap is cut off at it.
const claudeCodeOutput = 4096

// claudeCodeSettings gives Claude Code its fixed output cap in place of the window-scaled limit other agents get.
func claudeCodeSettings(name string, s settings) settings {
	if name == "claude-code" {
		s.Output = claudeCodeOutput
	}
	return s
}

// claudeCodeNeeds is the window Claude Code's first request and its output cap fill on their own.
func (h harness) claudeCodeNeeds() int { return h.FirstRequest + claudeCodeOutput }

// claudeCodeThreshold is the window above which connect says nothing about Claude Code's own compaction. The margin
// beyond claudeCodeNeeds is a quarter of the first request, the margin fit calls viable, because Claude Code compacts
// some way before its context plus the cap reaches the window and the rest has to hold at least one tool result
// (docs/measurements/integration-harness-2026-10/i3-night-2026-10-13.md, "I1's live gate").
func (h harness) claudeCodeThreshold() int { return h.claudeCodeNeeds() + h.FirstRequest/4 }

// claudeCodeWarnings is what connect adds for Claude Code about the window it is given: a warning when the first
// request and the output cap do not fit it, a one-line note when they fit with less than the margin, and a warning for
// a model id to which Claude Code does not apply CLAUDE_CODE_MAX_CONTEXT_TOKENS. Empty for every other agent.
func claudeCodeWarnings(name string, h harness, s settings) string {
	if name != "claude-code" {
		return ""
	}
	var b strings.Builder
	switch needs, want := h.claudeCodeNeeds(), h.claudeCodeThreshold(); {
	case s.Window < needs && s.MaxPos > 0 && s.MaxPos < needs:
		fmt.Fprintf(&b, "  warning: Claude Code's first request (%d tokens) and its output cap (%d) need %d, and this model cannot hold "+
			"them: its own maximum context is %d\n", h.FirstRequest, claudeCodeOutput, needs, s.MaxPos)
	case s.Window < needs:
		ctx := want
		if s.MaxPos > 0 {
			ctx = min(ctx, s.MaxPos)
		}
		fmt.Fprintf(&b, "  warning: Claude Code's first request (%d tokens) and its output cap (%d) need %d, more than this window: "+
			"restart goinfer-serve with -ctx %d or more (if this box cannot hold that, it cannot run Claude Code on this model)\n",
			h.FirstRequest, claudeCodeOutput, needs, ctx)
	case s.Window < want:
		fmt.Fprintf(&b, "  note: Claude Code will auto-compact often: %d tokens beside its first request and output cap (-ctx %d leaves %d)\n",
			s.Window-needs, want, want-needs)
	}
	// Claude Code applies CLAUDE_CODE_MAX_CONTEXT_TOKENS directly only to an id it cannot resolve to a Claude model, and
	// not to a [1m] variant (code.claude.com/docs/en/model-config, "Correct the window for a gateway or custom model ID").
	if id := strings.ToLower(s.Model); strings.Contains(id, "claude-") || strings.Contains(id, "[1m]") {
		fmt.Fprintf(&b, "  warning: Claude Code may read the model id %q as a Claude model, and then CLAUDE_CODE_MAX_CONTEXT_TOKENS "+
			"may not apply: serve the model under another name (-model name=path)\n", s.Model)
	}
	return b.String()
}
