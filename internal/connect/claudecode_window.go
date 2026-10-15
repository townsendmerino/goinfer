package connect

import (
	"fmt"
	"strings"
)

const (
	// claudeCodeMargin is kept free beside Claude Code's first request and its output cap, because Claude Code compacts
	// somewhat before its context plus the cap reaches the window
	// (docs/measurements/integration-harness-2026-10/i3-night-2026-10-13.md, "I1's live gate").
	claudeCodeMargin = 2048
	// claudeCodeMaxOutput is the most connect writes; claudeCodeMinOutput is the least it writes without a warning.
	claudeCodeMaxOutput = 8192
	claudeCodeMinOutput = 4096
	// claudeCodeFloor keeps the cap Claude Code is given positive in a window too small for its first request (a
	// -force write), at the least Claude Code's thinking budget takes (its env-vars reference, MAX_THINKING_TOKENS).
	claudeCodeFloor = 1024
)

// claudeCodeCap is the output cap connect writes for Claude Code (CLAUDE_CODE_MAX_OUTPUT_TOKENS): what the window
// leaves beside the first request and the margin, at most claudeCodeMaxOutput. Claude Code counts the cap against the
// window before it auto-compacts, so a cap the window cannot spare makes it compact at once.
func (h harness) claudeCodeCap(window int) int {
	return max(min(claudeCodeMaxOutput, window-h.FirstRequest-claudeCodeMargin), claudeCodeFloor)
}

// claudeCodeSettings gives Claude Code claudeCodeCap in place of the window-scaled limit other agents get.
func claudeCodeSettings(name string, h harness, s settings) settings {
	if name == "claude-code" {
		s.Output = h.claudeCodeCap(s.Window)
	}
	return s
}

// claudeCodeWarnings is what connect adds for Claude Code: a warning when its output cap comes out under
// claudeCodeMinOutput, and one for a model id to which Claude Code does not apply CLAUDE_CODE_MAX_CONTEXT_TOKENS.
// Empty for every other agent.
func claudeCodeWarnings(name string, h harness, s settings) string {
	if name != "claude-code" {
		return ""
	}
	var b strings.Builder
	if c := h.claudeCodeCap(s.Window); c < claudeCodeMinOutput {
		need := h.FirstRequest + claudeCodeMargin + claudeCodeMinOutput
		if s.MaxPos > 0 && s.MaxPos < need {
			fmt.Fprintf(&b, "  warning: Claude Code's output cap is %d; this model cannot hold Claude Code's first request (%d tokens) with "+
				"room to answer: its own maximum context is %d, under the %d needed\n", c, h.FirstRequest, s.MaxPos, need)
		} else {
			fmt.Fprintf(&b, "  warning: Claude Code's output cap is %d, under %d, and it will compact almost at once: restart goinfer-serve "+
				"with -ctx %d or more\n", c, claudeCodeMinOutput, need)
		}
	}
	// Claude Code applies CLAUDE_CODE_MAX_CONTEXT_TOKENS directly only to an id it cannot resolve to a Claude model, and
	// not to a [1m] variant (code.claude.com/docs/en/model-config, "Correct the window for a gateway or custom model ID").
	if id := strings.ToLower(s.Model); strings.Contains(id, "claude-") || strings.Contains(id, "[1m]") {
		fmt.Fprintf(&b, "  warning: Claude Code may read the model id %q as a Claude model, and then CLAUDE_CODE_MAX_CONTEXT_TOKENS "+
			"may not apply: serve the model under another name (-model name=path)\n", s.Model)
	}
	return b.String()
}
