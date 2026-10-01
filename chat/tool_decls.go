package chat

import "strings"

// toolDeclarations selects how the Hermes renderer writes each declared tool inside <tools>: "compact" (goinfer's own: sorted-key compact JSON,
// HTML-escaped — the default) or "template" (the model's own bytes: `{"type": "function", "function": {"name", "description", "parameters"}}`
// with Python's separators and the schema's own key order, as the Qwen2.5 / Qwen3 / Qwen3.5 templates' `tool | tojson` writes it).
//
// It is a string variable, and not a flag or an environment read, so that the two arms of the pre-registered A/B
// (docs/measurements/qwen-tool-declarations-2026-09-30/) are two builds of the SAME source differing only by
// `-ldflags -X github.com/townsendmerino/goinfer/chat.toolDeclarations=template`. It exists only for that measurement: it is removed when
// the result is in, whichever way it goes.
var toolDeclarations = "compact"

// ToolDeclarationsTemplate reports whether this build writes tool declarations in the template's bytes.
func ToolDeclarationsTemplate() bool { return toolDeclarations == "template" }

// detectPyDeclarations reports whether a ChatML template writes each tool declaration with `tool | tojson` — the marker of the Qwen templates
// whose declaration bytes the template renderer reproduces.
func detectPyDeclarations(tmpl string) bool { return strings.Contains(tmpl, "tool | tojson") }

// UsesTemplateDeclarations reports whether t's tool declarations are written in the template's bytes: this build selects them and t's own
// template is one that writes them with `tool | tojson`.
func (t *Template) UsesTemplateDeclarations() bool {
	return t != nil && t.pyDeclarations && ToolDeclarationsTemplate()
}
