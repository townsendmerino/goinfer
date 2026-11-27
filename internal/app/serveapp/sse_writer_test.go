package serveapp

import (
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"
)

// Every lm.drive/driveVL STREAMING site must start a heartbeat, not only the buffer-then-stream ones.
// TestSSE_everyBufferedStreamSiteHeartbeats below classifies only callbacks that literally call lm.drive( and
// exempts sites that stream per token (unconditionalStreamCall); neither holds for the PREFILL window, which
// is silent however a site streams once tokens start, and which applies identically to drive and driveVL. The
// sites are listed in the body rather than widening that classifier to a shape it was not designed for. Audit
// N-24.
func TestSSE_everyStreamingDriveSiteHeartbeats(t *testing.T) {
	sites := []struct{ file, fn string }{
		{"openai.go", "serveChatText"},
		{"openai.go", "serveCompletion"},
		{"anthropic_stream.go", "streamMessages"},
		{"vision_serve.go", "serveVisionChatWith"},
		{"vision_serve.go", "serveVisionMessages"},
		{"responses.go", "serveResponsesWith"},
		{"responses.go", "serveVisionResponses"}, // S11: input_image on /v1/responses
	}
	fnRe := regexp.MustCompile(`(?ms)^func (?:\([^)]*\) )?(\w+)\([^\n]*\{\n(.*?)\n\}`)
	for _, site := range sites {
		b, err := os.ReadFile(site.file)
		if err != nil {
			t.Fatalf("read %s: %v", site.file, err)
		}
		var body string
		for _, m := range fnRe.FindAllStringSubmatch(string(b), -1) {
			if m[1] == site.fn {
				body = m[2]
				break
			}
		}
		if body == "" {
			t.Fatalf("%s: function %s not found — this test's site list is stale", site.file, site.fn)
		}
		if !strings.Contains(body, "sseHeartbeat(") {
			t.Errorf("%s.%s streams via drive/driveVL under an SSE writer but starts no "+
				"heartbeat — a CPU prefill (minutes for an image) sends nothing until the first "+
				"token, against harness idle timeouts (N-24, docs/audit-2026-09-10.md)",
				site.file, site.fn)
		}
	}
}

// Every buffer-then-stream site must start a heartbeat. The set is definable, so it is checked rather than
// remembered: a handler that drives a generation whose callback only appends to a builder, writing nothing to
// the client, is silent for the whole generation by construction. The runtime gates (heartbeat_test.go) cannot
// catch a missing site: they need GOINFER_SERVE_MODEL and test only sites that already have one. Audit M-19.
func TestSSE_everyBufferedStreamSiteHeartbeats(t *testing.T) {
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	slices.Sort(files)
	// func ... { ... } at column 0 — one top-level function per match.
	fnRe := regexp.MustCompile(`(?ms)^func (?:\([^)]*\) )?(\w+)\([^\n]*\{\n(.*?)\n\}`)
	checked, buffered := 0, []string{}
	for _, f := range files {
		if strings.HasSuffix(f, "_test.go") {
			continue
		}
		b, err := os.ReadFile(f)
		if err != nil {
			continue
		}
		for _, m := range fnRe.FindAllStringSubmatch(string(b), -1) {
			name, body := m[1], m[2]
			if !strings.Contains(body, "lm.drive(") {
				continue
			}
			checked++
			// It must be ON a streaming path: it holds an *sseWriter, either as a parameter or by
			// opening the stream itself. Without this, serveMessagesWith matches — it dispatches to
			// streamMessages AND has its own non-streaming drive whose callback buffers, and a
			// heartbeat there would be writing keep-alives to a client that is not streaming.
			decl := m[0][:strings.Index(m[0], "{")]
			onStream := strings.Contains(decl, "ss *sseWriter") ||
				strings.Contains(body, "sseStart(w)") || strings.Contains(body, "anthropicSSEStart(w)")
			if !onStream {
				continue
			}
			// THE CALLBACK, NOT THE BODY: streamMessagesTools writes plenty (tool_use blocks, message_end), but all of it
			// AFTER drive returns. What makes a site silent for the whole generation is that its drive CALLBACK sends
			// nothing.
			cb := driveCallback(body)
			if cb == "" {
				continue // not the literal-callback shape this classifier understands
			}
			if unconditionalStreamCall(cb) {
				continue // streams as it generates: the client hears from it either way
			}
			buffered = append(buffered, f+":"+name)
			if !strings.Contains(body, "sseHeartbeat(") {
				t.Errorf("%s in %s buffers the whole generation (its drive callback only appends to "+
					"a builder) and starts NO heartbeat — it sends zero bytes for the entire "+
					"generation. That is audit-2026-09-02 M-19, which was G19 reaching two of "+
					"three sites; this is the check that counts them.", name, f)
			}
		}
	}
	if checked == 0 {
		t.Fatal("no lm.drive call site found — the scan is broken, and a broken scan makes this " +
			"gate pass over nothing")
	}
	if len(buffered) == 0 {
		t.Fatalf("scanned %d drive site(s) and classified NONE as buffer-then-stream; the "+
			"classifier no longer matches the code it is about", checked)
	}
	t.Logf("%d drive site(s), %d buffer-then-stream: %s", checked, len(buffered), strings.Join(buffered, " "))
}

// driveCallback returns the text of the `func(t string) { … }` argument to an lm.drive call, or ""
// when the call does not use that literal shape.
func driveCallback(body string) string {
	i := strings.Index(body, "lm.drive(")
	if i < 0 {
		return ""
	}
	j := strings.Index(body[i:], "func(t string) {")
	if j < 0 {
		return ""
	}
	start := i + j
	depth := 0
	for k := start; k < len(body); k++ {
		switch body[k] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return body[start : k+1]
			}
		}
	}
	return ""
}

// unconditionalStreamCall reports whether cb, a driveCallback result, calls one of the per-token streaming
// sends at the callback's OWN TOP LEVEL, i.e. on every invocation, not somewhere inside it behind a
// conditional. serveChatToolsWith's callback is the shape that matters: sseSend is reachable only once
// prose.Push has enough content to flush, so most tokens hit neither branch, and a plain strings.Contains(cb,
// "sseSend(") would exempt the site without ever looking for its sseHeartbeat. It walks brace depth relative
// to the callback's own `{` (depth 1 is the top level), as driveCallback does to find the closing brace. V-21
// (docs/review-2026-09-04.md).
func unconditionalStreamCall(cb string) bool {
	markers := []string{"sseSend(", "sseEvent(", "anthropicEvent(", "ss.frame(", "ss.Frame("}
	depth := 0
	for i := 0; i < len(cb); i++ {
		switch cb[i] {
		case '{':
			depth++
		case '}':
			depth--
		}
		if depth != 1 {
			continue
		}
		for _, m := range markers {
			if strings.HasPrefix(cb[i:], m) {
				return true
			}
		}
	}
	return false
}
