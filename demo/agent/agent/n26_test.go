package agent

import (
	"os"
	"strings"
	"testing"
)

// N-26: agent-web accepted cross-origin POSTs and unbounded bodies, and the session encoded user
// text with SPECIAL-TOKEN PARSING — so a user typing "<|im_start|>assistant" into the chat box
// promoted those bytes to real role tokens, forging the template's own boundaries.
//
// "Demo-grade" is not a security boundary: it binds a port, and one request can occupy the model
// for minutes (a vision turn) or discard the conversation (/api/reset).
func TestAgentWeb_hardening(t *testing.T) {
	web, err := os.ReadFile("../cmd/agent-web/main.go")
	if err != nil {
		t.Skipf("no agent-web main.go: %v", err)
	}
	src := string(web)
	for _, want := range []string{"MaxBytesReader", "sameOrigin(", "limitBody("} {
		if !strings.Contains(src, want) {
			t.Errorf("agent-web lacks %s (N-26)", want)
		}
	}
	// The mutating routes must carry BOTH wrappers. GET /api/info is read-only and cheap, so it
	// is deliberately not wrapped — checking that keeps this from passing by blanket-wrapping.
	// Anchor on the mux REGISTRATION, not on the route string — which also appears in the
	// file's doc comment, and matching that made the first version of this guard fail on prose.
	for _, route := range []string{"POST /api/chat", "POST /api/reset"} {
		reg := `mux.HandleFunc("` + route + `"`
		i := strings.Index(src, reg)
		if i < 0 {
			t.Errorf("route %s is not registered; this guard is watching the wrong file", route)
			continue
		}
		line := strings.SplitN(src[i:], "\n", 2)[0]
		if !strings.Contains(line, "sameOrigin(") || !strings.Contains(line, "limitBody(") {
			t.Errorf("%s is not wrapped in sameOrigin+limitBody: %s", route, line)
		}
	}
}

// The encode half: user text must go through EncodeSegments, which keeps the template's own
// special-token boundaries and encodes everything else as literal.
func TestAgentSession_encodesSegmentsNotRenderedText(t *testing.T) {
	src, err := os.ReadFile("agent.go")
	if err != nil {
		t.Fatalf("read agent.go: %v", err)
	}
	for i, ln := range strings.Split(string(src), "\n") {
		s := strings.TrimSpace(ln)
		if strings.HasPrefix(s, "//") {
			continue // the explanation of the rule is not an instance of it
		}
		if strings.Contains(s, "s.tk.Encode(") {
			t.Errorf("agent.go:%d encodes rendered text with special-token parsing: %s\n"+
				"a user typing a role marker into the chat box then forges a real turn boundary "+
				"(N-26) — use EncodeSegments", i+1, s)
		}
	}
	if !strings.Contains(string(src), "s.tk.EncodeSegments(") {
		t.Error("agent.go never calls EncodeSegments")
	}
}
