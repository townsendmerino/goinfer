package serveapp

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
)

// Gates for the `role: "developer"` → `role: "system"` alias (queue G12).
//
// The before-state these replace, verified at dc8355e and pinned by a test shown
// failing against the fix: "developer" matched no arm of messagesToTurns' switch
// and fell through `default:` to a USER turn — not a 400, not a drop. A harness
// that sends its system prompt as "developer" (OpenAI's newer APIs, and the
// agent harnesses following them) had its entire scaffold delivered as the
// user's first message, and the resulting behavior read as a bad model rather
// than a mangled request. The equality gates below are what keep that silent
// failure from returning: they assert the aliased form renders BYTE-IDENTICALLY
// to the equivalent system message, so a future edit that half-recognizes the
// role fails here rather than in someone's agent loop.

// templates covers every family the alias must be invisible to. It is invisible
// by construction — the alias resolves before a template is reached — so this is
// a guard against the alias ever being pushed down into a renderer.
func templates() map[string]*chat.Template {
	return map[string]*chat.Template{
		"chatml":  chat.ChatML(),
		"gemma3":  chat.Gemma3(),
		"gemma4":  chat.Gemma4(),
		"llama3":  chat.Llama3(),
		"mistral": chat.Mistral(),
		"harmony": chat.Harmony(),
	}
}

// TestDeveloperRoleAliasesSystem is the equality gate: for each shape, the
// request written with "developer" must render byte-identically to the same
// request written with "system", in every family.
func TestDeveloperRoleAliasesSystem(t *testing.T) {
	cases := []struct {
		name          string
		dev, sys      []chatMessage
		wantSystem    string
		wantTurnRoles []string
	}{
		{
			name:          "developer only",
			dev:           []chatMessage{{Role: "developer", Content: rawStr("SCAFFOLD")}, {Role: "user", Content: rawStr("hi")}},
			sys:           []chatMessage{{Role: "system", Content: rawStr("SCAFFOLD")}, {Role: "user", Content: rawStr("hi")}},
			wantSystem:    "SCAFFOLD",
			wantTurnRoles: []string{"user"},
		},
		{
			// Precedence is not a new concept: two "system" messages already
			// resolve last-one-wins, and the alias inherits exactly that.
			name:          "system then developer — last wins",
			dev:           []chatMessage{{Role: "system", Content: rawStr("FIRST")}, {Role: "developer", Content: rawStr("SECOND")}, {Role: "user", Content: rawStr("hi")}},
			sys:           []chatMessage{{Role: "system", Content: rawStr("FIRST")}, {Role: "system", Content: rawStr("SECOND")}, {Role: "user", Content: rawStr("hi")}},
			wantSystem:    "SECOND",
			wantTurnRoles: []string{"user"},
		},
		{
			name:          "developer then system — last wins",
			dev:           []chatMessage{{Role: "developer", Content: rawStr("FIRST")}, {Role: "system", Content: rawStr("SECOND")}, {Role: "user", Content: rawStr("hi")}},
			sys:           []chatMessage{{Role: "system", Content: rawStr("FIRST")}, {Role: "system", Content: rawStr("SECOND")}, {Role: "user", Content: rawStr("hi")}},
			wantSystem:    "SECOND",
			wantTurnRoles: []string{"user"},
		},
		{
			name: "developer mid-conversation",
			dev: []chatMessage{
				{Role: "user", Content: rawStr("q1")},
				{Role: "assistant", Content: rawStr("a1")},
				{Role: "developer", Content: rawStr("SCAFFOLD")},
				{Role: "user", Content: rawStr("q2")},
			},
			sys: []chatMessage{
				{Role: "user", Content: rawStr("q1")},
				{Role: "assistant", Content: rawStr("a1")},
				{Role: "system", Content: rawStr("SCAFFOLD")},
				{Role: "user", Content: rawStr("q2")},
			},
			wantSystem:    "SCAFFOLD",
			wantTurnRoles: []string{"user", "assistant", "user"},
		},
		{
			name:          "developer with array content parts",
			dev:           []chatMessage{{Role: "developer", Content: json.RawMessage(`[{"type":"text","text":"SCAF"},{"type":"text","text":"FOLD"}]`)}, {Role: "user", Content: rawStr("hi")}},
			sys:           []chatMessage{{Role: "system", Content: json.RawMessage(`[{"type":"text","text":"SCAF"},{"type":"text","text":"FOLD"}]`)}, {Role: "user", Content: rawStr("hi")}},
			wantSystem:    "SCAFFOLD",
			wantTurnRoles: []string{"user"},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			gotSystem, gotTurns := messagesToTurns(c.dev)
			if gotSystem != c.wantSystem {
				t.Errorf("system = %q, want %q", gotSystem, c.wantSystem)
			}
			if len(gotTurns) != len(c.wantTurnRoles) {
				t.Fatalf("got %d turns, want %d", len(gotTurns), len(c.wantTurnRoles))
			}
			for i, want := range c.wantTurnRoles {
				if gotTurns[i].Role != want {
					t.Errorf("turn %d role = %q, want %q", i, gotTurns[i].Role, want)
				}
			}

			// The equality gate, per family: byte-identical rendered prompt.
			wantSys, wantTurns := messagesToTurns(c.sys)
			for name, tmpl := range templates() {
				got := tmpl.Render(gotSystem, gotTurns)
				want := tmpl.Render(wantSys, wantTurns)
				if got != want {
					t.Errorf("%s: developer form is not byte-identical to the system form\n got: %q\nwant: %q", name, got, want)
				}
			}
			// And the raw-conversation fallback, for unrecognized families.
			if got, want := rawPrompt(gotSystem, gotTurns), rawPrompt(wantSys, wantTurns); got != want {
				t.Errorf("rawPrompt fallback differs\n got: %q\nwant: %q", got, want)
			}
		})
	}
}

// TestDeveloperRoleResponsesInput covers /v1/responses message items, which pass
// the role through verbatim into the same chokepoint.
func TestDeveloperRoleResponsesInput(t *testing.T) {
	dev, err := responseInputToMessages(json.RawMessage(`[{"role":"developer","content":"SCAFFOLD"},{"role":"user","content":"hi"}]`))
	if err != nil {
		t.Fatalf("responseInputToMessages: %v", err)
	}
	sys, err := responseInputToMessages(json.RawMessage(`[{"role":"system","content":"SCAFFOLD"},{"role":"user","content":"hi"}]`))
	if err != nil {
		t.Fatalf("responseInputToMessages: %v", err)
	}
	gotSystem, gotTurns := messagesToTurns(dev)
	wantSystem, wantTurns := messagesToTurns(sys)
	if gotSystem != "SCAFFOLD" {
		t.Errorf("system = %q, want %q", gotSystem, "SCAFFOLD")
	}
	if got, want := chat.ChatML().Render(gotSystem, gotTurns), chat.ChatML().Render(wantSystem, wantTurns); got != want {
		t.Errorf("responses developer item is not byte-identical to the system form\n got: %q\nwant: %q", got, want)
	}
}

// TestDeveloperRoleResponsesInstructions: the Responses `instructions` field is
// separate plumbing — it is constructed as a system message directly and never
// carried a wire role, so the alias must leave it exactly as it was.
func TestDeveloperRoleResponsesInstructions(t *testing.T) {
	msgs := []chatMessage{{Role: "system", Content: rawStr("FROM INSTRUCTIONS")}}
	inputMsgs, err := responseInputToMessages(json.RawMessage(`"hi"`))
	if err != nil {
		t.Fatalf("responseInputToMessages: %v", err)
	}
	system, turns := messagesToTurns(append(msgs, inputMsgs...))
	if system != "FROM INSTRUCTIONS" {
		t.Errorf("system = %q, want %q", system, "FROM INSTRUCTIONS")
	}
	if len(turns) != 1 || turns[0].Role != "user" || turns[0].Content != "hi" {
		t.Errorf("turns = %+v, want one user turn %q", turns, "hi")
	}
}

// TestDeveloperRoleIsNotGeneralRoleTolerance is the non-goal guard. The alias
// names exactly one role; every other unrecognized role keeps the default arm's
// behavior, unchanged by this work.
func TestDeveloperRoleIsNotGeneralRoleTolerance(t *testing.T) {
	for _, role := range []string{"deve1oper", "Developer", "DEVELOPER", "system_", "root", "instructions", ""} {
		system, turns := messagesToTurns([]chatMessage{{Role: role, Content: rawStr("X")}})
		if system != "" {
			t.Errorf("role %q was treated as a system message (system=%q); the alias must name exactly one role", role, system)
		}
		if len(turns) != 1 || turns[0].Role != "user" {
			t.Errorf("role %q: turns = %+v, want the unchanged default (one user turn)", role, turns)
		}
	}
}

// TestAnthropicRejectsIllegalRoles is the G12 pin, FLIPPED by G13.
//
// It used to assert that /v1/messages silently demoted a developer-role message
// to a user turn — pinned deliberately, so the behavior was visible rather than
// silent while the decision was pending. The decision came out the other way:
// the Anthropic Messages API accepts only "user" and "assistant" and rejects
// anything else, so demote-vs-alias was the wrong menu for this surface and
// rejection is the faithful answer.
//
// The class matters more than the instance. "developer" was never special here —
// ANY typo'd or invented role was folded into the conversation, restructuring
// what the model saw. Each case below is a shape that used to be swallowed.
func TestAnthropicRejectsIllegalRoles(t *testing.T) {
	for _, role := range []string{
		"developer", // the instance that exposed the class
		"system",    // the likeliest mistake: system is a top-level field on this API
		"Assistant", // case matters
		"USER",      //
		"sytem",     // a plain typo
		"tool",      // legal on the OpenAI surface, not this one
		"",          // omitted entirely
		"function",  //
	} {
		req := &anthropicReq{
			System:   json.RawMessage(`"TOP LEVEL SYSTEM"`),
			Messages: []anthropicMessage{{Role: role, Content: rawStr("SCAFFOLD")}},
		}
		_, _, aerr := anthropicTurns(req)
		if aerr == nil {
			t.Errorf("role %q was accepted; it must be a clean 400, not folded into the conversation", role)
			continue
		}
		if aerr.code != http.StatusBadRequest || aerr.kind != "invalid_request_error" {
			t.Errorf("role %q: got (%d, %q), want (400, invalid_request_error)", role, aerr.code, aerr.kind)
		}
		if !strings.Contains(aerr.msg, role) && role != "" {
			t.Errorf("role %q: error does not name the offending role: %s", role, aerr.msg)
		}
	}
}

// The legal roles must still work, including the shapes Claude Code actually
// sends — otherwise the validation above is a compatibility break wearing a
// correctness costume.
func TestAnthropicAcceptsLegalRoles(t *testing.T) {
	req := &anthropicReq{
		System: json.RawMessage(`"TOP LEVEL SYSTEM"`),
		Messages: []anthropicMessage{
			{Role: "user", Content: rawStr("q1")},
			{Role: "assistant", Content: rawStr("a1")},
			{Role: "user", Content: json.RawMessage(`[{"type":"text","text":"q2","cache_control":{"type":"ephemeral"}}]`)},
		},
	}
	system, turns, aerr := anthropicTurns(req)
	if aerr != nil {
		t.Fatalf("legal roles rejected: %+v", aerr)
	}
	if system != "TOP LEVEL SYSTEM" {
		t.Errorf("system = %q, want the top-level field", system)
	}
	if len(turns) != 3 {
		t.Fatalf("got %d turns, want 3: %+v", len(turns), turns)
	}
	for i, want := range []string{"user", "assistant", "user"} {
		if turns[i].Role != want {
			t.Errorf("turn %d role = %q, want %q", i, turns[i].Role, want)
		}
	}
}

// An illegal role must surface as a real HTTP 400 with the Anthropic error body,
// on /v1/messages AND on /v1/messages/count_tokens — validating inside
// anthropicTurns is what makes both true, and a test at the function alone would
// not have shown it.
func TestAnthropicIllegalRoleIsHTTP400(t *testing.T) {
	ts := newAnthropicTestServer(t)
	defer ts.Close()

	for _, path := range []string{"/v1/messages", "/v1/messages/count_tokens"} {
		body := `{"model":"test-model","max_tokens":8,"messages":[{"role":"developer","content":"x"}]}`
		resp, err := http.Post(ts.URL+path, "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatalf("post %s: %v", path, err)
		}
		var got struct {
			Type  string `json:"type"`
			Error struct {
				Type    string `json:"type"`
				Message string `json:"message"`
			} `json:"error"`
		}
		_ = json.NewDecoder(resp.Body).Decode(&got)
		resp.Body.Close()

		if resp.StatusCode != http.StatusBadRequest {
			t.Errorf("%s: status = %d, want 400", path, resp.StatusCode)
		}
		if got.Error.Type != "invalid_request_error" {
			t.Errorf("%s: error.type = %q, want invalid_request_error", path, got.Error.Type)
		}
		if !strings.Contains(got.Error.Message, "developer") {
			t.Errorf("%s: message does not name the offending role: %q", path, got.Error.Message)
		}
		t.Logf("%s -> %d %s: %s", path, resp.StatusCode, got.Error.Type, got.Error.Message)
	}
}
