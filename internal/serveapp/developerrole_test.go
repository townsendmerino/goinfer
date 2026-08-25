package serveapp

import (
	"encoding/json"
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

// TestAnthropicDeveloperRoleStaysUser pins /v1/messages, where the alias is
// DELIBERATELY not applied — a documented decision, not an omission.
//
// The task doc assumed a developer-role message was structurally impossible on
// this surface because Anthropic carries `system` as a top-level field. It is
// not impossible: anthropicMessage.Role is a free string and anthropicRole maps
// everything that is not "assistant" to a user turn, so "developer" is demoted
// here exactly as it was on the OpenAI surfaces — and there is no role
// validation anywhere in serveapp to catch it first.
//
// It is left demoted anyway. The Anthropic Messages API has no developer role,
// so honoring one would invent behavior upstream does not have on a surface
// whose compatibility bar is "works for the apps that matter"; and nothing sends
// it here, because a client speaking this shape puts its system prompt in the
// top-level field. This test exists so the behavior is pinned rather than
// silent: if it ever starts to matter, it fails here first.
func TestAnthropicDeveloperRoleStaysUser(t *testing.T) {
	req := &anthropicReq{
		System:   json.RawMessage(`"TOP LEVEL SYSTEM"`),
		Messages: []anthropicMessage{{Role: "developer", Content: rawStr("SCAFFOLD")}, {Role: "user", Content: rawStr("hi")}},
	}
	system, turns, aerr := anthropicTurns(req)
	if aerr != nil {
		t.Fatalf("anthropicTurns: %+v", aerr)
	}
	if system != "TOP LEVEL SYSTEM" {
		t.Errorf("system = %q, want the top-level field %q", system, "TOP LEVEL SYSTEM")
	}
	if len(turns) == 0 || turns[0].Role != "user" {
		t.Fatalf("turns = %+v, want the developer message pinned as a user turn", turns)
	}
	if system == "SCAFFOLD" {
		t.Error("the alias must not be applied on /v1/messages")
	}
}
