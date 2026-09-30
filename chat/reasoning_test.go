package chat

import (
	"encoding/json"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// splitAll runs a reply through a fresh splitter in the given chunks.
func splitAll(open, close string, forced bool, chunks []string) (reasoning, content string, inReasoning bool) {
	s := NewThinkSplitter(open, close, forced)
	var r, c strings.Builder
	for _, ch := range chunks {
		a, b := s.Push(ch)
		r.WriteString(a)
		c.WriteString(b)
	}
	inReasoning = s.InReasoning()
	a, b := s.Flush()
	r.WriteString(a)
	c.WriteString(b)
	return r.String(), c.String(), inReasoning
}

var thinkCases = []struct {
	name       string
	open, cls  string
	forced     bool
	reply      string
	wantR      string
	wantC      string
	wantInReas bool // the reply ended inside an unclosed block (truncated)
}{
	{"qwen3 self-opened", "<think>", "</think>", false, "<think>\nlet me see\n</think>\n\nThe answer is 4.", "let me see", "The answer is 4.", false},
	{"9b forced-open, only the close is generated", "<think>", "</think>", true, "Okay, the user wants.\n</think>\n\nHello!", "Okay, the user wants.", "Hello!", false},
	{"0.8B closed prefill: the reply is all content", "<think>", "</think>", false, "Hello there.", "", "Hello there.", false},
	{"empty reasoning block", "<think>", "</think>", false, "<think>\n\n</think>\n\nDone.", "", "Done.", false},
	{"truncated inside the block", "<think>", "</think>", true, "Okay, the user asks for a long", "Okay, the user asks for a long", "", true},
	{"truncated, trailing newlines are trimmed like the template's strip", "<think>", "</think>", false, "<think>\nhmm\n\n", "hmm", "", true},
	{"an open tag inside content is literal and never re-enters", "<think>", "</think>", false, "<think>\nr\n</think>\n\nUse <think> tags.", "r", "Use <think> tags.", false},
	{"a reply that merely mentions the tag mid-text stays content", "<think>", "</think>", false, "The <think> tag opens reasoning.", "", "The <think> tag opens reasoning.", false},
	{"leading whitespace before the open tag", "<think>", "</think>", false, "  \n<think>\nr\n</think>\nA", "r", "A", false},
	{"model re-opens a block the prompt opened", "<think>", "</think>", true, "<think>\nr\n</think>\n\nA", "r", "A", false},
	{"content keeps its own leading space and inner newlines", "<think>", "</think>", false, "<think>\nr\n</think>\n\nA\n\nB ", "r", "A\n\nB ", false},
	{"reasoning keeps inner newlines", "<think>", "</think>", false, "<think>\nline1\n\nline2\n</think>\n\nA", "line1\n\nline2", "A", false},
	{"partial open delimiter then plain text is content", "<think>", "</think>", false, "<thi is not a tag", "", "<thi is not a tag", false},
	{"a reply ending on a partial open delimiter flushes as content", "<think>", "</think>", false, "<thin", "", "<thin", false},
	{"whitespace-only reply flushes as content", "<think>", "</think>", false, " \n", "", " \n", false},
	{"gemma4 channel", "<|channel>thought\n", "<channel|>", false, "<|channel>thought\nwork it out\n<channel|>Final.", "work it out", "Final.", false},
	{"gemma4 closed scaffold: content direct", "<|channel>thought\n", "<channel|>", false, "Final.", "", "Final.", false},
	{"gemma4 empty thought", "<|channel>thought\n", "<channel|>", false, "<|channel>thought\n<channel|>Final.", "", "Final.", false},
}

// chunkings returns every two-way split of s, the byte-at-a-time split, and seeded random partitions.
func chunkings(s string) [][]string {
	var out [][]string
	out = append(out, []string{s})
	for i := 1; i < len(s); i++ {
		out = append(out, []string{s[:i], s[i:]})
	}
	bytewise := make([]string, 0, len(s))
	for i := 0; i < len(s); i++ {
		bytewise = append(bytewise, s[i:i+1])
	}
	out = append(out, bytewise)
	rng := rand.New(rand.NewSource(int64(len(s))*7919 + 1))
	for k := 0; k < 40; k++ {
		var parts []string
		for rest := s; rest != ""; {
			n := 1 + rng.Intn(6)
			if n > len(rest) {
				n = len(rest)
			}
			parts = append(parts, rest[:n])
			rest = rest[n:]
		}
		out = append(out, parts)
	}
	return out
}

func TestThinkSplitter_table(t *testing.T) {
	for _, tc := range thinkCases {
		t.Run(tc.name, func(t *testing.T) {
			r, c, in := splitAll(tc.open, tc.cls, tc.forced, []string{tc.reply})
			if r != tc.wantR || c != tc.wantC || in != tc.wantInReas {
				t.Fatalf("whole reply %q\n got reasoning=%q content=%q inReasoning=%v\nwant reasoning=%q content=%q inReasoning=%v",
					tc.reply, r, c, in, tc.wantR, tc.wantC, tc.wantInReas)
			}
		})
	}
}

// TestThinkSplitter_chunkIndependent is the property that makes streaming safe: however the decoder happens to cut the
// reply into fragments — every split point, one byte at a time, random partitions — the concatenated reasoning and
// content equal the whole-reply result, for every case above.
func TestThinkSplitter_chunkIndependent(t *testing.T) {
	n := 0
	for _, tc := range thinkCases {
		wantR, wantC, wantIn := splitAll(tc.open, tc.cls, tc.forced, []string{tc.reply})
		for _, chunks := range chunkings(tc.reply) {
			r, c, in := splitAll(tc.open, tc.cls, tc.forced, chunks)
			n++
			if r != wantR || c != wantC || in != wantIn {
				t.Fatalf("%s: chunks %q\n got reasoning=%q content=%q in=%v\nwant reasoning=%q content=%q in=%v",
					tc.name, chunks, r, c, in, wantR, wantC, wantIn)
			}
		}
	}
	if n < 1000 {
		t.Fatalf("only %d chunkings exercised; the property is not being tested", n)
	}
}

// A well-formed reply never leaks a delimiter into either output.
func TestThinkSplitter_noDelimiterLeaks(t *testing.T) {
	for _, tc := range thinkCases {
		if tc.wantInReas || strings.Contains(tc.wantC, tc.open) { // truncated, or content that legitimately names the tag
			continue
		}
		r, c, _ := splitAll(tc.open, tc.cls, tc.forced, []string{tc.reply})
		for _, s := range []string{r, c} {
			if strings.Contains(s, tc.open) || strings.Contains(s, tc.cls) {
				t.Errorf("%s: delimiter leaked into %q", tc.name, s)
			}
		}
	}
}

// Content must be emitted as soon as the reasoning closes, not held to the end: a streaming client would otherwise see
// nothing until Flush.
func TestThinkSplitter_streamsContentAfterClose(t *testing.T) {
	s := NewThinkSplitter("<think>", "</think>", false)
	if r, c := s.Push("<think>\nab"); r != "ab" || c != "" {
		t.Fatalf("got %q %q", r, c)
	}
	if r, c := s.Push("c</think>\n\nHel"); r != "c" || c != "Hel" {
		t.Fatalf("got %q %q", r, c)
	}
	if r, c := s.Push("lo"); r != "" || c != "lo" {
		t.Fatalf("got %q %q", r, c)
	}
}

// ---------------------------------------------------------------------------------------------------------------------

type thinkGoldenCase struct {
	Conv   string `json:"conv"`
	Mode   string `json:"mode"`
	Tools  bool   `json:"tools"`
	Prompt string `json:"prompt"`
	IDs    []int  `json:"ids"`
}
type thinkGolden struct {
	Checkpoint   string            `json:"checkpoint"`
	ChatTemplate string            `json:"chat_template"`
	Cases        []thinkGoldenCase `json:"cases"`
}

func loadThinkGoldens(t *testing.T) []thinkGolden {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "testdata", "chat_think_goldens", "think_modes.json"))
	if err != nil {
		t.Fatal(err)
	}
	var gs []thinkGolden
	if err := json.Unmarshal(raw, &gs); err != nil {
		t.Fatal(err)
	}
	if len(gs) != 4 {
		t.Fatalf("expected the four pinned checkpoints, got %d", len(gs))
	}
	return gs
}

func convTurns(conv string) (string, []Turn) {
	switch conv {
	case "user":
		return "", []Turn{{Role: "user", Content: "Hi"}}
	case "sys_user":
		return "Be brief.", []Turn{{Role: "user", Content: "Hi"}}
	case "multi":
		return "", []Turn{{Role: "user", Content: "Hi"}, {Role: "assistant", Content: "Hello!"}, {Role: "user", Content: "And?"}}
	}
	panic(conv)
}

var goldenMode = map[string]ThinkMode{"unset": ThinkTemplate, "false": ThinkOff, "true": ThinkOn}

var goldenTools = []Tool{{Name: "get_weather", Description: "Weather for a city",
	Parameters: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`)}}

// TestThinkModes_matchHF pins the PROMPT half, per checkpoint: Detect runs on the real chat template text, and every
// thinking setting renders byte-for-byte what HuggingFace renders from that same template. One checkpoint proving nothing
// about another is the lesson here — the 0.8B and 9B differ only in which way the enable_thinking test points.
func TestThinkModes_matchHF(t *testing.T) {
	ran := 0
	for _, g := range loadThinkGoldens(t) {
		tmpl, err := Detect(Meta{ChatTemplate: g.ChatTemplate})
		if err != nil {
			t.Fatalf("%s: Detect: %v", g.Checkpoint, err)
		}
		if tmpl.Reasoning() == nil {
			t.Fatalf("%s: thinking control not recognised from its real template", g.Checkpoint)
		}
		for _, c := range g.Cases {
			if c.Tools {
				continue // below
			}
			system, turns := convTurns(c.Conv)
			got := tmpl.WithThinking(goldenMode[c.Mode]).Render(system, turns)
			ran++
			if got != c.Prompt {
				t.Errorf("%s %s enable_thinking=%s:\n got %q\nwant %q", g.Checkpoint, c.Conv, c.Mode, got, c.Prompt)
			}
		}
	}
	if ran != 4*9 {
		t.Fatalf("ran %d text cases, want 36", ran)
	}
}

// TestThinkModes_toolsMatchHF: Gemma 4's tool prompt, byte-exact in every thinking setting. (Qwen's own tool syntax is
// XML, goinfer's is Hermes JSON — a known, separate gap — so for Qwen only the generation-prompt ending is checked, below.)
func TestThinkModes_toolsMatchHF(t *testing.T) {
	ran := 0
	for _, g := range loadThinkGoldens(t) {
		if !strings.HasPrefix(g.Checkpoint, "gemma-4") {
			continue
		}
		tmpl, _ := Detect(Meta{ChatTemplate: g.ChatTemplate})
		for _, c := range g.Cases {
			if !c.Tools {
				continue
			}
			system, turns := convTurns(c.Conv)
			got := tmpl.WithThinking(goldenMode[c.Mode]).RenderTools(system, turns, goldenTools)
			ran++
			if got != c.Prompt {
				t.Errorf("%s tools %s enable_thinking=%s:\n got %q\nwant %q", g.Checkpoint, c.Conv, c.Mode, got, c.Prompt)
			}
		}
	}
	if ran != 6 {
		t.Fatalf("ran %d tool cases, want 6", ran)
	}
}

// Every Qwen tool prompt must end with the same think suffix the no-tools prompt does in that mode.
func TestThinkModes_chatmlToolsEndLikeText(t *testing.T) {
	for _, g := range loadThinkGoldens(t) {
		if strings.HasPrefix(g.Checkpoint, "gemma-4") {
			continue
		}
		tmpl, _ := Detect(Meta{ChatTemplate: g.ChatTemplate})
		for mode, m := range goldenMode {
			text := tmpl.WithThinking(m).Render("", []Turn{{Role: "user", Content: "Hi"}})
			tools := tmpl.WithThinking(m).RenderTools("", []Turn{{Role: "user", Content: "Hi"}}, goldenTools)
			tail := text[strings.LastIndex(text, "<|im_start|>assistant\n"):]
			if !strings.HasSuffix(tools, tail) {
				t.Errorf("%s enable_thinking=%s: tool prompt ends %q, text prompt ends %q", g.Checkpoint, mode, tools[len(tools)-40:], tail)
			}
		}
	}
}

// ThinkAsIs is today's bytes: it must equal the generic rendering for every checkpoint, and be a strict prefix of
// every thinking rendering of a ChatML-family template (a suffix is only ever appended).
func TestThinkModes_asIsIsTodaysBytes(t *testing.T) {
	for _, g := range loadThinkGoldens(t) {
		tmpl, _ := Detect(Meta{ChatTemplate: g.ChatTemplate})
		var generic *Template
		if tmpl.Name() == "gemma4" {
			generic = Gemma4()
		} else {
			generic = ChatML()
		}
		for _, conv := range []string{"user", "sys_user", "multi"} {
			system, turns := convTurns(conv)
			asis := tmpl.Render(system, turns)
			if want := generic.Render(system, turns); asis != want {
				t.Errorf("%s %s: as-is %q != generic %q", g.Checkpoint, conv, asis, want)
			}
			if tmpl.Name() != "chatml" {
				continue
			}
			for _, m := range []ThinkMode{ThinkTemplate, ThinkOn, ThinkOff} {
				if got := tmpl.WithThinking(m).Render(system, turns); !strings.HasPrefix(got, asis) {
					t.Errorf("%s %s mode %s: %q does not extend the as-is prompt %q", g.Checkpoint, conv, m, got, asis)
				}
			}
		}
	}
}

// The defaults and the open/closed shape per checkpoint, stated as facts about the real templates (2026-09-30).
func TestThinkModes_detectedFacts(t *testing.T) {
	want := map[string]struct {
		defaultOn bool
		opensOn   bool // PromptOpensThink under ThinkTemplate
	}{
		"qwen3-4b":           {true, false},
		"qwen3.5-0.8b":       {false, false},
		"qwen3.5-9b":         {true, true},
		"gemma-4-26b-a4b-it": {false, false},
	}
	for _, g := range loadThinkGoldens(t) {
		tmpl, _ := Detect(Meta{ChatTemplate: g.ChatTemplate})
		w := want[g.Checkpoint]
		if got := tmpl.Reasoning().DefaultOn(); got != w.defaultOn {
			t.Errorf("%s: DefaultOn=%v, want %v", g.Checkpoint, got, w.defaultOn)
		}
		if got := tmpl.WithThinking(ThinkTemplate).PromptOpensThink(); got != w.opensOn {
			t.Errorf("%s: PromptOpensThink(template)=%v, want %v", g.Checkpoint, got, w.opensOn)
		}
		if tmpl.WithThinking(ThinkOff).PromptOpensThink() {
			t.Errorf("%s: a thinking-off prompt must never end inside an open block", g.Checkpoint)
		}
	}
}

// The fail-toward-today's-bytes rule, from both directions: templates that do not think get no spec, and a ChatML
// template whose control is not one of the three shapes read from real checkpoints is left unmanaged rather than guessed.
func TestThinkModes_unrecognisedIsUnmanaged(t *testing.T) {
	plain := "{% for m in messages %}<|im_start|>{{m.role}}\n{{m.content}}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
	weird := "<|im_start|>{% if add_generation_prompt %}<|im_start|>assistant\n{% if enable_thinking is defined and enable_thinking is none %}{{ '<think>\\n' }}{% endif %}{% endif %}"
	for name, tmplText := range map[string]string{"qwen2.5-like": plain, "unrecognised control": weird} {
		tmpl, err := Detect(Meta{ChatTemplate: tmplText})
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if tmpl.Reasoning() != nil {
			t.Errorf("%s: got a reasoning spec, want nil", name)
		}
		user := []Turn{{Role: "user", Content: "Hi"}}
		base := tmpl.Render("", user)
		for _, m := range []ThinkMode{ThinkTemplate, ThinkOn, ThinkOff} {
			if got := tmpl.WithThinking(m).Render("", user); got != base {
				t.Errorf("%s mode %s changed the bytes: %q vs %q", name, m, got, base)
			}
		}
		if tmpl.NewReasoningSplitter() != nil {
			t.Errorf("%s: an unmanaged template must not split", name)
		}
	}
}

// TestThinkModes_idsMatchHF pins that the tagged segments tokenize to the ids HuggingFace produces — the
// <think> / </think> / <|think|> / channel markers must come out as their control tokens. Needs the real tokenizer files;
// skipped (visibly) without them, so run with -v and read for PASS.
func TestThinkModes_idsMatchHF(t *testing.T) {
	home, _ := os.UserHomeDir()
	ran := 0
	for _, g := range loadThinkGoldens(t) {
		dir := filepath.Join(home, "models", g.Checkpoint)
		if _, err := os.Stat(filepath.Join(dir, "tokenizer.json")); err != nil {
			t.Logf("SKIP %s: no tokenizer at %s", g.Checkpoint, dir)
			continue
		}
		tk, err := tokenizer.Load(dir)
		if err != nil {
			t.Fatalf("%s: %v", g.Checkpoint, err)
		}
		tmpl, _ := Detect(Meta{ChatTemplate: g.ChatTemplate})
		for _, c := range g.Cases {
			system, turns := convTurns(c.Conv)
			tm := tmpl.WithThinking(goldenMode[c.Mode])
			segs := tm.RenderSegments(system, turns)
			if c.Tools {
				segs = tm.RenderToolsSegments(system, turns, goldenTools)
			}
			ids, err := tk.EncodeSegments(segs, false)
			if err != nil {
				t.Fatal(err)
			}
			ran++
			if len(ids) != len(c.IDs) {
				t.Errorf("%s %s/%s tools=%v: %d ids, HF %d", g.Checkpoint, c.Conv, c.Mode, c.Tools, len(ids), len(c.IDs))
				continue
			}
			for i := range ids {
				if ids[i] != c.IDs[i] {
					t.Errorf("%s %s/%s tools=%v: id[%d]=%d, HF %d", g.Checkpoint, c.Conv, c.Mode, c.Tools, i, ids[i], c.IDs[i])
					break
				}
			}
		}
	}
	t.Logf("%d prompts tokenized and compared against HF ids", ran)
}

// TestThinkModes_delimitersDecodeToTheirSurfaceForm: the splitter works on DECODED text, so the real tokenizer must decode
// the delimiter tokens to their literal surface form (serve's streamTokens appends DecodePiece(id) per token). Checked
// against each checkpoint's real tokenizer, because the serve-level splitter tests use a synthetic vocabulary and would pass
// even if a real tokenizer dropped special tokens on decode — the one thing that would make the whole split a no-op.
// Skipped, visibly, without the tokenizer files; run with -v.
func TestThinkModes_delimitersDecodeToTheirSurfaceForm(t *testing.T) {
	home, _ := os.UserHomeDir()
	want := map[string][]string{
		"qwen3-4b":           {"<think>", "</think>"},
		"qwen3.5-0.8b":       {"<think>", "</think>"},
		"qwen3.5-9b":         {"<think>", "</think>"},
		"gemma-4-26b-a4b-it": {"<|channel>", "<channel|>"},
	}
	ran := 0
	for name, toks := range want {
		dir := filepath.Join(home, "models", name)
		if _, err := os.Stat(filepath.Join(dir, "tokenizer.json")); err != nil {
			t.Logf("SKIP %s: no tokenizer at %s", name, dir)
			continue
		}
		tk, err := tokenizer.Load(dir)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		for _, lit := range toks {
			id, ok := tk.TokenID(lit)
			if !ok {
				t.Errorf("%s: %q is not a single token", name, lit)
				continue
			}
			piece, _ := tk.DecodePiece(id)
			ran++
			if piece != lit {
				t.Errorf("%s: DecodePiece(%d) = %q, want the literal %q — the splitter would never see the delimiter", name, id, piece, lit)
			}
		}
	}
	t.Logf("%d delimiter tokens decoded and compared", ran)
}
