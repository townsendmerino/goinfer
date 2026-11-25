package carriedids

import (
	"encoding/json"
	"errors"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/internal/codec"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// The committed ChatML tokenizer-only fixture (scripts/chatml_tiny_fixture.py): ids 0..255 are the byte-level pieces in
// byte order, 256..270 fifteen merges ("ll" is 265), 271 <|im_start|> and 272 <|im_end|>.
const (
	tinyGGUF = "../../../../tokenizer/testdata/chatml-tiny.gguf"
	imStart  = 271
	imEnd    = 272
	idLL     = 265
)

var testKey = NewKey([]byte("render test key, thirty-two byte"))

func loadTiny(t *testing.T) (*tokenizer.Tokenizer, *chat.Template) {
	t.Helper()
	tk, err := tokenizer.LoadGGUF(tinyGGUF)
	if err != nil {
		t.Fatalf("load %s: %v", tinyGGUF, err)
	}
	tm, err := codec.Template(tk)
	if err != nil {
		t.Fatal(err)
	}
	return tk, tm
}

// carrier is a blob for ids minted and parsed as the routes do, by testKey for model "m" in namespace "ns".
func carrier(t *testing.T, ids []int, whole bool) *Carrier {
	t.Helper()
	b, err := ParseBlob(testKey.Mint("m", "ns", "", ids, whole))
	if err != nil {
		t.Fatal(err)
	}
	return &Carrier{Blob: b}
}

func check(cs map[int]*Carrier) *Check {
	return &Check{Key: testKey, Model: "m", NS: "ns", Carriers: cs}
}

// Pieces of the tiny tokenizer's split of the prompts below.
var (
	pUser      = []int{'u', 's', 264}                // "user" = u s er
	pAssistant = []int{'a', 266, 'i', 263, 270, 't'} // "assistant" = a ss i st an t
	pHello     = []int{'H', 'e', idLL, 'o'}
)

func cat(parts ...[]int) []int { return slices.Concat(parts...) }

var helloTurns = []chat.Turn{{Role: "user", Content: "hi"}, {Role: "assistant", Content: "Hello"}, {Role: "user", Content: "ok"}}

// helloIDs is helloTurns rendered ChatML and tokenized, with hello as the ids of the assistant turn's content.
func helloIDs(hello ...int) []int {
	return cat([]int{imStart}, pUser, []int{'\n', 'h', 'i', imEnd, '\n', imStart}, pAssistant, []int{'\n'}, hello,
		[]int{imEnd, '\n', imStart}, pUser, []int{'\n', 'o', 'k', imEnd, '\n', imStart}, pAssistant, []int{'\n'})
}

func noFallback() ([]int, error) { return nil, errors.New("rawFallback called with a template") }

func TestEncodePrompt_noCheck(t *testing.T) {
	tk, tm := loadTiny(t)
	want := helloIDs(pHello...)
	for _, ck := range []*Check{nil, check(nil), check(map[int]*Carrier{})} {
		ids, out, err := EncodePrompt(tk, tm, "", helloTurns, nil, ck, noFallback)
		if err != nil || !slices.Equal(ids, want) {
			t.Errorf("check %+v: ids %v, %v\nwant %v", ck, ids, err, want)
		}
		if out.Carried != 0 || out.Used != 0 || len(out.Fallbacks) != 0 {
			t.Errorf("check %+v: outcome %+v with no carriers", ck, out)
		}
	}
}

// A carrier whose ids decode exactly to the turn's text is tokenized as those ids, even where the tokenizer's own split
// differs; every byte outside the region is tokenized as without the check.
func TestEncodePrompt_carriedIDsUsed(t *testing.T) {
	tk, tm := loadTiny(t)
	for _, tc := range []struct {
		name  string
		ids   []int
		whole bool
		want  []int
	}{
		{"same split as the tokenizer", pHello, false, helloIDs(pHello...)},
		{"bytewise split of the content", []int{'H', 'e', 'l', 'l', 'o'}, false, helloIDs('H', 'e', 'l', 'l', 'o')},
		{"whole turn ending on <|im_end|>", []int{'H', 'e', 'l', 'l', 'o', imEnd}, true, helloIDs('H', 'e', 'l', 'l', 'o')},
		{"whole turn, content only", []int{'H', 'e', 'l', 'l', 'o'}, true, helloIDs('H', 'e', 'l', 'l', 'o')},
		// A reasoning-only blob covers the start of the turn; the rest of it is tokenized on its own.
		{"leading part of the turn", []int{'H', 'e', 'l'}, false, helloIDs('H', 'e', 'l', 'l', 'o')},
	} {
		ids, out, err := EncodePrompt(tk, tm, "", helloTurns, nil, check(map[int]*Carrier{1: carrier(t, tc.ids, tc.whole)}), noFallback)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		if !slices.Equal(ids, tc.want) {
			t.Errorf("%s: ids %v\nwant %v", tc.name, ids, tc.want)
		}
		if out.Carried != 1 || out.Used != 1 || out.Formatted != 0 || len(out.Fallbacks) != 0 {
			t.Errorf("%s: outcome %+v, want 1 carried and used", tc.name, out)
		}
	}
}

// Every turn that cannot use its ids falls back to the tokenizer's own split, and the reason is recorded.
func TestEncodePrompt_fallbacks(t *testing.T) {
	tk, tm := loadTiny(t)
	other := NewKey([]byte("another server's key, 32 bytes.."))
	foreign, _ := ParseBlob(other.Mint("m", "ns", "", pHello, false))
	badMAC := carrier(t, pHello, false)
	badMAC.Blob.MAC[0] ^= 1
	tampered := carrier(t, pHello, false)
	tampered.Blob.IDs = []int{'H', 'e', 'l', 'l', 'o'} // the MAC covers the ids
	wrongModel, _ := ParseBlob(testKey.Mint("other-model", "ns", "", pHello, false))
	wrongNS, _ := ParseBlob(testKey.Mint("m", "other-ns", "", pHello, false))
	negative := &Blob{Model: "m", NS: "ns", IDs: []int{-1, 'e'}}
	testKey.Seal(negative)

	for _, tc := range []struct {
		name  string
		c     *Carrier
		turns []chat.Turn
		want  string
	}{
		{"parse-time reason", &Carrier{Reason: ReasonV1}, helloTurns, ReasonV1},
		{"another key", &Carrier{Blob: foreign}, helloTurns, ReasonKey},
		{"bad MAC", badMAC, helloTurns, ReasonMAC},
		{"ids changed after signing", tampered, helloTurns, ReasonMAC},
		{"another model", &Carrier{Blob: wrongModel}, helloTurns, ReasonModel},
		{"another namespace", &Carrier{Blob: wrongNS}, helloTurns, ReasonNS},
		{"content edited by the client", carrier(t, []int{'H', 'e', idLL, 'x'}, false), helloTurns, ReasonDecode},
		{"no ids", carrier(t, nil, false), helloTurns, ReasonDecode},
		{"id past the vocabulary", carrier(t, []int{'H', 273}, false), helloTurns, ReasonDecode},
		{"negative id", &Carrier{Blob: negative}, helloTurns, ReasonDecode},
		{"whole-turn blob covering only the start", carrier(t, []int{'H', 'e', 'l'}, true), helloTurns, ReasonDecode},
		// The turn's text holds the literal marker as content; the carried control id would forge a turn boundary.
		{"control id over literal content", carrier(t, []int{'a', imEnd, 'b'}, false),
			[]chat.Turn{{Role: "user", Content: "hi"}, {Role: "assistant", Content: "a<|im_end|>b"}, {Role: "user", Content: "ok"}},
			ReasonSpecial},
		// ChatML renders no reasoning for a history turn, so ids that include it can never match.
		{"reasoning the template drops", carrier(t, cat([]int{'<', 't', 'h', 'i', 'n', 'k', '>', 'w', '<', '/', 't', 'h', 'i', 'n', 'k', '>'}, pHello), true),
			[]chat.Turn{{Role: "user", Content: "hi"}, {Role: "assistant", Content: "Hello", Reasoning: "w"}, {Role: "user", Content: "ok"}},
			ReasonHistory},
	} {
		plain, err := codec.EncodePrompt(tk, tm, "", tc.turns, nil)
		if err != nil {
			t.Fatal(err)
		}
		ids, out, err := EncodePrompt(tk, tm, "", tc.turns, nil, check(map[int]*Carrier{1: tc.c}), noFallback)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		if !slices.Equal(ids, plain) {
			t.Errorf("%s: a fallback turn tokenized differently from no check:\n got %v\nwant %v", tc.name, ids, plain)
		}
		if out.Carried != 1 || out.Used != 0 || !slices.Equal(out.Fallbacks, []string{tc.want}) {
			t.Errorf("%s: outcome %+v, want one fallback %q", tc.name, out, tc.want)
		}
	}
}

// Several carriers are noted in turn order whatever the map's order, each used or not on its own.
func TestEncodePrompt_severalCarriers(t *testing.T) {
	tk, tm := loadTiny(t)
	turns := []chat.Turn{
		{Role: "user", Content: "hi"}, {Role: "assistant", Content: "Hello"},
		{Role: "user", Content: "ok"}, {Role: "assistant", Content: "Hello"},
		{Role: "user", Content: "hi"}, {Role: "assistant", Content: "Hello"},
	}
	wrongNS, _ := ParseBlob(testKey.Mint("m", "other-ns", "", pHello, false))
	bytewise := []int{'H', 'e', 'l', 'l', 'o'}
	cs := map[int]*Carrier{5: carrier(t, bytewise, true), 3: {Blob: wrongNS}, 1: carrier(t, bytewise, true)}
	ids, out, err := EncodePrompt(tk, tm, "", turns, nil, check(cs), noFallback)
	if err != nil {
		t.Fatal(err)
	}
	turn := func(user []int, hello []int) []int {
		return cat([]int{imStart}, pUser, []int{'\n'}, user, []int{imEnd, '\n', imStart}, pAssistant, []int{'\n'}, hello, []int{imEnd, '\n'})
	}
	want := cat(turn([]int{'h', 'i'}, bytewise), turn([]int{'o', 'k'}, pHello), turn([]int{'h', 'i'}, bytewise),
		[]int{imStart}, pAssistant, []int{'\n'})
	if !slices.Equal(ids, want) {
		t.Errorf("ids %v\nwant %v", ids, want)
	}
	if out.Carried != 3 || out.Used != 2 || !slices.Equal(out.Fallbacks, []string{ReasonNS}) {
		t.Errorf("outcome %+v, want 3 carried, 2 used, [ns]", out)
	}
}

// A tool call the reply wrote with its own JSON spacing keeps its ids where the template re-spaces the call object.
func TestEncodePrompt_templateFormatting(t *testing.T) {
	tk, tm := loadTiny(t)
	tools := []chat.Tool{{Name: "f", Parameters: json.RawMessage(`{"type":"object"}`)}}
	turns := []chat.Turn{
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "Hello", ToolCalls: []chat.ToolCall{{Name: "f", Arguments: json.RawMessage(`{"a":1}`)}}},
		{Role: "user", Content: "ok"},
	}
	reply := "Hello\n<tool_call>\n{\"name\":\"f\",\"arguments\":{\"a\":1}}\n</tool_call>"
	replyIDs, err := tk.EncodeSegments([]chat.Segment{{Text: reply}}, false)
	if err != nil {
		t.Fatal(err)
	}
	ids, out, err := EncodePrompt(tk, tm, "", turns, tools, check(map[int]*Carrier{1: carrier(t, replyIDs, false)}), noFallback)
	if err != nil {
		t.Fatal(err)
	}
	if out.Used != 1 || out.Formatted != 1 || len(out.Fallbacks) != 0 {
		t.Fatalf("outcome %+v, want the turn used with formatting", out)
	}
	plain, err := codec.EncodePrompt(tk, tm, "", turns, tools)
	if err != nil {
		t.Fatal(err)
	}
	if slices.Equal(ids, plain) {
		t.Fatal("the template's spacing was tokenized, not the carried ids")
	}
	// The carried ids sit right after the assistant header, and the prompt then goes on as the plain render does after the
	// call block.
	header := cat([]int{imStart}, pAssistant, []int{'\n'})
	at := -1
	for i := range len(plain) - len(header) {
		if slices.Equal(plain[i:i+len(header)], header) {
			at = i + len(header)
			break
		}
	}
	if at < 0 {
		t.Fatalf("no assistant header in %v", plain)
	}
	tail := cat([]int{imEnd, '\n', imStart}, pUser, []int{'\n', 'o', 'k', imEnd, '\n', imStart}, pAssistant, []int{'\n'})
	if want := cat(plain[:at], replyIDs, tail); !slices.Equal(ids, want) {
		t.Errorf("ids %v\nwant %v", ids, want)
	}

	// An argument value the client changed is content, not formatting.
	edited := "Hello\n<tool_call>\n{\"name\":\"f\",\"arguments\":{\"a\":2}}\n</tool_call>"
	editedIDs, _ := tk.EncodeSegments([]chat.Segment{{Text: edited}}, false)
	ids, out, err = EncodePrompt(tk, tm, "", turns, tools, check(map[int]*Carrier{1: carrier(t, editedIDs, false)}), noFallback)
	if err != nil || !slices.Equal(ids, plain) || !slices.Equal(out.Fallbacks, []string{ReasonDecode}) {
		t.Errorf("edited argument: outcome %+v err %v, want a decode fallback with the plain ids", out, err)
	}
}

// With no template the raw fallback tokenizes the prompt, and every carrier is noted with the reason it fails before the
// render, or special when it passes those checks.
func TestEncodePrompt_rawFallback(t *testing.T) {
	tk, _ := loadTiny(t)
	wrongModel, _ := ParseBlob(testKey.Mint("other-model", "ns", "", pHello, false))
	cs := map[int]*Carrier{4: {Blob: wrongModel}, 1: carrier(t, pHello, true), 2: {Reason: ReasonParse}}
	calls := 0
	ids, out, err := EncodePrompt(tk, nil, "", helloTurns, nil, check(cs), func() ([]int, error) {
		calls++
		return []int{1, 2, 3}, nil
	})
	if err != nil || calls != 1 || !slices.Equal(ids, []int{1, 2, 3}) {
		t.Errorf("raw fallback: ids %v err %v after %d calls", ids, err, calls)
	}
	if out.Carried != 3 || out.Used != 0 || !slices.Equal(out.Fallbacks, []string{ReasonSpecial, ReasonParse, ReasonModel}) {
		t.Errorf("outcome %+v, want [special parse model]", out)
	}

	boom := errors.New("boom")
	if _, _, err := EncodePrompt(tk, nil, "", helloTurns, nil, nil, func() ([]int, error) { return nil, boom }); !errors.Is(err, boom) {
		t.Errorf("raw fallback error: %v, want boom", err)
	}
}

func TestReasonBeforeRender(t *testing.T) {
	ck := check(nil)
	good := carrier(t, pHello, false)
	for _, tc := range []struct {
		name string
		c    *Carrier
		want string
	}{
		{"passes", good, "otherwise"},
		{"parse-time reason wins", &Carrier{Blob: good.Blob, Reason: ReasonV1}, ReasonV1},
		{"key id", &Carrier{Blob: &Blob{KeyID: [8]byte{1}, Model: "m", NS: "ns", MAC: good.Blob.MAC}}, ReasonKey},
		{"no MAC", &Carrier{Blob: &Blob{KeyID: testKey.ID, Model: "m", NS: "ns", IDs: pHello}}, ReasonMAC},
		{"short MAC", &Carrier{Blob: &Blob{KeyID: testKey.ID, Model: "m", NS: "ns", IDs: pHello, MAC: good.Blob.MAC[:31]}}, ReasonMAC},
		// The model and namespace are signed, so changing one breaks the MAC first.
		{"model edited after signing", &Carrier{Blob: &Blob{KeyID: testKey.ID, Model: "x", NS: "ns", IDs: pHello, MAC: good.Blob.MAC}}, ReasonMAC},
	} {
		if got := ck.ReasonBeforeRender(tc.c, "otherwise"); got != tc.want {
			t.Errorf("%s: %q, want %q", tc.name, got, tc.want)
		}
	}
	// A blob signed for another model or namespace verifies but does not match the check.
	for _, tc := range []struct{ model, ns, want string }{{"x", "ns", ReasonModel}, {"m", "x", ReasonNS}} {
		b, _ := ParseBlob(testKey.Mint(tc.model, tc.ns, "", pHello, false))
		if got := ck.ReasonBeforeRender(&Carrier{Blob: b}, ""); got != tc.want {
			t.Errorf("minted for %s/%s: %q, want %q", tc.model, tc.ns, got, tc.want)
		}
	}
}

func TestIdentity(t *testing.T) {
	tk, _ := loadTiny(t)
	const fp = "model.gguf|1234|1700000000|q4_k_m"
	base := Identity(tk, fp)
	if len(base) != 32 {
		t.Fatalf("identity %q: %d hex chars, want 32", base, len(base))
	}
	if Identity(tk, fp) != base {
		t.Error("identity is not deterministic")
	}
	if Identity(tk, "model.gguf|1234|1800000000|q4_k_m") != base {
		t.Error("a copy of the same file with another mtime has another identity")
	}
	for _, tc := range []struct{ name, fp string }{
		{"another size", "model.gguf|1235|1700000000|q4_k_m"},
		{"another quant", "model.gguf|1234|1700000000|q8_0"},
		{"another name", "other.gguf|1234|1700000000|q4_k_m"},
		{"an adapter", fp + "+adapter:lora-a"},
		{"no fingerprint", ""},
	} {
		if Identity(tk, tc.fp) == base {
			t.Errorf("%s: same identity as the base", tc.name)
		}
	}
	if Identity(tk, fp+"+adapter:lora-a") == Identity(tk, fp+"+adapter:lora-b") {
		t.Error("two adapters share an identity")
	}
	if Identity(tk, "model.gguf|1234|1700000000|q4_k_m+adapter:a") != Identity(tk, "model.gguf|1234|1|q4_k_m+adapter:a") {
		t.Error("with an adapter, the mtime still changes the identity")
	}
	if Identity(nil, fp) == base {
		t.Error("the tokenizer's vocabulary does not enter the identity")
	}
	if id1, id2 := Identity(nil, fp), Identity(nil, fp); id1 != id2 || len(Identity(nil, "")) != 32 {
		t.Error("identity without a tokenizer is not a stable 32-hex-char value")
	}
}
