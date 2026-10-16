package serveapp

import (
	"bufio"
	"compress/gzip"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/internal/modelload"
)

// G-ids-0 (docs/measurements/integration-harness-2026-10/id-roundtrip-step0-2026-10-15.md, §(c) and its Amendment 1): the
// 45 replies of the step-0 replay, rendered through the /v1/responses renderer with Mellum2.1's tokenizer and template and no
// weights. For each reply the next request carries the reply's items as emitted with a v2 blob signed by the test key, and
// then each planted defect (a) to (m), and an edit to the visible reasoning only. A carried, untouched blob must make the next
// prompt start with the slot's id record (the prompt, then every generated id) and count ids=1/1; every defect must give
// today's tokenization exactly, its named reason, and no error.
//
// "Today's tokenization" is the same request rendered with no ids used: the reasoning text the item carries (a parseable v2
// blob's text, as a v1 blob's is read today; a blob that does not parse is ignored and the visible content is read), re-tokenized.
//
// Fixture: testdata/ids-step0/replies.jsonl.gz, built from step 0's requests (prep.py) and its replay (run.jsonl.gz).
// Tokenizer: testdata/mellum2-tokenizer (Mellum2 and Mellum2.1 share tokenizer.json byte for byte); the test skips without it.
func TestIDsCarried_G0(t *testing.T) {
	tkDir := filepath.Join("..", "..", "testdata", "mellum2-tokenizer")
	if _, err := os.Stat(filepath.Join(tkDir, "tokenizer.json")); err != nil {
		t.Skipf("G-ids-0 needs the Mellum2 tokenizer at %s (a per-machine symlink): %v", tkDir, err)
	}
	tk, err := modelload.Tokenizer(tkDir)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "chat_think_goldens", "mellum21_think_history.json"))
	if err != nil {
		t.Fatal(err)
	}
	var gs []struct {
		ChatTemplate string `json:"chat_template"`
	}
	if err := json.Unmarshal(raw, &gs); err != nil || len(gs) != 1 {
		t.Fatalf("golden: %v", err)
	}
	tm, err := chat.Detect(chat.Meta{ChatTemplate: gs[0].ChatTemplate, HasToken: tk.Has})
	if err != nil {
		t.Fatal(err)
	}
	s := &server{}
	key := newIDsKey([]byte("G-ids-0 test key, 32 bytes long."))
	s.idsKey = key
	lm := &loadedModel{tk: tk, name: "mellum2.1-thinking"}
	lm.tmpl = s.cfg.tuneTemplate(tm)
	model := lm.idsIdentity()

	fx := loadIDsStep0(t)
	if len(fx.replies) != 45 {
		t.Fatalf("fixture holds %d replies, want 45", len(fx.replies))
	}
	render := func(body map[string]any, useIDs bool) ([]int, idsOutcome, error) {
		b, _ := json.Marshal(body)
		return kv0RenderIDs(s, lm, "/v1/responses", b, useIDs)
	}
	other := newIDsKey([]byte("a different key, also 32 bytes.."))
	cases := map[string]int{}
	held := 0
	for _, rp := range fx.replies {
		body := fx.body(rp.Session, rp.NInput)
		P, hist, err := render(body, true)
		if err != nil {
			t.Fatalf("%s: render: %v", rp.ID, err)
		}
		// The archived history carries the night's v1 blobs: each is a carried turn that falls back as v1 (G-ids-1's reading).
		h := hist.carried
		if hist.used != 0 || len(hist.fallbacks) != h || strings.Trim(strings.Repeat(idsV1+",", h), ",") != strings.Join(hist.fallbacks, ",") {
			t.Fatalf("%s: the request's own history counts %s; every archived blob is a v1", rp.ID, hist.fields())
		}
		want := func(reason string) string { // the next request's fields: h v1 fallbacks, then the reply's turn
			var o idsOutcome
			for range h {
				o.note(idsV1)
			}
			if reason != "none" {
				o.note(reason)
			}
			return o.fields()
		}
		if len(P) != rp.P { // void: the fixture's prompt is not the one the replay's slot started with
			t.Fatalf("VOID %s: the route renders %d prompt ids, the replay's slot started with %d", rp.ID, len(P), rp.P)
		}
		slot := append(append([]int(nil), P...), rp.Gen...)
		rsText, rsAt, callAt, msgAt := "", -1, -1, -1
		for i, it := range rp.Items {
			switch it["type"] {
			case "reasoning":
				rsAt = i
				for _, c := range it["content"].([]any) {
					rsText += c.(map[string]any)["text"].(string)
				}
			case "function_call":
				if callAt < 0 {
					callAt = i
				}
			case "message":
				msgAt = i
			}
		}
		if rsAt < 0 {
			t.Fatalf("%s: no reasoning item; the fixture's replies all reasoned", rp.ID)
		}
		valid := key.mint(model, "", rsText, rp.Gen, true)
		// next builds the next request from the emitted items (less status), one of them changed by edit, then follow.
		next := func(edit func(items []map[string]any) []map[string]any) map[string]any {
			items := make([]map[string]any, 0, len(rp.Items))
			for _, it := range rp.Items {
				e := map[string]any{}
				for k, v := range it {
					if k != "status" {
						e[k] = v
					}
				}
				if e["type"] == "reasoning" {
					e["encrypted_content"] = valid
				}
				items = append(items, e)
			}
			if edit != nil {
				items = edit(items)
			}
			nb := fx.body(rp.Session, rp.NInput)
			input := append([]any(nil), nb["input"].([]any)...)
			for _, it := range items {
				input = append(input, it)
			}
			input = append(input, rp.Follow...)
			nb["input"] = input
			return nb
		}
		setBlob := func(enc string) func([]map[string]any) []map[string]any {
			return func(items []map[string]any) []map[string]any {
				items[rsAt]["encrypted_content"] = enc
				return items
			}
		}
		// 1. Carried and untouched.
		base := next(nil)
		ids, out, err := render(base, true)
		if err != nil {
			t.Fatalf("%s: carried: %v", rp.ID, err)
		}
		if out.fields() != want("") {
			t.Errorf("%s carried: %s, want %s (the reply's turn used, v1 only on archived history)", rp.ID, out.fields(), want(""))
		}
		if len(ids) < len(slot) || !slices.Equal(ids[:len(slot)], slot) {
			t.Errorf("%s carried: the next prompt does not start with the slot's id record (first difference at %d of %d)", rp.ID, idsLCP(ids, slot), len(slot))
		} else {
			held++
		}
		cases["carried"]++
		todayBase, _, err := render(base, false)
		if err != nil {
			t.Fatal(err)
		}
		// 3. Edits to the visible reasoning only: the blob's text renders, and its ids are used.
		vis := next(func(items []map[string]any) []map[string]any {
			items[rsAt]["content"] = []any{map[string]any{"type": "reasoning_text", "text": "edited " + rsText}}
			return items
		})
		if ids, out, err := render(vis, true); err != nil || out.fields() != want("") || len(ids) < len(slot) || !slices.Equal(ids[:len(slot)], slot) {
			t.Errorf("%s visible edit: %s %v; the blob's text and ids must win over the visible content", rp.ID, out.fields(), err)
		}
		cases["visible-edit"]++

		// 2. Planted defects: each gives today's tokenization, its reason and no error.
		type defect struct {
			name, reason string
			body         map[string]any
			textEdit     bool // the defect edits rendered text, so a render equal to the untouched one means it did not plant
		}
		var ds []defect
		ds = append(ds, defect{"a", idsDecode, next(setBlob(key.mint(model, "", flipChar(rsText), rp.Gen, true))), true})
		if callAt >= 0 {
			args := rp.Items[callAt]["arguments"].(string)
			ds = append(ds, defect{"b", idsDecode, next(func(items []map[string]any) []map[string]any {
				items[callAt]["arguments"] = flipChar(args)
				return items
			}), true})
			if strings.Contains(args, `\\t`) {
				ds = append(ds, defect{"c", idsDecode, next(func(items []map[string]any) []map[string]any {
					items[callAt]["arguments"] = strings.Replace(args, `\\t`, `\t`, 1)
					return items
				}), true})
			}
		}
		if msgAt >= 0 {
			ds = append(ds, defect{"d", idsDecode, next(func(items []map[string]any) []map[string]any {
				c := items[msgAt]["content"].([]any)[0].(map[string]any)
				cc := map[string]any{}
				for k, v := range c {
					cc[k] = v
				}
				cc["text"] = flipChar(c["text"].(string))
				items[msgAt]["content"] = []any{cc}
				return items
			}), true})
		}
		forged := nonCanonicalSplit(t, lm, rp.Gen)
		noMAC := &idsBlob{whole: true, keyID: key.id, model: model, text: rsText, ids: forged}
		ds = append(ds, defect{"e", idsMAC, next(setBlob(noMAC.String())), false})
		randMAC := &idsBlob{whole: true, keyID: key.id, model: model, text: rsText, ids: forged, mac: make([]byte, 32)}
		_, _ = rand.Read(randMAC.mac)
		ds = append(ds, defect{"f", idsMAC, next(setBlob(randMAC.String())), false})
		ds = append(ds, defect{"g", idsKeyR, next(setBlob(other.mint(model, "", rsText, rp.Gen, true))), false})
		hText, hIDs := plantSpecialInReasoning(t, lm, rsText, rp.Gen)
		ds = append(ds, defect{"h", idsSpecial, next(setBlob(key.mint(model, "", hText, hIDs, true))), false})
		ds = append(ds, defect{"i", idsNS, next(setBlob(key.mint(model, "another-namespace", rsText, rp.Gen, true))), false})
		ds = append(ds, defect{"j", idsModel, next(setBlob(key.mint("another-model", "", rsText, rp.Gen, true))), false})
		ds = append(ds, defect{"k", idsV1, next(setBlob(reasoningBlobPrefix + base64.StdEncoding.EncodeToString([]byte(rsText)))), false})
		ds = append(ds, defect{"l", idsParse, next(setBlob(valid[:len(reasoningBlobV2Prefix)+(len(valid)-len(reasoningBlobV2Prefix))/2] + "!!")), false})
		ds = append(ds, defect{"m", "", next(func(items []map[string]any) []map[string]any {
			return append(items[:rsAt:rsAt], items[rsAt+1:]...)
		}), false})
		for _, d := range ds {
			got, out, err := render(d.body, true)
			if err != nil {
				t.Errorf("%s (%s): error %v; a defect must fall back, not fail", rp.ID, d.name, err)
				continue
			}
			today, _, err := render(d.body, false)
			if err != nil {
				t.Fatal(err)
			}
			if d.textEdit && slices.Equal(today, todayBase) {
				t.Fatalf("VOID %s (%s): the planted edit left the rendered prompt unchanged", rp.ID, d.name)
			}
			if !slices.Equal(got, today) {
				t.Errorf("%s (%s): the fallback is not today's tokenization (first difference at %d)", rp.ID, d.name, idsLCP(got, today))
			}
			wantFields := want(d.reason)
			if d.name == "m" {
				wantFields = want("none")
				probe := rsText[:min(len(rsText), 40)]
				if strings.Count(renderText(lm, got), probe) != strings.Count(renderText(lm, todayBase), probe)-1 {
					t.Errorf("%s (m): the dropped reasoning item's text is rendered", rp.ID)
				}
			}
			if out.fields() != wantFields {
				t.Errorf("%s (%s): %s, want %s", rp.ID, d.name, out.fields(), wantFields)
			}
			cases[d.name]++
		}
	}
	t.Logf("G-ids-0: carried and untouched, next prompt starts with the slot: %d of %d; cases run: %v", held, len(fx.replies), cases)
	if held != len(fx.replies) {
		t.Errorf("carried and untouched held in %d of %d, want all", held, len(fx.replies))
	}
	// Every case once: 45 carried, 45 visible edits, (a) (e)-(m) on all 45; (b) on the 44 with a call; (c) where an escape is; (d) on the one message.
	for _, c := range []string{"carried", "visible-edit", "a", "e", "f", "g", "h", "i", "j", "k", "l", "m"} {
		if cases[c] != 45 {
			t.Errorf("case %s ran %d times, want 45", c, cases[c])
		}
	}
	if cases["b"] != 44 || cases["d"] != 1 || cases["c"] == 0 {
		t.Errorf("cases b/c/d ran %d/%d/%d, want 44/at least 1/1", cases["b"], cases["c"], cases["d"])
	}
}

// idsStep0 is the G-ids-0 fixture.
type idsStep0 struct {
	template map[string]any
	sessions map[string][]any
	replies  []idsStep0Reply
}

type idsStep0Reply struct {
	ID      string           `json:"id"`
	Session string           `json:"session"`
	NInput  int              `json:"n_input"`
	Items   []map[string]any `json:"items"`
	Follow  []any            `json:"follow"`
	Gen     []int            `json:"gen"`
	P       int              `json:"P"`
}

func loadIDsStep0(t *testing.T) *idsStep0 {
	t.Helper()
	f, err := os.Open(filepath.Join("..", "..", "testdata", "ids-step0", "replies.jsonl.gz"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	zr, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	fx := &idsStep0{sessions: map[string][]any{}}
	sc := bufio.NewScanner(zr)
	sc.Buffer(make([]byte, 1<<20), 1<<26)
	for sc.Scan() {
		var line map[string]json.RawMessage
		if err := json.Unmarshal(sc.Bytes(), &line); err != nil {
			t.Fatal(err)
		}
		switch {
		case line["template"] != nil:
			_ = json.Unmarshal(line["template"], &fx.template)
		case line["input"] != nil:
			var s string
			var in []any
			_ = json.Unmarshal(line["session"], &s)
			_ = json.Unmarshal(line["input"], &in)
			fx.sessions[s] = in
		default:
			var r idsStep0Reply
			if err := json.Unmarshal(sc.Bytes(), &r); err != nil {
				t.Fatal(err)
			}
			fx.replies = append(fx.replies, r)
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	return fx
}

// body is the request a session sent with its first n input items, as Codex sent it.
func (fx *idsStep0) body(session string, n int) map[string]any {
	b := map[string]any{}
	for k, v := range fx.template {
		b[k] = v
	}
	b["input"] = append([]any(nil), fx.sessions[session][:n]...)
	return b
}

func idsLCP(a, b []int) int {
	n := 0
	for n < len(a) && n < len(b) && a[n] == b[n] {
		n++
	}
	return n
}

// flipChar changes one character in the middle of s (an ASCII letter or digit, so the JSON it may sit in stays valid).
func flipChar(s string) string {
	b := []byte(s)
	for off := 0; off < len(b); off++ {
		for _, i := range []int{len(b)/2 + off, len(b)/2 - off} {
			if i < 0 || i >= len(b) {
				continue
			}
			c := b[i]
			if c >= 'a' && c <= 'y' || c >= 'A' && c <= 'Y' || c >= '0' && c <= '8' {
				b[i] = c + 1
				return string(b)
			}
		}
	}
	return s + "x"
}

// nonCanonicalSplit is a client's own tokenization of the text gen decodes to: every run between special ids re-encoded one
// character at a time, which decodes to the same text and is not the split the model generated.
func nonCanonicalSplit(t *testing.T, lm *loadedModel, gen []int) []int {
	t.Helper()
	var out []int
	for i := 0; i < len(gen); {
		if lm.tk.IsAdded(gen[i]) {
			out = append(out, gen[i])
			i++
			continue
		}
		j := i
		for j < len(gen) && !lm.tk.IsAdded(gen[j]) {
			j++
		}
		text, err := lm.tk.DecodeContinuation(gen[i:j])
		if err != nil {
			t.Fatal(err)
		}
		for _, r := range text {
			ids, err := lm.tk.EncodeSegments([]chat.Segment{{Text: string(r)}}, false)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, ids...)
		}
		i = j
	}
	if slices.Equal(out, gen) {
		t.Fatal("the planted client split equals the generated ids")
	}
	return out
}

// plantSpecialInReasoning is defect (h): a special id inside the reasoning run, with the blob's text carrying the marker's
// surface at the same place, so the ids decode to the blob's text exactly while the template renders that text as content.
func plantSpecialInReasoning(t *testing.T, lm *loadedModel, text string, gen []int) (string, []int) {
	t.Helper()
	call, ok := lm.tk.TokenID("<tool_call>")
	think, ok2 := lm.tk.TokenID("<think>")
	end, ok3 := lm.tk.TokenID("</think>")
	if !ok || !ok2 || !ok3 || len(gen) == 0 || gen[0] != think {
		t.Fatal("defect (h) needs <think> first and the <tool_call> and </think> tokens")
	}
	e := slices.Index(gen, end)
	if e < 0 {
		t.Fatal("defect (h): no </think> in the generated ids")
	}
	// Generated: <think> "\n" + text + "\n" </think> ...; planted: <think> enc("\n"+text) <tool_call> enc("\n") </think> ...
	before, err := lm.tk.EncodeSegments([]chat.Segment{{Text: "\n" + text}}, false)
	if err != nil {
		t.Fatal(err)
	}
	nl, _ := lm.tk.EncodeSegments([]chat.Segment{{Text: "\n"}}, false)
	ids := append([]int{think}, before...)
	ids = append(ids, call)
	ids = append(ids, nl...)
	ids = append(ids, gen[e:]...)
	return text + "<tool_call>", ids
}

// renderText is ids decoded, for a containment check.
func renderText(lm *loadedModel, ids []int) string {
	s, _ := lm.tk.Decode(ids)
	if !utf8.ValidString(s) {
		return ""
	}
	return s
}
