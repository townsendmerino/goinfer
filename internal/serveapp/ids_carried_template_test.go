package serveapp

import (
	"compress/gzip"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/internal/modelload"
)

// G-ids-3's three `decode` turns, replayed through the /v1/responses renderer with Mellum2.1's tokenizer and template and no
// weights (ids-carried-2026-10-15.md, "G-ids-3: graded"). Each reply's own bytes differ from the template's render of its turn
// only in template-owned formatting (idsFormat): a newline before `</think>`, or a space after `"arguments":`. Carried and
// untouched, the next prompt must now start with the slot's id record (the previous prompt, then every generated id) with
// ids_fmt=1. Without the carrier it must not: the template's rewrite leaves the slot inside the reply.
//
// Planted defects on the reply's turn must still fall back as `decode` and tokenize the turn as a v1 blob of the same text
// would: a one-character content edit in the call's arguments and in the reasoning, whitespace added inside an argument string
// and inside the reasoning, and a space (not a newline) before `</think>`. A client re-spacing the call's JSON outside its
// strings is on the closed list and keeps the ids.
//
// Fixture: testdata/ids-template/cases.json.gz, built by
// docs/measurements/integration-harness-2026-10/ids-carried-tier2/build_template_fixture.py from the night's bodies.
func TestIDsCarried_templateFormat(t *testing.T) {
	tkDir := filepath.Join("..", "..", "testdata", "mellum2-tokenizer")
	if _, err := os.Stat(filepath.Join(tkDir, "tokenizer.json")); err != nil {
		t.Skipf("needs the Mellum2 tokenizer at %s (a per-machine symlink): %v", tkDir, err)
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
	// serve's flag defaults, which the night ran with (-thinking template, -tool-format auto): a zero config renders another
	// prompt.
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	sf := registerFlags(fs)
	if err := fs.Parse(nil); err != nil {
		t.Fatal(err)
	}
	s := &server{cfg: sf.cfg}
	key := newIDsKey([]byte("G-ids-3 template-format test key"))
	s.idsKey = key
	lm := &loadedModel{tk: tk, name: "mellum2.1-thinking"}
	lm.tmpl = s.cfg.tuneTemplate(tm)
	model := lm.idsIdentity()
	render := func(body tmplBody, useIDs bool) ([]int, idsOutcome) {
		t.Helper()
		ids, o, err := kv0RenderIDs(s, lm, "/v1/responses", body.bytes(t), useIDs)
		if err != nil {
			t.Fatal(err)
		}
		return ids, o
	}

	f, err := os.Open(filepath.Join("..", "..", "testdata", "ids-template", "cases.json.gz"))
	if err != nil {
		t.Fatal(err)
	}
	zr, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		ID           string `json:"id"`
		Prev         string `json:"prev"`
		Next         string `json:"next"`
		NPrev        int    `json:"n_prev"`
		NItems       int    `json:"n_items"`
		Gen          []int  `json:"gen"`
		PromptLen    int    `json:"prompt_len"`
		PromptSHA256 string `json:"prompt_sha256"`
	}
	if err := json.NewDecoder(zr).Decode(&cases); err != nil {
		t.Fatal(err)
	}
	f.Close()
	if len(cases) != 3 {
		t.Fatalf("fixture holds %d cases, want 3", len(cases))
	}

	// remint re-signs every v2 blob in a body with the test key, unchanged otherwise (the night's key was random per server).
	remint := func(body tmplBody) {
		for _, it := range body.input {
			m := it.(map[string]any)
			if enc, _ := m["encrypted_content"].(string); strings.HasPrefix(enc, reasoningBlobV2Prefix) {
				b, err := parseIDsBlob(enc)
				if err != nil {
					t.Fatal(err)
				}
				m["encrypted_content"] = key.mint(model, b.ns, b.text, b.ids, b.whole)
			}
		}
	}

	for _, c := range cases {
		prev, next := parseTmplBody(t, c.Prev), parseTmplBody(t, c.Next)
		remint(prev)
		remint(next)
		P0, _ := render(prev, true)
		if sum := sha256.Sum256(idsBytes(P0)); len(P0) != c.PromptLen || hex.EncodeToString(sum[:]) != c.PromptSHA256 {
			t.Fatalf("VOID %s: the previous request renders %d ids, not the night's prompt (%d ids, sha256 %s)", c.ID, len(P0), c.PromptLen, c.PromptSHA256)
		}
		slot := append(slices.Clone(P0), c.Gen...)

		rsAt, callAt := -1, -1
		input := next.input
		for i := c.NPrev; i < c.NPrev+c.NItems; i++ {
			switch input[i].(map[string]any)["type"] {
			case "reasoning":
				rsAt = i
			case "function_call":
				callAt = i
			}
		}
		if rsAt < 0 || callAt < 0 {
			t.Fatalf("%s: the reply has no reasoning item or no call", c.ID)
		}
		blob, err := parseIDsBlob(input[rsAt].(map[string]any)["encrypted_content"].(string))
		if err != nil || !slices.Equal(blob.ids, c.Gen) {
			t.Fatalf("%s: the carried ids are not the logged generation (%v)", c.ID, err)
		}

		// Carried and untouched: the ids replace the template's formatting and the slot is kept.
		ids, out := render(next, true)
		want := fmt.Sprintf("ids=%d/%d ids_fallback=- ids_fmt=1", out.carried, out.carried)
		if out.fields() != want {
			t.Errorf("%s carried: %s, want %s", c.ID, out.fields(), want)
		}
		if len(ids) < len(slot) || !slices.Equal(ids[:len(slot)], slot) {
			t.Errorf("%s carried: the next prompt leaves the slot at %d of %d", c.ID, idsLCP(ids, slot), len(slot))
		}

		// Without a carrier for this turn the template's rewrite breaks reuse inside the reply: the no-carrier path has the
		// same loss (the off arm's cases).
		v1 := func(text string) string { return reasoningBlobPrefix + base64.StdEncoding.EncodeToString([]byte(text)) }
		noCarrier := next.clone(t)
		noCarrier.item(rsAt)["encrypted_content"] = v1(blob.text)
		today, _ := render(noCarrier, true)
		if at := idsLCP(today, slot); at < len(P0) || at >= len(slot) {
			t.Errorf("%s without a carrier: the prompt leaves the slot at %d; want inside the reply [%d, %d)", c.ID, at, len(P0), len(slot))
		} else {
			t.Logf("%s without a carrier: the prompt leaves the slot at %d, %d ids before its end", c.ID, at, len(slot)-at)
		}

		// Planted defects. Each is applied to a copy of the next request; reasoning edits go in the blob's text, which the
		// render uses over the visible content.
		args := input[callAt].(map[string]any)["arguments"].(string)
		vi := strings.Index(args, `"cmd":"`) + len(`"cmd":"`)
		if vi < len(`"cmd":"`) || vi+2 > len(args) {
			t.Fatalf("%s: no cmd string in %s", c.ID, args)
		}
		mid := strings.Index(blob.text[len(blob.text)/2:], " ") + len(blob.text)/2
		type defect struct {
			name, reason string
			args, text   string // the call's arguments and the blob's text after the edit
		}
		ds := []defect{
			{"args one character", idsDecode, flipChar(args), blob.text},
			{"reasoning one character", idsDecode, args, flipChar(blob.text)},
			{"space inside an argument string", idsDecode, args[:vi+1] + " " + args[vi+1:], blob.text},
			{"space inside the reasoning", idsDecode, args, blob.text[:mid] + " " + blob.text[mid:]},
			{"space before the closing delimiter", idsDecode, args, blob.text + " "},
			{"the call's JSON re-spaced outside strings", "", strings.Replace(args, `"cmd":"`, `"cmd": "`, 1), blob.text},
		}
		for _, d := range ds {
			body := next.clone(t)
			body.item(callAt)["arguments"] = d.args
			body.item(rsAt)["encrypted_content"] = key.mint(model, "", d.text, blob.ids, blob.whole)
			got, out := render(body, true)
			plain := body.clone(t)
			plain.item(rsAt)["encrypted_content"] = v1(d.text)
			ref, _ := render(plain, true)
			if slices.Equal(ref, today) {
				t.Fatalf("VOID %s (%s): the edit left the rendered prompt unchanged", c.ID, d.name)
			}
			if d.reason == "" {
				want := fmt.Sprintf("ids=%d/%d ids_fallback=- ids_fmt=1", out.carried, out.carried)
				if out.fields() != want || len(got) < len(slot) || !slices.Equal(got[:len(slot)], slot) {
					t.Errorf("%s (%s): %s, slot kept to %d of %d; want %s and the whole slot", c.ID, d.name, out.fields(), idsLCP(got, slot), len(slot), want)
				}
				continue
			}
			if want := fmt.Sprintf("ids=%d/%d ids_fallback=%s", out.carried-1, out.carried, d.reason); out.fields() != want {
				t.Errorf("%s (%s): %s, want %s", c.ID, d.name, out.fields(), want)
			}
			if !slices.Equal(got, ref) {
				t.Errorf("%s (%s): the fallback is not the v1 tokenization of the same text (first difference at %d)", c.ID, d.name, idsLCP(got, ref))
			}
		}
	}
}

// tmplBody is a request body whose fields other than input are kept as the bytes the client sent: serve renders a tool's
// parameters verbatim, so a re-serialized body renders another prompt.
type tmplBody struct {
	raw   map[string]json.RawMessage
	input []any
}

func parseTmplBody(t *testing.T, s string) tmplBody {
	t.Helper()
	var b tmplBody
	if err := json.Unmarshal([]byte(s), &b.raw); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b.raw["input"], &b.input); err != nil {
		t.Fatal(err)
	}
	return b
}

func (b tmplBody) item(i int) map[string]any { return b.input[i].(map[string]any) }

// bytes writes every field but input verbatim (json.Marshal would compact and HTML-escape a RawMessage).
func (b tmplBody) bytes(t *testing.T) []byte {
	t.Helper()
	in, err := json.Marshal(b.input)
	if err != nil {
		t.Fatal(err)
	}
	out := []byte(`{"input":`)
	out = append(out, in...)
	for k, v := range b.raw {
		if k == "input" {
			continue
		}
		kb, _ := json.Marshal(k)
		out = append(append(append(append(out, ','), kb...), ':'), v...)
	}
	return append(out, '}')
}

func (b tmplBody) clone(t *testing.T) tmplBody {
	t.Helper()
	var in []any
	enc, _ := json.Marshal(b.input)
	if err := json.Unmarshal(enc, &in); err != nil {
		t.Fatal(err)
	}
	return tmplBody{raw: b.raw, input: in}
}
