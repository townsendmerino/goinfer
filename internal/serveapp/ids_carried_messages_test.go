package serveapp

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/internal/loadflags"
	"github.com/townsendmerino/goinfer/internal/modelload"
)

// Carried ids on /v1/messages: the same blob, signing and checks as /v1/responses, carried in a thinking block's signature, or
// for a reply that asked for thinking and had none in a redacted_thinking block's data. These are the Responses tier-0 gates
// on this route: the KV-independence and fallback arms on the tiny fixture, the carriers streamed and not, and G-ids-0's 45
// replies with their planted defects.

// thinkingReq is the thinking field a Messages client sends to ask for thinking.
var thinkingReq = map[string]any{"type": "enabled", "budget_tokens": 1024}

// messagesContent is a /v1/messages reply's content blocks: the body's, or a stream's assembled from its events.
func messagesContent(t *testing.T, evs []map[string]any, stream bool) []map[string]any {
	t.Helper()
	asMap := func(v any) map[string]any { m, _ := v.(map[string]any); return m }
	if !stream {
		var out []map[string]any
		for _, b := range evs[0]["content"].([]any) {
			out = append(out, asMap(b))
		}
		return out
	}
	var out []map[string]any
	for _, e := range evs {
		switch e["type"] {
		case "content_block_start":
			if i := int(e["index"].(float64)); i != len(out) {
				t.Fatalf("block %d opened at position %d", i, len(out))
			}
			b := map[string]any{}
			for k, v := range asMap(e["content_block"]) {
				b[k] = v
			}
			out = append(out, b)
		case "content_block_delta":
			b, d := out[int(e["index"].(float64))], asMap(e["delta"])
			switch d["type"] {
			case "thinking_delta":
				b["thinking"] = fmtString(b["thinking"]) + fmtString(d["thinking"])
			case "signature_delta":
				b["signature"] = d["signature"]
			case "text_delta":
				b["text"] = fmtString(b["text"]) + fmtString(d["text"])
			case "input_json_delta":
				b["partial_json"] = fmtString(b["partial_json"]) + fmtString(d["partial_json"])
			}
		}
	}
	return out
}

// TestIDsCarriedMessages_carriers: a reply that reasons carries the whole turn's ids in its thinking block's signature when the
// block closes after the generation (not streamed, or streamed straight into a tool call), and the ids up to the end of the
// reasoning when a streamed answer opens first. A reply that does not reason carries them in a redacted_thinking block, first,
// only when the request asked for thinking. Echoed in the loop the ids are used; after a new query the turn counts as
// history. On the scripted Mellum2.1 server.
func TestIDsCarriedMessages_carriers(t *testing.T) {
	const call = "<think>\nI will read calc.go.\n</think>\n\n<tool_call>\n{\"name\": \"exec_command\", \"arguments\": {\"cmd\": \"cat calc.go\"}}\n</tool_call>"
	ts, callIDs, _ := scriptedIDsServer(t, call)
	for _, stream := range []bool{false, true} {
		req := requestFor("/v1/messages", true, stream)
		req["thinking"] = thinkingReq
		c := messagesContent(t, postScripted(t, ts.URL+"/v1/messages", req), stream)
		if c[0]["type"] != "thinking" {
			t.Fatalf("stream=%v: first block %v", stream, c[0])
		}
		b, err := parseIDsBlob(fmtString(c[0]["signature"]))
		if err != nil || !b.whole || !slices.Equal(b.ids, callIDs) || b.text != "I will read calc.go." || c[0]["thinking"] != b.text {
			t.Errorf("stream=%v: thinking block %v, blob %+v (%v); want the whole turn's ids", stream, c[0], b, err)
		}
	}

	const plain = "<tool_call>\n{\"name\": \"exec_command\", \"arguments\": {\"cmd\": \"ls\"}}\n</tool_call>"
	ts2, plainIDs, _ := scriptedIDsServer(t, plain)
	for _, stream := range []bool{false, true} {
		req := requestFor("/v1/messages", true, stream)
		c := messagesContent(t, postScripted(t, ts2.URL+"/v1/messages", req), stream)
		if c[0]["type"] != "tool_use" {
			t.Errorf("stream=%v: a carrier without a thinking request: %v", stream, c[0])
		}
		req["thinking"] = thinkingReq
		c = messagesContent(t, postScripted(t, ts2.URL+"/v1/messages", req), stream)
		b, err := parseIDsBlob(fmtString(c[0]["data"]))
		if c[0]["type"] != "redacted_thinking" || err != nil || !slices.Equal(b.ids, plainIDs) || !b.whole || b.text != "" || c[1]["type"] != "tool_use" {
			t.Errorf("stream=%v: blocks %v (%v); want a redacted_thinking carrier with the turn's ids, then the call", stream, c, err)
		}
	}

	const answer = "<think>\nAdd returns a + b.\n</think>\n\n5"
	ts3, ids, s := scriptedIDsServer(t, answer)
	lm := s.models["m"]
	end, _ := lm.tk.TokenID("</think>")
	cut := slices.Index(ids, end) + 1
	for _, stream := range []bool{false, true} {
		req := requestFor("/v1/messages", false, stream)
		req["thinking"] = thinkingReq
		c := messagesContent(t, postScripted(t, ts3.URL+"/v1/messages", req), stream)
		b, err := parseIDsBlob(fmtString(c[0]["signature"]))
		if err != nil {
			t.Fatalf("stream=%v: %v", stream, err)
		}
		if stream && (b.whole || !slices.Equal(b.ids, ids[:cut])) || !stream && (!b.whole || !slices.Equal(b.ids, ids)) {
			t.Errorf("stream=%v: blob whole=%v ids %v; generated %v", stream, b.whole, b.ids, ids)
		}
		for _, tc := range []struct {
			name, want string
			tail       []any
		}{
			{"in the loop", "ids=1/1 ids_fallback=-", nil},
			{"after a new query", "ids=0/1 ids_fallback=history", []any{map[string]any{"role": "user", "content": "And 3 + 4?"}}},
		} {
			msgs := append([]any{req["messages"].([]any)[0], map[string]any{"role": "assistant", "content": []any{c[0], c[1]}}}, tc.tail...)
			body, _ := json.Marshal(map[string]any{"model": "m", "max_tokens": 64, "thinking": thinkingReq, "messages": msgs})
			_, o, err := kv0RenderIDs(s, lm, "/v1/messages", body, true)
			if err != nil || o.fields() != tc.want {
				t.Errorf("stream=%v, %s: %s (%v), want %s", stream, tc.name, o.fields(), err, tc.want)
			}
		}
	}
}

// G-ids-2's fallback arms and KV-independence check on /v1/messages, as TestIDsCarried_tinyFallbackAndKV runs them on
// /v1/responses: the tiny committed fixture, one KV session, the real handler. The tiny model does not reason, so turn 1's ids
// ride in a redacted_thinking carrier (the request asks for thinking).
func TestIDsCarriedMessages_tinyFallbackAndKV(t *testing.T) {
	logPath := filepath.Join(t.TempDir(), "token-ids.jsonl")
	srv, err := newServer(config{models: modelFlag{{name: "tiny", path: tinyFillerCheckpoint(t)}}, kvSessions: 1,
		logTokenIDs: logPath, load: loadflags.Flags{Backend: "cpu", Quant: "int8int8"}})
	if err != nil {
		t.Fatalf("newServer on the tiny fixture: %v", err)
	}
	defer srv.closeTokenLog()
	key := newIDsKey([]byte("tier-0 test key, 32 bytes long.."))
	srv.idsKey = key
	lm := srv.models["tiny"]
	var logbuf bytes.Buffer
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/messages", srv.traceRequests(&logbuf, srv.handleMessages))
	mux.HandleFunc("GET /admin/status", srv.handleAdminStatus)
	ts := httptest.NewServer(mux)
	defer ts.Close()

	lineRE := regexp.MustCompile(`^request: POST /v1/messages model=tiny status=200 prompt_tokens=(\d+) completion_tokens=(\d+) ttft=\S+ total=\d+ms reused=(\d+) slot=(\d+) div=(\S+) canon=([01]) ids=(\d+/\d+) ids_fallback=(\S+)$`)
	type arm struct {
		name                string
		content             []map[string]any
		text                string
		inTok, reused, slot int
		div, canon, ids, fb string
		promptSHA           string
		gen, promptIDs      []int
	}
	nTok := 0
	post := func(name string, body map[string]any) *arm {
		t.Helper()
		logbuf.Reset()
		b, _ := json.Marshal(body)
		resp, err := http.Post(ts.URL+"/v1/messages", "application/json", bytes.NewReader(b))
		if err != nil {
			t.Fatal(err)
		}
		raw, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode != 200 {
			t.Fatalf("%s: %d %s", name, resp.StatusCode, raw)
		}
		var msg map[string]any
		if err := json.Unmarshal(raw, &msg); err != nil {
			t.Fatal(err)
		}
		a := &arm{name: name, content: messagesContent(t, []map[string]any{msg}, false)}
		for _, c := range a.content {
			if c["type"] == "text" {
				a.text = fmtString(c["text"])
			}
		}
		a.inTok = int(msg["usage"].(map[string]any)["input_tokens"].(float64))
		m := lineRE.FindStringSubmatch(strings.TrimSpace(logbuf.String()))
		if m == nil {
			t.Fatalf("%s: the -log-requests line does not have the documented shape: %q", name, logbuf.String())
		}
		a.reused, _ = strconv.Atoi(m[3])
		a.slot, _ = strconv.Atoi(m[4])
		a.div, a.canon, a.ids, a.fb = m[5], m[6], m[7], m[8]
		f, err := os.Open(logPath)
		if err != nil {
			t.Fatal(err)
		}
		sc := bufio.NewScanner(f)
		sc.Buffer(make([]byte, 1<<20), 1<<20)
		var lines []map[string]any
		for sc.Scan() {
			var l map[string]any
			if err := json.Unmarshal(sc.Bytes(), &l); err != nil {
				t.Fatalf("token-id log: %v", err)
			}
			lines = append(lines, l)
		}
		f.Close()
		if len(lines) != nTok+1 {
			t.Fatalf("%s: the token-id log has %d lines after %d generations", name, len(lines), nTok+1)
		}
		nTok++
		tl := lines[len(lines)-1]
		if tl["route"] != "/v1/messages" {
			t.Errorf("%s: token-id line names route %v", name, tl["route"])
		}
		a.promptSHA = tl["prompt_sha256"].(string)
		for _, g := range tl["gen"].([]any) {
			a.gen = append(a.gen, int(g.(float64)))
		}
		if int(tl["reused"].(float64)) != a.reused || int(tl["slot"].(float64)) != a.slot {
			t.Errorf("%s: the token-id line and the request line disagree: %v vs reused=%d slot=%d", name, tl, a.reused, a.slot)
		}
		// the prompt ids the route renders for this body must be the ones the handler prefilled
		a.promptIDs, _, err = kv0RenderIDs(srv, lm, "/v1/messages", b, true)
		if err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(idsBytes(a.promptIDs))
		if len(a.promptIDs) != int(tl["prompt_len"].(float64)) || hex.EncodeToString(sum[:]) != a.promptSHA || len(a.promptIDs) != a.inTok {
			t.Fatalf("%s: the route's render (%d ids) is not what the handler prefilled (%v ids, input_tokens %d)", name, len(a.promptIDs), tl["prompt_len"], a.inTok)
		}
		return a
	}
	req := func(msgs []any, maxTok int) map[string]any {
		return map[string]any{"model": "tiny", "max_tokens": maxTok, "temperature": 0, "messages": msgs, "thinking": thinkingReq}
	}
	user := func(s string) map[string]any { return map[string]any{"role": "user", "content": s} }

	t1 := post("turn 1", req([]any{user("Hi")}, 12))
	if len(t1.content) != 2 || t1.content[0]["type"] != "redacted_thinking" || t1.content[1]["type"] != "text" {
		t.Fatalf("turn 1 content %v: want a redacted_thinking carrier, then the text", t1.content)
	}
	valid := fmtString(t1.content[0]["data"])
	blob, err := parseIDsBlob(valid)
	if err != nil || !slices.Equal(blob.ids, t1.gen) || !blob.whole || blob.text != "" {
		t.Fatalf("turn 1's carrier %+v (%v); the generation's ids are %v", blob, err, t1.gen)
	}
	canon, _ := lm.tk.EncodeSegments([]chat.Segment{{Text: t1.text}}, false)
	if slices.Equal(canon, t1.gen) || t1.canon != "0" {
		t.Fatalf("VOID: turn 1's reply %q holds no planted non-canonical split (generated %v, canonical %v)", t1.text, t1.gen, canon)
	}
	P1 := t1.promptIDs
	slot := append(slices.Clone(P1), t1.gen...)

	next := func(enc string) map[string]any {
		var content []any
		if enc != "" {
			content = append(content, map[string]any{"type": "redacted_thinking", "data": enc})
		}
		content = append(content, map[string]any{"type": "text", "text": t1.text})
		return req([]any{user("Hi"), map[string]any{"role": "assistant", "content": content}, user("More")}, 8)
	}
	other := req([]any{user("Another conversation")}, 4)
	forgedIDs := nonCanonicalSplit(t, lm, t1.gen)
	if slices.Equal(forgedIDs, canon) {
		forgedIDs = slices.Clone(t1.gen)
	}
	forged := (&idsBlob{whole: true, keyID: key.id, model: lm.idsIdentity(), ids: forgedIDs}).String()
	otherKey := newIDsKey([]byte("another key for the tier-0 test!")).mint(lm.idsIdentity(), "", "", t1.gen, true)

	aw := post("A-warm", next(valid))
	post("other (cold for A-cold)", other)
	ac := post("A-cold", next(valid))
	post("other (cold for B)", other)
	b := post("B", next(forged))
	post("other (cold for C)", other)
	c := post("C", next(otherKey))
	post("other (cold for D1)", other)
	d1 := post("D1", next(""))
	post("other (cold for D2)", other)
	d2 := post("D2", next(""))

	if d1.text != d2.text || d1.promptSHA != d2.promptSHA {
		t.Fatalf("VOID: the cold path is not reproducible: D1 %q, D2 %q", d1.text, d2.text)
	}
	for _, a := range []*arm{aw, ac} {
		if a.ids != "1/1" || a.fb != "-" {
			t.Errorf("%s: ids=%s ids_fallback=%s, want 1/1 and none", a.name, a.ids, a.fb)
		}
		if len(a.promptIDs) < len(slot) || !slices.Equal(a.promptIDs[:len(slot)], slot) {
			t.Errorf("%s: the prompt does not start with turn 1's prompt and generated ids", a.name)
		}
	}
	if ac.promptSHA != aw.promptSHA {
		t.Errorf("A-cold's prompt ids differ from A-warm's: they must not depend on the KV")
	}
	if at := idsLCP(ac.promptIDs, d1.promptIDs); at < len(P1) || at >= len(slot) {
		t.Errorf("A's prompt and D's first differ at %d, want inside the reply [%d, %d)", at, len(P1), len(slot))
	}
	if aw.reused < len(slot)-1 {
		t.Errorf("A-warm reused %d rows; warm means the whole previous prompt and reply (%d)", aw.reused, len(slot))
	}
	for _, a := range []*arm{ac, b, c, d1, d2} {
		if a.reused >= len(P1) || a.div == "-" {
			t.Errorf("%s: reused %d rows (div=%s, slot=%d); cold means at most the preamble shared with the other conversation (< %d)", a.name, a.reused, a.div, a.slot, len(P1))
		}
	}
	for _, x := range []struct {
		a      *arm
		reason string
	}{{b, idsMAC}, {c, idsKeyR}} {
		if x.a.ids != "0/1" || x.a.fb != x.reason {
			t.Errorf("%s: ids=%s ids_fallback=%s, want 0/1 %s", x.a.name, x.a.ids, x.a.fb, x.reason)
		}
		if x.a.promptSHA != d1.promptSHA || x.a.text != d1.text {
			t.Errorf("%s: prompt or output differ from D's (output %q, D's %q)", x.a.name, x.a.text, d1.text)
		}
	}
	if d1.ids != "0/0" || d1.fb != "-" {
		t.Errorf("D1: ids=%s ids_fallback=%s, want 0/0", d1.ids, d1.fb)
	}
	t.Logf("A-warm reused %d of %d; A-cold %d; B/C/D1/D2 %d/%d/%d/%d; outputs A %q/%q, D %q", aw.reused, aw.inTok, ac.reused, b.reused, c.reused, d1.reused, d2.reused, aw.text, ac.text, d1.text)

	resp, err := http.Get(ts.URL + "/admin/status")
	if err != nil {
		t.Fatal(err)
	}
	var st struct {
		IDs struct {
			Carried   int            `json:"carried"`
			Used      int            `json:"used"`
			Fallbacks map[string]int `json:"fallbacks"`
		} `json:"carried_ids"`
	}
	_ = json.NewDecoder(resp.Body).Decode(&st)
	resp.Body.Close()
	if st.IDs.Carried != 4 || st.IDs.Used != 2 || st.IDs.Fallbacks[idsMAC] != 1 || st.IDs.Fallbacks[idsKeyR] != 1 || len(st.IDs.Fallbacks) != 2 {
		t.Errorf("/admin/status carried_ids = %+v, want 4 carried, 2 used, mac 1, key 1", st.IDs)
	}
}

// messagesFromResponses is a step-0 Responses request as a Messages client sends the same conversation: developer messages
// before the first turn as the system field and later ones as system messages, each assistant turn one message (a reasoning
// item's text as a thinking block whose signature is the item's v2 blob, if any; a call as tool_use; text as a text block),
// and call outputs as tool_result blocks.
func messagesFromResponses(t *testing.T, body map[string]any) map[string]any {
	t.Helper()
	out := map[string]any{"model": body["model"], "max_tokens": 4096, "thinking": thinkingReq}
	var tools []any
	for _, x := range body["tools"].([]any) {
		if m := x.(map[string]any); m["type"] == "function" {
			tools = append(tools, map[string]any{"name": m["name"], "description": m["description"], "input_schema": m["parameters"]})
		}
	}
	out["tools"] = tools
	partsText := func(v any) string {
		if s, ok := v.(string); ok {
			return s
		}
		var b strings.Builder
		for _, p := range v.([]any) {
			b.WriteString(fmtString(p.(map[string]any)["text"]))
		}
		return b.String()
	}
	var system []string
	var msgs []any
	var cur map[string]any // the message blocks are being added to
	open := false          // cur is an assistant turn a reasoning item opened and nothing has closed
	add := func(role string, block map[string]any, fresh bool) {
		if fresh || cur == nil || cur["role"] != role {
			cur = map[string]any{"role": role, "content": []any{}}
			msgs = append(msgs, cur)
		}
		cur["content"] = append(cur["content"].([]any), block)
	}
	for _, x := range body["input"].([]any) {
		it := x.(map[string]any)
		switch it["type"] {
		case "reasoning":
			c, _ := json.Marshal(it["content"])
			s, _ := json.Marshal(it["summary"])
			enc := fmtString(it["encrypted_content"])
			text, _, err := reasoningItemCarried(enc, c, s)
			if err != nil {
				t.Fatal(err)
			}
			sig := ""
			if strings.HasPrefix(enc, reasoningBlobV2Prefix) {
				sig = enc
			}
			add("assistant", map[string]any{"type": "thinking", "thinking": text, "signature": sig}, true)
			open = true
		case "function_call":
			add("assistant", map[string]any{"type": "tool_use", "id": it["call_id"], "name": it["name"], "input": json.RawMessage(fmtString(it["arguments"]))}, !open)
			open = false
		case "function_call_output":
			o, _ := json.Marshal(it["output"])
			add("user", map[string]any{"type": "tool_result", "tool_use_id": it["call_id"], "content": toolOutputText(o)}, false)
			open = false
		default:
			text := partsText(it["content"])
			switch it["role"] {
			case "developer", "system":
				if len(msgs) == 0 {
					system = append(system, text)
				} else {
					msgs, cur = append(msgs, map[string]any{"role": "system", "content": text}), nil
				}
			case "assistant":
				add("assistant", map[string]any{"type": "text", "text": text}, !open)
			default:
				add("user", map[string]any{"type": "text", "text": text}, true)
			}
			open = false
		}
	}
	out["system"] = strings.Join(system, "\n\n")
	out["messages"] = msgs
	return out
}

// jsMarshal encodes as a JavaScript client re-sends JSON it parsed: compact, and without Go's HTML escaping.
func jsMarshal(t *testing.T, v any) []byte {
	t.Helper()
	var b bytes.Buffer
	e := json.NewEncoder(&b)
	e.SetEscapeHTML(false)
	if err := e.Encode(v); err != nil {
		t.Fatal(err)
	}
	return bytes.TrimSpace(b.Bytes())
}

// G-ids-0 on /v1/messages: step 0's 45 replies, each carried in its thinking block's signature in the next Messages request.
// Carried and untouched, the next prompt starts with the slot's id record (the previous request's prompt, then every generated
// id). The arguments come back compacted, as a client that parsed them sends them, so a reply whose arguments the model spaced
// keeps its ids by the closed list (ids_fmt=1). Every planted defect gives today's tokenization, its reason and no error. The
// Responses-only defect (k), a v1 blob, becomes a signature this server did not mint, which carries nothing.
func TestIDsCarriedMessages_G0(t *testing.T) {
	tkDir := filepath.Join("..", "..", "testdata", "mellum2-tokenizer")
	if _, err := os.Stat(filepath.Join(tkDir, "tokenizer.json")); err != nil {
		t.Skipf("G-ids-0 needs the Mellum2 tokenizer at %s (a per-machine symlink): %v", tkDir, err)
	}
	tk, err := modelload.Tokenizer(tkDir)
	if err != nil {
		t.Fatal(err)
	}
	s := &server{}
	key := newIDsKey([]byte("G-ids-0 test key, 32 bytes long."))
	s.idsKey = key
	lm := &loadedModel{tk: tk, name: "mellum2.1-thinking"}
	lm.tmpl = s.cfg.tuneTemplate(mellum21Template(t))
	model := lm.idsIdentity()
	render := func(body map[string]any, useIDs bool) ([]int, idsOutcome) {
		t.Helper()
		ids, o, err := kv0RenderIDs(s, lm, "/v1/messages", jsMarshal(t, body), useIDs)
		if err != nil {
			t.Fatal(err)
		}
		return ids, o
	}
	other := newIDsKey([]byte("a different key, also 32 bytes.."))
	fx := loadIDsStep0(t)
	cases := map[string]int{}
	held, fmtd := 0, 0
	for _, rp := range fx.replies {
		prev := messagesFromResponses(t, fx.body(rp.Session, rp.NInput))
		P, po := render(prev, true)
		if po.carried != 0 {
			t.Fatalf("%s: the previous request carries %d turns; step 0's archived history is v1 only", rp.ID, po.carried)
		}
		slot := append(slices.Clone(P), rp.Gen...)
		rsText, args, msgText := "", "", ""
		for _, it := range rp.Items {
			switch it["type"] {
			case "reasoning":
				for _, c := range it["content"].([]any) {
					rsText += c.(map[string]any)["text"].(string)
				}
			case "function_call":
				if args == "" {
					args = it["arguments"].(string)
				}
			case "message":
				msgText = it["content"].([]any)[0].(map[string]any)["text"].(string)
			}
		}
		valid := key.mint(model, "", rsText, rp.Gen, true)
		// next is the next request: the reply as an assistant message (thinking with sig, then text and calls), then the
		// session's follow-up items, translated.
		next := func(sig, thinking string, keepThinking bool, callArgs, text string) map[string]any {
			nb := fx.body(rp.Session, rp.NInput)
			var reply []any
			if keepThinking {
				reply = append(reply, map[string]any{"type": "reasoning", "content": []any{map[string]any{"type": "reasoning_text", "text": thinking}}, "summary": []any{}})
			}
			for _, it := range rp.Items {
				switch it["type"] {
				case "function_call":
					c := map[string]any{}
					for k, v := range it {
						c[k] = v
					}
					if it["arguments"] == args {
						c["arguments"] = callArgs
					}
					reply = append(reply, c)
				case "message":
					reply = append(reply, map[string]any{"type": "message", "role": "assistant", "content": []any{map[string]any{"type": "output_text", "text": text}}})
				}
			}
			nb["input"] = append(append(append([]any(nil), nb["input"].([]any)...), reply...), rp.Follow...)
			m := messagesFromResponses(t, nb)
			if keepThinking { // the translator put the visible text in the block; the signature is the case's
				for _, x := range m["messages"].([]any) {
					for _, bl := range x.(map[string]any)["content"].([]any) {
						if b, ok := bl.(map[string]any); ok && b["type"] == "thinking" && b["thinking"] == thinking && b["signature"] == "" {
							b["signature"] = sig
						}
					}
				}
			}
			return m
		}
		var compact bytes.Buffer
		formatted := args != "" && json.Compact(&compact, []byte(args)) == nil && compact.String() != args
		want := func(reason string) string {
			var o idsOutcome
			if reason != "none" {
				o.note(reason)
			}
			if reason == "" && formatted {
				o.formatted = 1
			}
			return o.fields()
		}
		base := next(valid, rsText, true, args, msgText)
		ids, out := render(base, true)
		if out.fields() != want("") {
			t.Errorf("%s carried: %s, want %s", rp.ID, out.fields(), want(""))
		}
		if len(ids) < len(slot) || !slices.Equal(ids[:len(slot)], slot) {
			t.Errorf("%s carried: the next prompt leaves the slot at %d of %d", rp.ID, idsLCP(ids, slot), len(slot))
		} else {
			held++
		}
		if formatted {
			fmtd++
		}
		cases["carried"]++
		if ids, out := render(next(valid, "edited "+rsText, true, args, msgText), true); out.fields() != want("") || !slices.Equal(ids[:min(len(ids), len(slot))], slot) {
			t.Errorf("%s visible edit: %s; the blob's text and ids must win over the visible thinking", rp.ID, out.fields())
		}
		cases["visible-edit"]++
		todayBase, _ := render(base, false)

		type defect struct {
			name, reason string
			body         map[string]any
			textEdit     bool
		}
		ds := []defect{{"a", idsDecode, next(key.mint(model, "", flipChar(rsText), rp.Gen, true), rsText, true, args, msgText), true}}
		if args != "" {
			ds = append(ds, defect{"b", idsDecode, next(valid, rsText, true, flipChar(args), msgText), true})
			if strings.Contains(args, `\\t`) {
				ds = append(ds, defect{"c", idsDecode, next(valid, rsText, true, strings.Replace(args, `\\t`, `\t`, 1), msgText), true})
			}
		}
		if msgText != "" {
			ds = append(ds, defect{"d", idsDecode, next(valid, rsText, true, args, flipChar(msgText)), true})
		}
		forged := nonCanonicalSplit(t, lm, rp.Gen)
		noMAC := &idsBlob{whole: true, keyID: key.id, model: model, text: rsText, ids: forged}
		randMAC := &idsBlob{whole: true, keyID: key.id, model: model, text: rsText, ids: forged, mac: bytes.Repeat([]byte{0x5a}, 32)}
		hText, hIDs := plantSpecialInReasoning(t, lm, rsText, rp.Gen)
		ds = append(ds,
			defect{"e", idsMAC, next(noMAC.String(), rsText, true, args, msgText), false},
			defect{"f", idsMAC, next(randMAC.String(), rsText, true, args, msgText), false},
			defect{"g", idsKeyR, next(other.mint(model, "", rsText, rp.Gen, true), rsText, true, args, msgText), false},
			defect{"h", idsSpecial, next(key.mint(model, "", hText, hIDs, true), rsText, true, args, msgText), false},
			defect{"i", idsNS, next(key.mint(model, "another-namespace", rsText, rp.Gen, true), rsText, true, args, msgText), false},
			defect{"j", idsModel, next(key.mint("another-model", "", rsText, rp.Gen, true), rsText, true, args, msgText), false},
			defect{"k", "none", next("EqQBCkYIBhgCKkBforeignSignatureNotMintedHere==", rsText, true, args, msgText), false},
			defect{"l", idsParse, next(valid[:len(reasoningBlobV2Prefix)+(len(valid)-len(reasoningBlobV2Prefix))/2]+"!!", rsText, true, args, msgText), false},
			defect{"m", "none", next("", rsText, false, args, msgText), false},
		)
		for _, d := range ds {
			got, out := render(d.body, true)
			today, _ := render(d.body, false)
			if d.textEdit && slices.Equal(today, todayBase) {
				t.Fatalf("VOID %s (%s): the planted edit left the rendered prompt unchanged", rp.ID, d.name)
			}
			if !slices.Equal(got, today) {
				t.Errorf("%s (%s): the fallback is not today's tokenization (first difference at %d)", rp.ID, d.name, idsLCP(got, today))
			}
			if out.fields() != want(d.reason) {
				t.Errorf("%s (%s): %s, want %s", rp.ID, d.name, out.fields(), want(d.reason))
			}
			cases[d.name]++
		}
	}
	t.Logf("G-ids-0 on /v1/messages: carried and untouched, next prompt starts with the slot: %d of %d (%d by the closed list); cases run: %v", held, len(fx.replies), fmtd, cases)
	if held != len(fx.replies) {
		t.Errorf("carried and untouched held in %d of %d, want all", held, len(fx.replies))
	}
	for _, c := range []string{"carried", "visible-edit", "a", "e", "f", "g", "h", "i", "j", "k", "l", "m"} {
		if cases[c] != 45 {
			t.Errorf("case %s ran %d times, want 45", c, cases[c])
		}
	}
	if cases["b"] != 44 || cases["d"] != 1 || cases["c"] == 0 {
		t.Errorf("cases b/c/d ran %d/%d/%d, want 44/at least 1/1", cases["b"], cases["c"], cases["d"])
	}
}
