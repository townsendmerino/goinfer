package serveapp

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/internal/loadflags"
)

// The v2 blob's format: every field survives a round trip, and a blob that is cut short or garbled does not parse.
func TestIDsBlob_roundTrip(t *testing.T) {
	k := newIDsKey([]byte("round-trip test key of 32 bytes!"))
	enc := k.mint("model-x", "ns-y", "reasoning text\nwith a newline", []int{0, 1, 127, 128, 300000, 5}, false)
	if !strings.HasPrefix(enc, reasoningBlobV2Prefix) {
		t.Fatalf("minted %q", enc)
	}
	b, err := parseIDsBlob(enc)
	if err != nil {
		t.Fatal(err)
	}
	if b.whole || b.keyID != k.id || b.model != "model-x" || b.ns != "ns-y" || b.text != "reasoning text\nwith a newline" ||
		!slices.Equal(b.ids, []int{0, 1, 127, 128, 300000, 5}) || len(b.mac) != 32 {
		t.Errorf("parsed %+v", b)
	}
	if !k.verifyMAC(b) {
		t.Error("the key does not verify its own MAC")
	}
	b.ids[2]++
	if k.verifyMAC(b) {
		t.Error("the MAC verifies after an id changed")
	}
	if text, err := reasoningItemText(enc, nil, nil); err != nil || text != "reasoning text\nwith a newline" {
		t.Errorf("reasoningItemText on a v2 blob: %q, %v", text, err)
	}
	for _, bad := range []string{enc[:len(enc)-6], enc[:len(reasoningBlobV2Prefix)+3], reasoningBlobV2Prefix + "%%%", reasoningBlobV2Prefix} {
		if _, err := parseIDsBlob(bad); err == nil {
			t.Errorf("%q parsed", bad)
		}
	}
}

// --ids-key-file: 32 bytes or more, mode 0600 or stricter, else refused; the key id is the first 8 bytes of SHA-256 of the key.
func TestIDsKeyFile(t *testing.T) {
	dir := t.TempDir()
	write := func(name string, b []byte, mode os.FileMode) string {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, b, mode); err != nil {
			t.Fatal(err)
		}
		if err := os.Chmod(p, mode); err != nil {
			t.Fatal(err)
		}
		return p
	}
	key := bytes.Repeat([]byte{7}, 40)
	k, err := readIDsKeyFile(write("ok", key, 0o600))
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(key)
	if !bytes.Equal(k.id[:], sum[:8]) || !bytes.Equal(k.key, key) {
		t.Errorf("key id %x, want %x", k.id, sum[:8])
	}
	if _, err := readIDsKeyFile(write("short", key[:31], 0o600)); err == nil {
		t.Error("a 31-byte key was accepted")
	}
	if _, err := readIDsKeyFile(write("open", key, 0o644)); err == nil && runtime.GOOS != "windows" { // Windows carries no mode bits
		t.Error("a key file readable by others was accepted")
	}
	if _, err := readIDsKeyFile(filepath.Join(dir, "missing")); err == nil {
		t.Error("a missing key file was accepted")
	}
	if w := idsKeyFileWarning("windows", "k"); !strings.Contains(w, "permissions not checked on Windows") {
		t.Errorf("windows: warning %q", w)
	}
	if idsKeyFileWarning("linux", "k") != "" || idsKeyFileWarning("windows", "") != "" {
		t.Error("a warning off Windows, or with no key file")
	}
	r1, err1 := randomIDsKey()
	r2, err2 := randomIDsKey()
	if err1 != nil || err2 != nil || len(r1.key) != 32 || bytes.Equal(r1.key, r2.key) {
		t.Error("the default key is not 32 random bytes per start")
	}
}

// tinyFillerTokenizer is tinyChatTokenizer filled to the tiny model's 256 ids. Every filler is a two-character piece that no
// merge reaches, so an id the model generates from them is a split the BPE never produces: the text it decodes to re-encodes
// as two one-character ids. That is the planted non-canonical split, by construction.
func tinyFillerTokenizer(t *testing.T) []byte {
	t.Helper()
	vocab := map[string]int{"Ġ": 0, "Ċ": 1}
	for c := '!'; c <= '~'; c++ {
		vocab[string(c)] = len(vocab)
	}
	var added []map[string]any
	for _, s := range []string{"<|endoftext|>", "<|im_start|>", "<|im_end|>"} {
		id := len(vocab)
		vocab[s] = id
		added = append(added, map[string]any{"id": id, "content": s, "special": true})
	}
	vocab["hi"] = len(vocab)
	for k := 0; len(vocab) < 256; k++ {
		vocab[string(rune('A'+k%26))+string(rune('a'+k/26))] = len(vocab)
	}
	b, err := json.Marshal(map[string]any{
		"model":         map[string]any{"type": "BPE", "vocab": vocab, "merges": []string{"h i"}},
		"pre_tokenizer": map[string]any{"type": "ByteLevel", "add_prefix_space": false},
		"decoder":       map[string]any{"type": "ByteLevel"},
		"added_tokens":  added,
	})
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func tinyFillerCheckpoint(t *testing.T) string {
	t.Helper()
	dir := tinyChatCheckpoint(t)
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), tinyFillerTokenizer(t), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

// G-ids-2's fallback half and its KV-independence check, at tier 0 (the record's §(c) Amendment 1), on the tiny committed
// fixture with one KV session, through the real /v1/responses handler. Turn 1 generates a reply whose ids hold a planted
// non-canonical split; the next request is then sent under each arm:
//
//	A-warm  the valid carried blob, right after its reply
//	A-cold  the valid carried blob, after another conversation took the session
//	B       a forged blob (the client's own split of the same text, no MAC), cold
//	C       the reply's ids signed by a different key, cold
//	D1, D2  the carrier removed, cold
//
// Rule: B and C counted mac and key, their prompt ids equal to D's, their output byte-identical to D's; A-cold's prompt ids
// equal to A-warm's and different from D's at the planted split; A-cold's reused rows show it was cold. Void: D1 and D2 differ.
// The counters read are the -log-requests fields, the token-id log (prompt ids by hash, generated ids) and /admin/status.
func TestIDsCarried_tinyFallbackAndKV(t *testing.T) {
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
	mux.HandleFunc("POST /v1/responses", srv.traceRequests(&logbuf, srv.handleResponses))
	mux.HandleFunc("GET /admin/status", srv.handleAdminStatus)
	ts := httptest.NewServer(mux)
	defer ts.Close()

	lineRE := regexp.MustCompile(`^request: POST /v1/responses model=tiny status=200 prompt_tokens=(\d+) completion_tokens=(\d+) ttft=\S+ total=\d+ms reused=(\d+) slot=(\d+) div=(\S+) canon=([01]) ids=(\d+/\d+) ids_fallback=(\S+)$`)
	type arm struct {
		name                 string
		body                 map[string]any
		resp                 map[string]any
		text                 string
		inTok, cached        int
		reused, slot         int
		div, canon, ids, fb  string
		promptLen            int
		promptSHA            string
		gen                  []int
		promptIDs, renderIDs []int
	}
	nTok := 0
	post := func(name string, body map[string]any) *arm {
		t.Helper()
		logbuf.Reset()
		b, _ := json.Marshal(body)
		resp, err := http.Post(ts.URL+"/v1/responses", "application/json", bytes.NewReader(b))
		if err != nil {
			t.Fatal(err)
		}
		raw, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode != 200 {
			t.Fatalf("%s: %d %s", name, resp.StatusCode, raw)
		}
		a := &arm{name: name, body: body}
		if err := json.Unmarshal(raw, &a.resp); err != nil {
			t.Fatal(err)
		}
		for _, it := range a.resp["output"].([]any) {
			if m := it.(map[string]any); m["type"] == "message" {
				a.text = m["content"].([]any)[0].(map[string]any)["text"].(string)
			}
		}
		u := a.resp["usage"].(map[string]any)
		a.inTok = int(u["input_tokens"].(float64))
		a.cached = int(u["input_tokens_details"].(map[string]any)["cached_tokens"].(float64))
		m := lineRE.FindStringSubmatch(strings.TrimSpace(logbuf.String()))
		if m == nil {
			t.Fatalf("%s: the -log-requests line does not have the documented shape: %q", name, logbuf.String())
		}
		a.reused, _ = strconv.Atoi(m[3])
		a.slot, _ = strconv.Atoi(m[4])
		a.div, a.canon, a.ids, a.fb = m[5], m[6], m[7], m[8]
		if a.reused != a.cached {
			t.Errorf("%s: reused=%d on the line, cached_tokens %d in the response", name, a.reused, a.cached)
		}
		// the token-id log's line for this generation
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
		if tl["route"] != "/v1/responses" || tl["id"] != a.resp["id"] {
			t.Errorf("%s: token-id line names %v %v", name, tl["route"], tl["id"])
		}
		a.promptLen = int(tl["prompt_len"].(float64))
		a.promptSHA = tl["prompt_sha256"].(string)
		for _, g := range tl["gen"].([]any) {
			a.gen = append(a.gen, int(g.(float64)))
		}
		if int(tl["reused"].(float64)) != a.reused || int(tl["slot"].(float64)) != a.slot || strconv.Itoa(int(tl["canon"].(float64))) != a.canon {
			t.Errorf("%s: the token-id line and the request line disagree: %v vs reused=%d slot=%d canon=%s", name, tl, a.reused, a.slot, a.canon)
		}
		// the prompt ids the route renders for this body, which must be the ones the handler prefilled
		a.renderIDs, _, err = kv0RenderIDs(srv, lm, "/v1/responses", b, true)
		if err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(idsBytes(a.renderIDs))
		if len(a.renderIDs) != a.promptLen || hex.EncodeToString(sum[:]) != a.promptSHA || a.promptLen != a.inTok {
			t.Fatalf("%s: the route's render (%d ids) is not what the handler prefilled (%d ids, input_tokens %d)", name, len(a.renderIDs), a.promptLen, a.inTok)
		}
		a.promptIDs = a.renderIDs
		return a
	}

	req := func(input []any, maxTok int) map[string]any {
		return map[string]any{"model": "tiny", "max_output_tokens": maxTok, "temperature": 0, "input": input,
			"include": []any{"reasoning.encrypted_content"}}
	}
	user := func(s string) map[string]any { return map[string]any{"role": "user", "content": s} }

	// Turn 1.
	t1 := post("turn 1", req([]any{user("Hi")}, 12))
	out := t1.resp["output"].([]any)
	if len(out) != 2 {
		t.Fatalf("turn 1 output %v: want a carrier-only reasoning item, then the message", out)
	}
	carrier, msg := out[0].(map[string]any), out[1].(map[string]any)
	if carrier["type"] != "reasoning" || len(carrier["content"].([]any)) != 0 || msg["type"] != "message" {
		t.Fatalf("turn 1 output %v", out)
	}
	blob, err := parseIDsBlob(carrier["encrypted_content"].(string))
	if err != nil {
		t.Fatalf("turn 1's carrier: %v", err)
	}
	if !slices.Equal(blob.ids, t1.gen) || !blob.whole || blob.text != "" {
		t.Fatalf("turn 1's carrier holds ids %v (whole %v, text %q); the generation's ids are %v", blob.ids, blob.whole, blob.text, t1.gen)
	}
	canon, _ := lm.tk.EncodeSegments([]chat.Segment{{Text: t1.text}}, false)
	if slices.Equal(canon, t1.gen) || t1.canon != "0" {
		t.Fatalf("VOID: turn 1's reply %q holds no planted non-canonical split (generated %v, canonical %v, canon=%s)", t1.text, t1.gen, canon, t1.canon)
	}
	if t1.ids != "0/0" || t1.fb != "-" || t1.div != "-" {
		t.Errorf("turn 1 line: ids=%s ids_fallback=%s div=%s", t1.ids, t1.fb, t1.div)
	}
	P1 := t1.promptIDs
	slot := append(append([]int(nil), P1...), t1.gen...)

	echo := map[string]any{}
	for k, v := range msg {
		if k != "status" {
			echo[k] = v
		}
	}
	carrierItem := func(enc string) map[string]any {
		return map[string]any{"type": "reasoning", "id": carrier["id"], "summary": []any{}, "content": []any{}, "encrypted_content": enc}
	}
	next := func(enc string) map[string]any {
		in := []any{user("Hi")}
		if enc != "" {
			in = append(in, carrierItem(enc))
		}
		return req(append(in, echo, user("More")), 8)
	}
	other := req([]any{user("Another conversation")}, 4)
	valid := carrier["encrypted_content"].(string)
	forgedIDs := nonCanonicalSplit(t, lm, t1.gen)
	if slices.Equal(forgedIDs, canon) {
		forgedIDs = append([]int(nil), t1.gen...) // the per-character split is the canonical one here; any split will do without a MAC
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
	if ac.promptSHA != aw.promptSHA || !slices.Equal(ac.promptIDs, aw.promptIDs) {
		t.Errorf("A-cold's prompt ids differ from A-warm's: they must not depend on the KV")
	}
	if at := idsLCP(ac.promptIDs, d1.promptIDs); at < len(P1) || at >= len(slot) {
		t.Errorf("A's prompt and D's first differ at %d, want inside the reply [%d, %d), where the planted split is", at, len(P1), len(slot))
	}
	if aw.cached < len(slot)-1 {
		t.Errorf("A-warm reused %d rows; warm means the whole previous prompt and reply (%d, the last generated id may be uncommitted)", aw.cached, len(slot))
	}
	for _, a := range []*arm{ac, b, c, d1, d2} {
		if a.cached >= len(P1) || a.div == "-" || a.div != strconv.Itoa(a.reused) {
			t.Errorf("%s: reused %d rows (div=%s, slot=%d); cold means at most the preamble shared with the other conversation (< %d)", a.name, a.cached, a.div, a.slot, len(P1))
		}
	}
	for _, x := range []struct {
		a      *arm
		reason string
	}{{b, idsMAC}, {c, idsKeyR}} {
		if x.a.ids != "0/1" || x.a.fb != x.reason {
			t.Errorf("%s: ids=%s ids_fallback=%s, want 0/1 %s", x.a.name, x.a.ids, x.a.fb, x.reason)
		}
		if x.a.promptSHA != d1.promptSHA || !slices.Equal(x.a.promptIDs, d1.promptIDs) {
			t.Errorf("%s: prompt ids differ from D's (first at %d)", x.a.name, idsLCP(x.a.promptIDs, d1.promptIDs))
		}
		if x.a.text != d1.text {
			t.Errorf("%s: output %q, D's %q", x.a.name, x.a.text, d1.text)
		}
	}
	if d1.ids != "0/0" || d1.fb != "-" {
		t.Errorf("D1: ids=%s ids_fallback=%s, want 0/0", d1.ids, d1.fb)
	}
	t.Logf("A-warm reused %d of %d; A-cold %d; B/C/D1/D2 %d/%d/%d/%d; outputs A %q/%q, D %q", aw.cached, aw.inTok, ac.cached, b.cached, c.cached, d1.cached, d2.cached, aw.text, ac.text, d1.text)

	// /admin/status: the totals by reason since start.
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

// A reply that reasons carries its ids in the reasoning item, streamed or not; a reply that does not reason carries them in a
// carrier-only item only when the request asked for reasoning.encrypted_content. On the scripted Mellum2.1 server.
func TestIDsCarried_carrierItems(t *testing.T) {
	const gen = "<think>\nI will read calc.go.\n</think>\n\n<tool_call>\n{\"name\": \"exec_command\", \"arguments\": {\"cmd\": \"cat calc.go\"}}\n</tool_call>"
	ts, genIDs, _ := scriptedIDsServer(t, gen)
	for _, stream := range []bool{false, true} {
		req := requestFor("/v1/responses", true, stream)
		evs := postScripted(t, ts.URL+"/v1/responses", req)
		out := lastOutput(t, evs, stream)
		b, err := parseIDsBlob(out[0].(map[string]any)["encrypted_content"].(string))
		if err != nil || !slices.Equal(b.ids, genIDs) || !b.whole || b.text != "I will read calc.go." {
			t.Errorf("stream=%v: reasoning item's blob %+v (%v); want the whole turn's ids", stream, b, err)
		}
		if stream { // the item as Codex reads it: from response.output_item.done
			for _, e := range evs {
				if e["type"] == "response.output_item.done" && e["item"].(map[string]any)["type"] == "reasoning" {
					if e["item"].(map[string]any)["encrypted_content"] != out[0].(map[string]any)["encrypted_content"] {
						t.Errorf("the reasoning item's done event carries a different blob from response.completed")
					}
				}
			}
		}
	}
	// No reasoning: a carrier-only item with include, none without.
	const plain = "<tool_call>\n{\"name\": \"exec_command\", \"arguments\": {\"cmd\": \"ls\"}}\n</tool_call>"
	ts2, ids2, _ := scriptedIDsServer(t, plain)
	for _, stream := range []bool{false, true} {
		req := requestFor("/v1/responses", true, stream)
		req["reasoning"] = map[string]any{"effort": "none"}
		out := lastOutput(t, postScripted(t, ts2.URL+"/v1/responses", req), stream)
		if out[0].(map[string]any)["type"] == "reasoning" {
			t.Errorf("stream=%v: a carrier-only item without include", stream)
		}
		req["include"] = []any{"reasoning.encrypted_content"}
		out = lastOutput(t, postScripted(t, ts2.URL+"/v1/responses", req), stream)
		it := out[0].(map[string]any)
		b, err := parseIDsBlob(fmtString(it["encrypted_content"]))
		if it["type"] != "reasoning" || len(it["content"].([]any)) != 0 || err != nil || !slices.Equal(b.ids, ids2) {
			t.Errorf("stream=%v: first item %v (%v); want a carrier-only reasoning item with the turn's ids", stream, it, err)
		}
	}
}

// A streamed answer opens its message while the turn is still generating, so the reasoning item closes first: it carries the
// ids up to the end of the reasoning block (whole false), and the answer re-tokenizes. Not streamed, it carries the whole
// turn. Echoed back with the turn still in the loop in progress, the ids are used; after a new query, Mellum2.1's history rule
// drops the turn's reasoning from the prompt, and the turn counts as history.
func TestIDsCarried_streamedAnswer(t *testing.T) {
	const gen = "<think>\nAdd returns a + b.\n</think>\n\n5"
	ts, ids, s := scriptedIDsServer(t, gen)
	lm := s.models["m"]
	end, _ := lm.tk.TokenID("</think>")
	cut := slices.Index(ids, end) + 1
	for _, stream := range []bool{false, true} {
		req := requestFor("/v1/responses", false, stream)
		out := lastOutput(t, postScripted(t, ts.URL+"/v1/responses", req), stream)
		rs := out[0].(map[string]any)
		b, err := parseIDsBlob(fmtString(rs["encrypted_content"]))
		if err != nil {
			t.Fatalf("stream=%v: %v", stream, err)
		}
		if want := ids[:cut]; stream && (b.whole || !slices.Equal(b.ids, want)) || !stream && (!b.whole || !slices.Equal(b.ids, ids)) {
			t.Errorf("stream=%v: blob whole=%v ids %v; generated %v", stream, b.whole, b.ids, ids)
		}
		msg := map[string]any{}
		for k, v := range out[1].(map[string]any) {
			if k != "status" {
				msg[k] = v
			}
		}
		for _, tc := range []struct {
			name, want string
			tail       []any
		}{
			{"in the loop", "ids=1/1 ids_fallback=-", nil},
			{"after a new query", "ids=0/1 ids_fallback=history", []any{map[string]any{"role": "user", "content": "And 3 + 4?"}}},
		} {
			input := append([]any{map[string]any{"role": "user", "content": req["input"]}, rs, msg}, tc.tail...)
			body, _ := json.Marshal(map[string]any{"model": "m", "input": input})
			_, o, err := kv0RenderIDs(s, lm, "/v1/responses", body, true)
			if err != nil || o.fields() != tc.want {
				t.Errorf("stream=%v, %s: %s (%v), want %s", stream, tc.name, o.fields(), err, tc.want)
			}
		}
	}
}

// scriptedIDsServer is scriptedServer's Mellum2.1 model with a test key, returning the reply's ids.
func scriptedIDsServer(t *testing.T, reply string) (*httptest.Server, []int, *server) {
	t.Helper()
	tk := scriptTokenizer(t)
	ids, err := tk.Encode(reply, false)
	if err != nil {
		t.Fatal(err)
	}
	s := &server{idsKey: newIDsKey([]byte("scripted test key, 32 bytes long"))}
	lm := &loadedModel{tk: tk, name: "m", scripted: append([]int{}, ids...)}
	lm.tmpl = s.cfg.tuneTemplate(mellum21Template(t))
	s.models = map[string]*loadedModel{"m": lm}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/responses", s.handleResponses)
	mux.HandleFunc("POST /v1/messages", s.handleMessages)
	ts := httptest.NewServer(mux)
	t.Cleanup(ts.Close)
	return ts, ids, s
}

func lastOutput(t *testing.T, evs []map[string]any, stream bool) []any {
	t.Helper()
	if !stream {
		return evs[0]["output"].([]any)
	}
	for _, e := range slices.Backward(evs) {
		if e["type"] == "response.completed" {
			return e["response"].(map[string]any)["output"].([]any)
		}
	}
	t.Fatal("no response.completed")
	return nil
}

func fmtString(v any) string { s, _ := v.(string); return s }

// --carry-ids defaults on (a server built from a zero config carries ids too); --carry-ids=false leaves the server with no key,
// so replies carry the v1 text blob and no echoed turn uses ids.
func TestIDsCarried_flag(t *testing.T) {
	for _, tc := range []struct {
		args []string
		off  bool
	}{{nil, false}, {[]string{"--carry-ids=false"}, true}, {[]string{"--carry-ids"}, false}} {
		fs := flag.NewFlagSet("serve", flag.ContinueOnError)
		sf := registerFlags(fs)
		if err := fs.Parse(tc.args); err != nil {
			t.Fatal(err)
		}
		if sf.cfg.noCarryIDs != tc.off {
			t.Errorf("%v: noCarryIDs %v", tc.args, sf.cfg.noCarryIDs)
		}
	}
	for _, off := range []bool{false, true} {
		srv, err := newServer(config{noCarryIDs: off})
		if err != nil {
			t.Fatal(err)
		}
		if (srv.idsKey == nil) != off {
			t.Errorf("noCarryIDs=%v: key %v", off, srv.idsKey != nil)
		}
	}
}
