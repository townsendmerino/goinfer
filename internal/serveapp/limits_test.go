package serveapp

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/tokenizer"
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"math/big"
)

// C-08: top_logprobs was the one sampling field `prepare` passed through without a range check.
//
// Each retained entry is a TokenLogprob and the response builder then materializes a
// map[string]any per entry BEFORE writing a byte, so {logprobs:true, top_logprobs:150000,
// max_tokens:4096} on a 152k-vocab model retains ~9.8 GB of them and OOM-kills the process — a
// fatal Go allocation failure, not a 500 anyone can catch. One request does it, and every other
// sampling field on the same struct was already validated.
func TestPrepare_rejectsUnboundedTopLogprobs(t *testing.T) {
	lm := &loadedModel{}
	n := func(v int) *int { return &v }

	for _, bad := range []int{-1, maxTopLogprobs + 1, 20000, 150000} {
		_, err := lm.prepare(sampling{Logprobs: true, TopLogprobs: n(bad)}, []int{1}, false)
		if err == nil {
			t.Errorf("top_logprobs=%d accepted; one such request retains max_tokens x %d entries "+
				"and materializes a map per entry before writing a byte", bad, bad)
			continue
		}
		if !strings.Contains(err.Error(), "top_logprobs") {
			t.Errorf("top_logprobs=%d rejected for the wrong reason: %v", bad, err)
		}
	}
	// The documented range must still be accepted — OpenAI's own ceiling, so this rejects nothing
	// a compatible client sends. A fix that clamped everything would pass the loop above.
	for _, ok := range []int{0, 1, maxTopLogprobs} {
		if _, err := lm.prepare(sampling{Logprobs: true, TopLogprobs: n(ok)}, []int{1}, false); err != nil {
			t.Errorf("top_logprobs=%d rejected, but it is inside OpenAI's documented range: %v", ok, err)
		}
	}
	// Absent stays absent.
	if _, err := lm.prepare(sampling{}, []int{1}, false); err != nil {
		t.Errorf("a request with no top_logprobs was rejected: %v", err)
	}
}

// M-23: graceful shutdown cancels every in-flight generation, and `genErr` treats every
// context.Canceled as a clean end — so the truncated text came back as a 200 with
// finish_reason:"stop". The client cannot tell it from a model that finished.
//
// srvCancel() runs BEFORE Shutdown, so the client is still connected and reads a partial answer as
// complete. A client disconnect looks identical here (both arrive as a cancelled request context),
// which is why the answer is "length" rather than an error: truthful in both cases, and the signal
// a client already knows how to act on.
func TestStreamTokens_externallyCancelledIsTruncatedNotClean(t *testing.T) {
	lm := &loadedModel{tk: &tokenizer.Tokenizer{}}
	gr := genRequest{maxTokens: 64}

	// A stream that ends without the budget being reached: the "clean stop" shape.
	closed := make(chan int)
	close(closed)

	ctx, cancel := context.WithCancel(context.Background())
	finish, _, stopHit := lm.streamTokens(ctx, cancel, closed, gr, nil, func(string) {})
	if finish != "stop" || stopHit != "" {
		t.Fatalf("premise broke: an uncancelled short generation reported %q/%q, want stop/\"\"", finish, stopHit)
	}

	// Same generation, parent cancelled from outside (shutdown, or the client vanished).
	cctx, ccancel := context.WithCancel(context.Background())
	ccancel()
	closed2 := make(chan int)
	close(closed2)
	finish, _, stopHit = lm.streamTokens(cctx, ccancel, closed2, gr, nil, func(string) {})
	if stopHit != "" {
		t.Fatalf("premise broke: stopHit=%q, this case is about NO stop string", stopHit)
	}
	if finish != "length" {
		t.Errorf("finish_reason = %q after an external cancel, want \"length\". \"stop\" tells the "+
			"client the model finished when the answer was cut off mid-sentence", finish)
	}
}

// M-21: EVERY ROUTE THAT TOKENIZES MUST REJECT AN OVERSIZED BODY FIRST.
//
// G1c added `promptTooLargeForContext` to three routes and left five running a full O(n) BPE over
// an arbitrary body before rejecting it — the G1c comment prices that at "~27 s of BPE + gigabytes
// of ids" on a multi-MiB body. anthropic.go's own comment called its half "the same defect this
// release claims to fix, left half-covered on one surface", and count_tokens was the worst of them:
// it never enters the per-model queue, so up to -max-inflight (128) such tokenizations run at once.
//
// Three routes guarded and five not is a COUNTING failure, so it is counted rather than remembered.
// A route here is a handler that tokenizes: it takes a ResponseWriter and calls one of the
// tokenizing entry points. Each must call the guard, or name itself as guarded by its caller.
func TestServe_everyTokenizingRouteGuardsItsInputSize(t *testing.T) {
	// Guarded by a caller, with the reason. A private helper reached from exactly one guarded
	// handler does not need its own check — but it has to say so here rather than be forgotten.
	guardedByCaller := map[string]string{
		"respondTools": "reached only from serveResponsesWith, which guards before the tools/plain branch",
	}
	// Direct calls, plus the prompt builders that tokenize TRANSITIVELY. The vision routes were the
	// hole in the first cut of this check: serveVisionChatWith contains no tokenizer call of its
	// own — visionPrompt does — so the gate passed over the very route whose guard it was written
	// to protect, and dropping that guard produced no failure. Caught by mutation, not by reading.
	tokenizers := []string{
		"lm.chatPrompt(", "lm.tk.EncodeSegments(", "lm.tk.Encode(", "lm.encode(", "lm.promptFor(",
		"lm.visionPrompt(", "lm.qwenVisionPrompt(",
	}

	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	sort.Strings(files)
	fnRe := regexp.MustCompile(`(?ms)^func (?:\([^)]*\) )?(\w+)\(([^\n]*)\{\n(.*?)\n\}`)
	routes, unguarded := 0, []string{}
	for _, f := range files {
		if strings.HasSuffix(f, "_test.go") {
			continue
		}
		b, err := os.ReadFile(f)
		if err != nil {
			continue
		}
		for _, m := range fnRe.FindAllStringSubmatch(string(b), -1) {
			name, params, body := m[1], m[2], m[3]
			if !strings.Contains(params, "http.ResponseWriter") {
				continue // not a route: helpers that tokenize are reached through one
			}
			tokenizes := false
			for _, c := range tokenizers {
				if strings.Contains(body, c) {
					tokenizes = true
					break
				}
			}
			if !tokenizes {
				continue
			}
			routes++
			if strings.Contains(body, "promptTooLargeForContext(") {
				continue
			}
			if _, ok := guardedByCaller[name]; ok {
				continue
			}
			unguarded = append(unguarded, f+":"+name)
		}
	}
	if routes == 0 {
		t.Fatal("no tokenizing route found — the scan is broken, and a broken scan makes this " +
			"gate pass over nothing")
	}
	for _, u := range unguarded {
		t.Errorf("%s tokenizes an arbitrary request body with no promptTooLargeForContext guard: "+
			"a body under the byte cap still pays a full O(n) BPE before being rejected. That is "+
			"audit-2026-09-02 M-21, which was G1c reaching three routes of eight; this is the "+
			"check that counts them.", u)
	}
	for name, why := range guardedByCaller {
		if strings.TrimSpace(why) == "" {
			t.Errorf("%s claims a caller guards it and gives no reason", name)
		}
	}
	t.Logf("%d tokenizing route(s), all guarded (%d by a caller)", routes, len(guardedByCaller))
}

// C-07, the serving half: the decoder-as-embedder must TRUNCATE to the model's context window.
//
// C-21 capped one input at 1 MiB of BYTES and left the token count unbounded, so ~1 MiB of short
// words is ~500k tokens: HiddenLast preallocates KV for every one of them and runs a sequential
// per-token forward with no context, under the embed mutex, until the process is OOM-killed. The
// decoder-side guard (TestHiddenLast_refusesMoreTokensThanTheContextWindow) makes that an error
// rather than an OOM — but an ERROR is not the right answer for the embedder, which should do what
// HF's truncation=True does and what the aikit encoder path already did. So both halves are pinned:
// the decoder refuses, and this one never sends it more than it can take.
func TestDecoderEmbedder_truncatesToTheContextWindow(t *testing.T) {
	const window = 16
	e := &decoderEmbedder{
		maxTokens: window, // what loadDecoderEmbedder now sets from m.Config().MaxPositions
		appendID:  -1,
	}
	if e.maxTokens == 0 {
		t.Fatal("premise broke: maxTokens 0 is the unbounded state this test is about")
	}
	// tokenize() needs a tokenizer; the truncation arithmetic is what matters and is exercised
	// directly, since a real BPE would only obscure the boundary.
	ids := make([]int, window*4)
	room := e.maxTokens
	if e.appendID >= 0 {
		room--
	}
	if room > 0 && len(ids) > room {
		ids = ids[:room]
	}
	if len(ids) != window {
		t.Fatalf("truncated to %d, want %d", len(ids), window)
	}

	// With an appended pooling token, the slot must be RESERVED — the appended token has to stay
	// last because it is the pooled position, so truncation must not be what drops it.
	e2 := &decoderEmbedder{maxTokens: window, appendID: 7}
	ids2 := make([]int, window*4)
	room2 := e2.maxTokens
	if e2.appendID >= 0 {
		room2--
	}
	if room2 > 0 && len(ids2) > room2 {
		ids2 = ids2[:room2]
	}
	ids2 = append(ids2, e2.appendID)
	if len(ids2) != window {
		t.Errorf("with an appended token the total is %d, want %d", len(ids2), window)
	}
	if ids2[len(ids2)-1] != 7 {
		t.Error("the appended pooling token is not last; truncation dropped the pooled position")
	}
}

// The wiring itself: loadDecoderEmbedder must set maxTokens from the model, not leave it 0.
// The arithmetic test above passes for any non-zero bound, so this is what pins the SOURCE of it.
func TestDecoderEmbedder_boundComesFromTheModelNotZero(t *testing.T) {
	b, err := os.ReadFile("decoder_embedder.go")
	if err != nil {
		t.Fatal(err)
	}
	src := string(b)
	if strings.Contains(src, "maxTokens: 0,") {
		t.Error("loadDecoderEmbedder still constructs the embedder with maxTokens: 0 — the token " +
			"count is unbounded and only C-21's 1 MiB byte cap stands between a request and ~500k " +
			"positions of sequential forward under the embed mutex (audit-2026-09-02 C-07)")
	}
	if !strings.Contains(src, "maxTokens: m.Config().MaxPositions,") {
		t.Error("the embedder's token bound no longer comes from the model's context window; " +
			"m.Config().MaxPositions is max_position_embeddings, and NOT Architecture.MaxPositions, " +
			"which is the GPT-2 learned-position table and 0 for every RoPE family")
	}
}

// N-16: BOTH decoders must shape a JSON error the same way, and neither may echo the raw one.
//
// M-06 and R-11 removed the Go struct/field/type leak from decodeJSON; decodeAnthropicJSON still
// appended err.Error() verbatim, so "invalid request body: json: cannot unmarshal string into Go
// struct field anthropicReq.max_tokens of type int" went straight to the client. §0 theme 2 again:
// one route hardened, its twin not.
func TestJSONDecodeMessage_neverLeaksGoTypeNames(t *testing.T) {
	type inner struct {
		N int `json:"n"`
	}
	type req struct {
		Flag     bool            `json:"flag"`
		Messages []inner         `json:"messages"`
		Raw      json.RawMessage `json:"raw"`
	}
	for name, body := range map[string]string{
		"scalar field":    `{"flag": "yes"}`,
		"composite field": `{"messages": "nope"}`,
		"malformed":       `{"flag":`,
	} {
		var v req
		err := json.Unmarshal([]byte(body), &v)
		if err == nil {
			t.Fatalf("%s: premise broke, %s decoded cleanly", name, body)
		}
		msg, tooLarge := jsonDecodeMessage(err)
		if tooLarge {
			t.Errorf("%s: reported as a body-size failure", name)
		}
		for _, leak := range []string{"Go struct", "serveapp.", "json:", "*serveapp"} {
			if strings.Contains(msg, leak) {
				t.Errorf("%s: message leaks internals (%q): %s", name, leak, msg)
			}
		}
		if msg == "" {
			t.Errorf("%s: empty message", name)
		}
	}
}

// The Anthropic decoder must USE it. The test above proves the shaper is clean and says nothing
// about the call site — the same gap that let M-21's vision route and M-22's call site pass.
func TestAnthropic_decodeUsesTheSharedShaping(t *testing.T) {
	b, err := os.ReadFile("anthropic.go")
	if err != nil {
		t.Fatal(err)
	}
	src := string(b)
	if strings.Contains(src, `"invalid request body: "+err.Error()`) {
		t.Error("decodeAnthropicJSON still appends the raw json error, which re-leaks the Go " +
			"struct/field/type names M-06 and R-11 removed from the OpenAI surface (N-16)")
	}
	if !strings.Contains(src, "jsonDecodeMessage(err)") {
		t.Error("decodeAnthropicJSON no longer routes through jsonDecodeMessage; the two surfaces " +
			"can drift again")
	}
}

// N-21: session persistence is the CONVERSATION. A .giw-kv blob replays what the user said and what
// the model answered, and it was written 0o644 inside a 0o755 directory — readable by every local
// account. The directory matters as much as the files: a readable one lists the session ids.
func TestSessions_persistedStateIsOwnerOnly(t *testing.T) {
	if sessionFilePerm != 0o600 {
		t.Errorf("session blobs are written %#o, want 0o600", sessionFilePerm)
	}
	if sessionDirPerm != 0o700 {
		t.Errorf("the session directory is created %#o, want 0o700", sessionDirPerm)
	}
	b, err := os.ReadFile("sessions.go")
	if err != nil {
		t.Fatal(err)
	}
	// Code lines only. The comment above the constants NAMES the old modes, and a substring scan
	// over the whole file matches its own explanation of the defect — the second time a check in
	// this batch did that, so it is worth doing deliberately rather than rediscovering.
	for _, line := range strings.Split(string(b), "\n") {
		if strings.HasPrefix(strings.TrimSpace(line), "//") {
			continue
		}
		for _, world := range []string{"0o644", "0o755", "0o666", "0o777"} {
			if strings.Contains(line, world) {
				t.Errorf("sessions.go still writes something %s — a KV blob is the conversation: %s",
					world, strings.TrimSpace(line))
			}
		}
	}
}

// M-24: two concurrent admin loads of the same name leaked the loser's model.
//
// loadDecoder runs OUTSIDE the registry lock (deliberately — it takes seconds), so two
// requests for the same name both load, and the post-check refuses to publish the second.
// The refused *loadedModel holds resident device memory, the .giw mmap and an uploaded block
// drafter, and dropping the pointer released none of it: purego installs no finalizers, which
// is the whole reason the drain design exists.
//
// Asserted on the SOURCE because the leak has no observable behaviour to test — nothing
// errors, nothing is slower, the memory is simply never returned. A test that loaded two real
// models to watch RSS would be measuring Darwin's UBC rather than the defect (the audit's own
// note about RSS-based guards inverting under pressure).
func TestAdminLoad_racedDuplicateIsClosed(t *testing.T) {
	fset := token.NewFileSet()
	af, err := parser.ParseFile(fset, "admin.go", nil, 0)
	if err != nil {
		t.Fatalf("parse admin.go: %v", err)
	}
	// The 409 branch that fires AFTER loadDecoder — identified by its body writing a
	// StatusConflict error, in a block that also unlocks regMu (the pre-check does not).
	var checked int
	ast.Inspect(af, func(n ast.Node) bool {
		blk, ok := n.(*ast.BlockStmt)
		if !ok {
			return true
		}
		// DIRECT statements only. Matching nested text made the enclosing function body
		// match as well, since it contains this block — a guard counting 2 where it meant 1.
		var unlocks, conflict bool
		for _, st := range blk.List {
			es, ok := st.(*ast.ExprStmt)
			if !ok {
				continue
			}
			src := &strings.Builder{}
			_ = printer.Fprint(src, fset, es)
			if strings.Contains(src.String(), "regMu.Unlock()") {
				unlocks = true
			}
			if strings.Contains(src.String(), "StatusConflict") {
				conflict = true
			}
		}
		if !unlocks || !conflict {
			return true
		}
		checked++
		var body strings.Builder
		_ = printer.Fprint(&body, fset, blk)
		for _, want := range []string{"model.Close()", "closeEntryNatives()"} {
			if !strings.Contains(body.String(), want) {
				t.Errorf("the raced-duplicate 409 path does not call %s: the loser's fully "+
					"loaded model is dropped without releasing device memory, the .giw mmap or "+
					"the block drafter, and purego has no finalizers to collect them (M-24)", want)
			}
		}
		return true
	})
	if checked != 1 {
		t.Errorf("found %d post-load 409 branches, want 1 — the guard is watching the wrong "+
			"code (the PRE-load duplicate check must not match: it has nothing to close)", checked)
	}
}

// N-17: response/message/tool-call ids were a SEQUENTIAL counter. Seeding it from UnixNano hid
// that without fixing it — each id is exactly one more than the previous, so a client holding
// its own `resp_<hex>` can walk ±1 onto other clients' ids. `previous_response_id` continues a
// stored conversation from an id, so with `-addr 0.0.0.0` and one shared key that reads back
// someone else's turns.
func TestReqID_isNotGuessable(t *testing.T) {
	const n = 512
	seen := make(map[string]bool, n)
	ids := make([]string, 0, n)
	for range n {
		id := reqID()
		if seen[id] {
			t.Fatalf("duplicate id %q in %d draws", id, n)
		}
		seen[id] = true
		ids = append(ids, id)
	}
	if len(ids[0]) < 24 {
		t.Errorf("id %q is %d chars — too short to resist enumeration", ids[0], len(ids[0]))
	}
	// THE PROPERTY THAT WAS BROKEN: consecutive ids must not be consecutive integers. Parsed as
	// big-endian hex, a counter gives a difference of exactly 1 every time.
	var consecutive int
	for i := 1; i < len(ids); i++ {
		a, ok1 := new(big.Int).SetString(ids[i-1], 16)
		b, ok2 := new(big.Int).SetString(ids[i], 16)
		if !ok1 || !ok2 {
			t.Fatalf("ids are not hex: %q %q", ids[i-1], ids[i])
		}
		if new(big.Int).Sub(b, a).CmpAbs(big.NewInt(1)) == 0 {
			consecutive++
		}
	}
	if consecutive > 0 {
		t.Errorf("%d of %d consecutive id pairs differ by exactly 1 — the ids are a counter, so "+
			"holding one id gives you the next (N-17)", consecutive, len(ids)-1)
	}
}

// N-20: demoteLoop mutates the session LRU on its own ticker while the unload drain saved it
// without holding lm.mu. The drain waits out in-flight REQUESTS, which is a different thing —
// the idle-demote goroutine is not a request and holds no ml.rw. sessions.go documents the LRU
// as not goroutine-safe, so this is a concurrently-mutated map: a process crash, not a wrong
// answer.
//
// Asserted on the source. Reproducing it needs -kv-idle-demote and -session-dir and an unload
// landing inside a tick, and a test that raced for it would be flaky in the direction that
// reports success.
func TestStartDrain_savesSessionsUnderTheLRULock(t *testing.T) {
	src, err := os.ReadFile("liveness.go")
	if err != nil {
		t.Fatalf("read liveness.go: %v", err)
	}
	lines := strings.Split(string(src), "\n")
	var found bool
	for i, ln := range lines {
		if !strings.Contains(ln, "lm.sessions.save(") {
			continue
		}
		found = true
		// lm.mu must be held: look for the Lock in the few lines above.
		var locked bool
		for j := i - 1; j >= 0 && j >= i-6; j-- {
			if strings.Contains(lines[j], "lm.mu.Lock()") {
				locked = true
				break
			}
		}
		if !locked {
			t.Errorf("liveness.go:%d saves the session LRU without taking lm.mu — demoteLoop "+
				"mutates the same map on its ticker (N-20):\n\t%s", i+1, strings.TrimSpace(ln))
		}
	}
	if !found {
		t.Error("no lm.sessions.save( in liveness.go — this guard is watching the wrong file")
	}
}

// N-19: admin-load set c.quant from the request but inherited c.quantSet from the CLI, and
// explicitQuant() — which drives the .giw baked-quant mismatch check — reads quantSet. Both
// directions were wrong, and they are opposite failures, so a fix has to be checked both ways:
//
//	no CLI --quant + admin asks int8  → not treated as explicit → the bundle's baked int4
//	                                    loads SILENTLY under an int8 request
//	CLI --quant given + admin asks nothing → treated as explicit → the request is rejected
//	                                    against a quant it never named
func TestAdminLoad_requestQuantIsTheExplicitOne(t *testing.T) {
	for name, tc := range map[string]struct {
		cliQuant    string
		cliQuantSet bool
		reqQuant    string
		want        string // what explicitQuant should report for this load
	}{
		"request names a quant, no CLI default": {"", false, "int8", "int8"},
		"request names a quant, CLI differs":    {"int4", true, "int8", "int8"},
		"request names nothing, CLI was passed": {"int4", true, "", ""},
		"request names nothing, no CLI either":  {"", false, "", ""},
	} {
		t.Run(name, func(t *testing.T) {
			c := config{quant: tc.cliQuant, quantSet: tc.cliQuantSet}
			// The admin handler's resolution, verbatim.
			if tc.reqQuant != "" {
				c.quant, c.quantSet = tc.reqQuant, true
			} else {
				c.quantSet = false
			}
			if got := (modelSpec{}).explicitQuant(c); got != tc.want {
				t.Errorf("explicitQuant = %q, want %q — the .giw mismatch check keys on this, so "+
					"a wrong answer either loads a differently-baked bundle silently or rejects "+
					"a request that named nothing (N-19)", got, tc.want)
			}
		})
	}

	// And the handler must actually do that resolution — the table above is the RULE, this is
	// the call site. Same gap that made M-25's component test vouch for nothing.
	src, err := os.ReadFile("admin.go")
	if err != nil {
		t.Fatalf("read admin.go: %v", err)
	}
	if !strings.Contains(string(src), "c.quant, c.quantSet = req.Quant, true") {
		t.Error("admin.go does not set quantSet from the request: c.quant moves and quantSet " +
			"stays inherited from the CLI, which is N-19 exactly")
	}
}

// N-18: a NAMED tool_choice whose function is not in tools decoded completely unconstrained,
// having asked for one specific function. The 2026-08-05 audit made "named but unconstrainable"
// a 400 and left "named but nonexistent" falling through — the louder of the two, since it
// means a typo or a stale tool list and the prose answer looks like a free choice.
func TestConstrainForcedTool_namedButNonexistentIs400(t *testing.T) {
	tools := []chat.Tool{{Name: "get_weather"}, {Name: "get_time"}}
	lm := &loadedModel{tmpl: chat.ChatML(), vocab: 32, tk: &tokenizer.Tokenizer{}}

	err := constrainForcedTool(lm, &genRequest{}, nil, true, tools)
	if err == nil {
		t.Fatal("a named tool_choice matching no tool returned nil (N-18)")
	}
	// The message must name what IS available, or the caller cannot tell a typo from a
	// server-side omission.
	for _, want := range []string{"get_weather", "get_time"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not list the available tool %q", err, want)
		}
	}
	// Not an error when no specific function was named — that is just "no lone-tool shortcut".
	if err := constrainForcedTool(lm, &genRequest{}, nil, false, tools); err != nil {
		t.Errorf("unnamed choice with no forced tool errored: %v", err)
	}
}
