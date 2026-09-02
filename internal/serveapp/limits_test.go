package serveapp

import (
	"context"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
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
