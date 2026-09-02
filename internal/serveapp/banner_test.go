package serveapp

import (
	"strings"
	"testing"
)

func bannerLine(lines []string, prefix string) string {
	for _, l := range lines {
		if strings.HasPrefix(l, prefix) {
			return l
		}
	}
	return ""
}

// TestBanner_sessionReuseMatchesTheDecodePath is G6's core assertion, and the one worth
// having: the banner may claim prefix reuse ONLY when the server would actually use it.
//
// decoder.Generate engages the resident DecodeRunner only when there is no session commit and
// no prefix reuse — the resident KV lives on the GPU while a session's prefix cache is
// CPU-side, and the two cannot both be the source of truth (loadedModel.drive). So a resident
// model is stateless and re-prefills every turn, whatever --kv-sessions says. A banner that
// reported "session reuse: on" there would be telling an agent-loop author the opposite of
// what their latency will do.
func TestBanner_sessionReuseMatchesTheDecodePath(t *testing.T) {
	for _, tc := range []struct {
		name       string
		resident   bool
		kvSessions int
		wantReuse  bool
	}{
		{"resident, sessions configured", true, 4, false}, // residency wins: stateless
		{"resident, sessions off", true, 0, false},
		{"staged, sessions configured", false, 4, true},
		{"staged, sessions off", false, 0, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			lines := modelBannerFrom(bannerFacts{resident: tc.resident, hasTemplate: true}, config{kvSessions: tc.kvSessions})
			line := bannerLine(lines, "session reuse:")
			if line == "" {
				t.Fatal("no session-reuse line in the banner")
			}
			// The server reuses a prefix iff it is NOT resident and sessions are configured.
			serverWouldReuse := !tc.resident && tc.kvSessions > 0
			if serverWouldReuse != tc.wantReuse {
				t.Fatalf("test table is wrong about the server's own rule")
			}
			saysOn := !strings.Contains(line, "OFF")
			if saysOn != serverWouldReuse {
				t.Errorf("banner says %q but the server would reuse=%v", line, serverWouldReuse)
			}
			// When it is off for a reason the user did not choose, the banner must say WHY —
			// an unexplained OFF sends someone to the flag they already set.
			if tc.resident && !strings.Contains(line, "re-prefills") {
				t.Errorf("resident OFF must explain the cost, got %q", line)
			}
		})
	}
}

// TestBanner_reportsResolvedNotRequested: the decode/prefill lines must carry what the model
// actually resolved to, not what was asked for. These fall back SILENTLY, which is the whole
// reason they are printed.
func TestBanner_reportsResolvedNotRequested(t *testing.T) {
	f := bannerFacts{decodePath: "cpu (int4)", prefillPath: "sequential — no batched prefill", hasTemplate: true}
	lines := modelBannerFrom(f, config{backend: "cuda"}) // asked for cuda, resolved to cpu
	if got := bannerLine(lines, "decode path:"); !strings.Contains(got, "cpu (int4)") {
		t.Errorf("decode line must report the RESOLVED path, got %q", got)
	}
	if strings.Contains(strings.Join(lines, "\n"), "cuda") {
		t.Error("the banner must not echo the REQUESTED backend — that is the fallback this line exists to expose")
	}
}

// TestBanner_contextAndKV: the two numbers that decide whether a harness's turn fits, and
// whether the KV is lossy.
func TestBanner_contextAndKV(t *testing.T) {
	for _, tc := range []struct {
		ctx        int
		kv         string
		wantSubstr []string
	}{
		{0, "", []string{"backend default", "KV f32"}},
		{4096, "f32", []string{"4096 tokens", "KV f32"}},
		{32768, "f16", []string{"32768 tokens", "KV f16", "lossy"}},
		{65536, "i8", []string{"65536 tokens", "KV i8", "lossy"}},
	} {
		lines := modelBannerFrom(bannerFacts{hasTemplate: true}, config{ctxSize: tc.ctx, kvPrec: tc.kv})
		got := bannerLine(lines, "context:")
		for _, want := range tc.wantSubstr {
			if !strings.Contains(got, want) {
				t.Errorf("ctx=%d kv=%q: line %q missing %q", tc.ctx, tc.kv, got, want)
			}
		}
	}
}

// TestBanner_toolSupportIsNotAssumed: M-20 (gemma-4's parse-only tool form) is exactly the
// case a harness must know about BEFORE it sends a named tool_choice and gets a 400.
func TestBanner_toolSupportIsNotAssumed(t *testing.T) {
	on := bannerLine(modelBannerFrom(bannerFacts{hasTemplate: true, toolCallForm: true}, config{}), "features:")
	if !strings.Contains(on, "tools (constrainable") {
		t.Errorf("a constrainable template should advertise tools, got %q", on)
	}
	off := bannerLine(modelBannerFrom(bannerFacts{hasTemplate: true, toolCallForm: false}, config{}), "features:")
	if !strings.Contains(off, "parse-only") {
		t.Errorf("a non-constrainable template must NOT claim plain tool support, got %q", off)
	}
	none := bannerLine(modelBannerFrom(bannerFacts{hasTemplate: false}, config{}), "features:")
	if !strings.Contains(none, "raw completion") {
		t.Errorf("no template must be stated, got %q", none)
	}
}

// TestServerBanner_routesMatchRegistration: the routes line must list the web routes exactly
// when -web registered them. Route lists drift from route registration; that is the M-07
// class applied to the one document every user reads.
func TestServerBanner_routesMatchRegistration(t *testing.T) {
	s := &server{models: map[string]*loadedModel{"m": {}}}
	withWeb := strings.Join(serverBanner(s, config{web: true}), "\n")
	if !strings.Contains(withWeb, "(web UI)") {
		t.Error("-web must appear in the routes line")
	}
	if strings.Contains(withWeb, "web UI: off") {
		t.Error("with -web on, the 'off' hint must not print")
	}
	without := strings.Join(serverBanner(s, config{}), "\n")
	if strings.Contains(without, "(web UI)") {
		t.Error("without -web the route must NOT be advertised — it returns 404")
	}
	if !strings.Contains(without, "-web enables") {
		t.Error("without -web, say how to turn it on")
	}
	// With no generative model loaded, the chat routes are not registered and must not be listed.
	empty := strings.Join(serverBanner(&server{}, config{}), "\n")
	if strings.Contains(empty, "/v1/chat/completions") {
		t.Error("no model ⇒ the chat routes are not registered, so they must not be advertised")
	}
}
