package serveapp

import (
	"fmt"
	"github.com/townsendmerino/goinfer/internal/loadflags"
	"os"
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
	lines := modelBannerFrom(f, config{load: loadflags.Flags{Backend: "cuda"}}) // asked for cuda, resolved to cpu
	if got := bannerLine(lines, "decode path:"); !strings.Contains(got, "cpu (int4)") {
		t.Errorf("decode line must report the RESOLVED path, got %q", got)
	}
	if strings.Contains(strings.Join(lines, "\n"), "cuda") {
		t.Error("the banner must not echo the REQUESTED backend — that is the fallback this line exists to expose")
	}
}

// TestBanner_contextAndKV: the two numbers that decide whether a harness's turn fits, and
// whether the KV is lossy. The context is the RESOLVED limit, never the request: "backend default"
// told a user nothing, and printing a -ctx above the model's maximum reported a limit that did not exist.
func TestBanner_contextAndKV(t *testing.T) {
	for _, tc := range []struct {
		name        string
		window, max int
		ctx         int
		kv          string
		want        []string
		notWant     []string
		resKV       string // the resident runner's own KV precision ("" = not resident)
	}{
		{"no -ctx, resident cap below the model maximum", 8192, 40960, 0, "", []string{"context: 8192 tokens (backend default; model maximum 40960 — raise with --ctx)", "KV f32"}, []string{"backend default ·"}, ""},
		{"no -ctx, model maximum binds", 32768, 32768, 0, "", []string{"context: 32768 tokens (model maximum)"}, []string{"backend default"}, ""},
		{"-ctx below the model maximum", 4096, 40960, 4096, "f32", []string{"context: 4096 tokens (--ctx; model maximum 40960)", "KV f32"}, nil, ""},
		{"-ctx above the model maximum", 8192, 8192, 100000, "f16", []string{"context: 8192 tokens (model maximum; --ctx 100000 is above it)", "KV f16", "lossy"}, []string{"100000 tokens"}, ""},
		{"-ctx equal to the model maximum", 65536, 65536, 65536, "i8", []string{"context: 65536 tokens (model maximum)", "KV i8", "lossy"}, nil, ""},
		{"unknown", 0, 0, 0, "", []string{"context: unknown"}, nil, ""},
		// Metal allocates f16 KV whatever -kv says: the banner reports what runs, not the request.
		{"metal resident, no -kv", 4096, 32768, 0, "", []string{"KV f16"}, []string{"KV f32", "lossy"}, "f16"},
		{"metal resident, -kv f32 requested", 4096, 32768, 0, "f32", []string{"KV f16"}, []string{"KV f32"}, "f16"},
		{"resident, -kv i8 requested and applied", 4096, 32768, 0, "i8", []string{"KV i8", "lossy"}, nil, "i8"},
	} {
		lines := modelBannerFrom(bannerFacts{hasTemplate: true, ctxWindow: tc.window, maxPositions: tc.max, kvPrec: tc.resKV}, config{load: loadflags.Flags{Ctx: tc.ctx, KV: tc.kv}})
		got := bannerLine(lines, "context:")
		for _, want := range tc.want {
			if !strings.Contains(got, want) {
				t.Errorf("%s: line %q missing %q", tc.name, got, want)
			}
		}
		for _, bad := range tc.notWant {
			if strings.Contains(got, bad) {
				t.Errorf("%s: line %q must not contain %q", tc.name, got, bad)
			}
		}
	}
}

// TestBanner_contextIsTheEnforcedWindow ties the banner to the limit itself, through the real
// factsOf on a loaded model: the number printed is contextWindow's — the same one prepare enforces
// and /v1/models publishes — not a value re-derived from flags.
func TestBanner_contextIsTheEnforcedWindow(t *testing.T) {
	_, lm := tinyServed(t)
	f := factsOf(lm)
	want := lm.contextWindow(lm.adapter == "")
	if want <= 0 || f.ctxWindow != want || f.maxPositions != lm.model.Config().MaxPositions {
		t.Fatalf("factsOf: ctxWindow=%d maxPositions=%d, want %d / %d", f.ctxWindow, f.maxPositions, want, lm.model.Config().MaxPositions)
	}
	got := bannerLine(modelBanner(lm, config{}), "context:")
	if !strings.Contains(got, fmt.Sprintf("context: %d tokens", want)) {
		t.Errorf("banner context line %q does not state the enforced window %d", got, want)
	}
	// On cpu the resident cap does not exist, so window == MaxPositions and a factsOf that read
	// MaxPositions directly would pass the check above. The GPU case — where they differ, 8192 against
	// 40960 on the 2070 — is held by the source: factsOf must take the window from contextWindow, and
	// the load path must hand the banner the -ctx THIS model resolved (a per-model ctx= override wins).
	src, err := os.ReadFile("banner.go")
	if err != nil {
		t.Fatal(err)
	}
	facts := string(src)[strings.Index(string(src), "func factsOf("):]
	if !strings.Contains(facts[:strings.Index(facts, "\n}\n")], "f.ctxWindow = lm.contextWindow(lm.adapter == \"\")") {
		t.Error("factsOf no longer takes ctxWindow from lm.contextWindow — the banner can drift from the enforced limit")
	}
	mainSrc, err := os.ReadFile("main.go")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(mainSrc), "bannerCfg.load.Ctx = opts.ResidentContext") {
		t.Error("the load path no longer passes the model's resolved -ctx to the banner — a per-model ctx= would be misreported")
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
