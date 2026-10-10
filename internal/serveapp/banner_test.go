package serveapp

import (
	"fmt"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/loadflags"
	"os"
	"path/filepath"
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

// TestBanner_sessionReuseMatchesTheDecodePath: the banner may claim prefix reuse ONLY when the server would
// actually use it. decoder.Generate engages the resident DecodeRunner only when there is no session commit and
// no prefix reuse (loadedModel.drive): the resident KV lives on the GPU and a session's prefix cache is
// CPU-side, so a resident model re-prefills every turn whatever --kv-sessions says. A banner reporting
// "session reuse: on" there would tell an agent-loop author the opposite of what their latency will do.
func TestBanner_sessionReuseMatchesTheDecodePath(t *testing.T) {
	for _, tc := range []struct {
		name       string
		resident   bool
		reuseOff   bool // GOINFER_NO_RESIDENT_REUSE
		kvSessions int
		wantReuse  bool
		kvSlots    int  // resident KV slots allocated (MC1); 0/1 = one conversation
		byDefault  bool // -kv-sessions not given (E-P09: Metal keeps 2 then)
	}{
		// A resident model reuses the most recent conversation's prefix on the device (decoder/resident_reuse.go);
		// the CPU session LRU does not apply to it, so --kv-sessions does not change the answer.
		{"resident, sessions configured", true, false, 4, true, 0, false},
		{"resident, sessions off", true, false, 0, true, 0, false},
		{"resident, resident reuse switched off", true, true, 4, false, 0, false},
		{"staged, sessions configured", false, false, 4, true, 0, false},
		{"staged, sessions off", false, false, 0, false, 0, false},
		{"resident, 4 KV slots", true, false, 4, true, 4, false},
		{"resident, 4 asked, guard allowed 2", true, false, 4, true, 2, false},
		{"resident, -kv-sessions not given, Metal's default 2", true, false, 4, true, 2, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			lines := modelBannerFrom(bannerFacts{resident: tc.resident, residentReuseOff: tc.reuseOff, hasTemplate: true, kvSlots: tc.kvSlots, decodePath: "metal-resident (int4)"}, config{kvSessions: tc.kvSessions, kvSessionsSet: !tc.byDefault})
			line := bannerLine(lines, "session reuse:")
			if line == "" {
				t.Fatal("no session-reuse line in the banner")
			}
			saysOn := !strings.Contains(line, "OFF")
			if saysOn != tc.wantReuse {
				t.Errorf("banner says %q, want reuse=%v", line, tc.wantReuse)
			}
			// On the resident path reuse is ONE conversation; the line must say so, or an operator
			// running several agents expects the LRU's behaviour and gets full re-prefills.
			if tc.resident && tc.wantReuse && tc.kvSlots <= 1 && (!strings.Contains(line, "one conversation") || !strings.Contains(line, "re-prefills")) {
				t.Errorf("resident reuse must say it covers one conversation and what switching costs, got %q", line)
			}
			// With several resident KV slots (MC1) it must say how many, and name the clamp when the guard allowed fewer.
			if tc.resident && tc.kvSlots > 1 {
				if !strings.Contains(line, fmt.Sprintf("%d conversations", tc.kvSlots)) {
					t.Errorf("resident with %d KV slots must say so, got %q", tc.kvSlots, line)
				}
				if tc.kvSessions > tc.kvSlots && !tc.byDefault && !strings.Contains(line, fmt.Sprintf("allowed %d", tc.kvSlots)) {
					t.Errorf("a clamped slot count must be named, got %q", line)
				}
				// Fewer slots than -kv-sessions' default, with the flag not given: the line must not blame the memory guard
				// alone, and must say how to ask for more.
				if tc.byDefault && tc.kvSessions > tc.kvSlots && (!strings.Contains(line, "by default") || !strings.Contains(line, fmt.Sprintf("--kv-sessions %d asks", tc.kvSessions))) {
					t.Errorf("a default slot count must say it is the default and how to ask for more, got %q", line)
				}
			}
			// Off for a reason the user did not type as a flag: the line must name it.
			if tc.resident && tc.reuseOff && !strings.Contains(line, "GOINFER_NO_RESIDENT_REUSE") {
				t.Errorf("resident reuse OFF must name why, got %q", line)
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
	// On cpu the resident cap does not exist, so window == MaxPositions and a factsOf that read MaxPositions
	// directly would pass the check above. The GPU case, where they differ, is held by the source: factsOf must
	// take the window from contextWindow, and the load path must hand the banner the -ctx THIS model resolved (a
	// per-model ctx= override wins).
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
	// With no generative model loaded, the chat routes ARE registered (a model can be loaded later —
	// TestServe_generationRoutesExistWithoutAStartupModel), so they are listed, with the condition.
	empty := strings.Join(serverBanner(&server{}, config{}), "\n")
	if !strings.Contains(empty, "/v1/chat/completions") || !strings.Contains(empty, "after a model is loaded") {
		t.Errorf("no model yet: the chat routes are registered and answer once a model loads — list them with that condition:\n%s", empty)
	}
	if strings.Contains(withWeb, "after a model is loaded") {
		t.Error("with a model loaded, the routes line must not carry the not-yet-loaded condition")
	}
}

// TestBanner_concurrency: the banner says how many generations run at once, names a -kv-sessions cap, and says why a
// model runs one when -max-concurrent asked for more (MC3c).
func TestBanner_concurrency(t *testing.T) {
	for _, tc := range []struct {
		name       string
		concurrent int
		cfg        config
		want       string // substring; "" = no concurrency line
	}{
		{"default", 1, config{}, ""},
		{"four", 4, config{maxConcurrent: 4, kvSessions: 4}, "4 generations at once"},
		{"capped", 2, config{maxConcurrent: 4, kvSessions: 2}, "capped by --kv-sessions 2"},
		{"asked, not eligible", 1, config{maxConcurrent: 4, kvSessions: 4}, "applies to CPU models"},
		{"resident, batched", 4, config{maxConcurrent: 4, kvSessions: 4}, "resident KV slot, decode tokens batched"},
		{"resident, batched, chunked prefill", 4, config{maxConcurrent: 4, kvSessions: 4, prefillChunk: 512}, "prefills in 512-token chunks"},
		{"resident, capped by slots", 2, config{maxConcurrent: 4, kvSessions: 2}, "capped by the 2 resident KV slots"},
		{"resident, cannot batch", 1, config{maxConcurrent: 4, kvSessions: 4}, "needs a resident that batches decode"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			resident := strings.HasPrefix(tc.name, "resident")
			f := bannerFacts{hasTemplate: true, concurrent: tc.concurrent, resident: resident, kvSlots: tc.cfg.kvSessions}
			if l := bannerLine(modelBannerFrom(f, tc.cfg), "concurrency:"); l != "" {
				t.Errorf("the load-time banner printed %q — concurrency is decided after load (setConcurrency)", l)
			}
			line := concurrencyLine(f, tc.cfg)
			if tc.want == "" {
				if line != "" {
					t.Errorf("unexpected concurrency line %q", line)
				}
				return
			}
			if !strings.Contains(line, tc.want) {
				t.Errorf("concurrency line %q, want it to contain %q", line, tc.want)
			}
		})
	}
}

// TestConcurrencyLine_cpuBatch pins the banner's MC3c step-2 wording: a CPU model whose concurrent generations batch
// their decode tokens says so and names -cpu-batch; one left on independent workers under -cpu-batch auto says why.
func TestConcurrencyLine_cpuBatch(t *testing.T) {
	batched := concurrencyLine(bannerFacts{concurrent: 4, cpuBatched: true}, config{maxConcurrent: 4, kvSessions: 4})
	if !strings.Contains(batched, "decode tokens batched on the CPU") || !strings.Contains(batched, "-cpu-batch") {
		t.Errorf("batched CPU model: %q", batched)
	}
	workers := concurrencyLine(bannerFacts{concurrent: 4}, config{maxConcurrent: 4, kvSessions: 4, cpuBatch: decoder.CPUBatchAuto})
	if !strings.Contains(workers, "independent workers") || strings.Contains(workers, "batched on the CPU") {
		t.Errorf("CPU model on the workers under auto: %q", workers)
	}
	off := concurrencyLine(bannerFacts{concurrent: 4}, config{maxConcurrent: 4, kvSessions: 4, cpuBatch: decoder.CPUBatchOff})
	if strings.Contains(off, "independent workers") || strings.Contains(off, "batched") {
		t.Errorf("-cpu-batch off: %q (want step 1's line unchanged)", off)
	}
}

// R20: on the CPU path the context figure is the model's whole maximum and the "fit:" KV cost is a ceiling, not memory held; the banner says so beside the
// figures, and — when a GPU backend declined — that a GPU-resident load would have capped it. Neither note appears for a resident model, nor where the window
// is below the model's maximum.
func TestBanner_cpuContextIsACeiling(t *testing.T) {
	const note = "the CPU path has no --ctx cap: KV is allocated per request, so this is a ceiling, not memory held"
	base := bannerFacts{hasTemplate: true, ctxWindow: 262144, maxPositions: 262144, fitKnown: true, fitCtx: 262144, fitKVBytes: 10 << 30, fitWeightBytes: 13 << 30, fitBudgetBytes: 25 << 30}
	line := func(f bannerFacts, prefix string) string { return bannerLine(modelBannerFrom(f, config{}), prefix) }

	cpu := line(base, "context:")
	if !strings.Contains(cpu, "262144 tokens (model maximum)") || !strings.Contains(cpu, note) || strings.Contains(cpu, "declined") {
		t.Errorf("plain CPU load: %q", cpu)
	}
	if fit := line(base, "fit:"); !strings.Contains(fit, "a ceiling, reached only by a request that fills the window") {
		t.Errorf("CPU fit line: %q", fit)
	}

	declined := base
	declined.residentDecline = "cuda: device allocation failed"
	d := line(declined, "context:")
	if !strings.Contains(d, note) || !strings.Contains(d, "a GPU-resident load caps it with --ctx, but this model declined one (see decode path)") {
		t.Errorf("CPU fallback after a decline: %q", d)
	}

	res := base
	res.resident = true
	if got := line(res, "context:"); strings.Contains(got, "CPU path") || strings.Contains(line(res, "fit:"), "ceiling") {
		t.Errorf("a resident model got the CPU note: %q / %q", got, line(res, "fit:"))
	}
	below := base
	below.ctxWindow, below.fitCtx = 8192, 8192
	if got := line(below, "context:"); strings.Contains(got, "CPU path") || strings.Contains(line(below, "fit:"), "ceiling") {
		t.Errorf("a window below the model maximum got the CPU note: %q", got)
	}
}

// Through the caller: factsOf reads the decline off a real model whose backend declined residency, so the note's second half is not a constant of the test.
func TestFactsOf_carriesTheResidentDecline(t *testing.T) {
	p := filepath.Join("..", "..", "testdata", "glm-tiny.gguf")
	if _, err := os.Stat(p); err != nil {
		t.Skipf("no committed tiny fixture at %s", p)
	}
	name := "fake-declines-r20"
	decoder.RegisterBackend(name, func() (decoder.Backend, error) {
		cpu, err := decoder.NewBackend("cpu")
		if err != nil {
			return nil, err
		}
		return &decliningBackend{Backend: cpu}, nil
	})
	m, err := decoder.Load(p, decoder.Options{Backend: name, Quant: "int8int8"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = m.Close() })
	lm := &loadedModel{name: "declined", model: m}
	f := factsOf(lm)
	if f.resident || !strings.Contains(f.residentDecline, "no card in this test") {
		t.Fatalf("factsOf: resident=%v residentDecline=%q, want the backend's own reason", f.resident, f.residentDecline)
	}
	if got := bannerLine(modelBanner(lm, config{}), "context:"); !strings.Contains(got, "this model declined one") {
		t.Errorf("the banner of a model that declined residency: %q", got)
	}
}

// A plain CPU load is not a decline: the CPU was the choice. decoder sets ResidentDecline for it too ("backend
// does not implement residency (… the CPU backend)"), so the banner must not report that as a GPU load that
// fell through.
func TestFactsOf_aPlainCPULoadDeclinedNothing(t *testing.T) {
	_, lm := tinyServed(t) // decoder.Load with Backend "cpu"
	if lm.model.ResidentDecline() == "" {
		t.Skip("decoder no longer records a decline for a CPU load, so this guard has nothing to guard against")
	}
	f := factsOf(lm)
	if f.residentDecline != "" {
		t.Errorf("factsOf reports a decline for a plain CPU load: %q", f.residentDecline)
	}
	got := bannerLine(modelBanner(lm, config{}), "context:")
	if strings.Contains(got, "declined") {
		t.Errorf("the banner of a plain CPU load claims a decline: %q", got)
	}
	if !strings.Contains(got, "the CPU path has no --ctx cap") {
		t.Errorf("the banner of a plain CPU load lost the ceiling note: %q", got)
	}
}

type decliningBackend struct{ decoder.Backend }

func (b *decliningBackend) BuildResident(m *decoder.Model) (decoder.ResidentForward, bool, error) {
	return nil, false, decoder.DeclineResident("no card in this test")
}

// TestKVPlanLine pins S18's "KV plan" line (docs/tasks/task-multimodal-support-2026-10.md): the resolved slots and positions, and the reasons the banner can state when the plan is less than was asked for.
func TestKVPlanLine(t *testing.T) {
	for _, tc := range []struct {
		name     string
		f        bannerFacts
		cfg      config
		want     string // "" = no line
		wantNone bool
	}{
		{"a tower held 2.4 GB back and the card allowed 2 of 4 (Gemma 3 4B on the 8 GB card, float32 tower)",
			bannerFacts{resident: true, kvSlots: 2, ctxWindow: 4096, decodePath: "cuda-resident (int4)", towerReserve: 2_400_000_000}, config{kvSessions: 4},
			"KV plan: 2 conversations x 4096 positions (4 asked for; 2.4 GB held back for the vision tower)", false},
		{"one slot", bannerFacts{resident: true, kvSlots: 1, ctxWindow: 4096, decodePath: "cuda-resident (int4)", towerReserve: 959_000_000}, config{kvSessions: 4},
			"KV plan: 1 conversation x 4096 positions (4 asked for; 1.0 GB held back for the vision tower)", false},
		{"nothing reduced, no tower: no parenthesis", bannerFacts{resident: true, kvSlots: 4, ctxWindow: 5057, decodePath: "cuda-resident (int4)"}, config{kvSessions: 4},
			"KV plan: 4 conversations x 5057 positions", false},
		{"an explicit -ctx is named", bannerFacts{resident: true, kvSlots: 2, ctxWindow: 8192, decodePath: "cuda-resident (int4)"}, config{kvSessions: 4, load: loadflags.Flags{Ctx: 8192}},
			"KV plan: 2 conversations x 8192 positions (4 asked for; --ctx 8192)", false},
		{"not resident: no line", bannerFacts{kvSlots: 0, ctxWindow: 4096, decodePath: "cpu (int4)"}, config{kvSessions: 4}, "", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := bannerLine(modelBannerFrom(tc.f, tc.cfg), "KV plan:")
			if tc.wantNone {
				if got != "" {
					t.Errorf("a KV plan line on a non-resident model: %q", got)
				}
				return
			}
			if got != tc.want {
				t.Errorf("KV plan line = %q, want %q", got, tc.want)
			}
		})
	}
}

// The session-reuse line names Metal only for a Metal decode path, not for a CUDA card's granted count.
func TestBanner_sessionReuseNamesMetalOnlyOnMetal(t *testing.T) {
	cfg := config{kvSessions: 4}
	cuda := bannerLine(modelBannerFrom(bannerFacts{resident: true, hasTemplate: true, kvSlots: 2, decodePath: "cuda-resident (int4)"}, cfg), "session reuse:")
	if strings.Contains(cuda, "Metal") || !strings.Contains(cuda, "the default asks for 4 and the memory guard allowed 2") {
		t.Errorf("CUDA session-reuse line = %q", cuda)
	}
	metal := bannerLine(modelBannerFrom(bannerFacts{resident: true, hasTemplate: true, kvSlots: 2, decodePath: "metal-resident (int4)"}, cfg), "session reuse:")
	if !strings.Contains(metal, "Metal keeps 2 by default") {
		t.Errorf("Metal session-reuse line = %q", metal)
	}
}

// A config that declares no maximum (Gemma 3's) must not have the resident capacity called the model maximum.
func TestBanner_contextWhenTheModelDeclaresNoMaximum(t *testing.T) {
	unpinned := bannerLine(modelBannerFrom(bannerFacts{hasTemplate: true, ctxWindow: 4096, maxPositions: 0}, config{}), "context:")
	if strings.Contains(unpinned, "model maximum") || !strings.Contains(unpinned, "declares no maximum") || !strings.Contains(unpinned, "raise with --ctx") {
		t.Errorf("unpinned: %q", unpinned)
	}
	pinned := bannerLine(modelBannerFrom(bannerFacts{hasTemplate: true, ctxWindow: 8192, maxPositions: 0}, config{load: loadflags.Flags{Ctx: 8192}}), "context:")
	if strings.Contains(pinned, "model maximum") || !strings.Contains(pinned, "--ctx") {
		t.Errorf("pinned: %q", pinned)
	}
}

// The head table's precision is on the banner: the same checkpoint is a different model under a different
// table, and the default differs by backend (--embed-int4). Two halves: the line itself from the facts, and
// the facts from a real load with the flag both ways.
func TestBanner_headTable(t *testing.T) {
	for _, c := range []struct{ kind, want string }{
		{"int4", "head table: int4 (--embed-int4; --embed-int4=false pins int8)"},
		{"int8", "head table: int8"},
	} {
		got := bannerLine(modelBannerFrom(bannerFacts{decodePath: "cpu (int4)", headTable: c.kind}, config{}), "head table:")
		if got != c.want {
			t.Errorf("head table %q: banner line %q, want %q", c.kind, got, c.want)
		}
	}
	if got := bannerLine(modelBannerFrom(bannerFacts{decodePath: "cpu (int4)"}, config{}), "head table:"); got != "" {
		t.Errorf("an unknown head table printed a line: %q", got)
	}
}

func TestFactsOf_headTableFollowsEmbedInt4(t *testing.T) {
	p := filepath.Join("..", "..", "testdata", "glm-tiny.gguf")
	if _, err := os.Stat(p); err != nil {
		t.Skipf("no committed tiny fixture at %s", p)
	}
	for _, embed4 := range []bool{true, false} {
		m, err := decoder.Load(p, decoder.Options{Backend: "cpu", Quant: "int4", EmbedInt4: embed4})
		if err != nil {
			t.Fatalf("load (embed-int4 %v): %v", embed4, err)
		}
		f := factsOf(&loadedModel{name: "tiny", model: m})
		_ = m.Close()
		want := "int8"
		if embed4 {
			want = "int4"
		}
		if f.headTable != want {
			t.Errorf("EmbedInt4=%v: facts report head table %q, want %q", embed4, f.headTable, want)
		}
	}
}
