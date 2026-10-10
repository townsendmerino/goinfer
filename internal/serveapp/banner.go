package serveapp

import (
	"fmt"
	"strings"

	"github.com/townsendmerino/goinfer/decoder"
)

// The startup banner is the UI.
//
// A harness user reads exactly one thing before their first request: the lines `serve` prints before it says it is
// listening. Everything below is a fact the runtime already knows, so nobody has to discover one request at a time
// that their agent loop re-prefills every turn (docs/tasks/task-embed-and-harness-ux.md section 3.3, and the dsh
// recipe in docs/server.md: "set expectations, don't let the harness discover them").
//
// The banner is built as a function returning lines, not a run of Fprintf calls, so a test can assert it against the
// runtime's own state (TestBanner_tellsTheTruth). A banner that drifts from what the server does is worse than no
// banner.

// bannerFacts is everything the banner reports, read off the runtime once. Split out so the
// banner is a pure function of resolved state and can be asserted against that state in a
// test — without a GPU, and without capturing stderr.
type bannerFacts struct {
	decodePath   string
	headTable    string // the embedding / LM-head table's precision (decoder.Model.HeadTable); "" when unknown
	prefillPath  string
	resident     bool // the model decodes through the resident runner (⇒ stateless, no prefix reuse)
	hasTemplate  bool
	toolCallForm bool // the template exposes a constrainable tool-call form
	spec         bool
	blockDrafter bool

	// The context cap, KV at that cap, and what remains of the RAM budget, printed at every load and not only when
	// something is tight: a load reading "79% of budget" and a first request swapping at 14 GB RSS are the same load,
	// with no line connecting them.
	fitKnown       bool
	fitCtx         int
	fitKVBytes     int64
	fitWeightBytes int64
	fitBudgetBytes int64

	// The context a text request is ACTUALLY held to (contextWindow — the function prepare enforces and
	// /v1/models publishes as context_window), and the model's own maximum, so the banner can say which of
	// the two bound it. 0 = unknown.
	ctxWindow    int
	maxPositions int

	// kvPrec is the KV precision the resident runner actually allocates (decoder.Model.ResidentKVPrecision),
	// "" off the resident path — where the requested -kv is what applies.
	kvPrec string

	// residentReuseOff: the resident path's own prefix reuse is switched off (GOINFER_NO_RESIDENT_REUSE).
	residentReuseOff bool

	// residentDecline: why a backend that was asked for did not build a resident path (decoder.Model.ResidentDecline), "" when it did or none was asked.
	residentDecline string

	// kvSlots: how many resident KV slots the model's generations choose among (decoder.Model.ResidentKVSlots); 1 for
	// a backend or family without them, 0 off the resident path.
	kvSlots int

	// towerReserve: the VRAM held back for this model's CUDA vision tower when the resident plan was made (towerReserve; 0 when there is none or it is not CUDA).
	towerReserve int64

	// concurrent: how many generations of this model run at once (loadedModel.concurrent).
	concurrent int

	// cpuBatched: a CPU model's concurrent generations join their decode tokens into batched steps
	// (decoder.Model.EnableCPUBatch).
	cpuBatched bool
}

func factsOf(lm *loadedModel) bannerFacts {
	_, prefillWhy := lm.model.PrefillPath()
	f := bannerFacts{
		decodePath:   lm.model.DecodePath(),
		headTable:    lm.model.HeadTable(),
		prefillPath:  prefillWhy,
		resident:     lm.model.ResidentActive(),
		hasTemplate:  lm.tmpl != nil,
		spec:         lm.spec,
		blockDrafter: lm.blockSpec != nil,
	}
	if lm.tmpl != nil {
		_, _, _, _, f.toolCallForm = lm.tmpl.ToolCallWrapper()
	}
	f.fitCtx, f.fitKVBytes, f.fitWeightBytes, f.fitBudgetBytes, f.fitKnown = lm.model.FitBudgetSummary()
	f.ctxWindow = lm.contextWindow(lm.adapter == "")
	f.maxPositions = lm.model.Config().MaxPositions
	f.kvPrec = lm.model.ResidentKVPrecision()
	f.kvSlots = lm.model.ResidentKVSlots()
	// A decline is only a decline when a GPU backend was built and then did not go resident (EffectiveBackend is that backend, not "cpu"). ResidentDecline is
	// also set, with its own "the CPU backend" reason, for an ordinary CPU load or one that -backend auto sent to the CPU — where nothing was declined, the CPU was the choice.
	if lm.model.EffectiveBackend() != "cpu" {
		f.residentDecline = lm.model.ResidentDecline()
	}
	f.concurrent = lm.concurrent
	if v, _ := lm.model.Knob("GOINFER_NO_RESIDENT_REUSE"); v != "" {
		f.residentReuseOff = true
	}
	return f
}

// modelBanner returns the resolved-state lines for one loaded model, in the order a harness user wants them: what it
// is, where it runs, how much context, whether turns are reused, what it can do. The caller indents each line two
// spaces to sit under the "loaded ..." line.
func modelBanner(lm *loadedModel, cfg config) []string {
	f := factsOf(lm)
	f.towerReserve = towerReserve(cfg, lm.source)
	return modelBannerFrom(f, cfg)
}

func modelBannerFrom(f bannerFacts, cfg config) []string {
	var out []string

	// Where it runs. RESOLVED, not requested: both the resident decode path and the batched
	// prefill fall back silently, so a model can report a GPU backend and still take one
	// forward per prompt token.
	out = append(out, "decode path: "+f.decodePath)
	out = append(out, "prefill path: "+f.prefillPath)
	if f.headTable != "" { // the table every token streams, and the one --embed-int4 changes by backend
		line := "head table: " + f.headTable
		if f.headTable == "int4" {
			line += " (--embed-int4; --embed-int4=false pins int8)"
		}
		out = append(out, line)
	}

	// How much context, and at what KV precision: the two numbers that decide whether a harness's turn fits at all.
	// The line prints the resolved limit, ctxWindow (what prepare enforces and /v1/models publishes), and says what
	// set it. The limit is min(model maximum, resident KV cap), and neither "backend default" nor the -ctx value is
	// that; an invisible default found out by degradation is the exact complaint users make about other local
	// servers. cfg.load.Ctx is the requested -ctx for this model (the caller passes the per-model ctx= override when
	// there is one).
	ctxLine := "context: "
	switch {
	case f.ctxWindow <= 0:
		ctxLine += "unknown (the model declares no maximum)"
	case f.maxPositions > 0 && f.ctxWindow < f.maxPositions && cfg.load.Ctx > 0:
		ctxLine += fmt.Sprintf("%d tokens (--ctx; model maximum %d)", f.ctxWindow, f.maxPositions)
	case f.maxPositions > 0 && f.ctxWindow < f.maxPositions:
		ctxLine += fmt.Sprintf("%d tokens (backend default; model maximum %d — raise with --ctx)", f.ctxWindow, f.maxPositions)
	case f.maxPositions == 0 && cfg.load.Ctx > 0:
		// A config that declares no max_position_embeddings (Gemma 3's): the figure is the resident KV capacity, not
		// a model limit.
		ctxLine += fmt.Sprintf("%d tokens (--ctx; the model declares no maximum)", f.ctxWindow)
	case f.maxPositions == 0:
		ctxLine += fmt.Sprintf("%d tokens (backend default; the model declares no maximum — raise with --ctx)", f.ctxWindow)
	case cfg.load.Ctx > f.ctxWindow:
		ctxLine += fmt.Sprintf("%d tokens (model maximum; --ctx %d is above it)", f.ctxWindow, cfg.load.Ctx)
	default:
		ctxLine += fmt.Sprintf("%d tokens (model maximum)", f.ctxWindow)
	}
	// On the CPU path the window is the model's whole maximum (-ctx is the GPU-resident KV capacity and caps nothing
	// here) and KV is allocated per request, so this figure and the "fit:" KV cost below are a ceiling a request
	// reaches only by filling the window. Said here, beside the figure: after a CUDA decline the two lines otherwise
	// read as the same question with different answers.
	cpuNote := ""
	if cpuCeiling := f.ctxWindow > 0 && !f.resident && f.ctxWindow == f.maxPositions; cpuCeiling {
		cpuNote += " — the CPU path has no --ctx cap: KV is allocated per request, so this is a ceiling, not memory held"
		if f.residentDecline != "" {
			cpuNote += "; a GPU-resident load caps it with --ctx, but this model declined one (see decode path)"
		}
	}
	// The precision that RUNS: a resident runner reports its own (Metal allocates f16 KV whatever -kv
	// says); off it, the requested -kv applies. "(lossy)" marks a precision the operator chose below f32.
	switch req := cfg.load.KV; {
	case f.kvPrec != "":
		ctxLine += " · KV " + f.kvPrec
		if f.kvPrec != "f32" && f.kvPrec == req {
			ctxLine += " (lossy)"
		}
	case req != "" && req != "f32":
		ctxLine += " · KV " + req + " (lossy)"
	default:
		ctxLine += " · KV f32"
	}
	ctxLine += cpuNote
	out = append(out, ctxLine)

	// The memory this context cap actually costs, and what is left: the line a harness user needs to judge whether
	// their own turn size fits, before finding out from swap. fitBudgetBytes is priced against currently available
	// memory (decoder.FitBudgetSummary), read after the model is already resident, so the weights are already
	// excluded from it by the OS's own accounting. Subtracting them again would double-count a footprint that is not
	// there (the bug AdmitPrefillMemory once had at request time). Weights are still shown, for the reader's own
	// arithmetic.
	if f.fitKnown {
		remaining := f.fitBudgetBytes - f.fitKVBytes
		fitLine := fmt.Sprintf(
			"fit: %d-token cap needs %.1f GB KV (weights %.1f GB, budget %.1f GB, %.1f GB left for a request's own prefill)",
			f.fitCtx, float64(f.fitKVBytes)/(1<<30), float64(f.fitWeightBytes)/(1<<30),
			float64(f.fitBudgetBytes)/(1<<30), float64(remaining)/(1<<30))
		if !f.resident && f.ctxWindow > 0 && f.fitCtx == f.maxPositions { // the CPU path's ceiling, as the context line says
			fitLine += " — a ceiling, reached only by a request that fills the window"
		}
		out = append(out, fitLine)
	}

	// Prefix reuse, and why when it is off or narrower than it sounds: the line that makes an agent loop's per-turn
	// re-prefill visible before it is paid for. A resident model does not use the CPU-side session LRU (its KV lives
	// on the device; see loadedModel.drive), but its own cache reuses the prefix committed by the last generation
	// (decoder/resident_reuse.go): a continuing conversation prefills only its new suffix, and a different
	// conversation re-prefills in full.
	if f.resident && f.kvSlots > 0 && f.ctxWindow > 0 {
		out = append(out, kvPlanLine(f, cfg))
	}
	switch {
	case f.resident && f.residentReuseOff:
		out = append(out, "session reuse: OFF — GOINFER_NO_RESIDENT_REUSE is set, so every turn re-prefills its whole prompt")
	case f.resident && f.kvSlots > 1:
		line := fmt.Sprintf("session reuse: on the GPU cache, %d conversations kept resident — each reuses its own prefix; "+
			"a further one takes the least recently used slot and re-prefills", f.kvSlots)
		if cfg.kvSessions > f.kvSlots && !cfg.kvSessionsSet && strings.HasPrefix(f.decodePath, "metal") {
			line += fmt.Sprintf(" (--kv-sessions not given: Metal keeps %d by default, and a memory guard may keep fewer; --kv-sessions %d asks for %d)",
				f.kvSlots, cfg.kvSessions, cfg.kvSessions)
		} else if cfg.kvSessions > f.kvSlots && !cfg.kvSessionsSet {
			// Not Metal: the Metal wording would misreport the count a CUDA card was granted.
			line += fmt.Sprintf(" (--kv-sessions not given: the default asks for %d and the memory guard allowed %d; --kv-sessions %d asks for %d)",
				cfg.kvSessions, f.kvSlots, cfg.kvSessions, cfg.kvSessions)
		} else if cfg.kvSessions > f.kvSlots {
			line += fmt.Sprintf(" (--kv-sessions %d; the memory guard allowed %d)", cfg.kvSessions, f.kvSlots)
		}
		out = append(out, line)
	case f.resident:
		why := "--kv-sessions applies to the CPU path only"
		if cfg.kvSessions > 1 {
			why = fmt.Sprintf("--kv-sessions %d asked for more, but this backend or model family keeps one GPU KV slot, "+
				"or the memory guard allowed only one", cfg.kvSessions)
		}
		out = append(out, "session reuse: on the GPU cache, one conversation — the most recent one's prefix is reused; "+
			"switching to another conversation re-prefills its whole prompt ("+why+")")
	case cfg.kvSessions > 0:
		out = append(out, fmt.Sprintf("session reuse: on (%d conversations kept prefilled)", cfg.kvSessions))
	default:
		out = append(out, "session reuse: OFF (--kv-sessions 0)")
	}

	// What it can do, in the terms a harness asks about.
	var feats []string
	switch {
	case !f.hasTemplate:
		feats = append(feats, "no chat template — raw completion only")
	case f.toolCallForm:
		feats = append(feats, "tools (constrainable call form)")
	default:
		feats = append(feats, "tools: parse-only (a named tool_choice cannot be constrained on this template)")
	}
	feats = append(feats, "structured output")
	if f.spec {
		feats = append(feats, "speculative: n-gram")
	}
	if f.blockDrafter {
		feats = append(feats, "speculative: block drafter")
	}
	out = append(out, "features: "+strings.Join(feats, " · "))
	return out
}

// serverBanner returns the once-per-process lines: which routes a client can actually call.
// A harness speaks exactly one of these families, and "which URL do I point it at" is the
// first question every integration recipe has to answer.
func serverBanner(s *server, cfg config) []string {
	// The generation routes are always registered (a model can be loaded after startup); with none
	// loaded yet they answer 404 "model not found", and the line says so.
	gen := "/v1/chat/completions /v1/completions /v1/responses /v1/messages"
	if len(s.models) == 0 {
		gen += " (after a model is loaded)"
	}
	routes := []string{gen, "/v1/embeddings"}
	if cfg.web {
		routes = append(routes, "/ (web UI)")
	}
	if cfg.allowAdmin {
		routes = append(routes, "/admin/models/{load,unload}", "/admin/generations", "/admin/generations/{id}/cancel")
	}
	out := []string{"routes: " + strings.Join(routes, " ")}
	// Load cost, split by phase. The split is what makes it actionable: "load 9.2s" is a number to be annoyed by,
	// "9.2s, 82% build" says the disk is not the problem and a different quant might be. Printed only for a model
	// whose loader instrumented it (docs/tasks/task-embed-and-harness-ux.md section 3.3 names the banner as the
	// harness user's UI).
	for _, lm := range s.models {
		if sum := lm.model.LoadProfile().Summary(); sum != "" {
			out = append(out, sum)
			break // one line; a zoo would otherwise print one per model
		}
	}
	if !cfg.web {
		out = append(out, "web UI: off (-web enables a browser UI at / for chat and model pulls)")
	}
	return out
}

// concurrencyLine says how many generations of a model run at once (-max-concurrent) and why fewer than asked when
// that happens; "" when nothing needs saying. It is not part of the load-time banner: setConcurrency decides after
// every model, adapter and vision tower has loaded, so a line printed with the banner would report the undecided
// value.
func concurrencyLine(f bannerFacts, cfg config) string {
	switch {
	case f.concurrent > 1 && f.resident:
		line := fmt.Sprintf("concurrency: %d generations at once, each on its own resident KV slot, decode tokens batched (-max-concurrent)", f.concurrent)
		if cfg.maxConcurrent > f.concurrent {
			line += fmt.Sprintf("; %d asked, capped by the %d resident KV slots (--kv-sessions)", cfg.maxConcurrent, f.kvSlots)
		}
		if cfg.prefillChunk > 0 {
			line += fmt.Sprintf("; a long prompt arriving mid-decode prefills in %d-token chunks (-prefill-chunk)", cfg.prefillChunk)
		}
		return line
	case f.concurrent > 1:
		line := fmt.Sprintf("concurrency: %d generations at once, each on its own session KV (-max-concurrent)", f.concurrent)
		if f.cpuBatched {
			line = fmt.Sprintf("concurrency: %d generations at once, each on its own session KV, decode tokens batched on the CPU (-max-concurrent, -cpu-batch)", f.concurrent)
		} else if cfg.cpuBatch == decoder.CPUBatchAuto {
			line += "; decode runs as independent workers (-cpu-batch auto batches models with 2 GiB or more of weights)"
		}
		if cfg.maxConcurrent > f.concurrent {
			line += fmt.Sprintf("; %d asked, capped by --kv-sessions %d", cfg.maxConcurrent, cfg.kvSessions)
		}
		return line
	case cfg.maxConcurrent > 1 && f.resident:
		return fmt.Sprintf("concurrency: one generation at a time — -max-concurrent %d needs a resident that "+
			"batches decode (a dense family on 2+ resident KV slots, no speculation or adapter)", cfg.maxConcurrent)
	case cfg.maxConcurrent > 1:
		return fmt.Sprintf("concurrency: one generation at a time — -max-concurrent %d applies to CPU models "+
			"and to GPU-resident models that batch decode (a weight-streaming or vision model runs one)", cfg.maxConcurrent)
	}
	return ""
}

// kvPlanLine is the "KV plan" line: what the resident plan chose and, when it is less than was asked for, why. The
// numbers are the resolved ones (decoder.Model.ResidentKVSlots and the enforced context window), not the request; the
// reasons are those the banner can state without the planner's own arithmetic: slots asked for beyond those granted,
// an explicit -ctx, and the VRAM held back for a CUDA vision tower.
func kvPlanLine(f bannerFacts, cfg config) string {
	noun := "conversations"
	if f.kvSlots == 1 {
		noun = "conversation"
	}
	line := fmt.Sprintf("KV plan: %d %s x %d positions", f.kvSlots, noun, f.ctxWindow)
	var why []string
	if cfg.kvSessions > f.kvSlots {
		why = append(why, fmt.Sprintf("%d asked for", cfg.kvSessions))
	}
	if cfg.load.Ctx > 0 {
		why = append(why, fmt.Sprintf("--ctx %d", cfg.load.Ctx))
	}
	if f.towerReserve > 0 {
		why = append(why, fmt.Sprintf("%.1f GB held back for the vision tower", float64(f.towerReserve)/1e9))
	}
	if len(why) > 0 {
		line += " (" + strings.Join(why, "; ") + ")"
	}
	return line
}
