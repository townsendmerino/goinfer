package serveapp

import (
	"fmt"
	"strings"
)

// The startup banner is the UI.
//
// A harness user reads exactly one thing before their first request: the lines `serve` prints
// before it says it is listening. Everything below is a fact the runtime already knows and
// used to keep private — which is how someone discovers, one request at a time, that their
// agent loop re-prefills every turn (docs/task-embed-and-harness-ux.md §3.3, and
// docs/server.md's dsh recipe: "set expectations, don't let the harness discover them").
//
// Built as a function returning lines rather than a run of Fprintf calls so a test can assert
// the banner against the runtime's own state. A banner that drifts from what the server does
// is worse than no banner: it is the M-07 class (the doc says exact, the code does not)
// applied to the one document every user reads. See TestBanner_tellsTheTruth.

// bannerFacts is everything the banner reports, read off the runtime once. Split out so the
// banner is a pure function of resolved state and can be asserted against that state in a
// test — without a GPU, and without capturing stderr.
type bannerFacts struct {
	decodePath   string
	prefillPath  string
	resident     bool // the model decodes through the resident runner (⇒ stateless, no prefix reuse)
	hasTemplate  bool
	toolCallForm bool // the template exposes a constrainable tool-call form
	spec         bool
	blockDrafter bool

	// R13 (docs/measurements/cold-user-2026-09-07-macbook-arm64.md): the context cap, KV at that
	// cap, and what remains of the RAM budget — printed at every load, not only when something is
	// tight, because "79% of budget" at load time and "14 GB RSS, swapping" on the first real
	// request were the same load with no line connecting them.
	fitKnown       bool
	fitCtx         int
	fitKVBytes     int64
	fitWeightBytes int64
	fitBudgetBytes int64
}

func factsOf(lm *loadedModel) bannerFacts {
	_, prefillWhy := lm.model.PrefillPath()
	f := bannerFacts{
		decodePath:   lm.model.DecodePath(),
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
	return f
}

// modelBanner returns the resolved-state lines for one loaded model, in the order §3.3 asks
// for: what it is, where it runs, how much context, whether turns are reused, what it can do.
// Each line is indented two spaces by the caller to sit under the "loaded ..." line.
func modelBanner(lm *loadedModel, cfg config) []string {
	return modelBannerFrom(factsOf(lm), cfg)
}

func modelBannerFrom(f bannerFacts, cfg config) []string {
	var out []string

	// Where it runs. RESOLVED, not requested: both the resident decode path and the batched
	// prefill fall back silently, so a model can report a GPU backend and still take one
	// forward per prompt token.
	out = append(out, "decode path: "+f.decodePath)
	out = append(out, "prefill path: "+f.prefillPath)

	// How much context, and at what KV precision — the two numbers that decide whether a
	// harness's turn fits at all.
	ctxLine := "context: "
	if cfg.ctxSize > 0 {
		ctxLine += fmt.Sprintf("%d tokens (--ctx)", cfg.ctxSize)
	} else {
		ctxLine += "backend default"
	}
	if p := cfg.kvPrec; p != "" && p != "f32" {
		ctxLine += " · KV " + p + " (lossy)"
	} else {
		ctxLine += " · KV f32"
	}
	out = append(out, ctxLine)

	// R13: the memory this context cap actually costs, and what is left — the line a harness
	// user needs to judge whether their own turn size fits, before finding out from swap.
	//
	// R13-follow-on: fitBudgetBytes is now priced against CURRENTLY AVAILABLE memory
	// (decoder.FitBudgetSummary), read after the model is already resident — so weightBytes is
	// ALREADY excluded from it by the OS's own accounting. Subtracting it again here would
	// double-count a footprint that is not there to subtract (the same bug AdmitPrefillMemory
	// had at request time). weights are still shown, for the reader's own arithmetic, not this
	// function's.
	if f.fitKnown {
		remaining := f.fitBudgetBytes - f.fitKVBytes
		out = append(out, fmt.Sprintf(
			"fit: %d-token cap needs %.1f GB KV (weights %.1f GB, budget %.1f GB, %.1f GB left for a request's own prefill)",
			f.fitCtx, float64(f.fitKVBytes)/(1<<30), float64(f.fitWeightBytes)/(1<<30),
			float64(f.fitBudgetBytes)/(1<<30), float64(remaining)/(1<<30)))
	}

	// Session reuse, and WHY when it is off. This is the line that makes an agent loop's
	// per-turn re-prefill visible before it is paid for: decoder.Generate engages the resident
	// DecodeRunner only when there is no session commit and no prefix reuse, because the
	// resident KV lives on the GPU while a session's prefix cache is CPU-side and the two
	// cannot both be the source of truth (see loadedModel.drive). Resident decode is the much
	// larger win, so the trade is deliberate — but it should not be a surprise.
	switch {
	case f.resident:
		out = append(out, "session reuse: OFF — resident decode is stateless, so every turn re-prefills its whole prompt")
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
	var routes []string
	if len(s.models) > 0 {
		routes = append(routes, "/v1/chat/completions", "/v1/completions", "/v1/responses", "/v1/messages")
	}
	routes = append(routes, "/v1/embeddings")
	if cfg.web {
		routes = append(routes, "/ (web UI)")
	}
	if cfg.allowAdmin {
		routes = append(routes, "/admin/models/{load,unload}")
	}
	out := []string{"routes: " + strings.Join(routes, " ")}
	// Load cost, split by phase. task-embed-and-harness-ux.md 3.3 already names the banner as the
	// UI for a harness user; this is one line of it. The SPLIT is what makes it actionable — "load
	// 9.2s" is a number to be annoyed by, "9.2s, 82% build" says the disk is not the problem and a
	// different quant might be. Printed only for a model whose loader instrumented it.
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
