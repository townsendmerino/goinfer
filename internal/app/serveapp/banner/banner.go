package banner

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
)

// Harness context thresholds and CUDA int8 KV trade-off note.
const (
	HarnessCtx          = 16384
	HarnessFirstRequest = 11137
	KVI8Cost            = "about f16's quality on a 7B at 8K keys, a measurable loss on a 1.5B; decode up to 28% slower than f16 at short context, prefill near f16's"
)

// Facts holds the resolved runtime properties of a loaded model reported by the banner.
type Facts struct {
	DecodePath   string
	HeadTable    string
	PrefillPath  string
	Resident     bool
	Snapshots    bool
	HasTemplate  bool
	ToolCallForm bool
	Spec         bool
	BlockDrafter bool

	FitKnown       bool
	FitCtx         int
	FitKVBytes     int64
	FitWeightBytes int64
	FitBudgetBytes int64

	CtxWindow    int
	MaxPositions int
	CtxNote      string

	HarnessKVBytes int64
	KV             []decoder.KVPrecisionReport

	ResidentReuseOff bool
	ResidentDecline  string

	KVSlots      int
	KVPool       int
	KVShare      int
	KVPoolDetail decoder.KVPoolDetail
	KVShareState string
	KVShareWhy   string

	TowerReserve int64
	Concurrent   int
	CPUBatched   bool
	Sampling     *decoder.RecommendedSampling
}

// Config holds the serving flags that shape model and concurrency banner lines.
type Config struct {
	Ctx            int
	KVSessions     int
	KVSessionsSet  bool
	KVHostBytes    int64
	KVHostBytesHow string
	KVDemote       bool
	MaxConcurrent  int
	PrefillChunk   int
	CPUBatch       int
}

// ServerConfig holds the process-level route and load facts for ServerLines.
type ServerConfig struct {
	HasModels   bool
	HasSpeech   bool
	Web         bool
	AllowAdmin  bool
	LoadSummary string
}

// ModelLines formats the startup banner lines for one loaded model.
func ModelLines(f Facts, cfg Config) []string {
	var out []string

	out = append(out, "decode path: "+f.DecodePath)
	out = append(out, "prefill path: "+f.PrefillPath)
	if f.HeadTable != "" {
		line := "head table: " + f.HeadTable
		if f.HeadTable == "int4" {
			line += " (--embed-int4; --embed-int4=false pins int8)"
		}
		out = append(out, line)
	}
	out = append(out, SamplingLine(f.Sampling))

	ctxLine := "context: "
	byDefault := "backend default"
	if f.CtxNote != "" {
		byDefault = f.CtxNote
	}
	switch {
	case f.CtxWindow <= 0:
		ctxLine += "unknown (the model declares no maximum)"
	case f.MaxPositions > 0 && f.CtxWindow < f.MaxPositions && cfg.Ctx > 0:
		ctxLine += fmt.Sprintf("%d tokens (--ctx; model maximum %d)", f.CtxWindow, f.MaxPositions)
	case f.MaxPositions > 0 && f.CtxWindow < f.MaxPositions:
		ctxLine += fmt.Sprintf("%d tokens (%s; model maximum %d — raise with --ctx)", f.CtxWindow, byDefault, f.MaxPositions)
	case f.MaxPositions == 0 && cfg.Ctx > 0:
		ctxLine += fmt.Sprintf("%d tokens (--ctx; the model declares no maximum)", f.CtxWindow)
	case f.MaxPositions == 0:
		ctxLine += fmt.Sprintf("%d tokens (%s; the model declares no maximum — raise with --ctx)", f.CtxWindow, byDefault)
	case cfg.Ctx > f.CtxWindow:
		ctxLine += fmt.Sprintf("%d tokens (model maximum; --ctx %d is above it)", f.CtxWindow, cfg.Ctx)
	default:
		ctxLine += fmt.Sprintf("%d tokens (model maximum)", f.CtxWindow)
	}
	cpuNote := ""
	if cpuCeiling := f.CtxWindow > 0 && !f.Resident && f.CtxWindow == f.MaxPositions; cpuCeiling {
		cpuNote += " — the CPU path has no --ctx cap: KV is allocated per request, so this is a ceiling, not memory held"
		if f.ResidentDecline != "" {
			cpuNote += "; a GPU-resident load caps it with --ctx, but this model declined one (see decode path)"
		}
	}
	ctxLine += KVBannerPart(f.KV, strings.HasPrefix(f.DecodePath, "cuda-resident"))
	ctxLine += cpuNote
	out = append(out, ctxLine)
	if l := HarnessLine(f); l != "" {
		out = append(out, l)
	}

	if f.FitKnown {
		kv, what := f.FitKVBytes, fmt.Sprintf("%d-token cap needs", f.FitCtx)
		if f.Resident && f.KVSlots > 1 && f.KVPoolDetail.Bytes > 0 {
			kv, what = f.KVPoolDetail.Bytes, fmt.Sprintf("%d KV slots at a %d-token cap hold", f.KVSlots, f.FitCtx)
		}
		fitLine := fmt.Sprintf(
			"fit: %s %s KV (weights %s, budget %s, %s left for a request's own prefill)",
			what, mem.GiB(kv), mem.GiB(f.FitWeightBytes),
			mem.GiB(f.FitBudgetBytes), mem.GiB(f.FitBudgetBytes-kv))
		if !f.Resident && f.CtxWindow > 0 && f.FitCtx == f.MaxPositions {
			fitLine += " — a ceiling, reached only by a request that fills the window"
		}
		out = append(out, fitLine)
	}

	if f.Resident && f.KVSlots > 0 && f.CtxWindow > 0 {
		out = append(out, KVPlanLine(f, cfg))
		if l := KVPoolBannerLine(f); l != "" {
			out = append(out, l)
		}
	}
	switch {
	case f.Resident && f.ResidentReuseOff:
		out = append(out, "session reuse: OFF — GOINFER_NO_RESIDENT_REUSE is set, so every turn re-prefills its whole prompt")
	case f.Resident && f.KVSlots > 1:
		line := fmt.Sprintf("session reuse: on the GPU cache, %d conversations kept resident — each reuses its own prefix; "+
			"a further one takes the least recently used slot and re-prefills", f.KVSlots)
		if cfg.KVSessions > f.KVSlots && !cfg.KVSessionsSet && strings.HasPrefix(f.DecodePath, "metal") {
			line += fmt.Sprintf(" (--kv-sessions not given: Metal keeps %d by default, and a memory guard may keep fewer; --kv-sessions %d asks for %d)",
				f.KVSlots, cfg.KVSessions, cfg.KVSessions)
		} else if cfg.KVSessions > f.KVSlots && !cfg.KVSessionsSet {
			line += fmt.Sprintf(" (--kv-sessions not given: the default asks for %d and the memory guard allowed %d; --kv-sessions %d asks for %d)",
				cfg.KVSessions, f.KVSlots, cfg.KVSessions, cfg.KVSessions)
		} else if cfg.KVSessions > f.KVSlots {
			line += fmt.Sprintf(" (--kv-sessions %d; the memory guard allowed %d)", cfg.KVSessions, f.KVSlots)
		}
		out = append(out, line)
	case f.Resident:
		why := "--kv-sessions applies to the CPU path only"
		if cfg.KVSessions > 1 {
			why = fmt.Sprintf("--kv-sessions %d asked for more, but this backend or model family keeps one GPU KV slot, "+
				"or the memory guard allowed only one", cfg.KVSessions)
		}
		out = append(out, "session reuse: on the GPU cache, one conversation — the most recent one's prefix is reused; "+
			"switching to another conversation re-prefills its whole prompt ("+why+")")
	case cfg.KVSessions > 0:
		out = append(out, fmt.Sprintf("session reuse: on (%d conversations kept prefilled)", cfg.KVSessions))
	default:
		out = append(out, "session reuse: OFF (--kv-sessions 0)")
	}

	if f.Resident {
		switch {
		case cfg.KVHostBytes <= 0:
			how := cfg.KVHostBytesHow
			if how == "" {
				how = "-kv-host-bytes 0"
			}
			out = append(out, "prefix snapshots: OFF ("+how+")")
		case !f.Snapshots:
			out = append(out, "prefix snapshots: not on this backend or model family")
		default:
			out = append(out, fmt.Sprintf("prefix snapshots: budget %s (%s), 0 B held; host memory is held only while a snapshot is "+
				"(taken after a cold first prefill of 2048 ids or more; /admin/status reports the use)", mem.GiB(cfg.KVHostBytes), cfg.KVHostBytesHow))
			if cfg.KVDemote {
				out = append(out, "released conversations: kept in that budget where the backend can, and brought back when they return")
			}
		}
	}

	var feats []string
	switch {
	case !f.HasTemplate:
		feats = append(feats, "no chat template — raw completion only")
	case f.ToolCallForm:
		feats = append(feats, "tools (constrainable call form)")
	default:
		feats = append(feats, "tools: parse-only (a named tool_choice cannot be constrained on this template)")
	}
	feats = append(feats, "structured output")
	if f.Spec {
		feats = append(feats, "speculative: n-gram")
	}
	if f.BlockDrafter {
		feats = append(feats, "speculative: block drafter")
	}
	out = append(out, "features: "+strings.Join(feats, " · "))
	return out
}

// ServerLines formats the once-per-process routes and web-UI summary lines.
func ServerLines(cfg ServerConfig) []string {
	gen := "/v1/chat/completions /v1/completions /v1/responses /v1/messages"
	if !cfg.HasModels {
		gen += " (after a model is loaded)"
	}
	routes := []string{gen, "/v1/embeddings"}
	if cfg.HasSpeech {
		routes = append(routes, "/v1/audio/transcriptions")
	}
	if cfg.Web {
		routes = append(routes, "/ (web UI)")
	}
	if cfg.AllowAdmin {
		routes = append(routes, "/admin/models/{load,unload}", "/admin/generations", "/admin/generations/{id}/cancel")
	}
	out := []string{"routes: " + strings.Join(routes, " ")}
	if cfg.LoadSummary != "" {
		out = append(out, cfg.LoadSummary)
	}
	if !cfg.Web {
		out = append(out, "web UI: off (-web enables a browser UI at / for chat and model pulls)")
	}
	return out
}

// ConcurrencyLine formats the post-load concurrency summary line, or "" when 1x default.
func ConcurrencyLine(f Facts, cfg Config) string {
	switch {
	case f.Concurrent > 1 && f.Resident:
		line := fmt.Sprintf("concurrency: %d generations at once, each on its own resident KV slot, decode tokens batched (-max-concurrent)", f.Concurrent)
		if cfg.MaxConcurrent > f.Concurrent {
			line += fmt.Sprintf("; %d asked, capped by the %d resident KV slots (--kv-sessions)", cfg.MaxConcurrent, f.KVSlots)
		}
		if cfg.PrefillChunk > 0 {
			line += fmt.Sprintf("; a long prompt arriving mid-decode prefills in %d-token chunks (-prefill-chunk)", cfg.PrefillChunk)
		}
		return line
	case f.Concurrent > 1:
		line := fmt.Sprintf("concurrency: %d generations at once, each on its own session KV (-max-concurrent)", f.Concurrent)
		if f.CPUBatched {
			line = fmt.Sprintf("concurrency: %d generations at once, each on its own session KV, decode tokens batched on the CPU (-max-concurrent, -cpu-batch)", f.Concurrent)
		} else if cfg.CPUBatch == decoder.CPUBatchAuto {
			line += "; decode runs as independent workers (-cpu-batch auto batches models with 2 GiB or more of weights)"
		}
		if cfg.MaxConcurrent > f.Concurrent {
			line += fmt.Sprintf("; %d asked, capped by --kv-sessions %d", cfg.MaxConcurrent, cfg.KVSessions)
		}
		return line
	case cfg.MaxConcurrent > 1 && f.Resident:
		return fmt.Sprintf("concurrency: one generation at a time — -max-concurrent %d needs a resident that "+
			"batches decode (a dense family on 2+ resident KV slots, no speculation or adapter)", cfg.MaxConcurrent)
	case cfg.MaxConcurrent > 1:
		return fmt.Sprintf("concurrency: one generation at a time — -max-concurrent %d applies to CPU models "+
			"and to GPU-resident models that batch decode (a weight-streaming or vision model runs one)", cfg.MaxConcurrent)
	}
	return ""
}

// HarnessLine warns when a GPU-resident model's context window is below HarnessCtx.
func HarnessLine(f Facts) string {
	if !f.Resident || f.CtxWindow <= 0 || f.CtxWindow >= HarnessCtx {
		return ""
	}
	line := fmt.Sprintf("harness: %d tokens is short for a coding agent — its first request is about 11k tokens (opencode: %d)",
		f.CtxWindow, HarnessFirstRequest)
	if f.CtxWindow <= HarnessFirstRequest {
		line += " and will be refused (context_length_exceeded)"
	} else {
		line += fmt.Sprintf(", which leaves %d for the session", f.CtxWindow-HarnessFirstRequest)
	}
	switch {
	case f.MaxPositions > 0 && f.MaxPositions < HarnessCtx:
		return line + fmt.Sprintf("; the model's own maximum is %d, so --ctx cannot raise it to %d", f.MaxPositions, HarnessCtx)
	case f.HarnessKVBytes > 0:
		slots := max(f.KVSlots, 1)
		noun := "slots"
		if slots == 1 {
			noun = "slot"
		}
		return line + fmt.Sprintf("; --ctx %d holds it for %s of KV (%d %s)", HarnessCtx,
			mem.GiB(int64(slots)*f.HarnessKVBytes), slots, noun)
	}
	return line + fmt.Sprintf("; --ctx %d holds it", HarnessCtx)
}

// KVPlanLine formats the "KV plan:" summary line for a resident model.
func KVPlanLine(f Facts, cfg Config) string {
	noun := "conversations"
	if f.KVSlots == 1 {
		noun = "conversation"
	}
	line := fmt.Sprintf("KV plan: %d %s x %d positions", f.KVSlots, noun, f.CtxWindow)
	if f.KVPool > 0 && f.KVShare > 0 && f.KVShare < f.CtxWindow {
		line = fmt.Sprintf("KV plan: a %d-position pool for %d %s; each is guaranteed %d, and one can use up to %d", f.KVPool, f.KVSlots, noun,
			f.KVShare, f.CtxWindow)
	}
	var why []string
	if cfg.KVSessions > f.KVSlots {
		why = append(why, fmt.Sprintf("%d asked for", cfg.KVSessions))
	}
	if cfg.Ctx > 0 {
		why = append(why, fmt.Sprintf("--ctx %d", cfg.Ctx))
	}
	if f.TowerReserve > 0 {
		why = append(why, fmt.Sprintf("%s held back for the vision tower", mem.GiB(f.TowerReserve)))
	}
	if len(why) > 0 {
		line += " (" + strings.Join(why, "; ") + ")"
	}
	return line
}

// KVPoolBannerLine formats the "KV pool:" detail line when a resident model uses a block pool.
func KVPoolBannerLine(f Facts) string {
	d := f.KVPoolDetail
	if d.Positions <= 0 || d.BlockRows <= 0 {
		return ""
	}
	groups := "group: "
	if len(d.Groups) > 1 {
		groups = "groups: "
	}
	groups += strings.Join(d.Groups, ", ")
	if d.WindowRows > 0 {
		groups += fmt.Sprintf(", whose sliding-window layers keep a conversation's newest %s rows", commaInt(d.WindowRows))
	}
	scope := "a prompt can take another conversation's KV rows"
	if f.KVShareState != "on" {
		scope = "no conversation takes another's KV rows (" + f.KVShareWhy + ")"
	}
	use := "one conversation can use all of it"
	if f.CtxWindow > 0 && (f.CtxWindow < d.Positions || d.WindowRows > 0) {
		use = fmt.Sprintf("one conversation can use up to %s of them", commaInt(f.CtxWindow))
	}
	return fmt.Sprintf("KV pool: %s tokens in blocks of %d (%s, %s; %s); %s; %s.", commaInt(d.Positions), d.BlockRows, d.Precision,
		kvPoolBytes(d.Bytes), groups, scope, use)
}

func kvPoolBytes(n int64) string {
	if n >= 1<<30 {
		return mem.GiB(n)
	}
	return fmt.Sprintf("%.1f MiB", float64(n)/(1<<20))
}

func commaInt(n int) string {
	s := strconv.Itoa(n)
	for i := len(s) - 3; i > 0; i -= 3 {
		s = s[:i] + "," + s[i:]
	}
	return s
}

// SamplingLine describes the checkpoint's recommended sampling defaults or server fallback.
func SamplingLine(rs *decoder.RecommendedSampling) string {
	if rs == nil {
		return "sampling: the model recommends none; a request that sends no sampling fields gets temperature 1.0 over the whole vocabulary"
	}
	set := rs.Applied()
	line := "sampling: " + rs.Source + " recommends "
	if len(set) == 0 {
		line += "nothing the sampler applies"
	} else {
		line += strings.Join(set, ", ") + ", applied to the fields a request leaves out"
	}
	if rs.RepetitionPenalty != nil {
		line += "; not applied: repetition_penalty " + strconv.FormatFloat(*rs.RepetitionPenalty, 'g', -1, 64) +
			" (it divides logits; the sampler's penalties are subtractive)"
	}
	return line
}

// KVBannerPart formats the KV precision suffix on the context line.
func KVBannerPart(reps []decoder.KVPrecisionReport, cudaResident bool) string {
	if len(reps) == 0 {
		return " · KV f32"
	}
	var s strings.Builder
	for i, r := range reps {
		label := "KV"
		if i > 0 {
			label = "CPU KV"
		}
		note := "requested " + r.Requested
		if r.Reason != "" && r.Requested != r.Delivered {
			note += ": " + r.Reason
		}
		if (r.Delivered == "f16" || r.Delivered == "i8") && r.Delivered == r.Requested {
			note += ", lossy"
			if r.Delivered == "i8" && r.Cache == "resident" && cudaResident {
				note += ": " + KVI8Cost
			}
		}
		s.WriteString(" · " + label + " " + r.Delivered + " (" + note + ")")
	}
	return s.String()
}
