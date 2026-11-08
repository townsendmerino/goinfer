// Package fitcmd is the `fit` subcommand shared by the goinfer binaries — task-fit-to-
// hardware.md §3's dry run: "prints the plan without loading[, wait — this phase DOES load;
// see below]: bytes by class, the chosen placement per available backend, the ctx cap, and what
// was pinned."
//
// PHASE 1 SCOPE NOTE (docs/tasks/task-gpu-paths-2026-09.md's G11 entry has the full reasoning): the
// doc wants this header-only ("two seconds, no model in memory") so `pull` and the web UI can
// check fit before a multi-GB download. Nothing in the repo separates dense from routed-expert
// bytes at the checkpoint-header level today, so this phase loads the model like any other
// command does — real numbers, real quantization, but not the zero-load speed the doc's `pull`/
// web-UI integration eventually wants. That header-only work is still open.
package fitcmd

import (
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/decoder/fit"
	"github.com/townsendmerino/goinfer/internal/runtime/modelload"
	"github.com/townsendmerino/goinfer/internal/runtime/prequant"
)

const fitUsage = `%[1]s fit <path> — show how this checkpoint would be placed on this machine, per backend

  fit <file.gguf|dir>            report at the default quant/context
  fit hf:owner/repo[:quant]      the same, for a reference serve --model accepts (demo:<name> too)
  fit <file.gguf|dir> -ctx 32768 report at a specific context (refused if it doesn't fit, not
                                  silently shrunk — pass a smaller -ctx to see what DOES fit)
  fit <file.gguf|dir> -quant int8int8
  fit <file.gguf|dir> -profile testdata/machines/<box>.json [-backend B] [-json]
                                 price it against a recorded machine as Load's own check would,
                                 without loading it; exit 0 when it proceeds, 4 when refused
  fit <file.gguf|dir> -measure   also self-measure decode rate on the best admitted backend
                                  (tasks/task-fit-to-hardware.md §5) — a REAL load + a short decode,
                                  not free like the report above; off by default

This loads the checkpoint (unlike a future header-only version): real bytes, real quantization,
same load time %[1]s itself would pay. It reports EVERY backend compiled into this binary —
a CPU-only build only ever reports "cpu"; the metal/cuda release assets report their own GPU
backend too. WebGPU is planned correctly (tasks/task-fit-to-hardware.md Phase 3, M-32 fixed) but has
no live free-memory probe yet — WebGPU exposes no portable query for it — so it reports "no
memory probe available... skipped" until one exists.
`

// defaultCtx is `fit`'s -ctx default: the GPU planners' candidate context (cuda/resident.go's fitDefaultCtx, decoder.MetalFitDefaultCtx), so the dry
// run prices what the real load picks. TestDefaultCtx_agreesWithTheCudaPlannerAndHoldsAnAgentTurn keeps it equal to CUDA's.
const defaultCtx = decoder.MetalFitDefaultCtx

// Run implements `fit`. args excludes the program name and the "fit" word. Returns an exit code.
func Run(args []string) int {
	fs := flag.NewFlagSet("fit", flag.ContinueOnError)
	fs.Usage = func() { fmt.Fprintf(os.Stderr, fitUsage, self()) }
	ctx := fs.Int("ctx", defaultCtx, "context to plan for (tasks/task-fit-to-hardware.md §8: 16384 is the size of a coding agent's turn — opencode's first request was 11,137 tokens — not the model's max; pass the model's own window explicitly if you want that priced instead)")
	quant := fs.String("quant", "int4", "weight quant to plan at: int4 | int4mix | int8int8 | int8 | \"\" (f32)")
	kvF16 := fs.Bool("kv-f16", false, "plan KV at f16 instead of f32 (halves KV bytes; a lossy precision choice, never chosen silently)")
	kvI8 := fs.Bool("kv-i8", false, "plan KV at int8 instead of f32 (further shrinks KV bytes; lossy)")
	slots := fs.Int("moe-cache-slots", 0, "an explicit expert-cache slot count to plan against instead of letting Plan choose the largest that fits")
	profile := fs.String("profile", "", "price the load against a recorded machine (a testdata/machines/*.json entry) as Load's own memory check would, without loading it; exit 4 when it would be refused")
	backend := fs.String("backend", "", "with -profile: the backend the load will use (cpu, cuda, metal, webgpu); what Load prices depends on it and on the backends linked into this binary")
	moeCache := fs.Bool("moe-cache-experts", false, "with -profile: the load builds a CUDA expert cache")
	stream := fs.Bool("stream-weights", false, "with -profile: the load streams its weights")
	asJSON := fs.Bool("json", false, "with -profile: print the answer as one line of JSON")
	measure := fs.Bool("measure", false, "self-measure decode rate on the best admitted backend (tasks/task-fit-to-hardware.md §5) — a real load + short decode, not free")

	var path string
	rest := args
	if len(args) > 0 && !strings.HasPrefix(args[0], "-") {
		path, rest = args[0], args[1:]
	}
	if err := fs.Parse(rest); err != nil {
		return 2
	}
	if path == "" {
		fmt.Fprintf(os.Stderr, "%s: fit needs a checkpoint path\n\n", self())
		fs.Usage()
		return 2
	}
	if *profile != "" {
		opts := decoder.Options{Quant: *quant, Backend: *backend, MoECacheExperts: *moeCache, StreamWeights: *stream}
		fs.Visit(func(f *flag.Flag) {
			if f.Name == "ctx" {
				opts.ResidentContext = *ctx // an explicit -ctx is a pin, as it is for Load
			}
		})
		if *kvF16 {
			opts.KVPrecision = "f16"
		}
		if *kvI8 {
			opts.KVQuant = "i8"
		}
		return runProfile(os.Stdout, path, *profile, opts, *asJSON)
	}
	// An hf:/demo: reference resolves exactly as serve's and chat's --model does (internal/modelload):
	// fetched, or found in the cache, before anything is measured. A plain path is untouched.
	resolved, rerr := modelload.Resolve(context.Background(), path)
	if rerr != nil {
		fmt.Fprintf(os.Stderr, "%s: %v\n", self(), rerr)
		return 1
	}
	path = resolved
	// An explicit -ctx pins it (declines rather than auto-shrinks); the default is NOT pinned.
	ctxPinned := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == "ctx" {
			ctxPinned = true
		}
	})

	// M-19 (docs/audit-2026-09-10.md): the CPU free-RAM probe must be read BEFORE decoder.Load,
	// or the budget it reports has already been depleted by the very model being planned for —
	// a guard that inverts under the condition it exists for (CLAUDE.md's own name for this
	// class of bug). GPU probes read DEVICE memory, which a CPU-resident load never touches, so
	// only "cpu" needs its free bytes captured early; freeBytesFor still queries the rest live.
	hostFreeBeforeLoad := decoder.HostRAMAvailableBytes()

	explicitQuant := ""
	fs.Visit(func(f *flag.Flag) {
		if f.Name == "quant" {
			explicitQuant = *quant
		}
	})
	residentCtx := 0
	if ctxPinned {
		residentCtx = *ctx // an explicit -ctx is a pin, as it is for Load
	}

	// Each backend is planned on what a load on it opens (sourceFor). Every probe is read before any load, which on unified
	// memory takes from what the GPU probe reports too.
	type line struct {
		backend string
		free    int64
		known   bool
		src     fitSource
		plan    fit.Plan
	}
	var lines []line
	for _, backend := range decoder.CompiledBackends() {
		l := line{backend: backend}
		if l.free, l.known = freeBytesFor(backend, hostFreeBeforeLoad); l.known {
			src, err := sourceFor(path, *quant, explicitQuant, backend)
			if err != nil {
				fmt.Fprintf(os.Stderr, "%s: %v\n", self(), err)
				return 1
			}
			l.src = src
		}
		lines = append(lines, l)
	}
	// One load per distinct source, closed before the next: two quantized copies at once could trip the guard being priced.
	req := fit.PlanRequest{Ctx: *ctx, CtxPinned: ctxPinned, KVF16: *kvF16, KVI8: *kvI8, Slots: *slots}
	planned := make([]bool, len(lines))
	for i := range lines {
		if !lines[i].known || planned[i] {
			continue
		}
		src := lines[i].src
		m, err := decoder.Load(src.path, fitLoadOptions(src, residentCtx))
		if err != nil {
			fmt.Fprintf(os.Stderr, "load %s: %v\n", src.path, err)
			return 1
		}
		for j := i; j < len(lines); j++ {
			if lines[j].known && !planned[j] && sameSource(lines[j].src, src) {
				lines[j].plan, planned[j] = planFor(m, lines[j].backend, lines[j].free, req), true
			}
		}
		_ = m.Close()
	}

	fmt.Printf("%s, ctx=%d%s\n\n", path, *ctx, pinnedNote(ctxPinned))
	printed := map[string]bool{}
	var admitted []string // in CompiledBackends() order; "cpu" always eligible, sorted last
	srcOf := map[string]fitSource{}
	for _, l := range lines {
		if !l.known {
			fmt.Printf("%-6s  no memory probe available (driver/device not found) — skipped\n", l.backend)
			continue
		}
		for _, n := range l.src.notes {
			if !printed[n] {
				fmt.Printf("%s\n", n)
				printed[n] = true
			}
		}
		via := ""
		if l.src.path != path {
			via = ", read through its sidecar " + filepath.Base(l.src.path)
		}
		fmt.Printf("%-6s  %s @ %s%s\n        %s\n", l.backend, strings.ToUpper(l.plan.Placement.String()), quantLabel(l.src.quant), via, l.plan.Reason)
		if l.plan.Placement != fit.PlacementDecline {
			admitted = append(admitted, l.backend)
		}
		srcOf[l.backend] = l.src
	}
	if *measure {
		selfMeasure(admitted, srcOf)
	}
	return 0
}

// fitSource is what `fit` opens to plan one backend, and at which quant.
type fitSource struct {
	path        string // the checkpoint, or the sidecar .giw read through
	quant       string
	group       int    // the activation group (decoder.Options.ActQuantGroup)
	loadBackend string // the backend the source is opened with: a cpu-target sidecar can be row4-only, which only "cpu" reads
	notes       []string
}

// sourceFor is what a load on backend opens, as the load chooses both: the quant modelload.Plan resolves (the per-family default when
// -quant was not given, with the note the load prints), and a fresh sidecar of that quant, Metal's own on Metal (an always-canonical
// target, opened with no backend), else freshSidecar's. Without one it is path, opened directly: a .gguf read so holds its weights on
// the heap, which the load, through the sidecar it builds first, does not.
func sourceFor(path, quant, explicitQuant, backend string) (fitSource, error) {
	pl, err := modelload.Plan(modelload.Request{Spec: path, Opts: decoder.Options{Quant: quant, Backend: backend}, ExplicitQuant: explicitQuant})
	if err != nil {
		return fitSource{}, err
	}
	s := fitSource{path: path, quant: pl.Opts.Quant, group: pl.Opts.ActQuantGroup, notes: pl.Notes}
	switch {
	case !strings.HasSuffix(path, ".gguf"):
	case backend == "metal":
		if cached, ok := sidecarIfFresh(path, s.quant, "metal"); ok {
			s.path = cached
		}
	default:
		if cached, be, ok := freshSidecar(path, s.quant); ok {
			s.path, s.loadBackend = cached, be
		}
	}
	return s, nil
}

// sameSource reports whether two backends are planned on one load.
func sameSource(a, b fitSource) bool {
	return a.path == b.path && a.quant == b.quant && a.group == b.group && a.loadBackend == b.loadBackend
}

// fitLoadOptions is what `fit` opens src with: its quant and activation group, the backend it is read on, a pinned -ctx as the
// load's own pin (residentCtx, 0 for none), and the KV slot request of a serve load without --kv-sessions (the caller's default,
// which Metal keeps to decoder.MetalDefaultKVSlots), which planFor's Metal line prices.
func fitLoadOptions(src fitSource, residentCtx int) decoder.Options {
	return decoder.Options{Quant: src.quant, ActQuantGroup: src.group, Backend: src.loadBackend, DirectLoad: true, ResidentContext: residentCtx,
		ResidentKVSlots: decoder.MetalDefaultKVSlots, ResidentKVSlotsDefault: true}
}

// planFor is the plan `fit` prints for one backend. On Metal it is decoder.Model.MetalLoadPlan's, the load a serve without
// --kv-sessions makes: the context an unpinned load takes and the KV of every slot the build keeps, and the reason says both.
func planFor(m *decoder.Model, backend string, free int64, req fit.PlanRequest) fit.Plan {
	if backend != "metal" {
		return m.Plan(backend, free, req)
	}
	l := m.MetalLoadPlan(free, req)
	p := l.Plan
	if want := m.MetalKVSlotsWanted(); want > 1 && p.Placement != fit.PlacementDecline {
		further := m.MetalKVBytes(l.KVSlots, p.Ctx) - m.ResidentKVBytes("metal", p.Ctx, req.KVF16, req.KVI8)
		p.Reason += fmt.Sprintf(" — a default load keeps %d of the %d KV slots it asks for (KV %.2f GB in all)", l.KVSlots, want,
			float64(p.KVBytes+further)/(1<<30))
	}
	if !req.CtxPinned {
		p.Reason += fmt.Sprintf(" — an unpinned load takes %d positions (%s)", l.Fit.Ctx, l.Fit.Why)
	}
	return p
}

// selfMeasure is tasks/task-fit-to-hardware.md §5's "self-measure" option: after load, decode a fixed
// probe and print the rate as "measured on this machine" — the SAME model this Model was already
// planned against, loaded again on the backend actually chosen so the probe runs through the
// real decode path (not the plan's hypothetical byte arithmetic above). Prefers any admitted
// non-cpu backend over cpu (cpu is always eligible and always last in admitted, so it is only
// picked when nothing else was) — a user asking "how fast will it go" almost always means the
// GPU they compiled in, not the CPU fallback every build has.
func selfMeasure(admitted []string, srcOf map[string]fitSource) {
	if len(admitted) == 0 {
		fmt.Println("\n(measure) no backend admitted this checkpoint — nothing to probe")
		return
	}
	backend := admitted[0]
	for _, b := range admitted {
		if b != "cpu" {
			backend = b
			break
		}
	}

	src := srcOf[backend] // what fit planned this backend on: the load's quant, and its sidecar when one is fresh
	pm, err := decoder.Load(src.path, decoder.Options{Backend: backend, Quant: src.quant, ActQuantGroup: src.group, DirectLoad: true})
	if err != nil {
		fmt.Printf("\n(measure) %s: load failed, skipping probe: %v\n", backend, err)
		return
	}
	defer pm.Close()

	const probePrompt, probeDecode = 64, 32
	_, _, _, _, _, _, vocab := pm.Dims()
	prompt := make([]int, probePrompt)
	for i := range prompt {
		prompt[i] = (i*2654435761 + 1) % vocab // deterministic, in-range — a throughput probe, not a real prompt
	}

	// M-20 (docs/audit-2026-09-10.md): time from the FIRST generated token, not from before
	// Generate is called — the first token's own latency includes the fixed one-time prompt
	// prefill (probePrompt tokens), so folding it into the rate understated true decode
	// throughput by roughly a third on the CPU staged path (the audit's own measurement). Every
	// step after the first is pure decode, so that's what the rate reports.
	t0 := time.Now()
	out, gen := pm.Generate(context.Background(), prompt, probeDecode, decoder.SamplingParams{Temperature: 0})
	n := 0
	var firstTokAt time.Time
	for range out {
		n++
		if n == 1 {
			firstTokAt = time.Now()
		}
	}
	now := time.Now()
	totalElapsed := now.Sub(t0)
	if gen.Err() != nil {
		fmt.Printf("\n(measure) %s: probe failed: %v\n", backend, gen.Err())
		return
	}
	if n < 2 {
		fmt.Printf("\n(measure) %-6s only %d/%d decode step(s) produced — not enough after the "+
			"first token to report a decode-only rate\n", backend, n, probeDecode)
		return
	}
	decodeSteps := n - 1
	decodeElapsed := now.Sub(firstTokAt)
	rate := float64(decodeSteps) / decodeElapsed.Seconds()
	fmt.Printf("\n(measure) %-6s %.1f tok/s over %d decode steps after the first token (excludes "+
		"the %d-token prompt prefill and the first decode step; total wall time for all %d steps "+
		"including that prefill: %s; measured on this machine, not projected)\n",
		backend, rate, decodeSteps, probePrompt, n, totalElapsed.Round(time.Millisecond))
}

// freeBytesFor is CompiledBackends' "cpu" special case (HostRAMAvailableBytes, which predates
// the RegisterMemoryProbe registry and lives directly on decoder) plus everything else via the
// registry backends register their own probe into (metal/backend.go, cuda/backend.go).
// hostFreeBeforeLoad is the CPU free-RAM probe captured BEFORE decoder.Load — see the M-19
// comment at the call site. GPU backends are unaffected by a CPU-resident load, so they still
// query live.
func freeBytesFor(backend string, hostFreeBeforeLoad int64) (int64, bool) {
	if backend == "cpu" {
		if hostFreeBeforeLoad > 0 {
			return hostFreeBeforeLoad, true
		}
		return 0, false
	}
	return decoder.FreeBytesFor(backend)
}

func quantLabel(q string) string {
	if q == "" {
		return "f32"
	}
	return q
}

func pinnedNote(pinned bool) string {
	if pinned {
		return " (pinned — refused rather than shrunk if it doesn't fit)"
	}
	return ""
}

func self() string {
	if len(os.Args) > 0 && os.Args[0] != "" {
		return filepath.Base(os.Args[0])
	}
	return "goinfer-chat"
}

// freshSidecar finds an already-fresh sidecar .giw that a chat or serve load built for this .gguf, and
// the Backend to load it with. fit has no --backend of its own (it reports every compiled backend), and
// it used to look only for the canonical target — but a default chat or serve run (--backend cpu)
// writes this host's cpu target, cpu-amd64 or cpu-arm64, so fit never found the sidecar they had
// already paid for and did a full direct load instead. Tried in the order a user is likely to have
// built them: this host's cpu target (loaded with Backend "cpu": an arm64 cpu sidecar is row4-only,
// which only the literal cpu backend reads), then cuda's and the canonical one (canonical bytes, any
// backend). A metal sidecar is not used: its fused kinds are Metal's alone.
func freshSidecar(path, quant string) (giw, backend string, ok bool) {
	if !strings.HasSuffix(path, ".gguf") {
		return "", "", false
	}
	for _, be := range []string{"cpu", "cuda", ""} {
		if p, fresh := sidecarIfFresh(path, quant, be); fresh {
			if be == "cuda" {
				be = ""
			}
			return p, be, true
		}
	}
	return "", "", false
}

// sidecarIfFresh is the sidecar a load of path at quant on backend built, whichever head precision it used. fit has no
// --embed-int4 and cannot know the caller's, and the default differs by backend: since 2026-09-28 an int4 load keeps the
// embedding/LM-head table at int4 (its sidecar is <base>.int4.e4h.<target>.giw), except on Metal (loadflags). Looking
// only for the plain-head sidecar, as fit did, missed the one every default CPU, CUDA and WebGPU load writes, and fell
// back to a full direct load. The backend's default is tried first, then the other.
func sidecarIfFresh(path, quant, backend string) (string, bool) {
	def := quant == "int4" && backend != "metal"
	for _, e4 := range []bool{def, !def} {
		if p, ok := prequant.SidecarPathIfFresh(path, quant, backend, e4); ok {
			return p, true
		}
	}
	return "", false
}
