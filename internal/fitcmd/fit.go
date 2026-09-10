// Package fitcmd is the `fit` subcommand shared by the goinfer binaries — task-fit-to-
// hardware.md §3's dry run: "prints the plan without loading[, wait — this phase DOES load;
// see below]: bytes by class, the chosen placement per available backend, the ctx cap, and what
// was pinned."
//
// PHASE 1 SCOPE NOTE (docs/task-gpu-paths-2026-09.md's G11 entry has the full reasoning): the
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
)

const fitUsage = `%[1]s fit <path> — show how this checkpoint would be placed on this machine, per backend

  fit <file.gguf|dir>            report at the default quant/context
  fit <file.gguf|dir> -ctx 32768 report at a specific context (refused if it doesn't fit, not
                                  silently shrunk — pass a smaller -ctx to see what DOES fit)
  fit <file.gguf|dir> -quant int8int8
  fit <file.gguf|dir> -measure   also self-measure decode rate on the best admitted backend
                                  (task-fit-to-hardware.md §5) — a REAL load + a short decode,
                                  not free like the report above; off by default

This loads the checkpoint (unlike a future header-only version): real bytes, real quantization,
same load time %[1]s itself would pay. It reports EVERY backend compiled into this binary —
a CPU-only build only ever reports "cpu"; the metal/cuda release assets report their own GPU
backend too. WebGPU is planned correctly (task-fit-to-hardware.md Phase 3, M-32 fixed) but has
no live free-memory probe yet — WebGPU exposes no portable query for it — so it reports "no
memory probe available... skipped" until one exists.
`

// Run implements `fit`. args excludes the program name and the "fit" word. Returns an exit code.
func Run(args []string) int {
	fs := flag.NewFlagSet("fit", flag.ContinueOnError)
	fs.Usage = func() { fmt.Fprintf(os.Stderr, fitUsage, self()) }
	ctx := fs.Int("ctx", 8192, "context to plan for (task-fit-to-hardware.md §8: 8192 is the agent-turn size, not the model's max — pass the model's own window explicitly if you want that priced instead)")
	quant := fs.String("quant", "int4", "weight quant to plan at: int4 | int4mix | int8int8 | int8 | \"\" (f32)")
	kvF16 := fs.Bool("kv-f16", false, "plan KV at f16 instead of f32 (halves KV bytes; a lossy precision choice, never chosen silently)")
	kvI8 := fs.Bool("kv-i8", false, "plan KV at int8 instead of f32 (further shrinks KV bytes; lossy)")
	slots := fs.Int("moe-cache-slots", 0, "an explicit expert-cache slot count to plan against instead of letting Plan choose the largest that fits")
	measure := fs.Bool("measure", false, "self-measure decode rate on the best admitted backend (task-fit-to-hardware.md §5) — a real load + short decode, not free")

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
	// An explicit -ctx pins it (declines rather than auto-shrinks); the default is NOT pinned.
	ctxPinned := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == "ctx" {
			ctxPinned = true
		}
	})

	m, err := decoder.Load(path, decoder.Options{Quant: *quant})
	if err != nil {
		fmt.Fprintf(os.Stderr, "load %s: %v\n", path, err)
		return 1
	}
	defer m.Close()

	req := decoder.PlanRequest{Ctx: *ctx, CtxPinned: ctxPinned, KVF16: *kvF16, KVI8: *kvI8, Slots: *slots}
	fmt.Printf("%s @ %s, ctx=%d%s\n\n", path, quantLabel(*quant), *ctx, pinnedNote(ctxPinned))

	var admitted []string // in CompiledBackends() order; "cpu" always eligible, sorted last
	for _, backend := range decoder.CompiledBackends() {
		free, known := freeBytesFor(backend)
		if !known {
			fmt.Printf("%-6s  no memory probe available (driver/device not found) — skipped\n", backend)
			continue
		}
		p := m.Plan(backend, free, req)
		fmt.Printf("%-6s  %s\n        %s\n", backend, strings.ToUpper(p.Placement.String()), p.Reason)
		if p.Placement != decoder.PlacementDecline {
			admitted = append(admitted, backend)
		}
	}

	if *measure {
		selfMeasure(path, *quant, admitted)
	}
	return 0
}

// selfMeasure is task-fit-to-hardware.md §5's "self-measure" option: after load, decode a fixed
// probe and print the rate as "measured on this machine" — the SAME model this Model was already
// planned against, loaded again on the backend actually chosen so the probe runs through the
// real decode path (not the plan's hypothetical byte arithmetic above). Prefers any admitted
// non-cpu backend over cpu (cpu is always eligible and always last in admitted, so it is only
// picked when nothing else was) — a user asking "how fast will it go" almost always means the
// GPU they compiled in, not the CPU fallback every build has.
func selfMeasure(path, quant string, admitted []string) {
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

	pm, err := decoder.Load(path, decoder.Options{Backend: backend, Quant: quant})
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

	t0 := time.Now()
	out, gen := pm.Generate(context.Background(), prompt, probeDecode, decoder.SamplingParams{Temperature: 0})
	n := 0
	for range out {
		n++
	}
	elapsed := time.Since(t0)
	if gen.Err() != nil {
		fmt.Printf("\n(measure) %s: probe failed: %v\n", backend, gen.Err())
		return
	}
	rate := float64(n) / elapsed.Seconds()
	fmt.Printf("\n(measure) %-6s %.1f tok/s over %d/%d decode steps (includes a %d-token prompt prefill; measured on this machine, not projected)\n",
		backend, rate, n, probeDecode, probePrompt)
}

// freeBytesFor is CompiledBackends' "cpu" special case (HostRAMAvailableBytes, which predates
// the RegisterMemoryProbe registry and lives directly on decoder) plus everything else via the
// registry backends register their own probe into (metal/backend.go, cuda/backend.go).
func freeBytesFor(backend string) (int64, bool) {
	if backend == "cpu" {
		if b := decoder.HostRAMAvailableBytes(); b > 0 {
			return b, true
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
