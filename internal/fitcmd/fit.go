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
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/townsendmerino/goinfer/decoder"
)

const fitUsage = `%[1]s fit <path> — show how this checkpoint would be placed on this machine, per backend

  fit <file.gguf|dir>            report at the default quant/context
  fit <file.gguf|dir> -ctx 32768 report at a specific context (refused if it doesn't fit, not
                                  silently shrunk — pass a smaller -ctx to see what DOES fit)
  fit <file.gguf|dir> -quant int8int8

This loads the checkpoint (unlike a future header-only version): real bytes, real quantization,
same load time %[1]s itself would pay. It reports EVERY backend compiled into this binary —
a CPU-only build only ever reports "cpu"; the metal/cuda release assets report their own GPU
backend too. WebGPU is not reported yet (needs its own KV-precision fix first).
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

	for _, backend := range decoder.CompiledBackends() {
		free, known := freeBytesFor(backend)
		if !known {
			fmt.Printf("%-6s  no memory probe available (driver/device not found) — skipped\n", backend)
			continue
		}
		p := m.Plan(backend, free, req)
		fmt.Printf("%-6s  %s\n        %s\n", backend, strings.ToUpper(p.Placement.String()), p.Reason)
	}
	return 0
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
