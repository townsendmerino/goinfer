// Command prequant builds a goinfer prequant bundle (.giw) from a GGUF model.
//
// It loads the model at a fixed quant (default int8int8), serializes the
// already-quantized resident weights, and pairs them with a metadata-only GGUF
// (the source truncated at the tensor-data boundary) that carries the tokenizer.
// The demo's -tags prequant build embeds the bundle and loads it with NO
// dequant/requant — the int8 weights are aliased straight from the binary image,
// so a 4B model no longer needs a multi-GB heap copy on every launch. serve also
// builds these on the fly for --stream-weights (see internal/prequant.EnsureCachedGIW).
//
// Usage:
//
//	go run ./cmd/prequant -o demo/chat/model.giw ~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/prequant"
)

func main() {
	out := flag.String("o", "", "output .giw bundle path (required)")
	quant := flag.String("quant", "int8int8", "weight quant baked into the bundle: int8int8 | int8 | int4")
	embedInt4 := flag.Bool("embed-int4", false, "in int4 mode, quantize the token-embedding/LM-head to int4 too (else int8-pinned); ~½ the head's per-token traffic, coherence-safe on big-vocab models (verified on gemma4-26b: trigram 0.911, Paris survives)")
	targetFlag := flag.String("target", "", ".giw target: cpu-arm64 | cpu-amd64 | metal | cuda | webgpu | cpu | canonical (docs/task-int4-layout-2026-09.md's L2). Default \"\" resolves like \"cpu\": this build box's own CPU arch. Only cpu-arm64 currently changes what's written — every eligible int4 tensor (dense Q/K/V/gate/up/down projections, embed/lm_head, router, shared-expert(+gate), gemma4 PLE; MoE-paged experts and a few not-yet-scoped mixer projections stay kind 3 always, see decoder/serialize.go's weightMatKind3Only) bakes kind 5 (row4-only, NO canonical bytes at all) for shapes RepackW4A8Row4/RepackW4A8Row4Scales accept — every other shape/target still writes kind 3 (canonical), same as today. Bit-identical dispatch either way — this changes the bundle's size and load characteristics, never decode output. A kind-5 bundle is a promise to ONE core: loading it under a different arch, or under a GPU backend that needs canonical, fails loudly at load rather than silently. Pass canonical explicitly for a bundle more than one consumer/arch will read (e.g. one baked once and embedded into several cross-compiled release binaries) — one .giw always serves exactly one target.")
	flag.Parse()
	in := flag.Arg(0)
	if in == "" || *out == "" {
		fmt.Fprintln(os.Stderr, "usage: prequant -o <out.giw> <model.gguf | safetensors-dir>")
		flag.Usage()
		os.Exit(2)
	}
	target, terr := decoder.ParseGIWTarget(*targetFlag)
	if terr != nil {
		fmt.Fprintf(os.Stderr, "prequant: %v\n", terr)
		os.Exit(2)
	}
	// Ctrl-C / SIGTERM aborts the (minutes-long, tens-of-GB) transcode and removes the
	// partial .giw, instead of running to completion after the user gives up (audit M-21).
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := prequant.Transcode(ctx, in, *out, *quant, *embedInt4, target); err != nil {
		fmt.Fprintf(os.Stderr, "prequant: %v\n", err)
		os.Exit(1)
	}
	if fi, err := os.Stat(*out); err == nil {
		fmt.Fprintf(os.Stderr, "wrote %s: %.0f MB (%s)\n", *out, float64(fi.Size())/1048576, *quant)
	}
}
