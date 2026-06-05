// Command prequant builds a goinfer prequant bundle (.giw) from a GGUF model.
//
// It loads the model at a fixed quant (default int8int8), serializes the
// already-quantized resident weights, and pairs them with a metadata-only GGUF
// (the source truncated at the tensor-data boundary) that carries the tokenizer.
// The demo's -tags prequant build embeds the bundle and loads it with NO
// dequant/requant — the int8 weights are aliased straight from the binary image,
// so a 4B model no longer needs a multi-GB heap copy on every launch.
//
// Usage:
//
//	go run ./cmd/prequant -o demo/chat/model.giw ~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/giw"
	"github.com/townsendmerino/goinfer/tokenizer"
)

func main() {
	out := flag.String("o", "", "output .giw bundle path (required)")
	quant := flag.String("quant", "int8int8", "weight quant baked into the bundle: int8int8 | int8 | int4")
	flag.Parse()
	in := flag.Arg(0)
	if in == "" || *out == "" {
		fmt.Fprintln(os.Stderr, "usage: prequant -o <out.giw> <model.gguf>")
		flag.Usage()
		os.Exit(2)
	}
	if err := run(in, *out, *quant); err != nil {
		fmt.Fprintf(os.Stderr, "prequant: %v\n", err)
		os.Exit(1)
	}
}

func run(in, out, quant string) error {
	// 1) Tokenizer half: the source GGUF truncated at the tensor-data boundary —
	// all metadata + tensor infos, no weight bytes.
	raw, err := os.ReadFile(in)
	if err != nil {
		return fmt.Errorf("read gguf: %w", err)
	}
	prefix, err := metadataPrefixLen(raw)
	if err != nil {
		return fmt.Errorf("locate gguf metadata: %w", err)
	}
	tokBytes := raw[:prefix]
	if _, err := tokenizer.LoadGGUFBytes(tokBytes); err != nil {
		return fmt.Errorf("metadata GGUF does not load a tokenizer: %w", err)
	}

	// 2) Weights half: load at the target quant and serialize the resident bundle.
	m, err := decoder.Load(in, decoder.Options{Quant: quant})
	if err != nil {
		return fmt.Errorf("load model: %w", err)
	}
	blob, err := decoder.SerializeWeights(m.Weights(), filepath.Base(in))
	if err != nil {
		return fmt.Errorf("serialize weights: %w", err)
	}

	bundle := giw.Write(blob, tokBytes)

	// 3) Verify the bundle round-trips before writing it.
	wb, tb, err := giw.Read(bundle)
	if err != nil {
		return fmt.Errorf("self-check: %w", err)
	}
	if _, err := decoder.LoadSerializedWeights(wb); err != nil {
		return fmt.Errorf("self-check weights: %w", err)
	}
	if _, err := tokenizer.LoadGGUFBytes(tb); err != nil {
		return fmt.Errorf("self-check tokenizer: %w", err)
	}

	if err := os.WriteFile(out, bundle, 0o644); err != nil {
		return fmt.Errorf("write %s: %w", out, err)
	}
	mb := func(n int) float64 { return float64(n) / 1048576 }
	fmt.Fprintf(os.Stderr, "wrote %s: %.0f MB (weights %.0f MB %s + tokenizer %.1f MB)\n",
		out, mb(len(bundle)), mb(len(blob)), quant, mb(len(tokBytes)))
	return nil
}
