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
	"io"
	"os"
	"path/filepath"
	"runtime"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/giw"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// metaPrefixCap bounds how much of the source GGUF we read to extract the
// tokenizer half. The metadata + tensor directory live at the front of the file
// (KVs + tensor infos, no weight bytes), a few MB even for a 256-expert MoE — so
// reading the whole multi-GB model just to slice its header wastes that much heap
// (and OOMs on a 35B). 64 MiB is comfortably more than any real header.
const metaPrefixCap = 64 << 20

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
	// all metadata + tensor infos, no weight bytes. Only the file's head is read
	// (the directory lives there), so this stays a few MB even for a 35B.
	head, err := readHead(in, metaPrefixCap)
	if err != nil {
		return fmt.Errorf("read gguf head: %w", err)
	}
	prefix, err := metadataPrefixLen(head)
	if err != nil {
		return fmt.Errorf("locate gguf metadata (header > %d MB?): %w", metaPrefixCap>>20, err)
	}
	tokBytes := head[:prefix]
	if _, err := tokenizer.LoadGGUFBytes(tokBytes); err != nil {
		return fmt.Errorf("metadata GGUF does not load a tokenizer: %w", err)
	}

	// 2) Weights half: load at the target quant and stream the bundle directly to
	// the output file. Streaming (vs build-blob-then-write) keeps peak RAM at ~the
	// resident weight size instead of 2×+ — the difference between fitting and
	// thrashing for a 35B (each of resident/blob/bundle is tens of GB).
	m, err := decoder.Load(in, decoder.Options{Quant: quant})
	if err != nil {
		return fmt.Errorf("load model: %w", err)
	}
	f, err := os.Create(out)
	if err != nil {
		_ = m.Close()
		return fmt.Errorf("create %s: %w", out, err)
	}
	werr := giw.WriteStream(f, tokBytes, func(w io.Writer) (int64, error) {
		return decoder.SerializeWeightsTo(w, m.Weights(), filepath.Base(in))
	})
	_ = m.Close()
	m = nil
	runtime.GC()
	if cerr := f.Close(); werr == nil {
		werr = cerr
	}
	if werr != nil {
		_ = os.Remove(out)
		return fmt.Errorf("write bundle: %w", werr)
	}

	// 3) Verify the bundle round-trips through the real (mmap) load path — cheap
	// (lazy faults), and confirms the streamed weights deserialize. The tokenizer
	// half was already validated at step 1.
	if err := selfCheck(out); err != nil {
		_ = os.Remove(out)
		return fmt.Errorf("self-check: %w", err)
	}

	fi, _ := os.Stat(out)
	mb := func(n int64) float64 { return float64(n) / 1048576 }
	fmt.Fprintf(os.Stderr, "wrote %s: %.0f MB (%s weights + tokenizer %.1f MB)\n",
		out, mb(fi.Size()), quant, mb(int64(len(tokBytes))))
	return nil
}

// selfCheck verifies a freshly written bundle loads through the real mmap path
// (lazy, so it doesn't pull the whole file into RAM). It confirms the streamed
// weights deserialize; the tokenizer half is validated earlier, before the load.
func selfCheck(path string) error {
	m, err := decoder.Load(path, decoder.Options{})
	if err != nil {
		return err
	}
	return m.Close()
}

// readHead reads up to capBytes from the front of path (a GGUF's metadata + tensor
// directory live at the front, before the weight data). Returns fewer bytes for a
// smaller file.
func readHead(path string, capBytes int) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}
	n := capBytes
	if sz := int(fi.Size()); sz < n {
		n = sz
	}
	buf := make([]byte, n)
	if _, err := io.ReadFull(f, buf); err != nil {
		return nil, err
	}
	return buf, nil
}
