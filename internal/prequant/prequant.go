// Package prequant builds a goinfer prequant bundle (.giw) from a GGUF model: it
// loads the model at a fixed quant, streams the already-quantized resident weights
// to disk, and pairs them with a metadata-only GGUF (the source truncated at the
// tensor-data boundary) that carries the tokenizer. Shared by cmd/prequant (the
// explicit CLI) and the serve-side transparent cache (EnsureCachedGIW), so a user
// who points --stream-weights at a plain .gguf never has to run prequant by hand.
package prequant

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/giw"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// metaPrefixCap bounds how much of the source GGUF we read to extract the tokenizer
// half. The metadata + tensor directory live at the front of the file (KVs + tensor
// infos, no weight bytes), a few MB even for a 256-expert MoE — so reading the whole
// multi-GB model just to slice its header wastes that much heap (and OOMs on a 35B).
// 64 MiB is comfortably more than any real header.
const metaPrefixCap = 64 << 20

// Transcode writes a .giw bundle at out from the GGUF at in, quantized to quant
// ("int8int8" | "int8" | "int4" | "" for f32). It streams the weights straight to
// the file (peak RAM ≈ the resident weight size, not 2×+), so it fits a 35B on a
// modest box. A failed write removes the partial output.
func Transcode(in, out, quant string) error {
	// 1) Tokenizer half: the source GGUF truncated at the tensor-data boundary —
	// metadata + tensor infos, no weight bytes. Only the file's head is read.
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
	// the output file, freeing the resident model before the self-check so peak RAM
	// stays ~resident, not resident+blob+bundle.
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
	// (lazy faults), confirms the streamed weights deserialize.
	if err := selfCheck(out); err != nil {
		_ = os.Remove(out)
		return fmt.Errorf("self-check: %w", err)
	}
	return nil
}

// EnsureCachedGIW returns a .giw for the GGUF at ggufPath quantized to quant,
// transcoding once into a sidecar cache (alongside the GGUF, "<base>.<quant>.giw")
// when no fresh cache exists. The cache is fresh when it's newer than the source,
// so replacing the GGUF rebuilds it. The one-time transcode is logged to stderr
// (it can take minutes and write tens of GB) so a slow first start isn't mistaken
// for a hang. Returns the .giw path to load.
func EnsureCachedGIW(ggufPath, quant string) (string, error) {
	cache := streamCachePath(ggufPath, quant)
	if cacheFresh(cache, ggufPath) {
		return cache, nil
	}
	fmt.Fprintf(os.Stderr, "stream-weights: transcoding %s → %s (%s, one-time — minutes + ~model-size on disk)…\n",
		filepath.Base(ggufPath), filepath.Base(cache), quantLabel(quant))
	t0 := time.Now()
	if err := Transcode(ggufPath, cache, quant); err != nil {
		return "", err
	}
	if fi, e := os.Stat(cache); e == nil {
		fmt.Fprintf(os.Stderr, "stream-weights: cache ready (%.0f MB) in %s\n",
			float64(fi.Size())/1048576, time.Since(t0).Round(time.Second))
	}
	return cache, nil
}

// streamCachePath is the sidecar cache for a GGUF at a quant: "<base>.<quant>.giw".
func streamCachePath(ggufPath, quant string) string {
	base := ggufPath[:len(ggufPath)-len(filepath.Ext(ggufPath))]
	return base + "." + quantLabel(quant) + ".giw"
}

// cacheFresh reports whether cache exists and is newer than src (so an updated GGUF
// invalidates a stale cache).
func cacheFresh(cache, src string) bool {
	cs, err1 := os.Stat(cache)
	ss, err2 := os.Stat(src)
	return err1 == nil && err2 == nil && cs.ModTime().After(ss.ModTime())
}

func quantLabel(q string) string {
	if q == "" {
		return "f32"
	}
	return q
}

// selfCheck verifies a freshly written bundle loads through the real mmap path
// (lazy, low RAM) — the streamed weights deserialize.
func selfCheck(path string) error {
	m, err := decoder.Load(path, decoder.Options{})
	if err != nil {
		return err
	}
	return m.Close()
}

// readHead reads up to capBytes from the front of path (a GGUF's metadata + tensor
// directory live at the front). Returns fewer bytes for a smaller file.
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
