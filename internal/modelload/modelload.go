// Package modelload is the one path serve, chat and fit take from a --model value to a loaded
// decoder.Model and its tokenizer. Each app used to do this its own way, and the copies drifted:
// serve's .giw tokenizer fallback still swallowed the GGUF error chat had fixed as N-25, fit did not
// resolve hf:/demo: references at all, and serve alone retried a fit decline with weight streaming.
// The steps, in order: resolve a reference, pick the path to load (sidecar .giw or streaming
// transcode for a .gguf), load the tokenizer, load the model under the swap guard, retry once with
// streaming on a dense fit decline, and refuse an explicit --quant a prequant .giw cannot honour.
package modelload

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/giw"
	"github.com/townsendmerino/goinfer/internal/prequant"
	"github.com/townsendmerino/goinfer/internal/swapguard"
	"github.com/townsendmerino/goinfer/pull"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// Resolve turns a --model value into a local path: an hf:/demo: reference is fetched (or found in the
// cache) with progress on stderr; anything else is returned untouched, so no existing path changes
// meaning.
func Resolve(ctx context.Context, spec string) (string, error) {
	if !pull.IsRef(spec) {
		return spec, nil
	}
	path, err := pull.ResolveVerbose(ctx, spec)
	if err != nil {
		return "", fmt.Errorf("resolving %s: %w", spec, err)
	}
	return path, nil
}

// Tokenizer loads the tokenizer for a model path, by extension: a .giw carries it in its tok half, a
// .gguf in its own metadata, and anything else is an HF / SentencePiece directory.
func Tokenizer(path string) (*tokenizer.Tokenizer, error) {
	switch {
	case strings.HasSuffix(path, ".giw"):
		raw, err := giw.ReadTokFile(path)
		if err != nil {
			return nil, err
		}
		return TokenizerFromTok(raw)
	case strings.HasSuffix(path, ".gguf"):
		return tokenizer.LoadGGUF(path)
	default:
		return tokenizer.Load(path)
	}
}

// TokenizerFromTok parses a .giw bundle's tok half, which is GGUF metadata for a GGUF-sourced bundle
// and the raw tokenizer.json for a safetensors-sourced one. When neither parses, BOTH errors are
// reported (N-25): with only the JSON one, a corrupt GGUF-sourced bundle reads as "invalid JSON" and
// sends the reader to the wrong half of the file.
func TokenizerFromTok(raw []byte) (*tokenizer.Tokenizer, error) {
	tk, gerr := tokenizer.LoadGGUFBytes(raw)
	if gerr == nil {
		return tk, nil
	}
	tk, jerr := tokenizer.LoadJSONBytes(raw)
	if jerr != nil {
		return nil, fmt.Errorf("not a GGUF (%v) and not tokenizer.json (%v)", gerr, jerr)
	}
	return tk, nil
}

// Request is one model load. Opts is final except for StreamWeights, which Load may turn on (see
// Result.Opts).
type Request struct {
	Spec string          // the --model value: a path, or an hf:/demo: reference
	Opts decoder.Options // backend, quant, knobs, … as the app resolved them
	// DirectLoad is -direct-load / GOINFER_GGUF_DIRECT: load a .gguf straight into the heap instead of
	// through its sidecar .giw (prequant.DefaultToSidecar).
	DirectLoad bool
	// ExplicitQuant is the --quant the user actually typed ("" when it is the default): a prequant .giw
	// carries its own quant, so an explicit different one is refused rather than silently ignored (T1-7).
	ExplicitQuant string
	// GuardAdvice is appended to the swap guard's abort message: what the user can do instead.
	GuardAdvice string
}

// Result is a loaded model.
type Result struct {
	Source    string // the resolved source path (a reference already fetched); name and fingerprint derive from this
	LoadPath  string // what was actually loaded: Source, or its sidecar / streaming .giw
	Tokenizer *tokenizer.Tokenizer
	Model     *decoder.Model
	Opts      decoder.Options // as loaded — StreamWeights is true after the automatic streaming retry
	LoadTime  time.Duration   // the decoder.Load call that succeeded (not resolve, transcode or tokenizer)
}

// startLoadGuard is swapguard.StartLoad, as a variable so a test can hand Load an already-tripped guard
// and check the abort really reaches decoder.Load (a guard armed but never passed through would be
// silent in every other test).
var startLoadGuard = swapguard.StartLoad

// Load runs the whole path from a --model value to a loaded model. The caller owns Result.Model.
func Load(ctx context.Context, req Request) (*Result, error) {
	src, err := Resolve(ctx, req.Spec)
	if err != nil {
		return nil, err
	}
	opts := req.Opts
	ensureGIW := func() (string, error) {
		if opts.EmbedInt4 {
			fmt.Fprintln(os.Stderr, "note: embed-int4 is ignored with stream-weights (the cached .giw keeps the int8 pin); prequant the model with embed-int4 to bake it")
		}
		return prequant.EnsureCachedGIW(ctx, src, opts.Quant, opts.Backend)
	}

	// Weight streaming needs the read-only mmap only a .giw provides, so a .gguf is transcoded to a
	// sidecar once. Without streaming, a .gguf still resolves to its sidecar by default (S1,
	// task-never-swap-2026-09.md) so resident weights are zero-copy mmap aliases rather than heap
	// copies; -embed-int4 implies a direct load there rather than silently losing its int4 embed pin.
	loadPath := src
	if opts.StreamWeights {
		if strings.HasSuffix(src, ".gguf") {
			if loadPath, err = ensureGIW(); err != nil {
				return nil, fmt.Errorf("stream-weights cache (%s): %w", src, err)
			}
		} else if fi, serr := os.Stat(src); serr == nil && fi.IsDir() {
			// M-30: -stream-weights is a no-op for a safetensors directory; say so rather than leave a
			// later fit refusal reading as "should have worked".
			fmt.Fprintf(os.Stderr, "note: -stream-weights only applies to a .gguf source; %q is a "+
				"safetensors directory — see cmd/prequant to build a streamable .giw from it\n", src)
		}
	} else if strings.HasSuffix(src, ".gguf") && !opts.EmbedInt4 && prequant.DefaultToSidecar(req.DirectLoad) {
		if loadPath, err = ensureGIW(); err != nil {
			return nil, fmt.Errorf("sidecar cache (%s): %w — pass -direct-load (or GOINFER_GGUF_DIRECT=1) to load this .gguf straight into the heap instead", src, err)
		}
	}

	tk, err := Tokenizer(loadPath)
	if err != nil {
		return nil, fmt.Errorf("load tokenizer (%s): %w", loadPath, err)
	}
	t0 := time.Now()
	model, err := loadGuarded(loadPath, opts, req.GuardAdvice)
	if err != nil {
		// One automatic retry with weight streaming for a plain .gguf that does not fit resident RAM
		// (tasks/task-fit-to-hardware.md's CPU placement piece) — unless the model is MoE or an
		// own-forward family (FitDeclineError.DenseStreamable: CPU expert paging is a measured
		// failure), streaming was already on, or --fit=off asked for today's refusal instead.
		var fde *decoder.FitDeclineError
		if opts.StreamWeights || opts.DisableFit || !strings.HasSuffix(src, ".gguf") ||
			!errors.As(err, &fde) || !fde.DenseStreamable {
			return nil, fmt.Errorf("load model (%s): %w", loadPath, err)
		}
		fmt.Fprintf(os.Stderr, "note: %q does not fit resident RAM; automatically retrying with weight streaming (pass --fit=off to keep today's refusal instead)\n", src)
		declineErr := err
		opts.StreamWeights = true
		if loadPath, err = ensureGIW(); err != nil {
			return nil, fmt.Errorf("load model (%s): %w (auto weight-streaming retry also failed: %v)", src, declineErr, err)
		}
		if tk, err = Tokenizer(loadPath); err != nil {
			return nil, fmt.Errorf("load tokenizer (%s): %w", loadPath, err)
		}
		t0 = time.Now()
		if model, err = decoder.Load(loadPath, opts); err != nil {
			return nil, fmt.Errorf("load model (%s): %w (auto weight-streaming retry also failed after transcode: %v)", src, declineErr, err)
		}
	}
	loadTime := time.Since(t0)
	if err := model.CheckGiwQuantMatch(req.ExplicitQuant); err != nil {
		model.Close()
		return nil, fmt.Errorf("--model %q: %w", req.Spec, err)
	}
	return &Result{Source: src, LoadPath: loadPath, Tokenizer: tk, Model: model, Opts: opts, LoadTime: loadTime}, nil
}

// loadGuarded is decoder.Load under S3's load-time swap tripwire (docs/tasks/task-never-swap-2026-09.md).
// Only a .gguf direct build observes Options.LoadAbort, so a .giw / HF-dir / streamed load arms nothing.
func loadGuarded(path string, opts decoder.Options, advice string) (*decoder.Model, error) {
	loadOpts := opts
	wrapLoadErr := func(err error) error { return err }
	stopLoadGuard := func() {}
	if !opts.StreamWeights && strings.HasSuffix(path, ".gguf") {
		loadOpts.LoadAbort, wrapLoadErr, stopLoadGuard = startLoadGuard(path, opts, advice)
	}
	model, err := decoder.Load(path, loadOpts)
	stopLoadGuard() // done with this attempt either way — never left running through a retry
	return model, wrapLoadErr(err)
}
