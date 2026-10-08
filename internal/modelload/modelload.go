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
	"io"
	"os"
	"path/filepath"
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

// loadNoticeOut is where loadGuarded says a load is not covered by the tripwire; a variable so a test can read it.
var loadNoticeOut io.Writer = os.Stderr

// Load runs the whole path from a --model value to a loaded model. The caller owns Result.Model.
func Load(ctx context.Context, req Request) (*Result, error) {
	src, err := Resolve(ctx, req.Spec)
	if err != nil {
		return nil, err
	}
	opts := req.Opts
	// A merged LoRA needs a safetensors base (PEFT targets HF module names). A .gguf used to reach
	// decoder.Load's own refusal, but with the sidecar default it becomes a .giw first, and a .giw has
	// no base to merge into — the adapter was dropped without a word. Refuse before any transcode.
	if opts.LoRA != "" && (strings.HasSuffix(src, ".gguf") || strings.HasSuffix(src, ".giw")) {
		return nil, fmt.Errorf("--lora %s: a LoRA adapter is merged into a safetensors base at load, and %q is not one (a .gguf or .giw is already quantized) — pass the base model's safetensors directory, or use serve's --adapter", opts.LoRA, src)
	}
	// Before a sidecar is chosen: a .giw bakes its quant, so this cannot be fixed after the transcode.
	var msg string
	if opts.Quant, opts.ActQuantGroup, msg = activationSafeQuant(src, opts.Quant, opts.ActQuantGroup, req.ExplicitQuant, opts.Backend); msg != "" {
		fmt.Fprintln(os.Stderr, msg)
	}
	ensureGIW := func() (string, error) {
		return prequant.EnsureCachedGIW(ctx, src, opts.Quant, opts.Backend, opts.EmbedInt4)
	}

	// Weight streaming needs the read-only mmap only a .giw provides, so a .gguf is transcoded to a
	// sidecar once. Without streaming, a .gguf still resolves to its sidecar by default (S1,
	// task-never-swap-2026-09.md) so resident weights are zero-copy mmap aliases rather than heap
	// copies. -embed-int4 bakes its int4 head into the sidecar too (streamCachePath's "e4h" cache
	// key keeps it distinct from a plain-head sidecar of the same source and quant) — it no longer
	// forces a direct load.
	loadPath := src
	if opts.Quant == "q4k" && strings.HasSuffix(src, ".gguf") {
		// q4k has no .giw form yet (docs/tasks/task-int4-weight-quality-2026-09.md): load the .gguf
		// directly, never through the sidecar cache.
		if opts.StreamWeights {
			return nil, fmt.Errorf("--quant q4k cannot stream weights yet (it has no .giw form); drop -stream-weights")
		}
	} else if opts.StreamWeights {
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
	} else if strings.HasSuffix(src, ".gguf") && prequant.DefaultToSidecar(req.DirectLoad) {
		if loadPath, err = ensureGIW(); err != nil {
			return nil, fmt.Errorf("sidecar cache (%s): %w — pass -direct-load (or GOINFER_GGUF_DIRECT=1) to load this .gguf straight into the heap instead", src, err)
		}
	} else if dirSidecarApplies(src, opts) && prequant.DefaultToSidecar(req.DirectLoad) {
		// S18 (docs/tasks/task-multimodal-support-2026-10.md): a safetensors directory resolves to a sidecar too, so its
		// weights are mmap aliases rather than a heap copy, which on Metal's unified memory is the difference between the
		// weights counted once and counted twice. The one-time build loads the directory whole (no streaming transcode
		// for safetensors yet), so a build that fails, the fit guard's refusal included, keeps today's direct load.
		if p, gerr := ensureGIW(); gerr == nil {
			loadPath = p
		} else {
			fmt.Fprintf(os.Stderr, "note: loading %s directly: its sidecar .giw could not be built (%v)\n", src, gerr)
		}
	}

	// A directory's tokenizer stays the directory's: a sidecar's tok half is its tokenizer.json alone, and
	// tokenizer_config.json (the chat template, the BOS/EOS flags) is beside it, not in it.
	tokPath := loadPath
	if src != loadPath && isDir(src) {
		tokPath = src
	}
	tk, err := Tokenizer(tokPath)
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

// dirSidecarApplies reports whether a safetensors directory source takes the sidecar default (S18): not with a LoRA to
// merge (the merge needs the safetensors base, and a plain sidecar would drop the adapter), not at q4k (no .giw form),
// and not under -stream-weights, which serve already explains is .gguf-only for a directory.
func dirSidecarApplies(src string, opts decoder.Options) bool {
	if opts.LoRA != "" || opts.Quant == "q4k" || opts.StreamWeights || !isDir(src) {
		return false
	}
	m, _ := filepath.Glob(filepath.Join(src, "*.safetensors"))
	return len(m) > 0
}

func isDir(p string) bool {
	fi, err := os.Stat(p)
	return err == nil && fi.IsDir()
}

// activationSafeQuant is the precision and activation group a load should use for src. For a family
// that per-vector int8 activations break (decoder.ActivationQuantHazard), the default becomes:
//   - q4k for a .gguf on the CPU or CUDA backend: the file's Q4_K tensors exact, the rest int8,
//     per-32 (docs/tasks/task-int4-weight-quality-2026-09.md: on phi3-mini, CPU decode 1.31× and
//     CUDA resident decode 1.26× / 1.17× (depth 128 / 2048) the int8int8 default's, and CUDA fits the
//     default context resident where int8int8 does not);
//   - int8int8 with per-32 activation scales anywhere else (Metal, WebGPU: no Q4_K kernel yet), which
//     those backends run resident (docs/tasks/task-actquant-pergroup-2026-09.md).
//
// An explicit int8int8 gets per-32 too. An explicit int4/int4mix is honoured with a warning, because
// per-32 does not clear int4's weight error. msg is "" when nothing applies.
func activationSafeQuant(src, quant string, group int, explicitQuant, backend string) (string, int, string) {
	why := decoder.ActivationQuantHazard(decoder.PeekModelType(src))
	if why == "" || !decoder.QuantizesActivations(quant) || group == 32 && quant == "int8int8" || quant == "q4k" {
		// q4k is per-32 throughout by construction (decoder.modelFromOptions stamps it).
		return quant, group, ""
	}
	switch {
	case explicitQuant == "" && strings.HasSuffix(src, ".gguf") && (backend == "" || backend == "cpu" || backend == "cuda"):
		return "q4k", 0, fmt.Sprintf("note: loading at --quant q4k (the file's Q4_K tensors kept exact, the rest int8, per-32 activations) instead of the default %s: %s.", quant, why)
	case explicitQuant == "":
		return "int8int8", 32, fmt.Sprintf("note: loading at --quant int8int8 with per-32 activation scales instead of the default %s: %s.", quant, why)
	case quant == "int8int8":
		return quant, 32, fmt.Sprintf("note: using per-32 activation scales for --quant int8int8: %s.", why)
	}
	return quant, group, fmt.Sprintf("warning: --quant %s quantizes activations to int8, and %s; expect degraded output (--quant int8int8 runs it with per-32 activation scales).", quant, why)
}

// loadGuarded is decoder.Load under S3's load-time swap tripwire (docs/tasks/task-never-swap-2026-09.md).
// Only a .gguf direct build observes Options.LoadAbort, so a .giw / HF-dir / streamed load arms nothing.
func loadGuarded(path string, opts decoder.Options, advice string) (*decoder.Model, error) {
	loadOpts := opts
	wrapLoadErr := func(err error) error { return err }
	stopLoadGuard := func() {}
	if !opts.StreamWeights && strings.HasSuffix(path, ".gguf") {
		loadOpts.LoadAbort, wrapLoadErr, stopLoadGuard = startLoadGuard(path, opts, advice)
	} else {
		// Say so: a cold-user run (2026-10-05, gemma-4-26b-a4b) loaded a non-.gguf source for 20 s with an
		// empty log, and its operator could not tell "guarded and quiet" from "not guarded at all".
		fmt.Fprintf(loadNoticeOut, "swap guard (load): not armed for %s — only a direct .gguf build is watched, "+
			"so a swap-growing load is not stopped here; the serving watch arms once the server is up\n", path)
	}
	model, err := decoder.Load(path, loadOpts)
	stopLoadGuard() // done with this attempt either way — never left running through a retry
	return model, wrapLoadErr(err)
}
