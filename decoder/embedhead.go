package decoder

import (
	"github.com/townsendmerino/goinfer/internal/runtime/wbuild"
	"os"
	"strings"

	"github.com/townsendmerino/goinfer/decoder/fit"
)

// embedInt4DefaultMaxParams is the model size, in total parameters, from which the default output head is int8 rather than
// int4 (EmbedInt4For). It sits between the largest model whose int4 head held long-context agreement and the smallest whose
// did not (docs/quantization.md, "--embed-int4"). A variable so a test can move it.
var embedInt4DefaultMaxParams int64 = 1e9

// EmbedInt4For is the head precision a load of src takes under opts: opts.EmbedInt4, except that when it is the caller's
// default (EmbedInt4Default) and src has embedInt4DefaultMaxParams parameters or more, the default int4 head becomes the
// int8 one. A size it cannot read (a .giw, whose head is baked, or an unreadable file) keeps opts.EmbedInt4. It reads only
// the file's tensor directory.
func EmbedInt4For(src string, opts Options) bool {
	if !opts.EmbedInt4 || !opts.EmbedInt4Default {
		return opts.EmbedInt4
	}
	n := PeekParamCount(src)
	return n == 0 || n < embedInt4DefaultMaxParams
}

// resolveEmbedInt4 applies EmbedInt4For to opts for src, after which EmbedInt4 is the choice, not a default.
func resolveEmbedInt4(src string, opts Options) Options {
	opts.EmbedInt4, opts.EmbedInt4Default = EmbedInt4For(src, opts), false
	return opts
}

// PeekParamCount is the total parameter count of a .gguf or a safetensors checkpoint directory, from its tensor shapes
// alone (a tied head counted once, as the file stores it), or 0 when src is neither or cannot be read. A safetensors
// count unpacks quantized layouts as the fit estimate does (estimateSafetensorsWeightBytes at one byte per element),
// which counts a 1-D or vision-tower tensor four times: noise beside the text matrices.
func PeekParamCount(src string) int64 {
	if strings.HasSuffix(src, ".gguf") {
		g, err := OpenGGUFMmap(src)
		if err != nil {
			return 0
		}
		defer g.Close()
		var n int64
		for _, name := range g.Names() {
			d, ok := g.Dims(name)
			if !ok {
				continue
			}
			e := int64(1)
			for _, x := range d {
				e *= int64(x)
			}
			n += e
		}
		return n
	}
	if strings.HasSuffix(src, ".giw") {
		return 0
	}
	return estimateSafetensorsParams(src)
}

// estimateSafetensorsParams is PeekParamCount for a safetensors directory.
func estimateSafetensorsParams(dir string) int64 {
	st, err := openCheckpointMmap(dir)
	if err != nil {
		return 0
	}
	defer st.Close()
	r := fit.Rates{Matrix: 1, Embedding: 1, Int8: 1}
	if cfg, cerr := loadConfig(os.DirFS(dir), "config.json"); cerr == nil {
		if qc, qerr := wbuild.ParseCheckpointQuantConfig(cfg); qerr == nil && qc != nil && qc.Method == "mlx" {
			r.MLXBits = qc.Bits
		}
	}
	return fit.SafetensorsWeightBytes(st.Names(), func(name string) ([]int, bool) {
		t, err := st.Tensor(name)
		if err != nil {
			return nil, false
		}
		return t.Shape, true
	}, r)
}
