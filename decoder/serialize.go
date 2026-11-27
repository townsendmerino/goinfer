package decoder

import (
	"fmt"
	"io"

	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/giwcodec"
	"github.com/townsendmerino/goinfer/internal/runtime/logs"
	"github.com/townsendmerino/goinfer/internal/runtime/wbuild"
)

// GIWFormatVersion is the weights format version this build writes and reads.
const GIWFormatVersion = giwcodec.FormatVersion

const (
	giwMagic              = giwcodec.Magic
	giwVersion            = giwcodec.Version
	giwEncF32             = giwcodec.EncF32
	giwKindF32            = giwcodec.KindF32
	maxSerializedLayers   = giwcodec.MaxSerializedLayers
	maxSnapshotCacheBytes = giwcodec.MaxSnapshotCacheBytes
)

// SerializeError is returned by LoadSerializedWeights on any format/CRC/shape mismatch.
type SerializeError = giwcodec.SerializeError

var errCopiedF32Bundle = giwcodec.ErrCopiedF32Bundle

// giwF32Kind1ForTest forces f32 weight matrices to be written and accepted as kind 1 (copied f32).
var giwF32Kind1ForTest bool

// giwIslandsF32ForTest writes bf16 island matrices as standalone f32 arrays for round-trip tests.
var giwIslandsF32ForTest bool

// giwWriter wraps wbuild.StreamWriter so decoder/session_giwkv.go can attach snapshot helpers.
type giwWriter struct {
	wbuild.StreamWriter
}

func newGIWWriter(sink io.Writer, target GIWTarget, layerBudget int64, hooks wbuild.StreamTestHooks) *giwWriter {
	w := &giwWriter{StreamWriter: *wbuild.NewStreamWriter(sink, target, layerBudget, hooks, wbuildHooks())}
	w.InitHeadTable()
	return w
}

// giwReader wraps giwcodec.Reader so decoder/session_giwkv.go can attach snapshot helpers.
type giwReader struct {
	giwcodec.Reader
}

// SerializeWeights writes the resident weight bundle to a flat little-endian blob.
func SerializeWeights(w *darch.Weights, id string) ([]byte, error) {
	return giwcodec.SerializeWeights(w, id, giwF32Kind1ForTest, giwIslandsF32ForTest)
}

// SerializeWeightsTo streams the weight bundle directly to out.
func SerializeWeightsTo(out io.Writer, w *darch.Weights, id string) (int64, error) {
	return giwcodec.SerializeWeightsTo(out, w, id, giwF32Kind1ForTest, giwIslandsF32ForTest)
}

// SerializeWeightsForTarget is SerializeWeights for a single target consumer.
func SerializeWeightsForTarget(w *darch.Weights, id string, target GIWTarget) ([]byte, error) {
	return giwcodec.SerializeWeightsForTarget(w, id, target, giwF32Kind1ForTest, giwIslandsF32ForTest)
}

// SerializeWeightsToForTarget streams a target-aware weight bundle directly to out.
func SerializeWeightsToForTarget(out io.Writer, w *darch.Weights, id string, target GIWTarget) (int64, error) {
	return giwcodec.SerializeWeightsToForTarget(out, w, id, target, giwF32Kind1ForTest, giwIslandsF32ForTest)
}

// LoadSerializedWeights reconstructs a *Weights from a SerializeWeights blob.
func LoadSerializedWeights(data []byte) (*darch.Weights, error) {
	return giwcodec.LoadSerializedWeights(data, giwF32Kind1ForTest)
}

func loadSerializedWeightsAbortable(data []byte, crcAlreadyVerified bool, beforeWalk func(*darch.Architecture), abort <-chan struct{}) (*darch.Weights, error) {
	return giwcodec.LoadSerializedWeightsAbortable(data, crcAlreadyVerified, beforeWalk, abort, giwF32Kind1ForTest, ErrLoadAborted)
}

func readGIWHeadReader(data []byte) (*giwReader, []byte, string, error) {
	r, cfgJSON, baked, err := giwcodec.ReadHeadReader(data)
	if err != nil {
		return nil, nil, "", err
	}
	return &giwReader{Reader: *r}, cfgJSON, baked, nil
}

func scanGIWWeights(blob []byte) (string, error) { return giwcodec.ScanWeights(blob) }

// Weights exposes the loaded weight bundle, e.g. so a build-time tool can SerializeWeights(m.Weights(), ...). It returns
// the live bundle (a defensive copy would duplicate gigabytes and the device buffers), so the forward pass depends on it
// being treated as IMMUTABLE: the derived *Architecture, RoPE tables and resident buffers are built from it at load and
// are not rebuilt, so any mutation silently desyncs them. Read-only.
func (m *Model) Weights() *darch.Weights { return m.w }

// Quant names the precision the model's matmul weights are resident in
// ("int8int8" | "int8" | "int4" | "native"). For a prequant model the runtime
// --quant flag is moot — this reports what was actually baked in.
func (m *Model) Quant() string {
	// A direct load records the requested quant — accurate for the KV-snapshot
	// fingerprint (so int4 / int4mix / int8 don't collide on the same file). A
	// prequant .giw leaves it empty and derives from the resident weight kinds.
	switch m.quant {
	case "int8", "int8int8", "int4", "int4mix", "q4k":
		return m.quant
	}
	// A v5 .giw records the resolved label at bake time — prefer it over re-inferring. Empty for
	// a pre-v5 or streamed bundle, which fall back to the (corrected) tensor-kind inference.
	if m.w.BakedQuant != "" {
		return m.w.BakedQuant
	}
	return m.w.QuantLabel()
}

// CheckGiwQuantMatch returns a startup error when an explicit weight-quant request cannot take effect because the model
// is an already-baked prequant .giw whose quant differs. A .giw is serialized at a fixed precision, so --quant is inert
// for it (Load ignores it); this surfaces the mismatch instead of dropping the flag silently.
//
// `requested` is the quant the user explicitly asked for: pass "" when they did not (relied on the default). A bare
// default must not conflict: a .giw carries its own quant, and running it with process defaults is the normal
// cross-format case (the caller, which alone knows whether the flag was set, passes "" then). For a non-.giw model
// (GGUF/safetensors), where --quant is honored at load, this is a no-op.
//
// The comparison uses the Quant() label (the resident weight kinds), not the raw .giw header field. The message mirrors
// the safetensors int4mix decline (weights.go): the constraint, the requested value, the baked value, and the file.
func (m *Model) CheckGiwQuantMatch(requested string) error {
	path := m.GiwPath()
	if requested == "" || path == "" {
		return nil
	}
	if baked := flagQuant(m.Quant()); flagQuant(requested) != baked {
		return fmt.Errorf("decoder: --quant %q cannot apply to the prequantized .giw bundle %s — it is baked at %q, and a .giw carries its own quant; pass --quant %s or omit --quant", requested, path, baked, baked)
	}
	return nil
}

// flagQuant spells a quant the way --quant does. Quant() reports an unquantized model as "native", a
// word --quant does not accept; the flag says "f32". Compared raw, an explicit --quant f32 was refused
// against an unquantized .giw, and the refusal told the user to pass --quant native.
func flagQuant(q string) string {
	if q == "" || q == "native" {
		return "f32"
	}
	return q
}

// NewModel wraps an already-built *Weights (e.g. from LoadSerializedWeights)
// into a runnable *Model: it attaches a compute backend and resolves the EOS
// ids from the config. The weights are used as-is — their quantization is fixed.
func NewModel(w *darch.Weights, backend string) (*Model, error) {
	return NewModelWithOptions(w, Options{Backend: backend})
}

// NewModelWithOptions is NewModel with load options: the per-model ones a .giw load through Load
// honours — KV precision, resident context, fit, MoE expert caching, exact prefill, knobs — applied
// the same way (modelFromOptions). Options that say how to BUILD weights do not apply to weights
// already built, exactly as for a .giw: Quant is inert (CheckGiwQuantMatch reports an explicit one
// that disagrees) and LoRA has no base to merge into (refuse it before calling). StreamWeights is
// refused, not ignored: paging reads a .giw's file mapping, and in-memory weights have none.
func NewModelWithOptions(w *darch.Weights, opts Options) (*Model, error) {
	opts = opts.withAutoBackend()
	if opts.StreamWeights {
		return nil, fmt.Errorf("decoder: StreamWeights pages weights out of a .giw file mapping, and prequantized weights held in memory have none — load the .giw file with Load to stream it")
	}
	be, beErr := NewBackend(opts.Backend)
	if beErr != nil {
		logs.Warnf("%v", beErr) // webgpu/cuda requested but fell back — not fatal.
	}
	return modelFromOptions(w, be, opts).withBackendNames(opts.Backend, beErr).bindKnobs(opts.Knobs.values()).withResidency(), nil
}
