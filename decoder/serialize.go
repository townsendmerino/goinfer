package decoder

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"

	"github.com/townsendmerino/aikit/linalg"
	"hash/crc32"
	"io"
	"math"
	"unsafe"
)

// This file defines the binary format of an already-quantized *Weights bundle (a ".giw", goinfer weights), so the
// resident weights can be produced once and mapped, skipping the GGUF dequant and requant on every launch. The big
// int8/int4 weight arrays and their scale arrays are aliased over the input slice at load (zero-copy: this is the speed
// and RAM win); the norm and bias vectors and f32 matrices are copied.
//
// There is one layout. This build writes giwVersion and reads only giwVersion; a file at any other version is refused
// with a *SerializeError, and a sidecar that fails its freshness check is rebuilt from its source (internal/prequant).
// Every tensor record starts with its encoding (a weight matrix's kind, or giwEncF32 for a standalone array), and a reader
// refuses only an encoding it does not read, so a new encoding needs no new version (docs/giw-bundles.md, "From v17").
// There is no fallback to the GGUF: an embedded or explicitly named bundle that does not load is an error. The
// history of the earlier versions is in docs/code-notes/decoder.md ("The .giw format before v16").
//
// Format (little-endian throughout):
//
//	magic   [5]byte = "GINFW"
//	version uint32   (giwVersion)
//	id      str      (model identity, the source's basename; not validated on read)
//	config  str      (Config as JSON; arch is re-derived from it on load)
//	quantLabel str   (the resolved quant label (int4|int4mix|int8int8|int8|native), or "" for a streamed bundle,
//	                  whose label the reader infers from the tensors)
//	zero padding to a 16-byte boundary
//	Embed, LMHead, PosEmbed     weightMat
//	FinalNorm, FinalNormBias    f32
//	(gemma4 only: the model-level Per-Layer-Embedding inputs)
//	numLayers uint32
//	  per layer: the LayerWeights fields in the order giwWriter.layer writes them, with the per-family tails
//	crc     uint32   (CRC32-IEEE over every preceding byte)
//
// str  = uint32 len + len bytes
// f32  = uint8 encoding (giwEncF32) + uint32 len + len*4 LE-float32 bytes   (len 0 => nil on load); a weight matrix's own
//        arrays carry no encoding byte, the matrix's kind being theirs
// i8   = uint32 len + len bytes                (aliased on load)
// raw  = uint32 len + len bytes                (aliased on load)
// u16  = uint32 len + len*2 LE-uint16 bytes    (aliased on load)
//
// weightMat = uint8 kind; if non-empty: int32 rows, cols, group; uint8 w8a8; then the kind's arrays. Every aliased array
// is preceded by zero padding so its payload starts 16-aligned relative to the blob start (giwArrayPad); the v3 bundle
// frame puts the blob at file offset 64, so the payload is aligned in the mapping too.
//
//	0  empty
//	1  f32: an f32 array, not aligned (it is copied)
//	2  int8: f32 row scales, then i8 codes
//	3  int4, canonical: u16 binary16 group scales, then raw nibbles
//	5  int4, row4-only: u16 scales and raw nibbles in the arm64 split-half + 4-row-interleaved layout, with no canonical
//	   arrays. Written per tensor on a cpu-arm64 target only, for a tensor whose call site allows it (weightMat, not
//	   weightMatKind3Only) and whose shape repackRow4ForEmit accepts; everything else is kind 3. Loaded with
//	   linalg.WrapInt4Row4OnlyF16, which declines (a named *SerializeError) on a core that cannot run that layout. A
//	   storage choice, not a numerics one: dispatch is bit-identical either way.
//	6  int4, fused-group member (Metal target): a header with no arrays; the group's nibbles and then its binary16
//	   scales follow the group's last header as two unprefixed 16-aligned blocks (giwWriter.fusedGroup).
//
// Kinds 4 and 7 existed before v16 and are not read.

// GIWFormatVersion is the weights format version this build writes and reads: the "vN" a sidecar's file name carries
// (internal/prequant), so builds at different versions keep separate sidecars of one model.
const GIWFormatVersion = giwVersion

const (
	giwMagic = "GINFW"
	// giwVersion is the one format version this build writes and reads. A file at any other version is refused, and a
	// sidecar is then rebuilt from its source (docs/giw-bundles.md, "One version").
	giwVersion = 17
	// giwEncF32 is a standalone f32 array's encoding byte, in the weight-matrix kinds' namespace (kind 1 is f32). Every
	// tensor record starts with its encoding, and a reader refuses only an encoding it does not know (unknownEncoding).
	giwEncF32 = 1
	// Sanity ceilings on the count fields, generous against any real checkpoint but low enough that a corrupt or hostile blob
	// cannot drive a multi-GB make() before the body reader hits its first short read. A LayerWeights is a large struct, so an
	// unbounded layer count is the worst offender.
	maxSerializedLayers  = 4096
	maxSerializedExperts = 4096

	// maxSnapshotCacheBytes caps the KV cache a session snapshot may ask LoadSession to allocate (kvsnapshot.go). It bounds the
	// allocation rather than the blob, because a well-formed snapshot body can be small while pos is large: a never-written
	// ring and a KV-shared layer each serialise zero KV bytes, and a ring layer stores only min(count, W) rows. 16 GiB is far
	// above any legitimate session and far below the TBs the unbounded path could reach.
	maxSnapshotCacheBytes int64 = 1 << 34
)

// SerializeError is returned by LoadSerializedWeights on any magic/version/
// quant/CRC mismatch. It is distinct so callers can fall back to building from
// the source GGUF rather than treating a stale blob as fatal.
type SerializeError struct{ Reason string }

func (e *SerializeError) Error() string { return "decoder: serialized weights: " + e.Reason }

// canSerialize reports why a model's per-layer state cannot round-trip through the .giw format, or nil if it can. It
// returns nil for every registered family today. Refusing a family up front beats emitting a CRC-valid bundle that
// nil-derefs at the first forward.
func canSerialize(a *Architecture) *SerializeError {
	// Empty on purpose: every registered family is representable. A hand-maintained blocklist drifted silently (a family could
	// ride it while the writer dropped state, so bundles loaded clean and generated wrong text); the guard against drift is
	// TestSerializeCensus_noSilentFieldDrop, which asks the struct whether a round-trip lost anything. Keep the function: a
	// future family may genuinely be unrepresentable (a new per-layer state with no field here), and refusing is the correct
	// answer for it.
	return nil
}

// SerializeWeights writes the resident weight bundle (already quantized to its
// current precision) to a flat little-endian blob suitable for embedding. id is
// an opaque model-identity string (e.g. the source filename) stored for tooling.
func SerializeWeights(w *Weights, id string) ([]byte, error) {
	wr := &giwWriter{}
	if err := wr.writeBundle(w, id); err != nil {
		return nil, err
	}
	wr.u32(crc32.ChecksumIEEE(wr.buf)) // CRC over the body; appended last
	return wr.buf, nil
}

// SerializeWeightsTo streams the same bundle directly to out, never materializing
// the whole blob in memory — so prequantizing a large model peaks at ~the resident
// weight size, not 2× (resident + blob). It returns the number of bytes written
// (body + trailing CRC). The big int8/int4 arrays are written straight from the
// resident slices (one write each); only the small per-tensor scale/norm vectors
// are buffered transiently. out should be a regular file for the prequant path.
func SerializeWeightsTo(out io.Writer, w *Weights, id string) (int64, error) {
	wr := &giwWriter{sink: out}
	if err := wr.writeBundle(w, id); err != nil {
		return wr.n, err
	}
	var crc [4]byte
	binary.LittleEndian.PutUint32(crc[:], wr.crc) // running CRC over the body
	if _, err := out.Write(crc[:]); err != nil {
		return wr.n, err
	}
	return wr.n + 4, nil
}

// SerializeWeightsForTarget is SerializeWeights for a bundle promised to ONE consumer (docs/tasks/task-int4-layout-2026-09.md):
// on a cpu-arm64 target, every eligible int4 tensor (see weightMat vs weightMatKind3Only) writes kind 5 (row4-only)
// instead of kind 3; every other target, including GIWTargetNone, writes kind 3 for every int4 tensor exactly like
// SerializeWeights. internal/prequant.Transcode/EnsureCachedGIW and cmd/prequant drive this.
func SerializeWeightsForTarget(w *Weights, id string, target GIWTarget) ([]byte, error) {
	wr := &giwWriter{target: target}
	if err := wr.writeBundle(w, id); err != nil {
		return nil, err
	}
	wr.u32(crc32.ChecksumIEEE(wr.buf))
	return wr.buf, nil
}

// SerializeWeightsToForTarget is SerializeWeightsTo with SerializeWeightsForTarget's
// target-aware kind-5 opt-in — see that function's doc for the contract.
func SerializeWeightsToForTarget(out io.Writer, w *Weights, id string, target GIWTarget) (int64, error) {
	wr := &giwWriter{sink: out, target: target}
	if err := wr.writeBundle(w, id); err != nil {
		return wr.n, err
	}
	var crc [4]byte
	binary.LittleEndian.PutUint32(crc[:], wr.crc)
	if _, err := out.Write(crc[:]); err != nil {
		return wr.n, err
	}
	return wr.n + 4, nil
}

// writeBundle writes the bundle body (everything but the trailing CRC) via the
// writer's current sink (buffer or stream). Shared by SerializeWeights and
// SerializeWeightsTo so the field order can't drift between them or from the reader.
func (wr *giwWriter) writeBundle(w *Weights, id string) error {
	if err := canSerialize(w.arch); err != nil {
		return err
	}
	wr.arch = w.arch // gates the gemma4 model-level PLE + per-layer tail
	if err := wr.writeHeadGlobals(w, id); err != nil {
		return err
	}
	for i := range w.Layers {
		wr.layer(&w.Layers[i])
	}
	return wr.err
}

// writeHeadGlobals writes everything up to and including the layer count: the
// header (magic/version/id/config/label), the global tensors, and u32(len(Layers)).
// Split out so a streaming transcode can emit the head, then produce-write-free each
// layer one at a time (peak RAM ~one layer, not the whole model) before the loop in
// writeBundle. The layer count is len(w.Layers), so the streamer must allocate
// w.Layers to NumLayers up front even though it never fills them all at once.
func (wr *giwWriter) writeHeadGlobals(w *Weights, id string) error {
	cfgJSON, err := json.Marshal(w.Cfg)
	if err != nil {
		return fmt.Errorf("decoder: marshal config: %w", err)
	}
	wr.raw([]byte(giwMagic))
	wr.u32(giwVersion)
	wr.str(id)
	wr.bytesField(cfgJSON)

	// The resolved quant label, recorded so the reader need not re-infer it. It is gated on whether w.Layers is
	// populated, not on which writer is in use. The incremental GGUF transcode calls writeHeadGlobals on a freshly make()'d,
	// all-zero Layers slice before any layer streams, where quantLabel() cannot see real data and its default case returns
	// "native", a real quant mode, so calling it unconditionally would bake a false "native" label into every streamed
	// bundle. A caller that already holds a fully loaded *Weights (internal/prequant) and merely chooses the streaming API
	// for its I/O shape does get the label, so a buffered and a streamed call on the same model produce identical bytes.
	label := ""
	if w.hasPopulatedLayers() {
		label = w.quantLabel()
	}
	wr.str(label)
	wr.alignHead() // the variable-length header (the label, present or not) must not shift what follows

	wr.weightMat(&w.Embed)
	wr.weightMat(&w.LMHead)
	wr.weightMatKind3Only(&w.PosEmbed)
	wr.f32(w.FinalNorm)
	wr.f32(w.FinalNormBias)

	// Gemma 4 model-level Per-Layer-Embedding inputs (empty on the PLE-free
	// E-model/26B variants, but present as empty WeightMats so the layout is stable).
	// Written for gemma4 only; no other family's bundle carries them.
	if wr.arch != nil && wr.arch.gemma4 != nil {
		wr.weightMat(&w.PerLayerTokenEmbed)
		wr.weightMat(&w.PerLayerModelProj)
		wr.f32(w.PerLayerProjNorm)
		// FFNPerLayer (the E-models' variable per-layer FFN widths) is json:"-" on
		// Config, so it does NOT survive the config-JSON round-trip — without it ffnAt()
		// falls back to IntermediateDim and mis-sizes the MLP matmuls. Carry it here.
		ffn := wr.arch.gemma4.FFNPerLayer
		wr.u32(uint32(len(ffn)))
		for _, v := range ffn {
			wr.u32(uint32(v))
		}
	}

	wr.u32(uint32(len(w.Layers)))
	return wr.err
}

// LoadSerializedWeights reconstructs a *Weights from a SerializeWeights blob without any dequant or requant. Big
// int8/int4 arrays are aliased into data (zero-copy); float arrays are copied. data MUST stay alive for the returned
// model's lifetime (the aliased slices point into it). On any magic/version/quant/arch/CRC mismatch it returns a
// *SerializeError, which distinguishes a corrupt or stale bundle from an I/O failure. There is no fallback to the GGUF:
// callers return the error.
func LoadSerializedWeights(data []byte) (*Weights, error) {
	return loadSerializedWeights(data, false)
}

// loadSerializedWeights is LoadSerializedWeights with the whole-payload CRC optionally skipped —
// only for a caller that already holds proof THIS file passed it (see giwverify.go). Everything
// else (magic, version, header, bounds, arch, per-tensor validation) still runs.
func loadSerializedWeights(data []byte, crcAlreadyVerified bool) (*Weights, error) {
	return loadSerializedWeightsHook(data, crcAlreadyVerified, nil)
}

// loadSerializedWeightsHook is loadSerializedWeights with beforeWalk, when non-nil, called once the header is read
// and its architecture resolved, before any weight matrix is: the point at which a caller that holds the file's
// mapping can prepare the page cache for the walk over the matrices (prepareColdLoad).
func loadSerializedWeightsHook(data []byte, crcAlreadyVerified bool, beforeWalk func(*Architecture)) (*Weights, error) {
	r := &giwReader{data: data}
	if got := r.rawN(len(giwMagic)); string(got) != giwMagic {
		return nil, &SerializeError{fmt.Sprintf("bad magic %q (want %q)", got, giwMagic)}
	}
	if v := r.u32(); r.err == nil && v != giwVersion {
		return nil, &SerializeError{fmt.Sprintf("format v%d, this build reads v%d — delete the sidecar (it rebuilds) "+
			"or rebuild with `go run ./cmd/prequant …`", v, giwVersion)}
	}
	_ = r.str() // id — stored for tooling, not validated here
	cfgJSON := r.bytesField()
	bakedQuant := r.str() // the resolved quant label; "" for a streamed bundle, which is then inferred from the tensors
	r.alignHead()
	if r.err != nil {
		return nil, &SerializeError{"truncated header"}
	}

	// CRC: verify the whole payload (everything before the trailing crc word)
	// before trusting any offsets/lengths.
	if len(data) < 4 {
		return nil, &SerializeError{"too short"}
	}
	if !crcAlreadyVerified {
		body, want := data[:len(data)-4], binary.LittleEndian.Uint32(data[len(data)-4:])
		if got := crc32.ChecksumIEEE(body); got != want {
			return nil, &SerializeError{fmt.Sprintf("CRC mismatch (got %08x want %08x) — corrupt or truncated", got, want)}
		}
	}

	var cfg Config
	if err := json.Unmarshal(cfgJSON, &cfg); err != nil {
		return nil, &SerializeError{"config json: " + err.Error()}
	}
	if cfg.Sampling != nil {
		cfg.Sampling = cfg.Sampling.normalized() // the bundle's bytes are input, held to the same ranges as the checkpoint's
	}
	arch, _, err := resolveArchitecture(&cfg)
	if err != nil {
		return nil, &SerializeError{"arch: " + err.Error()}
	}
	r.arch = arch // gates the gemma4 model-level inputs and per-layer tail
	if beforeWalk != nil {
		beforeWalk(arch)
	}

	w := &Weights{Cfg: cfg, arch: arch, backing: data, bakedQuant: bakedQuant}
	w.Embed = r.weightMat()
	w.LMHead = r.weightMat()
	// A tied checkpoint (no output.weight) round-trips with an empty LMHead; every other loader sets TiedLMHead from lm_head
	// presence, so mirror that here. Without it the head reads as untied+empty and the forward emits all-zero logits (greedy
	// loops on token 0, sampling is uniform noise) with no error.
	arch.TiedLMHead = w.LMHead.Rows() == 0
	w.PosEmbed = r.weightMat()
	w.FinalNorm = r.f32()
	w.FinalNormBias = r.f32()
	if arch.gemma4 != nil { // gemma4 model-level PLE inputs
		w.PerLayerTokenEmbed = r.weightMat()
		w.PerLayerModelProj = r.weightMat()
		w.PerLayerProjNorm = r.f32()
		nf := int(r.u32())
		if nf < 0 || nf > maxSerializedLayers {
			return nil, &SerializeError{"implausible ffn-per-layer count"}
		}
		if nf > 0 {
			ffn := make([]int, nf)
			for i := range ffn {
				ffn[i] = int(r.u32())
			}
			arch.gemma4.FFNPerLayer = ffn
		}
	}
	n := int(r.u32())
	if n < 0 || n > maxSerializedLayers {
		return nil, &SerializeError{"implausible layer count"}
	}
	// Compare with the arch before allocating, not after. validateShapes catches a mismatched count, but only once every layer
	// struct exists, and maxSerializedLayers is ~40x a real model, so a hostile count amplifies that far on an exported entry
	// point before any check runs. The arch is already resolved by this point, so the comparison is free.
	if arch != nil && arch.NumLayers > 0 && n != arch.NumLayers {
		return nil, &SerializeError{fmt.Sprintf(
			"layer count: blob has %d, arch expects %d", n, arch.NumLayers)}
	}
	w.Layers = make([]LayerWeights, n)
	for i := range w.Layers {
		r.layer(&w.Layers[i])
	}
	if r.err != nil {
		return nil, &SerializeError{"truncated body: " + r.err.Error()}
	}
	if err := validateShapes(w, arch); err != nil {
		return nil, err
	}
	return w, nil
}

// validateShapes cross-checks the deserialized tensors against the architecture's expected dims. The .giw reader
// validates only internal consistency (array length vs the blob's own rows/cols), so a bundle whose Router declares
// rows = NumExperts+K, or an Embed/LMHead with the wrong vocab, passes every reader check and then writes past a
// config-sized scratch slice at decode (moeMLP's `make([]float32, NumExperts)`, the qDim/kvDim/vocab decodeScratch
// buffers): heap corruption from caller-supplied bytes (LoadSerializedWeights is exported). The GGUF and safetensors
// loaders do this cross-check too.
//
// Universal invariants (vocab and expert count are uniform in every serializable family) are always checked. The
// attention/FFN projection dims are checked per layer through the per-layer accessors, so gemma-4's per-layer geometry
// (FFNPerLayer, two-geom head dims) is covered without a model-level dim that would false-reject it.
func validateShapes(w *Weights, arch *Architecture) *SerializeError {
	eq := func(name string, got, want int) *SerializeError {
		if got != want {
			return &SerializeError{fmt.Sprintf("%s: %d rows, arch expects %d", name, got, want)}
		}
		return nil
	}
	// vec checks a per-layer f32 vector (bias / norm weight) whose length the blob controls but the forward indexes at an
	// arch-derived width: addBias iterates over the projection output and rmsNorm indexes weight[0:dim], so a short vector
	// slice-panics in the decode goroutine and a long one is silently mis-consumed. 0 = absent (a family that omits it),
	// allowed.
	vec := func(name string, got, want int) *SerializeError {
		if got != 0 && got != want {
			return &SerializeError{fmt.Sprintf("%s: len %d, arch expects %d", name, got, want)}
		}
		return nil
	}
	// req is vec for a vector the family's forward dereferences unconditionally, where "absent" is not a family that omits it
	// but a bundle that is missing it. vec's `got == 0 => allowed` is what would let a pre-v6 gpt-oss sidecar through: it is
	// within giwMinReadV, "fresh" by mtime, reads AttnSinks as nil, passes validateShapes, and panics at forward_gptoss.go's
	// `lw.AttnSinks[qh]` on the first request. Same shape as the LFM2 conv presence check below.
	req := func(name string, got, want int) *SerializeError {
		if got == 0 {
			return &SerializeError{fmt.Sprintf("%s: absent, arch requires len %d — the bundle "+
				"predates this field (rewrite the .giw sidecar; mtime freshness cannot see a "+
				"missing tensor)", name, want)}
		}
		return vec(name, got, want)
	}
	// cols checks a table's row width. The embedding lookup widens one row into a HiddenDim-sized scratch (WeightMat.Row), so a
	// table the blob declares wider than that is refused here, by name, instead of reaching the kernel; the LM head is
	// multiplied against a HiddenDim activation, and a gpt-2 position row is added to one.
	cols := func(name string, got, want int) *SerializeError {
		if got != want {
			return &SerializeError{fmt.Sprintf("%s: %d cols, arch expects %d", name, got, want)}
		}
		return nil
	}
	if w.Embed.Rows() > 0 {
		if e := eq("Embed", w.Embed.Rows(), arch.VocabSize); e != nil {
			return e
		}
		if e := cols("Embed", w.Embed.Cols(), arch.HiddenDim); e != nil {
			return e
		}
	}
	if w.LMHead.Rows() > 0 { // 0 = tied (validated via Embed above)
		if e := eq("LMHead", w.LMHead.Rows(), arch.VocabSize); e != nil {
			return e
		}
		if e := cols("LMHead", w.LMHead.Cols(), arch.HiddenDim); e != nil {
			return e
		}
	}
	if w.PosEmbed.Rows() > 0 {
		if e := cols("PosEmbed", w.PosEmbed.Cols(), arch.HiddenDim); e != nil {
			return e
		}
	}
	// The blob controls len(w.Layers), but the forward indexes arch.NumLayers: a short blob is an out-of-bounds layer read the
	// per-layer checks below never reach.
	if len(w.Layers) != arch.NumLayers {
		return &SerializeError{fmt.Sprintf("layer count: blob has %d, arch expects %d", len(w.Layers), arch.NumLayers)}
	}
	// The per-layer accessors (headDimAt/kvHeadsAt/ffnAt) collapse to the uniform Architecture fields for every non-gemma-4
	// family, so one per-layer check set covers all families, gemma-4 included. Each check is guarded by Rows()>0, so a family
	// that legitimately omits a projection (a routed layer's empty dense FFN, gemma-4's MLP living in the MoE sub-block) is
	// not false-rejected.
	for i := range w.Layers {
		lw := &w.Layers[i]
		hd := arch.headDimAt(i)
		// headsAt(i), not NumHeads: some families (Laguna) vary the query head count per layer, so a uniform NumHeads would
		// reject a correctly written bundle. headDimAt/kvHeadsAt/ffnAt are per-layer here too.
		qDim, kvDim, ffn := arch.headsAt(i)*hd, arch.kvHeadsAt(i)*hd, arch.ffnAt(i)
		for _, c := range []struct {
			name string
			got  int
			want int
		}{
			{"QProj", lw.QProj.Rows(), qDim},
			{"KProj", lw.KProj.Rows(), kvDim},
			{"VProj", lw.VProj.Rows(), kvDim},
			{"OProj", lw.OProj.Rows(), arch.HiddenDim},
			{"GateProj", lw.GateProj.Rows(), ffn},
			{"UpProj", lw.UpProj.Rows(), ffn},
			{"DownProj", lw.DownProj.Rows(), arch.HiddenDim},
		} {
			if c.got > 0 {
				if e := eq(fmt.Sprintf("layer %d %s", i, c.name), c.got, c.want); e != nil {
					return e
				}
			}
		}
		// GProj (Laguna's attention output gate) is legal at EITHER width — per-head (one scalar per
		// query head) or per-element (the full qDim) — and which one ships is decided by the tensor,
		// not the config (XS.2 declares per-element and ships per-head). So both are accepted and
		// anything else is rejected, rather than leaving the field unchecked.
		if g := lw.GProj.Rows(); g > 0 && g != arch.headsAt(i) && g != qDim {
			return &SerializeError{fmt.Sprintf("layer %d GProj: %d rows, arch expects %d (per-head) or %d (per-element)",
				i, g, arch.headsAt(i), qDim)}
		}
		// Per-layer f32 vectors: biases feed addBias over the projection output, norm weights feed rmsNorm at their consumed width
		// (QK-norm is per-head-dim hd normally, but the whole projected width when the family sets QKNormWhole: olmo3's
		// whole-vector QK-norm, which reuses rmsNorm at rows=1/dim=qDim; the block norms are hidden). The blob controls their
		// length; the forward indexes an arch width with no check.
		qNormWant, kNormWant := hd, hd
		if arch.QKNormWhole {
			qNormWant, kNormWant = qDim, kvDim
		}
		for _, c := range []struct {
			name string
			got  int
			want int
		}{
			{"QBias", len(lw.QBias), qDim}, {"KBias", len(lw.KBias), kvDim},
			{"VBias", len(lw.VBias), kvDim}, {"OBias", len(lw.OBias), arch.HiddenDim},
			{"QNorm", len(lw.QNorm), qNormWant}, {"KNorm", len(lw.KNorm), kNormWant},
			{"PreAttnNorm", len(lw.PreAttnNorm), arch.HiddenDim}, {"PostAttnNorm", len(lw.PostAttnNorm), arch.HiddenDim},
			{"PreMLPNorm", len(lw.PreMLPNorm), arch.HiddenDim}, {"PostMLPNorm", len(lw.PostMLPNorm), arch.HiddenDim},
			{"UpBias", len(lw.UpBias), ffn}, {"DownBias", len(lw.DownBias), arch.HiddenDim},
		} {
			if e := vec(fmt.Sprintf("layer %d %s", i, c.name), c.got, c.want); e != nil {
				return e
			}
		}
		// LFM2's short-conv mixer. Its three tensors are flat f32 slices the forward indexes at arch-derived widths, so a short one
		// slice-panics in the decode goroutine, as for the bias vectors. The presence check is the load-bearing half: a conv layer
		// whose mixer is absent is what a pre-v8 .giw hands back, and the panic it produces at the first forward is the defect
		// this validation converts into a refusal at load.
		if arch.lfm2 != nil {
			cd, k := arch.lfm2.ConvDim, arch.lfm2.ConvLCache
			isConv := lw.QProj.Rows() == 0 // a conv layer loads no attention projections
			switch {
			case isConv && lw.shortConv == nil:
				return &SerializeError{fmt.Sprintf("layer %d: lfm2 conv layer has no short-conv "+
					"mixer — the bundle was written before the v8 tail and would nil-deref at the "+
					"first forward", i)}
			case lw.shortConv == nil:
			default:
				c := lw.shortConv
				for _, ck := range []struct {
					name string
					got  int
					want int
				}{
					{"shortConv.inProj", len(c.inProj), 3 * cd * arch.HiddenDim},
					{"shortConv.convW", len(c.convW), cd * k},
					{"shortConv.outProj", len(c.outProj), arch.HiddenDim * cd},
				} {
					if e := eq(fmt.Sprintf("layer %d %s", i, ck.name), ck.got, ck.want); e != nil {
						return e
					}
				}
			}
		}
		// gemma-4 per-layer-embedding (PLE) branch: PLEGate/PLEProj are matmul'd and PostPLENorm rmsNorm'd at fixed widths; a
		// wrong-rows blob OOB-panics or silently truncates the PLE activation.
		if arch.gemma4 != nil {
			if pleDim := arch.gemma4.HiddenSizePerLayerInput; pleDim > 0 {
				if lw.PLEGate.Rows() > 0 {
					if e := eq(fmt.Sprintf("layer %d PLEGate", i), lw.PLEGate.Rows(), pleDim); e != nil {
						return e
					}
				}
				if lw.PLEProj.Rows() > 0 {
					if e := eq(fmt.Sprintf("layer %d PLEProj", i), lw.PLEProj.Rows(), arch.HiddenDim); e != nil {
						return e
					}
				}
			}
			if e := vec(fmt.Sprintf("layer %d PostPLENorm", i), len(lw.PostPLENorm), arch.HiddenDim); e != nil {
				return e
			}
		}
		// gemma-4 dense||MoE sub-block (l.gemma4moe): the standard Router/Experts are empty for gemma-4, so the arch.MoE block
		// below never validates it. The router feeds make([]float32, nE) and each expert index runs to nE: a short router
		// mis-routes silently, a long one or ne != NumExperts panics in the decode goroutine. Cross-check against the arch.
		if mo := lw.gemma4moe; mo != nil && arch.MoE != nil {
			nE, moeInter := arch.MoE.NumExperts, arch.MoE.IntermediateDim
			if e := eq(fmt.Sprintf("layer %d gemma4moe router", i), mo.routerProj.Rows(), nE); e != nil {
				return e
			}
			if len(mo.expertsGateUp) != nE || len(mo.expertsDown) != nE {
				return &SerializeError{fmt.Sprintf("layer %d gemma4moe expert count: gate|up %d / down %d, arch expects %d", i, len(mo.expertsGateUp), len(mo.expertsDown), nE)}
			}
			if e := vec(fmt.Sprintf("layer %d gemma4moe perExpertScale", i), len(mo.perExpertScale), nE); e != nil {
				return e
			}
			// routerScale scales the [hidden] router input; the three branch norms rmsNorm at hidden. A short slice OOB-panics in the
			// decode goroutine.
			for _, c := range []struct {
				name string
				got  int
			}{
				{"gemma4moe routerScale", len(mo.routerScale)},
				{"gemma4moe postFFNNorm1", len(mo.postFFNNorm1)},
				{"gemma4moe preFFNNorm2", len(mo.preFFNNorm2)},
				{"gemma4moe postFFNNorm2", len(mo.postFFNNorm2)},
			} {
				if e := vec(fmt.Sprintf("layer %d %s", i, c.name), c.got, arch.HiddenDim); e != nil {
					return e
				}
			}
			for xe := range mo.expertsGateUp {
				if e := eq(fmt.Sprintf("layer %d gemma4moe expert %d gate|up", i, xe), mo.expertsGateUp[xe].Rows(), 2*moeInter); e != nil {
					return e
				}
				if e := eq(fmt.Sprintf("layer %d gemma4moe expert %d down", i, xe), mo.expertsDown[xe].Rows(), arch.HiddenDim); e != nil {
					return e
				}
			}
		}
		if arch.MoE != nil {
			// RouterBias is addBias'd over the [NumExperts] router logits; the shared expert's Gate/Up write SharedIntermediateDim
			// scratch, Down writes hidden, and SharedGate is the scalar sigmoid gate (1 row). A short blob panics; validate them.
			if e := vec(fmt.Sprintf("layer %d RouterBias", i), len(lw.RouterBias), arch.MoE.NumExperts); e != nil {
				return e
			}
			for _, c := range []struct {
				name string
				got  int
				want int
			}{
				{"SharedExpert Gate", lw.SharedExpert.Gate.Rows(), arch.MoE.SharedIntermediateDim},
				{"SharedExpert Up", lw.SharedExpert.Up.Rows(), arch.MoE.SharedIntermediateDim},
				{"SharedExpert Down", lw.SharedExpert.Down.Rows(), arch.HiddenDim},
				{"SharedGate", lw.SharedGate.Rows(), 1},
			} {
				if c.got > 0 {
					if e := eq(fmt.Sprintf("layer %d %s", i, c.name), c.got, c.want); e != nil {
						return e
					}
				}
			}
			// Router feeds moeMLP's make([]float32, NumExperts) — the exploit the audit names.
			if lw.Router.Rows() > 0 {
				if e := eq(fmt.Sprintf("layer %d Router", i), lw.Router.Rows(), arch.MoE.NumExperts); e != nil {
					return e
				}
			}
			// Per-expert Gate/Up (moe intermediate width) and Down (hidden) — same config-sized-scratch
			// write class as the dense projections. Empty for families that stack experts elsewhere.
			for xe := range lw.Experts {
				ex := &lw.Experts[xe]
				for _, c := range []struct {
					name string
					got  int
					want int
				}{
					{"Gate", ex.Gate.Rows(), arch.MoE.IntermediateDim},
					{"Up", ex.Up.Rows(), arch.MoE.IntermediateDim},
					{"Down", ex.Down.Rows(), arch.HiddenDim},
				} {
					if c.got > 0 {
						if e := eq(fmt.Sprintf("layer %d expert %d %s", i, xe, c.name), c.got, c.want); e != nil {
							return e
						}
					}
				}
			}
		}

		// The v6 completeness tail. forward_gptoss.go indexes AttnSinks[qh] for every head and addBias iterates every expert bias
		// with no nil check, so for this family these are required, not optional; a gpt-oss sidecar written before v6 is
		// sink-free, within giwMinReadV, and judged fresh by mtime.
		if arch.gptoss != nil {
			if e := req(fmt.Sprintf("layer %d AttnSinks", i), len(lw.AttnSinks), arch.NumHeads); e != nil {
				return e
			}
			for xe := range lw.Experts {
				ex := &lw.Experts[xe]
				if ex.Gate.Rows() == 0 {
					continue // an expert the bundle stacks elsewhere; its biases live there too
				}
				for _, c := range []struct {
					name string
					got  int
					want int
				}{
					{"GateBias", len(ex.GateBias), arch.MoE.IntermediateDim},
					{"UpBias", len(ex.UpBias), arch.MoE.IntermediateDim},
					{"DownBias", len(ex.DownBias), arch.HiddenDim},
				} {
					if e := req(fmt.Sprintf("layer %d expert %d %s", i, xe, c.name), c.got, c.want); e != nil {
						return e
					}
				}
			}
		}
	}

	// The model-level PLE tail: gemma4's forward reads all three unconditionally when the arch declares PLE, and a bundle from
	// before v4 has none of them.
	//
	// PerLayerModelProj is [NumLayers*HiddenSizePerLayerInput, HiddenDim]: gguf.go builds it as
	// mat("per_layer_model_proj.weight", pleTotal, hidden) and forward_gemma4.go's matmul(&PerLayerModelProj, ...) agrees that
	// Rows() is pleTotal, not HiddenDim.
	if arch.gemma4 != nil && w.PerLayerTokenEmbed.Rows() > 0 {
		pleTotal := arch.NumLayers * arch.gemma4.HiddenSizePerLayerInput
		if e := eq("PerLayerModelProj", w.PerLayerModelProj.Rows(), pleTotal); e != nil {
			return e
		}
		if got := w.PerLayerModelProj.Cols(); got != arch.HiddenDim {
			return &SerializeError{fmt.Sprintf("PerLayerModelProj: %d cols, arch expects %d", got, arch.HiddenDim)}
		}
		// PerLayerTokenEmbed rows are pleTotal wide (gguf.go builds it as embMat(..., VocabSize, pleTotal)); the lookup widens one
		// into a pleTotal-sized scratch.
		if e := cols("PerLayerTokenEmbed", w.PerLayerTokenEmbed.Cols(), pleTotal); e != nil {
			return e
		}
		// PerLayerProjNorm is sized HiddenSizePerLayerInput (the RMSNorm runs over one pleDim-wide row of ctxAware), not
		// HiddenDim; gguf.go's vnorm and forward_gemma4.go's normalize agree.
		if e := req("PerLayerProjNorm", len(w.PerLayerProjNorm), arch.gemma4.HiddenSizePerLayerInput); e != nil {
			return e
		}
	}
	return nil
}

// Weights exposes the loaded weight bundle, e.g. so a build-time tool can SerializeWeights(m.Weights(), ...). It returns
// the live bundle (a defensive copy would duplicate gigabytes and the device buffers), so the forward pass depends on it
// being treated as IMMUTABLE: the derived *Architecture, RoPE tables and resident buffers are built from it at load and
// are not rebuilt, so any mutation silently desyncs them. Read-only.
func (m *Model) Weights() *Weights { return m.w }

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
	if m.w.bakedQuant != "" {
		return m.w.bakedQuant
	}
	return m.w.quantLabel()
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

// hasPopulatedLayers reports whether w's body matmul weights actually hold data, as opposed to a freshly make()'d Layers
// slice whose elements are still zero-valued (the streaming GGUF transcode's state at header-write time, before any layer
// has streamed in). Check it before trusting quantLabel(): its "nothing matched" case returns "native", a real quant mode,
// not an empty string, so an unpopulated w would silently produce a false "native" label.
func (w *Weights) hasPopulatedLayers() bool {
	for _, m := range w.bodyMatmulWeights() {
		if m.Rows() > 0 {
			return true
		}
	}
	return false
}

// quantLabel names the precision of the resident matmul weights for display and the KV-snapshot fingerprint, accounting
// for mixed bundles. It scans the body matmuls (the per-layer attention/FFN projections, experts and routers: exactly
// what `-quant int4|int4mix|int8int8` selects and what the batched-prefill gate inspects) and returns "int4mix" only
// when int4 coexists with an int8 body weight, which is what `-quant int4mix` produces. Pure bundles collapse to int4 /
// int8int8 / int8 / native.
//
// A float32 body weight does not make a mix. No quant mode chooses one: a MoE's routers stay float32 at every quant, so
// counting them would label every int4 MoE "int4mix" and refuse an explicit `-quant int4` against its own sidecar.
//
// The token embedding and LM head are excluded. int4 mode pins them to int8 by default (logit-critical; the EmbedInt4
// knob relaxes them), so their precision is orthogonal to the int4-vs-int4mix distinction: a plain `-quant int4` bundle
// keeps an int8 head. Including them would mislabel such a bundle "int4mix" even though every projection is int4, and the
// label would contradict the path the prefill gate takes. The .giw header's quant field derives from the first weight
// (the int8 embed), so it is not a substitute.
func (w *Weights) quantLabel() string {
	var hasInt4, hasInt8I8, hasInt8 bool
	// bodyMatmulWeights excludes the int8-pinned logit tables — the single definition of that
	// exclusion (weights.go), so the label and the .giw resolved-quant field can never disagree.
	for _, m := range w.bodyMatmulWeights() {
		if m.Rows() == 0 {
			continue
		}
		switch m.Kind() {
		case "int4":
			hasInt4 = true
		case "int8":
			if _, _, w8a8, _ := m.Int8(); w8a8 {
				hasInt8I8 = true
			} else {
				hasInt8 = true
			}
		}
	}
	switch {
	case hasInt4 && (hasInt8I8 || hasInt8):
		return "int4mix"
	case hasInt4:
		return "int4"
	case hasInt8I8:
		return "int8int8"
	case hasInt8:
		return "int8"
	default:
		return "native"
	}
}

// NewModel wraps an already-built *Weights (e.g. from LoadSerializedWeights)
// into a runnable *Model: it attaches a compute backend and resolves the EOS
// ids from the config. The weights are used as-is — their quantization is fixed.
func NewModel(w *Weights, backend string) (*Model, error) {
	return NewModelWithOptions(w, Options{Backend: backend})
}

// NewModelWithOptions is NewModel with load options: the per-model ones a .giw load through Load
// honours — KV precision, resident context, fit, MoE expert caching, exact prefill, knobs — applied
// the same way (modelFromOptions). Options that say how to BUILD weights do not apply to weights
// already built, exactly as for a .giw: Quant is inert (CheckGiwQuantMatch reports an explicit one
// that disagrees) and LoRA has no base to merge into (refuse it before calling). StreamWeights is
// refused, not ignored: paging reads a .giw's file mapping, and in-memory weights have none.
func NewModelWithOptions(w *Weights, opts Options) (*Model, error) {
	opts = opts.withAutoBackend()
	if opts.StreamWeights {
		return nil, fmt.Errorf("decoder: StreamWeights pages weights out of a .giw file mapping, and prequantized weights held in memory have none — load the .giw file with Load to stream it")
	}
	be, beErr := NewBackend(opts.Backend)
	if beErr != nil {
		fmt.Fprintln(os.Stderr, beErr) // webgpu/cuda requested but fell back — not fatal.
	}
	return modelFromOptions(w, be, opts).withBackendNames(opts.Backend, beErr).bindKnobs(opts.Knobs.values()).withResidency(), nil
}

// --- writer ---

// giwWriter serializes the .giw / .giw-kv formats in one of two modes: buffer mode
// (sink == nil) accumulates into buf — the caller reads buf and appends its own CRC
// (SerializeWeights, kvsnapshot); stream mode (sink != nil) writes straight to sink
// while maintaining a running CRC and byte count, so a large bundle never lives in
// memory (SerializeWeightsTo). Both modes route every byte through raw, so the two
// produce identical bytes.
type giwWriter struct {
	buf  []byte        // buffer mode
	sink io.Writer     // stream mode (nil ⇒ buffer mode)
	crc  uint32        // running CRC32-IEEE over bytes written (stream mode)
	n    int64         // bytes written (stream mode)
	err  error         // first sink error (stream mode)
	arch *Architecture // set in writeBundle; gates the gemma4 model-level inputs and per-layer tail

	// target opts weightMat into emitting kind 5 (row4-only, no canonical arrays) for every eligible int4 tensor on a
	// cpu-arm64 target, instead of kind 3: the one-representation-per-target policy, .giw's counterpart to
	// wantsCanonicalInt4's in-RAM decision (docs/tasks/task-int4-layout-2026-09.md). GIWTargetNone (the zero value) keeps
	// kind 3 for every int4 tensor.
	target GIWTarget

	// layerBudget is the memory streamLayers may hold in built, unwritten layers (stream mode). Zero builds one layer at
	// a time.
	layerBudget int64
}

func (w *giwWriter) raw(b []byte) {
	if w.sink == nil {
		w.buf = append(w.buf, b...)
		return
	}
	if w.err == nil {
		if _, err := w.sink.Write(b); err != nil {
			w.err = err
		}
	}
	w.crc = crc32.Update(w.crc, crc32.IEEETable, b)
	w.n += int64(len(b))
}

// pos is how many bytes have been written since the blob start, in either mode — the quantity the
// alignment padding is defined over.
func (w *giwWriter) pos() int64 {
	if w.sink == nil {
		return int64(len(w.buf))
	}
	return w.n
}

// giwArrayPad is the number of zero bytes to insert at blob offset off so that an array written as
// `u32 count | payload` has its PAYLOAD (off+pad+4) on a 16-byte boundary. A pure function of the
// offset, so writer and reader compute the same answer without recording it.
func giwArrayPad(off int64) int { return int((-(off + 4)) & 15) }

// alignHead ends the variable-length header on a 16-byte boundary. The header carries a length-prefixed quant label
// that the resident writer emits and the streaming writer deliberately does not, so the two paths' offsets differ by the
// label's length. Every later array's padding is a function of its absolute offset, so without this the two would lay out
// the same weights with different padding and stop being byte-identical (internal/prequant
// TestStreamTranscodeMatchesResident). With it, everything after the header is at the same offsets in both.
func (w *giwWriter) alignHead() { w.align16() }

// align16 pads to a 16-byte boundary: before a block that carries no count (a fused group's nibbles and scales).
func (w *giwWriter) align16() {
	if pad := int((-w.pos()) & 15); pad > 0 {
		var z [16]byte
		w.raw(z[:pad])
	}
}

// alignArray pads before the next array's count.
func (w *giwWriter) alignArray() {
	if pad := giwArrayPad(w.pos()); pad > 0 {
		var z [16]byte
		w.raw(z[:pad])
	}
}

func (w *giwWriter) u32(v uint32) {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], v)
	w.raw(b[:])
}
func (w *giwWriter) str(s string) { w.u32(uint32(len(s))); w.raw([]byte(s)) }

func (w *giwWriter) bytesField(b []byte) { w.u32(uint32(len(b))); w.raw(b) }

// u16s writes s as little-endian uint16s, with no count.
func (w *giwWriter) u16s(s []uint16) {
	if len(s) == 0 {
		return
	}
	if nativeLittleEndian {
		w.raw(unsafe.Slice((*byte)(unsafe.Pointer(&s[0])), 2*len(s)))
		return
	}
	b := make([]byte, 2*len(s))
	for i, v := range s {
		binary.LittleEndian.PutUint16(b[2*i:], v)
	}
	w.raw(b)
}

// scales16 writes binary16 int4 scales, the WeightMat's own storage, as a u32 count + little-endian uint16 payload.
func (w *giwWriter) scales16(s []uint16) {
	w.u32(uint32(len(s)))
	w.u16s(s)
}

// f32 writes a standalone f32 array: its encoding byte (giwEncF32), then the array (f32Body).
func (w *giwWriter) f32(s []float32) {
	w.raw([]byte{giwEncF32})
	w.f32Body(s)
}

// f32Body batches into one raw write (a per-tensor temp), not 4 bytes at a time — the
// stream path would otherwise do millions of tiny writes. A weight matrix's own arrays use it directly: the matrix's
// kind byte is their encoding.
func (w *giwWriter) f32Body(s []float32) {
	w.u32(uint32(len(s)))
	if len(s) == 0 {
		return
	}
	b := make([]byte, 4*len(s))
	for i, v := range s {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(v))
	}
	w.raw(b)
}

func (w *giwWriter) i8(s []int8) {
	w.u32(uint32(len(s)))
	if len(s) > 0 {
		w.raw(unsafe.Slice((*byte)(unsafe.Pointer(&s[0])), len(s)))
	}
}

// weightMat writes m, eligible for kind 5 (row4-only) under a cpu-arm64 target —
// see weightMatKind3Only for the tensors that must never take kind 5.
func (w *giwWriter) weightMat(m *linalg.WeightMat) { w.weightMatKind(m, true) }

// weightMatKind3Only is weightMat for a tensor that must never take kind 5 regardless of target: it always writes kind 3
// for an int4 tensor. Two reasons put a call site here:
//
//   - MoE-paged experts (l.Experts[*], gemma4's mo.expertsGateUp/expertsDown): moepaging.go reads these off the mmap with
//     no load-time repack step, so the file must carry whatever layout the pager needs, chosen once at write time, not
//     per reader. (layerpaging.go's dense per-layer pager prefers WeightMat.MappedSpanRow4 over MappedSpan, so the dense
//     QProj..DownProj stay kind-5-eligible through plain weightMat.)
//   - Not yet scoped: KDA/DeltaNet/qattn mixer projections and gemma4's fused-MoE router (mo.routerProj). These are absent
//     from Weights.matmulWeights(), which decoder.Load's post-load kind-5-vs-backend check walks (see
//     repackedOnlyInt4Count), so a kind-5 instance would slip past that check and rely solely on the soft downstream
//     residency decline. They stay kind-3-only until they get their own entry in that census.
func (w *giwWriter) weightMatKind3Only(m *linalg.WeightMat) { w.weightMatKind(m, false) }

func (w *giwWriter) weightMatKind(m *linalg.WeightMat, eligible bool) {
	if m.Kind() == "q4k" {
		// quantQ4K has no .giw kind yet (docs/tasks/task-int4-weight-quality-2026-09.md): refuse
		// rather than write a tensor with no payload. Every serialize/stream entry returns w.err.
		if w.err == nil {
			w.err = fmt.Errorf("decoder: --quant q4k has no .giw form yet; load the .gguf directly")
		}
		return
	}
	if m.Rows() == 0 {
		w.raw([]byte{0}) // empty
		return
	}
	q4, q4s, group, isQ4 := m.Int4F16()
	q8, scales, w8a8, isQ8 := m.Int8()
	f32, _ := m.F32()
	var kind byte
	switch {
	case isQ4:
		kind = 3
		// Kind 5 is a function of the writer's target and this tensor's shape, never of whatever repack state already
		// sits in RAM (repackW4A8Row4IfEligible repacks every heap-built int4 tensor on arm64, whether or not this call
		// is a prequant path). So the row4 arrays are recomputed here from the canonical ones.
		if eligible && w.target == GIWTargetCPUArm64 {
			if r4, r4s, ok := repackRow4ForEmit(q4, q4s, m.Rows(), m.Cols(), group); ok {
				kind, q4, q4s = 5, r4, r4s
			}
		}
	case isQ8:
		kind = 2
	default:
		kind = 1 // f32
	}
	w.raw([]byte{kind})
	w.u32(uint32(m.Rows()))
	w.u32(uint32(m.Cols()))
	w.u32(uint32(group)) // 0 unless int4
	if w8a8 {
		w.raw([]byte{1})
	} else {
		w.raw([]byte{0})
	}
	// Kind 1 (an f32 matrix) is deliberately NOT aligned: the reader copies it (an aliased f32
	// matrix would sit in a read-only mapping, and nothing here proves no caller mutates one),
	// so there is nothing to gain. Every other array below is aliased by the reader and is padded
	// so the alias is legal (see giwArrayPad).
	switch kind {
	case 1:
		w.f32Body(f32)
	case 2:
		w.alignArray()
		w.f32Body(scales)
		w.alignArray()
		w.i8(q8)
	case 3, 5: // binary16 group scales, then the nibbles: canonical for kind 3, the row4 layout for kind 5
		w.alignArray()
		w.scales16(q4s)
		w.alignArray()
		w.bytesField(q4)
	}
}

// fusedEligible reports whether ms can be written as one fused group (kind 6): a Metal-target writer, at least
// two members, every member a canonical group-32 int4 with the same K (K%32==0, so every member's nibble bytes are a
// multiple of 16 and the members stay 16-aligned back to back). Anything else (an absent V on a K=V layer, an int8
// tensor, a mixed K) is written the ordinary way, one record at a time.
func (w *giwWriter) fusedEligible(ms []*linalg.WeightMat) bool {
	if w.target != GIWTargetMetal || len(ms) < 2 {
		return false
	}
	k := ms[0].Cols()
	for _, m := range ms {
		if m.Rows() == 0 || m.Cols() != k || k%32 != 0 {
			return false
		}
		if _, _, group, ok := m.Int4F16(); !ok || group != 32 {
			return false
		}
	}
	return true
}

// fusedGroup writes ms, in the order a resident backend fuses them (q,k,v / gate,up), as ONE group whose nibbles are
// contiguous, and whose binary16 scales are too: a Metal fused GEMV wants its rows in one buffer, and an mmap'd file can
// only be aliased into one if the rows are adjacent there. Layout:
//
//	per member:  kind 6 | rows | cols | group | w8a8=0                                      (no arrays)
//	then:        pad to 16 | nibbles of member 0 | nibbles of member 1 | ...                (no length prefixes)
//	then:        pad to 16 | scales of member 0 | scales of member 1 | ...                  (little-endian uint16)
//
// Both blocks' lengths are a pure function of each member's shape (rows*cols/2 nibble bytes, rows*cols/group scales),
// so they need no prefixes: a prefix between two members would be exactly the gap this exists to remove. A group that
// is not eligible is written as plain consecutive records, byte-for-byte what weightMat writes.
func (w *giwWriter) fusedGroup(ms ...*linalg.WeightMat) {
	if !w.fusedEligible(ms) {
		for _, m := range ms {
			w.weightMat(m)
		}
		return
	}
	for _, m := range ms {
		_, _, group, _ := m.Int4F16()
		w.raw([]byte{6})
		w.u32(uint32(m.Rows()))
		w.u32(uint32(m.Cols()))
		w.u32(uint32(group))
		w.raw([]byte{0})
	}
	w.align16()
	for _, m := range ms {
		q4, _, _, _ := m.Int4F16()
		w.raw(q4)
	}
	w.align16()
	for _, m := range ms {
		w.u16s(m.Int4ScalesF16())
	}
}

func (w *giwWriter) layer(l *LayerWeights) {
	w.fusedGroup(&l.QProj, &l.KProj, &l.VProj)
	w.weightMat(&l.OProj)
	w.f32(l.QBias)
	w.f32(l.KBias)
	w.f32(l.VBias)
	w.f32(l.OBias)
	w.f32(l.QNorm)
	w.f32(l.KNorm)
	w.f32(l.PreAttnNorm)
	w.f32(l.PreAttnNormBias)
	w.f32(l.PostAttnNorm)
	w.fusedGroup(&l.GateProj, &l.UpProj)
	w.weightMat(&l.DownProj)
	w.f32(l.UpBias)
	w.f32(l.DownBias)
	w.f32(l.PreMLPNorm)
	w.f32(l.PreMLPNormBias)
	w.f32(l.PostMLPNorm)
	w.weightMat(&l.Router)
	w.f32(l.RouterBias) // DeepSeek/GLM e_score_correction_bias
	w.u32(uint32(len(l.Experts)))
	for e := range l.Experts {
		w.weightMatKind3Only(&l.Experts[e].Gate)
		w.weightMatKind3Only(&l.Experts[e].Up)
		w.weightMatKind3Only(&l.Experts[e].Down)
	}
	w.weightMat(&l.SharedExpert.Gate)
	w.weightMat(&l.SharedExpert.Up)
	w.weightMat(&l.SharedExpert.Down)
	w.weightMat(&l.SharedGate)
	w.hybridLayer(l)
	if w.arch != nil && w.arch.gemma4 != nil {
		w.gemma4Layer(l)
	}
	w.v6Layer(l) // the completeness tail
	w.v8Layer(l) // the LFM2 short-conv tail
	w.v9Layer(l) // the KDA and MLA-gate tail
}

// v6Layer writes the state that made five families unrepresentable, in one unconditional tail. Unconditional rather than
// arch-gated like the gemma4 tail: every field here is empty on the families that do not use it, so the cost is a handful
// of zero lengths per layer, and an arch-gated tail is how a family's state goes missing (the gate is another place to
// remember). TestSerializeCensus_noSilentFieldDrop checks that claim against the struct itself.
func (w *giwWriter) v6Layer(l *LayerWeights) {
	w.weightMat(&l.GProj) // Laguna's attention output gate (per-head or per-element)
	w.f32(l.AttnSinks)    // gpt-oss per-head attention sinks
	// gpt-oss per-expert biases. The expert loop above writes only the three matrices; these are
	// nil for every other family, so this is three zero lengths per expert elsewhere.
	for e := range l.Experts {
		w.f32(l.Experts[e].GateBias)
		w.f32(l.Experts[e].UpBias)
		w.f32(l.Experts[e].DownBias)
	}
	w.f32(l.SharedExpert.GateBias)
	w.f32(l.SharedExpert.UpBias)
	w.f32(l.SharedExpert.DownBias)
	// MLA (DeepSeek / Kimi): presence byte then the eight tensors.
	if l.mla == nil {
		w.raw([]byte{0})
	} else {
		w.raw([]byte{1})
		m := l.mla
		w.f32(m.qAProj)
		w.f32(m.qALayernorm)
		w.f32(m.qBProj)
		w.f32(m.qProj)
		w.f32(m.kvAProj)
		w.f32(m.kvALayernorm)
		w.f32(m.kvBProj)
		w.f32(m.oProj)
	}
	// Mamba-2 (Granite / Nemotron): presence byte then the eight tensors. The recurrent STATE is
	// not written and must not be — it is per-sequence, rebuilt at load; only the weights are here.
	if l.mamba == nil {
		w.raw([]byte{0})
	} else {
		w.raw([]byte{1})
		m := l.mamba
		w.mambaProj(m.inProj, &m.inQ)
		w.f32(m.convW)
		w.f32(m.convB)
		w.f32(m.aLog)
		w.f32(m.d)
		w.f32(m.dtBias)
		w.f32(m.normW)
		w.mambaProj(m.outProj, &m.outQ)
	}
}

// mambaProj writes a Mamba projection in the encoding it is held in: an int8 weight matrix (kind 2) when the load quantized
// it, else a standalone f32 array (giwEncF32). Both start with their encoding byte, so the reader tells them apart.
func (w *giwWriter) mambaProj(f []float32, q *linalg.WeightMat) {
	if q.Rows() > 0 {
		w.weightMatKind3Only(q)
		return
	}
	w.f32(f)
}

// v8Layer writes the LFM2 gated short-convolution mixer: presence byte then the three tensors. Unconditional like the v6
// tail and for the same reason; the cost is one zero byte per layer on every other family. A writer that omits this field
// produces a CRC-valid bundle whose first forward nil-dereferences on conv layer 0 (lw.shortConv == nil), and prequant's
// selfCheck cannot see it because it only Loads.
//
// As with mamba, only the weights are here: the rolling conv window is per-sequence state that lives in the KVCache and
// is rebuilt at load.
func (w *giwWriter) v8Layer(l *LayerWeights) {
	if l.shortConv == nil {
		w.raw([]byte{0})
		return
	}
	w.raw([]byte{1})
	c := l.shortConv
	w.f32(c.inProj)
	w.f32(c.convW)
	w.f32(c.outProj)
}

// v9Layer writes Bailing Hybrid's (Ling 3.0) per-layer v9 tail: MLA's optional attention-output gate (l.mla.gProj, added to
// mlaWeights after v6Layer's MLA block shipped, so it rides a new version rather than retrofitting v6Layer's fixed byte
// layout, which would corrupt the read of every existing v6/v7/v8 file), then the KDA mixer (presence byte + the thirteen
// tensors, kdaWeights' own field count). Without it a round-tripped bailing_hybrid bundle nil-dereferences in
// kdaMixerStep on the first KDA layer; TestSerializeCensus_noSilentFieldDrop catches a missing field.
func (w *giwWriter) v9Layer(l *LayerWeights) {
	var gProj []float32
	if l.mla != nil {
		gProj = l.mla.gProj // nil for deepseek_v2/v3/kimi_k2 (no gate); set for Bailing Hybrid
	}
	w.f32(gProj)
	if l.kda == nil {
		w.raw([]byte{0})
		return
	}
	w.raw([]byte{1})
	k := l.kda
	w.weightMatKind3Only(&k.qProj)
	w.weightMatKind3Only(&k.kProj)
	w.weightMatKind3Only(&k.vProj)
	w.f32(k.qConvW)
	w.f32(k.kConvW)
	w.f32(k.vConvW)
	w.weightMatKind3Only(&k.fProj)
	w.f32(k.dtBias)
	w.f32(k.aLog)
	w.f32(k.bProj)
	w.weightMatKind3Only(&k.gProj)
	w.f32(k.oNormW)
	w.weightMatKind3Only(&k.oProj)
}

// gemma4Layer writes the v4 Gemma 4 per-layer tail: the PLE branch, the per-layer
// output scalar, the KV-share flags, and (when enable_moe_block) the gemma4moe
// sub-block's OWN tensors — the ones NOT already covered by the standard block. The
// dense-branch MLP + its pre/post norms are l.GateProj/UpProj/DownProj +
// l.PreMLPNorm/PostMLPNorm (already written), which the reader re-aliases into
// gemma4moe; only the three parallel-branch norms, the router (l.Router is empty for
// gemma4), the per-expert scale, and the fused experts are new here.
func (w *giwWriter) gemma4Layer(l *LayerWeights) {
	w.weightMat(&l.PLEGate)
	w.weightMat(&l.PLEProj)
	w.f32(l.PostPLENorm)
	w.u32(math.Float32bits(l.LayerScalar))
	w.raw([]byte{b2u8(l.KVShared), b2u8(l.VFromK)})
	if l.gemma4moe == nil {
		w.raw([]byte{0})
		return
	}
	w.raw([]byte{1})
	mo := l.gemma4moe
	w.f32(mo.postFFNNorm1)
	w.f32(mo.preFFNNorm2)
	w.f32(mo.postFFNNorm2)
	w.weightMatKind3Only(&mo.routerProj)
	w.f32(mo.routerScale)
	w.f32(mo.perExpertScale)
	w.u32(uint32(len(mo.expertsGateUp)))
	for e := range mo.expertsGateUp {
		w.weightMatKind3Only(&mo.expertsGateUp[e])
		w.weightMatKind3Only(&mo.expertsDown[e])
	}
}

func b2u8(b bool) byte {
	if b {
		return 1
	}
	return 0
}

// hybridLayer writes the qwen3_5_moe per-layer extras (v2): a kind byte then the
// DeltaNet (linear-layer) or gated-softmax (qattn) f32 tensor set. Every other
// family writes kind 0 and nothing more, so the field stays one byte per layer.
func (w *giwWriter) hybridLayer(l *LayerWeights) {
	switch {
	case l.delta != nil:
		w.raw([]byte{1})
		d := l.delta
		w.weightMatKind3Only(&d.inProjQKV)
		w.weightMatKind3Only(&d.inProjZ)
		w.f32(d.inProjB)
		w.f32(d.inProjA)
		w.f32(d.convW)
		w.f32(d.dtBias)
		w.f32(d.negExpA) // the precomputed −exp(A_log); stored as-is, no recompute on load
		w.f32(d.normW)
		w.weightMatKind3Only(&d.outProj)
	case l.qattn != nil:
		w.raw([]byte{2})
		q := l.qattn
		w.weightMatKind3Only(&q.qProj)
		w.weightMatKind3Only(&q.kProj)
		w.weightMatKind3Only(&q.vProj)
		w.weightMatKind3Only(&q.oProj)
		w.f32(q.qNorm)
		w.f32(q.kNorm)
	default:
		// The "no hybrid extras" marker. MLA and Mamba-2 are written by v6Layer.
		w.raw([]byte{0})
	}
}

// --- reader (cursor over data; big arrays aliased, floats copied) ---

type giwReader struct {
	data []byte
	off  int
	err  error
	arch *Architecture // resolved from the serialized config; gates the gemma4 tail
}

func (r *giwReader) fail(msg string) {
	if r.err == nil {
		r.err = &SerializeError{msg}
	}
}

func (r *giwReader) need(n int) bool {
	if r.err != nil {
		return false
	}
	if n < 0 || r.off+n > len(r.data) {
		r.fail("unexpected end of data")
		return false
	}
	return true
}

func (r *giwReader) u32() uint32 {
	if !r.need(4) {
		return 0
	}
	v := binary.LittleEndian.Uint32(r.data[r.off:])
	r.off += 4
	return v
}

func (r *giwReader) rawN(n int) []byte {
	if !r.need(n) {
		return nil
	}
	b := r.data[r.off : r.off+n]
	r.off += n
	return b
}

func (r *giwReader) str() string        { return string(r.rawN(int(r.u32()))) }
func (r *giwReader) bytesField() []byte { return r.rawN(int(r.u32())) }

func (r *giwReader) u8() byte {
	if !r.need(1) {
		return 0
	}
	b := r.data[r.off]
	r.off++
	return b
}

// align16 skips the padding the writer put before a 16-aligned position (giwWriter.align16): the end of the header and
// the two blocks of a fused group.
func (r *giwReader) align16() {
	if r.err != nil {
		return
	}
	if pad := int((-int64(r.off)) & 15); r.need(pad) {
		r.off += pad
	}
}

func (r *giwReader) alignHead() { r.align16() }

// alignArray skips the zero padding the writer put before an aligned array's count (see giwArrayPad).
func (r *giwReader) alignArray() {
	if r.err != nil {
		return
	}
	if pad := giwArrayPad(int64(r.off)); r.need(pad) {
		r.off += pad
	}
}

// nativeLittleEndian reports whether float32s can be read straight out of the little-endian blob.
var nativeLittleEndian = func() bool { x := uint16(1); return *(*byte)(unsafe.Pointer(&x)) == 1 }()

// f32Alias reads a padded f32 array as an ALIAS of the mapping when its payload is 4-byte aligned
// in memory (always true for a blob inside a v3-bundle mapping), and copies otherwise, so an
// oddly-placed blob still loads, onto the heap. For weight-matrix scale arrays only: the mapping is read-only, and a
// scale array is read-only by construction (its sibling nibble/int8 arrays are already aliased).
// The caller must have called alignArray first.
func (r *giwReader) f32Alias() []float32 {
	n := int(r.u32())
	if n == 0 || !r.need(n*4) {
		return nil
	}
	if p := unsafe.Pointer(&r.data[r.off]); nativeLittleEndian && uintptr(p)%4 == 0 {
		r.off += n * 4
		return unsafe.Slice((*float32)(p), n)
	}
	out := make([]float32, n)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(r.data[r.off:]))
		r.off += 4
	}
	return out
}

// f32 reads a standalone f32 array (giwWriter.f32): its encoding byte, which must be giwEncF32, then the array.
func (r *giwReader) f32() []float32 {
	if enc := r.u8(); enc != giwEncF32 {
		r.unknownEncoding(enc)
		return nil
	}
	return r.f32Body()
}

// unknownEncoding refuses a tensor whose encoding this build does not read. Only that encoding is refused: a newer writer
// adds an encoding without a version bump, and a bundle that never uses it still loads.
func (r *giwReader) unknownEncoding(enc byte) {
	r.fail(fmt.Sprintf("a tensor's encoding %d is not one this build reads (written by a newer goinfer; the sidecar is rebuilt)", enc))
}

// f32Body copies (the input isn't guaranteed aligned). len 0 ⇒ nil, preserving the
// "absent ⇒ nil" convention the forward pass checks for biases/norms.
func (r *giwReader) f32Body() []float32 {
	n := int(r.u32())
	if n == 0 || !r.need(n*4) {
		return nil
	}
	out := make([]float32, n)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(r.data[r.off:]))
		r.off += 4
	}
	return out
}

// i8 ALIASES the int8 weight bytes over data (zero-copy). Read-only at inference.
func (r *giwReader) i8() []int8 {
	n := int(r.u32())
	if n == 0 || !r.need(n) {
		return nil
	}
	b := r.data[r.off : r.off+n]
	r.off += n
	return unsafe.Slice((*int8)(unsafe.Pointer(&b[0])), n)
}

// u16View returns n little-endian uint16s at the cursor as an ALIAS of the blob when 2-byte aligned on a little-endian
// host, else a copy: f32Alias's rule for f16 scales.
func (r *giwReader) u16View(n int) []uint16 {
	if n == 0 || !r.need(2*n) {
		return nil
	}
	if p := unsafe.Pointer(&r.data[r.off]); nativeLittleEndian && uintptr(p)%2 == 0 {
		r.off += 2 * n
		return unsafe.Slice((*uint16)(p), n)
	}
	out := make([]uint16, n)
	for i := range out {
		out[i] = binary.LittleEndian.Uint16(r.data[r.off:])
		r.off += 2
	}
	return out
}

// int4Scales reads an int4 scale array (kinds 3 and 5) as binary16, aliased from the mapping. The caller must have
// called alignArray first.
func (r *giwReader) int4Scales() []uint16 { return r.u16View(int(r.u32())) }

// rawAlias ALIASES a []byte (int4 packed nibbles): a plain subslice of data.
func (r *giwReader) rawAlias() []byte {
	n := int(r.u32())
	if n == 0 || !r.need(n) {
		return nil
	}
	b := r.data[r.off : r.off+n]
	r.off += n
	return b
}

func (r *giwReader) weightMat() linalg.WeightMat {
	if !r.need(1) {
		return linalg.WeightMat{}
	}
	kind := r.data[r.off]
	r.off++
	if kind == 0 {
		return linalg.WeightMat{}
	}
	if kind != 1 && kind != 2 && kind != 3 && kind != 5 { // kind 6 is read as a fused group, never here
		r.unknownEncoding(kind)
		return linalg.WeightMat{}
	}
	rows, cols, group := int(r.u32()), int(r.u32()), int(r.u32())
	if !r.need(1) {
		return linalg.WeightMat{}
	}
	w8a8 := r.data[r.off] == 1
	r.off++
	// rows/cols/group are blob-controlled. linalg.Wrap{Int8,Int4} PANIC on a length mismatch or group<=0, and a wrong-length
	// WrapF32 defers the panic to first use, so a one-byte flip with a recomputed CRC would crash the loader. Validate dims
	// and array lengths here (the arrays were already bounded by r.need); a mismatch is a *SerializeError via r.fail, not a
	// panic. The maxWeightDim cap keeps rows*cols from overflowing int before the equality check.
	const maxWeightDim = 1 << 26
	if rows <= 0 || cols <= 0 || rows > maxWeightDim || cols > maxWeightDim {
		r.fail(fmt.Sprintf("weightMat implausible dims %d×%d", rows, cols))
		return linalg.WeightMat{}
	}
	switch kind {
	case 1:
		f := r.f32Body()
		if len(f) != rows*cols {
			r.fail(fmt.Sprintf("f32 weightMat %d×%d has %d values", rows, cols, len(f)))
			return linalg.WeightMat{}
		}
		return linalg.WrapF32(f, rows, cols)
	case 2:
		r.alignArray()
		scales := r.f32Alias() // read order matches the writer (scales, then codes)
		r.alignArray()
		q8 := r.i8()
		if len(q8) != rows*cols || len(scales) != rows {
			r.fail(fmt.Sprintf("int8 weightMat %d×%d: q8=%d (want %d) scales=%d (want %d)", rows, cols, len(q8), rows*cols, len(scales), rows))
			return linalg.WeightMat{}
		}
		return linalg.WrapInt8(q8, scales, rows, cols, w8a8)
	case 3:
		r.alignArray()
		q4s := r.int4Scales()
		r.alignArray()
		q4 := r.rawAlias() // zero-copy alias into the mmap'd blob (WrapInt4 keeps it)
		if group <= 0 {
			r.fail(fmt.Sprintf("int4 weightMat group %d ≤ 0", group))
			return linalg.WeightMat{}
		}
		wantQ4, wantScales := rows*((cols+1)/2), rows*((cols+group-1)/group)
		if len(q4) != wantQ4 || len(q4s) != wantScales {
			r.fail(fmt.Sprintf("int4 weightMat %d×%d group=%d: q4=%d (want %d) q4s=%d (want %d)", rows, cols, group, len(q4), wantQ4, len(q4s), wantScales))
			return linalg.WeightMat{}
		}
		return linalg.WrapInt4F16(q4, q4s, rows, cols, group)
	case 5:
		if group <= 0 {
			r.fail(fmt.Sprintf("int4-row4-only weightMat group %d ≤ 0", group))
			return linalg.WeightMat{}
		}
		// The row4 arrays have kind 3's lengths: a repack, not a requant.
		wantQ4, wantScales := rows*((cols+1)/2), rows*((cols+group-1)/group)
		r.alignArray()
		q4Row4Scales := r.int4Scales()
		r.alignArray()
		q4Row4 := r.rawAlias() // zero-copy — no canonical bytes exist in this file at all
		if len(q4Row4) != wantQ4 || len(q4Row4Scales) != wantScales {
			r.fail(fmt.Sprintf("int4-row4-only weightMat %d×%d group=%d: q4Row4=%d (want %d) q4Row4Scales=%d (want %d)", rows, cols, group, len(q4Row4), wantQ4, len(q4Row4Scales), wantScales))
			return linalg.WeightMat{}
		}
		wm, ok := linalg.WrapInt4Row4OnlyF16(q4Row4, q4Row4Scales, rows, cols, group)
		if !ok {
			// Named and actionable: a kind-5 file is a promise to one target. ok=false here means Int4Row4Usable rejected
			// this core: wrong arch (the file was written for arm64), or a shape this core's build cannot run the row4
			// kernel for.
			r.fail(fmt.Sprintf("int4 weightMat %d×%d group=%d is stored row4-only (kind 5, a cpu-arm64 prequant target) but this core cannot use that layout — rebuild with `go run ./cmd/prequant -target <this core>` (or delete the stream-weights cache so it rebuilds automatically)", rows, cols, group))
			return linalg.WeightMat{}
		}
		return wm
	default:
		r.unknownEncoding(kind)
		return linalg.WeightMat{}
	}
}

// mambaProj reads giwWriter.mambaProj's record: an f32 array (giwEncF32) or a weight matrix, by its encoding byte.
func (r *giwReader) mambaProj() ([]float32, linalg.WeightMat) {
	if r.need(1) && r.data[r.off] == giwEncF32 {
		return r.f32(), linalg.WeightMat{}
	}
	return nil, r.weightMat()
}

// fusedPending is a kind-6 member whose header has been read; its nibbles and scales are in the group's two
// blocks, which follow the group's last record.
type fusedPending struct {
	dst               *linalg.WeightMat
	rows, cols, group int
}

// fusedGroup reads what giwWriter.fusedGroup wrote: for each member either an ordinary weightMat record or a
// kind-6 header, then, if any member was kind 6, the shared nibble block and the shared binary16 scale block, each
// member's part aliased out of them in member order.
func (r *giwReader) fusedGroup(dst ...*linalg.WeightMat) {
	var pend []fusedPending
	for _, d := range dst {
		if r.err == nil && r.need(1) && r.data[r.off] == 6 {
			r.off++
			rows, cols, group := int(r.u32()), int(r.u32()), int(r.u32())
			if !r.need(1) {
				return
			}
			r.off++ // w8a8 flag: unused for int4
			const maxWeightDim = 1 << 26
			if rows <= 0 || cols <= 0 || rows > maxWeightDim || cols > maxWeightDim || group <= 0 {
				r.fail(fmt.Sprintf("fused int4 member implausible dims %d×%d group=%d", rows, cols, group))
				return
			}
			pend = append(pend, fusedPending{dst: d, rows: rows, cols: cols, group: group})
			continue
		}
		*d = r.weightMat()
	}
	if len(pend) == 0 || r.err != nil {
		return
	}
	// Each block starts 16-aligned (the writer pads with the position, not with a length prefix). A member's nibbles
	// are a multiple of 16 bytes (K%32==0), so every member's nibbles stay 16-aligned and adjacent.
	r.align16()
	nibbles := make([][]byte, len(pend))
	for i, p := range pend {
		if nibbles[i] = r.rawN(p.rows * ((p.cols + 1) / 2)); r.err != nil {
			return
		}
	}
	r.align16()
	for i, p := range pend {
		scales := r.u16View(p.rows * ((p.cols + p.group - 1) / p.group))
		if r.err != nil {
			return
		}
		*p.dst = linalg.WrapInt4F16(nibbles[i], scales, p.rows, p.cols, p.group)
	}
}

func (r *giwReader) layer(l *LayerWeights) {
	r.fusedGroup(&l.QProj, &l.KProj, &l.VProj)
	l.OProj = r.weightMat()
	l.QBias = r.f32()
	l.KBias = r.f32()
	l.VBias = r.f32()
	l.OBias = r.f32()
	l.QNorm = r.f32()
	l.KNorm = r.f32()
	l.PreAttnNorm = r.f32()
	l.PreAttnNormBias = r.f32()
	l.PostAttnNorm = r.f32()
	r.fusedGroup(&l.GateProj, &l.UpProj)
	l.DownProj = r.weightMat()
	l.UpBias = r.f32()
	l.DownBias = r.f32()
	l.PreMLPNorm = r.f32()
	l.PreMLPNormBias = r.f32()
	l.PostMLPNorm = r.f32()
	l.Router = r.weightMat()
	l.RouterBias = r.f32() // DeepSeek/GLM e_score_correction_bias
	ne := int(r.u32())
	if ne < 0 || ne > maxSerializedExperts {
		r.fail("implausible expert count")
		return
	}
	// The per-layer half of the same bound, multiplied by the layer count: the reader already holds the resolved arch, so a
	// blob claiming more experts than the family has is refused before the structs exist rather than after.
	if r.arch != nil && r.arch.MoE != nil && r.arch.MoE.NumExperts > 0 && ne > r.arch.MoE.NumExperts {
		r.fail(fmt.Sprintf("expert count: blob has %d, arch expects at most %d",
			ne, r.arch.MoE.NumExperts))
		return
	}
	if ne > 0 {
		l.Experts = make([]expertWeights, ne)
		for e := range l.Experts {
			l.Experts[e].Gate = r.weightMat()
			l.Experts[e].Up = r.weightMat()
			l.Experts[e].Down = r.weightMat()
		}
	}
	l.SharedExpert.Gate = r.weightMat()
	l.SharedExpert.Up = r.weightMat()
	l.SharedExpert.Down = r.weightMat()
	l.SharedGate = r.weightMat()
	r.hybridLayer(l)
	if r.arch != nil && r.arch.gemma4 != nil {
		r.gemma4Layer(l)
	}
	r.v6Layer(l) // the completeness tail
	r.v8Layer(l) // the LFM2 short-conv tail
	r.v9Layer(l) // the KDA and MLA-gate tail
}

// v6Layer mirrors giwWriter.v6Layer.
func (r *giwReader) v6Layer(l *LayerWeights) {
	l.GProj = r.weightMat()
	l.AttnSinks = r.f32()
	for e := range l.Experts {
		l.Experts[e].GateBias = r.f32()
		l.Experts[e].UpBias = r.f32()
		l.Experts[e].DownBias = r.f32()
	}
	l.SharedExpert.GateBias = r.f32()
	l.SharedExpert.UpBias = r.f32()
	l.SharedExpert.DownBias = r.f32()
	if r.u8() != 0 {
		m := &mlaWeights{}
		m.qAProj = r.f32()
		m.qALayernorm = r.f32()
		m.qBProj = r.f32()
		m.qProj = r.f32()
		m.kvAProj = r.f32()
		m.kvALayernorm = r.f32()
		m.kvBProj = r.f32()
		m.oProj = r.f32()
		l.mla = m
	}
	if r.u8() != 0 {
		m := &mamba2Weights{}
		m.inProj, m.inQ = r.mambaProj()
		m.convW = r.f32()
		m.convB = r.f32()
		m.aLog = r.f32()
		m.d = r.f32()
		m.dtBias = r.f32()
		m.normW = r.f32()
		m.outProj, m.outQ = r.mambaProj()
		l.mamba = m
	}
}

// v8Layer mirrors giwWriter.v8Layer. Version-gated, so a v3-v7 bundle's layer block ends where it
// always did — and an LFM2 bundle written at v7 or earlier is one whose conv weights were never in
// the file at all, which validateShapes rejects rather than loading into a nil-deref.
func (r *giwReader) v8Layer(l *LayerWeights) {
	if r.u8() == 0 {
		return
	}
	c := &shortConvWeights{}
	c.inProj = r.f32()
	c.convW = r.f32()
	c.outProj = r.f32()
	l.shortConv = c
}

// v9Layer mirrors giwWriter.v9Layer. gProj is always present in the byte stream (possibly
// zero-length) regardless of l.mla, since the writer runs unconditionally — read it first and
// only attach it when this layer actually has MLA weights (v6Layer, read earlier, already set
// l.mla by the time this runs). Version-gated the same way v8Layer is, so a pre-v9 bundle's layer
// block ends where it always did.
func (r *giwReader) v9Layer(l *LayerWeights) {
	gProj := r.f32()
	if l.mla != nil {
		l.mla.gProj = gProj
	}
	if r.u8() == 0 {
		return
	}
	k := &kdaWeights{}
	k.qProj = r.weightMat()
	k.kProj = r.weightMat()
	k.vProj = r.weightMat()
	k.qConvW = r.f32()
	k.kConvW = r.f32()
	k.vConvW = r.f32()
	k.fProj = r.weightMat()
	k.dtBias = r.f32()
	k.aLog = r.f32()
	k.bProj = r.f32()
	k.gProj = r.weightMat()
	k.oNormW = r.f32()
	k.oProj = r.weightMat()
	l.kda = k
}

// gemma4Layer reads the v4 Gemma 4 per-layer tail and, when the MoE sub-block is
// present, reconstructs gemma4moe — re-aliasing the dense-branch MLP + pre/post norms
// already read into the standard block, and taking the fixed dims from the resolved
// arch (denseInter = IntermediateDim, moe/experts/topK from arch.MoE).
func (r *giwReader) gemma4Layer(l *LayerWeights) {
	l.PLEGate = r.weightMat()
	l.PLEProj = r.weightMat()
	l.PostPLENorm = r.f32()
	l.LayerScalar = math.Float32frombits(r.u32())
	l.KVShared = r.u8() != 0
	l.VFromK = r.u8() != 0
	if r.u8() == 0 { // no gemma4moe sub-block (dense E-model layer)
		return
	}
	mo := &gemma4MoEWeights{
		preFFNNorm:  l.PreMLPNorm,  // dense pre-norm (already read)
		postFFNNorm: l.PostMLPNorm, // joint post-norm (already read)
		mlpGate:     l.GateProj,
		mlpUp:       l.UpProj,
		mlpDown:     l.DownProj,
		layerScalar: l.LayerScalar,
		denseInter:  r.arch.IntermediateDim,
	}
	if m := r.arch.MoE; m != nil {
		mo.moeInter, mo.nE, mo.topK = m.IntermediateDim, m.NumExperts, m.TopK
	}
	mo.postFFNNorm1 = r.f32()
	mo.preFFNNorm2 = r.f32()
	mo.postFFNNorm2 = r.f32()
	mo.routerProj = r.weightMat()
	mo.routerScale = r.f32()
	mo.perExpertScale = r.f32()
	ne := int(r.u32())
	if ne < 0 || ne > maxSerializedExperts {
		r.fail("implausible gemma4moe expert count")
		return
	}
	mo.expertsGateUp = make([]linalg.WeightMat, ne)
	mo.expertsDown = make([]linalg.WeightMat, ne)
	for e := range ne {
		mo.expertsGateUp[e] = r.weightMat()
		mo.expertsDown[e] = r.weightMat()
	}
	l.gemma4moe = mo
}

// hybridLayer reconstructs the qwen3_5_moe per-layer extras written by
// giwWriter.hybridLayer. negExpA was stored precomputed, so it loads straight in.
func (r *giwReader) hybridLayer(l *LayerWeights) {
	switch r.u8() {
	case 0:
		// no hybrid extras (every non-qwen3_5_moe family)
	case 1:
		// Field order is the wire order: a struct literal evaluates its fields in source order, so these must be read in exactly
		// the order hybridLayer writes them.
		d := &deltaNetWeights{}
		d.inProjQKV = r.weightMat()
		d.inProjZ = r.weightMat()
		d.inProjB = r.f32()
		d.inProjA = r.f32()
		d.convW = r.f32()
		d.dtBias = r.f32()
		d.negExpA = r.f32()
		d.normW = r.f32()
		d.outProj = r.weightMat()
		l.delta = d
	case 2:
		a := &qwenAttnWeights{}
		a.qProj = r.weightMat()
		a.kProj = r.weightMat()
		a.vProj = r.weightMat()
		a.oProj = r.weightMat()
		a.qNorm = r.f32()
		a.kNorm = r.f32()
		l.qattn = a
	default:
		r.fail("unknown per-layer hybrid kind")
	}
}
