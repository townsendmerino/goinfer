package giwcodec

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/crc32"
	"io"

	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/decoder/config"
	"github.com/townsendmerino/goinfer/internal/runtime/families"
)

// This file defines the binary format of an already-quantized *Weights bundle (a ".giw", goinfer weights), so the
// resident weights can be produced once and mapped, skipping the GGUF dequant and requant on every launch. The big
// Weight arrays and their scale arrays are aliased over the input slice at load (zero-copy: this is the speed and RAM win);
// the norm and bias vectors and kind 1's f32 matrices are copied.
//
// There is one layout. This build writes Version and reads only Version; a file at any other version is refused
// with a *SerializeError, and a sidecar that fails its freshness check is rebuilt from its source (decoder/sidecar.go).
// Every tensor record starts with its encoding (a weight matrix's kind, or EncF32 for a standalone array), and a reader
// refuses only an encoding it does not read, so a new encoding needs no new version (docs/giw-bundles.md, "From v17").
// There is no fallback to the GGUF: an embedded or explicitly named bundle that does not load is an error. The
// history of the earlier versions is in docs/code-notes/decoder.md ("The .giw format before v16").
//
// Format (little-endian throughout):
//
//	magic   [5]byte = "GINFW"
//	version uint32   (Version)
//	id      str      (model identity, the source's basename; not validated on read)
//	config  str      (Config as JSON; arch is re-derived from it on load)
//	quantLabel str   (the resolved quant label (int4|int4mix|int8int8|int8|native), or "" for a streamed bundle,
//	                  whose label the reader infers from the tensors)
//	zero padding to a 16-byte boundary
//	Embed, LMHead, PosEmbed     weightMat
//	FinalNorm, FinalNormBias    f32
//	(gemma4 only: the model-level Per-Layer-Embedding inputs)
//	numLayers uint32
//	  per layer: the LayerWeights fields in the order Writer.layer writes them, with the per-family tails
//	crc     uint32   (CRC32-IEEE over every preceding byte)
//
// str  = uint32 len + len bytes
// f32  = uint8 encoding (EncF32) + uint32 len + len*4 LE-float32 bytes   (len 0 => nil on load); a weight matrix's own
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
//	1  f32: an f32 array, not aligned (it is copied). Written by builds before kind 10. Read for an MoE router; a bundle whose
//	   f32 body is in this kind is refused (ErrCopiedF32Bundle), and rebuilt where it is a sidecar.
//	10 f32: an f32 array, aligned, aliased on load
//	2  int8: f32 row scales, then i8 codes
//	3  int4, canonical: u16 binary16 group scales, then raw nibbles
//	5  int4, row4-only: u16 scales and raw nibbles in the arm64 split-half + 4-row-interleaved layout, with no canonical
//	   arrays. Written per tensor on a cpu-arm64 target only, for a tensor whose call site allows it (weightMat, not
//	   weightMatKind3Only) and whose shape repackRow4ForEmit accepts; everything else is kind 3. Loaded with
//	   linalg.WrapInt4Row4OnlyF16, which declines (a named *SerializeError) on a core that cannot run that layout. A
//	   storage choice, not a numerics one: dispatch is bit-identical either way.
//	6  int4, fused-group member (Metal target): a header with no arrays; the group's nibbles and then its binary16
//	   scales follow the group's last header as two unprefixed 16-aligned blocks (Writer.fusedGroup).
//	8  Q4_K (a --quant q4k load's native tensors): raw, the GGUF super-blocks as they are (144 bytes per 256 values),
//	   aliased on load through linalg.WrapQ4K. group is 0.
//	9  bf16: u16, rows×cols bf16 values, aliased on load through linalg.WrapBF16. group is 0. Written for an island
//	   matrix the load holds bf16 (Writer.island); an f32 island is a standalone f32 array instead.
//	11 Q5_K, 12 Q6_K: raw, as kind 8, for the file's Q5_K and Q6_K tensors a q4k load keeps (q4kNative56), aliased through
//	   linalg.WrapQ5K / WrapQ6K. Written on the cpu-amd64 and cuda targets only; every other target widens them to kind 2.
//
// Kinds 4 and 7 existed before v16 and are not read.

// FormatVersion is the weights format version this build writes and reads: the "vN" a sidecar's file name carries
// (decoder/sidecar.go:streamCachePath), so builds at different versions keep separate sidecars of one model.
const FormatVersion = Version

const (
	Magic = "GINFW"
	// Version is the one format version this build writes and reads. A file at any other version is refused, and a
	// sidecar is then rebuilt from its source (docs/giw-bundles.md, "One version").
	Version = 18
	// EncF32 is a standalone f32 array's encoding byte, in the weight-matrix kinds' namespace (kind 1 is f32). Every
	// tensor record starts with its encoding, and a reader refuses only an encoding it does not know (unknownEncoding).
	EncF32 = 1
	// KindF32 is an f32 weight matrix's kind: one aligned array of rows×cols values, aliased on load. No production F32() caller
	// writes through one, and the mapping is read-only, so one that did would fault (TestGIW_f32WeightsAreMapped).
	KindF32 = 10
	// giwKindBF16 is a bf16 weight matrix's kind: one aligned binary16 array of rows×cols values, aliased on load.
	giwKindBF16 = 9
	// giwKindQ5K and giwKindQ6K are a q4k load's native Q5_K and Q6_K tensors (q4kNative56): the GGUF super-blocks as they
	// are (176 and 210 bytes per 256 values), aliased on load. Written for the cpu-amd64 and cuda targets only.
	giwKindQ5K = 11
	giwKindQ6K = 12
	// Sanity ceilings on the count fields, generous against any real checkpoint but low enough that a corrupt or hostile blob
	// cannot drive a multi-GB make() before the body reader hits its first short read. A LayerWeights is a large struct, so an
	// unbounded layer count is the worst offender.
	MaxSerializedLayers  = 4096
	maxSerializedExperts = 4096

	// MaxSnapshotCacheBytes caps the KV cache a session snapshot may ask LoadSession to allocate (kvsnapshot.go). It bounds the
	// allocation rather than the blob, because a well-formed snapshot body can be small while pos is large: a never-written
	// ring and a KV-shared layer each serialise zero KV bytes, and a ring layer stores only min(count, W) rows. 16 GiB is far
	// above any legitimate session and far below the TBs the unbounded path could reach.
	MaxSnapshotCacheBytes int64 = 1 << 34
)

// SerializeError is returned by LoadSerializedWeights on any magic/version/
// quant/CRC mismatch, and by the writer for weights the format cannot hold (a bf16 or f16 load). It is distinct so callers can
// fall back to building from the source GGUF rather than treating a stale blob as fatal.
type SerializeError struct{ Reason string }

func (e *SerializeError) Error() string { return "decoder: serialized weights: " + e.Reason }

// CanSerialize reports why a model's per-layer state cannot round-trip through the .giw format, or nil if it can. It
// returns nil for every registered family today. Refusing a family up front beats emitting a CRC-valid bundle that
// nil-derefs at the first forward.
func CanSerialize(a *darch.Architecture) *SerializeError {
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
func SerializeWeights(w *darch.Weights, id string, f32Kind1, islandsF32 bool) ([]byte, error) {
	wr := &Writer{F32Kind1: f32Kind1, IslandsF32: islandsF32}
	if err := wr.WriteBundle(w, id); err != nil {
		return nil, err
	}
	wr.U32(crc32.ChecksumIEEE(wr.Buf)) // CRC over the body; appended last
	return wr.Buf, nil
}

// SerializeWeightsTo streams the same bundle directly to out, never materializing
// the whole blob in memory — so prequantizing a large model peaks at ~the resident
// weight size, not 2× (resident + blob). It returns the number of bytes written
// (body + trailing CRC). The big int8/int4 arrays are written straight from the
// resident slices (one write each); only the small per-tensor scale/norm vectors
// are buffered transiently. out should be a regular file for the prequant path.
func SerializeWeightsTo(out io.Writer, w *darch.Weights, id string, f32Kind1, islandsF32 bool) (int64, error) {
	wr := &Writer{Sink: out, F32Kind1: f32Kind1, IslandsF32: islandsF32}
	if err := wr.WriteBundle(w, id); err != nil {
		return wr.N, err
	}
	var crc [4]byte
	binary.LittleEndian.PutUint32(crc[:], wr.CRC) // running CRC over the body
	if _, err := out.Write(crc[:]); err != nil {
		return wr.N, err
	}
	return wr.N + 4, nil
}

// SerializeWeightsForTarget is SerializeWeights for a bundle promised to ONE consumer:
// on a cpu-arm64 target, every eligible int4 tensor (see weightMat vs weightMatKind3Only) writes kind 5 (row4-only)
// instead of kind 3; every other target, including GIWTargetNone, writes kind 3 for every int4 tensor exactly like
// SerializeWeights. TranscodeGIW/EnsureCachedGIW (sidecar.go) and cmd/prequant drive this.
func SerializeWeightsForTarget(w *darch.Weights, id string, target Target, f32Kind1, islandsF32 bool) ([]byte, error) {
	wr := &Writer{Target: target, F32Kind1: f32Kind1, IslandsF32: islandsF32}
	if err := wr.WriteBundle(w, id); err != nil {
		return nil, err
	}
	wr.U32(crc32.ChecksumIEEE(wr.Buf))
	return wr.Buf, nil
}

// SerializeWeightsToForTarget is SerializeWeightsTo with SerializeWeightsForTarget's
// target-aware kind-5 opt-in — see that function's doc for the contract.
func SerializeWeightsToForTarget(out io.Writer, w *darch.Weights, id string, target Target, f32Kind1, islandsF32 bool) (int64, error) {
	wr := &Writer{Sink: out, Target: target, F32Kind1: f32Kind1, IslandsF32: islandsF32}
	if err := wr.WriteBundle(w, id); err != nil {
		return wr.N, err
	}
	var crc [4]byte
	binary.LittleEndian.PutUint32(crc[:], wr.CRC)
	if _, err := out.Write(crc[:]); err != nil {
		return wr.N, err
	}
	return wr.N + 4, nil
}

// WriteBundle writes the bundle body (everything but the trailing CRC) via the
// writer's current sink (buffer or stream). Shared by SerializeWeights and
// SerializeWeightsTo so the field order can't drift between them or from the reader.
func (wr *Writer) WriteBundle(w *darch.Weights, id string) error {
	if err := CanSerialize(w.Arch); err != nil {
		return err
	}
	// The format has no 16-bit matmul weight kind (LFM2's bf16 conv islands, kind 9, are not matmul weights): refused before a
	// byte is written, not emitted as a bundle that leaves them out.
	for _, m := range w.MatmulWeights() {
		if k := m.Kind(); k == "bf16" || k == "f16" {
			return &SerializeError{"a .giw has no " + k.String() + " matmul weight kind: serialize a load at int4, int8, int8int8 or f32"}
		}
	}
	wr.Arch = w.Arch // gates the gemma4 model-level PLE + per-layer tail
	if err := wr.WriteHeadGlobals(w, id); err != nil {
		return err
	}
	for i := range w.Layers {
		wr.Layer(&w.Layers[i])
	}
	return wr.Err
}

// WriteHeadGlobals writes everything up to and including the layer count: the
// header (magic/version/id/config/label), the global tensors, and u32(len(Layers)).
// Split out so a streaming transcode can emit the head, then produce-write-free each
// layer one at a time (peak RAM ~one layer, not the whole model) before the loop in
// writeBundle. The layer count is len(w.Layers), so the streamer must allocate
// w.Layers to NumLayers up front even though it never fills them all at once.
func (wr *Writer) WriteHeadGlobals(w *darch.Weights, id string) error {
	cfgJSON, err := json.Marshal(w.Cfg)
	if err != nil {
		return fmt.Errorf("decoder: marshal config: %w", err)
	}
	wr.Raw([]byte(Magic))
	wr.U32(Version)
	wr.Str(id)
	wr.BytesField(cfgJSON)

	// The resolved quant label, recorded so the reader need not re-infer it. It is gated on whether w.Layers is
	// populated, not on which writer is in use. The incremental GGUF transcode calls writeHeadGlobals on a freshly make()'d,
	// all-zero Layers slice before any layer streams, where quantLabel() cannot see real data and its default case returns
	// "native", a real quant mode, so calling it unconditionally would bake a false "native" label into every streamed
	// bundle. A caller that already holds a fully loaded *Weights (residentDirBody) and merely chooses the streaming API
	// for its I/O shape does get the label, so a buffered and a streamed call on the same model produce identical bytes.
	label := ""
	if w.HasPopulatedLayers() {
		label = w.QuantLabel()
	}
	wr.Str(label)
	wr.AlignHead() // the variable-length header (the label, present or not) must not shift what follows

	wr.HeadTable(&w.Embed)
	wr.HeadTable(&w.LMHead)
	wr.WeightMatKind3Only(&w.PosEmbed)
	wr.F32(w.FinalNorm)
	wr.F32(w.FinalNormBias)

	// Gemma 4 model-level Per-Layer-Embedding inputs (empty on the PLE-free
	// E-model/26B variants, but present as empty WeightMats so the layout is stable).
	// Written for gemma4 only; no other family's bundle carries them.
	if wr.Arch != nil && wr.Arch.Gemma4 != nil {
		wr.HeadTable(&w.PerLayerTokenEmbed)
		wr.WeightMat(&w.PerLayerModelProj)
		wr.F32(w.PerLayerProjNorm)
		// FFNPerLayer (the E-models' variable per-layer FFN widths) is json:"-" on
		// Config, so it does NOT survive the config-JSON round-trip — without it ffnAt()
		// falls back to IntermediateDim and mis-sizes the MLP matmuls. Carry it here.
		ffn := wr.Arch.Gemma4.FFNPerLayer
		wr.U32(uint32(len(ffn)))
		for _, v := range ffn {
			wr.U32(uint32(v))
		}
	}

	wr.U32(uint32(len(w.Layers)))
	return wr.Err
}

// LoadSerializedWeights reconstructs a *Weights from a SerializeWeights blob without any dequant or requant. Big
// int8/int4 arrays are aliased into data (zero-copy); float arrays are copied. data MUST stay alive for the returned
// model's lifetime (the aliased slices point into it). On any magic/version/quant/arch/CRC mismatch it returns a
// *SerializeError, which distinguishes a corrupt or stale bundle from an I/O failure. There is no fallback to the GGUF:
// callers return the error.
func LoadSerializedWeights(data []byte, f32Kind1 bool) (*darch.Weights, error) {
	return LoadSerializedWeightsWithCRC(data, false, f32Kind1)
}

// LoadSerializedWeightsWithCRC is LoadSerializedWeights with the whole-payload CRC optionally skipped —
// only for a caller that already holds proof THIS file passed it (see giwverify.go). Everything
// else (magic, version, header, bounds, arch, per-tensor validation) still runs.
func LoadSerializedWeightsWithCRC(data []byte, crcAlreadyVerified bool, f32Kind1 bool) (*darch.Weights, error) {
	return LoadSerializedWeightsHook(data, crcAlreadyVerified, nil, f32Kind1)
}

// ReadHeadReader reads a bundle's header (its Config JSON and baked quant label, "" for a streamed bundle, whose quant is inferred
// from the tensors), returning the reader positioned after it.
func ReadHeadReader(data []byte) (*Reader, []byte, string, error) {
	r := &Reader{Data: data}
	if got := r.RawN(len(Magic)); string(got) != Magic {
		return nil, nil, "", &SerializeError{fmt.Sprintf("bad magic %q (want %q)", got, Magic)}
	}
	if v := r.U32(); r.Err == nil && v != Version {
		return nil, nil, "", &SerializeError{fmt.Sprintf("format v%d, this build reads v%d — delete the sidecar (it rebuilds) "+
			"or rebuild with `go run ./cmd/prequant …`", v, Version)}
	}
	_ = r.Str() // id — stored for tooling, not validated here
	cfgJSON := r.BytesField()
	bakedQuant := r.Str()
	r.AlignHead()
	if r.Err != nil {
		return nil, nil, "", &SerializeError{"truncated header"}
	}
	return r, cfgJSON, bakedQuant, nil
}

// LoadSerializedWeightsHook is LoadSerializedWeightsWithCRC with beforeWalk, when non-nil, called once the header is read
// and its architecture resolved, before any weight matrix is: the point at which a caller that holds the file's
// mapping can prepare the page cache for the walk over the matrices (prepareColdLoad).
func LoadSerializedWeightsHook(data []byte, crcAlreadyVerified bool, beforeWalk func(*darch.Architecture), f32Kind1 bool) (*darch.Weights, error) {
	return LoadSerializedWeightsAbortable(data, crcAlreadyVerified, beforeWalk, nil, f32Kind1, nil)
}

// giwCRCChunk is the span the CRC pass checksums between looks at the abort channel: 64 MiB, a few milliseconds of checksum, so an abort
// is seen within a blink of the watch closing it.
const giwCRCChunk = 64 << 20

// LoadSerializedWeightsAbortable is LoadSerializedWeightsHook with abort, Options.LoadAbort, polled during the CRC pass (which reads every
// page of the mapping) and between layers of the walk. A closed abort returns an error wrapping abortErr; nil never fires.
func LoadSerializedWeightsAbortable(data []byte, crcAlreadyVerified bool, beforeWalk func(*darch.Architecture), abort <-chan struct{}, f32Kind1 bool, abortErr error) (*darch.Weights, error) {
	return WalkSerializedWeights(data, crcAlreadyVerified, beforeWalk, abort, false, f32Kind1, abortErr)
}

// ScanWeights reads a bundle's weight quant with no copy of its kind-1 matrices: they are skipped and come back empty, and shapes are
// not validated, so the Weights behind it answers what its matrices are, not a forward. It costs a walk of every tensor header, so it
// is for a question only a few loads ask.
func ScanWeights(blob []byte) (string, error) {
	w, err := WalkSerializedWeights(blob, true, nil, nil, true, false, nil)
	if err != nil {
		return "", err
	}
	return w.QuantLabel(), nil
}

// WalkSerializedWeights is the bundle reader; scan is ScanWeights' mode.
func WalkSerializedWeights(data []byte, crcAlreadyVerified bool, beforeWalk func(*darch.Architecture), abort <-chan struct{}, scan bool, f32Kind1 bool, abortErr error) (*darch.Weights, error) {
	aborted := func() bool {
		select {
		case <-abort:
			return true
		default:
			return false
		}
	}
	r, cfgJSON, bakedQuant, err := ReadHeadReader(data)
	if err != nil {
		return nil, err
	}
	var cfg config.Config // parsed ahead of the refusal below, so a corrupt head reads as corrupt and not as a bundle to rebuild
	if err := json.Unmarshal(cfgJSON, &cfg); err != nil {
		return nil, &SerializeError{"config json: " + err.Error()}
	}
	r.Scan = scan
	r.F32Kind1 = f32Kind1
	if r.OpensCopiedF32() { // before the CRC pass, which would read the whole file to reach the same answer
		return nil, ErrCopiedF32Bundle
	}

	// CRC: verify the whole payload (everything before the trailing crc word)
	// before trusting any offsets/lengths.
	if len(data) < 4 {
		return nil, &SerializeError{"too short"}
	}
	if !crcAlreadyVerified {
		body, want := data[:len(data)-4], binary.LittleEndian.Uint32(data[len(data)-4:])
		var got uint32
		for off := 0; off < len(body); off += giwCRCChunk {
			if aborted() {
				return nil, fmt.Errorf("decoder: .giw load aborted during the CRC pass: %w", abortErr)
			}
			got = crc32.Update(got, crc32.IEEETable, body[off:min(off+giwCRCChunk, len(body))])
		}
		if got != want {
			return nil, &SerializeError{fmt.Sprintf("CRC mismatch (got %08x want %08x) — corrupt or truncated", got, want)}
		}
	}

	if cfg.Sampling != nil {
		cfg.Sampling = cfg.Sampling.Normalized() // the bundle's bytes are input, held to the same ranges as the checkpoint's
	}
	arch, _, err := families.Resolve(&cfg)
	if err != nil {
		return nil, &SerializeError{"arch: " + err.Error()}
	}
	r.Arch = arch // gates the gemma4 model-level inputs and per-layer tail
	if beforeWalk != nil {
		beforeWalk(arch)
	}

	w := &darch.Weights{Cfg: cfg, Arch: arch, Backing: data, BakedQuant: bakedQuant}
	w.Embed = r.WeightMat()
	w.LMHead = r.WeightMat()
	// A tied checkpoint (no output.weight) round-trips with an empty LMHead; every other loader sets TiedLMHead from lm_head
	// presence, so mirror that here. Without it the head reads as untied+empty and the forward emits all-zero logits (greedy
	// loops on token 0, sampling is uniform noise) with no error.
	arch.TiedLMHead = w.LMHead.Rows() == 0
	w.PosEmbed = r.WeightMat()
	w.FinalNorm = r.F32()
	w.FinalNormBias = r.F32()
	if arch.Gemma4 != nil { // gemma4 model-level PLE inputs
		w.PerLayerTokenEmbed = r.WeightMat()
		w.PerLayerModelProj = r.WeightMat()
		w.PerLayerProjNorm = r.F32()
		nf := int(r.U32())
		if nf < 0 || nf > MaxSerializedLayers {
			return nil, &SerializeError{"implausible ffn-per-layer count"}
		}
		if nf > 0 {
			ffn := make([]int, nf)
			for i := range ffn {
				ffn[i] = int(r.U32())
			}
			arch.Gemma4.FFNPerLayer = ffn
		}
	}
	n := int(r.U32())
	if n < 0 || n > MaxSerializedLayers {
		return nil, &SerializeError{"implausible layer count"}
	}
	// Compare with the arch before allocating, not after. ValidateShapes catches a mismatched count, but only once every layer
	// struct exists, and MaxSerializedLayers is ~40x a real model, so a hostile count amplifies that far on an exported entry
	// point before any check runs. The arch is already resolved by this point, so the comparison is free.
	if arch != nil && arch.NumLayers > 0 && n != arch.NumLayers {
		return nil, &SerializeError{fmt.Sprintf(
			"layer count: blob has %d, arch expects %d", n, arch.NumLayers)}
	}
	w.Layers = make([]darch.LayerWeights, n)
	for i := range w.Layers {
		if aborted() {
			return nil, fmt.Errorf("decoder: .giw load aborted at layer %d of %d: %w", i, n, abortErr)
		}
		r.Layer(&w.Layers[i])
	}
	if r.Err != nil {
		return nil, &SerializeError{"truncated body: " + r.Err.Error()}
	}
	if scan {
		return w, nil
	}
	if err := ValidateShapes(w, arch); err != nil {
		return nil, err
	}
	return w, nil
}

// ValidateShapes cross-checks the deserialized tensors against the architecture's expected dims. The .giw reader
// validates only internal consistency (array length vs the blob's own rows/cols), so a bundle whose Router declares
// rows = NumExperts+K, or an Embed/LMHead with the wrong vocab, passes every reader check and then writes past a
// config-sized scratch slice at decode (moeMLP's `make([]float32, NumExperts)`, the qDim/kvDim/vocab decodeScratch
// buffers): heap corruption from caller-supplied bytes (LoadSerializedWeights is exported). The GGUF and safetensors
// loaders do this cross-check too.
//
// Universal invariants (vocab and expert count are uniform in every serializable family) are always checked. The
// attention/FFN projection dims are checked per layer through the per-layer accessors, so gemma-4's per-layer geometry
// (FFNPerLayer, two-geom head dims) is covered without a model-level dim that would false-reject it.
func ValidateShapes(w *darch.Weights, arch *darch.Architecture) *SerializeError {
	if w.Embed.Rows() > 0 {
		if e := shapeEq("Embed", w.Embed.Rows(), arch.VocabSize); e != nil {
			return e
		}
		if e := shapeCols("Embed", w.Embed.Cols(), arch.HiddenDim); e != nil {
			return e
		}
	}
	if w.LMHead.Rows() > 0 { // 0 = tied (validated via Embed above)
		if e := shapeEq("LMHead", w.LMHead.Rows(), arch.VocabSize); e != nil {
			return e
		}
		if e := shapeCols("LMHead", w.LMHead.Cols(), arch.HiddenDim); e != nil {
			return e
		}
	}
	if w.PosEmbed.Rows() > 0 {
		if e := shapeCols("PosEmbed", w.PosEmbed.Cols(), arch.HiddenDim); e != nil {
			return e
		}
	}
	// The blob controls len(w.Layers), but the forward indexes arch.NumLayers: a short blob is an out-of-bounds layer read the
	// per-layer checks below never reach.
	if len(w.Layers) != arch.NumLayers {
		return &SerializeError{fmt.Sprintf("layer count: blob has %d, arch expects %d", len(w.Layers), arch.NumLayers)}
	}
	// The per-layer accessors (headDimAt/kvHeadsAt/ffnAt) collapse to the uniform Architecture fields for every non-gemma-4
	// family, so one per-layer check set covers all families, gemma-4 included. A projection is mandatory only where the
	// layer's program multiplies by it (needsAt), so a family that legitimately omits one (a routed layer's empty dense FFN,
	// gemma-4's MLP living in the MoE sub-block, a KV-shared layer's k/v) is not false-rejected.
	for i := range w.Layers {
		lw := &w.Layers[i]
		n := needsAt(w, arch, i)
		if e := validateLayerAttnShapes(lw, arch, i, n); e != nil {
			return e
		}
		if e := validateLayerMLPShapes(lw, arch, i, n); e != nil {
			return e
		}
		if e := validateLayerHybridShapes(lw, arch, i, n); e != nil {
			return e
		}
	}

	// The model-level PLE tail: gemma4's forward reads all three unconditionally when the arch declares PLE, and a bundle from
	// before v4 has none of them.
	//
	// PerLayerModelProj is [NumLayers*HiddenSizePerLayerInput, HiddenDim]: gguf.go builds it as
	// mat("per_layer_model_proj.weight", pleTotal, hidden) and forward_gemma4.go's matmul(&PerLayerModelProj, ...) agrees that
	// Rows() is pleTotal, not HiddenDim.
	if arch.Gemma4 != nil && w.PerLayerTokenEmbed.Rows() > 0 {
		pleTotal := arch.NumLayers * arch.Gemma4.HiddenSizePerLayerInput
		if e := shapeEq("PerLayerModelProj", w.PerLayerModelProj.Rows(), pleTotal); e != nil {
			return e
		}
		if got := w.PerLayerModelProj.Cols(); got != arch.HiddenDim {
			return &SerializeError{fmt.Sprintf("PerLayerModelProj: %d cols, arch expects %d", got, arch.HiddenDim)}
		}
		// PerLayerTokenEmbed rows are pleTotal wide (gguf.go builds it as embMat(..., VocabSize, pleTotal)); the lookup widens one
		// into a pleTotal-sized scratch.
		if e := shapeCols("PerLayerTokenEmbed", w.PerLayerTokenEmbed.Cols(), pleTotal); e != nil {
			return e
		}
		// PerLayerProjNorm is sized HiddenSizePerLayerInput (the RMSNorm runs over one pleDim-wide row of ctxAware), not
		// HiddenDim; gguf.go's vnorm and forward_gemma4.go's normalize agree.
		if e := shapeReq("PerLayerProjNorm", len(w.PerLayerProjNorm), arch.Gemma4.HiddenSizePerLayerInput); e != nil {
			return e
		}
	}
	return nil
}

func (w *Writer) Layer(l *darch.LayerWeights) {
	w.FusedGroup(&l.QProj, &l.KProj, &l.VProj)
	w.WeightMat(&l.OProj)
	w.F32(l.QBias)
	w.F32(l.KBias)
	w.F32(l.VBias)
	w.F32(l.OBias)
	w.F32(l.QNorm)
	w.F32(l.KNorm)
	w.F32(l.PreAttnNorm)
	w.F32(l.PreAttnNormBias)
	w.F32(l.PostAttnNorm)
	w.FusedGroup(&l.GateProj, &l.UpProj)
	w.WeightMat(&l.DownProj)
	w.F32(l.UpBias)
	w.F32(l.DownBias)
	w.F32(l.PreMLPNorm)
	w.F32(l.PreMLPNormBias)
	w.F32(l.PostMLPNorm)
	w.WeightMat(&l.Router)
	w.F32(l.RouterBias) // DeepSeek/GLM e_score_correction_bias
	w.U32(uint32(len(l.Experts)))
	for e := range l.Experts {
		w.WeightMatKind3Only(&l.Experts[e].Gate)
		w.WeightMatKind3Only(&l.Experts[e].Up)
		w.WeightMatKind3Only(&l.Experts[e].Down)
	}
	w.WeightMat(&l.SharedExpert.Gate)
	w.WeightMat(&l.SharedExpert.Up)
	w.WeightMat(&l.SharedExpert.Down)
	w.WeightMat(&l.SharedGate)
	w.hybridLayer(l)
	if w.Arch != nil && w.Arch.Gemma4 != nil {
		w.gemma4Layer(l)
	}
	w.v6Layer(l) // the completeness tail
	w.v8Layer(l) // the LFM2 short-conv tail
	w.v9Layer(l) // the KDA and MLA-gate tail
}
