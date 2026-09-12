//go:build realckpt

// Per-tensor SOURCE quantization types, read from a GGUF's tensor-info table.
//
// Why a weightDiff test needs this: a GGUF's quantization is per-tensor, not per-file, and
// the "UD" (Unsloth dynamic) builds push that hard — unsloth/Qwen3.8-27B-GGUF UD-Q4_K_M
// carries NINE ggml types in one file (Q5_K, IQ4_XS, Q8_0, Q4_K, Q6_K, Q3_K, IQ4_NL, IQ3_S,
// F32), assigning a different bit-width to each tensor by sensitivity. So a single
// whole-file cosine floor is not a property of the loader: it silently becomes a floor on
// whichever tensor the quantizer chose to spend the fewest bits on. A weightDiff gate that
// wants to separate a loader TRANSFORM bug from dequant noise has to know which it is
// looking at, per tensor.
//
// aikit's embed.GGUFFile keeps the type unexported and offers no accessor, so this parses
// the tensor-info table directly. That table is the stable, documented front of the format
// (magic, version, tensor count, kv count, the kv block, then one info record per tensor)
// and nothing here touches tensor DATA — the loader under test is still the only thing that
// dequantizes. Keeping it test-only and realckpt-tagged is deliberate: it is an instrument
// for a gate, not a second reader for production to drift against.
package decoder

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

// ggmlTypeNames maps the ggml type enum to the names llama.cpp prints. Kept complete rather
// than minimal so an unrecognised type in a future file reports as a NUMBER we can look up,
// not as a silently-skipped tensor.
var ggmlTypeNames = map[uint32]string{
	0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
	10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
	16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL",
	21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64",
	28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4",
}

// ggufHeadReader is a sticky-error little-endian reader over the GGUF header. Sticky because
// the header is a chain of length-prefixed records: once one read is short every subsequent
// offset is meaningless, and checking each one inline would bury the parse.
type ggufHeadReader struct {
	r   io.Reader
	err error
}

func (h *ggufHeadReader) u32() uint32 {
	var v uint32
	h.read(&v)
	return v
}

func (h *ggufHeadReader) u64() uint64 {
	var v uint64
	h.read(&v)
	return v
}

func (h *ggufHeadReader) read(v any) {
	if h.err != nil {
		return
	}
	h.err = binary.Read(h.r, binary.LittleEndian, v)
}

func (h *ggufHeadReader) skip(n int64) {
	if h.err != nil || n == 0 {
		return
	}
	if n < 0 {
		h.err = fmt.Errorf("negative skip %d (corrupt length prefix)", n)
		return
	}
	if _, err := io.CopyN(io.Discard, h.r, n); err != nil {
		h.err = err
	}
}

func (h *ggufHeadReader) str() string {
	n := h.u64()
	if h.err != nil {
		return ""
	}
	if n > 1<<20 { // a tensor name or metadata key this long means we have lost the offset
		h.err = fmt.Errorf("implausible string length %d", n)
		return ""
	}
	b := make([]byte, n)
	if _, err := io.ReadFull(h.r, b); err != nil {
		h.err = err
		return ""
	}
	return string(b)
}

// ggufScalarWidth is the byte width of each non-string, non-array GGUF metadata value type.
var ggufScalarWidth = map[uint32]int64{
	0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8,
}

// skipValue advances past one metadata value of the given type. Metadata is not what this
// reader is after — it only has to be traversed exactly to reach the tensor-info table.
func (h *ggufHeadReader) skipValue(typ uint32) {
	if h.err != nil {
		return
	}
	switch typ {
	case 8: // string
		h.str()
	case 9: // array
		elem := h.u32()
		n := h.u64()
		if h.err != nil {
			return
		}
		if elem == 8 || elem == 9 { // strings and nested arrays are not fixed-width
			for i := uint64(0); i < n && h.err == nil; i++ {
				h.skipValue(elem)
			}
			return
		}
		w, ok := ggufScalarWidth[elem]
		if !ok {
			h.err = fmt.Errorf("unknown GGUF array element type %d", elem)
			return
		}
		h.skip(int64(n) * w)
	default:
		w, ok := ggufScalarWidth[typ]
		if !ok {
			h.err = fmt.Errorf("unknown GGUF value type %d", typ)
			return
		}
		h.skip(w)
	}
}

// ggufTensorQuants reports the source ggml quantization type of every tensor in the GGUF at
// path, keyed by tensor name (e.g. "blk.3.attn_k.weight" -> "Q4_K").
func ggufTensorQuants(path string) (map[string]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	h := &ggufHeadReader{r: f}
	var magic [4]byte
	h.read(&magic)
	if h.err != nil {
		return nil, h.err
	}
	if magic != [4]byte{'G', 'G', 'U', 'F'} {
		return nil, fmt.Errorf("%s: not a GGUF (magic %q)", path, magic)
	}
	ver := h.u32()
	if ver != 2 && ver != 3 {
		return nil, fmt.Errorf("%s: GGUF version %d, this reader handles 2 and 3", path, ver)
	}
	nTensor := h.u64()
	nKV := h.u64()
	if h.err != nil {
		return nil, h.err
	}
	for i := uint64(0); i < nKV && h.err == nil; i++ {
		h.str() // key
		h.skipValue(h.u32())
	}
	if h.err != nil {
		return nil, fmt.Errorf("%s: metadata block: %w", path, h.err)
	}
	out := make(map[string]string, nTensor)
	for i := uint64(0); i < nTensor && h.err == nil; i++ {
		name := h.str()
		nDims := h.u32()
		if h.err != nil {
			break
		}
		if nDims > 4 {
			h.err = fmt.Errorf("tensor %q reports %d dims", name, nDims)
			break
		}
		h.skip(int64(nDims) * 8) // dims
		typ := h.u32()
		h.skip(8) // data offset
		if h.err != nil {
			break
		}
		if n, ok := ggmlTypeNames[typ]; ok {
			out[name] = n
		} else {
			out[name] = fmt.Sprintf("type%d", typ)
		}
	}
	if h.err != nil {
		return nil, fmt.Errorf("%s: tensor-info table: %w", path, h.err)
	}
	return out, nil
}

// ggufQuantCosFloor is the per-source-quant agreement floor a CORRECT loader must clear when
// its f32 reconstruction is diffed against the bf16 safetensors reference.
//
// These are dequant noise budgets, not quality targets. Each sits at roughly 2x the MEASURED
// (1-cos) for that format, from TestQwen38GGUF_weightDiff over layers 0-3 of
// unsloth/Qwen3.8-27B-GGUF UD-Q4_K_M against the bf16 safetensors (2026-09-12, 44 s,
// goinfer-logs/qwen38-weightdiff-20260912-103901.log):
//
//	quant  n   measured cosine      1-cos      floor   budget used
//	F32    18  1.000000 (maxAbs 0)  0          0.999999  0%
//	Q8_0    6  0.999982-0.999986    1.7e-5     0.9999    17%
//	Q6_K    2  0.999742-0.999758    2.5e-4     0.9995    52%
//	Q5_K    6  0.999186-0.999257    8.0e-4     0.998     40%
//	Q4_K    5  0.996974-0.997047    3.0e-3     0.995     61%
//
// What makes that table evidence rather than a curve fit: the cosine is a function of the
// SOURCE QUANT ALONE. Q4_K spans 7e-5 across two different layer kinds (DeltaNet in_proj_qkv
// and in_proj_z, softmax k_proj) and five tensor roles and shapes; Q5_K likewise. A transform
// defect cannot produce agreement that tracks bit-width and ignores what the tensor is for.
//
// The quants not present in that file (Q4_0/Q4_1/Q5_0/Q5_1, the IQ and TQ families, MXFP4)
// are set from the same first-principles model the measured ones confirm to within 3e-4 — for
// a k-quant with 32-element sub-blocks over roughly-Gaussian weights, a b-bit code has step
// ~4.2*sigma/(2^b-1), so relL2 ~ step/sqrt(12)/sigma and 1-cos ~ relL2^2/2, predicting Q4_K
// 0.9967 / Q5_K 0.99924 / Q6_K 0.99982 / Q8_0 0.99995 against the measurements above.
//
// The margin matters less than it looks, because the defect class these gates exist for does
// not produce a near-miss. A wrong un-tile order PERMUTES elements, a missing (1+w) norm
// un-bake shifts every element by one, a sign error on -exp(A_log) inverts: all land near
// zero or negative cosine, orders of magnitude below even the 2-bit floor. The floors only
// have to sit above dequant noise and below "cratered", and that gap is enormous.
//
// THAT CLAIM WAS MEASURED, NOT ASSUMED — a loosened gate that can no longer go red is worth
// less than the tight one it replaced, and "it passes now" is exactly what that looks like.
// Deleting ONE real transform (the untileVHeads on in_proj_z in decoder/gguf.go's loadQ35)
// and rerunning took that tensor from 0.996974 to 0.045919 — 47704% of its budget, ~160x past
// the floor — while every other tensor stayed green, so the failure named the one broken
// transform. Dequant noise tops out at 61% of budget; a transform bug is three orders of
// magnitude past it. Run 2026-09-12, goinfer-logs/qwen38-weightdiff-MUTATION-20260912-*.log.
//
// An unrecognised quant deliberately gets the old whole-file bar: a format nobody has
// calibrated must not silently widen a gate.
func ggufQuantCosFloor(q string) float64 {
	switch q {
	case "F32", "F64":
		return 0.999999
	case "F16", "BF16", "Q8_0", "Q8_1", "Q8_K":
		return 0.9999
	case "Q6_K":
		return 0.9995
	case "Q5_0", "Q5_1", "Q5_K":
		return 0.998
	case "Q4_0", "Q4_1", "Q4_K", "IQ4_NL", "IQ4_XS":
		return 0.995
	case "MXFP4":
		return 0.99
	case "Q3_K", "IQ3_S", "IQ3_XXS":
		return 0.98
	case "Q2_K", "IQ2_S", "IQ2_XS", "IQ2_XXS", "TQ2_0":
		return 0.95
	case "IQ1_S", "IQ1_M", "TQ1_0":
		return 0.90
	default:
		return 0.999
	}
}
