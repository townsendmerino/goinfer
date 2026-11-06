package fit

import "strings"

// Rates are the per-element prices a checkpoint estimate is made at. The caller (package decoder) measures them through the loader's own
// quantizer, so this package carries no quant mode: it prices tensors, and the loader says what a tensor costs.
type Rates struct {
	// Matrix is the resident bytes of one element of an ordinary 2-D or larger matrix under the load's quant.
	Matrix float64
	// Embedding is the rate of a token-embedding table or LM head (safetensors only; the loaders hold them at their own precision).
	Embedding float64
	// Int8 is the W8A8 rate, used for the projections a quantized load holds at int8. It is read only when MambaProjInt8 or MLAProjInt8 is set.
	Int8 float64
	// MambaProjInt8: a quantized Nemotron-H holds its two Mamba projections at int8 rather than f32.
	MambaProjInt8 bool
	// MLAProjInt8: a load armed for int8 MLA holds the q-LoRA pair and kv-A projections at int8 rather than f32.
	MLAProjInt8 bool
	// Kept (GGUF only, nil for none) prices a tensor the load keeps in its stored encoding, and reports false for any other.
	Kept func(name string, dims []int) (bytes float64, ok bool)
}

// elems is the element count of a tensor of the given shape, 0 for an empty or non-positive dimension.
func elems(dims []int) int {
	n := 1
	for _, d := range dims {
		if d <= 0 {
			return 0
		}
		n *= d
	}
	return n
}

// GGUFWeightBytes sums the resident weight bytes of a GGUF's tensors from their names and shapes alone, and returns the share that is
// routed-expert weight (names carry "_exps"). arch is general.architecture. 1-D tensors stay f32 whatever the quant. A tensor dims
// does not know (ok false) is skipped, as an unreadable shape is unknown, not zero.
func GGUFWeightBytes(names []string, dims func(name string) ([]int, bool), arch string, r Rates) (total, experts int64) {
	nemoInt8 := (arch == "nemotron_h" || arch == "nemotron_h_moe") && r.MambaProjInt8
	var sum, exp float64
	for _, name := range names {
		d, ok := dims(name)
		if !ok {
			continue
		}
		n := elems(d)
		if n == 0 {
			continue
		}
		if len(d) < 2 {
			sum += 4 * float64(n)
			continue
		}
		if nemoInt8 && (strings.HasSuffix(name, ".ssm_in.weight") || strings.HasSuffix(name, ".ssm_out.weight")) {
			sum += r.Int8 * float64(n)
			continue
		}
		if r.MLAProjInt8 && MLAInt8TensorName(name) {
			sum += r.Int8 * float64(n)
			continue
		}
		// Mamba-2 and MLA tensors stay f32 whatever the requested quant: price them at what they cost.
		if F32PinnedTensorName(name) {
			sum += 4 * float64(n)
			continue
		}
		b := r.Matrix * float64(n)
		if r.Kept != nil {
			if kept, ok := r.Kept(name, d); ok {
				b = kept
			}
		}
		sum += b
		if strings.Contains(name, "_exps") {
			exp += b
		}
	}
	return int64(sum), int64(exp)
}

// SafetensorsWeightBytes is GGUFWeightBytes' safetensors twin: shape-only, 2-D+ tensors at r.Matrix, 1-D norms and biases at f32.
// It prices the requested quant, never the on-disk bytes: a checkpoint is usually f32 or bf16 on disk and shrinks several-fold when
// quantized on load, and an estimate wrong in the refusing direction is worse than none. shape reports ok false for a tensor it cannot read.
func SafetensorsWeightBytes(names []string, shape func(name string) ([]int, bool), r Rates) int64 {
	var total float64
	for _, name := range names {
		s, ok := shape(name)
		if !ok {
			continue
		}
		n := elems(s)
		if n == 0 {
			continue
		}
		// A GPTQ/AWQ auxiliary tensor is consumed during reconstruction and never kept resident.
		if gptqAWQAuxSuffix(name) {
			continue
		}
		// A bundled vision tower or projector loads at its own precision, independent of the text quant; f32 is the default.
		if visionTowerTensorName(name) {
			total += 4 * float64(n)
			continue
		}
		if len(s) < 2 {
			total += 4 * float64(n)
			continue
		}
		if r.MambaProjInt8 && (strings.HasSuffix(name, "mixer.in_proj.weight") || strings.HasSuffix(name, "mixer.out_proj.weight")) {
			total += r.Int8 * float64(n)
			continue
		}
		if r.MLAProjInt8 && MLAInt8TensorName(name) {
			total += r.Int8 * float64(n)
			continue
		}
		if F32PinnedTensorName(name) {
			total += 4 * float64(n)
			continue
		}
		// A GPTQ/AWQ qweight's on-disk shape is the packed [in/8, out] layout; the load unpacks it to the logical [in, out].
		if strings.HasSuffix(name, ".qweight") {
			n *= gptqAWQPackFactor
		}
		if embeddingTensorName(name) {
			total += r.Embedding * float64(n)
			continue
		}
		total += r.Matrix * float64(n)
	}
	return int64(total)
}

// KVBytesPerPosition is the flat KV cost of one position: K and V for every layer at kvHeads*headDim, f32 by default, f16 or int8
// (1.125 bytes per element including the per-row scale) when asked. It is 0 when the geometry is unknown. A model whose layers differ
// (hybrid, sliding-window, MLA) is priced per layer by the caller instead; this is the fallback for a geometry with no architecture.
func KVBytesPerPosition(layers, kvHeads, headDim int, kvF16, kvI8 bool) int64 {
	if layers <= 0 || kvHeads <= 0 {
		return 0
	}
	perElem := 4.0
	switch {
	case kvI8:
		perElem = 1.125
	case kvF16:
		perElem = 2
	}
	return int64(2 * perElem * float64(kvHeads*headDim) * float64(layers))
}

// F32PinnedTensorName reports whether a checkpoint tensor name is one the loaders ALWAYS keep resident at f32 regardless of the requested
// quant: a Mamba-2 mixer (Granite-4.0-H's "mamba.*" prefix, GGUF "ssm_*") or an MLA attention projection (GGUF
// "attn_q_a"/"attn_q_b"/"attn_kv_a_mqa"/"attn_kv_b", safetensors "q_a_proj"/"q_b_proj"/"kv_a_proj"/"kv_b_proj").
//
// DELIBERATELY NOT a bare "mixer."/"in_proj"/"out_proj" match: Nemotron-H shares the "mixer." prefix across its Mamba-2 (f32), attention,
// MLP and MoE layer types, and bare "in_proj"/"out_proj" collide with LFM2's conv mixer and qwen3.5/qwen3_next's DeltaNet hybrid, where
// SOME "linear_attn.in_proj_*" sub-tensors are f32 and OTHERS are genuinely quantized, so a name-only classifier cannot tell them apart.
// Nemotron-H's Mamba-2 in_proj/out_proj/conv1d ARE matched, via the full "mixer.in_proj"/"mixer.out_proj"/"mixer.conv1d" compound
// (never bare "mixer." or bare "in_proj"), which no other Nemotron block kind contains.
//
// A load armed for int8 MLA holds its q-LoRA pair and kv-A at int8 instead (MLAInt8TensorName, checked first).
//
// KNOWN RESIDUAL GAP, left unfixed rather than over-matched: MLA's output projection (self_attn.o_proj.weight / Bailing's dense.weight)
// and a LoRA-less q_proj are held f32, but "o_proj"/"q_proj"/"dense.weight" are ordinary quantizable tensor names in every non-MLA
// family, so they are priced at the load's rate: a small, honest under-estimate at int4
// (TestFitEstimate_f32PinnedMixersAgreeWithResident).
func F32PinnedTensorName(name string) bool {
	for _, s := range []string{
		"ssm_", "mamba.",
		"mixer.in_proj", "mixer.out_proj", "mixer.conv1d", // Nemotron-H Mamba-2 only (see doc comment)
		"attn_q_a", "attn_q_b", "attn_kv_a", "attn_kv_b",
		"q_a_proj", "q_b_proj", "kv_a_proj", "kv_b_proj",
		"_exps.bias", // gpt-oss's expert biases: stacked 2-D in the file, f32 per expert after the load
	} {
		if strings.Contains(name, s) {
			return true
		}
	}
	return false
}

// MLAInt8TensorName reports whether name is an MLA q-LoRA or kv-A projection, which a load armed for int8 MLA holds as per-row int8.
// kv_b is not one: it stays f32.
func MLAInt8TensorName(name string) bool {
	for _, s := range []string{"attn_q_a.", "attn_q_b.", "attn_kv_a_mqa.", "q_a_proj.", "q_b_proj.", "kv_a_proj"} {
		if strings.Contains(name, s) {
			return true
		}
	}
	return false
}

// visionTowerTensorName reports whether name belongs to a bundled multimodal vision tower or projector rather than the text decoder.
// The text loader never requests these names, but an estimate walks EVERY tensor in the checkpoint, so it must exclude them or they
// are swept into the text model's per-element quant price.
func visionTowerTensorName(name string) bool {
	return strings.Contains(name, "vision_tower.") || strings.Contains(name, "multi_modal_projector.")
}

// gptqAWQAuxSuffix reports whether name is a GPTQ/AWQ auxiliary tensor (qzeros/g_idx/scales) that is consumed during the one-time
// reconstruct-then-requantize step and never kept resident: pricing it as a weight matrix would double-count storage the load discards.
func gptqAWQAuxSuffix(name string) bool {
	for _, s := range []string{".qzeros", ".g_idx", ".scales"} {
		if strings.HasSuffix(name, s) {
			return true
		}
	}
	return false
}

// gptqAWQPackFactor is how many 4-bit codes GPTQ/AWQ pack into each int32 along qweight's packed dimension, so qweight's on-disk shape
// under-counts the logical [in, out] matrix by exactly this factor; the load unpacks it and requantizes.
const gptqAWQPackFactor = 8

// embeddingTensorName reports a checkpoint tensor the loaders keep at the embedding precision: a token-embedding table or an untied
// LM head, under the names the supported safetensors families use.
func embeddingTensorName(name string) bool {
	if name == "output.weight" { // InternLM2's LM head
		return true
	}
	for _, s := range []string{"embed_tokens.weight", "lm_head.weight", "tok_embeddings.weight", "wte.weight", "word_embeddings.weight", "embed_in.weight", "embed_out.weight"} {
		if name == s || strings.HasSuffix(name, "."+s) {
			return true
		}
	}
	return false
}
