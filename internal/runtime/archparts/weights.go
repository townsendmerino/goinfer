package archparts

import (
	"github.com/townsendmerino/aikit/linalg"
)

// QwenAttnWeights holds a qwen3_5_moe softmax layer's gated attention, f32.
// q_proj is DOUBLE-WIDTH — per head it emits [query ‖ gate]; the forward splits
// it and gates the attention output by sigmoid(gate). See docs/qwen3_5_moe.md.
type QwenAttnWeights struct {
	// Quantizable: these stay f32 when no quant is requested.
	QProj linalg.WeightMat // [numHeads*headDim*2, hidden]  (query ‖ gate per head)
	KProj linalg.WeightMat // [numKVHeads*headDim, hidden]
	VProj linalg.WeightMat // [numKVHeads*headDim, hidden]
	OProj linalg.WeightMat // [hidden, numHeads*headDim]
	QNorm []float32        // [headDim]
	KNorm []float32        // [headDim]
}

// MLAWeights holds one DeepSeek MLA layer's projections, stored f32 (parity-first,
// like qwenAttnWeights). Queries route through the q_a/q_b LoRA bottleneck when
// qAProj != nil, else the direct qProj (V2-Lite). K/V down-project to the latent
// (kvAProj: the kv_lora_rank latent ‖ the qk_rope_head_dim rope key) and up-project
// per head from the normalized latent (kvBProj → k_nope ‖ v per head).
type MLAWeights struct {
	QAProj       []float32 // [q_lora_rank, hidden]                 (nil ⇒ direct qProj)
	QALayernorm  []float32 // [q_lora_rank]
	QBProj       []float32 // [numHeads*qk_head_dim, q_lora_rank]
	QProj        []float32 // [numHeads*qk_head_dim, hidden]        (V2-Lite direct path)
	KVAProj      []float32 // [kv_lora_rank+qk_rope_head_dim, hidden]
	KVALayernorm []float32 // [kv_lora_rank]
	KVBProj      []float32 // [numHeads*(qk_nope+v_head_dim), kv_lora_rank]
	OProj        []float32 // [hidden, numHeads*v_head_dim]
	GProj        []float32 // [numHeads] (head_wise) or [numHeads*v_head_dim] (element_wise); nil ⇒ no gate (every DeepSeek family)
	// qAQ, qBQ, qQ, kvAQ and oQ are those projections as per-row int8 on a load armed for it (mlaProjInt8, quantizeProjections)
	// or read from a .giw that carries them, their f32 slices then nil.
	QAQ, QBQ, QQ, KVAQ, OQ linalg.WeightMat
}

// ExpertWeights is one MoE expert: a gated (SwiGLU) MLP. Mixtral names these
// w1=gate, w3=up, w2=down and has no biases; gpt-oss adds a per-expert bias on
// each projection (the *Bias slices, nil for every other family).
type ExpertWeights struct {
	Gate linalg.WeightMat // [IntermediateDim, HiddenDim] (w1)
	Up   linalg.WeightMat // [IntermediateDim, HiddenDim] (w3)
	Down linalg.WeightMat // [HiddenDim, IntermediateDim] (w2)

	// gpt-oss per-expert biases (nil elsewhere).
	GateBias []float32 // [IntermediateDim]
	UpBias   []float32 // [IntermediateDim]
	DownBias []float32 // [HiddenDim]
}
