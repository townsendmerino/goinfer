package archparts

import (
	"github.com/townsendmerino/aikit/linalg"
)

// KDAWeights holds one Bailing Hybrid (Ling 3.0) Kimi Delta Attention layer's parameters,
// row-major, as HF stores them. Unlike Gated DeltaNet, q/k/v are three FULLY SEPARATE
// projections AND three separate depthwise causal convs (verified against the real
// modeling_bailing_moe_v3.py's BailingMoeV3KimiDeltaAttention — independent self.q_proj/k_proj/
// v_proj and self.q_conv1d/k_conv1d/v_conv1d, not Gated DeltaNet's single concatenated in_proj_qkv
// + combined conv). num_k_heads == num_heads and head_k_dim == head_v_dim == head_dim always for
// this family (no GVA-style key/value head expansion), so there is only one width to track.
type KDAWeights struct {
	QProj, KProj, VProj    linalg.WeightMat // [numHeads*headDim, hidden] each
	QConvW, KConvW, VConvW []float32        // [numHeads*headDim, K] each, depthwise, bias-free (fla's ShortConvolution default)
	FProj                  linalg.WeightMat // [numHeads*headDim, hidden] — decay-gate raw input (no_kda_lora path: config.no_kda_lora=true selects this single-linear form over a LoRA'd a/b split, which this family does not implement — see kdaArchitecture)
	DtBias                 []float32        // [numHeads*headDim] — PER-CHANNEL (Gated DeltaNet's is per-head)
	ALog                   []float32        // [numHeads] — raw A_log (kdaLowerBoundGate exponentiates internally, unlike Gated DeltaNet's precomputed −exp(A_log))
	BProj                  []float32        // [numHeads, hidden] — beta write-gate, per-head scalar (small; not quantized, mirrors qwen35's inProjB)
	GProj                  linalg.WeightMat // [numHeads*headDim, hidden] — output gate raw input (no_kda_lora path)
	ONormW                 []float32        // [headDim] gated-RMSNorm weight (sigmoid-activated — Gated DeltaNet's is SiLU)
	OProj                  linalg.WeightMat // [hidden, numHeads*headDim]
}
