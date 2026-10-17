package decoder

import "os"

// SSM forward test/isolation seams, read once at init (the package convention) rather
// than per layer/token inside the decode loop. The first two shape Granite's program (Architecture.graniteLayerOps).
var (
	ssmNoMul   = os.Getenv("GOINFER_SSM_NOMUL") != ""
	ssmSkipFFN = os.Getenv("GOINFER_SSM_SKIPFFN") != ""
	// ssmStopLayer truncates the per-token walker after this layer (-1 = run every layer): the resident-SSM bring-up's layer-sweep
	// seam. Only tests set it, through SetSSMStopLayerForTest (testhooks.go).
	ssmStopLayer = -1
)

// Granite-4.0-H (granitemoehybrid): each layer is a Mamba-2 selective-scan mixer (recurrent state in the cache) or GQA softmax
// attention (KV cache), every layer a routed+shared MoE. Four Granite scalars apply: embedding_multiplier scales the embedding,
// attention_multiplier is the attention q·k scale (Architecture.AttnScale), residual_multiplier scales every residual add, and
// logits_scaling divides the final logits (at the shared head). Its program is Architecture.graniteLayerOps.
//
// RoPE is skipped on a layer when the checkpoint sets position_embedding_type "nope", which the released granite-4.0-h models do
// (HF then builds no rotary_emb). The tiny fixture is roped, so only the real-checkpoint oracle exercises the nope path.
