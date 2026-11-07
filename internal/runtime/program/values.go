package program

// The constants of program.go's types. A program fingerprint hashes the values a program holds, not what a constant means
// to the code that reads it, so a changed constant could change execution while every program stays the same: this file
// stays in every parity key, and program.go, which declares only types, is a builder (paritykey.ProgramTypesFile).

const (
	OpNorm       OpKind = iota + 1 // in → out, the normalisation (or a copy, for post-only placement)
	OpProjQKV                      // in → the stream's q, k, v
	OpQKNorm                       // q, k in place
	OpRope                         // q, k in place
	OpAttnTemp                     // q in place
	OpAttention                    // stores this position's k, v; q → the stream's attention context
	OpOutGate                      // context in place, gated by a projection of in
	OpProjO                        // context → out
	OpFFN                          // in → out
	OpResidual                     // the residual += regA (and regB)
	OpMLA                          // in → out, the whole Multi-head Latent Attention sublayer (mlaAttentionWith)
	OpKDA                          // in → out, the whole Kimi Delta Attention mixer (kdaMixerStep), on the layer's recurrent state
	OpShortConv                    // in → out, LFM2's gated short convolution (shortConvStep), on the layer's conv window
	OpMamba2                       // in → out, the Mamba-2 selective-scan mixer (mamba2Step), on the layer's SSM state
	OpDeltaNet                     // in → out, the Gated DeltaNet mixer (gatedDeltaNetStep), on the layer's recurrent state
	OpGatedAttn                    // in → out, qwen3.5's gated attention sublayer: [query ‖ gate] q_proj, QK-norm, rope, attendQuery, ×sigmoid(gate), o_proj
	OpGemma4Attn                   // in → out, Gemma 4's attention sublayer (gemma4AttnToken): per-layer widths, shared KV, scale-less v-norm
	OpGemma4MoE                    // the residual in place: Gemma 4's parallel dense ‖ MoE block, which returns the whole layer output
	OpPLE                          // the residual in place: Gemma 4's per-layer-embedding branch
	OpLayerScale                   // the residual in place: × the layer's learned output scalar, when it has one
)

const (
	RegH Reg = iota // the residual stream
	RegX            // a sublayer's input
	RegA            // a sublayer's output
	RegB            // the parallel block's second output
)

const (
	SlotPreAttn NormSlot = iota + 1
	SlotPreMLP
	SlotPostAttn
	SlotPostMLP
	SlotFinal
)

const (
	NormRMS NormForm = iota + 1
	NormLayer
	NormCopy
)

const (
	QKRMS QKNormForm = iota + 1 // weighted RMSNorm (rmsNorm), four-way sums
	QKL2                        // Llama 4: parameter-free RMS over each head, sequential float64 (l2NormHeads)
)

const (
	RopeAtKernel  RopeKernel = iota + 1 // ropeAt: NeoX or GPT-J pairing, m-RoPE
	RopeMLAKernel                       // mlaRope: GPT-J pairing by de-interleaving, then NeoX (Llama 4, MLA)
)

const (
	AttnBatchedAcc64 AttnOrder = iota + 1 // attendBatchedHeads with float64 accumulation
	AttnSinksF64                          // attendSinksToken: inline float64 dots and float32 AV with a learned sink per head; gpt-oss's order before KS-O4, kept as its fidelity gate's exact arm
	AttnQuery                             // attendQuery: float64 dots, float32 accumulation of the values
	AttnMLA                               // the family attends through opMLA, over a latent cache
	AttnGemma4                            // the family attends through opGemma4Attn (gemma4Attend: float32 matmuls)
)

const (
	FfnGated FFNForm = iota + 1
	FfnNonGated
	FfnMoE
	FfnGptOss      // gpt-oss: a router with a logit bias, softmax over the top k, clamped SwiGLU experts with biases (gptOssMoEWith)
	FfnLlama4MoE   // Llama 4: top-k by logit, each expert's INPUT scaled by sigmoid(logit), plus an ungated shared expert (llama4MoEWith)
	FfnNemotronMoE // Nemotron 3 Nano: routeExperts over non-gated relu² experts, plus an ungated shared expert (nemotronMoEWith)
	FfnGemma4Dense // Gemma 4's dense GeGLU at the layer's own width, in its own call form (gemma4DenseFFN)
)

const (
	ActGeluTanh ActKind = iota
	ActSiLU
	ActReLU2 // ReLU-squared (relu(x)²) — Nemotron-H's non-gated MLP
	// ActGelu is the EXACT erf GELU — HF's "gelu", a different function from
	// "gelu_new"/"gelu_pytorch_tanh" (ActGeluTanh) rather than a spelling of it.
	// Appended deliberately: GatedActResident passes the ordinal straight to resident
	// kernels that know 0 (GELU-tanh) and 1 (SiLU), so the existing values cannot be
	// renumbered. Gated and non-gated archs both use it, and no resident kernel
	// implements it: residentFeatures derives FeatGatedGELUErf or FeatNonGatedGELUErf,
	// which no backend declares.
	ActGelu
)
