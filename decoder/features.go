package decoder

import "slices"

import "maps"

// ResidentFeature is one architecture capability a resident (GPU) decode path must implement
// in order to run a model CORRECTLY.
//
// The failure mode this taxonomy exists to prevent is SILENT. A backend that admits a model
// needing a feature it has not implemented raises no error — it simply drops the feature and
// emits wrong logits (docs/metal-model-coverage.md found exactly this: Qwen3 running with
// QK-norm ignored, Mistral running full-attention past its window). Admission must therefore
// be a subset check: RequiredResidentFeatures(model) ⊆ the backend's implemented set, else
// decline to the staged/CPU path.
//
// Requirements are DERIVED from the loaded Architecture's own flags — never a hand-maintained
// per-arch list. That is the point: a newly registered arch is classified automatically, and a
// backend that has not implemented its features declines by default instead of mis-running.
// Adding a field to Architecture that changes the math means adding it here too; the
// registry-driven test (features_test.go) is what makes forgetting expensive.
//
// Note this layer sits ABOVE the arch flags, so it cannot catch an arch that fails to CLAIM a
// feature it needs — e.g. phi3Architecture once dropped sliding_window entirely, which made
// every path (CPU included) silently wrong. That class is a registry bug, caught by
// per-family parity, not by admission.
type ResidentFeature string

const (
	FeatQKNorm            ResidentFeature = "qk-norm"             // per-head Q/K RMSNorm before RoPE (Qwen3, GLM, Mellum)
	FeatSlidingWindow     ResidentFeature = "sliding-window"      // windowed attention (Mistral, Mellum, Phi-3-mini)
	FeatPartialRotary     ResidentFeature = "partial-rotary"      // RotaryDim < HeadDim (GLM-dense, some Phi)
	FeatPerLayerRoPE      ResidentFeature = "per-layer-rope"      // inv-freq/mscale differ per layer type (Mellum)
	FeatRopeMscale        ResidentFeature = "yarn-mscale"         // YaRN attention_factor != 1 (Mellum, long-ctx)
	FeatRMSAddOne         ResidentFeature = "rms-add-one"         // Gemma's (1+w) RMSNorm offset
	FeatEmbedScale        ResidentFeature = "embed-scale"         // √hidden embedding multiplier (Gemma)
	FeatAttnLogitSoftcap  ResidentFeature = "attn-logit-softcap"  // per-layer attention-score softcap (Gemma 2) — a real attention kernel
	FeatFinalLogitSoftcap ResidentFeature = "final-logit-softcap" // softcap·tanh(logits/softcap) applied ONCE after the LM head, host-side (Gemma 2/4)
	FeatSandwichNorm      ResidentFeature = "sandwich-norm"       // post-attn / post-MLP norms (Gemma)
	FeatLayerNorm         ResidentFeature = "layer-norm"          // mean-subtracting LayerNorm instead of RMSNorm (Cohere, GPT-2)
	FeatParallelBlock     ResidentFeature = "parallel-block"      // one shared input norm; attn+MLP sum into a single residual add (Cohere, GPT-J)
	FeatNoPE              ResidentFeature = "nope"                // per-layer NoPE: some layers skip RoPE entirely (Cohere2 global layers, Llama-4 iRoPE)
	FeatGatedGELU         ResidentFeature = "gated-gelu"          // gated MLP whose activation is GELU-tanh, not SiLU (Gemma)
	FeatNonGatedMLP       ResidentFeature = "non-gated-mlp"       // up→act→down, no gate (GPT-2, Nemotron relu²)
	FeatLearnedPos        ResidentFeature = "learned-pos"         // learned position embeddings, no RoPE (GPT-2)
	FeatOutBias           ResidentFeature = "out-bias"            // additive bias on the attn output proj (GPT-2)
	FeatLogitScale        ResidentFeature = "logit-scale"         // logits_scaling divisor (Granite)
	FeatMoE               ResidentFeature = "moe"                 // sparse mixture-of-experts FFN
	FeatMoEGatedShared    ResidentFeature = "moe-gated-shared"    // sigmoid-GATED always-on shared expert (Qwen2-MoE); ungated (GLM/DeepSeek) needs only FeatMoE
	FeatMLA               ResidentFeature = "mla"                 // latent-KV attention (DeepSeek, Kimi)
	FeatSSM               ResidentFeature = "ssm"                 // Mamba-2 mixer (Granite-4.0-H, Nemotron-H)
	// FeatDeltaNet bundles the TWO departures of the Gated-DeltaNet hybrids (qwen3_5_moe,
	// qwen3_next, qwen3_5), for the same reason FeatAttnSink bundles gpt-oss's three: they always
	// co-occur, and a backend that implemented one without the other would be admitted and then
	// silently run half the model wrong.
	//
	//  1. the recurrent delta rule replacing softmax attention on 3 of every 4 layers, with a
	//     fixed-size per-head matrix state (no KV cache, not position-truncatable); and
	//  2. attn_output_gate on the REMAINING layers — q_proj emits [query ‖ gate] per head at
	//     double width and the context is scaled by sigmoid(gate) before o_proj.
	//
	// (2) is spelled differently from Laguna's FeatAttnOutputGate — a separate g_proj through
	// softplus, not a fused double-width q through sigmoid — so the two are NOT interchangeable
	// and declaring one must not admit the other.
	FeatDeltaNet ResidentFeature = "deltanet" // Gated-DeltaNet mixer + fused attn output gate
	// FeatAttnSink bundles gpt-oss's THREE departures: the learned per-head sink in the softmax
	// denominator, the clamped interleaved-SwiGLU expert, and a router whose bias reaches the
	// WEIGHT rather than only the selection. No resident backend DECLARES it yet, so CUDA/Metal/
	// WebGPU all decline (TestGptOss_cudaWebgpuDecline).
	//
	// Metal HAS all three as end-to-end PROVEN kernels (2026-08-18: kernels.go's `attention`
	// sink term, moe.go's route_gptoss/swiglu_quant_gptoss/gemv_w4a8_moe_wacc_bias, the moe.go
	// isGptOss dispatch split — TestGptOssResidentParity, 8/8 argmax-exact, min cosine 0.9989 on
	// the tiny fixture) but still does not declare this: gpt-oss also needs FeatRopeMscale
	// (YaRN), which is held pending Mellum's own end-to-end gate — see the FeatRopeMscale note on
	// metal's residentBackendFeatures entry. Declaring FeatAttnSink alone would still correctly
	// decline gpt-oss via the missing FeatRopeMscale check, but the two are declared together
	// once Mellum unblocks it, so a reader sees one coherent "gpt-oss shipped" commit rather than
	// a half-declared feature set.
	//
	// CUDA has all three as gated KERNELS (cuda/gptoss_act.cu, plus the sink argument on both
	// attention kernels) and does not declare this either — but NOT for the reason this comment
	// used to give. It said the kernels were "LOADED but never DISPATCHED into a forward pass",
	// which the code contradicts: sinkArg is threaded into BOTH attention launches
	// (cuda/resident.go:1536 split-KV, :2243 decode) and launchGluSplitExpert is dispatched from
	// the MoE expert loop (:1720, :1844). The bridge is WRITTEN. What has never happened is a
	// real gpt-oss forward EXECUTING it, because the 20b MXFP4 checkpoint (~13.8 GB) does not fit
	// the CUDA box's 8 GB VRAM without the host<->VRAM MoE-streaming path. Written-but-unexercised
	// is a different, smaller gap than not-yet-attempted, and the distinction is the estimate.
	//
	// The genuinely missing CUDA piece is FeatOutBias: no o_proj-bias kernel and no wiring exists
	// there at all (grep OBias/out_bias across cuda/*.go). Kernel-level parity is still not
	// end-to-end parity — 2224441 declared on kernel evidence and was correctly reverted, which
	// is why neither flag is set here. WebGPU has none of this.
	FeatAttnSink       ResidentFeature = "attn-sink"        // see above: CPU-only until FeatOutBias exists on CUDA and ONE real gpt-oss forward has run resident
	FeatAttnOutputGate ResidentFeature = "attn-output-gate" // Laguna: ctx *= softplus(g_proj·h) applied BEFORE o_proj, plus a per-layer QUERY head count. CPU-only — no resident backend implements either, so CUDA/Metal/WebGPU all decline. Without this the family needs nothing CUDA lacks and would be ADMITTED-but-mis-run: the resident path would skip the gate entirely and still produce plausible logits.
	FeatShortConv      ResidentFeature = "short-conv"       // LFM2/LFM2.5: the gated short-convolution mixer that replaces attention on 22 of 30 layers (B,C,x = in_proj(h); conv = depthwise_causal_conv(B*x), no activation; out_proj(C*conv)), carrying a per-layer rolling window of the last K-1 inputs. CPU-only — no resident backend implements the conv OR its recurrent state. Declared for the SAME reason as FeatAttnOutputGate above: LFM2 is otherwise a plain GQA+QK-norm+SwiGLU model that needs nothing CUDA lacks, so without this it would be ADMITTED and then mis-run, with the resident path treating every conv layer as attention. The window also makes it stateful, so a resident runner would need the state plumbing FeatSSM/FeatDeltaNet have and this has not.
	FeatGemma4EModel   ResidentFeature = "gemma4-e-model"   // Gemma-4 E2B/E4B shape: per-layer embeddings (PLE, hidden_size_per_layer_input>0) + cross-layer shared-KV + variable per-layer FFN. runLayersGemma4 injects PLE per layer; the resident bridges (built for the PLE-free dense 12B/26B) implement NONE of it, so a resident runner would SKIP the PLE branch and silently mis-run. No resident backend declares it ⇒ all decline (CPU-only) until an E-model bridge lands.
	// FeatAttnTemp (Ministral 3, batch 2 G3): the Llama4-style attn-temp query scale
	// (AttnTempBeta/AttnTempOrigMaxPos) is new to the GENERIC forward path (decoder/attention.go,
	// decoder/forwardn.go) — no resident (GPU) backend's own kernels apply it, so admitting
	// mistral3 to any resident path today would silently drop the scale for every position past
	// original_max_position_embeddings, producing plausible-but-wrong logits at exactly the
	// context lengths the mechanism exists for. Same shape as FeatAttnOutputGate/FeatShortConv
	// above: otherwise a plain GQA+YaRN model needing nothing else CUDA/Metal/WebGPU lack, so
	// without this declaration it would be ADMITTED and mis-run rather than correctly declined.
	FeatAttnTemp ResidentFeature = "attn-temp"
	// FeatPostOnlyNorm (Olmo 3/Olmo Hybrid, batch 2 G2): NormPostOnly — no pre-norm at all, the
	// sublayer's OUTPUT is normalized before the residual add. Genuinely different from
	// FeatSandwichNorm (which normalizes BOTH the input and the output); no resident backend
	// implements it, so this stays CPU-only until one does, the same shape as FeatAttnTemp.
	FeatPostOnlyNorm ResidentFeature = "post-only-norm"
	// FeatQKNormWhole (Olmo 3/Olmo Hybrid): QK-norm computed over the FULL projected q/k vector
	// (one RMSNorm over num_heads*head_dim) rather than per-head — verified against the real
	// modeling_olmo3.py (`Olmo3RMSNorm(config.num_attention_heads * self.head_dim, ...)`), not
	// the standard per-head FeatQKNorm (Qwen3/Gemma3/Mellum). Different statistic AND a
	// differently-shaped weight tensor, so it is its own feature, not a variant of FeatQKNorm.
	FeatQKNormWhole ResidentFeature = "qk-norm-whole"
)

// residentFeatures derives the features this architecture actually needs from its own flags.
func (a *Architecture) residentFeatures() []ResidentFeature {
	var f []ResidentFeature
	add := func(need bool, x ResidentFeature) {
		if need {
			f = append(f, x)
		}
	}
	// FeatQKNorm is the PER-HEAD kernel; QKNormWhole needs FeatQKNormWhole instead (added
	// separately below), not both — a backend implementing only the per-head kernel must not
	// be admitted for the whole-vector families.
	add(a.QKNorm && !a.QKNormWhole, FeatQKNorm)
	add(a.SlidingWindow > 0, FeatSlidingWindow)
	// MLA (DeepSeek/Kimi) carries rope on a decoupled qk_rope slice handled INSIDE the MLA kernel
	// (FeatMLA), so its RotaryDim<HeadDim is not the generic partial-rope path — don't double-count
	// it as FeatPartialRotary (C6: the hand table lists MLA archs as [mla moe]).
	add(a.mla == nil && a.RotaryDim != 0 && a.RotaryDim < a.HeadDim, FeatPartialRotary)
	add(!a.ropeUniform(), FeatPerLayerRoPE)
	// YaRN attention_factor on ANY layer, not just layer 0 (C6). Mellum interleaves 3:1 and puts
	// YaRN only on its full_attention layers — layer 0 is a sliding/default layer (mscale 1), so a
	// layer-0 sample missed it and over-admitted Mellum2 as resident on backends that don't apply
	// the full-layer scaling. Mirrors ropeUniform's all-layer loop just above.
	yarnMscale := false
	for i := 0; i < a.NumLayers; i++ {
		if a.ropeMscale(i) != 1 {
			yarnMscale = true
			break
		}
	}
	add(yarnMscale, FeatRopeMscale)
	add(a.RMSAddOne, FeatRMSAddOne)
	add(a.AttnTempBeta != 0, FeatAttnTemp)
	add(a.EmbedScale > 1, FeatEmbedScale)
	// Split (9a-P2): the old single FeatLogitSoftcap conflated two different capabilities. The
	// attention-score softcap is a per-layer KERNEL; the final-logit softcap is one host-side
	// tanh after the LM head (like FeatEmbedScale's √hidden). Gemma 4 needs ONLY the latter, so
	// declining it for the former it does not use was over-broad. Backends declare each
	// separately — a backend that ships the host tanh but no attention-softcap kernel gets
	// FeatFinalLogitSoftcap alone.
	add(a.AttnLogitSoftcap != 0, FeatAttnLogitSoftcap)
	add(a.FinalLogitSoftcap != 0, FeatFinalLogitSoftcap)
	// Gate on the SPECIFIC placement, not "anything but Pre2" — NormParallel is a
	// third placement whose kernel is FeatParallelBlock, not sandwich norms.
	add(a.NormPlacement == NormSandwich4, FeatSandwichNorm)
	add(a.NormPlacement == NormPostOnly, FeatPostOnlyNorm)
	add(a.QKNormWhole, FeatQKNormWhole)
	add(a.NormPlacement == NormParallel, FeatParallelBlock)
	// Per-layer NoPE (Cohere2 global layers skip RoPE). A backend that ropes every
	// layer would corrupt the NoPE layers, so it is a distinct implemented-or-decline
	// capability. (Not the same as FeatPerLayerRoPE, which is differing rope TABLES.)
	add(a.layerNoPE != nil, FeatNoPE)
	// Mean-subtracting LayerNorm (Cohere/GPT-2) is a distinct kernel from RMSNorm;
	// no resident backend implements it, so carriers decline to CPU. GPT-2 already
	// carried the other GPT-2 declines (learned-pos, non-gated, out-bias); this
	// just makes the norm itself explicit.
	add(a.Norm != NormRMS, FeatLayerNorm)
	// The MLP ACTIVATION, not just the gate's presence. A backend whose glue kernel hardcodes
	// SwiGLU would run Gemma's gated GELU-tanh block silently wrong — the exact failure this
	// taxonomy exists to catch, and the one it missed until CUDA went looking at Gemma. Scoped to
	// GATED archs on purpose: a non-gated arch's activation (GPT-2 gelu, Nemotron relu²) is
	// already implied by FeatNonGatedMLP, whose kernel is written for that family's activation.
	//
	// Tested against SiLU rather than for GeluTanh deliberately: ActGeluTanh is ActKind's ZERO
	// value, so an arch that forgets to set Act reads as GELU and lands here — declined, CPU
	// fallback, correct-but-slow. That is the safe direction; matching on GeluTanh would instead
	// let a forgotten Act sail through onto a SwiGLU kernel.
	add(!a.NonGatedMLP && a.Act != ActSiLU, FeatGatedGELU)
	add(a.NonGatedMLP, FeatNonGatedMLP)
	add(a.LearnedPosEmbed, FeatLearnedPos)
	add(a.OutBias, FeatOutBias)
	add(a.LogitScale != 0 && a.LogitScale != 1, FeatLogitScale)
	add(a.MoE != nil, FeatMoE)
	// A sigmoid-GATED always-on shared expert (Qwen2-MoE: out += sigmoid(SharedGate·h)·shared(h))
	// is a distinct kernel from the ungated add (GLM/DeepSeek: out += shared(h)). CUDA implements
	// only the ungated combine and DECLINES the gated one (cuda/backend.go); Metal and WebGPU
	// implement both. Splitting it out moves that decline from a hand-coded backend check into the
	// shared taxonomy, so the hardware matrix matches admission.
	add(a.MoE != nil && a.MoE.SharedIntermediateDim > 0 && !a.MoE.SharedUngated, FeatMoEGatedShared)
	add(a.mla != nil, FeatMLA)
	add(a.granite != nil || a.nemotron != nil, FeatSSM)
	add(a.qwen35 != nil, FeatDeltaNet)
	add(a.lfm2 != nil, FeatShortConv)
	add(a.gptoss != nil, FeatAttnSink)
	// Laguna's attention output gate AND its per-layer query-head count. Both live on
	// arch.laguna and neither has a resident implementation; either one alone would be
	// silently skipped by a resident runner, so the whole family is CPU-only for now.
	add(a.laguna != nil, FeatAttnOutputGate)
	// Gemma-4 E-model (E2B/E4B) shape: PLE, cross-layer shared-KV, variable per-layer FFN — all
	// co-present and NONE ported to the resident bridges (built/validated on the PLE-free dense 12B and
	// 26B-A4B). Without this, an E-model needs no feature CUDA lacks ⇒ admitted-but-mis-run (the PLE
	// branch silently skipped). PLE alone catches every real E-model; the shared-KV/FFN disjuncts make
	// the decline complete against a hypothetical PLE-less E-variant.
	add(a.gemma4 != nil && (a.gemma4.HiddenSizePerLayerInput > 0 || a.gemma4.SharedKVLayers > 0 || len(a.gemma4.FFNPerLayer) > 0), FeatGemma4EModel)
	slices.Sort(f)
	return f
}

// ropeUniform reports whether every layer shares one inv-freq table and mscale. False ⇒ the
// backend must dispatch RoPE per layer type (Mellum's YaRN-on-global vs default-local).
func (a *Architecture) ropeUniform() bool {
	if a.NumLayers <= 1 {
		return true
	}
	base := a.ropeInvFreq(0)
	m0 := a.ropeMscale(0)
	for i := 1; i < a.NumLayers; i++ {
		if a.ropeMscale(i) != m0 {
			return false
		}
		inv := a.ropeInvFreq(i)
		if len(inv) != len(base) {
			return false
		}
		for j := range inv {
			if inv[j] != base[j] {
				return false
			}
		}
	}
	return true
}

// RequiredResidentFeatures returns the features a resident backend must implement to run this
// model correctly. Derived from the arch — see ResidentFeature.
func (m *Model) RequiredResidentFeatures() []ResidentFeature { return m.w.arch.residentFeatures() }

// MissingResidentFeatures returns the required features that `implemented` does not contain,
// sorted. Empty ⇒ a backend declaring that set can run m correctly, and admission is granted.
// Backends call this instead of hand-rolling their own decline list, so the taxonomy has one
// source of truth (three copies is how the bug recurs).
func (m *Model) MissingResidentFeatures(implemented map[ResidentFeature]bool) []ResidentFeature {
	return missingFeatures(m.w.arch.residentFeatures(), implemented)
}

func missingFeatures(required []ResidentFeature, implemented map[ResidentFeature]bool) []ResidentFeature {
	var missing []ResidentFeature
	for _, r := range required {
		if !implemented[r] {
			missing = append(missing, r)
		}
	}
	return missing
}

// ResidentEligible reports whether `backend` can run architecture `a` on its resident (GPU)
// decode path — the CAPABILITY predicate, model-free (arch flags only). It is exactly the two
// gates every resident admission already applies, composed in one place: the arch is a shape the
// runner supports (decodeRunnerEligible) AND the backend implements every feature the arch needs
// (the shared taxonomy). The runtime and the hardware-matrix generator both derive from these
// same pieces, so the published table can never disagree with what a backend can run.
//
// Scope note (deliberate, matches capability_matrix's GPUResident = decodeRunnerEligible): this is
// arch-level CAPABILITY, not a runtime admission. The runtime additionally applies load-time
// POLICY that a model-free predicate cannot know — the Nemotron int4-only / GOINFER_SSM_RESIDENT
// precision gate (Model.DecodeRunnerEligible). Those are precision choices, not "can this backend
// run this family", so the matrix shows capability and footnotes the policy.
func ResidentEligible(a *Architecture, backend string) bool {
	impl, ok := residentBackendFeatures[backend]
	if !ok {
		return false
	}
	return a.decodeRunnerEligible() &&
		len(missingFeatures(a.residentFeatures(), impl)) == 0 &&
		residentMoECapacityOK(a, backend)
}

// residentBackendMoECap is the router-kernel capacity of each backend whose MoE scoreboard is a
// FIXED-SIZE array — the numeric twin of ResidentBackendFeatures (a feature is "implemented at
// all"; a cap is "implemented up to N"). gpu/moe.go scores into array<f32,256> and its group-limited
// path into array<f32,32>/array<bool,32>; cuda/backend.go rejects >256 identically. A model past the
// cap would route on only the first N experts (or index groups out of bounds) — plausible-looking
// WRONG output, no error — so it must decline to the staged/CPU path. This is exactly the runtime
// guard M22 added in gpu/backend.go BuildResident; declared here too so the hardware-matrix
// generator derives the same answer the runtime gives (the C6 one-source-of-truth discipline — a
// feature-only predicate silently over-admitted Kimi K2's 384 experts as WebGPU-resident). Absent
// entry = no fixed-size router cap (a backend that declines these archs on features never reaches
// this — Metal/CUDA decline Kimi on FeatMLA).
var residentBackendMoECap = map[string]struct{ experts, groups int }{
	"webgpu": {experts: 512, groups: 32}, // gpu/moe.go: MAXE 512, array<f32,512> score/sel / array<f32,32> gscore
	"cuda":   {experts: 512, groups: 64}, // cuda/moe.cu: MOE_MAX_E 512 / MOE_MAX_G 64. 256→512 raised deliberately (see below); groups was 32→64 by audit M-17
	"metal":  {experts: 256, groups: 64}, // metal/moe.go: float score[256]/sel[256], gscore[64]/keep[64]; guarded at build (moe.go:206-211)
}

// WHY cuda is 512 and the others are not — this is three shader constants plus this map, and only
// one of them is expensive to change:
//
//   - cuda   moe.cu MOE_MAX_E, raised 256→512. The constant bounds moe_route's per-thread scratch
//            (score[]/sel[]), and moe_route lives in the AUDITED 12.6.85 cuda/testdata/moe.ptx — so
//            raising it required regenerating that artifact. Done at a PINNED, IDENTICAL toolchain
//            with a byte-identical rebuild-unchanged control first; the resulting diff touches only
//            moe_route's stack depot (2368→4416 B) and every other kernel in the file is
//            byte-identical. Procedure: cuda/testdata/REGEN.md. 512 covers Kimi-K2's 384 routed
//            experts (its whole point) and DeepSeek-V4-Pro's 384; it deliberately stops short of
//            Kimi-K3's 896, which is an unbuilt family that should not set validated limits.
//   - metal  DECLARED here for the first time, NOT raised. Its shader really is capped at 256
//            (metal/moe.go:48-49) and rejects above it, so the old "absent entry = no fixed-size
//            router cap" claim was FALSE for metal and the hardware-matrix generator derived a
//            different answer than the runtime gives — exactly the C6 one-source-of-truth defect
//            this map exists to prevent. Raising metal's shader needs Mac validation: future leg.
//   - webgpu ALSO raised 256→512, and this is the one that actually unblocks Kimi-K2: webgpu is the
//            only backend declaring FeatMLA, so on cuda/metal K2 declines on FEATURES no matter what
//            this cap says. Its WGSL compiles at runtime (no frozen artifact), and it was validated
//            on this box. Groups stay 32 — array<f32,32> gscore is untouched and no target needs more.

// ResidentBackendMoECap returns backend's declared router-kernel capacity. ok is false for a
// backend with no fixed-size router.
//
// Exported for the BACKENDS to read (M-31). gpu/residency.go had its own hardcoded 256/32 while
// this map said 512, so ResidentEligible admitted a 384-expert Kimi-K2 or DeepSeek-V4-Pro — "✅
// resident" in both generated matrices — and BuildResident then declined it to CPU with a
// message naming 256, or refused to start under -require-be webgpu. Two pin tests were green
// throughout: one greps gpu/moe.go, the other asserts ResidentEligible; neither reads
// residency.go. That is exactly the drift this map exists to prevent, happening one file over.
func ResidentBackendMoECap(backend string) (experts, groups int, ok bool) {
	c, ok := residentBackendMoECap[backend]
	return c.experts, c.groups, ok
}

// residentMoECapacityOK reports whether backend's router kernel can route arch's MoE (M22). True for
// a dense arch or a backend without a fixed-size router.
func residentMoECapacityOK(a *Architecture, backend string) bool {
	cap, ok := residentBackendMoECap[backend]
	if !ok || a.MoE == nil {
		return true
	}
	if cap.experts > 0 && a.MoE.NumExperts > cap.experts {
		return false
	}
	if cap.groups > 0 && a.MoE.NGroup > cap.groups {
		return false
	}
	return true
}

// ResidentBackendFeatures returns a COPY of the feature set a resident backend implements
// (nil if the backend is unknown). Returning a copy keeps the source map read-only from
// outside the package: an external caller or third-party init() cannot add a feature claim
// its kernels don't implement — precisely the silent-wrong-output failure the registry exists
// to prevent — nor trigger a fatal concurrent map write during a Load (audit B-09). Callers
// look up one backend by name; the package's own admission path reads the unexported map.
func ResidentBackendFeatures(backend string) map[ResidentFeature]bool {
	src := residentBackendFeatures[backend]
	if src == nil {
		return nil
	}
	out := make(map[ResidentFeature]bool, len(src))
	maps.Copy(out, src)
	return out
}

// residentBackendFeatures declares what each resident backend's decode path implements.
//
// These live HERE, not in the backends, for two reasons. First, one source of truth: three
// hand-maintained copies of this logic is precisely how the silent-wrong-output bug recurs
// (Metal had it; CUDA had it; the audit found them independently). Second, testability — the
// backends are build-tagged (`-tags cuda`, `-tags gpu`, darwin-only metal), so a test that
// could see their sets could not run in CI. Declared here, the registry-driven admission gate
// (features_test.go) checks every (arch × backend) pair with no GPU present.
//
// A backend adds an entry ONLY when it ships the kernel that implements it. Overclaiming here
// is exactly the lie the gate exists to catch.
var residentBackendFeatures = map[string]map[ResidentFeature]bool{
	// cgo-free CUDA (cuda/): the dense Qwen2/Llama block, plus QK-norm, sliding window, the
	// Gemma set ((1+w) RMS, sandwich norms, GeGLU, embed scale, per-layer RoPE base), partial
	// rotary, and MoE (routed + ungated shared expert). Still NOT implemented: per-layer rotary
	// WIDTH (only per-layer base); no YaRN mscale; no logit softcap; no MLA/SSM.
	//
	// TRAP — before adding FeatMLA here (or otherwise making a GROUP-ROUTED family
	// CUDA-admissible), verify the nGroup/topkGroup mapping at cuda/resident.go's moe_route
	// launch END TO END. DeepSeek and Kimi are the only families with nGroup != topkGroup, and
	// they decline on this line today, which is the sole reason that mapping has never run.
	// Every admissible MoE model has nGroup == topkGroup, so a transposition there is a no-op
	// and passes every gate in the repo (verified by transposing it). Adding MLA arms it: the
	// first group-routed forward would route through the wrong groups, and NOTHING would go red
	// — expert selection is discrete, so the output is unrelated rather than slightly off, the
	// exact class the Granite SSM investigation cost 66% agreement to. Gate the mapping first.
	//
	// FeatMoE covers the ROUTED block (router + stacked experts + every routing flavour the
	// route kernel handles) AND the always-on UNGATED shared expert (GLM/DeepSeek). The GATED
	// shared expert (Qwen-MoE's sigmoid(SharedGate·h) scaling) is NOT wired — BuildResident
	// declines it at load, since no committed fixture gates it end to end and FeatMoE is one flag
	// that cannot express the sub-shape. That decline is the honest "admitted, but this variant
	// is not wired" in a table whose whole job is to not lie.
	//
	// FeatPartialRotary and the shared expert land together on purpose: every partial-rotary arch
	// (glm4_moe) also has a shared expert, so neither is independently reachable — glm-tiny is the
	// joint end-to-end gate (TestGLMResidentParity), and declaring partial rotary before the
	// shared expert existed would have admitted glm onto a path no model could exercise.
	"cuda": {
		FeatQKNorm:            true, // qk_norm kernel — per-head Q/K RMSNorm before RoPE (Qwen3)
		FeatSlidingWindow:     true, // attention `window` uniform, per-layer via LayerIsLocalResident
		FeatPartialRotary:     true, // rope_kv rhalf = rotaryDim/2 + un-rotated tail cached (GLM/Phi)
		FeatRMSAddOne:         true, // (1+w) offset, threaded through rmsnorm_quant/fused_rms_*/qk_norm
		FeatSandwichNorm:      true, // rmsnorm_f32 on each sublayer output (breaks the accum epilogue)
		FeatGatedGELU:         true, // glu_quant `act` — GeGLU as well as SwiGLU
		FeatEmbedScale:        true, // √hidden applied host-side in embedResident
		FeatFinalLogitSoftcap: true, // softcap·tanh(logits/softcap) host-side after readback (finalSoftcap)
		FeatPerLayerRoPE:      true, // per-layer invFreq buffer (Gemma local 10k vs global 1M base)
		FeatMoE:               true, // moe_route + indexed stacked experts + ungated shared expert
		// The SIGMOID-GATED always-on shared expert (Qwen-MoE): out += sigmoid(SharedGate·h)·shared(h).
		// Declared 2026-08-20. The kernel was always here — moe.cu's shared_gate_combine has an
		// `ungated` flag and its comment names the gated case "Qwen-MoE" — so what this backend
		// actually lacked was the [1,hidden] gate weight in the build, not any device code. Worth
		// recording: the feature table said "CUDA implements only the ungated combine", which was
		// true of the WIRING and false of the kernel, and nothing reconciled the two.
		//
		// GATED BY qwen3_5_moe-tiny (cuda.TestQwen35ResidentParityCUDA), whose MoE block IS
		// Qwen2-MoE's — transformers derives Qwen3_5MoeSparseMoeBlock from it, shared_expert_gate
		// included. So declaring this ALSO admits qwen2_moe on cuda as a documented side effect,
		// the same shape as Metal's FeatRopeMscale/Mellum note above: no qwen2_moe fixture exists
		// in this tree, so that family's CUDA admission rests on the inheritance, not on its own
		// end-to-end run. Add one if that ever stops being good enough.
		FeatMoEGatedShared: true,
		// Gated-DeltaNet: the deltanet.ptx mixer (conv ring + delta rule + gated norm) plus the
		// family's fused double-width q_proj and sigmoid output gate. Declared 2026-08-20 with
		// the end-to-end gate, not ahead of it — the same discipline the GPT-2/gpt-oss entries
		// record. This admits the DENSE sibling only: qwen3_5_moe and qwen3_next additionally
		// need FeatMoEGatedShared, which CUDA still does not implement.
		FeatDeltaNet: true,
		// FeatRopeMscale: YaRN's attention_factor, folded into cos/sin by rope / rope_kv /
		// rope_kv_batched (cuda/glue.cu, gemv_fwd.cu, prefill_batched.cu) and threaded per LAYER
		// from Model.RopeMscaleLayer via cudaLayer.mscale. Proven in isolation first by
		// TestRopeMscale (scale=1 reproduces the unscaled rotation to 8.9e-08; scale=0.85 matches
		// the scaled reference AND is provably different from unscaled), then end-to-end on real
		// weights by TestMellumResidentParityCUDA — in that order, because the kernels took NO
		// scale parameter at all until 2026-08-31 and declaring this on them would have admitted
		// families onto a path that silently drops the factor.
		//
		// DECLARING THIS ADMITS MELLUM, which is not a side effect but the point of validating it
		// first: mellumArchitecture needs exactly {FeatMoE, FeatPerLayerRoPE, FeatQKNorm,
		// FeatRopeMscale, FeatSlidingWindow} and CUDA already declared the other four, so this
		// single flag is the whole admission. Metal hit the identical coupling (G10) and resolved
		// it by an explicit call because no Mellum checkpoint was reachable there; here one is, so
		// it was measured instead.
		FeatRopeMscale: true,
		// gpt-oss's two remaining departures, declared TOGETHER on 2026-08-31 because the family
		// needs both and neither admits anything on its own:
		//
		//   FeatAttnSink  the learned per-head softmax sink, the clamped interleaved-SwiGLU
		//                 expert, and the router whose bias reaches the selection WEIGHT. Kernels
		//                 in cuda/gptoss_act.cu; sinkArg threaded into BOTH attention launches
		//                 (decode + prefill) and launchGluSplitExpert dispatched from the MoE
		//                 expert loop.
		//   FeatOutBias   the o_proj bias. NO new kernel was needed: aikit's gemv_quant.cu and
		//                 goinfer's batched gemv_w4a8_rn already fold bias into the value BEFORE
		//                 the accumulate select, so bias-plus-residual is one instruction here.
		//                 It was pure wiring, at four launch sites (two decode, two prefill).
		//
		// DECLARED ONLY AFTER A REAL gpt-oss-20b FORWARD RAN ON THIS PATH — the thing G7 had been
		// blocked on since 2026-08-18, and the reason 2224441's earlier declaration was reverted:
		// kernel-level parity is not end-to-end parity. TestGptOssResidentParityCUDA on the real
		// 20B, resident on an 8 GB card via --moe-cache-experts: 7/8 argmax-exact, min cosine
		// 0.996392. For scale, the same harness measures 0.982 on a 40-layer qwen3.6-35b-a3b and
		// 0.974 on a 24-layer dense 0.5B, so this is at the top of the range, not scraping a bar.
		//
		// Getting there took THREE silent defects, none of which any kernel test could see,
		// because each was a term the wiring dropped rather than a kernel computing it wrongly:
		//   d9829ce  the gate‖up bias table indexed by SLOT id under expert caching
		//   610ce7f  the per-expert DOWN bias never applied at all (0.750 -> 0.9993 on the tiny)
		//   this     route_gptoss never LOADED, so the router fell back to moe.cu's moe_route,
		//            which takes the mixing weight from the UNBIASED score. Same experts
		//            selected, different weights (0.895 -> 0.9964 on the real 20B).
		FeatAttnSink: true,
		FeatOutBias:  true,
	},

	// WebGPU (gpu/): the richest runner — the levers in docs/gpu-residency-coverage.md.
	"webgpu": {
		FeatQKNorm:         true, // C1  per-head QK-norm before RoPE
		FeatPartialRotary:  true, // C5  rotary_dim < head_dim
		FeatSlidingWindow:  true, // C6  per-layer windowed start
		FeatPerLayerRoPE:   true, // C7  differing invFreq per layer type
		FeatRopeMscale:     true, // C7  YaRN attention_factor
		FeatMoE:            true, // C3a-d router / stacked experts / shared expert
		FeatMoEGatedShared: true, // sharedGatedCombine — sigmoid-gated shared expert (gpu/moe.go)
		FeatMLA:            true, // C4a-d latent-KV attention
		FeatSSM:            true, // Mamba-2 engine (Granite-4.0-H, Nemotron-H)
		FeatDeltaNet:       true, // Gated-DeltaNet engine + fused attn output gate (gpu/deltanet.go)
		FeatNonGatedMLP:    true, // relu2Quant (Nemotron-H squared-ReLU)
		FeatLogitScale:     true, // Granite logits_scaling
		FeatRMSAddOne:      true, // (1+w) RMS offset
	},

	// cgo-free Metal (metal/): dense Qwen2/Llama plus qk-norm, sliding-window, partial-rotary,
	// MoE (router + stacked experts + shared expert; metal/moe.go), the full Gemma set —
	// sandwich norms, GeGLU, (1+w) RMS, √hidden embed scale, per-layer RoPE base — GPT-2's
	// LayerNorm/non-gated-MLP/learned-pos/out-bias (2026-08-18: layernorm_quant, act_quant,
	// gemv_w4a8_resid_bias/gemv_w4a8_sa_bias_resid, encodeLayer/encodeAttention wiring, the
	// ForwardArgmax V%8!=0 fallback for GPT-2's 50257 vocab — TestGPT2ResidentParity, min cosine
	// 0.999) — and gpt-oss's attention sink + clamped-SwiGLU MoE + custom router (2026-08-18:
	// kernels.go's `attention` sink term, moe.go's route_gptoss/swiglu_quant_gptoss/
	// gemv_w4a8_moe_wacc_bias, the moe.go isGptOss dispatch split, gpt-oss's YaRN rope mscale
	// riding the already-wired-everywhere rope kernel param — TestGptOssResidentParity, 8/8
	// argmax-exact, min cosine 0.9989 on the tiny fixture). Gemma parity was gated on the
	// GELU-tanh overflow fix (glu_act clamp, 38a2b7c): logit cosine 0.818→0.994. Still declines
	// MLA and SSM.
	//
	// FeatRopeMscale is SHARED with Mellum (same required-feature set minus
	// FeatAttnSink/FeatOutBias) — declaring it for gpt-oss's YaRN ALSO admits Mellum on Metal, a
	// path with ZERO end-to-end validation here: no real Mellum checkpoint on this box (~24GB),
	// and a synthetic-random-weight structural test was tried and abandoned as inconclusive (even
	// a plain dense qwen2 with NO QK-norm fails the same cosine bar against fully-random untrained
	// weights at realistic dims — the methodology can't discriminate a real bug from quantization
	// noise on unstructured weights, so it proves nothing either way). Declared anyway (explicit
	// user call, 2026-08-18): Mellum already has a trusted GPU path on WebGPU, this only adds an
	// unvalidated SECOND path on Metal, and the flag is one boolean — trivially reversible if a
	// real Mellum checkpoint later surfaces a problem. If you're chasing a Mellum-on-Metal bug,
	// start here.
	"metal": {
		FeatQKNorm:            true, // qk_norm kernels
		FeatSlidingWindow:     true, // attention window uniform
		FeatPartialRotary:     true, // rope rhalf = rotaryDim/2
		FeatMoE:               true, // moe_route + indexed stacked-expert W4A8 GEMVs + shared expert (metal/moe.go)
		FeatMoEGatedShared:    true, // shared_gate_combine — sigmoid-gated shared expert (metal/moe.go)
		FeatSandwichNorm:      true, // rmsnorm_f32 on each sublayer output (Gemma)
		FeatGatedGELU:         true, // GeGLU — clamped-tanh geglu (glu_act, 38a2b7c)
		FeatRMSAddOne:         true, // (1+w) RMS offset
		FeatEmbedScale:        true, // √hidden embedding multiplier (embedResident)
		FeatPerLayerRoPE:      true, // per-layer invFreq (Gemma local 10k vs global 1M base)
		FeatFinalLogitSoftcap: true, // softcap·tanh(logits/softcap) host-side after readback (metal/model.go finalizeLogits)
		FeatLayerNorm:         true, // layernorm_quant — mean-centered norm+quant (GPT-2, generalized with a hasBias flag for Cohere)
		FeatNonGatedMLP:       true, // act_quant — up→act→down, no gate (GPT-2)
		FeatLearnedPos:        true, // addLearnedPos — host-side wpe[pos] add, RoPE dispatch skipped (GPT-2)
		FeatOutBias:           true, // gemv_w4a8_sa_bias_resid (o-proj) — GPT-2's attention output bias
		FeatRopeMscale:        true, // rope kernel's scale param (kernels.go), proven in isolation (TestRope_mscale) and end-to-end via gpt-oss's YaRN (TestGptOssResidentParity) — ALSO admits Mellum, see note above
		FeatAttnSink:          true, // attention sink term + clamped-SwiGLU MoE + custom router (gpt-oss) — TestGptOssResidentParity
		FeatDeltaNet:          true, // Gated-DeltaNet mixer + fused attn output gate (deltanet.go/deltanet_kernels.go) — TestQwen35ResidentParityMetal
	},
}
