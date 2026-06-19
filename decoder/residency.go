package decoder

import (
	"fmt"
	"math"
	"os"
)

// GPU full-residency decode support. When the backend is webgpu AND the arch is
// DecodeRunner-eligible (dense Qwen2/Llama shape), the per-token forward can run
// entirely on the GPU (the gpu package's DecodeRunner) instead of the per-matmul
// staged path. This file is the decoder-side seam: the interfaces the gpu backend
// implements, the eligibility gate, and (in Generate) the routing. Sampler /
// constrain / Session all stay CPU-side — they consume the logits that come back.
//
// v1 scope (documented limitations):
//   - stateless Model.Generate only. Session prefix-reuse and GenerateSpeculative
//     drive a CPU-resident KVCache that the GPU-resident KV can't transparently
//     share, so those requests fall back to the staged path (serve warm-sessions
//     and GPU residency are mutually exclusive in v1).
//   - prompt prefill runs on the CPU (batched forwardLayersN) and its post-RoPE
//     K/V is uploaded into the GPU caches; only decode runs on the GPU.

// ResidentForward is one token's GPU forward: embedding[hidden] in, logits[vocab]
// out, with the model + KV resident on the device. Implemented by the gpu package.
type ResidentForward interface {
	// Forward runs token-at-pos: returns logits for the embedding at absolute
	// position pos (the runner appends this position's K/V to its resident cache).
	Forward(embedding []float32, pos int) (logits []float32, err error)
	// ForwardN runs K tokens at consecutive positions startPos..startPos+K-1 in ONE
	// command buffer (one Submit/Poll), appending K KV positions and returning K
	// logit rows — the batched verify for speculative decoding. Causal: row i attends
	// to positions [0, startPos+i]. Bit-identical to K sequential Forward calls
	// (TestResidentForwardN_parity), so it amortizes the cgo-encode glue over K
	// without changing numerics. nil/empty embeddings ⇒ no-op.
	ForwardN(embeddings [][]float32, startPos int) (logits [][]float32, err error)
	// UploadKV writes a layer's post-RoPE K and raw V (each [n*kvDim], positions
	// 0..n-1) into the resident GPU caches — the prefill bridge.
	UploadKV(layer int, keys, vals []float32) error
	// TruncateTo drops resident KV positions ≥ pos — the rollback after a partial
	// speculative accept. The resident cache is positional and Forward sets
	// nKeys=pos+1, so stale entries past pos are simply never read and get overwritten
	// next round; this is therefore a no-op (the caller tracks pos), kept for the
	// interface symmetry with the CPU KVCache.
	TruncateTo(pos int)
	// Reset clears resident KV for a fresh generation (positions overwritten).
	Reset()
	// Close releases the resident GPU buffers.
	Close() error
}

// ResidencyBackend is the optional capability a Backend advertises to build a
// ResidentForward from a loaded Model. The webgpu backend implements it; ok is
// false when the arch is not DecodeRunner-eligible.
type ResidencyBackend interface {
	BuildResident(m *Model) (rf ResidentForward, ok bool, err error)
}

// DecodeRunnerEligible reports whether this model's arch is a dense Qwen2/Llama
// shape the GPU DecodeRunner supports: no MoE/Gemma4/qwen3_5 special path, gated
// SwiGLU MLP, pre-2 RMSNorm, full RoPE, standard GQA, no QK-norm / learned-pos /
// sliding-window / logit-softcap / output-bias. q/k/v bias (Qwen2) and the (1+w)
// RMS offset are handled, so they're allowed. Anything else → staged fallback.
func (m *Model) DecodeRunnerEligible() bool {
	return m.w.arch.decodeRunnerEligible()
}

// decodeRunnerEligible is the arch-only predicate behind DecodeRunnerEligible —
// usable directly from a resolved Architecture (e.g. the capability matrix)
// without a loaded Model.
func (a *Architecture) decodeRunnerEligible() bool {
	// Families with their own non-uniform forward (hybrid mixers, Gemma-4, Llama-4,
	// qwen3.5) keep the staged path regardless.
	if a.gemma4 != nil || a.qwen35 != nil || a.llama4 != nil {
		return false
	}
	// Granite-4.0-H resident SSM hybrid (P5b): its own mixer-kind path (Mamba-2 ⊕
	// attention), so it bypasses the standard GQA gates below. Guarded during the
	// parity bring-up (P5b.3/P5b.4); P6 makes this unconditional.
	if a.granite != nil {
		return os.Getenv("GOINFER_SSM_RESIDENT") != ""
	}
	// Nemotron-H resident (dense squared-ReLU hybrid): single-op-per-block Mamba-2 / NoPE-GQA /
	// relu² MLP, reusing the granite SSM engine. Guarded during parity bring-up (P3–P5).
	if a.nemotron != nil {
		return os.Getenv("GOINFER_SSM_RESIDENT") != ""
	}
	if a.MoE != nil && !a.moeResidentEligible() {
		return false
	}
	// FFN / norm / head constraints common to both attention paths.
	if a.NonGatedMLP || a.LearnedPosEmbed || a.OutBias ||
		a.NormPlacement != NormPre2 || a.FinalLogitSoftcap != 0 || a.AttnLogitSoftcap != 0 {
		return false
	}
	// MLA (DeepSeek/Kimi) runs its own latent-attention path on the resident runner
	// (Lever C4 — gpu/mla.go), so the standard GQA/RoPE/QK-norm/sliding-window checks
	// don't apply; its decoupled RoPE rides a separate qk_rope slice, not HeadDim.
	if a.mla != nil {
		return true
	}
	// Standard GQA attention: QK-norm (Qwen3/GLM/Mellum) is handled (Lever C1 — per-head
	// RMSNorm before RoPE), q/k/v bias (Qwen2) and the (1+w) RMS offset too. Partial RoPE
	// (GLM/Phi rotary_dim < HeadDim) is handled (Lever C5), sliding window (Lever C6), and
	// per-layer-type RoPE (Mellum YaRN-on-global vs default-local invFreq + mscale, Lever
	// C7). The remaining gate: the local/global rope tables must be the SAME LENGTH so the
	// runner's single rotaryDim/2 dispatch covers both (a per-layer rotary width — Gemma's
	// global/local head-dim split — is not handled; Gemma is also softcapped out anyway).
	return a.ropeResidentCompatible()
}

// ropeResidentCompatible reports whether the resident runner's per-layer RoPE binding can
// represent this arch. It binds a per-layer invFreq buffer + per-layer cos/sin scale, so
// local/global layers MAY differ in RoPE base or YaRN scaling (Mellum) — but the two
// inverse-freq tables must share a length, since the rope dispatch's rotaryDim/2 is one
// model-level value. Single-rope families and uniform sliding-window Mistral pass trivially.
func (a *Architecture) ropeResidentCompatible() bool {
	return len(a.ropeInvFreqLocal) == len(a.ropeInvFreqGlobal)
}

// moeResidentEligible reports whether this arch's MoE is a shape the GPU resident
// runner handles. The route kernel now covers every routing flavor — softmax/sigmoid,
// optional selection bias, norm_topk_prob, routed scaling, and DeepSeek group-limited
// top-k (NGroup > 1, Lever C4c) — plus the always-on shared expert (C3d) and dense
// prefix layers (FirstKDense > 0), so any int8-stackable MoE qualifies.
func (a *Architecture) moeResidentEligible() bool {
	return a.MoE != nil
}

// MoEResidentParams returns the MoE knobs the GPU resident runner needs (ok=false for
// a dense model). sharedInter > 0 marks an always-on shared expert; sharedUngated picks
// the GLM/DeepSeek (no sigmoid gate) combine; nGroup > 1 is DeepSeek group-limited
// routing (topkGroup groups kept).
func (m *Model) MoEResidentParams() (nE, k, inter, sharedInter int, sigmoid, norm, sharedUngated bool, scale float64, nGroup, topkGroup int, ok bool) {
	mo := m.w.arch.MoE
	if mo == nil {
		return 0, 0, 0, 0, false, false, false, 0, 0, 0, false
	}
	return mo.NumExperts, mo.TopK, mo.IntermediateDim, mo.SharedIntermediateDim,
		mo.RouterSigmoid, mo.NormTopKProb, mo.SharedUngated, mo.RoutedScale, mo.NGroup, mo.TopkGroup, true
}

// MLAResidentParams returns the DeepSeek/Kimi MLA geometry the GPU resident runner
// needs (ok=false when the arch is not MLA). attnScale is the resolved q·k softmax
// multiplier (qk_head_dim^-0.5, with any YaRN mscale²); ropeScale is the YaRN
// attention factor folded into the decoupled RoPE cos/sin (1.0 when none).
func (m *Model) MLAResidentParams() (qLoRA, kvLoRA, qkNope, qkRope, vHead int, interleave bool, attnScale, ropeScale float64, ok bool) {
	p := m.w.arch.mla
	if p == nil {
		return 0, 0, 0, 0, 0, false, 0, 0, false
	}
	return p.QLoRARank, p.KVLoRARank, p.QKNopeHeadDim, p.QKRopeHeadDim, p.VHeadDim,
		p.ropeInterleave, m.w.arch.AttnScale, m.w.arch.ropeMscale(0), true
}

// MLALayerWeights returns layer l's MLA projection weights (f32) for the resident
// bridge: the q-LoRA bottleneck (qA/qANorm/qB) or direct qProj, the KV down-proj
// (kvA) + its latent norm (kvANorm), the per-head kv up-proj kvB (sliced into W_UK /
// W_UV on the GPU side), and the output proj. nil qA ⇒ the direct qProj (V2-Lite).
func (m *Model) MLALayerWeights(l int) (qA, qANorm, qB, qProj, kvA, kvANorm, kvB, oProj []float32) {
	w := m.w.Layers[l].mla
	return w.qAProj, w.qALayernorm, w.qBProj, w.qProj, w.kvAProj, w.kvALayernorm, w.kvBProj, w.oProj
}

// SlidingWindowResident returns the sliding-window size for the resident runner (0 when
// the model is full-attention). Local (windowed) layers attend only the last `window`
// positions; the runner computes the per-token window start (Lever C6).
func (m *Model) SlidingWindowResident() int { return m.w.arch.SlidingWindow }

// LayerIsLocalResident reports whether layer i is a sliding-window (local) layer — i.e. a
// window is set AND the arch marks the layer non-global. Mistral is all-local; Mellum
// interleaves (3:1 sliding/full).
func (m *Model) LayerIsLocalResident(i int) bool {
	return m.w.arch.SlidingWindow > 0 && !m.w.arch.isGlobalLayer(i)
}

// LayerRopeGlobal reports whether layer i uses the global RoPE table (vs the local one) —
// the per-layer-type RoPE selector (Lever C7). For single-rope models this is always true.
func (m *Model) LayerRopeGlobal(i int) bool { return m.w.arch.isGlobalLayer(i) }

// RopeInvFreqLayer returns layer i's RoPE inverse-frequency table as float32 — the global
// or local table per the layer's attention type (Mellum YaRN-on-global vs default-local).
func (m *Model) RopeInvFreqLayer(i int) []float32 {
	inv := m.w.arch.ropeInvFreq(i)
	out := make([]float32, len(inv))
	for j, v := range inv {
		out[j] = float32(v)
	}
	return out
}

// RopeMscaleLayer returns layer i's RoPE cos/sin scale (YaRN attention_factor; 1.0 for
// non-YaRN layers) — folded into the resident rope kernel per layer.
func (m *Model) RopeMscaleLayer(i int) float64 { return m.w.arch.ropeMscale(i) }

// RopeInvFreq returns the rotary inverse frequencies (layer 0; uniform across an
// eligible model's dense layers) as float32 — the GPU bridge's RoPE table.
func (m *Model) RopeInvFreq() []float32 {
	inv := m.w.arch.ropeInvFreq(0)
	out := make([]float32, len(inv))
	for i, v := range inv {
		out[i] = float32(v)
	}
	return out
}

// AttnScale is the attention softmax scale (1/√headDim for eligible archs).
func (m *Model) AttnScale() float32 {
	return float32(1.0 / math.Sqrt(float64(m.w.arch.HeadDim)))
}

// RMSAddOne reports the Gemma (1+w) RMSNorm offset (false for Qwen/Llama).
func (m *Model) RMSAddOne() bool { return m.w.arch.RMSAddOne }

// Dims returns the model shape from the resolved Architecture — authoritative,
// unlike Config() (Cfg), which a GGUF-loaded or .giw model may leave partly
// zero. The forward pass reads these.
func (m *Model) Dims() (hidden, nLayers, nH, nKV, hd, inter, vocab int) {
	a := m.w.arch
	return a.HiddenDim, a.NumLayers, a.NumHeads, a.NumKVHeads, a.HeadDim, a.IntermediateDim, a.VocabSize
}

// NormEps is the RMSNorm epsilon (arch-backed).
func (m *Model) NormEps() float32 {
	if e := float32(m.w.arch.NormEps); e != 0 {
		return e
	}
	return 1e-6
}

// withResidency builds the GPU resident decoder if the backend supports it and
// the arch is eligible, then returns m. A no-op for the CPU backend / ineligible
// archs (m.resident stays nil → staged/CPU path). Called at every model-load site.
func (m *Model) withResidency() *Model {
	if os.Getenv("GOINFER_NO_RESIDENCY") != "" {
		return m // force the per-matmul staged path (decision-matrix measurement)
	}
	rb, ok := m.be.(ResidencyBackend)
	if !ok || !m.DecodeRunnerEligible() {
		return m
	}
	rf, ok, err := rb.BuildResident(m)
	if err != nil || !ok {
		if os.Getenv("GOINFER_RESIDENT_DEBUG") != "" {
			fmt.Fprintf(os.Stderr, "[resident-debug] BuildResident ok=%v err=%v\n", ok, err)
		}
		return m // build refused/failed → silently fall back
	}
	m.resident = rf
	return m
}

// --- Granite-4.0-H resident SSM bridge (P5b: resident Mamba decode, docs/ssm-residency-scope.md) ---
// These expose the hybrid's per-layer mixer-kind + Mamba-2 mixer weights + scalar multipliers
// to the gpu resident builder. Additive accessors; the eligibility flip (P6) is separate and
// still gated, so production routing is unchanged until then.

// GraniteResidentParams returns the Mamba-2 geometry + the four Granite scalar multipliers the
// resident SSM path needs (ok=false for non-granite). logitScale divides the final logits;
// attnScale is the attention q·k multiplier (folded into the resident attention scale). Granite
// gated-RMSNorm is over the full dInner (NormGroups=1).
func (m *Model) GraniteResidentParams() (nHeads, headDim, dState, nGroups, dConv int, embMul, residMul, logitScale, attnScale float32, ok bool) {
	g := m.w.arch.granite
	if g == nil {
		return 0, 0, 0, 0, 0, 0, 0, 0, 0, false
	}
	ls := float32(1)
	if m.w.arch.LogitScale != 0 {
		ls = float32(m.w.arch.LogitScale)
	}
	em, rm := g.EmbMul, g.ResidMul
	if os.Getenv("GOINFER_SSM_NOMUL") != "" { // isolation seam: identity multipliers on both sides
		em, rm, ls = 1, 1, 1
	}
	return g.NHeads, g.HeadDim, g.DState, g.NGroups, g.DConv, em, rm, ls, float32(m.w.arch.AttnScale), true
}

// GraniteMambaLayer reports whether layer i is a Mamba-2 mixer (vs GQA attention).
func (m *Model) GraniteMambaLayer(i int) bool { return m.w.arch.isMambaLayer(i) }

// NemotronResidentParams returns the resident geometry for Nemotron-H (the dense squared-ReLU
// hybrid: Mamba-2 / NoPE-GQA / non-gated relu² MLP, single-op-per-block, no multipliers, no MoE).
// attnScale is the GQA q·k multiplier (1/√head_dim). normGroups is the Mamba gated-RMSNorm group
// count — Nemotron normalizes PER GROUP (NGroups), unlike Granite's full-dInner (1). ok=false for
// non-nemotron archs.
func (m *Model) NemotronResidentParams() (nHeads, headDim, dState, nGroups, dConv, normGroups int, attnScale float32, ok bool) {
	np := m.w.arch.nemotron
	if np == nil {
		return 0, 0, 0, 0, 0, 0, 0, false
	}
	return np.NHeads, np.HeadDim, np.DState, np.NGroups, np.DConv, np.NGroups, float32(m.w.arch.AttnScale), true
}

// NemotronBlockKind returns layer i's single-op kind: 0 mamba · 1 attention · 2 mlp (the
// arch's nemoMamba/nemoAttn/nemoMLP). Mamba weights reuse GraniteMambaWeights (model-agnostic).
func (m *Model) NemotronBlockKind(i int) uint8 { return m.w.arch.nemotron.blockKind[i] }

// GraniteMambaWeights returns layer i's f32 Mamba-2 mixer tensors (nil if the layer is an
// attention layer). The resident build quantizes inProj/outProj to W8A8 and uploads
// convW/convB/aLog/d/dtBias/normW as f32; aLog is folded to Aexp=-exp(aLog) at build.
func (m *Model) GraniteMambaWeights(i int) (inProj, convW, convB, aLog, dW, dtBias, normW, outProj []float32) {
	w := m.w.Layers[i].mamba
	if w == nil {
		return
	}
	return w.inProj, w.convW, w.convB, w.aLog, w.d, w.dtBias, w.normW, w.outProj
}

// ResidentActive reports whether the GPU full-residency decode path is built and
// will run for a plain stateless Generate (webgpu backend + eligible arch).
func (m *Model) ResidentActive() bool { return m.resident != nil }

// ResidentForwardForTest exposes the resident forward (nil when not resident) for the
// GPU ForwardN-vs-Forward parity gate. Test-only seam; production code routes through
// Generate / GenerateSpeculative, not this.
func (m *Model) ResidentForwardForTest() ResidentForward { return m.resident }

// ForwardForTest / EmbedResidentForTest are CPU-reference seams for the gpu-package
// resident parity gates (which can't be in package decoder — import cycle via gpu).
// ForwardForTest is the CPU per-token logits; EmbedResidentForTest is the resident
// input embedding (incl. Granite EmbMul). Test-only.
func (m *Model) ForwardForTest(id int, cache *KVCache) ([]float32, error) {
	return m.forward(id, cache)
}
func (m *Model) EmbedResidentForTest(id int) []float32 { return m.embedResident(id) }

// embedResident returns the raw input embedding [hidden] for token id — the CPU
// half of the residency forward (eligible archs have no embedding scale).
func (m *Model) embedResident(id int) []float32 {
	h := make([]float32, m.w.arch.HiddenDim)
	m.w.Embed.Row(id, h)
	// Granite embedding_multiplier — the resident residual stream starts at emb·EmbMul
	// (P5b; the only Granite scalar applied outside the resident runner / weight folds).
	if g := m.w.arch.granite; g != nil && g.EmbMul != 0 && g.EmbMul != 1 {
		for i := range h {
			h[i] *= g.EmbMul
		}
	}
	return h
}
