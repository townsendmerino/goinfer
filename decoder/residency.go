package decoder

import (
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
	// UploadKV writes a layer's post-RoPE K and raw V (each [n*kvDim], positions
	// 0..n-1) into the resident GPU caches — the prefill bridge.
	UploadKV(layer int, keys, vals []float32) error
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
	return a.MoE == nil && a.gemma4 == nil && a.qwen35 == nil &&
		!a.NonGatedMLP && !a.QKNorm && !a.LearnedPosEmbed && !a.OutBias &&
		a.NormPlacement == NormPre2 && a.SlidingWindow == 0 &&
		a.FinalLogitSoftcap == 0 && a.AttnLogitSoftcap == 0 &&
		(a.RotaryDim == 0 || a.RotaryDim == a.HeadDim)
}

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
		return m // build refused/failed → silently fall back
	}
	m.resident = rf
	return m
}

// ResidentActive reports whether the GPU full-residency decode path is built and
// will run for a plain stateless Generate (webgpu backend + eligible arch).
func (m *Model) ResidentActive() bool { return m.resident != nil }

// embedResident returns the raw input embedding [hidden] for token id — the CPU
// half of the residency forward (eligible archs have no embedding scale).
func (m *Model) embedResident(id int) []float32 {
	h := make([]float32, m.w.arch.HiddenDim)
	m.w.Embed.Row(id, h)
	return h
}
