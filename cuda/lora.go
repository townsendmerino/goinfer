//go:build cuda

package cuda

import (
	"fmt"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/goinfer/decoder"
)

// Compute-time LoRA on the resident path (G3, docs/task-gpu-paths-2026-09.md).
//
// Architecturally this backend is like Metal, not WebGPU: launchToken re-issues live kernel
// launches every token (segA/segB/segBFFN), so binding/clearing an adapter is just a Go-side
// `if r.loraLayers != nil` at each of the 7 projection sites — no fixed dispatch-plan surgery
// needed (contrast gpu/lora_resident.go's rebuildSteps, forced by WebGPU's fixed-plan-replay
// architecture). The one thing THIS backend has that neither of the others does is CUDA graphs
// (r.graphs, opt-in via GOINFER_CUDA_GRAPHS, off by default): a captured graph replays the exact
// launches recorded at capture time, which happened with r.loraLayers == nil, so a graph replay
// would silently skip every LoRA dispatch — SetAdapter refuses rather than bind under graphs (see
// resident.go's r.loraLayers field comment).
//
// The delta itself, and its two-kernel down/up split, mirror metal/lora.go and gpu/lora.go
// exactly: y[o] += scale·Σ_r B[o,r]·(A·x)[r], applied AFTER the base projection's matmul on the
// SAME quantized activation the base projection consumed.

// loraRMax bounds the LoRA rank this backend accepts — generous against real adapters
// (typically rank 4-64, rarely above 128). SetAdapter declines (does not silently truncate) a
// projection whose rank exceeds it.
const loraRMax = 256

// cudaLoraProj is one projection's bound delta: the device-resident A[R,In]/B[Out,R] buffers
// (row-major f32, PEFT's own layout) plus the Go-side scalars applyLora needs to size its two
// launches. Built fresh per SetAdapter call.
type cudaLoraProj struct {
	a, b  Buffer
	rank  int
	outn  int
	scale float32
}

// cudaLoraLayer is one transformer layer's per-projection bound deltas; a nil field is a
// projection this adapter does not target, matching decoder.ResidentAdapterLayer's own
// nil-means-untargeted convention exactly.
type cudaLoraLayer struct {
	q, k, v, o, gate, up, down *cudaLoraProj
}

func projForCUDA(l *decoder.ResidentAdapterLayer) [7]*decoder.ResidentAdapterProj {
	return [7]*decoder.ResidentAdapterProj{l.Q, l.K, l.V, l.O, l.Gate, l.Up, l.Down}
}

// releaseLoraLayers frees every device buffer a previously-bound adapter allocated. Unlike
// Metal/WebGPU, this backend's Device ledger normally frees every allocation ONLY at Close
// (ReleaseObjects) — r.dev.ReleaseBuf exists precisely for a dynamic, per-call allocation like
// this one, and skipping it here would leak VRAM on every rebind for the resident's whole life.
func (r *cudaResident) releaseLoraLayers() {
	for _, l := range r.loraLayers {
		for _, p := range []*cudaLoraProj{l.q, l.k, l.v, l.o, l.gate, l.up, l.down} {
			if p == nil {
				continue
			}
			r.dev.ReleaseBuf(p.a)
			r.dev.ReleaseBuf(p.b)
		}
	}
}

// SetAdapter implements decoder.ResidentAdapter: uploads (layers != nil) or clears (layers ==
// nil) the compute-time LoRA delta for every subsequent Forward call, until the next
// SetAdapter. Runs on the executor thread (r.do) like every other device-touching call in this
// file — CUDA contexts are thread-affine, and BuildResident's own setup follows the same rule.
func (r *cudaResident) SetAdapter(layers []decoder.ResidentAdapterLayer) error {
	return r.do(func() error {
		r.releaseLoraLayers()
		r.loraLayers = nil
		if layers == nil {
			return nil
		}
		if r.graphs {
			return fmt.Errorf("cuda: SetAdapter declined — this resident was built with CUDA " +
				"graphs enabled (GOINFER_CUDA_GRAPHS); a captured graph replays the exact launches " +
				"recorded without an adapter bound, so binding one here would silently apply no " +
				"delta. Disable GOINFER_CUDA_GRAPHS to use compute-time LoRA on this backend")
		}
		if len(layers) != len(r.layers) {
			return fmt.Errorf("cuda: SetAdapter got %d layers, model has %d", len(layers), len(r.layers))
		}
		built := make([]cudaLoraLayer, len(layers))
		for i := range layers {
			projs := projForCUDA(&layers[i])
			dst := []**cudaLoraProj{&built[i].q, &built[i].k, &built[i].v, &built[i].o, &built[i].gate, &built[i].up, &built[i].down}
			for j, proj := range projs {
				if proj == nil {
					continue
				}
				if proj.R <= 0 || proj.R > loraRMax {
					r.releaseLoraLayers2(built[:i+1])
					return fmt.Errorf("cuda: SetAdapter: LoRA rank %d out of range (want 1..%d)", proj.R, loraRMax)
				}
				*dst[j] = &cudaLoraProj{a: r.up32(proj.A), b: r.up32(proj.B), rank: proj.R, outn: proj.Out, scale: proj.Scale}
				if r.setupErr != nil {
					err := r.setupErr
					r.setupErr = nil
					r.releaseLoraLayers2(built[:i+1])
					return fmt.Errorf("cuda: SetAdapter: uploading layer %d projection %d: %w", i, j, err)
				}
			}
		}
		r.loraLayers = built
		return nil
	})
}

// releaseLoraLayers2 is releaseLoraLayers over an explicit slice (not r.loraLayers) — used to
// clean up a partially-built set on a mid-loop error in SetAdapter, so a rejected bind (bad
// rank, a failed upload) never leaks the projections it already allocated.
func (r *cudaResident) releaseLoraLayers2(layers []cudaLoraLayer) {
	for _, l := range layers {
		for _, p := range []*cudaLoraProj{l.q, l.k, l.v, l.o, l.gate, l.up, l.down} {
			if p == nil {
				continue
			}
			r.dev.ReleaseBuf(p.a)
			r.dev.ReleaseBuf(p.b)
		}
	}
}

// applyLora dispatches one projection's compute-time LoRA delta into dst, ADDITIVELY — no-op if
// p is nil (an untargeted projection, or no adapter bound at all). aq/ascale must be the SAME
// quantized activation the base projection this delta rides alongside already consumed; dst
// must already hold the base projection's result — the delta is added on top, matching
// applyLoRA's CPU "matmul, then add" order exactly.
func (r *cudaResident) applyLora(p *cudaLoraProj, aq, ascale Buffer, k int, dst Buffer) error {
	if p == nil {
		return nil
	}
	if err := r.launch(r.fLoraDown, onecfg(256, 256*4),
		Arg(aq), Arg(ascale), Arg(p.a), Arg(r.loraT),
		gpu.ArgValue(int32(k)), gpu.ArgValue(int32(p.rank))); err != nil {
		return err
	}
	return r.launch(r.fLoraUp, g1cfg(p.outn, 256),
		Arg(p.b), Arg(r.loraT), Arg(dst),
		gpu.ArgValue(int32(p.rank)), gpu.ArgValue(int32(p.outn)), gpu.ArgValue(p.scale))
}
