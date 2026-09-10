package decoder

import "github.com/townsendmerino/aikit/linalg"

// L-01 hybrid CPU/GPU MoE expert execution (docs/task-l01-hybrid-moe-cpu-gpu.md) — the export
// seam a GPU backend needs to run one missed expert's SwiGLU MLP on the CPU from bytes it
// already holds pinned in host memory, instead of DMA-fetching it. Nothing in this file is
// called from goinfer's own decode path; it exists for cuda's l01_cpu_offload.go to call.

// F16BitsToF32 exports the canonical f16-bits decoder every resident backend's int4 group
// scales agree on bit-for-bit (audit C-15) — a GPU backend decoding its OWN f16 scales for a
// CPU-side WeightMat needs this same conversion, not a re-derivation that could drift from it.
func F16BitsToF32(h uint16) float32 { return f16bitsToF32(h) }

// CPUExpertWeights is what ComputeExpertMLP needs to run one MoE expert's SwiGLU MLP on the
// CPU. The caller builds these from whatever pinned/resident memory it already holds (e.g.
// cuda's own C′ pinned host expert stack, unpermuted back to the plain int4 byte layout every
// backend agrees on) — this package does not reach into another backend's buffers itself.
type CPUExpertWeights struct {
	Gate, Up, Down linalg.WeightMat
}

// ComputeExpertMLP runs one expert's SwiGLU MLP (gate*silu(up) → down) on the CPU into dst.
// gate/up are scratch of length inter, reused across calls the same way moeMLP's own
// sequential loop already does (mlp.go) — every OTHER caller in this package uses the
// unexported swiGLUExpert directly; this is L-01's own exported entry point.
func ComputeExpertMLP(w CPUExpertWeights, h, dst []float32, inter int, gate, up []float32) {
	ex := &expertWeights{Gate: w.Gate, Up: w.Up, Down: w.Down}
	swiGLUExpert(ex, h, dst, inter, &cpuBackend{}, gate, up)
}
