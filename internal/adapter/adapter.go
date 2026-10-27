// Package adapter holds the compute-time LoRA delta a resident backend applies, as plain data: decoder builds it and the
// backends consume it. It imports nothing, so a backend's kernel layer can take it without reaching decoder. decoder
// re-exports both types under their old names (decoder.ResidentAdapterProj, decoder.ResidentAdapterLayer).
package adapter

// ResidentAdapterProj is one projection's low-rank compute-time LoRA delta (G3,
// docs/tasks/task-gpu-paths-2026-09.md): the exported twin of decoder's package-private loraDelta, since a resident
// backend lives in another module and cannot see unexported fields. Same layout and semantics as applyLoRA's CPU
// reference (decoder/lora.go): y[o] += Scale · Σ_k B[o,k]·(A·x)[k].
type ResidentAdapterProj struct {
	A, B       []float32 // lora_A [R,In] row-major, lora_B [Out,R] row-major
	R, In, Out int
	Scale      float32
}

// ResidentAdapterLayer is one transformer layer's per-projection deltas; a nil field is a projection this adapter does
// not target, matching decoder's loraLayerDelta "nil ⇒ no-op" convention, so a backend's dispatch loop is a direct
// nil-check per projection.
type ResidentAdapterLayer struct {
	Q, K, V, O, Gate, Up, Down *ResidentAdapterProj
}
