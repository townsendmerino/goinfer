package decoder

import "math"

// F4 (docs/task-families-2026-09.md): KDA (Kimi Delta Attention) recurrence rehearsal for
// Ling-3.0-tiny / Kimi K3's linear-attention mixer. NOT wired to any registered family — this is
// the scoped bring-up the F4 brief asked for: prove the one genuinely new piece of KDA's math
// against a real reference before any registry work.
//
// Verified against fla-org/flash-linear-attention's actual source (fla/ops/kda/{naive,gate}.py,
// not the HF modeling file's paraphrase, which only calls the opaque Triton kernel): KDA's
// delta-rule recurrence is structurally IDENTICAL to the Gated DeltaNet this repo already ships
// for qwen3_5_moe (gatedDeltaNetStep, decoder/deltanet.go) -- same beta write-gate, same
// outer-product delta update, same q/k L2-norm-in-kernel, same final q·S read -- except the decay
// that Gated DeltaNet applies as ONE SCALAR to the whole [head_k_dim, head_v_dim] state block is,
// in KDA, PER-CHANNEL: one decay value per row of S (one per key-dimension), not one for the
// entire block. That is the one new primitive; everything else composes from what qwen3_5_moe
// already validated.

// kdaLowerBoundGate computes KDA's "safe_gate" per-channel log-decay:
//
//	g = lower_bound * sigmoid(exp(A_log) * (rawGate + dtBias))
//
// the exact function Ling-3.0-tiny's released config selects (kda_safe_gate: true,
// kda_lower_bound: -5), verified against fla's naive_kda_lowerbound_gate. rawGate/dtBias/the
// output are all per-channel ([head_k_dim] for one head); aLog is that head's single scalar
// parameter. The result is a log-decay in (lowerBound, 0), exponentiated by the caller before
// use in kdaRecurrentStep.
func kdaLowerBoundGate(rawGate, dtBias []float32, aLog, lowerBound float32) []float32 {
	out := make([]float32, len(rawGate))
	scale := float32(math.Exp(float64(aLog)))
	for i, g := range rawGate {
		out[i] = lowerBound * sigmoidf(scale*(g+dtBias[i]))
	}
	return out
}

// kdaRecurrentStep advances the KDA recurrence by one timestep for a single head, updating S
// (row-major [K*V], one head's state block) in place and returning the output [V]. decay is
// exp(kdaLowerBoundGate(...)) — one value per row of S (per key-channel) — where
// gatedDeltaNetStep's equivalent step multiplies every row by the SAME scalar. q/k are
// pre-L2-normalized (q already carrying the 1/√K scale, matching l2normScaled's convention);
// beta is sigmoid(b_proj(h)) for this head, same as Gated DeltaNet's own beta gate.
//
// Verified against fla's naive_recurrent_kda: decay S row-wise FIRST, then read kv = k·S(decayed)
// exactly as gatedDeltaNetStep does, then the identical outer-product delta write and q·S(final)
// read. The three-phase structure is unchanged from Gated DeltaNet; only phase 1 (decay) differs.
func kdaRecurrentStep(q, k, v, decay []float32, beta float32, S []float32) []float32 {
	K, V := len(q), len(v)

	// 1. Per-channel decay — KDA's departure from Gated DeltaNet's single scalar.
	for kd := range K {
		row := S[kd*V : kd*V+V]
		dk := decay[kd]
		for vd := range V {
			row[vd] *= dk
		}
	}
	// 2. kv = k · S(decayed); delta = beta·(v − kv) — identical to gatedDeltaNetStep.
	kv := make([]float32, V)
	for kd := range K {
		row := S[kd*V : kd*V+V]
		kk := k[kd]
		for vd := range V {
			kv[vd] += row[vd] * kk
		}
	}
	delta := kv // aliased and overwritten in place, same convention as gatedDeltaNetStep
	for vd := range V {
		delta[vd] = (v[vd] - kv[vd]) * beta
	}
	// 3. S += k ⊗ delta; out = q · S(final) — identical to gatedDeltaNetStep.
	out := make([]float32, V)
	for kd := range K {
		row := S[kd*V : kd*V+V]
		kk, qq := k[kd], q[kd]
		for vd := range V {
			row[vd] += kk * delta[vd]
			out[vd] += row[vd] * qq
		}
	}
	return out
}
