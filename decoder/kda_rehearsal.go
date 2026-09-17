package decoder

import "math"

// F4 (docs/completed/task-families-2026-09.md): KDA (Kimi Delta Attention) recurrence rehearsal for
// Ling-3.0-tiny / Kimi K3's linear-attention mixer, written as the scoped bring-up the F4 brief
// asked for: prove the one genuinely new piece of KDA's math against a real reference before any
// registry work. N-08 (audit-2026-09-10): "NOT wired to any registered family" went stale as of
// e6b31cc — kdaMixerStep (decoder/kda.go, called from forward_bailing.go's served path for
// bailing_hybrid/Ling 3.0) reuses kdaLowerBoundGate and kdaRecurrentStep below directly, so this
// is production code now, not just a bring-up rehearsal.
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
// kdaLowerBoundGateInto computes KDA's "safe_gate" per-channel log-decay directly into out.
func kdaLowerBoundGateInto(out, rawGate, dtBias []float32, aLog, lowerBound float32) {
	n := len(rawGate)
	if n == 0 {
		return
	}
	_ = out[n-1]
	_ = rawGate[n-1]
	_ = dtBias[n-1]
	scale := float32(math.Exp(float64(aLog)))
	i := 0
	for ; i+3 < n; i += 4 {
		out[i] = lowerBound * sigmoidf(scale*(rawGate[i]+dtBias[i]))
		out[i+1] = lowerBound * sigmoidf(scale*(rawGate[i+1]+dtBias[i+1]))
		out[i+2] = lowerBound * sigmoidf(scale*(rawGate[i+2]+dtBias[i+2]))
		out[i+3] = lowerBound * sigmoidf(scale*(rawGate[i+3]+dtBias[i+3]))
	}
	for ; i < n; i++ {
		out[i] = lowerBound * sigmoidf(scale*(rawGate[i]+dtBias[i]))
	}
}

func kdaLowerBoundGate(rawGate, dtBias []float32, aLog, lowerBound float32) []float32 {
	out := make([]float32, len(rawGate))
	kdaLowerBoundGateInto(out, rawGate, dtBias, aLog, lowerBound)
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
	out := make([]float32, len(v))
	kdaRecurrentStepInto(out, q, k, v, decay, beta, S)
	return out
}

// kdaRecurrentStepInto is the zero-allocation, 4-way vectorized implementation of kdaRecurrentStep
// writing directly into the caller's out buffer.
func kdaRecurrentStepInto(out, q, k, v, decay []float32, beta float32, S []float32) {
	K, V := len(q), len(v)

	// 1. Per-channel decay — KDA's departure from Gated DeltaNet's single scalar.
	for kd := range K {
		row := S[kd*V : kd*V+V]
		dk := decay[kd]
		_ = row[V-1]
		vd := 0
		for ; vd+3 < V; vd += 4 {
			row[vd] *= dk
			row[vd+1] *= dk
			row[vd+2] *= dk
			row[vd+3] *= dk
		}
		for ; vd < V; vd++ {
			row[vd] *= dk
		}
	}

	// 2. kv = k · S(decayed); delta = beta·(v − kv) — identical to gatedDeltaNetStep.
	var kvBuf [256]float32
	var kv []float32
	if V <= len(kvBuf) {
		kv = kvBuf[:V]
	} else {
		kv = make([]float32, V)
	}
	clear(kv)
	for kd := range K {
		row := S[kd*V : kd*V+V]
		kk := k[kd]
		addScaled(kv, row, kk)
	}

	delta := kv // aliased and overwritten in place, same convention as gatedDeltaNetStep
	_ = delta[V-1]
	_ = v[V-1]
	vd := 0
	for ; vd+3 < V; vd += 4 {
		delta[vd] = (v[vd] - delta[vd]) * beta
		delta[vd+1] = (v[vd+1] - delta[vd+1]) * beta
		delta[vd+2] = (v[vd+2] - delta[vd+2]) * beta
		delta[vd+3] = (v[vd+3] - delta[vd+3]) * beta
	}
	for ; vd < V; vd++ {
		delta[vd] = (v[vd] - delta[vd]) * beta
	}

	// 3. S += k ⊗ delta; out = q · S(final) — identical to gatedDeltaNetStep.
	clear(out)
	_ = out[V-1]
	for kd := range K {
		row := S[kd*V : kd*V+V]
		kk, qq := k[kd], q[kd]
		_ = row[V-1]
		_ = delta[V-1]
		vd := 0
		for ; vd+3 < V; vd += 4 {
			r0 := row[vd] + kk*delta[vd]
			r1 := row[vd+1] + kk*delta[vd+1]
			r2 := row[vd+2] + kk*delta[vd+2]
			r3 := row[vd+3] + kk*delta[vd+3]
			row[vd] = r0
			row[vd+1] = r1
			row[vd+2] = r2
			row[vd+3] = r3
			out[vd] += r0 * qq
			out[vd+1] += r1 * qq
			out[vd+2] += r2 * qq
			out[vd+3] += r3 * qq
		}
		for ; vd < V; vd++ {
			r := row[vd] + kk*delta[vd]
			row[vd] = r
			out[vd] += r * qq
		}
	}
}
