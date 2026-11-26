//go:build cuda

package cuda

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/townsendmerino/goinfer/decoder"
)

// prefillStaticDecline reports why the batched path cannot run for THIS model, or nil. It covers the model-dependent
// guards (arch, per-layer geometry, weight kind, kernel availability) and deliberately not the prompt-dependent one
// (checkCap needs M/startPos), so PrefillPath can answer at load time from the code prefillCore enforces at call time.
// Per-head QK norm and Gemma sandwich norms are batched and do not decline (backend.go asserts qNorm/kNorm length ==
// headDim before residency); MoE and the Gemma parallel dense||MoE take the sequential path.
func (r *cudaResident) prefillStaticDecline() error {
	if err := r.prefillFlagsDecline(); err != nil {
		return err
	}
	// Every layer must have batched steps (cudaPrefillStepsOf); prefillFlagsDecline declines every model that has a layer without.
	for l := range r.layers {
		if len(r.layers[l].prefillSteps) == 0 {
			return fmt.Errorf("cuda prefill: layer %d has no batched steps: %w", l, errPrefillDeclined)
		}
	}
	return nil
}

// prefillFlagsDecline is prefillStaticDecline's guards read from the layers' flags and buffers, without the check on their batched
// steps; TestPrefillSteps_equalTheFlags compares the two.
func (r *cudaResident) prefillFlagsDecline() error {
	if !r.prefillReady {
		return fmt.Errorf("cuda prefill: batched kernels unavailable: %w", errPrefillDeclined)
	}
	if r.actG32 && !r.prefillG32Ready {
		// The per-vector batched quantizers would write one scale a row where the per-32 GEMVs read 2·K/32.
		return fmt.Errorf("cuda prefill: the per-32 batched kernels did not load: %w", errPrefillDeclined)
	}
	// MoE batches the attention half; its FFN runs row by row off the batched residual (prefillCore) and keeps decode's
	// exact per-token expert sequence. The per-token debug seams would fire M times per layer in that row loop, so they
	// decline: hidCapTaps (one residual per tap per token, for a block drafter) and layerCap (one snapshot per layer, for
	// the divergence probe) would silently return Mx the rows their consumers expect. The guard is scoped to MoE on purpose:
	// prefillCore calls layerTail only inside the per-row MoE FFN loop, and a block drafter arms hidCapTaps and verifies
	// through the batched path on a DENSE model (TestDFlashRoundComposition, TestDFlashCompositionResidual), so widening it
	// breaks that.
	if r.moe || r.gemma4Moe {
		if len(r.hidCapTaps) > 0 {
			return fmt.Errorf("cuda prefill: MoE per-row FFN with per-token hidden-state taps armed "+
				"would record M rows per tap instead of one: %w", errPrefillDeclined)
		}
		if r.layerCap {
			return fmt.Errorf("cuda prefill: MoE per-row FFN with per-layer residual capture armed "+
				"would append M snapshots per layer: %w", errPrefillDeclined)
		}
	}
	// Gated-DeltaNet layers, whose conv ring and matrix state must advance one token at a time in order, are handled inside
	// the pass: prefillCore branches on Ly.isDeltaNet() and runs the recurrence through row-batched twins of decode's kernels.
	// Dispatch is on the layer's kind, not on which
	// weights happen to be absent (a dense attention stack once ran over conv layers that load no q/k/v/o), and
	// nonBatchableKind checks the projections each kind binds. ForwardN (spec verify) stays sequential for this family:
	// verify rewinds and recurrent state does not.
	//
	// Geometry is per layer: the batched launches bind each layer's own hd/nKV/qDim/kvDim/rhalf, and the M-sized scratch is
	// sized by the maximum across layers (prefillMaxGeom), so a family whose layers differ strides correctly.
	for l := range r.layers {
		Ly := &r.layers[l]
		// K=V layers (Gemma 4 global) batch as decode does: a second k-projection GEMV into the V buffer, then a scale-less
		// v_norm over it before rope rotates k. That needs the unit-weight buffer, allocated only when some layer is kEqV, so
		// refuse rather than bind a null weight.
		if Ly.vNorm && r.vNormUnit == (Buffer{}) {
			return fmt.Errorf("cuda prefill: v_norm layer at %d but no v_norm unit weight: %w", l, errPrefillDeclined)
		}
		if k := nonBatchableKind(Ly, r.actG32); k != "" {
			return fmt.Errorf("cuda prefill: %s weight at layer %d needs the sequential path: %w", k, l, errPrefillDeclined)
		}
	}
	return nil
}

// prefillMaxGeom returns the largest qDim and kvDim across layers: the row strides the M-sized Q/K/V/context scratch is
// allocated at. Each launch binds its own layer's dims, so a narrower layer uses a prefix of each row; the scratch has
// no cross-layer meaning, which is what lets one allocation serve a non-uniform stack.
func (r *cudaResident) prefillMaxGeom() (maxQDim, maxKvDim int) {
	for l := range r.layers {
		maxQDim = max(maxQDim, r.layers[l].qDim)
		maxKvDim = max(maxKvDim, r.layers[l].kvDim)
	}
	return maxQDim, maxKvDim
}

// checkPrefillShmem is prefillStaticDecline's prompt-dependent twin. The exact batched attention (attn_batched and its image
// twin) sizes its dynamic shared memory as decode's does, (maxNWin+128)*4 bytes, so a launch past singleBlockAttnShmemLimit
// is not survivable and must be a named decline; otherwise the launch fails in the driver and the caller silently falls to
// the sequential path. prefillCore asks it only when the row-batched exact attention did not load: with it, an ordinary pass and an image-block
// pass both have no layer attn_batched or its image twin would run past the bound (prefillSplitKVServes). Checked per layer, because a sliding-window layer's maxNWin is clamped to its
// window. It mirrors prefillCore's per-layer maxNWin exactly, so it neither declines a shape the launch would run nor misses
// one it would fail.
//
// fused says, for a layer's head dimension, whether the fused attention (attn_fused*) serves it in this pass: that kernel
// stages 64 keys at a time and its shared memory does not grow with the span, so such a layer needs nothing of it. nil means
// no layer is fused-served, the exact kernel's bound on every layer.
func (r *cudaResident) checkPrefillShmem(startPos, M int, fused func(hd int) bool) error {
	for l := range r.layers {
		Ly := &r.layers[l]
		maxNWin := startPos + M
		if Ly.window > 0 && int(Ly.window) < maxNWin {
			maxNWin = int(Ly.window)
		}
		if splitKVRequired(maxNWin) {
			if fused != nil && fused(Ly.hd) {
				continue
			}
			return fmt.Errorf("cuda prefill: layer %d attention at %d attended keys needs %d B of "+
				"shared memory, past this device's %d B limit — the row-batched exact attention does not serve "+
				"this pass (its module did not load) and neither does the fused attention (it needs head dim 64 "+
				"or 128, an ordinary last-row prefill, 16 rows or more, no image block), so this prompt length needs the sequential path: %w",
				l, maxNWin, attnShmemBytes(maxNWin), singleBlockAttnShmemLimit, errPrefillDeclined)
		}
	}
	return nil
}

// nonBatchableKind returns the weight kind of the layer's first projection the batched GEMVs cannot handle, or "" when
// every projection is int4 or int8 (bGemvB: int4 uses gemv_w4a8_batched/_rn, int8 gemv_w8a8_batched; a uniform or mixed
// int4/int8 bundle all batch). Naming the kind makes the startup message actionable.
func nonBatchableKind(Ly *cudaLayer, actG32 bool) string {
	// A Mamba-2 mixer and a feed-forward-only block have no batched kernels: their prompts take the sequential path, named here
	// and not left to the absent q/k/v weights reading as "absent/unquantized".
	switch Ly.mixer {
	case decoder.MixerMamba:
		return "Mamba-2 mixer"
	case decoder.MixerNone:
		return "feed-forward-only block"
	}
	// Check exactly the projections this layer's batched path binds, and no others. An absent weight has kind "", which the
	// caller reads as "no problem", so checking an absent weight passes vacuously and tightening the sentinel would turn
	// that into a wrong refusal; narrow the list instead. Ly.v is absent on a K=V layer (V is derived from k), and Ly.g/u/d
	// on a pure-MoE layer (its FFN runs per row on the expert stacks). The expert stacks are deliberately not checked: the
	// per-row MoE FFN issues decode's launches on decode's weights, so whatever kind decode accepts it accepts here.
	ws := []cudaWQ{Ly.q, Ly.o}
	if !Ly.kvShared { // a KV-shared layer (Gemma 4 E-model) has no k_proj or v_proj: it attends over its source's cache
		ws = append(ws, Ly.k)
		if !Ly.kEqV {
			ws = append(ws, Ly.v)
		}
	}
	if Ly.pleGate.N > 0 || Ly.pleProj.N > 0 { // the E-model per-layer embedding branch runs through the batched GEMVs too
		ws = append(ws, Ly.pleGate, Ly.pleProj)
	}
	if Ly.isDeltaNet() {
		// A Gated-DeltaNet layer has no q/k/v/o; prefillDeltaNetRows binds these five instead.
		ws = []cudaWQ{Ly.dnQKV, Ly.dnB, Ly.dnA, Ly.dnZ, Ly.dnOut}
	}
	if !Ly.g4moe && !Ly.isMoE {
		ws = append(ws, Ly.g, Ly.u, Ly.d)
	}
	for _, w := range ws {
		if w.kind == "" {
			return "absent/unquantized"
		}
		if !batchableKind(w.kind, actG32) {
			return w.kind
		}
	}
	return ""
}

// batchableKind reports whether bGemvRows has a batched GEMV for a weight of this kind: int4 and int8 under per-vector
// activations, the q4k lane's Q4_K, Q5_K, Q6_K and int8 under per-32 (prefill_g32.cu has no int4 twin).
func batchableKind(kind string, actG32 bool) bool {
	if actG32 {
		return kind == "q4k" || kind == "q5k" || kind == "q6k" || kind == "int8"
	}
	return kind == "int4" || kind == "int8"
}

// PrefillPath (decoder.PrefillPathReporter) answers at load, before a request is served, whether this model gets the
// batched prefill, and says why not. It declines what prefillStaticDecline declines, including a native/f32 projection.
func (r *cudaResident) PrefillPath() (bool, string) {
	err := r.prefillStaticDecline()
	if err == nil {
		// Say rows per pass, not "one pass": the report is read as a promise about how a long prompt is ingested, and a prompt
		// past the chunk width is several weight-stationary passes (TestPrefillChunked_bitIdentical).
		return true, fmt.Sprintf("batched (weight-stationary CUDA passes of up to %d rows)%s",
			r.prefillPassRows(), r.fusedAttnNote())
	}
	// Detail without the wrapped sentinel, which says nothing a user can act on.
	detail := strings.TrimPrefix(strings.TrimSuffix(err.Error(), ": "+errPrefillDeclined.Error()), "cuda prefill: ")
	for l := range r.layers {
		if r.layers[l].mixer == decoder.MixerMamba {
			// Not a precision problem: the Mamba-2 recurrence has no batched kernel, so a prompt runs one forward per token at any quant.
			return false, "sequential — the Mamba-2 mixer has no batched kernel, so a prompt runs one forward per token (slower TTFT)"
		}
	}
	if len(r.layers) > 0 {
		if k := nonBatchableKind(&r.layers[0], r.actG32); k != "" {
			if r.actG32 {
				return false, fmt.Sprintf("sequential — batched prefill under per-32 activations needs Q4_K, Q5_K, Q6_K or int8 projections (%s weights) (slower TTFT: one forward per prompt token)", k)
			}
			return false, fmt.Sprintf("sequential — batched prefill needs int4 or int8 projections (%s weights) — ~9× slower TTFT, 20× CPU on a 300-token prompt", k)
		}
	}
	return false, "sequential — " + detail + " (slower TTFT: one forward per prompt token)"
}

// fusedAttnNote describes the attention kernel PrefillPath reports on. It states the condition, not a verdict: the
// kernel depends on head dim and on M, and M is a call-time property PrefillPath never sees, so a flat "fused" would be
// the same untestable promise as "one pass".
func (r *cudaResident) fusedAttnNote() string {
	if !r.fastAttn && !r.fastGemm {
		return ""
	}
	if r.eModel {
		// prefillCore forces the exact kernels for an E-model at every length; saying "fused" here would be the untestable promise this note avoids.
		return "; exact kernels (the fast levers are not fidelity-gated on Gemma 4 E-models)"
	}
	if r.actG32 {
		return "; exact kernels (the fast levers are not fidelity-gated on per-32 activations)"
	}
	if !r.fastAttn {
		return "; weight term: gemm_w4a8_mma (L3) where M>=" + strconv.Itoa(gemmMMAMinRows) +
			"; attention: attn_batched (exact)"
	}
	if r.bAttnFused64 == (Pipeline{}) && r.bAttnFused128 == (Pipeline{}) {
		return "; fused attention REQUESTED but its module did not load — attn_batched (exact) throughout"
	}
	var served []string
	for _, hd := range []int{64, 128} {
		if p, _ := r.attnFusedFor(hd); p != (Pipeline{}) {
			served = append(served, strconv.Itoa(hd))
		}
	}
	return fmt.Sprintf("; attention: attn_fused (L2, head dim %s, M>=%d) else attn_batched (exact)",
		strings.Join(served, "/"), attnFusedMinRows)
}

// prefillEModelDecline is a Gemma 4 E-model's decline from a pass, nil for any other model.
func (r *cudaResident) prefillEModelDecline(embeddings [][]float32, blocks [][2]int, mropePos [][3]int, rows []stepRow) error {
	if r.eModel {
		// Gemma 4 E-model: text prompts batch. A step (its rows' PLE tails and KV slots are not staged by the step path), an
		// image block and m-RoPE (no E-model has it) stay sequential.
		if rows != nil || len(blocks) > 0 || mropePos != nil {
			return fmt.Errorf("cuda prefill: Gemma 4 E-model batches text prompts only (no batched multi-conversation step, image block or m-RoPE yet): %w", errPrefillDeclined)
		}
		for m, e := range embeddings {
			if len(e) != r.embLen {
				// An E-model row is [hidden ‖ nLayers*P] (launchToken refuses the same by length): a hidden-sized row would be read as a full one with a short tail.
				return fmt.Errorf("cuda prefill: Gemma 4 E-model embedding row %d has %d floats, want %d (hidden %d + %d layers x PLE width %d)", m, len(e), r.embLen, r.hidden, r.nLayers, r.pleP)
			}
		}
	}
	return nil
}

// prefillRoomDecline is the pass's position and KV checks: a step's rows (one decode token each, at its own position), or an
// ordinary pass against the cap and the windowed KV's room.
func (r *cudaResident) prefillRoomDecline(startPos, M, tail int, blocks [][2]int, mropePos [][3]int, rows []stepRow) error {
	if rows != nil {
		// A step: each row is one decode token at its own position, and its attention is decode's (whose own launch
		// checks the shared-memory limit per row), so the whole-pass checks below do not apply.
		if len(rows) != M || tail != tailAllLogits || len(blocks) > 0 || mropePos != nil {
			return fmt.Errorf("cuda step: %d rows for %d embeddings, tail %d — a step is all-logits, text only", len(rows), M, tail)
		}
		for _, rw := range rows {
			if e := r.checkCap(rw.pos, 1); e != nil {
				return e
			}
		}
	} else if e := r.checkCap(startPos, M); e != nil {
		return e
	} else if !r.kvRoomFor(startPos, M) {
		return fmt.Errorf("cuda prefill: %d rows at position %d do not fit the windowed KV (window %d + %d slack): %w", M, startPos, r.kvWindow, r.kvSlack, errPrefillDeclined)
	}
	return nil
}
