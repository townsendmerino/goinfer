//go:build cuda

package cuda

import (
	"fmt"

	gpu "github.com/townsendmerino/goinfer/backend/cuda/internal/gpudev"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

// segA / segB / segC are the three graph-STATIC launch runs of one layer, factored out of launchToken so a SINGLE
// source of truth serves both the live path (call them in sequence) and the graph path (replay Ly.gSegA/B/C).
// Every launch here binds only fixed buffers and per-layer constants; the per-token dynamics (rope_kv/attention
// bind pos/nKeys; the g4x2 clear and the routing readback are host copies) stay live in the gaps between them. See
// cudaLayer.gSegA and captureGraphs.

// segA: QKV projection (fused QKV super-kernel when every projection is int4, else rmsnorm+quant +
// three GEMVs) + per-head QK-norm + scale-less K=V v-norm. Ends before rope_kv.
func (r *cudaResident) segA(Ly *cudaLayer, l int) error {
	nullBias := ArgNull()
	qb, kb, vb := nullBias, nullBias, nullBias
	if Ly.hasBias {
		qb, kb, vb = Arg(Ly.qb), Arg(Ly.kb), Arg(Ly.vb)
	}
	if r.floatAct {
		if e := r.floatActQKV(Ly, qb, kb, vb); e != nil {
			return e
		}
	} else if r.fuseQKV {
		// Fused QKV: rmsnorm+quant redundantly per block + this block's Q/K/V rows — one launch instead of four.
		nrows := Ly.qDim + 2*Ly.kvDim
		cfg := LaunchConfig{GridX: uint32((nrows + 7) / 8), GridY: 1, GridZ: 1,
			BlockX: 256, BlockY: 1, BlockZ: 1,
			SharedMemBytes: uint32((r.hidden + 256 + r.hidden/4) * 4)}
		// Rows-per-warp variant (fused_qkv_rows.cu): each warp walks several rows off one shared activation, so fewer blocks pay the redundant
		// rmsnorm+quant prologue. Bit-identical to the one-row-per-warp kernel (TestFusedQKVRowsBitIdentical), so this only ever changes speed.
		rpw := r.waveRowsPerWarp(nrows, int(cfg.SharedMemBytes))
		if rpw == 0 {
			rpw = fusedQKVRowsPerWarp(nrows) // device shape unread: the static rule
		}
		if rpw > 1 && r.fQKVRows != (Pipeline{}) {
			rcfg := cfg
			rcfg.GridX = uint32((nrows + 8*rpw - 1) / (8 * rpw))
			if e := r.launch(r.fQKVRows, rcfg,
				Arg(r.x), Arg(Ly.preNorm), gpu.ArgValue(int32(r.hidden)), gpu.ArgValue(r.eps),
				gpu.ArgValue(r.addOneArg()),
				Arg(Ly.q.W), Arg(Ly.q.ws16), qb,
				Arg(Ly.k.W), Arg(Ly.k.ws16), kb,
				Arg(Ly.v.W), Arg(Ly.v.ws16), vb,
				gpu.ArgValue(int32(Ly.qDim)), gpu.ArgValue(int32(Ly.kvDim)),
				gpu.ArgValue(int32(r.hidden/8)), gpu.ArgValue(int32(r.hidden/32)), gpu.ArgValue(int32(rpw)),
				Arg(r.qB), Arg(r.kB), Arg(r.vB)); e != nil {
				return e
			}
		} else if e := r.launch(r.fQKV, cfg,
			Arg(r.x), Arg(Ly.preNorm), gpu.ArgValue(int32(r.hidden)), gpu.ArgValue(r.eps),
			gpu.ArgValue(r.addOneArg()),
			Arg(Ly.q.W), Arg(Ly.q.ws16), qb,
			Arg(Ly.k.W), Arg(Ly.k.ws16), kb,
			Arg(Ly.v.W), Arg(Ly.v.ws16), vb,
			gpu.ArgValue(int32(Ly.qDim)), gpu.ArgValue(int32(Ly.kvDim)),
			gpu.ArgValue(int32(r.hidden/8)), gpu.ArgValue(int32(r.hidden/32)),
			Arg(r.qB), Arg(r.kB), Arg(r.vB)); e != nil {
			return e
		}
	} else if r.fuseG32 {
		// Per-32 twin of the fused QKV kernel: rmsnorm + per-32 quant redundantly per block + Q/K/V rows of any per-32
		// kind — one launch instead of four. Bit-identical to the unfused chain below (shared row code).
		nrows := Ly.qDim + 2*Ly.kvDim
		rpw := fusedQKVRowsPerWarp(nrows)
		cfg := LaunchConfig{GridX: uint32((nrows + 8*rpw - 1) / (8 * rpw)), GridY: 1, GridZ: 1,
			BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: fusedG32Shmem(r.hidden)}
		if e := r.launch(r.fQKVg32, cfg,
			Arg(r.x), Arg(Ly.preNorm), gpu.ArgValue(int32(r.hidden)), gpu.ArgValue(r.eps),
			gpu.ArgValue(r.addOneArg()),
			Arg(Ly.q.W), wkScale(Ly.q), qb, gpu.ArgValue(wkCode(Ly.q.kind)),
			Arg(Ly.k.W), wkScale(Ly.k), kb, gpu.ArgValue(wkCode(Ly.k.kind)),
			Arg(Ly.v.W), wkScale(Ly.v), vb, gpu.ArgValue(wkCode(Ly.v.kind)),
			gpu.ArgValue(int32(Ly.qDim)), gpu.ArgValue(int32(Ly.kvDim)), gpu.ArgValue(int32(rpw)),
			Arg(r.qB), Arg(r.kB), Arg(r.vB)); e != nil {
			return e
		}
	} else {
		aw := &Ly.walk.Attn
		if e := r.emitInputNorm(aw.InputForm, r.x, Ly.preNorm, r.aq, r.aSc); e != nil {
			return e
		}
		if aw.Gated {
			// attn_output_gate: q_proj emits [query ‖ gate] PER HEAD at double width. Project into
			// a 2*qDim scratch and split on the ACTIVATION — the weight stays fused because it is
			// quantized, and slicing rows out of an int4 bundle with its per-group scales is real
			// surgery. Interleaved per head, NOT two concatenated blocks.
			if e := r.doG(Ly.q, r.aq, r.aSc, qb, r.dnQg, 0); e != nil {
				return e
			}
			if e := r.launch(r.dnQSplit, g1cfg(Ly.qDim, 256),
				Arg(r.dnQg), Arg(r.qB), Arg(r.dnAGate),
				gpu.ArgValue(int32(Ly.qDim)), gpu.ArgValue(int32(Ly.hd))); e != nil {
				return e
			}
		} else if e := r.doG(Ly.q, r.aq, r.aSc, qb, r.qB, 0); e != nil {
			return e
		}
		// A K/V-shared layer (Gemma 4 E-model) has no k_proj and no v_proj: K/V come from its source layer's cache.
		if !Ly.kvShared {
			if err := r.doG(Ly.k, r.aq, r.aSc, kb, r.kB, 0); err != nil {
				return err
			}
			if Ly.kEqV {
				// K=V: vB is the RAW (pre-norm) k projection, which v_norm consumes below. A copy of kB is bit-identical to
				// projecting k a second time, and is a launch like every other op here, so it records into the CUDA graph and is
				// ordered on this stream between the projection and qk_norm (which rewrites kB in place).
				if err := r.copyF32(r.kB, r.vB, Ly.kvDim); err != nil {
					return err
				}
			} else {
				if err := r.doG(Ly.v, r.aq, r.aSc, vb, r.vB, 0); err != nil {
					return err
				}
			}
		}
	}
	// Compute-time LoRA: q/k/v deltas, right after the base projection and
	// before qk_norm/RoPE below, matching applyLoRA's CPU order exactly. Not reached for gated attention (a documented gap,
	// see cuda/lora.go's file comment): gated q_proj takes a different path entirely, into r.dnQg, split afterward,
	// so no single hook point covers both.
	if r.loraLayers != nil && !Ly.walk.Attn.Gated {
		if r.fuseQKV {
			// The fused QKV kernel computes rmsnorm+quant INTERNALLY and never writes r.aq/r.aSc
			// out — materialize them here (redundant work, correct result: rmsnorm_quant is a
			// deterministic function of r.x/Ly.preNorm) so the LoRA kernels have a valid input.
			if e := r.norm(r.x, Ly.preNorm, r.aq, r.aSc); e != nil {
				return e
			}
		}
		LA := &r.loraLayers[l]
		if e := r.applyLora(LA.q, r.aq, r.aSc, r.hidden, r.qB); e != nil {
			return e
		}
		if e := r.applyLora(LA.k, r.aq, r.aSc, r.hidden, r.kB); e != nil {
			return e
		}
		if e := r.applyLora(LA.v, r.aq, r.aSc, r.hidden, r.vB); e != nil {
			return e
		}
	}
	if Ly.walk.Attn.QKNorm { // per-head (or, for QKNormWhole, whole-vector) Q/K RMSNorm before RoPE
		addOne := int32(0)
		if Ly.walk.Attn.QKNormParams.AddOne {
			addOne = 1
		}
		// QKNormWhole (Olmo 3/Olmo Hybrid): the SAME per-head kernel, grid collapsed to one Q
		// block + one K block by passing nH=1,nKV=1,hd=nH_orig*hd — qk_norm's block h<nH reduces
		// [q+h*hd, +hd), so h=0 with hd=nH_orig*hd_orig spans the WHOLE contiguous Q vector, and
		// h=nH=1 spans the whole K vector, matching decoder/attention.go's rmsNorm(q,QNorm,1,
		// nH*hd,...) exactly. Only correct when nH==Ly.nKV (MHA) — BuildResident already declines
		// otherwise (backend.go), so that invariant holds here unconditionally.
		qkNH, qkNKV, qkHD := r.nH, Ly.nKV, Ly.hd
		if Ly.walk.Attn.QKNormParams.Whole {
			qkNH, qkNKV, qkHD = 1, 1, r.nH*Ly.hd
		}
		kNormW := Ly.kNorm
		if Ly.kvShared {
			// Q only: nKV=0 launches no K blocks, and a shared layer has no k_norm weight to bind (an empty Buffer is a nil deref).
			qkNKV, kNormW = 0, Ly.qNorm
		}
		if e := r.launch(r.fQKN, LaunchConfig{GridX: uint32(qkNH + qkNKV), GridY: 1, GridZ: 1,
			BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: 128 * 8},
			Arg(r.qB), Arg(r.kB), Arg(Ly.qNorm), Arg(kNormW),
			gpu.ArgValue(int32(qkNH)), gpu.ArgValue(int32(qkNKV)), gpu.ArgValue(int32(qkHD)),
			gpu.ArgValue(r.eps), gpu.ArgValue(addOne)); e != nil {
			return e
		}
	}
	if Ly.walk.Attn.G4VNorm && (Ly.kEqV || !g4DropVNormForTest) { // scale-less v_norm on vB (the raw k copy on a K=V layer, the v_proj output otherwise), BEFORE rope_kv rotates k
		if err := r.launch(r.fQKN, LaunchConfig{GridX: uint32(Ly.nKV), GridY: 1, GridZ: 1,
			BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: 128 * 8},
			Arg(r.vB), Arg(r.vB), Arg(r.vNormUnit), Arg(r.vNormUnit),
			gpu.ArgValue(int32(0)), gpu.ArgValue(int32(Ly.nKV)), gpu.ArgValue(int32(Ly.hd)),
			gpu.ArgValue(r.eps), gpu.ArgValue(int32(0))); err != nil {
			return err
		}
	}
	return nil
}

// segAMLA runs the MLA attention block: pre-norm -> Q (direct or LoRA bottleneck) ->
// KV-A projection -> latent store -> W_UK absorb -> Q RoPE -> MLA latent attention ->
// W_UV lift. Leaves the attention context in r.cctx [Ly.qDim], ready for segB (o-proj + FFN).
func (r *cudaResident) segAMLA(Ly *cudaLayer, l, pos, ropePos int) error {
	nullBias := ArgNull()
	// 1. Pre-norm on r.x -> (r.aq, r.aSc)
	if e := r.norm(r.x, Ly.preNorm, r.aq, r.aSc); e != nil {
		return e
	}
	// 2. Query projection: direct or LoRA bottleneck (q_a -> norm -> q_b)
	if r.mlaQLoRA > 0 {
		if e := r.doG(Ly.mlaQA, r.aq, r.aSc, nullBias, r.mlaQAOut, 0); e != nil {
			return e
		}
		if e := r.rmsDim(r.mlaQAOut, Ly.mlaQANorm, r.mlaQLoRA, r.mlaQAQ, r.mlaQASc); e != nil {
			return e
		}
		if e := r.doG(Ly.mlaQB, r.mlaQAQ, r.mlaQASc, nullBias, r.qB, 0); e != nil {
			return e
		}
	} else {
		if e := r.doG(Ly.mlaQ, r.aq, r.aSc, nullBias, r.qB, 0); e != nil {
			return e
		}
	}
	// 3. KV-A projection -> r.mlaKVDown [latDim]
	if e := r.doG(Ly.mlaKVA, r.aq, r.aSc, nullBias, r.mlaKVDown, 0); e != nil {
		return e
	}
	// 4. Latent Store: RMSNorm(kvDown[:rank]) + decoupled RoPE(kvDown[rank:]) -> the latent row of position pos
	var intl int32
	if r.mlaInterleave {
		intl = 1
	}
	if err := r.launch(r.fMlaStore, onecfg(64, 0),
		Arg(r.mlaKVDown), Arg(Ly.mlaKVANorm), Arg(Ly.invF), Arg(r.kc[l]), Arg(r.kvT(l)),
		gpu.ArgValue(int32(r.mlaRank)), gpu.ArgValue(int32(r.mlaQKRope)), gpu.ArgValue(int32(ropePos)),
		gpu.ArgValue(r.eps), gpu.ArgValue(int32(pos)), gpu.ArgValue(r.mlaRopeScale), gpu.ArgValue(intl)); err != nil {
		return err
	}
	// 5. W_UK Absorb: per-head block-diagonal matvec of W_UKᵀ with Q's qkNope slice -> r.qAbs
	elemCountAbs := r.nH * r.mlaRank
	cfgAbs := LaunchConfig{GridX: uint32(elemCountAbs), GridY: 1, GridZ: 1, BlockX: 64, BlockY: 1, BlockZ: 1}
	if err := r.launch(r.fMlaHeadMV, cfgAbs,
		Arg(r.qB), Arg(Ly.mlaWUK), Arg(r.qAbs),
		gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(r.mlaRank)), gpu.ArgValue(int32(r.mlaQKNope)),
		gpu.ArgValue(int32(r.mlaQKHead)), gpu.ArgValue(int32(r.mlaLatDim))); err != nil {
		return err
	}
	// 6. Q RoPE: decoupled RoPE on Q's qkRope slice -> r.qAbs[h*latDim + rank]
	half := r.mlaQKRope / 2
	cfgRope := LaunchConfig{GridX: uint32((r.nH*half + 63) / 64), GridY: 1, GridZ: 1, BlockX: 64, BlockY: 1, BlockZ: 1}
	if err := r.launch(r.fMlaQRope, cfgRope,
		Arg(r.qB), Arg(Ly.invF), Arg(r.qAbs),
		gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(r.mlaQKHead)), gpu.ArgValue(int32(r.mlaQKNope)),
		gpu.ArgValue(int32(r.mlaQKRope)), gpu.ArgValue(int32(r.mlaRank)), gpu.ArgValue(int32(r.mlaLatDim)),
		gpu.ArgValue(int32(ropePos)), gpu.ArgValue(intl), gpu.ArgValue(r.mlaRopeScale)); err != nil {
		return err
	}
	// 7. MLA Latent Attention: causal dot over latDim, two-pass softmax, value accumulation over rank prefix -> r.wsum
	nKeys := pos + 1
	nWin := nKeys
	if Ly.window > 0 && nKeys > int(Ly.window) {
		nWin = int(Ly.window)
	}
	cfgAttn := LaunchConfig{
		GridX: uint32(r.nH), GridY: 1, GridZ: 1,
		BlockX: 128, BlockY: 1, BlockZ: 1,
		SharedMemBytes: uint32((nWin + 128) * 4),
	}
	if err := r.launch(r.fMlaAttn, cfgAttn,
		Arg(r.qAbs), Arg(r.kc[l]), Arg(r.kvT(l)), Arg(r.wsum),
		gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(r.mlaLatDim)), gpu.ArgValue(int32(r.mlaRank)),
		gpu.ArgValue(int32(pos)), gpu.ArgValue(r.attnScale), gpu.ArgValue(Ly.window), gpu.ArgValue(int32(1))); err != nil {
		return err
	}
	// 8. W_UV Lift: per-head block-diagonal matvec of W_UV with r.wsum -> r.cctx [Ly.qDim]
	elemCountLift := r.nH * r.mlaVHead
	cfgLift := LaunchConfig{GridX: uint32(elemCountLift), GridY: 1, GridZ: 1, BlockX: 64, BlockY: 1, BlockZ: 1}
	if err := r.launch(r.fMlaHeadMV, cfgLift,
		Arg(r.wsum), Arg(Ly.mlaWUV), Arg(r.cctx),
		gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(r.mlaVHead)), gpu.ArgValue(int32(r.mlaRank)),
		gpu.ArgValue(int32(r.mlaRank)), gpu.ArgValue(int32(r.mlaVHead))); err != nil {
		return err
	}
	return nil
}

// segB: context-quant + o-proj (accum into the residual, or sandwich-norm then add) + the MLP up to
// the router readback — the whole dense MLP for a dense layer (no readback gap, segC is nil), or the
// MoE pre-readback half (moeMLPPre / gemma4MoeMLPPre) for a routed layer.
func (r *cudaResident) segB(Ly *cudaLayer, l int, x Buffer) error {
	if Ly.walk.Attn.Gated { // ctx *= sigmoid(gate), before o_proj (matches the CPU qwen35Attention)
		if err := r.launch(r.dnAttnGate, g1cfg(Ly.qDim, 256),
			Arg(r.cctx), Arg(r.dnAGate), gpu.ArgValue(int32(Ly.qDim))); err != nil {
			return err
		}
	}
	if r.floatAct {
		// the attention output is already f32: no quantize, the o-projection reads it as is
	} else if err := r.launch(r.fQ, r.actCfg(256, 256*4, Ly.qDim), Arg(r.cctx), gpu.ArgValue(int32(Ly.qDim)), Arg(r.cq), Arg(r.cSc)); err != nil {
		return err
	}
	if Ly.attnOutUnfused() {
		if r.floatAct {
			if err := r.gemvF(Ly.o, r.cctx, r.oBiasArg(Ly), r.oO, 0); err != nil {
				return err
			}
		} else if err := r.doG(Ly.o, r.cq, r.cSc, r.oBiasArg(Ly), r.oO, 0); err != nil {
			return err
		}
		// LoRA delta added into the RAW o-proj output, BEFORE the post-attn norm below, matching applyLoRA's CPU order (delta added to `out`, caller norms afterward).
		if r.loraLayers != nil {
			if e := r.applyLora(r.loraLayers[l].o, r.cq, r.cSc, Ly.qDim, r.oO); e != nil {
				return e
			}
		}
		if Ly.walk.Attn.OutMode == program.OutPostNormResidual {
			if err := r.normF32(r.oO, Ly.postAttnNorm); err != nil {
				return err
			}
		}
		if r.subCap {
			r.capVec(r.oO, r.subAttnC, l, r.hidden)
		}
		if err := r.launch(r.fRes, g1cfg(r.hidden, 256), Arg(r.x), Arg(r.oO), gpu.ArgValue(int32(r.hidden))); err != nil {
			return err
		}
	} else {
		if r.floatAct {
			if err := r.gemvF(Ly.o, r.cctx, r.oBiasArg(Ly), r.x, 1); err != nil {
				return err
			}
		} else if err := r.doG(Ly.o, r.cq, r.cSc, r.oBiasArg(Ly), r.x, 1); err != nil {
			return err
		}
		if r.loraLayers != nil {
			if e := r.applyLora(r.loraLayers[l].o, r.cq, r.cSc, Ly.qDim, r.x); e != nil {
				return e
			}
		}
	}
	if !Ly.walk.HasFFN {
		return nil // a single-operation block (Nemotron-H attention): the o-projection above was the whole layer
	}
	return r.segBFFN(Ly, l, x)
}

// segBFFN is segB's FFN half, split out so the Gated-DeltaNet mixer can reuse it. The mixer
// replaces everything segB does BEFORE this point (ctx-quant, o-proj, residual) and nothing after
// it: a DeltaNet layer's FFN sub-block is the ordinary one, dense or MoE, and the router readback
// gap + segC that follow in launchToken are likewise unchanged.
func (r *cudaResident) segBFFN(Ly *cudaLayer, l int, x Buffer) error {
	nullBias := ArgNull()
	fw := &Ly.walk.FFN
	switch fw.Kind {
	case program.SublayerGemma4MoE:
		return r.gemma4MoeMLPPre(Ly, l, x)
	case program.SublayerMoE:
		return r.moeMLPPre(Ly, x)
	}
	rawIn := fw.InputMode == program.FFNInRaw
	// This layer's own FFN width: Gemma 4 E-models mix 6144 and 12288, and the
	// scratch is sized to the widest layer, so every launch below takes the layer's width and never the model-level r.inter.
	ffnI := Ly.ffnI
	if ffnI == 0 {
		ffnI = r.inter
	}
	// Dense MLP (whole): no readback gap, so segC is nil for this layer.
	if r.fuseQKV {
		cfg := LaunchConfig{GridX: uint32((2*ffnI + 63) / 64), GridY: 1, GridZ: 1,
			BlockX: 256, BlockY: 1, BlockZ: 1,
			SharedMemBytes: uint32((r.hidden + 256 + r.hidden/4) * 4)}
		// Rows-per-warp variant (fused_gu_rows.cu) for the geometries measured to win; bit-identical to the original (TestFusedGURowsBitIdentical).
		rpw := r.waveRowsPerWarp(2*ffnI, int(cfg.SharedMemBytes))
		if rpw == 0 {
			rpw = fusedGURowsPerWarp(r.hidden, ffnI) // device shape unread: the measured-geometry table
		}
		if rpw != 8 && r.fGURows != (Pipeline{}) {
			rcfg := cfg
			rcfg.GridX = uint32((2*ffnI + 8*rpw - 1) / (8 * rpw))
			if e := r.launch(r.fGURows, rcfg,
				Arg(x), Arg(Ly.postNorm), gpu.ArgValue(int32(r.hidden)), gpu.ArgValue(r.eps),
				gpu.ArgValue(r.addOneArg()),
				Arg(Ly.g.W), Arg(Ly.g.ws16),
				Arg(Ly.u.W), Arg(Ly.u.ws16),
				gpu.ArgValue(int32(ffnI)), gpu.ArgValue(int32(r.hidden/8)), gpu.ArgValue(int32(r.hidden/32)), gpu.ArgValue(int32(rpw)),
				Arg(r.gO), Arg(r.uO)); e != nil {
				return e
			}
		} else if e := r.launch(r.fGU, cfg,
			Arg(x), Arg(Ly.postNorm), gpu.ArgValue(int32(r.hidden)), gpu.ArgValue(r.eps),
			gpu.ArgValue(r.addOneArg()),
			Arg(Ly.g.W), Arg(Ly.g.ws16),
			Arg(Ly.u.W), Arg(Ly.u.ws16),
			gpu.ArgValue(int32(ffnI)), gpu.ArgValue(int32(r.hidden/8)), gpu.ArgValue(int32(r.hidden/32)),
			Arg(r.gO), Arg(r.uO)); e != nil {
			return e
		}
	} else if r.fuseG32 && !rawIn {
		// Per-32 twin of fGU (fused_rms_gu_g32): bit-identical to the unfused chain below.
		rpw := fusedGURowsPerWarp(r.hidden, ffnI)
		cfg := LaunchConfig{GridX: uint32((2*ffnI + 8*rpw - 1) / (8 * rpw)), GridY: 1, GridZ: 1,
			BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: fusedG32Shmem(r.hidden)}
		if e := r.launch(r.fGUg32, cfg,
			Arg(x), Arg(Ly.postNorm), gpu.ArgValue(int32(r.hidden)), gpu.ArgValue(r.eps),
			gpu.ArgValue(r.addOneArg()),
			Arg(Ly.g.W), wkScale(Ly.g), gpu.ArgValue(wkCode(Ly.g.kind)),
			Arg(Ly.u.W), wkScale(Ly.u), gpu.ArgValue(wkCode(Ly.u.kind)),
			gpu.ArgValue(int32(ffnI)), gpu.ArgValue(int32(rpw)),
			Arg(r.gO), Arg(r.uO)); e != nil {
			return e
		}
	} else {
		gq, gSc := r.mq, r.mSc
		switch fw.InputMode {
		case program.FFNInShared:
			// Parallel block (Cohere/Command-R): reuse segA's shared input norm (r.aq/r.aSc) — the MLP
			// consumes the same normed+quantized activation the attention branch computed.
			gq, gSc = r.aq, r.aSc
		default:
			if err := r.emitInputNorm(fw.InputNorm.Form, x, Ly.postNorm, r.mq, r.mSc); err != nil {
				return err
			}
		}
		if err := r.doG(Ly.g, gq, gSc, nullBias, r.gO, 0); err != nil {
			return err
		}
		if err := r.doG(Ly.u, gq, gSc, nullBias, r.uO, 0); err != nil {
			return err
		}
	}
	// Gate/up deltas BEFORE the SwiGLU activation below, matching decoder/mlp.go ("delta into gate/up before the activation").
	if r.loraLayers != nil {
		if r.fuseQKV {
			// fGU computes rmsnorm+quant INTERNALLY (same reasoning as segA's fQKV case) — this
			// branch only runs when InputMode is FFNInNorm (both raw and shared inputs force fuseQKV off at
			// BuildResident), so Ly.postNorm is unambiguously the right weight to materialize with.
			if e := r.norm(x, Ly.postNorm, r.mq, r.mSc); e != nil {
				return e
			}
		}
		LA := &r.loraLayers[l]
		if e := r.applyLora(LA.gate, r.mq, r.mSc, r.hidden, r.gO); e != nil {
			return e
		}
		if e := r.applyLora(LA.up, r.mq, r.mSc, r.hidden, r.uO); e != nil {
			return e
		}
	}
	if err := r.launch(r.fSw, r.actCfg(glueQuantThreads, glueQuantThreads*4, ffnI), Arg(r.gO), Arg(r.uO), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(ffnI)),
		gpu.ArgValue(int32(fw.FFN.Act)), Arg(r.dq), Arg(r.dSc), Arg(r.dScr)); err != nil {
		return err
	}
	if fw.OutMode != program.OutFusedResidual {
		if e := r.doG(Ly.d, r.dq, r.dSc, nullBias, r.dO, 0); e != nil {
			return e
		}
		if r.loraLayers != nil {
			// Added BEFORE any post-MLP norm or deferred residual add — same "delta before any subsequent norm" order.
			if e := r.applyLora(r.loraLayers[l].down, r.dq, r.dSc, ffnI, r.dO); e != nil {
				return e
			}
		}
		if r.subCap {
			r.capVec(r.dO, r.subMLPpreC, l, r.hidden)
		}
		if fw.OutMode == program.OutPostNormResidual {
			if err := r.normF32(r.dO, Ly.postMLPNorm); err != nil {
				return err
			}
		}
		if r.subCap {
			r.capVec(r.dO, r.subMLPC, l, r.hidden)
		}
		if err := r.launch(r.fRes, g1cfg(r.hidden, 256), Arg(x), Arg(r.dO), gpu.ArgValue(int32(r.hidden))); err != nil {
			return err
		}
	} else {
		if e := r.doG(Ly.d, r.dq, r.dSc, nullBias, x, 1); e != nil {
			return e
		}
		if r.loraLayers != nil {
			if e := r.applyLora(r.loraLayers[l].down, r.dq, r.dSc, ffnI, x); e != nil {
				return e
			}
		}
	}
	if Ly.walk.HasPLE && !g4SkipPLEForTest {
		// Gemma 4 E-model per-layer embedding branch (decoder/forward_gemma4.go: gate on the RAW residual, geluTanh(gate)*this layer's input,
		// projection, post-norm, residual add), between the FFN residual and the layer scalar. Existing launches only: quant_vec, the GEMV,
		// glu_quant with a GELU-tanh act (0) and the layer's slice of the row's tail as its `up`, the GEMV, the post-norm, the add.
		if e := r.launch(r.fQ, r.actCfg(256, 256*4, r.hidden), Arg(x), gpu.ArgValue(int32(r.hidden)), Arg(r.mq), Arg(r.mSc)); e != nil {
			return e
		}
		if e := r.doG(Ly.pleGate, r.mq, r.mSc, nullBias, r.pleG, 0); e != nil {
			return e
		}
		if e := r.launch(r.fSw, r.actCfg(glueQuantThreads, glueQuantThreads*4, r.pleP), Arg(r.pleG), Arg(r.x), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(r.hidden+(l+pleLayerShiftForTest)*r.pleP)), gpu.ArgValue(int32(r.pleP)),
			gpu.ArgValue(int32(0)), Arg(r.dq), Arg(r.dSc), Arg(r.dScr)); e != nil {
			return e
		}
		if e := r.doG(Ly.pleProj, r.dq, r.dSc, nullBias, r.dO, 0); e != nil {
			return e
		}
		if e := r.normF32(r.dO, Ly.postPLENorm); e != nil {
			return e
		}
		if e := r.launch(r.fRes, g1cfg(r.hidden, 256), Arg(x), Arg(r.dO), gpu.ArgValue(int32(r.hidden))); e != nil {
			return e
		}
	}
	// Per-layer output scalar (Gemma 4 dense: `if lw.LayerScalar != 0 { h *= lw.LayerScalar }`),
	// applied after the MLP residual add (and after the PLE branch on an E-model).
	if Ly.walk.HasLayerScale && Ly.layerScalar != 0 && !g4DropLayerScalarForTest {
		if err := r.launch(r.fScaleVec, g1cfg(r.hidden, 256), Arg(x), gpu.ArgValue(Ly.layerScalar), gpu.ArgValue(int32(r.hidden))); err != nil {
			return err
		}
	}
	return nil
}

// segC: the post-readback MoE half (expert loop + combine/join). nil for a dense layer.
func (r *cudaResident) segC(Ly *cudaLayer, l int, x Buffer) error {
	switch Ly.walk.FFN.Kind {
	case program.SublayerGemma4MoE:
		return r.gemma4MoeMLPPost(Ly, l, x)
	case program.SublayerMoE:
		return r.moeMLPPost(Ly, x)
	}
	return nil
}

// captureGraphs records each layer's three static segments once (on the executor thread, so the
// thread-local capture matches the replay thread). Off unless r.graphs; incompatible with the subCap/
// g4cap diagnostics (they sync inside a segment, which stream capture forbids) — the caller gates on
// !g4cap and launchToken's useGraphs gates on !subCap. A capture failure is a build error → the
// resident declines to the staged path, never runs a half-captured chain.
func (r *cudaResident) captureGraphs() error {
	for l := range r.layers {
		Ly, ll := &r.layers[l], l
		switch Ly.walk.Mixer {
		case program.MixerDeltaNet:
			// A Gated-DeltaNet layer is MORE graph-static than an attention one: no rope and no attention, so no per-token
			// dynamic uniform. The recurrence's only per-token input is the residual, which is buffer CONTENTS, and replay
			// reads current contents (TestCUDA_graphReplay). So the whole pre-routing half captures as ONE segment, mixer +
			// FFN-pre, and segB stays nil (no attention gap to split around).
			gM, e := r.stream.Capture(func() error {
				if err := r.deltaNetMixer(Ly, ll); err != nil {
					return err
				}
				return r.segBFFN(Ly, ll, r.x)
			})
			if e != nil {
				return fmt.Errorf("layer %d deltanet mixer+ffn: %w", l, e)
			}
			r.layers[l].gSegA = gM
			if Ly.routedFFN() {
				gC, e := r.stream.Capture(func() error { return r.segC(Ly, ll, r.x) })
				if e != nil {
					return fmt.Errorf("layer %d segC: %w", l, e)
				}
				r.layers[l].gSegC = gC
			}
		case program.MixerAttention:
			gA, e := r.stream.Capture(func() error { return r.segA(Ly, ll) })
			if e != nil {
				return fmt.Errorf("layer %d segA: %w", l, e)
			}
			gB, e := r.stream.Capture(func() error { return r.segB(Ly, ll, r.x) })
			if e != nil {
				return fmt.Errorf("layer %d segB: %w", l, e)
			}
			r.layers[l].gSegA, r.layers[l].gSegB = gA, gB
			if Ly.routedFFN() {
				gC, e := r.stream.Capture(func() error { return r.segC(Ly, ll, r.x) })
				if e != nil {
					return fmt.Errorf("layer %d segC: %w", l, e)
				}
				r.layers[l].gSegC = gC
			}
		}
	}
	r.launchErr = nil // capture invoked launch() to RECORD (not execute); clear the sticky accumulator
	return nil
}

// mlaState is latent attention's geometry, kernels and per-token scratch (DeepSeek / Kimi, FeatMLA). Zero unless isMLA.
type mlaState struct {
	fMlaStore, fMlaHeadMV, fMlaQRope, fMlaAttn Pipeline // MLA (mla module, FeatMLA)
	// MLA (DeepSeek / Kimi, FeatMLA) geometry & params
	isMLA         bool
	mlaRank       int // kv_lora_rank
	mlaLatDim     int // kv_lora_rank + qk_rope_head_dim
	mlaQKHead     int // qk_nope + qk_rope
	mlaQKNope     int // qk_nope_head_dim
	mlaQKRope     int // qk_rope_head_dim
	mlaVHead      int // v_head_dim
	mlaQLoRA      int // q_lora_rank (0 if direct q_proj)
	mlaInterleave bool
	mlaRopeScale  float32
	// MLA per-token scratch (allocated only when isMLA).
	mlaQAOut, mlaQAQ, mlaQASc Buffer
	mlaKVDown, qAbs, wsum     Buffer
}
