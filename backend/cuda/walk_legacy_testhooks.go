//go:build cuda && goinfer_testhooks

package cuda

import (
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// legacyLayer is launchToken's layer loop body as it was before the walk (OP4cu-b2), kept for B2-1's comparison and B2-6's A/B
// only: it dispatches on the layer's mixer where the walk runs its steps.
func legacyLayer(r *cudaResident, Ly *cudaLayer, l, pos, ropePos int, qTempScale float32, gA, gB, gC bool) error {
	if Ly.mixer == decoder.MixerDeltaNet {
		// Gated-DeltaNet mixer: replaces segA + rope + attention + o-proj (no KV cache, nothing positional: the state IS
		// the history). The FFN sub-block and the router-readback gap below are the ordinary ones, so this rejoins the
		// shared path at segBFFN. Captured as ONE segment (gSegA) when graphs are on: with no rope/attention gap there is
		// nothing dynamic to split around.
		if gA {
			if e := Ly.gSegA.Replay(); e != nil {
				return e
			}
			if r.graphsSync {
				if e := r.stream.Sync(); e != nil {
					return e
				}
			}
		} else {
			if e := r.deltaNetMixer(Ly, l); e != nil {
				return e
			}
			if e := r.segBFFN(Ly, l, r.x); e != nil {
				return e
			}
		}
		if e := r.layerTail(Ly, l, gC, r.x); e != nil {
			return e
		}
		return nil
	}
	if Ly.mixer == decoder.MixerMamba || Ly.mixer == decoder.MixerNone {
		// A Nemotron-H block carries one operation. Mamba-2 replaces segA + attention + o-proj (no KV, no position); a feed-forward-only
		// block is the up GEMV, relu2_quant and the down GEMV. Neither has a router readback gap, so both go straight to the layer tail.
		var e error
		if Ly.mixer == decoder.MixerMamba {
			e = r.mambaMixer(Ly)
		} else {
			e = r.nonGatedMLP(Ly)
		}
		if e != nil {
			return e
		}
		if e := r.layerTail(Ly, l, gC, r.x); e != nil {
			return e
		}
		return nil
	}
	if Ly.mixer == decoder.MixerMLA {
		if e := r.segAMLA(Ly, l, pos, ropePos); e != nil {
			return e
		}
	} else {
		// segA: QKV proj + qk/v-norm (pre-RoPE, static).
		if gA {
			if e := Ly.gSegA.Replay(); e != nil {
				return e
			}
			if r.graphsSync {
				if err := r.stream.Sync(); err != nil {
					return err
				}
			}
		} else if e := r.segA(Ly, l); e != nil {
			return e
		}
		// --- dynamic gap: rope_kv + attention (decodeAttnGap; MC3 on CUDA runs it per sequence inside a step).
		if e := r.decodeAttnGap(Ly, l, pos, ropePos, qTempScale); e != nil {
			return e
		}
	}
	if r.subCap { // pre-o-proj attention context (qDim), before quant — the cross-box discriminator (live path only)
		r.capVec(r.cctx, r.subCtxC, l, Ly.qDim)
	}
	// segB: ctx-quant + o-proj + MLP up to the router readback (whole dense MLP for a dense layer).
	if gB {
		if e := Ly.gSegB.Replay(); e != nil {
			return e
		}
		if r.graphsSync {
			if err := r.stream.Sync(); err != nil {
				return err
			}
		}
	} else if e := r.segB(Ly, l, r.x); e != nil {
		return e
	}
	// --- dynamic gap: clear the g4moe accumulator (H2D), then, if caching, read the routing back
	// to the host and DMA the routed experts into their VRAM slots (D2H+H2D). Host copies, not
	// graph-capturable; synchronous, so the slots are filled before segC replays.
	if e := r.layerTail(Ly, l, gC, r.x); e != nil {
		return e
	}
	return nil
}

// setWalkPlantForTest plants p in the walk for the rest of tb.
func setWalkPlantForTest(tb testing.TB, p walkPlantKind) {
	tb.Helper()
	walkPlant = p
	tb.Cleanup(func() { walkPlant = walkPlantNone })
}
