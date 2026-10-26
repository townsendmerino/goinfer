//go:build cuda

package cuda

import "github.com/townsendmerino/goinfer/decoder"

// The context margin computed from its parts (docs/tasks/task-kernel-survey-2026-10.md, "KS-H6 build"): what the build allocates
// after the KV pools, one 512-row prefill pass, the allocation floor and a decode allowance, beside the KV priced as the card
// charges it. slotMarginBytes is not this.

// allocFloorBytes is the free VRAM cuMemAlloc will not hand out at any request size: the top of TestAllocFloor's recorded band
// (148–155 MiB, docs/code-notes/cuda.md, "TestAllocFloor").
const allocFloorBytes = 155 << 20

// decodeAllowanceBytes is decode's transient allocations beyond the prefill pass: H6-R measured none on a dense decode
// (docs/tasks/task-kernel-survey-2026-10.md, "H6-R read").
const decodeAllowanceBytes = 16 << 20

// grantReserveBytes is the reserve the runtime grants (kvGrantRollback, allocRecurrentCheckpoint) keep beside two prefill passes.
// It is the flat figure the context margin used to be; the owner's KS-H6 decision did not change these grants.
const grantReserveBytes = 384 << 20

// chargedBytes is what the device charges for one allocation of n bytes: the request, rounded up to the allocation quantum from a
// quantum upward (allocRoundSlack).
func chargedBytes(n int64) int64 { return n + allocRoundSlack(n) }

// kvChargedBytes is the device bytes kvAllocPools takes for slots slots at capacity cap: every owning layer's K and V pool (one
// latent pool for MLA), each holding its group's blocks for every slot plus the null block and charged as the device rounds it,
// and the slots' block tables. It mirrors kvAllocPools and kvSlotBlocks; TestKSH6C_marginPriced holds it to the pools' allocations.
func kvChargedBytes(cap, slots int, layers []cudaLayer) int64 {
	full := int64(slots*kvBlocksFor(cap) + 1)
	var n int64
	for i := range layers {
		Ly := &layers[i]
		if Ly.kvShared || Ly.noKV() || Ly.kvDim == 0 {
			continue
		}
		elem := int64(4)
		if Ly.kvHalf {
			elem = 2
		}
		blocks := full
		if Ly.kvWin {
			blocks = int64(slots*kvWindowBlocks(cap, int(Ly.window)+kvWindowSlack) + 1)
		}
		buf := chargedBytes(blocks * kvBlock * int64(Ly.kvDim) * elem)
		if Ly.isMLA() {
			n += buf
			continue
		}
		n += 2 * buf
	}
	return n + chargedBytes(int64(slots*kvGroups*kvBlocksFor(cap))*4)
}

// kvChargedCapForBytes is the largest capacity in [lo, hi] whose slots slots of pools charge at most budget (ok=false when even lo
// does not). kvChargedBytes is non-decreasing in the capacity, so a search finds the edge.
func kvChargedCapForBytes(budget int64, slots, lo, hi int, layers []cudaLayer) (c int, ok bool) {
	if hi < lo || kvChargedBytes(lo, slots, layers) > budget {
		return lo, false
	}
	for hi > lo {
		mid := lo + (hi-lo+1)/2
		if kvChargedBytes(mid, slots, layers) <= budget {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	return lo, true
}

// kvChargedSlotsFit is the largest n in [1, want] whose n slots of pools charge at most free less reserve; never below 1 (the first
// slot is checkKVFits' own and is refused there). free is read after the weights are on the device and the pools count KV only:
// a weights-inclusive base compared with it counts the weights twice.
func kvChargedSlotsFit(want int, free int64, cap int, layers []cudaLayer, reserve int64) int {
	n := 1
	for k := 2; k <= want; k++ {
		if kvChargedBytes(cap, k, layers)+reserve > free {
			break
		}
		n = k
	}
	return n
}

// kvRoundingBound is the most the device's rounding can add to the linear KV price the plan uses (decoder's per-position KV):
// one allocation quantum per pool buffer, and one for the tables. owners and mla count the layers that own a K/V pool pair and
// those that own one latent pool.
func kvRoundingBound(owners, mla int) int64 {
	return int64(2*owners+mla+1) * allocQuantumBytes
}

// ctxMarginFor is the context margin at load for a context of ctx positions (docs/tasks/task-kernel-survey-2026-10.md, "KS-H6
// build"): the build's own allocations after the KV pools (postPoolBytes), one 512-row prefill pass at that context, the
// allocation floor and the decode allowance. BuildResident computes it once, on the planned geometry (planLayerGeometry) before the
// context is planned, and keeps it in ctxMarginBytes: the plan, checkKVFits, checkWeightsFit and their declines read that one
// figure. TestKSH6C_marginPriced holds it to the built resident's.
func (r *cudaResident) ctxMarginFor(ctx int) int64 {
	return r.postPoolBytes() + r.PrefillScratchBytes(prefillDefaultChunk) + r.PrefillSpanScratchBytes(prefillDefaultChunk, ctx) +
		allocFloorBytes + decodeAllowanceBytes
}

// postPoolBytes is what the setup job allocates after the KV pools, short of the grants that size themselves from what is left (the
// recurrent checkpoint, the expert slots, the windowed grant), as the card charges it: buffers under the allocation quantum come out
// of the driver's 2 MiB pages, so their sum is rounded up to whole pages, and each larger one is rounded up to the quantum. It
// mirrors the setup job's allocations from kvAllocPoolsFit to allocRecurrentCheckpoint. The float-activation buffers are priced on
// every Gemma 4 MoE build, the per-weight checks of floatActDecline being unknown before the upload. TestKSH6C_marginPriced holds it
// to the allocations.
func (r *cudaResident) postPoolBytes() int64 {
	var small, large int64
	f := func(n int) {
		if b := int64(max(n, 1)) * 4; b < allocQuantumBytes {
			small += b
		} else {
			large += chargedBytes(b)
		}
	}
	maxQ, maxHd, maxI, vNorm := 0, 0, r.inter, false
	for l := range r.layers {
		Ly := &r.layers[l]
		maxQ, maxHd, maxI = max(maxQ, Ly.qDim), max(maxHd, Ly.hd), max(maxI, Ly.ffnI)
		vNorm = vNorm || Ly.vNorm
	}
	H := r.hidden
	f(maxQ)
	f(r.actScaleLen(maxQ))
	f(maxQ / 4)
	if vNorm {
		f(maxHd)
	}
	f(H)
	f(r.actScaleLen(H))
	f(H / 4)
	if r.layerNorm {
		f(H)
	}
	if maxI > 0 {
		f(maxI)
		f(maxI)
		f(r.actScaleLen(maxI))
		f(maxI)
		f(maxI / 4)
	}
	if r.moe {
		f(r.nE)
		f(r.topK)
		f(r.topK)
		if r.cacheExperts {
			f(r.topK)
		}
		f(2 * r.moeInter)
		f(1)
		f(r.moeInter)
		f(r.moeInter / 4)
		if r.sharedInter > 0 {
			f(2 * r.sharedInter)
			f(1)
			f(r.sharedInter)
			f(r.sharedInter / 4)
			f(H)
			f(1)
		}
	}
	if r.gemma4Moe {
		f(H)
		f(H)
		f(H)
		if !floatActOffForTest && !r.eModel && !r.actG32 {
			f(H)
			f(max(maxI, 1))
			f(r.moeInter)
		}
	}
	f(H)
	f(r.vocab)
	f(1)
	f(1)
	f(2*topkMaxK + 2)
	f(topkCap)
	f(topkCap)
	f(1)
	f(topkThreads)
	f(1)
	f(1)
	nGb := gumbelBlocks(r.vocab)
	f(nGb)
	f(nGb)
	f(1)
	return (small+allocQuantumBytes-1)/allocQuantumBytes*allocQuantumBytes + large
}

// planLayerGeometry is the per-layer geometry the margin reads, planned from the same accessors and host plans the setup job
// builds its layers from (the mixer, head dims and widths, the window, the dense FFN width, KV sharing, K=V, v_norm and the KV
// precision), so the margin exists before the context is planned. The setup job replaces r.layers with the built layers.
func planLayerGeometry(m *decoder.Model, hls []hlayer, nH int, r *cudaResident) []cudaLayer {
	ls := make([]cudaLayer, len(hls))
	for l := range hls {
		h := &hls[l]
		L := cudaLayer{idx: l, mixer: h.mixer, ffn: h.ffn, window: h.window}
		if h.mixer == decoder.MixerNone {
			L.ffnI = h.u.N
		} else if !h.isMoE && h.ffn {
			L.ffnI = h.g.N
		}
		switch h.mixer {
		case decoder.MixerDeltaNet, decoder.MixerMamba, decoder.MixerNone:
		case decoder.MixerMLA:
			L.hd, L.nKV, L.qDim, L.kvDim = r.mlaQKHead, 1, nH*r.mlaVHead, r.mlaLatDim
		default:
			L.hd, L.nKV = m.HeadDimAtResident(l), m.KVHeadsAtResident(l)
			L.qDim, L.kvDim = nH*L.hd, L.nKV*L.hd
			if src := m.KVSrcAtResident(l); src != l {
				L.kvShared, L.kvSrc = true, src
			}
			L.kEqV = m.VFromKResident(l) && !L.kvShared
			L.vNorm = !L.kvShared && (L.kEqV || m.IsGemma4Resident())
		}
		L.kvHalf = r.kvF16 && !L.noKV() && !L.kvShared
		ls[l] = L
	}
	return ls
}

// kvLayerCounts is the number of layers owning a K/V pool pair and the number owning one latent pool (MLA), for kvRoundingBound.
func kvLayerCounts(layers []cudaLayer) (owners, mla int) {
	for l := range layers {
		Ly := &layers[l]
		switch {
		case Ly.kvShared || Ly.noKV() || Ly.kvDim == 0:
		case Ly.isMLA():
			mla++
		default:
			owners++
		}
	}
	return owners, mla
}

// marginBesidePass is the context margin less the 512-row pass it holds: what the width chooser keeps free beside the wider pass it
// prices instead (choosePrefillWidth). It reads the built resident's figures, which equal the planned ones (TestKSH6C_marginPriced).
func (r *cudaResident) marginBesidePass() int64 {
	return r.ctxMarginBytes - r.PrefillScratchBytes(prefillDefaultChunk) - r.PrefillSpanScratchBytes(prefillDefaultChunk, r.ctxMarginCtx)
}

// ctxMargin is the context margin this build keeps (ctxMarginFor at the planned context).
func (r *cudaResident) ctxMargin() int64 { return r.ctxMarginBytes }
