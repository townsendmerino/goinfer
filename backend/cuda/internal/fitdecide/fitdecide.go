// Package fitdecide holds the CUDA backend's pure memory-fit decisions: what each takes is the free-VRAM reading and plain
// numbers, and what it returns is the verdict, the lines to log and the refusal. It imports only the standard library, so the arithmetic is testable without a device or a model.
package fitdecide

import (
	"errors"
	"fmt"
)

// AllocQuantumBytes is the driver's allocation granularity; package cuda holds it equal to gpudev's by a compile-time check.
const AllocQuantumBytes = 2 << 20

// CtxCapDefault is the resident KV capacity in positions when nothing asks for more.
const CtxCapDefault = 4096

// CapForBytes is the largest capacity in [lo, hi] whose slots slots of pools charge at most budget (ok=false when even lo
// does not). charged is non-decreasing in the capacity, so a search finds the edge.
func CapForBytes(charged func(cap, slots int) int64, budget int64, slots, lo, hi int) (c int, ok bool) {
	if hi < lo || charged(lo, slots) > budget {
		return lo, false
	}
	for hi > lo {
		mid := lo + (hi-lo+1)/2
		if charged(mid, slots) <= budget {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	return lo, true
}

// SlotsFit is the largest n in [1, want] whose n slots of pools charge at most free less reserve; never below 1 (the first
// slot is the KV check's own and is refused there). free is read after the weights are on the device and the pools count KV
// only: a weights-inclusive base compared with it counts the weights twice.
func SlotsFit(charged func(cap, slots int) int64, want int, free int64, cap int, reserve, slotState int64) int {
	n := 1
	for k := 2; k <= want; k++ {
		if charged(cap, k)+int64(k-1)*slotState+reserve > free {
			break
		}
		n = k
	}
	return n
}

// SlotRequirement is the device VRAM n slots/layer actually costs: each buffer rounded up to its own
// quantum, times the layer count. Monotone non-decreasing in n, which is what makes CapSlots'
// binary search valid.
func SlotRequirement(n int, nLayers int64, strides []int64) int64 {
	var perLayer int64
	for _, p := range strides {
		b := int64(n) * p
		perLayer += (b + AllocQuantumBytes - 1) / AllocQuantumBytes * AllocQuantumBytes
	}
	return nLayers * perLayer
}

// SlotMarginBytes is the launch-time headroom the cap must leave free. It is a flat constant that TestAllocFloor
// CHECKS against the measured driver allocation floor, not one derived from what it must cover: the floor is
// unattributed and sits inside it (docs/queue-release.md, the "margin derived rather than asserted" item;
// docs/code-notes/cuda.md#slotMarginBytes).
const SlotMarginBytes = 384 << 20

// CapSlots is the sizing arithmetic, and it is a SEARCH rather than a division. Per-buffer 2 MiB rounding makes
// the requirement a step function of the slot count, and `(free - SlotMarginBytes) / nLayers / perLayer` cannot
// invert a step function: it is wrong precisely at the boundaries the failure lives on (on the real 26B it
// returned 34 slots where the true requirement exceeded free by ~200 MB, because all four buffers tip a quantum
// AT ONCE). A division plus a correction term would reproduce the same class one boundary over, so there is no
// fudge factor here. The requirement is monotone non-decreasing in n, so bisect it. A pure function of its
// inputs, so synthetic free-VRAM figures exercise a branch that in production binds only on models far larger
// than any fixture.
//
// Returns the slot count to use, or decline=true when not even topK fits.
func CapSlots(free, nLayers int64, strides []int64, topK, request int) (slots int, decline bool) {
	if nLayers <= 0 || len(strides) == 0 {
		return request, false
	}
	var any bool
	for _, p := range strides {
		if p > 0 {
			any = true
		}
	}
	if !any {
		return request, false
	}
	fits := func(n int) bool { return SlotRequirement(n, nLayers, strides)+SlotMarginBytes <= free }
	if fits(request) {
		return request, false
	}
	// Largest n in [0, request) that fits. lo always fits (n=0 costs nothing), hi never does.
	lo, hi := 0, request
	for hi-lo > 1 {
		mid := (lo + hi) / 2
		if fits(mid) {
			lo = mid
		} else {
			hi = mid
		}
	}
	if lo < topK {
		return lo, true
	}
	return lo, false
}

// SlotsOut is allocSlots' decision: the slots per layer, the lines the build logs, and the refusal when not even the minimum fits.
type SlotsOut struct {
	Slots int
	Logs  []string
	Err   error
}

// SlotsDecision is allocSlots' decision over the free VRAM it read (ok false: no reading, which keeps want): want slots per layer on
// nLayers routed layers of the given strides, capped to what fits beside extra (a companion attach), refused below topK.
func SlotsDecision(free uint64, ok bool, extra int64, nLayers int, strides []int64, topK, want int) SlotsOut {
	out := SlotsOut{Slots: want}
	if !ok {
		return out
	}
	// CapSlots is the ONE implementation of this arithmetic: a former inline copy let the gate corroborate a
	// parallel copy, so a change to either was uncontradicted by the other (the sibling-drift class,
	// docs/parity-coverage-policy.md).
	//
	// budget, not free, goes into CapSlots: extra reserves room for a companion attach (--drafter) coming after
	// this build (Model.ExtraResidentBytes's doc; docs/tasks/task-fit-to-hardware.md §2). Without it a 26B auto-sized
	// to 31 slots/layer left the drafter no room. 0 when nothing attaches, so budget == free then.
	budget := ReservedBudget(int64(free), extra)
	fit, decline := CapSlots(budget, int64(nLayers), strides, topK, want)
	if decline {
		// FLOOR. topK slots is the minimum that can work (one token's routed set must be simultaneously resident), so if
		// even that does not fit, DECLINE naming the shortfall rather than clamp UP to a figure just computed not to fit:
		// that would allocate it and surface the failure later as CUDA_ERROR_OUT_OF_MEMORY from cuLaunchKernel or an
		// empty generation, neither of which points back here.
		need := SlotRequirement(topK, int64(nLayers), strides)
		if extra > 0 {
			out.Err = fmt.Errorf("expert cache (C′) cannot fit its MINIMUM: top-%d routed experts across %d MoE "+
				"layers need %.2f GB of slots, but only %.2f GB is free (%.2f GB reserved for a companion "+
				"attach, --drafter) — %d slots/layer fit. Free VRAM, lower --ctx, or drop "+
				"GOINFER_MOE_CACHE_EXPERTS and use a card that holds the experts outright",
				topK, nLayers, float64(need)/1e9, float64(free)/1e9, float64(extra)/1e9, fit)
			return out
		}
		out.Err = fmt.Errorf("expert cache (C′) cannot fit its MINIMUM: top-%d routed experts across %d MoE "+
			"layers need %.2f GB of slots, but only %.2f GB is free — %d slots/layer fit. Free VRAM, "+
			"lower --ctx, or drop GOINFER_MOE_CACHE_EXPERTS and use a card that holds the experts outright",
			topK, nLayers, float64(need)/1e9, float64(free)/1e9, fit)
		return out
	}
	if fit < want {
		if extra > 0 {
			out.Logs = append(out.Logs, fmt.Sprintf("[cuda] C′ cache: %d slots/layer would need %.1f GB VRAM but only %.1f GB free "+
				"(%.1f GB of %.1f GB total reserved for a companion attach, --drafter) — capping to %d (%.1f GB)\n",
				want, float64(SlotRequirement(want, int64(nLayers), strides))/1e9,
				float64(budget)/1e9, float64(extra)/1e9, float64(free)/1e9,
				fit, float64(SlotRequirement(fit, int64(nLayers), strides))/1e9))
		} else {
			out.Logs = append(out.Logs, fmt.Sprintf("[cuda] C′ cache: %d slots/layer would need %.1f GB VRAM but only %.1f GB free — "+
				"capping to %d (%.1f GB)\n", want,
				float64(SlotRequirement(want, int64(nLayers), strides))/1e9, float64(free)/1e9,
				fit, float64(SlotRequirement(fit, int64(nLayers), strides))/1e9))
		}
		out.Slots = fit
	}
	return out
}

// ReservedBudget is what allocSlots' elastic expert-cache sizing gets to work with once a companion allocation
// (extraBytes: a --drafter attaching after this build, Model.ExtraResidentBytes's doc) has reserved its share of
// free VRAM. Split out, like FitsWeightsBudget below, so the clamp-at-zero (extraBytes can exceed free on a card
// too small for the pairing, and CapSlots must then see 0, not a negative budget) is unit-testable without a
// device. extraBytes <= 0 returns free unchanged.
func ReservedBudget(free, extraBytes int64) int64 {
	b := free - extraBytes
	if b < 0 {
		return 0
	}
	return b
}

// FitsWeightsBudget is checkWeightsFit's arithmetic alone, split out so it is unit-testable with
// synthetic numbers rather than a checkpoint large enough to swing a real device's verdict — same
// reasoning as metal/backend.go's fitsResidentBudget, which this mirrors. Unknown inputs (need or
// free <= 0) always fit: the guard's failure mode must be letting a doomed load through, never
// refusing one that would have run.
func FitsWeightsBudget(need, free, margin int64) bool {
	if need <= 0 || free <= 0 {
		return true
	}
	return need+margin <= free
}

// WeightsFitDecision is checkWeightsFit's decision over the free VRAM it read (ok false: no reading, which passes, as the
// allocation itself still errors if it truly cannot fit). need is the dense weights' device bytes
// (decoder.Model.ResidentDenseWeightBytesFor); extra is a companion attach (a --drafter's weights,
// decoder.Model.ExtraResidentBytes), which lands on this device after BuildResident returns and so is priced here too.
func WeightsFitDecision(need, extra, free int64, ok bool, margin int64) error {
	if !ok || FitsWeightsBudget(need+extra, free, margin) {
		return nil
	}
	if extra > 0 {
		return fmt.Errorf("cuda: dense weights need %.2f GB plus %.2f GB reserved for a companion "+
			"attach (--drafter) but only %.2f GB is free on the device (plus %.0f MB reserved for "+
			"driver and decode scratch) — use a smaller/more-quantized model, or the staged/CPU path",
			float64(need)/1e9, float64(extra)/1e9, float64(free)/1e9, float64(margin)/(1<<20))
	}
	return fmt.Errorf("cuda: dense weights need %.2f GB but only %.2f GB is free on the device "+
		"(plus %.0f MB reserved for driver and decode scratch) — use a smaller/more-quantized "+
		"model, or the staged/CPU path",
		float64(need)/1e9, float64(free)/1e9, float64(margin)/(1<<20))
}

// KVFitIn is what checkKVFits decides from besides the free VRAM: the planned context and whether it was asked for, the KV
// slots requested, the device price of slots slots of pools at a capacity (Charged, non-decreasing in both) and the number of
// layers it covers (Layers, for the refusal's per-position figure), the companion reserve, the context margin and a further
// slot's recurrent state. NoTrim turns the one-slot context trim off (a test plant).
type KVFitIn struct {
	CtxCap     int
	Explicit   bool
	SlotsReq   int
	Charged    func(cap, slots int) int64
	Layers     int
	NoTrim     bool
	ExtraBytes int64
	Margin     int64
	SlotState  int64 // each KV slot past the first's recurrent state (recurrentStateCharged); 0 for a model with none
}

// KVFitOut is checkKVFits' decision: the context the pools are sized for, the slots granted (>= 1), the lines the build logs
// (each ends in a newline) and the refusal, if any.
type KVFitOut struct {
	CtxCap int
	Slots  int
	Logs   []string
	Err    error
}

// KVFitDecision is checkKVFits' decision over the free VRAM it read (ok false: no reading, which grants one slot at the
// planned context and refuses nothing).
func KVFitDecision(in KVFitIn, free uint64, ok bool) KVFitOut {
	out := KVFitOut{CtxCap: in.CtxCap, Slots: 1}
	if !ok {
		// No MemInfo ⇒ no fit check possible. Do not fail the load on that: the default cap has
		// always been allocated without one, and a hard failure here would regress every driver
		// that does not report memory. The allocation itself still errors if it truly cannot fit.
		// MC1's further slots are not allocated blind: one slot, as before.
		return out
	}
	need := in.Charged(out.CtxCap, 1)
	// extraBytes reserves room for a companion attach (--drafter) coming after this build — see
	// cudaResident.extraBytes. 0 for every load without one, so this is unchanged then.
	reserve := in.ExtraBytes + in.Margin
	if !in.Explicit && out.CtxCap > CtxCapDefault {
		// An unpinned context that even ONE slot of KV overshoots (by the build's own scratch, which ctxForSlots cannot
		// see) gives up positions, never below CtxCapDefault, instead of declining the whole resident path to the CPU
		// with a log line. Only the multi-slot arm below used to shrink to what the card holds, so a plain decoder.Load of
		// a 7B or Gemma-3 on an 8 GB card fell to the CPU (docs/code-notes/cuda.md#checkKVFits.trim). serve plans its own
		// context and never hit it.
		c, fits := 0, false
		if need+reserve > int64(free) && !in.NoTrim {
			c, fits = CapForBytes(in.Charged, int64(free)-reserve, 1, CtxCapDefault, out.CtxCap-1)
		}
		if fits {
			out.Logs = append(out.Logs, fmt.Sprintf("cuda: resident context %d, trimmed from %d at the build, so one KV slot fits beside the weights (%.0f MB free, less %.0f MB reserved)\n",
				c, out.CtxCap, float64(free)/(1<<20), float64(reserve)/(1<<20)))
			out.CtxCap, need = c, in.Charged(c, 1)
		}
	}
	if need+reserve <= int64(free) {
		// MC1: as many of the requested KV slots as fit the same budget; the slots share each layer's pool buffer.
		out.Slots = SlotsFit(in.Charged, in.SlotsReq, int64(free), out.CtxCap, reserve, in.SlotState)
		if out.Slots < in.SlotsReq && !in.Explicit && out.CtxCap > CtxCapDefault {
			// Slots before context (owner decision): an unpinned context gives up positions, never below CtxCapDefault,
			// until every requested slot fits. ctxForSlots planned this with Plan, which prices the KV linearly, so its choice can
			// be a few positions long; this trims it against the real free VRAM, before any KV exists. Nothing allocated so far
			// depends on ctxCap exactly: the split-KV score scratch is indexed by the attended span, so a smaller cap uses less of it.
			c := CtxCapDefault
			if cc, fits := CapForBytes(in.Charged, int64(free)-reserve-int64(in.SlotsReq-1)*in.SlotState, in.SlotsReq, CtxCapDefault, out.CtxCap); fits {
				c = cc
			}
			if c < out.CtxCap {
				out.Logs = append(out.Logs, fmt.Sprintf("cuda: resident context %d, trimmed from %d at the build, so the %d requested KV slots "+
					"fit beside the weights (%.0f MB free, less %.0f MB reserved)\n",
					c, out.CtxCap, in.SlotsReq, float64(free)/(1<<20), float64(reserve)/(1<<20)))
				out.CtxCap, need = c, in.Charged(c, 1)
				out.Slots = SlotsFit(in.Charged, in.SlotsReq, int64(free), out.CtxCap, reserve, in.SlotState)
			}
		}
		if out.Slots < in.SlotsReq {
			out.Logs = append(out.Logs, fmt.Sprintf("cuda: %d resident KV slots of %d requested — one slot's pools cost %.0f MB at the resident "+
				"context %d, and free VRAM beside the weights (%.0f MB, less %.0f MB reserved) allows %d\n",
				out.Slots, in.SlotsReq, float64(need)/(1<<20), out.CtxCap, float64(free)/(1<<20),
				float64(reserve)/(1<<20), out.Slots))
		}
		return out
	}
	perPos := float64(need) / float64(max(out.CtxCap, 1)) / 1024
	msg := fmt.Sprintf("cuda: resident context %d positions needs %.2f GB of KV "+
		"(%.1f KB/position across %d layers) but only %.2f GB is free on the device beside the weights "+
		"(plus %.0f MB reserved for driver and decode scratch) — lower the serve context setting (-ctx), "+
		"or use a smaller/more-quantized model",
		out.CtxCap, float64(need)/1e9, perPos, in.Layers, float64(free)/1e9, float64(in.Margin)/(1<<20))
	if in.ExtraBytes > 0 {
		msg = fmt.Sprintf("cuda: resident context %d positions needs %.2f GB of KV "+
			"(%.1f KB/position across %d layers) plus %.2f GB reserved for a companion attach "+
			"(--drafter) but only %.2f GB is free on the device beside the weights (plus %.0f MB "+
			"reserved for driver and decode scratch) — lower the serve context setting (-ctx), or "+
			"use a smaller/more-quantized model",
			out.CtxCap, float64(need)/1e9, perPos, in.Layers, float64(in.ExtraBytes)/1e9,
			float64(free)/1e9, float64(in.Margin)/(1<<20))
	}
	if in.Explicit {
		out.Err = &KVWontFitError{Msg: msg}
		return out
	}
	// Default cap: keep the historical decline. Strip the sentinel so BuildResident treats it as
	// an ordinary "cannot host this here" and the staged path takes over, as it always has.
	if in.ExtraBytes > 0 {
		out.Err = fmt.Errorf("cuda: default resident context %d positions does not fit (%.2f GB of KV "+
			"plus %.2f GB reserved for a companion attach, --drafter, %.2f GB free) — staged path",
			out.CtxCap, float64(need)/1e9, float64(in.ExtraBytes)/1e9, float64(free)/1e9)
		return out
	}
	out.Err = fmt.Errorf("cuda: default resident context %d positions does not fit (%.2f GB of KV, %.2f GB free) — staged path",
		out.CtxCap, float64(need)/1e9, float64(free)/1e9)
	return out
}

// ErrKVWontFit marks the one setup failure that must NOT degrade quietly to the staged path: an
// explicitly configured resident context whose KV does not fit. BuildResident turns it into a hard
// startup error instead of a decline. An UNCONFIGURED (default-cap) miss stays a decline, because
// that is the historical behaviour for "this device cannot host this model".
var ErrKVWontFit = errors.New("resident KV does not fit in device memory")

// KVWontFitError carries the full operator-facing message while still matching ErrKVWontFit under
// errors.Is. A plain %w wrap would append the sentinel's text to a message that already says all of
// this, so the reader sees it twice — the classification must not leak into the prose.
type KVWontFitError struct{ Msg string }

func (e *KVWontFitError) Error() string        { return e.Msg }
func (e *KVWontFitError) Is(target error) bool { return target == ErrKVWontFit }
