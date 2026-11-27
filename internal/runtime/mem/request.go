package mem

import "fmt"

// Request is one generation's memory beyond a resident model: its KV (0 when a resident path holds it) and its prefill
// scratch, for PromptTokens + MaxTokens positions, on a server running up to Share generations at once.
type Request struct {
	KVBytes, ScratchBytes   int64
	PromptTokens, MaxTokens int
	Share                   int // concurrent generations sharing the margin; 1 or less is one
}

// AdmitRequest refuses r, with the message a client sees, when its KV and scratch exceed its share of the policy's fraction
// of available memory. "" admits it, and so does an unknown reading.
func AdmitRequest(r Request, m Machine, p Policy) string {
	avail := m.Available(HostAnon)
	if avail <= 0 {
		return ""
	}
	remaining := int64(float64(avail) * p.RequestFraction / float64(max(1, r.Share)))
	need := r.KVBytes + r.ScratchBytes
	if need <= remaining {
		return ""
	}
	return fmt.Sprintf(
		"decoder: this request needs ~%s (KV %s + prefill scratch %s for %d prompt + %d max_tokens positions) "+
			"but only %s of this machine's %s currently-available memory would be left as a safety margin — "+
			"rejected before prefill rather than paging.\n"+
			"  Send a smaller prompt, lower max_tokens, pass -ctx to cap the context, close other "+
			"applications to free memory, or -stream-weights to free more of the budget for KV.\n"+
			"  Set GOINFER_NO_FIT_GUARD=1 to allow it anyway if this machine really has the room",
		GiB(need), GiB(r.KVBytes), GiB(r.ScratchBytes), r.PromptTokens, r.MaxTokens,
		GiB(remaining), GiB(avail))
}

// HostBudget is the resident budget on m: the policy's fraction of available host memory, 0 when unknown. Its callers
// outside mem are listed as RAMShare's are.
func HostBudget(m Machine, p Policy) int64 {
	return int64(float64(m.Available(HostAnon)) * p.WeightsFraction)
}

// RAMShare is the resident budget a unified-memory backend's guards and slot clamps price against: WeightsFraction of ram,
// the machine's physical RAM in bytes (HostBudget's arithmetic over physical rather than available memory). Calling it
// applies the threshold, so the files outside mem that may are listed (TestNoGuardThresholdOutsideMem's ruleCallers).
func RAMShare(ram int64) int64 { return int64(float64(ram) * WeightsFraction) }

// WeightsPercent is WeightsFraction as the percentage a refusal or a clamp's log line prints; its callers outside mem are
// listed as RAMShare's are.
func WeightsPercent() float64 { return WeightsFraction * 100 }

// KVSlotsWithin is the largest slot count up to want whose resident build, base bytes for the first slot plus perSlot
// for each further one, fits budget, and never below 1: pricing the first slot is the caller's guard's job.
func KVSlotsWithin(want int, budget, base, perSlot int64) int {
	n := max(1, want)
	for n > 1 && base+int64(n-1)*perSlot > budget {
		n--
	}
	return n
}

// WeightCacheBudget is a weight cache's byte budget: requested when it is positive, else half of available host memory, 0
// when that is unknown too.
func WeightCacheBudget(requested int64, m Machine) int64 {
	if requested > 0 {
		return requested
	}
	if a := m.Available(HostAnon); a > 0 {
		return a / 2
	}
	return 0
}

// SnapshotHostBudget is the prefix-snapshot store's default host budget (-kv-host-bytes auto) for a machine with ram bytes
// of physical RAM under p, and the startup-banner phrase explaining how it was reached. Calling it applies KVHostFraction
// and KVHostCap, so its callers outside mem are listed in TestNoGuardThresholdOutsideMem's ruleCallers.
func SnapshotHostBudget(ram int64, p Policy) (int64, string) {
	if ram <= 0 || p.KVHostFraction <= 0 {
		return 0, "auto needs the machine's physical RAM, which is unknown here"
	}
	b := min(ram/int64(p.KVHostFraction), p.KVHostCap)
	return b, fmt.Sprintf("auto: 1/%d of %s of RAM, at most %s", p.KVHostFraction, GiB(ram), GiB(p.KVHostCap))
}
