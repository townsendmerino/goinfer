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
		"decoder: this request needs ~%.2f GB (KV %.2f GB + prefill scratch %.2f GB for %d prompt + %d max_tokens positions) "+
			"but only %.2f GB of this machine's %.2f GB currently-available memory would be left as a safety margin — "+
			"rejected before prefill rather than paging.\n"+
			"  Send a smaller prompt, lower max_tokens, pass -ctx to cap the context, close other "+
			"applications to free memory, or -stream-weights to free more of the budget for KV.\n"+
			"  Set GOINFER_NO_FIT_GUARD=1 to allow it anyway if this machine really has the room",
		float64(need)/gb, float64(r.KVBytes)/gb, float64(r.ScratchBytes)/gb, r.PromptTokens, r.MaxTokens,
		float64(remaining)/gb, float64(avail)/gb)
}

// HostBudget is the resident budget on m: the policy's fraction of available host memory, 0 when unknown.
func HostBudget(m Machine, p Policy) int64 {
	return int64(float64(m.Available(HostAnon)) * p.WeightsFraction)
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
