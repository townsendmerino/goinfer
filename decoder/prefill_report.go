package decoder

import (
	"context"
	"sync"
)

// PrefillReport records how one request's prompt prefilled, for the caller's request log. A caller attaches one with
// WithPrefillReport; while a generation on that ctx prefills, the decoder, and a backend through NotePrefillFallback,
// record a declined batched prefill in it. The first fallback noted is the one kept.
type PrefillReport struct {
	mu           sync.Mutex
	path, reason string
}

type prefillReportKey struct{}

// WithPrefillReport returns ctx carrying a fresh report, and the report.
func WithPrefillReport(ctx context.Context) (context.Context, *PrefillReport) {
	r := &PrefillReport{}
	return context.WithValue(ctx, prefillReportKey{}, r), r
}

// NotePrefillFallback records on ctx's report, when it carries one, that the batched prefill declined for reason and
// the prompt ran on path instead: "sequential" (the decoder's token-by-token loop) or a backend's own pass. tracked says
// whether ctx carries a report; first, whether this call recorded the request's first fallback, so a backend that
// announces its fallback can do so once per request.
func NotePrefillFallback(ctx context.Context, path, reason string) (first, tracked bool) {
	if ctx == nil {
		return false, false
	}
	r, _ := ctx.Value(prefillReportKey{}).(*PrefillReport)
	if r == nil {
		return false, false
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.path == "" {
		r.path, r.reason = path, reason
		return true, true
	}
	return false, true
}

// Fallback is the path the prompt fell back to and why; both are "" when no batched prefill declined.
func (r *PrefillReport) Fallback() (path, reason string) {
	if r == nil {
		return "", ""
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.path, r.reason
}
