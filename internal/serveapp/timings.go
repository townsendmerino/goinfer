package serveapp

import (
	"math"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// turnTiming is one request's X1 timings (docs/tasks/task-turn-telemetry-2026-09.md, "X1, registered 2026-10-14"):
// prepare makes it, drive and driveVL start it at the generation's start (admission granted), streamTokens notes the
// first token and, once the stream has closed, takes the decoder's stage timings. A nil *turnTiming reports nothing.
type turnTiming struct {
	prepared, started, first time.Time
	gen                      *decoder.TurnTiming
}

func newTurnTiming() *turnTiming { return &turnTiming{prepared: time.Now()} }

func (t *turnTiming) begin() {
	if t != nil && t.started.IsZero() {
		t.started = time.Now()
	}
}

func (t *turnTiming) firstToken() {
	if t != nil && t.first.IsZero() {
		t.first = time.Now()
	}
}

func (t *turnTiming) finish(gen *decoder.Generation) {
	if t != nil && gen != nil {
		t.gen = gen.Timing
	}
}

// timingsJSON is `goinfer_timings` inside a response's usage (Experimental). Every field is absent when the turn did
// not measure it, never zero for "unknown".
type timingsJSON struct {
	QueueMS             *float64 `json:"queue_ms,omitempty"`
	PrefillMS           *float64 `json:"prefill_ms,omitempty"`
	PrefillTokens       *int     `json:"prefill_tokens,omitempty"`
	PrefillReusedTokens *int     `json:"prefill_reused_tokens,omitempty"`
	TTFTMS              *float64 `json:"ttft_ms,omitempty"`
	DecodeMS            *float64 `json:"decode_ms,omitempty"`
	DecodeTokens        *int     `json:"decode_tokens,omitempty"`
	HostGapMS           *float64 `json:"host_gap_ms,omitempty"`
	DecodeOverlapped    *bool    `json:"decode_overlapped,omitempty"`
}

func msOf(d time.Duration) *float64 {
	v := math.Round(float64(d.Nanoseconds())/1e3) / 1e3
	return &v
}

func ptr[T any](v T) *T { return &v }

// json is the usage block, or nil when the turn never started.
func (t *turnTiming) json() *timingsJSON {
	if t == nil || t.started.IsZero() {
		return nil
	}
	out := &timingsJSON{QueueMS: msOf(t.started.Sub(t.prepared))}
	if !t.first.IsZero() {
		out.TTFTMS = msOf(t.first.Sub(t.started))
	}
	if g := t.gen; g != nil {
		out.PrefillMS, out.PrefillTokens, out.PrefillReusedTokens = msOf(g.Prefill), ptr(g.PrefillTokens), ptr(g.PrefillReused)
		out.DecodeTokens = ptr(g.DecodeTokens)
		if g.DecodeTokens > 0 {
			out.DecodeMS, out.HostGapMS = msOf(g.Decode), msOf(max(0, g.Decode-g.ForwardWait))
			out.DecodeOverlapped = ptr(g.Overlapped)
		}
	}
	return out
}

// addTo puts the block into a route's usage map, or leaves the key out when the turn never started.
func (t *turnTiming) addTo(usage map[string]any) {
	if j := t.json(); j != nil {
		usage["goinfer_timings"] = j
	}
}
