package decoder

import (
	"bufio"
	"encoding/json"
	"io"
	"math"
)

// SpecTrace is one speculated position's record — the labeled row the acceptance
// analysis (docs/spec/06) consumes. The key label is AcceptProb = 1 − TV(p, q),
// the *exact* per-position acceptance (a low-variance continuous signal), not just
// the realized Accepted bit. For the n-gram drafter q is the point mass on the
// proposed token, so TV(p, q) = 1 − p(token) and AcceptProb = p(token); QTop1 is 1.
// Field json tags match the §06 schema so dumps are directly analyzable offline.
type SpecTrace struct {
	Step       int     `json:"step"`            // verify round index within the sequence
	Pos        int     `json:"pos"`             // depth within the draft block (0-based)
	Source     string  `json:"source"`          // grammar | ngram | model | head
	Token      int     `json:"token"`           // the drafted token
	NgramMatch int     `json:"ngram_match_len"` // suffix-match length (n-gram source; 0 otherwise)
	Streak     int     `json:"streak"`          // consecutive accepts so far this block
	QTop1      float64 `json:"q_top1"`          // draft top-1 prob (1.0 for a deterministic drafter)
	PTop1      float64 `json:"p_top1"`          // target top-1 prob
	PEntropy   float64 `json:"p_entropy"`       // target entropy (nats)
	TV         float64 `json:"tv"`              // total-variation(p, q)
	AcceptProb float64 `json:"accept_prob"`     // = 1 − TV (exact per-position acceptance)
	Accepted   bool    `json:"accepted"`        // realized greedy outcome (draft == target argmax)
}

// specTracer receives one SpecTrace per verified draft position. nil disables
// tracing (and the target-softmax cost it implies) on the production path.
type specTracer func(SpecTrace)

// targetDist returns the target top-1 probability, the distribution entropy (nats),
// and the probability mass on `token`, from one position's logits. Used only when
// tracing — it pays for a full softmax over the vocab.
func targetDist(logits []float32, token int) (pTop1, pEntropy, pToken float64) {
	p := softmaxStable(logits, 1) // temperature 1 → the model's true next-token distribution
	for i, v := range p {
		if v > pTop1 {
			pTop1 = v
		}
		if v > 0 {
			pEntropy -= v * math.Log(v)
		}
		if i == token {
			pToken = v
		}
	}
	return pTop1, pEntropy, pToken
}

// TraceCollector buffers SpecTraces in memory and (optionally) streams them as
// JSONL. It is the harness-side sink for genNgram's tracer; not used in serving.
type TraceCollector struct {
	Rows []SpecTrace
	w    *bufio.Writer // optional JSONL sink
}

// NewTraceCollector returns a collector that also writes each row as JSON to w if
// w is non-nil. Call Flush before reading the sink.
func NewTraceCollector(w io.Writer) *TraceCollector {
	c := &TraceCollector{}
	if w != nil {
		c.w = bufio.NewWriter(w)
	}
	return c
}

// Record appends a trace (and writes it as a JSONL line when a sink is set).
func (c *TraceCollector) Record(t SpecTrace) {
	c.Rows = append(c.Rows, t)
	if c.w != nil {
		b, _ := json.Marshal(t)
		c.w.Write(b)
		c.w.WriteByte('\n')
	}
}

// Flush flushes the JSONL sink, if any.
func (c *TraceCollector) Flush() error {
	if c.w != nil {
		return c.w.Flush()
	}
	return nil
}

// MeanAccept is α̅ — the mean exact per-position acceptance (1 − TV) over all
// traced positions. The machine-independent headline metric (docs/spec/06 §6).
func (c *TraceCollector) MeanAccept() float64 {
	if len(c.Rows) == 0 {
		return 0
	}
	var s float64
	for _, r := range c.Rows {
		s += r.AcceptProb
	}
	return s / float64(len(c.Rows))
}
