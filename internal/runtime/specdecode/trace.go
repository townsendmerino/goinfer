package specdecode

import (
	"bufio"
	"encoding/json"
	"io"
	"math"

	"github.com/townsendmerino/goinfer/internal/runtime/sampling"
)

// SpecTrace is one speculated position's record — the labeled row the acceptance
// analysis (docs/spec/06) consumes.
type SpecTrace struct {
	Step       int     `json:"step"`            // verify round index within the sequence
	Pos        int     `json:"pos"`             // depth within the draft block (0-based)
	Source     string  `json:"source"`          // grammar | ngram | model | head
	Forced     bool    `json:"forced"`          // grammar pinned these bytes (the α̂_grammar support)
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

// Tracer receives one SpecTrace per verified draft position. nil disables tracing.
type Tracer func(SpecTrace)

// TargetDist returns the target top-1 probability, the distribution entropy (nats),
// and the probability mass on token, from one position's logits.
func TargetDist(logits []float32, token int) (pTop1, pEntropy, pToken float64) {
	p := sampling.SoftmaxStable(logits, 1)
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

// TraceFromDist builds a SpecTrace from a full normalized target distribution p.
func TraceFromDist(step, pos, matchLen, token int, p []float64, accepted bool) SpecTrace {
	var pTop1, pEnt, pTok float64
	for i, v := range p {
		if v > pTop1 {
			pTop1 = v
		}
		if v > 0 {
			pEnt -= v * math.Log(v)
		}
		if i == token {
			pTok = v
		}
	}
	return SpecTrace{
		Step: step, Pos: pos, Source: "ngram", Token: token,
		NgramMatch: matchLen, Streak: pos, QTop1: 1,
		PTop1: pTop1, PEntropy: pEnt, TV: 1 - pTok, AcceptProb: pTok, Accepted: accepted,
	}
}

// TraceCollector buffers SpecTraces in memory and (optionally) streams them as JSONL.
type TraceCollector struct {
	Rows []SpecTrace
	w    *bufio.Writer
}

// NewCollector returns a collector that also writes each row as JSON to w if w is non-nil.
func NewCollector(w io.Writer) *TraceCollector {
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

// MeanAccept is α̅ — the mean exact per-position acceptance (1 − TV) over all traced positions.
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
