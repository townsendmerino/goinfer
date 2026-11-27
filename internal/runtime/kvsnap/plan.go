package kvsnap

import "github.com/townsendmerino/goinfer/internal/runtime/kvmgr"

// BoundaryConfig carries the geometry and scope parameters for planning boundary and recurrent snapshot marks.
type BoundaryConfig struct {
	PromptLen   int
	Boundary    int
	Keep        int
	MsgBoundary int
	Align       int
	MinPrompt   int
	Slack       int
	MinTail     int
	Recurrent   bool
}

// BoundaryPlan computes the prefill position at which a boundary or recurrent Keep snapshot should be taken instead of at the end of the prompt.
func BoundaryPlan(cfg BoundaryConfig) (q int, ok bool) {
	n := cfg.PromptLen
	if cfg.Recurrent {
		if cfg.Align <= 0 || n < cfg.MinPrompt || cfg.Keep <= 0 || cfg.Keep >= n {
			return 0, false
		}
		return RecurrentCut(n, cfg.Keep, cfg.MinTail)
	}
	if cfg.Boundary <= 0 {
		return 0, false
	}
	if cfg.Align <= 0 || n < cfg.MinPrompt || cfg.Boundary >= n || n-cfg.Boundary <= cfg.Slack {
		return 0, false
	}
	q = cfg.Boundary / cfg.Align * cfg.Align
	if q < MinShared || q%kvmgr.BlockRows != 0 {
		return 0, false
	}
	return q, true
}

// MsgPlan computes the prefill position for a recurrent model's message-boundary snapshot.
func MsgPlan(cfg BoundaryConfig) (q int, ok bool) {
	if !cfg.Recurrent || cfg.Align <= 0 || cfg.PromptLen < cfg.MinPrompt || cfg.MsgBoundary <= 0 || cfg.MsgBoundary >= cfg.PromptLen {
		return 0, false
	}
	return RecurrentCut(cfg.PromptLen, cfg.MsgBoundary, cfg.MinTail)
}

// RecurrentCut rounds point down to a block boundary in a prompt of n tokens, stepping back by whole blocks while fewer than minTail tokens follow.
func RecurrentCut(n, point, minTail int) (q int, ok bool) {
	q = point / kvmgr.BlockRows * kvmgr.BlockRows
	for q > 0 && n-q < minTail {
		q -= kvmgr.BlockRows
	}
	return q, q >= MinShared
}

// CutSegmentOK reports whether prefilling rows tokens in passes of align keeps both the segment and its final pass at or above minTail.
func CutSegmentOK(rows, align, minTail int) bool {
	last := rows % align
	return rows >= minTail && (last == 0 || last >= minTail)
}

// HoldsTarget reports whether held snapshot c already serves a snapshot of the first b rows.
func HoldsTarget(c *Entry, b int, recurrent bool) bool {
	if recurrent {
		return c.Rows() == b
	}
	return c.Rows() >= b
}

// CoverSlack returns the rollback slack a covering search allows (0 for recurrent models, slack otherwise).
func CoverSlack(slack int, recurrent bool) int {
	if recurrent {
		return 0
	}
	return slack
}

// ClipRuns restricts rs to rows [0, rows).
func ClipRuns(rs []kvmgr.ClassRun, rows int) []kvmgr.ClassRun {
	var out []kvmgr.ClassRun
	for _, r := range rs {
		if r.Start >= rows {
			break
		}
		r.Len = min(r.Len, rows-r.Start)
		out = append(out, r)
	}
	return out
}

// ClassesEqual reports whether two class-run slices are identical.
func ClassesEqual(a, b []kvmgr.ClassRun) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
