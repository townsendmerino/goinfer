package decoder

import (
	"errors"
	"slices"
	"time"

	"github.com/townsendmerino/goinfer/internal/kvmgr"
)

// Sharing between CPU sessions (docs/tasks/task-kv-architecture-2026-10.md §6, KV2; the CPU joins through copies, D2):
// a new conversation copies the whole blocks of another session's KV that its own cold prefill would compute bit for
// bit, and prefills the rest. A CPU prefill pass of fastAttnMinPrompt rows or more runs the fast prefill attention, a
// class of its own; passes at or above it are invariant to where they start (docs/measurements/kv2-2026-10-14.md, G0).

// cpuClassMap is the class rule for a CPU session's prefill passes.
func (m *Model) cpuClassMap() kvmgr.ClassMap {
	return kvmgr.ClassMap{Floor: fastAttnMinPrompt, MinRows: fastAttnMinPrompt}
}

// errCopyLayout is copyPrefixFrom's refusal of a cache layout it does not copy.
var errCopyLayout = errors.New("decoder: this KV cache layout is not copied between sessions")

// copyPrefixFrom makes c hold src's first n positions and nothing else. It copies append-forever layers (f32 or int8)
// and the MLA latent; a sliding-window ring, recurrent state, image blocks or m-RoPE positions are refused before c is
// touched.
func (c *KVCache) copyPrefixFrom(src *KVCache, n int) error {
	if n <= 0 || n > src.pos || src.localAny || c.localAny || src.hasRecurrentState() || c.hasRecurrentState() ||
		src.imgBlocks != nil || src.mropePos != nil || src.deepstack != nil || src.manualPos || c.manualPos ||
		src.quant != c.quant || src.numLayers != c.numLayers || (src.mlaLatent == nil) != (c.mlaLatent == nil) {
		return errCopyLayout
	}
	c.TruncateTo(0)
	for l := range c.numLayers {
		if src.mlaLatent != nil {
			c.mlaLatent[l] = append(c.mlaLatent[l][:0], src.mlaLatent[l][:n*src.latentDim]...)
			continue
		}
		st := src.stride[l]
		c.stride[l] = st
		if st == 0 {
			continue // a KV-shared layer appends nothing
		}
		if src.quant == kvI8 {
			nKV := st / src.headDim
			c.keysQ[l] = append(c.keysQ[l][:0], src.keysQ[l][:n*st]...)
			c.valsQ[l] = append(c.valsQ[l][:0], src.valsQ[l][:n*st]...)
			c.keyScale[l] = append(c.keyScale[l][:0], src.keyScale[l][:n*nKV]...)
			c.valScale[l] = append(c.valScale[l][:0], src.valScale[l][:n*nKV]...)
		}
		if len(src.keys[l]) >= n*st { // f32 rows, which an int8 layer may also keep
			c.keys[l] = append(c.keys[l][:0], src.keys[l][:n*st]...)
			c.vals[l] = append(c.vals[l][:0], src.vals[l][:n*st]...)
		}
	}
	if src.mlaLatent != nil {
		c.latentDim = src.latentDim
	}
	c.pos = n
	return nil
}

// recordClasses sets the row classes of what a generation left in the session: the reused rows keep theirs, the pass
// over prompt[matched:] is one prefill pass, and decode's rows are exact. A sequential prefill (an adapter, or a family
// the batched pass does not take) writes decode's rows.
func (s *Session) recordClasses(prompt []int, matched int, batched bool) {
	n := len(s.tokens)
	out := clipClassRuns(s.classes, min(matched, n))
	cls := kvmgr.ClassExact
	if batched {
		cls = s.m.cpuClassMap().Pass(len(prompt)-matched, len(prompt))
	}
	if end := min(len(prompt), n); end > matched {
		out = append(out, kvmgr.ClassRun{Start: matched, Len: end - matched, Class: cls})
	}
	if n > len(prompt) {
		out = append(out, kvmgr.ClassRun{Start: len(prompt), Len: n - len(prompt), Class: kvmgr.ClassExact})
	}
	s.classes = out
}

// SharedHit is how many leading tokens of prompt this session can give another session by copy (CopyPrefixFrom): whole
// blocks of a common token prefix whose rows are the fast class, which the requester's own cold prefill writes, with
// the requester's suffix left on the fast class too (kvmgr.ClassMap.CapHit). 0 when prompt's cold prefill is not the
// fast class, when this session's KV was built under an adapter, or when nothing qualifies.
func (s *Session) SharedHit(prompt []int) int {
	cm := s.m.cpuClassMap()
	n := len(prompt)
	if n == 0 || s.kvAdapter != nil || !s.m.canBatchN(n) || cm.Pass(n, n) != kvmgr.ClassFast {
		return 0
	}
	hit := min(commonPrefixLen(s.tokens, prompt), n-1) / kvmgr.BlockRows * kvmgr.BlockRows
	for p := 0; p < hit; p++ {
		if !slices.ContainsFunc(s.classes, func(r kvmgr.ClassRun) bool {
			return r.Class == kvmgr.ClassFast && p >= r.Start && p < r.Start+r.Len
		}) {
			hit = p / kvmgr.BlockRows * kvmgr.BlockRows
			break
		}
	}
	return cm.CapHit(n, hit)
}

// CopyPrefixFrom makes s hold donor's first n positions (a SharedHit of donor), so s's next Generate reuses them. The
// caller holds both sessions: neither may be generating. s's adapter must be the donor's (the base weights).
func (s *Session) CopyPrefixFrom(donor *Session, n int) error {
	if n <= 0 || n > len(donor.tokens) || s.cache.lora != donor.cache.lora {
		return errCopyLayout
	}
	t0 := time.Now()
	if err := s.cache.copyPrefixFrom(donor.cache, n); err != nil {
		return err
	}
	s.tokens = append(s.tokens[:0], donor.tokens[:n]...)
	s.classes = clipClassRuns(donor.classes, n)
	s.kvAdapter, s.cleanCache = donor.kvAdapter, false
	s.shared, s.copyDur = n, time.Since(t0)
	return nil
}

// clipClassRuns is runs restricted to rows [0, n), as a new slice.
func clipClassRuns(runs []kvmgr.ClassRun, n int) []kvmgr.ClassRun {
	var out []kvmgr.ClassRun
	for _, r := range runs {
		if r.Start < n {
			r.Len = min(r.Len, n-r.Start)
			out = append(out, r)
		}
	}
	return out
}

// KVShareOff reports Options.KVShareOff: no request takes KV rows another conversation computed.
func (m *Model) KVShareOff() bool { return m.kvShareOff }

// KeepPrefix cuts the session back to its first n positions (a SharedHit of its own), so a request that takes it in
// place reuses exactly the rows a copy would have given it, and not a longer token match its cold prefill would compute
// on another class.
func (s *Session) KeepPrefix(n int) {
	if n < 0 || n > len(s.tokens) || !s.cache.TruncateTo(n) {
		s.Reset()
		return
	}
	s.tokens = s.tokens[:n]
	s.classes = clipClassRuns(s.classes, n)
}
