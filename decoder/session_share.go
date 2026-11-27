package decoder

import (
	"errors"
	"github.com/townsendmerino/goinfer/internal/runtime/cpuengine"
	"time"

	"github.com/townsendmerino/goinfer/internal/runtime/kvmgr"
)

// Sharing between CPU sessions:
// a new conversation copies the whole blocks of another session's KV that its own cold prefill would compute bit for
// bit, and prefills the rest. A CPU prefill pass of cpuengine.FastAttnMinPrompt rows or more runs the fast prefill attention, a
// class of its own; passes at or above it are invariant to where they start (docs/measurements/kv2-2026-10-14.md, "Split invariance").

// cpuClassMap is the class rule for a CPU session's prefill passes.
func (m *Model) cpuClassMap() kvmgr.ClassMap {
	return kvmgr.ClassMap{Floor: cpuengine.FastAttnMinPrompt, MinRows: cpuengine.FastAttnMinPrompt}
}

// errCopyLayout is copyPrefixFrom's refusal of a cache layout it does not copy.
var errCopyLayout = errors.New("decoder: this KV cache layout is not copied between sessions")

// copyPrefixFrom makes c hold src's first n positions and nothing else. It copies append-forever layers (f32 or int8)
// and the MLA latent; a sliding-window ring, recurrent state, image blocks or m-RoPE positions are refused before c is
// touched.
func (c *KVCache) copyPrefixFrom(src *KVCache, n int) error {
	if n <= 0 || n > src.c.PosN || src.c.LocalAny || c.c.LocalAny || src.c.HasRecurrentState() || c.c.HasRecurrentState() ||
		src.c.ImgBlocks != nil || src.c.MRopePos != nil || src.c.Deepstack != nil || src.c.ManualPos || c.c.ManualPos ||
		src.c.Quant != c.c.Quant || src.c.NumLayers != c.c.NumLayers || (src.c.MLALatent == nil) != (c.c.MLALatent == nil) {
		return errCopyLayout
	}
	c.TruncateTo(0)
	c.c.RowPad = src.c.RowPad // rows are copied at src's stride
	for l := range c.c.NumLayers {
		if src.c.MLALatent != nil {
			c.c.MLALatent[l] = append(c.c.MLALatent[l][:0], src.c.MLALatent[l][:n*src.c.LatentDim]...)
			continue
		}
		st := src.c.Stride[l]
		c.c.Stride[l] = st
		if st == 0 {
			continue // a KV-shared layer appends nothing
		}
		if src.c.Quant == kvqF16 {
			c.c.KeysH[l] = append(c.c.KeysH[l][:0], src.c.KeysH[l][:n*st]...)
			c.c.ValsH[l] = append(c.c.ValsH[l][:0], src.c.ValsH[l][:n*st]...)
		}
		if src.c.Quant == kvI8 {
			nKV := st / src.c.HeadDim
			c.c.KeysQ[l] = append(c.c.KeysQ[l][:0], src.c.KeysQ[l][:n*st]...)
			c.c.ValsQ[l] = append(c.c.ValsQ[l][:0], src.c.ValsQ[l][:n*st]...)
			c.c.KeyScale[l] = append(c.c.KeyScale[l][:0], src.c.KeyScale[l][:n*nKV]...)
			c.c.ValScale[l] = append(c.c.ValScale[l][:0], src.c.ValScale[l][:n*nKV]...)
		}
		if len(src.c.KeysBuf[l]) >= n*st { // f32 rows, which an int8 layer may also keep
			c.c.KeysBuf[l] = append(c.c.KeysBuf[l][:0], src.c.KeysBuf[l][:n*st]...)
			c.c.ValsBuf[l] = append(c.c.ValsBuf[l][:0], src.c.ValsBuf[l][:n*st]...)
		}
	}
	if src.c.MLALatent != nil {
		c.c.LatentDim = src.c.LatentDim
	}
	c.c.PosN = n
	return nil
}

// recordClasses sets the row classes of what a generation left in the session: the reused rows keep theirs, the pass
// over prompt[matched:] is one prefill pass, and decode's rows are exact. A sequential prefill (an adapter, or a family
// the batched pass does not take) writes decode's rows.
func (s *Session) recordClasses(prompt []int, matched int, batched bool) {
	cls := kvmgr.ClassExact
	if batched {
		cls = s.m.cpuClassMap().Pass(len(prompt)-matched, len(prompt))
	}
	s.classes = kvmgr.RecordTurnClasses(s.classes, matched, len(prompt), len(s.tokens), cls)
}

// SharedHit is how many leading tokens of prompt this session can give another session by copy (CopyPrefixFrom): whole
// blocks of a common token prefix whose rows are the fast class, which the requester's own cold prefill writes, with
// the requester's suffix left on the fast class too (kvmgr.ClassMap.CapHit). 0 when prompt's cold prefill is not the
// fast class, when this session's KV was built under an adapter, or when nothing qualifies.
func (s *Session) SharedHit(prompt []int) int {
	n := len(prompt)
	if n == 0 || s.kvAdapter != nil {
		return 0
	}
	return kvmgr.SharedPrefixHit(s.SequenceState(), prompt, s.m.cpuClassMap(), s.m.canBatchN(n))
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
	s.classes = kvmgr.ClipClassRuns(donor.classes, n)
	s.kvAdapter, s.cleanCache = donor.kvAdapter, false
	s.shared, s.copyDur = n, time.Since(t0)
	return nil
}

// KVShareOff reports Options.KVShareOff: no request takes KV rows another conversation computed.
func (m *Model) KVShareOff() bool { return m.opts.KVShareOff }

// KeepPrefix cuts the session back to its first n positions (a SharedHit of its own), so a request that takes it in
// place reuses exactly the rows a copy would have given it, and not a longer token match its cold prefill would compute
// on another class.
func (s *Session) KeepPrefix(n int) {
	if n < 0 || n > len(s.tokens) || !s.cache.TruncateTo(n) {
		s.Reset()
		return
	}
	s.tokens = s.tokens[:n]
	s.classes = kvmgr.ClipClassRuns(s.classes, n)
}
