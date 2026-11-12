package decoder

import (
	"fmt"

	"github.com/townsendmerino/goinfer/internal/runtime/kvmgr"
)

// Compile-time assertions that both CPU Session KV and GPU resident KV slots implement kvmgr.SequenceState.
var (
	_ kvmgr.SequenceState = cpuSequenceState{}
	_ kvmgr.SequenceState = residentSequenceState{}
)

// SequenceState returns the session's live KV sequence view under kvmgr.SequenceState.
func (s *Session) SequenceState() KVSequenceState {
	return cpuSequenceState{s: s}
}

// ResidentSequenceState returns slot's live KV sequence view under kvmgr.SequenceState.
func (m *Model) ResidentSequenceState(slot int) KVSequenceState {
	return residentSequenceState{m: m, slot: slot}
}

// cpuSequenceState adapts a CPU *Session and its *KVCache to kvmgr.SequenceState.
type cpuSequenceState struct {
	s *Session
}

func (c cpuSequenceState) Len() int {
	if c.s == nil {
		return 0
	}
	return len(c.s.tokens)
}

func (c cpuSequenceState) Tokens() []int {
	if c.s == nil {
		return nil
	}
	return c.s.tokens
}

func (c cpuSequenceState) Root() kvmgr.Root {
	if c.s == nil {
		return kvmgr.Root{}
	}
	return c.s.sequenceRoot(c.s.kvAdapter)
}

func (c cpuSequenceState) Images() []kvmgr.ImageSpan { return nil }

func (c cpuSequenceState) Classes() []kvmgr.ClassRun {
	if c.s == nil {
		return nil
	}
	return c.s.classes
}

func (c cpuSequenceState) Recurrent() bool {
	return c.s != nil && c.s.cache != nil && c.s.cache.hasRecurrentState()
}

func (c cpuSequenceState) Checkpoint() int { return 0 }

func (c cpuSequenceState) Floor() int { return 0 }

func (c cpuSequenceState) ReusablePrefix(prefix int) int {
	if prefix <= 0 || c.s == nil || c.s.cache == nil {
		return 0
	}
	kv := c.s.cache
	if prefix > kv.Pos() {
		return 0
	}
	if prefix < kv.Pos() && kv.holdsStateHandled(lcRewind, hInexact) {
		return 0
	}
	for _, r := range kv.rings {
		if r != nil && !kvmgr.RingTruncateExact(r.count, r.w, prefix) {
			return 0
		}
	}
	return prefix
}

func (c cpuSequenceState) Truncate(target int) bool {
	if c.s == nil || c.s.cache == nil || target < 0 || target > c.s.cache.Pos() {
		return false
	}
	return c.s.cache.TruncateTo(target)
}

// sequenceRoot returns the kvmgr.Root describing s's CPU cache under lora.
func (s *Session) sequenceRoot(lora *loraRuntime) kvmgr.Root {
	r := kvmgr.Root{Adapter: kvAdapterID(lora), Precision: s.cachePrecision()}
	if s.m != nil && s.m.w != nil && s.m.w.arch != nil {
		a := s.m.w.arch
		r.Model = fmt.Sprintf("%s/%dx%d/%d@cpu", a.Name, a.NumLayers, a.HiddenDim, a.VocabSize)
	}
	return r
}

func (s *Session) cachePrecision() string {
	if s == nil || s.cache == nil {
		return "f32"
	}
	switch s.cache.quant {
	case kvqF16:
		return "f16"
	case kvI8:
		return "i8"
	default:
		return "f32"
	}
}

// residentSequenceState adapts one resident KV slot on *Model to kvmgr.SequenceState.
type residentSequenceState struct {
	m    *Model
	slot int
}

func (r residentSequenceState) inner() kvmgr.SequenceState {
	if r.m == nil {
		return nil
	}
	slots := r.m.kvSlots()
	return slots.StateWith(
		r.slot,
		r.m.hasRecurrentState(),
		func(prefix int) int {
			if prefix <= 0 {
				return 0
			}
			if r.slot == slots.Bound() {
				return r.m.residentReuseFloor(prefix)
			}
			kv := r.m.resKV()
			if kv.ReusableKVSlotAvailable() {
				return kv.ReusableKVSlot(r.slot, prefix)
			}
			return prefix
		},
		func(target int) bool {
			if target == 0 {
				if r.slot == slots.Bound() {
					r.m.residentForgetIDs()
				} else {
					slots.Forget(r.slot)
				}
				return true
			}
			n := slots.Len(r.slot)
			return target >= 0 && target <= n && (!r.m.hasRecurrentState() || target == n)
		},
	)
}

func (r residentSequenceState) Len() int {
	if in := r.inner(); in != nil {
		return in.Len()
	}
	return 0
}

func (r residentSequenceState) Tokens() []int {
	if in := r.inner(); in != nil {
		return in.Tokens()
	}
	return nil
}

func (r residentSequenceState) Root() kvmgr.Root {
	if in := r.inner(); in != nil {
		return in.Root()
	}
	return kvmgr.Root{}
}

func (r residentSequenceState) Images() []kvmgr.ImageSpan {
	if in := r.inner(); in != nil {
		return in.Images()
	}
	return nil
}

func (r residentSequenceState) Classes() []kvmgr.ClassRun {
	if in := r.inner(); in != nil {
		return in.Classes()
	}
	return nil
}

func (r residentSequenceState) Recurrent() bool {
	return r.m != nil && r.m.hasRecurrentState()
}

func (r residentSequenceState) Checkpoint() int {
	if in := r.inner(); in != nil {
		return in.Checkpoint()
	}
	return 0
}

func (r residentSequenceState) Floor() int {
	if in := r.inner(); in != nil {
		return in.Floor()
	}
	return 0
}

func (r residentSequenceState) ReusablePrefix(prefix int) int {
	if in := r.inner(); in != nil {
		return in.ReusablePrefix(prefix)
	}
	return 0
}

func (r residentSequenceState) Truncate(target int) bool {
	if in := r.inner(); in != nil {
		return in.Truncate(target)
	}
	return false
}
