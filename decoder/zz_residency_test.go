package decoder

import (
	"time"

	"github.com/townsendmerino/goinfer/internal/runtime/kvmgr"
	"github.com/townsendmerino/goinfer/internal/runtime/residency"
)

func (m *Model) kvSlots() *kvmgr.Slots { h := m.resHost(); return h.KVSlots() }
func (m *Model) kvRoot(lora *loraRuntime) kvmgr.Root {
	h := m.resHost()
	return h.KVRoot(kvAdapterID(lora))
}
func (m *Model) kvRootIn(lora *loraRuntime, ns string) kvmgr.Root {
	h := m.resHost()
	return h.KVRootIn(kvAdapterID(lora), ns)
}
func (m *Model) kvClassMap() kvmgr.ClassMap { h := m.resHost(); return h.KVClassMap() }
func (m *Model) residentSlotCount() int     { h := m.resHost(); return h.SlotCount() }
func (m *Model) residentReuseLen(prompt []int, imgs []residentImageClaim, lora *loraRuntime) int {
	h := m.resHost()
	return h.ReuseLen(prompt, imgs, kvAdapterID(lora))
}
func (m *Model) residentAcquire(prompt []int, imgs []residentImageClaim, lora *loraRuntime) int {
	h := m.resHost()
	return h.Acquire(prompt, imgs, kvAdapterID(lora))
}
func (m *Model) residentAcquireSlot(prompt []int, imgs []residentImageClaim, lora *loraRuntime, busy []bool) (int, int) {
	h := m.resHost()
	return h.AcquireSlot(prompt, imgs, kvAdapterID(lora), busy)
}
func (m *Model) residentAcquireIn(prompt []int, imgs []residentImageClaim, lora *loraRuntime, busy []bool, sc KVScope) kvAcquired {
	h := m.resHost()
	return h.AcquireIn(prompt, imgs, kvAdapterID(lora), busy, sc)
}
func (m *Model) residentBind(slot int) error { h := m.resHost(); return h.Bind(slot) }
func (m *Model) residentCommitIDs(prompt, generated []int, newBlocks []residentImageBlock, lora *loraRuntime) {
	h := m.resHost()
	h.CommitIDs(prompt, generated, newBlocks, kvAdapterID(lora))
}
func (m *Model) residentForgetIDs()           { h := m.resHost(); h.ForgetIDs() }
func (m *Model) checkpointHeld(slot int) bool { h := m.resHost(); return h.CheckpointHeld(slot) }
func (m *Model) recurrentCheckpointer() ResidentKVSubsystem {
	h := m.resHost()
	return h.RecurrentCheckpointer()
}
func (m *Model) residentCheckpointSave(prompt []int) (time.Duration, bool) {
	h := m.resHost()
	return h.CheckpointSave(prompt)
}

func kvSpansOfBlocks(blocks []residentImageBlock) []kvmgr.ImageSpan {
	return residency.KVSpansOfBlocks(blocks)
}

func kvChooseDonor(res ResidentForward, hits []int, cm kvmgr.ClassMap, n, dst, reuse int) (int, int, string) {
	return residency.KVChooseDonor(res, hits, cm, n, dst, reuse)
}
func kvOtherDonors(rk ResidentKVSubsystem, hits []int, cm kvmgr.ClassMap, n, src, dst, reuse int) string {
	return residency.KVOtherDonors(rk, hits, cm, n, src, dst, reuse)
}
func pickResidentSlot(scores, lens []int, uses []uint64, busy []bool) int {
	return residency.PickResidentSlot(scores, lens, uses, busy)
}
func (m *Model) residentShare(prompt []int, imgs []residentImageClaim, lora *loraRuntime, sc KVScope, acq *kvAcquired) {
	h := m.resHost()
	h.Share(prompt, imgs, kvAdapterID(lora), sc, acq)
}

const (
	minLeadReuseFast   = residency.MinLeadReuseFast
	fastPrefillMinRows = residency.FastPrefillMinRows
)

func (m *Model) declineShortLeadReuse(prompt []int, reuse int) int {
	h := m.resHost()
	return h.DeclineShortLeadReuse(prompt, reuse)
}

func (m *Model) kvNSOf(slot int) string { h := m.resHost(); return h.KVNSOf(slot) }
