package decoder

import (
	"io"
	"os"
	"sync/atomic"

	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/kvsnap"
	"github.com/townsendmerino/goinfer/internal/runtime/residency"
)

type (
	kvAcquired         = residency.KVAcquired
	KVCopyDecline      = residency.KVCopyDecline
	residentImageBlock = residency.ImageBlock
	residentImageClaim = residency.ImageClaim
	prefillMark        = residency.PrefillMark
)

const (
	kvSnapMinShared = kvsnap.MinShared
)

var (
	ErrKVPoolFull                     = residency.ErrKVPoolFull
	stderrKV                io.Writer = os.Stderr
	kvSnapMinPrompt                   = 2048
	kvAdapterIDs            atomic.Uint64
	scratchSlotPlantForTest bool
	kvAllowanceOverride     int
	kvInPlaceCapOff         bool
)

func kvAdapterID(rt *loraRuntime) uint64 {
	if rt == nil {
		return 0
	}
	if id := rt.kvID.Load(); id != 0 {
		return id
	}
	rt.kvID.CompareAndSwap(0, kvAdapterIDs.Add(1))
	return rt.kvID.Load()
}

func (m *Model) resHost() *residency.Host {
	var arch *darch.Architecture
	if m.w != nil {
		arch = m.w.Arch
	}
	var beName string
	if m.be != nil {
		beName = m.be.Name()
	}
	return &residency.Host{
		Resident:          m.resident,
		KV:                &m.kv,
		KVNS:              &m.kvNS,
		KVSnaps:           &m.kvSnaps,
		CkptPrompt:        &m.resCkptPrompt,
		DrafterSynced:     &m.resDrafterSynced,
		Arch:              arch,
		BackendName:       beName,
		KVPrecision:       m.ResidentKVPrecision(),
		ReuseDisabled:     m.knobs.get(knobNoResidentReuse) != "",
		RecurrentSlotsOK:  m.be != nil && m.recurrentSlotsOn(beName),
		ShareOff:          m.opts.KVShareOff,
		KVDemote:          m.opts.KVDemote,
		ScratchSlotPlant:  scratchSlotPlantForTest,
		AllowanceOverride: kvAllowanceOverride,
		InPlaceCapOff:     kvInPlaceCapOff,
		KVHostBytesOpt:    m.opts.KVHostBytes,
		PrefillTailMin:    m.prefillTailMinRows(),
		Stderr:            stderrKV,
		SnapMinPrompt:     kvSnapMinPrompt,
	}
}

func (m *Model) ResidentKVPool() KVPoolDetail {
	h := m.resHost()
	return h.KVPool()
}

func (m *Model) KVShareState() (state, why string) {
	h := m.resHost()
	return h.KVShareState()
}

func (m *Model) KeepsRecurrentSnapshots() bool {
	h := m.resHost()
	return h.KeepsRecurrentSnapshots()
}

func (m *Model) KVSnapshotStats() (KVSnapshotStats, bool) {
	h := m.resHost()
	return h.KVSnapshotStats()
}

func (m *Model) KVSnapshotsSupported() bool {
	h := m.resHost()
	return h.KVSnapshotsSupported()
}

func (m *Model) KVHostBytes() int64 {
	h := m.resHost()
	return h.KVHostBytes()
}
