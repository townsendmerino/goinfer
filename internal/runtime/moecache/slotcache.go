package moecache

// PendingAdmit records one provisional slot claim made before its host-to-device transfer completes.
type PendingAdmit struct {
	Slot   int
	Expert uint32
}

// SlotCache tracks per-layer LRU residency across a fixed count of device expert slots. Not safe for concurrent access;
// the owning backend executor serializes calls per layer.
type SlotCache struct {
	nSlots       int
	slotOf       []int32
	inSlot       []int32
	used         []uint64
	clock        uint64
	hits, misses uint64
	stages       uint64
	distinct     uint64
}

// NewSlotCache allocates an empty SlotCache mapping nExperts expert IDs across nSlots device slots.
func NewSlotCache(nExperts, nSlots int) *SlotCache {
	c := &SlotCache{
		nSlots: nSlots,
		slotOf: make([]int32, nExperts),
		inSlot: make([]int32, nSlots),
		used:   make([]uint64, nSlots),
	}
	for i := range c.slotOf {
		c.slotOf[i] = -1
	}
	for i := range c.inSlot {
		c.inSlot[i] = -1
	}
	return c
}

// Admit returns the slot holding expert e, evicting an empty or least-recently-used slot on a miss. When hit is false,
// the caller must transfer e's weights into slot or call Unadmit if the transfer fails.
func (c *SlotCache) Admit(e uint32) (slot int, hit bool) {
	c.clock++
	if s := c.slotOf[e]; s >= 0 {
		c.used[s] = c.clock
		c.hits++
		return int(s), true
	}
	c.misses++
	victim, oldest := 0, ^uint64(0)
	for s := range c.nSlots {
		if c.inSlot[s] < 0 {
			victim = s
			break
		}
		if c.used[s] < oldest {
			oldest = c.used[s]
			victim = s
		}
	}
	if old := c.inSlot[victim]; old >= 0 {
		c.slotOf[old] = -1
	}
	c.inSlot[victim] = int32(e)
	c.slotOf[e] = int32(victim)
	c.used[victim] = c.clock
	return victim, false
}

// Unadmit marks slot empty and clears expert e's residency mapping after a failed transfer so a subsequent Admit
// re-transfers the weights.
func (c *SlotCache) Unadmit(slot int, e uint32) {
	if c.inSlot[slot] == int32(e) {
		c.inSlot[slot] = -1
	}
	if c.slotOf[e] == int32(slot) {
		c.slotOf[e] = -1
	}
}

// Rollback undoes every provisional admission in pending.
func (c *SlotCache) Rollback(pending []PendingAdmit) {
	for _, p := range pending {
		c.Unadmit(p.Slot, p.Expert)
	}
}

// RecordStageDemand records one routing stage and counts the unique expert IDs requested in ids before cache admission.
func (c *SlotCache) RecordStageDemand(ids []uint32) {
	c.stages++
	for j, id := range ids {
		dup := false
		for k := range j {
			if ids[k] == id {
				dup = true
				break
			}
		}
		if !dup {
			c.distinct++
		}
	}
}

// Hits returns cumulative cache-hit admissions.
func (c *SlotCache) Hits() uint64 { return c.hits }

// Misses returns cumulative cache-miss admissions.
func (c *SlotCache) Misses() uint64 { return c.misses }

// Stages returns cumulative staging events recorded via RecordStageDemand.
func (c *SlotCache) Stages() uint64 { return c.stages }

// Distinct returns the cumulative sum of per-stage unique expert IDs recorded via RecordStageDemand.
func (c *SlotCache) Distinct() uint64 { return c.distinct }

// CopyStatsFrom copies cumulative hit, miss, and demand counters from src into c.
func (c *SlotCache) CopyStatsFrom(src *SlotCache) {
	if src == nil {
		return
	}
	c.hits, c.misses = src.hits, src.misses
	c.stages, c.distinct = src.stages, src.distinct
}

// ResetStats zeroes cumulative hit, miss, and demand counters without changing slot residency.
func (c *SlotCache) ResetStats() {
	c.hits, c.misses, c.stages, c.distinct = 0, 0, 0, 0
}
