package decoder

import (
	"fmt"
	"math/rand"
	"slices"
	"testing"
)

// The reuse bookkeeping the KV manager replaced, kept here as the reference it is compared with: the per-slot id
// lists, the longest-common-prefix scan, the slot choice and the short-lead guard, as they were before the manager
// (resident_reuse.go at the commit this file was added in). TestResidentReuse_equalsTheBookkeepingItReplaced walks
// both at random and requires the same answer to every question a generation asks.
type oldReuse struct {
	resident   any
	recurrent  bool
	resIDs     []int
	resIDsLora *loraRuntime
	resImg     []residentImageBlock
	resSlots   []residentSlot
	resCur     int
	resTick    uint64
}

type residentSlot struct {
	ids       []int
	lora      *loraRuntime
	imgBlocks []residentImageBlock
	lastUse   uint64
}

func (m *oldReuse) residentReuseLen(prompt []int, imgs []residentImageClaim, lora *loraRuntime) int {
	return m.residentReuseFloor(m.reuseLenOf(m.resIDs, m.resIDsLora, m.resImg, prompt, imgs, lora))
}

func (m *oldReuse) residentReuseFloor(reuse int) int {
	if reuse <= 0 {
		return reuse
	}
	if rk, ok := m.resident.(ResidentReusableKV); ok {
		return rk.ReusableKV(reuse)
	}
	return reuse
}

func (m *oldReuse) reuseLenOf(resIDs []int, resLora *loraRuntime, resImgBlocks []residentImageBlock, prompt []int, imgs []residentImageClaim, lora *loraRuntime) int {
	if len(resIDs) == 0 || len(prompt) == 0 {
		return 0
	}
	if lora != resLora {
		return 0
	}
	if m.recurrent {
		n := len(resIDs)
		if len(prompt) <= n {
			return 0
		}
		for i := range n {
			if resIDs[i] != prompt[i] {
				return 0
			}
		}
		for _, blk := range resImgBlocks {
			c, ok := findImageClaim(imgs, blk.start)
			if !ok || blk.hash == 0 || c.Hash != blk.hash || c.Len != blk.end-blk.start {
				return 0
			}
		}
		for _, c := range imgs {
			if c.Start >= n {
				continue
			}
			if c.Start+c.Len > n {
				return 0
			}
			ok := false
			for _, blk := range resImgBlocks {
				if blk.start == c.Start && blk.end-blk.start == c.Len {
					ok = true
					break
				}
			}
			if !ok {
				return 0
			}
		}
		return n
	}
	n := min(len(prompt)-1, len(resIDs))
	i, bi := 0, 0
	for i < n {
		for bi < len(resImgBlocks) && resImgBlocks[bi].end <= i {
			bi++
		}
		if bi < len(resImgBlocks) && resImgBlocks[bi].start == i {
			blk := resImgBlocks[bi]
			if claim, ok := findImageClaim(imgs, blk.start); ok &&
				claim.Len == blk.end-blk.start && blk.hash != 0 && claim.Hash == blk.hash && blk.end <= n {
				i = blk.end
				bi++
				continue
			}
			return i
		}
		if resIDs[i] != prompt[i] {
			return i
		}
		i++
	}
	return i
}

func findImageClaim(imgs []residentImageClaim, pos int) (residentImageClaim, bool) {
	for _, c := range imgs {
		if c.Start == pos {
			return c, true
		}
	}
	return residentImageClaim{}, false
}

func (m *oldReuse) residentSlotCount() int {
	sl, ok := m.resident.(ResidentKVSlotter)
	if !ok || m.recurrent {
		return 1
	}
	return max(1, sl.KVSlots())
}

func (m *oldReuse) residentAcquireSlot(prompt []int, imgs []residentImageClaim, lora *loraRuntime, busy []bool) (reuse, slot int) {
	reuse, slot = m.residentAcquireSlotAll(prompt, imgs, lora, busy)
	return m.residentReuseFloor(m.declineShortLeadReuse(prompt, reuse)), slot
}

func (m *oldReuse) declineShortLeadReuse(prompt []int, reuse int) int {
	if reuse <= 0 || reuse >= 64 {
		return reuse
	}
	fp, ok := m.resident.(ResidentFastPrefill)
	if !ok {
		return reuse
	}
	if floor := fp.FastPrefillFloor(); floor <= 0 || len(prompt) < floor {
		return reuse
	}
	return 0
}

func (m *oldReuse) residentAcquireSlotAll(prompt []int, imgs []residentImageClaim, lora *loraRuntime, busy []bool) (reuse, slot int) {
	n := m.residentSlotCount()
	if n <= 1 {
		return m.residentReuseLen(prompt, imgs, lora), 0
	}
	if len(m.resSlots) != n {
		m.resSlots, m.resCur = make([]residentSlot, n), 0
	}
	scores, lens, uses := make([]int, n), make([]int, n), make([]uint64, n)
	for i := range n {
		if i == m.resCur {
			scores[i], lens[i] = m.residentReuseLen(prompt, imgs, lora), len(m.resIDs)
		} else {
			st := &m.resSlots[i]
			scores[i], lens[i] = m.reuseLenOf(st.ids, st.lora, st.imgBlocks, prompt, imgs, lora), len(st.ids)
		}
		uses[i] = m.resSlots[i].lastUse
	}
	pick := oldPickResidentSlot(scores, lens, uses, busy)
	if pick < 0 {
		return 0, -1
	}
	if err := m.residentBind(pick); err != nil {
		m.residentForgetIDs()
		return 0, -1
	}
	m.resTick++
	m.resSlots[pick].lastUse = m.resTick
	return scores[pick], pick
}

func (m *oldReuse) residentBind(slot int) error {
	if slot == m.resCur || m.residentSlotCount() <= 1 {
		return nil
	}
	sl := m.resident.(ResidentKVSlotter)
	if err := sl.UseKVSlot(slot); err != nil {
		return err
	}
	if len(m.resSlots) != m.residentSlotCount() {
		m.resSlots = make([]residentSlot, m.residentSlotCount())
	}
	cur := &m.resSlots[m.resCur]
	cur.ids, cur.lora, cur.imgBlocks = m.resIDs, m.resIDsLora, m.resImg
	in := &m.resSlots[slot]
	m.resIDs, m.resIDsLora, m.resImg = in.ids, in.lora, in.imgBlocks
	in.ids, in.lora, in.imgBlocks = nil, nil, nil
	m.resCur = slot
	return nil
}

func oldPickResidentSlot(scores, lens []int, uses []uint64, busy []bool) int {
	free := func(i int) bool { return i >= len(busy) || !busy[i] }
	best := -1
	for i, s := range scores {
		if free(i) && s > 0 && (best < 0 || s > scores[best]) {
			best = i
		}
	}
	if best >= 0 && scores[best] >= lens[best]-scores[best] {
		return best
	}
	for i, l := range lens {
		if free(i) && l == 0 {
			return i
		}
	}
	lru := -1
	for i := range uses {
		if free(i) && (lru < 0 || uses[i] < uses[lru]) {
			lru = i
		}
	}
	return lru
}

func (m *oldReuse) residentCommitIDs(prompt, generated []int, newBlocks []residentImageBlock, lora *loraRuntime) {
	ids := make([]int, 0, len(prompt)+len(generated))
	ids = append(ids, prompt...)
	ids = append(ids, generated...)
	m.resIDs = ids
	m.resIDsLora = lora
	if len(newBlocks) > 0 {
		kept := m.resImg[:0:0]
		for _, b := range m.resImg {
			if b.end <= newBlocks[0].start {
				kept = append(kept, b)
			}
		}
		m.resImg = append(kept, newBlocks...)
	}
}

func (m *oldReuse) residentForgetIDs() {
	m.resIDsLora, m.resIDs, m.resImg = nil, nil, nil
}

// equivResident is a fake resident with every capability the reuse code asks about: slots, a bind that can fail, a
// fast-prefill floor (0: off) and a windowed floor that drops some prefixes.
type equivResident struct {
	*fakeResident
	n, bound int
	fail     bool
	floor    int
	window   int // 0: every position is kept; else a prefix is served only when prefix%window != 0
}

func (r *equivResident) KVSlots() int { return r.n }
func (r *equivResident) UseKVSlot(i int) error {
	if r.fail {
		return fmt.Errorf("equivResident: forced bind failure")
	}
	r.bound = i
	return nil
}
func (r *equivResident) FastPrefillFloor() int { return r.floor }
func (r *equivResident) ReusableKV(prefix int) int {
	if r.window > 0 && prefix%r.window == 0 {
		return 0
	}
	return prefix
}

func TestResidentReuse_equalsTheBookkeepingItReplaced(t *testing.T) {
	// KV2's choice 3 (docs/measurements/kv2-2026-10-14.md, G1's exception) is off here: this walk compares the rest of
	// the slot choice with KV1's bookkeeping; choice 3 has tests of its own. So is the in-place class cap (KV2-FU3), which
	// KV1's bookkeeping had no classes for; TestResidentAcquire_capsInPlaceReuseToTheColdClass and TestCapInPlace hold it.
	defer func(old int) { kvAllowanceOverride = old }(kvAllowanceOverride)
	kvAllowanceOverride = 1 << 30
	defer func(old bool) { kvInPlaceCapOff = old }(kvInPlaceCapOff)
	kvInPlaceCapOff = true
	loras := []*loraRuntime{nil, {name: "a"}, {name: "b"}}
	steps, binds, reused := 0, 0, 0
	for seed := int64(1); seed <= 24; seed++ {
		rng := rand.New(rand.NewSource(seed))
		cfg := equivResident{n: 1 + rng.Intn(4), floor: []int{0, 0, 96, 512}[rng.Intn(4)], window: []int{0, 0, 7}[rng.Intn(3)]}
		rn, ro := cfg, cfg
		m, o := &Model{resident: &rn}, &oldReuse{resident: &ro}
		if seed%6 == 0 {
			m = recurrentTestModel(nil)
			m.resident = &rn
			o.recurrent = true
		}
		var history [][]int
		tokens := func(n int) []int {
			out := make([]int, n)
			for i := range out {
				out[i] = rng.Intn(3)
			}
			return out
		}
		prompt := func() ([]int, []residentImageClaim) {
			var p []int
			if len(history) > 0 && rng.Intn(5) != 0 {
				h := history[rng.Intn(len(history))]
				p = append(p, h[:rng.Intn(len(h)+1)]...)
			}
			p = append(p, tokens(rng.Intn(700)>>uint(rng.Intn(4)))...)
			var imgs []residentImageClaim
			if len(p) > 6 && rng.Intn(3) == 0 {
				at := rng.Intn(len(p) - 4)
				imgs = append(imgs, residentImageClaim{Start: at, Len: 1 + rng.Intn(min(len(p)-at, 30)), Hash: uint64(rng.Intn(3))})
			}
			return p, imgs
		}
		same := func(what string, op int) {
			t.Helper()
			if !slices.Equal(m.resIDs, o.resIDs) || m.resIDsLora != o.resIDsLora || !slices.Equal(m.resImgBlocks, o.resImg) || (m.resIDs == nil) != (o.resIDs == nil) {
				t.Fatalf("seed %d op %d, after %s: the bound record differs\n new: %d ids, adapter %p, images %v\n old: %d ids, adapter %p, images %v",
					seed, op, what, len(m.resIDs), m.resIDsLora, m.resImgBlocks, len(o.resIDs), o.resIDsLora, o.resImg)
			}
			if rn.bound != ro.bound {
				t.Fatalf("seed %d op %d, after %s: slot %d is bound, the old code bound %d", seed, op, what, rn.bound, ro.bound)
			}
		}
		for op := 0; op < 1500; op++ {
			steps++
			switch k := rng.Intn(20); {
			case k < 13: // a generation
				p, imgs := prompt()
				lora := loras[rng.Intn(len(loras))]
				var busy []bool
				if rng.Intn(4) == 0 {
					busy = make([]bool, cfg.n)
					for i := range busy {
						busy[i] = rng.Intn(3) == 0
					}
				}
				rn.fail = rng.Intn(25) == 0
				ro.fail = rn.fail
				gr, gs := m.residentAcquireSlot(p, imgs, lora, busy)
				wr, ws := o.residentAcquireSlot(p, imgs, lora, busy)
				if gr != wr || gs != ws {
					t.Fatalf("seed %d op %d: a %d-token prompt reuses %d on slot %d; the old code said %d on slot %d (slots %d, floor %d, window %d, busy %v)",
						seed, op, len(p), gr, gs, wr, ws, cfg.n, cfg.floor, cfg.window, busy)
				}
				rn.fail, ro.fail = false, false
				same("acquire", op)
				if gr > 0 {
					reused++
				}
				m.residentForgetIDs()
				o.residentForgetIDs()
				if gs < 0 || len(p) == 0 || rng.Intn(6) == 0 {
					break // refused, or cancelled: nothing is committed
				}
				gen := tokens(rng.Intn(80))
				var blocks []residentImageBlock
				for _, c := range imgs {
					blocks = append(blocks, residentImageBlock{start: c.Start, end: c.Start + c.Len, hash: c.Hash})
				}
				m.residentCommitIDs(p, gen, blocks, lora)
				o.residentCommitIDs(p, gen, blocks, lora)
				same("commit", op)
				history = append(history, append(slices.Clone(p), gen...))
				if len(history) > 12 {
					history = history[1:]
				}
			case k < 15: // the reuse length of the bound slot, asked directly (the image paths do)
				p, imgs := prompt()
				lora := loras[rng.Intn(len(loras))]
				if g, w := m.residentReuseLen(p, imgs, lora), o.residentReuseLen(p, imgs, lora); g != w {
					t.Fatalf("seed %d op %d: residentReuseLen = %d, the old code said %d", seed, op, g, w)
				}
			case k < 17: // bind a slot by number (an MC3 generation coming back to commit)
				sl := rng.Intn(cfg.n)
				ge, we := m.residentBind(sl), o.residentBind(sl)
				if (ge == nil) != (we == nil) {
					t.Fatalf("seed %d op %d: bind %d: %v, the old code %v", seed, op, sl, ge, we)
				}
				binds++
				same("bind", op)
			case k < 18: // the fields written directly, as tests do
				ids := tokens(1 + rng.Intn(200))
				m.resIDs, o.resIDs = slices.Clone(ids), slices.Clone(ids)
				lora := loras[rng.Intn(len(loras))]
				m.resIDsLora, o.resIDsLora = lora, lora
				m.resImgBlocks, o.resImg = nil, nil
			default:
				m.residentForgetIDs()
				o.residentForgetIDs()
				same("forget", op)
			}
		}
	}
	if reused < steps/20 || binds == 0 {
		t.Fatalf("%d steps reused a prefix %d times and bound by number %d times: the walk is not exercising reuse", steps, reused, binds)
	}
}
