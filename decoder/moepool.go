package decoder

import (
	"fmt"
	"io"
	"os"
	"unsafe"

	"github.com/townsendmerino/aikit/linalg"
)

// Lever 1b (docs/completed/task-moe-streaming.md): an owned-buffer pread cache for MoE expert
// weights, replacing expertPager's mmap+madvise mode. Darwin's MADV_DONTNEED is a documented
// no-op (madvise_darwin.go) -- an mmap-aliased cache can WILLNEED bytes in but can never
// actually release them, so it gives NO real RAM cap on macOS. Owned buffers the pool itself
// allocates and reuses give a firm cap on every platform: eviction here means "this slot's
// bytes may now be overwritten", genuinely freeing that RAM for the next miss.
//
// Mirrors metal/expertpool.go's shape (measured 1.26x faster darwin cold-stage vs mmap
// byte-copy, zero page faults -- gemma4_moe.go) translated from GPU Buffer writes to plain
// []byte + WeightMat repointing via aikit/linalg's exported Wrap*/accessor API. No aikit API
// change is needed: scales/rows/cols/group are already heap copies untouched by paging (only
// the packed nibble/code payload bytes ever alias the mmap -- decoder/serialize.go's
// giwReader.weightMat), and a WeightMat can be rebuilt from those plus a freshly pread payload.
//
// Opt-in only (GOINFER_MOE_PREAD_CPU=1) pending a same-session measurement against the mmap
// path on real hardware -- see task-moe-streaming.md's Lever 1a for why "should be faster"
// alone was not trusted here without a same-machine A/B.

// expertFieldKind selects which WeightMat representation a paged field holds, and therefore
// which Wrap* constructor rebuilds it after a pread refill.
type expertFieldKind int

const (
	fieldInt4Canonical expertFieldKind = iota
	fieldInt4Row4Only
	fieldInt8
)

// expertField is one paged payload span inside an expert's weight bundle: the WeightMat field
// to repoint on refill, its exact (unaligned) byte range in the .giw file, and the metadata
// needed to rebuild an equivalent WeightMat. A Mixtral-style expert has three fields (Gate,
// Up, Down -- separate, non-adjacent .giw tensors); a Gemma4 expert has two (the fused
// gate‖up bundle, Down).
type expertField struct {
	wm         *linalg.WeightMat
	kind       expertFieldKind
	off        int64
	n          int
	scales     []float32
	group      int  // int4 only
	w8a8       bool // int8 only
	rows, cols int
}

// buildExpertField resolves wm's paged payload span against the mmap window [base,end) and
// returns the field descriptor, or ok=false if wm has no pageable (mmap-aliased, quantized)
// payload. Row4 is preferred over canonical when both are present -- the M=1 decode kernel
// reads only row4 when it exists (moepaging.go's addExpert has the same preference, for the
// same reason: registering both would double-fetch on every miss).
//
// Deliberately uses the RAW unaligned accessor slices (Int4Row4/Int4/Int8), never
// MappedSpan/MappedSpanRow4 -- those round to a page-aligned interior, which is correct for an
// madvise hint but WRONG for a pread offset/length, which must be byte-exact.
func buildExpertField(wm *linalg.WeightMat, base, end uintptr) (expertField, bool) {
	if packed4, scales4, ok := wm.Int4Row4(); ok {
		if off, n, ok := containedOffset(packed4, base, end); ok {
			_, _, group, _ := wm.Int4() // group is a plain shared field, valid even when Int4()'s own ok is false
			return expertField{wm: wm, kind: fieldInt4Row4Only, off: off, n: n,
				scales: scales4, group: group, rows: wm.Rows(), cols: wm.Cols()}, true
		}
	}
	if q4, q4s, group, ok := wm.Int4(); ok {
		if off, n, ok := containedOffset(q4, base, end); ok {
			return expertField{wm: wm, kind: fieldInt4Canonical, off: off, n: n,
				scales: q4s, group: group, rows: wm.Rows(), cols: wm.Cols()}, true
		}
	}
	if q8, scales, w8a8, ok := wm.Int8(); ok && len(q8) > 0 {
		raw := unsafe.Slice((*byte)(unsafe.Pointer(&q8[0])), len(q8))
		if off, n, ok := containedOffset(raw, base, end); ok {
			return expertField{wm: wm, kind: fieldInt8, off: off, n: n,
				scales: scales, w8a8: w8a8, rows: wm.Rows(), cols: wm.Cols()}, true
		}
	}
	return expertField{}, false
}

// containedOffset returns raw's file offset within a mapping spanning [base,end) -- the
// mapping covers the WHOLE .giw file from byte 0 (aikit/mmap.MapReadOnly maps the file at
// offset 0), so a byte's file offset is simply its mapping address minus base. ok=false if raw
// is empty or heap-backed (not part of the mapping).
func containedOffset(raw []byte, base, end uintptr) (off int64, n int, ok bool) {
	if len(raw) == 0 {
		return 0, 0, false
	}
	start := uintptr(unsafe.Pointer(&raw[0]))
	if start < base || start+uintptr(len(raw)) > end {
		return 0, 0, false
	}
	return int64(start - base), len(raw), true
}

// refill re-preads this field's payload into dst (already sized to f.n) and repoints f.wm at
// it, reconstructing the same rows/cols/group/scales the original WeightMat had -- only the
// payload bytes' physical location changes.
func (f *expertField) refill(fd *os.File, dst []byte) error {
	if err := preadFull(fd, dst, f.off); err != nil {
		return err
	}
	switch f.kind {
	case fieldInt4Row4Only:
		wm, ok := linalg.WrapInt4Row4Only(dst, f.scales, f.rows, f.cols, f.group)
		if !ok {
			return fmt.Errorf("WrapInt4Row4Only rejected rows=%d cols=%d group=%d", f.rows, f.cols, f.group)
		}
		*f.wm = wm
	case fieldInt4Canonical:
		*f.wm = linalg.WrapInt4(dst, f.scales, f.rows, f.cols, f.group)
	case fieldInt8:
		q8 := unsafe.Slice((*int8)(unsafe.Pointer(&dst[0])), len(dst))
		*f.wm = linalg.WrapInt8(q8, f.scales, f.rows, f.cols, f.w8a8)
	}
	return nil
}

// preadFull reads exactly len(dst) bytes from f at off, looping on short reads. os.File.ReadAt
// is pread(2) on unix -- positional, so concurrent preads on the same *os.File from different
// goroutines are safe (unlike Read, it never moves a shared file offset).
func preadFull(f *os.File, dst []byte, off int64) error {
	for done := 0; done < len(dst); {
		n, err := f.ReadAt(dst[done:], off+int64(done))
		done += n
		if err != nil {
			if err == io.EOF && done >= len(dst) {
				break
			}
			return err
		}
	}
	return nil
}

// poolMember is one paged expert's field set, keyed by the same stable identity
// newExpertPager's mmap-mode members use (the address of the expert's primary weight struct).
type poolMember struct {
	key    unsafe.Pointer
	fields []expertField
}

// expertBufferPool is a bounded, fixed-N-slot owned-buffer pread cache for MoE expert weights.
//
// Concurrency: the pool has NO internal locking of its own -- expertPager.Lock/Unlock wraps
// every touch() AND the caller's subsequent matmul reads of the repointed WeightMat fields (see
// expertPager's doc comment). A slot's buffer must not be refilled by a competing miss while a
// concurrent decode stream is still reading it; holding the pager's lock across one MoE FFN
// call's touch+compute is what prevents that. This serializes MoE compute globally across
// concurrent streams while pread mode is active -- a coarser trade than the mmap path (whose
// bytes never move, so it never needed this). A finer per-slot pin/refcount scheme is a named
// follow-up in task-moe-streaming.md, not implemented here.
type expertBufferPool struct {
	fd *os.File

	slotSize int // sum of one member's field byte lengths -- identical for every member

	slotBuf    [][]byte         // nSlots, each slotSize bytes, owned (never aliases the mmap)
	slotExpert []unsafe.Pointer // slot -> resident member key, nil if free
	where      map[unsafe.Pointer]int
	lru        []int // slot indices, least-recently-used first

	members map[unsafe.Pointer][]expertField

	hits, misses, evictions int64
	bytesRead               int64 // total bytes actually pread from disk over every miss
}

// newExpertBufferPool builds a pool over members, opening its own fd on giwPath for pread
// (independent of the mmap's own fd, which aikit/mmap.MapReadOnly closes right after mapping --
// decoder/model.go's GiwPath doc comment). budget bytes are divided into slots sized to one
// member's total field length; the slot count is raised to at least minSlots (the model's
// top-k, so a single token's routing can never thrash the pool it just filled) and capped at
// len(members) (never more slots than there are distinct experts to hold).
func newExpertBufferPool(giwPath string, members []poolMember, budget int64, minSlots int) (*expertBufferPool, error) {
	if len(members) == 0 {
		return nil, fmt.Errorf("no pageable members")
	}
	layout := make([]int, len(members[0].fields))
	slotSize := 0
	for i, f := range members[0].fields {
		layout[i] = f.n
		slotSize += f.n
	}
	if slotSize == 0 {
		return nil, fmt.Errorf("zero-byte member field layout")
	}
	for _, m := range members {
		if len(m.fields) != len(layout) {
			return nil, fmt.Errorf("member field count %d != %d (heterogeneous expert shapes)", len(m.fields), len(layout))
		}
		for i, f := range m.fields {
			if f.n != layout[i] {
				return nil, fmt.Errorf("member field %d length %d != %d (heterogeneous expert shapes)", i, f.n, layout[i])
			}
		}
	}
	nSlots := int(budget / int64(slotSize))
	if nSlots < minSlots {
		nSlots = minSlots
	}
	if nSlots < 1 {
		nSlots = 1
	}
	if nSlots > len(members) {
		nSlots = len(members)
	}
	fd, err := os.Open(giwPath)
	if err != nil {
		return nil, fmt.Errorf("open %s for pread: %w", giwPath, err)
	}
	p := &expertBufferPool{
		fd:         fd,
		slotSize:   slotSize,
		slotBuf:    make([][]byte, nSlots),
		slotExpert: make([]unsafe.Pointer, nSlots),
		where:      make(map[unsafe.Pointer]int, nSlots),
		lru:        make([]int, 0, nSlots),
		members:    make(map[unsafe.Pointer][]expertField, len(members)),
	}
	for s := range nSlots {
		p.slotBuf[s] = make([]byte, slotSize)
	}
	for _, m := range members {
		p.members[m.key] = m.fields
	}
	return p, nil
}

// ensure makes key resident: MRU + no-op on a hit, or picks a slot (evicting the LRU tail if
// the pool is full), preads every field into it, and repoints key's WeightMat fields at the
// filled buffer. Caller must hold expertPager's lock (see expertPager's doc comment) --
// ensure does no locking of its own.
func (p *expertBufferPool) ensure(key unsafe.Pointer) error {
	if s, ok := p.where[key]; ok {
		p.hits++
		p.touchLRU(s)
		return nil
	}
	fields, ok := p.members[key]
	if !ok {
		return nil // not a managed member
	}
	p.misses++
	s := p.pickSlot()
	if old := p.slotExpert[s]; old != nil {
		delete(p.where, old)
		p.evictions++
	}
	buf := p.slotBuf[s]
	off := 0
	for i := range fields {
		f := &fields[i]
		dst := buf[off : off+f.n]
		if err := f.refill(p.fd, dst); err != nil {
			return fmt.Errorf("refill field %d/%d (%d bytes @ %d): %w", i, len(fields), f.n, f.off, err)
		}
		p.bytesRead += int64(f.n)
		off += f.n
	}
	p.slotExpert[s] = key
	p.where[key] = s
	p.touchLRU(s)
	return nil
}

// pickSlot returns a free slot if one exists, else evicts and returns the LRU tail.
func (p *expertBufferPool) pickSlot() int {
	for s, e := range p.slotExpert {
		if e == nil {
			return s
		}
	}
	victim := p.lru[0]
	p.lru = p.lru[1:]
	return victim
}

// touchLRU marks slot s most-recently-used.
func (p *expertBufferPool) touchLRU(s int) {
	for i, v := range p.lru {
		if v == s {
			p.lru = append(p.lru[:i], p.lru[i+1:]...)
			break
		}
	}
	p.lru = append(p.lru, s)
}

func (p *expertBufferPool) stats() (hits, misses, evictions int64) {
	return p.hits, p.misses, p.evictions
}

// budget returns the pool's actual resident cap in bytes: slot count × slot size.
func (p *expertBufferPool) budget() int64 { return int64(len(p.slotBuf)) * int64(p.slotSize) }

func (p *expertBufferPool) close() error {
	if p.fd != nil {
		return p.fd.Close()
	}
	return nil
}
