package paging

import (
	"errors"
	"fmt"
	"io"
	"os"
	"unsafe"

	"github.com/townsendmerino/aikit/linalg"
)

// ExpertBufferPool is an owned-buffer pread cache for MoE expert weights, replacing expertPager's mmap+madvise mode
// where that mode cannot cap RAM: darwin's MADV_DONTNEED is a documented no-op (madvise_darwin.go), so an mmap-aliased
// cache can WILLNEED bytes in but never release them. Owned buffers the pool allocates and reuses give a firm cap on
// every platform: eviction here means "this slot's bytes may now be overwritten", which genuinely frees that RAM for
// the next miss. It is the default on darwin; elsewhere --moe-pager=pool (Options.MoEPager) selects it.
// Darwin rates: docs/measurements/moe-pager-mode-darwin-2026-09-23.md.
//
// It mirrors metal/expertpool.go's shape, translated from GPU Buffer writes to plain []byte plus WeightMat repointing
// via aikit/linalg's exported Wrap*/accessor API. No aikit API change is needed: scales, rows, cols and group are heap
// copies untouched by paging (only the packed nibble/code payload bytes alias the mmap, decoder/serialize.go's
// giwReader.weightMat), and a WeightMat can be rebuilt from those plus a freshly pread payload.

// ExpertFieldKind selects which WeightMat representation a paged field holds, and therefore
// which Wrap* constructor rebuilds it after a pread refill.
type ExpertFieldKind int

const (
	FieldInt4Canonical ExpertFieldKind = iota
	FieldInt4Row4Only
	FieldInt8
)

// ExpertField is one paged payload span inside an expert's weight bundle: the WeightMat field
// to repoint on refill, its exact (unaligned) byte range in the .giw file, and the metadata
// needed to rebuild an equivalent WeightMat. A Mixtral-style expert has three fields (Gate,
// Up, Down -- separate, non-adjacent .giw tensors); a Gemma4 expert has two (the fused
// gate‖up bundle, Down).
type ExpertField struct {
	wm         *linalg.WeightMat
	kind       ExpertFieldKind
	off        int64
	n          int
	scales     []float32 // int8 only
	scales16   []uint16  // int4 only (binary16, as the WeightMat stores them)
	group      int       // int4 only
	w8a8       bool      // int8 only
	rows, cols int
}

// BuildExpertField resolves wm's paged payload span against the mmap window [base,end) and
// returns the field descriptor, or ok=false if wm has no pageable (mmap-aliased, quantized)
// payload. Row4 is preferred over canonical when both are present -- the M=1 decode kernel
// reads only row4 when it exists (moepaging.go's addExpert has the same preference, for the
// same reason: registering both would double-fetch on every miss).
//
// Deliberately uses the RAW unaligned accessor slices (Int4Row4/Int4/Int8), never
// MappedSpan/MappedSpanRow4 -- those round to a page-aligned interior, which is correct for an
// madvise hint but WRONG for a pread offset/length, which must be byte-exact.
func BuildExpertField(wm *linalg.WeightMat, base, end uintptr) (ExpertField, bool) {
	if packed4, scales4, ok := wm.Int4Row4F16(); ok {
		if off, n, ok := containedOffset(packed4, base, end); ok {
			_, _, group, _ := wm.Int4F16() // group is a plain shared field, valid even when Int4F16()'s own ok is false
			return ExpertField{wm: wm, kind: FieldInt4Row4Only, off: off, n: n,
				scales16: scales4, group: group, rows: wm.Rows(), cols: wm.Cols()}, true
		}
	}
	if q4, q4s, group, ok := wm.Int4F16(); ok {
		if off, n, ok := containedOffset(q4, base, end); ok {
			return ExpertField{wm: wm, kind: FieldInt4Canonical, off: off, n: n,
				scales16: q4s, group: group, rows: wm.Rows(), cols: wm.Cols()}, true
		}
	}
	if q8, scales, w8a8, ok := wm.Int8(); ok && len(q8) > 0 {
		raw := unsafe.Slice((*byte)(unsafe.Pointer(&q8[0])), len(q8))
		if off, n, ok := containedOffset(raw, base, end); ok {
			return ExpertField{wm: wm, kind: FieldInt8, off: off, n: n,
				scales: scales, w8a8: w8a8, rows: wm.Rows(), cols: wm.Cols()}, true
		}
	}
	return ExpertField{}, false
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

// wrap reconstructs a WeightMat backed by dst with this field's layout and scales.
func (f *ExpertField) wrap(dst []byte) (linalg.WeightMat, error) {
	switch f.kind {
	case FieldInt4Row4Only:
		wm, ok := linalg.WrapInt4(dst, f.scales16, f.rows, f.cols, f.group, linalg.LayoutRow4)
		if !ok {
			return linalg.WeightMat{}, fmt.Errorf("WrapInt4Row4Only rejected rows=%d cols=%d group=%d", f.rows, f.cols, f.group)
		}
		return wm, nil
	case FieldInt4Canonical:
		wm4, _ := linalg.WrapInt4(dst, f.scales16, f.rows, f.cols, f.group, linalg.LayoutCanonical)
		return wm4, nil
	case FieldInt8:
		q8 := unsafe.Slice((*int8)(unsafe.Pointer(&dst[0])), len(dst))
		return linalg.WrapInt8(q8, f.scales, f.rows, f.cols, f.w8a8), nil
	default:
		return linalg.WeightMat{}, fmt.Errorf("unknown ExpertFieldKind %d", f.kind)
	}
}

// preadFull reads exactly len(dst) bytes from f at off, looping on short reads. os.File.ReadAt
// is pread(2) on unix -- positional, so concurrent preads on the same *os.File from different
// goroutines are safe (unlike Read, it never moves a shared file offset).
func preadFull(f *os.File, dst []byte, off int64) error {
	for done := 0; done < len(dst); {
		n, err := f.ReadAt(dst[done:], off+int64(done))
		done += n
		if err != nil {
			if errors.Is(err, io.EOF) && done >= len(dst) {
				break
			}
			return err
		}
	}
	return nil
}

// PoolMember is one paged expert's field set, keyed by the same stable identity
// newExpertPager's mmap-mode members use (the address of the expert's primary weight struct).
type PoolMember struct {
	Key    unsafe.Pointer
	Fields []ExpertField
}

// ExpertBufferPool is a bounded, fixed-N-slot owned-buffer pread cache for MoE expert weights.
//
// Concurrency: the pool has NO internal locking of its own -- expertPager.Lock/Unlock wraps
// every touch() AND the caller's subsequent matmul reads of the repointed WeightMat fields (see
// expertPager's doc comment). A slot's buffer must not be refilled by a competing miss while a
// concurrent decode stream is still reading it; holding the pager's lock across one MoE FFN
// call's touch+compute is what prevents that. This serializes MoE compute globally across
// concurrent streams while pread mode is active -- a coarser trade than the mmap path (whose
// bytes never move, so it never needed this). A finer per-slot pin/refcount scheme is a named
// follow-up in task-moe-streaming.md, not implemented here.
type ExpertBufferPool struct {
	fd *os.File

	slotSize int // sum of one member's field byte lengths -- identical for every member

	slotBuf    [][]byte         // nSlots, each slotSize bytes, owned (never aliases the mmap)
	slotExpert []unsafe.Pointer // slot -> resident member key, nil if free
	where      map[unsafe.Pointer]int
	lru        []int // slot indices, least-recently-used first

	members map[unsafe.Pointer][]ExpertField

	hits, misses, evictions int64
	bytesRead               int64 // total bytes actually pread from disk over every miss
}

// NewExpertBufferPool builds a pool over members, opening its own fd on giwPath for pread
// (independent of the mmap's own fd, which aikit/mmap.MapReadOnly closes right after mapping --
// decoder/model.go's GiwPath doc comment). budget bytes are divided into slots sized to one
// member's total field length; the slot count is raised to at least minSlots (the model's
// top-k, so a single token's routing can never thrash the pool it just filled) and capped at
// len(members) (never more slots than there are distinct experts to hold).
func NewExpertBufferPool(giwPath string, members []PoolMember, budget int64, minSlots int) (*ExpertBufferPool, error) {
	if len(members) == 0 {
		return nil, fmt.Errorf("no pageable members")
	}
	layout := make([]int, len(members[0].Fields))
	slotSize := 0
	for i, f := range members[0].Fields {
		layout[i] = f.n
		slotSize += f.n
	}
	if slotSize == 0 {
		return nil, fmt.Errorf("zero-byte member field layout")
	}
	for _, m := range members {
		if len(m.Fields) != len(layout) {
			return nil, fmt.Errorf("member field count %d != %d (heterogeneous expert shapes)", len(m.Fields), len(layout))
		}
		for i, f := range m.Fields {
			if f.n != layout[i] {
				return nil, fmt.Errorf("member field %d length %d != %d (heterogeneous expert shapes)", i, f.n, layout[i])
			}
		}
	}
	nSlots := min(max(max(int(budget/int64(slotSize)), minSlots), 1), len(members))
	fd, err := os.Open(giwPath)
	if err != nil {
		return nil, fmt.Errorf("open %s for pread: %w", giwPath, err)
	}
	p := &ExpertBufferPool{
		fd:         fd,
		slotSize:   slotSize,
		slotBuf:    make([][]byte, nSlots),
		slotExpert: make([]unsafe.Pointer, nSlots),
		where:      make(map[unsafe.Pointer]int, nSlots),
		lru:        make([]int, 0, nSlots),
		members:    make(map[unsafe.Pointer][]ExpertField, len(members)),
	}
	for s := range nSlots {
		p.slotBuf[s] = make([]byte, slotSize)
	}
	for _, m := range members {
		p.members[m.Key] = m.Fields
	}
	return p, nil
}

// ensure makes key resident: MRU + no-op on a hit, or picks a slot (evicting the LRU tail if
// the pool is full), preads every field into it, and repoints key's WeightMat fields at the
// filled buffer. Caller must hold expertPager's lock (see expertPager's doc comment) --
// ensure does no locking of its own.
func (p *ExpertBufferPool) Ensure(key unsafe.Pointer) error {
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
		p.slotExpert[s] = nil
		p.evictions++
	}
	buf := p.slotBuf[s]
	off := 0
	for i := range fields {
		f := &fields[i]
		dst := buf[off : off+f.n]
		if err := preadFull(p.fd, dst, f.off); err != nil {
			return fmt.Errorf("refill field %d/%d (%d bytes @ %d): %w", i, len(fields), f.n, f.off, err)
		}
		p.bytesRead += int64(f.n)
		off += f.n
	}
	var rebuiltBuf [4]linalg.WeightMat
	rebuilt := rebuiltBuf[:0]
	if len(fields) > len(rebuiltBuf) {
		rebuilt = make([]linalg.WeightMat, 0, len(fields))
	}
	off = 0
	for i := range fields {
		f := &fields[i]
		wm, err := f.wrap(buf[off : off+f.n])
		if err != nil {
			return fmt.Errorf("refill field %d/%d (%d bytes @ %d): %w", i, len(fields), f.n, f.off, err)
		}
		rebuilt = append(rebuilt, wm)
		off += f.n
	}
	for i := range fields {
		*fields[i].wm = rebuilt[i]
	}
	p.slotExpert[s] = key
	p.where[key] = s
	p.touchLRU(s)
	return nil
}

// pickSlot returns a free slot if one exists, else evicts and returns the LRU tail.
func (p *ExpertBufferPool) pickSlot() int {
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
func (p *ExpertBufferPool) touchLRU(s int) {
	for i, v := range p.lru {
		if v == s {
			p.lru = append(p.lru[:i], p.lru[i+1:]...)
			break
		}
	}
	p.lru = append(p.lru, s)
}

func (p *ExpertBufferPool) Stats() (hits, misses, evictions int64) {
	return p.hits, p.misses, p.evictions
}

// budget returns the pool's actual resident cap in bytes: slot count × slot size.
func (p *ExpertBufferPool) Budget() int64 { return int64(len(p.slotBuf)) * int64(p.slotSize) }

func (p *ExpertBufferPool) Close() error {
	if p.fd != nil {
		return p.fd.Close()
	}
	return nil
}

// ByteLen returns the packed payload length in bytes.
func (f ExpertField) ByteLen() int { return f.n }

// NewInt8ExpertField builds an ExpertField for an int8 WeightMat at a known file offset (used by tests).
func NewInt8ExpertField(wm *linalg.WeightMat, off int64, n int, scales []float32, rows, cols int) ExpertField {
	return ExpertField{wm: wm, kind: FieldInt8, off: off, n: n, scales: scales, rows: rows, cols: cols}
}

// SlotCount returns the number of owned buffer slots in the pool.
func (p *ExpertBufferPool) SlotCount() int { return len(p.slotBuf) }

// SlotOf returns the resident slot index for key, or -1 if not currently loaded.
func (p *ExpertBufferPool) SlotOf(key unsafe.Pointer) int {
	if s, ok := p.where[key]; ok {
		return s
	}
	return -1
}
