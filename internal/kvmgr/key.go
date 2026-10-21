// Package kvmgr is the KV manager: the one owner of KV metadata for a model's resident sequences. It holds sequences,
// their tables of logical blocks, the content-addressed index of published blocks, reference counts and the LRU of
// cached blocks, and it decides what a new request may reuse. It stores no KV: a backend does, and the manager says
// where each logical block lives. The design is docs/tasks/task-kv-architecture-2026-10.md §5.
//
// It imports nothing from decoder, so its tests run apart from decoder's; decoder derives the Geometry from an
// Architecture and re-exports the types a backend needs.
package kvmgr

import (
	"crypto/sha256"
	"encoding/binary"
)

// BlockRows is B: the positions in one logical block. A multiple of 64 is what every kernel's key tile allows
// (docs/measurements/kv0-tile-census-2026-10-13.md); changing it re-keys every index.
const BlockRows = 64

// Hash is one link of a sequence's chain: the identity of everything that decides the KV bytes of a block and of
// every block before it.
type Hash [sha256.Size]byte

// Root is what a whole chain is built under. Two requests whose roots differ share no block.
type Root struct {
	Model     string // the model's numerics identity: checkpoint, quantisation, backend
	Namespace string // the sharing scope (one per API key; "" is the single local namespace)
	Adapter   uint64 // the compute-time adapter's identity, 0 for the base weights
	Precision string // the KV element type the backend stores ("f32", "f16", "i8")
}

// ImageSpan is one image's rows in a sequence. Its rows attended each other bidirectionally, so a reuse boundary
// never falls inside it. Hash is the content hash of the image bytes; 0 claims nothing and never matches.
type ImageSpan struct {
	Start, Len int
	Hash       uint64
}

func (s ImageSpan) end() int { return s.Start + s.Len }

// Class is the kernel class that wrote a row. Rows of different classes are not bit-identical for the same tokens.
type Class uint8

const (
	// ClassUnknown is a row whose producer the manager was not told. It is reusable by its own sequence and
	// matches no other request.
	ClassUnknown Class = iota
	// ClassExact is a decode row, or a prefill row from the kernels that are bit-identical to decode.
	ClassExact
	// ClassFast is a prefill row from a backend's batched kernels that are not bit-identical to decode.
	ClassFast
)

// ClassRun says which class wrote rows [Start, Start+Len).
type ClassRun struct {
	Start, Len int
	Class      Class
}

// ClassMap is a backend's rule for which class a prefill pass writes. The zero value is a backend with one class at
// every length. The rule is the one docs/measurements/kv0-invariance-2026-10-13.md measured on CUDA: a pass runs the
// fast kernels when it has at least MinRows rows and the prompt length it is judged on is at least Floor.
type ClassMap struct {
	Floor   int // the judged prompt length the fast kernels start at; 0: they are off
	MinRows int // the fewest rows a pass needs to run them
	// Windowed: the model has sliding-window layers. A fast pass over such a layer tiles its keys from where the pass
	// starts, so one that does not start on a block boundary writes rows no other fast pass writes
	// (docs/measurements/kv1-2026-10-13.md, G2b). PassAt records those as ClassUnknown.
	Windowed bool
}

// Pass is the class a prefill pass of rows rows writes when it is judged on a prompt of judged tokens.
func (c ClassMap) Pass(rows, judged int) Class {
	if c.Floor > 0 && judged >= c.Floor && rows >= c.MinRows {
		return ClassFast
	}
	return ClassExact
}

// PassAt is Pass for a pass that starts at position start. On a model with sliding-window layers a fast pass that starts
// off a block boundary is ClassUnknown: its rows are valid for the sequence that wrote them and equal to no other
// request's.
func (c ClassMap) PassAt(start, rows, judged int) Class {
	cls := c.Pass(rows, judged)
	if cls == ClassFast && c.Windowed && start%BlockRows != 0 {
		return ClassUnknown
	}
	return cls
}

// MinLeadReuse is the shortest reused prefix kept when the rest of the prompt would run a different class: a
// shorter one is a shared lead, cheap to recompute, and keeping it mixes two classes in one prompt.
const MinLeadReuse = 64

// DeclineShortLead returns reuse, or 0 when reuse is a short lead that the prompt's own cold prefill would write
// with the fast class. A prompt whose cold prefill is exact (under the floor, or under MinRows rows) keeps it: the
// reused rows are the same class.
func (c ClassMap) DeclineShortLead(promptLen, reuse int) int {
	if reuse <= 0 || reuse >= MinLeadReuse {
		return reuse
	}
	if c.Pass(promptLen, promptLen) != ClassFast {
		return reuse
	}
	return 0
}

// keyPart names one input of the key, so a test can drop it and show the wrong hit that follows.
type keyPart uint8

const (
	partAdapter keyPart = 1 << iota
	partImage
	partClass
	partPrecision
	partNamespace
	allParts = partAdapter | partImage | partClass | partPrecision | partNamespace
)

// hasher writes a canonical byte encoding: every field is tagged and every variable-length field carries its length,
// so two different inputs never serialise to the same bytes.
type hasher struct {
	buf []byte
}

func (h *hasher) tag(t byte)   { h.buf = append(h.buf, t) }
func (h *hasher) u64(v uint64) { h.buf = binary.LittleEndian.AppendUint64(h.buf, v) }
func (h *hasher) i64(v int)    { h.u64(uint64(int64(v))) }
func (h *hasher) str(s string) { h.i64(len(s)); h.buf = append(h.buf, s...) }
func (h *hasher) sum() Hash    { return sha256.Sum256(h.buf) }

// rootHash is link −1 of a chain.
// checkpointKey is the key of a recurrent checkpoint at position len(tokens): the chain under root through those tokens.
func checkpointKey(root Hash, tokens []int) Hash {
	h := hasher{buf: make([]byte, 0, 48+8*len(tokens))}
	h.tag('K')
	h.buf = append(h.buf, root[:]...)
	h.i64(len(tokens))
	for _, t := range tokens {
		h.i64(t)
	}
	return h.sum()
}

func rootHash(r Root, parts keyPart) Hash {
	h := hasher{buf: make([]byte, 0, 96+len(r.Model)+len(r.Namespace))}
	h.tag('R')
	h.i64(BlockRows)
	h.str(r.Model)
	if parts&partNamespace != 0 {
		h.tag('n')
		h.str(r.Namespace)
	}
	if parts&partAdapter != 0 {
		h.tag('a')
		h.u64(r.Adapter)
	}
	if parts&partPrecision != 0 {
		h.tag('p')
		h.str(r.Precision)
	}
	return h.sum()
}

// blockHash is link i of a chain: the previous link, block i's tokens, every image span that touches the block (its
// content hash, where it starts relative to the block, its length), and the class of each of the block's rows.
// tokens is the whole sequence; block i must be full.
func blockHash(h *hasher, prev Hash, i int, tokens []int, images []ImageSpan, classes []ClassRun, parts keyPart) Hash {
	lo, hi := i*BlockRows, (i+1)*BlockRows
	h.buf = h.buf[:0]
	h.tag('B')
	h.buf = append(h.buf, prev[:]...)
	for _, t := range tokens[lo:hi] {
		h.i64(t)
	}
	if parts&partImage != 0 {
		for _, s := range images {
			if s.Start < hi && s.end() > lo {
				h.tag('i')
				h.u64(s.Hash)
				h.i64(s.Start - lo)
				h.i64(s.Len)
			}
		}
	}
	if parts&partClass != 0 {
		// Row by row, run-length encoded inside the block, so the same classes split into different runs hash alike.
		cur, n := classAt(classes, lo), 0
		for p := lo; p < hi; p++ {
			c := classAt(classes, p)
			if c != cur {
				h.tag('c')
				h.buf = append(h.buf, byte(cur))
				h.i64(n)
				cur, n = c, 0
			}
			n++
		}
		h.tag('c')
		h.buf = append(h.buf, byte(cur))
		h.i64(n)
	}
	return h.sum()
}

// classAt is the class of row p; a row no run covers is ClassUnknown.
func classAt(classes []ClassRun, p int) Class {
	for _, r := range classes {
		if p >= r.Start && p < r.Start+r.Len {
			return r.Class
		}
	}
	return ClassUnknown
}

// clipClasses is classes restricted to rows [0, n), with adjacent runs of one class merged.
func clipClasses(classes []ClassRun, n int) []ClassRun {
	var out []ClassRun
	for _, r := range classes {
		if r.Start >= n || r.Len <= 0 {
			continue
		}
		if r.Start+r.Len > n {
			r.Len = n - r.Start
		}
		out = appendClass(out, r)
	}
	return out
}

// appendClass appends r, merging it into the last run when it continues it with the same class.
func appendClass(runs []ClassRun, r ClassRun) []ClassRun {
	if r.Len <= 0 {
		return runs
	}
	if k := len(runs) - 1; k >= 0 && runs[k].Class == r.Class && runs[k].Start+runs[k].Len == r.Start {
		runs[k].Len += r.Len
		return runs
	}
	return append(runs, r)
}
