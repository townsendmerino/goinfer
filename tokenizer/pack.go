package tokenizer

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"sync"
)

// A pack is a prebuilt binary form of a SentencePiece-style (modeGemma) BPE tokenizer: the tables parseTokenizerJSON
// builds from a tokenizer.json, stored so LoadPack can rebuild them without parsing JSON or looking up merge pieces by
// text. BuildPack makes one from tokenizer.json bytes; the same input always gives the same bytes, so a pack's SHA-256
// can be pinned beside the tokenizer.json it came from. A pack carries no chat template and reads no sibling file.
//
// Layout, little-endian, every count bounded before anything is allocated:
//
//	magic "GITOKPK1", version u32, flags u32 (packFlag*)
//	special: BOS, EOS, Pad, StartOfTurn, EndOfTurn, unkID as i32
//	strings: count u32, blob length u32, offsets [count+1]u32, blob
//	pieces u32 (idToPiece is strings[0:pieces]); unkPiece string index u32
//	vocab: count u32, then (string index u32, id i32) per entry
//	merges: count u32, then (left i32, right i32) per merge in list order; a value >= 0 is a piece id,
//	        a negative value v names string -(v+1) (a piece outside the vocabulary, mergeTable.byStr)
//	added: count u32, then (string index u32, id i32, flags u32) per added token (packAddedRStrip)
//	SHA-256 of every byte before it
const packMagic = "GITOKPK1"

const packVersion = 1

const (
	packFlagByteFallback = 1 << iota
	packFlagIgnoreMerges
	packFlagPrependSpace
	packFlagStripLeadingSpace
)

const packAddedRStrip = 1

// maxPackStrings bounds the string table: every piece, plus the few strings outside it.
const maxPackStrings = 2*maxTokenID + 1024

// ErrPack is the error class of a pack LoadPack refuses.
var ErrPack = errors.New("tokenizer: invalid pack")

// BuildPack parses tokenizer.json bytes and returns its pack. It accepts the SentencePiece-style byte-fallback BPE
// family with an explicit merge list (Gemma); a byte-level or score-ranked tokenizer is an error.
func BuildPack(raw []byte) ([]byte, error) {
	t, err := LoadJSONBytes(raw)
	if err != nil {
		return nil, err
	}
	if t.mode != modeGemma || t.scoreRank != nil {
		return nil, fmt.Errorf("tokenizer.BuildPack: only the SentencePiece-style BPE family is packed")
	}
	var tj tokenizerJSON
	if err := json.Unmarshal(raw, &tj); err != nil {
		return nil, fmt.Errorf("tokenizer.BuildPack: %w", err)
	}
	merges, err := parseMerges(tj.Model.Merges)
	if err != nil {
		return nil, fmt.Errorf("tokenizer.BuildPack: %w", err)
	}

	// String table: every piece by id, then the strings outside it, each once, in first-use order.
	strs := append([]string(nil), t.idToPiece...)
	extra := map[string]uint32{}
	ref := func(s string, id int32) uint32 {
		if id >= 0 && int(id) < len(t.idToPiece) && t.idToPiece[id] == s {
			return uint32(id)
		}
		if i, ok := extra[s]; ok {
			return i
		}
		i := uint32(len(strs))
		strs = append(strs, s)
		extra[s] = i
		return i
	}
	unkRef := ref(t.unkPiece, t.unkID)

	// Vocabulary entries by id, then by text, so the bytes do not depend on map order.
	type vent struct {
		str uint32
		id  int32
		s   string
	}
	vocab := make([]vent, 0, len(t.vocab))
	for s, id := range t.vocab {
		vocab = append(vocab, vent{id: id, s: s})
	}
	slices.SortFunc(vocab, func(a, b vent) int {
		if a.id != b.id {
			return int(a.id) - int(b.id)
		}
		return strings.Compare(a.s, b.s)
	})
	for i := range vocab {
		vocab[i].str = ref(vocab[i].s, vocab[i].id)
	}
	mergeRefs := make([][2]int32, len(merges))
	for i, m := range merges {
		for k := range 2 {
			if id := t.pieceID(m[k]); id >= 0 {
				mergeRefs[i][k] = id
			} else {
				mergeRefs[i][k] = -int32(ref(m[k], -1)) - 1
			}
		}
	}
	type aent struct {
		str   uint32
		id    int32
		flags uint32
	}
	added := make([]aent, len(tj.AddedTokens))
	for i, a := range tj.AddedTokens {
		added[i] = aent{str: ref(a.Content, a.ID), id: a.ID}
		if a.RStrip {
			added[i].flags = packAddedRStrip
		}
	}

	var b bytes.Buffer
	w := func(v any) { _ = binary.Write(&b, binary.LittleEndian, v) }
	b.WriteString(packMagic)
	w(uint32(packVersion))
	var flags uint32
	if t.byteFallback {
		flags |= packFlagByteFallback
	}
	if t.ignoreMerges {
		flags |= packFlagIgnoreMerges
	}
	if t.prependSpace {
		flags |= packFlagPrependSpace
	}
	if t.stripLeadingSpace {
		flags |= packFlagStripLeadingSpace
	}
	w(flags)
	sp := t.special
	w([6]int32{int32(sp.BOS), int32(sp.EOS), int32(sp.Pad), int32(sp.StartOfTurn), int32(sp.EndOfTurn), t.unkID})
	w(uint32(len(strs)))
	blobLen := 0
	for _, s := range strs {
		blobLen += len(s)
	}
	if blobLen > math.MaxUint32 {
		return nil, fmt.Errorf("tokenizer.BuildPack: string table too large")
	}
	w(uint32(blobLen))
	off := uint32(0)
	for _, s := range strs {
		w(off)
		off += uint32(len(s))
	}
	w(off)
	for _, s := range strs {
		b.WriteString(s)
	}
	w(uint32(len(t.idToPiece)))
	w(unkRef)
	w(uint32(len(vocab)))
	for _, v := range vocab {
		w(v.str)
		w(v.id)
	}
	w(uint32(len(mergeRefs)))
	for _, m := range mergeRefs {
		w(m)
	}
	w(uint32(len(added)))
	for _, a := range added {
		w(a.str)
		w(a.id)
		w(a.flags)
	}
	sum := sha256.Sum256(b.Bytes())
	b.Write(sum[:])
	return b.Bytes(), nil
}

// packReader reads a pack front to back. Every read is bounds-checked; the first failure sticks.
type packReader struct {
	b   []byte
	off int
	err error
}

func (r *packReader) fail(what string) {
	if r.err == nil {
		r.err = fmt.Errorf("%w: %s at byte %d", ErrPack, what, r.off)
	}
}

func (r *packReader) take(n int, what string) []byte {
	if r.err != nil {
		return nil
	}
	if n < 0 || n > len(r.b)-r.off {
		r.fail(what)
		return nil
	}
	p := r.b[r.off : r.off+n]
	r.off += n
	return p
}

func (r *packReader) u32(what string) uint32 {
	p := r.take(4, what)
	if p == nil {
		return 0
	}
	return binary.LittleEndian.Uint32(p)
}

func (r *packReader) i32(what string) int32 { return int32(r.u32(what)) }

// count reads a u32 count of records of size bytes each and checks that many records can follow.
func (r *packReader) count(size int, limit int, what string) int {
	n := int(r.u32(what))
	if r.err != nil {
		return 0
	}
	if n > limit || n*size > len(r.b)-r.off {
		r.fail(what)
		return 0
	}
	return n
}

// LoadPack builds a Tokenizer from a pack made by BuildPack. The result encodes and decodes exactly as the
// tokenizer.json the pack was built from. A malformed pack is an ErrPack error, never a panic. The pack bytes are
// not retained: strings are copied into one block that the vocabulary and the piece table share.
func LoadPack(pack []byte) (*Tokenizer, error) {
	if len(pack) < len(packMagic)+sha256.Size {
		return nil, fmt.Errorf("%w: too short", ErrPack)
	}
	body, sum := pack[:len(pack)-sha256.Size], pack[len(pack)-sha256.Size:]
	if got := sha256.Sum256(body); !bytes.Equal(got[:], sum) {
		return nil, fmt.Errorf("%w: checksum mismatch", ErrPack)
	}
	r := &packReader{b: body}
	if string(r.take(len(packMagic), "magic")) != packMagic {
		return nil, fmt.Errorf("%w: not a tokenizer pack", ErrPack)
	}
	if v := r.u32("version"); r.err == nil && v != packVersion {
		return nil, fmt.Errorf("%w: version %d, want %d", ErrPack, v, packVersion)
	}
	flags := r.u32("flags")
	var special [6]int32
	for i := range special {
		special[i] = r.i32("special tokens")
	}

	nStr := r.count(4, maxPackStrings, "string count")
	blobLen := int(r.u32("string blob length"))
	offs := r.take(4*(nStr+1), "string offsets")
	blobBytes := r.take(blobLen, "string blob")
	if r.err != nil {
		return nil, r.err
	}
	blob := string(blobBytes) // one copy; every piece and vocabulary key is a substring of it
	strs := make([]string, nStr)
	prev := uint32(0)
	for i := 0; i <= nStr; i++ {
		o := binary.LittleEndian.Uint32(offs[4*i:])
		if o < prev || int(o) > blobLen || (i == 0 && o != 0) {
			return nil, fmt.Errorf("%w: string offsets", ErrPack)
		}
		if i > 0 {
			strs[i-1] = blob[prev:o]
		}
		prev = o
	}
	if int(prev) != blobLen {
		return nil, fmt.Errorf("%w: string blob length", ErrPack)
	}
	str := func(i uint32) (string, bool) {
		if int(i) >= nStr {
			return "", false
		}
		return strs[i], true
	}

	nPieces := int(r.u32("piece count"))
	unkRef := r.u32("unk piece")
	if r.err != nil {
		return nil, r.err
	}
	if nPieces < 1 || nPieces > nStr || nPieces > maxTokenID+1 {
		return nil, fmt.Errorf("%w: piece count", ErrPack)
	}
	unkPiece, ok := str(unkRef)
	if !ok {
		return nil, fmt.Errorf("%w: unk piece", ErrPack)
	}
	nVocab := r.count(8, maxPackStrings, "vocabulary count")
	vrec := r.take(8*nVocab, "vocabulary")
	nMerges := r.count(8, math.MaxInt32, "merge count")
	mrec := r.take(8*nMerges, "merges")
	nAdded := r.count(12, maxTokenID+1, "added-token count")
	arec := r.take(12*nAdded, "added tokens")
	if r.err != nil {
		return nil, r.err
	}
	if r.off != len(body) {
		return nil, fmt.Errorf("%w: %d trailing bytes", ErrPack, len(body)-r.off)
	}
	inID := func(id int32) bool { return id >= 0 && int(id) < nPieces }
	for _, id := range special[:5] {
		if id < -1 || id >= int32(nPieces) {
			return nil, fmt.Errorf("%w: special token id %d", ErrPack, id)
		}
	}
	if !inID(special[5]) {
		return nil, fmt.Errorf("%w: unk id %d", ErrPack, special[5])
	}

	t := &Tokenizer{
		mode:              modeGemma,
		idToPiece:         strs[:nPieces],
		byteFallback:      flags&packFlagByteFallback != 0,
		ignoreMerges:      flags&packFlagIgnoreMerges != 0,
		prependSpace:      flags&packFlagPrependSpace != 0,
		stripLeadingSpace: flags&packFlagStripLeadingSpace != 0,
		unkPiece:          unkPiece,
		unkID:             special[5],
		byteToVal:         make(map[int32]byte, 256),
		special: SpecialTokens{BOS: int(special[0]), EOS: int(special[1]), Pad: int(special[2]),
			StartOfTurn: int(special[3]), EndOfTurn: int(special[4])},
	}

	// The vocabulary map and the merge table are independent: build them at once.
	var wg sync.WaitGroup
	var vocabErr, mergeErr error
	wg.Go(func() {
		vocab := make(map[string]int32, nVocab)
		for i := range nVocab {
			s, ok := str(binary.LittleEndian.Uint32(vrec[8*i:]))
			id := int32(binary.LittleEndian.Uint32(vrec[8*i+4:]))
			if !ok || !inID(id) {
				vocabErr = fmt.Errorf("%w: vocabulary entry %d", ErrPack, i)
				return
			}
			vocab[s] = id
		}
		t.vocab = vocab
	})
	wg.Go(func() {
		m := mergeTable{flat: newFlatMerges(nMerges)}
		for i := range nMerges {
			l := int32(binary.LittleEndian.Uint32(mrec[8*i:]))
			rr := int32(binary.LittleEndian.Uint32(mrec[8*i+4:]))
			if l >= 0 && rr >= 0 {
				if !inID(l) || !inID(rr) {
					mergeErr = fmt.Errorf("%w: merge %d", ErrPack, i)
					return
				}
				m.flat.put(uint64(l)<<32|uint64(rr), int32(i))
				continue
			}
			ls, lok := packMergeStr(l, strs, nPieces)
			rs, rok := packMergeStr(rr, strs, nPieces)
			if !lok || !rok {
				mergeErr = fmt.Errorf("%w: merge %d", ErrPack, i)
				return
			}
			if m.byStr == nil {
				m.byStr = map[bigram]int32{}
			}
			m.byStr[bigram{ls, rs}] = int32(i)
		}
		t.merges = m
	})
	wg.Wait()
	if err := errors.Join(vocabErr, mergeErr); err != nil {
		return nil, err
	}

	for b := range 256 {
		p := fmt.Sprintf("<0x%02X>", b)
		t.bytePiece[b] = p
		if id, ok := t.vocab[p]; ok {
			t.byteToVal[id] = byte(b)
		} else if t.byteFallback {
			return nil, fmt.Errorf("%w: byte_fallback set but %q missing", ErrPack, p)
		}
	}

	t.added = newAddedTrie()
	for i := range nAdded {
		s, ok := str(binary.LittleEndian.Uint32(arec[12*i:]))
		id := int32(binary.LittleEndian.Uint32(arec[12*i+4:]))
		fl := binary.LittleEndian.Uint32(arec[12*i+8:])
		if !ok || !inID(id) {
			return nil, fmt.Errorf("%w: added token %d", ErrPack, i)
		}
		t.added.add(s, id)
		t.markAdded(int(id))
		if fl&packAddedRStrip != 0 {
			t.setRstrip(id, true)
		}
	}
	return t, nil
}

// packMergeStr resolves one side of a merge: a piece id v >= 0 is that id's piece, a negative v names string -(v+1).
func packMergeStr(v int32, strs []string, nPieces int) (string, bool) {
	if v >= 0 {
		if int(v) >= nPieces {
			return "", false
		}
		return strs[v], true
	}
	i := -int64(v) - 1
	if i >= int64(len(strs)) {
		return "", false
	}
	return strs[i], true
}

// flatMerges is an open-addressing hash table from a merge's packed id pair to its rank, filled once by LoadPack. It
// answers exactly as mergeTable.byID would for the same adds, and builds several times faster than a Go map of the
// same size. A key packs two ids below maxTokenID, so flatEmpty can never be a key.
type flatMerges struct {
	keys  []uint64
	ranks []int32
	shift uint
	n     int
}

const flatEmpty = ^uint64(0)

// newFlatMerges sizes the table to a power of two at least twice n, so probes stay short.
func newFlatMerges(n int) *flatMerges {
	bits := uint(4)
	for 1<<bits < 2*n {
		bits++
	}
	keys := make([]uint64, 1<<bits)
	for i := range keys {
		keys[i] = flatEmpty
	}
	return &flatMerges{keys: keys, ranks: make([]int32, 1<<bits), shift: 64 - bits}
}

func (f *flatMerges) slot(k uint64) uint64 { return (k * 0x9E3779B97F4A7C15) >> f.shift }

// put sets k's rank; a later put of the same key replaces it, as a later merge in the list does.
func (f *flatMerges) put(k uint64, rank int32) {
	mask := uint64(len(f.keys) - 1)
	for i := f.slot(k); ; i = (i + 1) & mask {
		switch f.keys[i] {
		case flatEmpty:
			f.keys[i], f.ranks[i] = k, rank
			f.n++
			return
		case k:
			f.ranks[i] = rank
			return
		}
	}
}

func (f *flatMerges) get(k uint64) (int32, bool) {
	mask := uint64(len(f.keys) - 1)
	for i := f.slot(k); ; i = (i + 1) & mask {
		switch f.keys[i] {
		case flatEmpty:
			return 0, false
		case k:
			return f.ranks[i], true
		}
	}
}
