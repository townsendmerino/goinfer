package prequant

import (
	"encoding/binary"
	"fmt"
)

// metadataPrefixLen returns the byte length of a GGUF's header — magic, version,
// counts, all metadata key-values, and all tensor INFOS — up to the aligned
// start of the tensor DATA section. raw[:n] is therefore a self-contained GGUF
// with the same metadata + tensor directory but zero weight bytes; aikit's
// parseGGUF accepts it (it only checks the data start is within the file), and
// the tokenizer loads from it (it never reads tensor data). This is how the
// prequant bundle carries the tokenizer in ~MBs instead of the whole model.
//
// It mirrors aikit/embed.parseGGUF's cursor exactly; if that format changes,
// the prequant generator must follow (the version guards catch a stale bundle).
func metadataPrefixLen(raw []byte) (int, error) {
	c := &ggufCur{b: raw}
	if c.u32() != 0x46554747 { // "GGUF" little-endian
		return 0, fmt.Errorf("not a GGUF file (bad magic)")
	}
	if v := c.u32(); v != 2 && v != 3 {
		return 0, fmt.Errorf("unsupported GGUF version %d", v)
	}
	tensorCount := c.u64()
	kvCount := c.u64()

	align := uint64(32)
	for i := uint64(0); i < kvCount && c.err == nil; i++ {
		key := c.str()
		vtype := c.u32()
		if key == "general.alignment" && vtype == 4 {
			if a := uint64(c.u32()); a > 0 {
				align = a
			}
			continue
		}
		c.skipValue(vtype)
	}
	for i := uint64(0); i < tensorCount && c.err == nil; i++ {
		c.str()            // name
		nd := int(c.u32()) // n_dims
		for range nd {
			c.u64() // dim
		}
		c.u32() // ggml type
		c.u64() // offset
	}
	if c.err != nil {
		return 0, c.err
	}
	start := uint64(c.off)
	if start%align != 0 {
		start += align - start%align
	}
	if start > uint64(len(raw)) {
		return 0, fmt.Errorf("computed data start %d past EOF %d", start, len(raw))
	}
	return int(start), nil
}

type ggufCur struct {
	b   []byte
	off int
	err error
}

func (c *ggufCur) need(n int) bool {
	if c.err != nil {
		return false
	}
	if n < 0 || n > len(c.b)-c.off { // overflow-safe form (a crafted u64 length can wrap c.off+n); see internal/giw/bundle.go
		c.err = fmt.Errorf("gguf header: unexpected EOF")
		return false
	}
	return true
}

func (c *ggufCur) u32() uint32 {
	if !c.need(4) {
		return 0
	}
	v := binary.LittleEndian.Uint32(c.b[c.off:])
	c.off += 4
	return v
}

func (c *ggufCur) u64() uint64 {
	if !c.need(8) {
		return 0
	}
	v := binary.LittleEndian.Uint64(c.b[c.off:])
	c.off += 8
	return v
}

// str skips a GGUF string (u64 length + bytes).
func (c *ggufCur) str() string {
	n := int(c.u64())
	if !c.need(n) {
		return ""
	}
	s := string(c.b[c.off : c.off+n])
	c.off += n
	return s
}

// ggufScalarSize is the byte size of a GGUF scalar value type, or 0 for
// string(8)/array(9).
var ggufScalarSize = map[uint32]int{0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

func (c *ggufCur) skipValue(vtype uint32) {
	switch vtype {
	case 8: // string
		c.str()
	case 9: // array: elem type + count + elements
		et := c.u32()
		n := int(c.u64())
		for i := 0; i < n && c.err == nil; i++ {
			c.skipValue(et)
		}
	default:
		sz, ok := ggufScalarSize[vtype]
		if !ok {
			c.err = fmt.Errorf("gguf header: unknown value type %d", vtype)
			return
		}
		if c.need(sz) {
			c.off += sz
		}
	}
}
