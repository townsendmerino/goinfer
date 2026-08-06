// Package giw frames a prequant "goinfer weights" bundle: the serialized,
// already-quantized decoder weights (decoder.SerializeWeights) plus a tiny
// metadata-only GGUF carrying the tokenizer (the source GGUF truncated at the
// tensor-data boundary — ~MBs, no weights). The demo embeds one bundle and
// loads both halves with their existing load paths; the generator (cmd/prequant)
// writes it. Shared so writer and reader agree on the frame.
package giw

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

const (
	bundleMagic = "GINFB"
	// v2 writes the weights length as u64; v1 used u32 and so truncated weight
	// blobs > 4 GiB (a 7B int4 blob is ~5.2 GB → corruption). v1 bundles still
	// load. tok stays u32 — it's a metadata-only GGUF, always well under 4 GiB.
	bundleVersion = 2
)

// Write frames the weights blob + tokenizer GGUF into a single bundle:
//
//	magic "GINFB" | u32 version=2 | u64 len(weights) | weights | u32 len(tok) | tok
func Write(weights, tok []byte) []byte {
	out := make([]byte, 0, len(bundleMagic)+16+len(weights)+len(tok))
	out = append(out, bundleMagic...)
	out = binary.LittleEndian.AppendUint32(out, bundleVersion)
	out = binary.LittleEndian.AppendUint64(out, uint64(len(weights)))
	out = append(out, weights...)
	out = binary.LittleEndian.AppendUint32(out, uint32(len(tok)))
	out = append(out, tok...)
	return out
}

// WriteStream frames a bundle straight to f without ever holding the weights blob
// in memory — for prequantizing a large model where that blob is tens of GB. It
// writes the header with a placeholder weights length, calls writeWeights to stream
// the weights directly into f (returning the bytes written), appends tok, then
// patches the length field via WriteAt. The result is byte-identical to Write and
// loads with Read. f must be a regular, seekable file opened for writing.
func WriteStream(f *os.File, tok []byte, writeWeights func(io.Writer) (int64, error)) error {
	if _, err := f.Write([]byte(bundleMagic)); err != nil {
		return err
	}
	var hdr [12]byte // u32 version + u64 placeholder weights length
	binary.LittleEndian.PutUint32(hdr[0:4], bundleVersion)
	if _, err := f.Write(hdr[:]); err != nil {
		return err
	}
	lenOff := int64(len(bundleMagic) + 4) // offset of the u64 weights-length field
	n, err := writeWeights(f)
	if err != nil {
		return fmt.Errorf("stream weights: %w", err)
	}
	var tl [4]byte
	binary.LittleEndian.PutUint32(tl[:], uint32(len(tok)))
	if _, err := f.Write(tl[:]); err != nil {
		return err
	}
	if _, err := f.Write(tok); err != nil {
		return err
	}
	var nb [8]byte
	binary.LittleEndian.PutUint64(nb[:], uint64(n))
	if _, err := f.WriteAt(nb[:], lenOff); err != nil {
		return fmt.Errorf("patch weights length: %w", err)
	}
	return nil
}

// ReadTokFile reads only the tokenizer half of a bundle file, without pulling the
// (potentially many-GB) weights half into memory: it parses the small header, then
// ReadAts the trailing tokenizer GGUF past the weights. For serving a large .giw
// where the weights are mmap'd by the decoder — the tokenizer is a few MB, the
// weights tens of GB, so Read (which aliases the whole file) would be ruinous here.
func ReadTokFile(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var hdr [17]byte // magic(5) + u32 version + u64 weights-len (v2; v1 uses 4 of the 8)
	if _, err := f.ReadAt(hdr[:], 0); err != nil {
		return nil, fmt.Errorf("giw: read header: %w", err)
	}
	if string(hdr[:len(bundleMagic)]) != bundleMagic {
		return nil, fmt.Errorf("giw: bad bundle magic %q", hdr[:len(bundleMagic)])
	}
	var tokOff int64
	switch ver := binary.LittleEndian.Uint32(hdr[5:9]); ver {
	case 1:
		tokOff = 13 + int64(binary.LittleEndian.Uint32(hdr[9:13])) // magic+u32ver+u32len
	case 2:
		tokOff = 17 + int64(binary.LittleEndian.Uint64(hdr[9:17])) // magic+u32ver+u64len
	default:
		return nil, fmt.Errorf("giw: bundle version %d, this build reads 1–2", ver)
	}
	var tl [4]byte
	if _, err := f.ReadAt(tl[:], tokOff); err != nil {
		return nil, fmt.Errorf("giw: read tok length: %w", err)
	}
	// Bound the allocation by the file (C-31): the tokenizer body starts at tokOff+4, so it can hold at
	// most (fileSize - tokOff - 4) bytes. Without this a crafted u32 tokLen (e.g. 0xFFFFFFFF in a 21-byte
	// file) makes a 4 GiB allocation before the ReadAt below fails. The ReadAt above succeeded, so
	// tokOff+4 ≤ fileSize and avail ≥ 0.
	tokLen := int64(binary.LittleEndian.Uint32(tl[:]))
	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if avail := fi.Size() - (tokOff + 4); tokLen > avail {
		return nil, fmt.Errorf("giw: tokenizer length %d exceeds the %d bytes remaining in the file (truncated or corrupt)", tokLen, avail)
	}
	tok := make([]byte, tokLen)
	if _, err := f.ReadAt(tok, tokOff+4); err != nil {
		return nil, fmt.Errorf("giw: read tok: %w", err)
	}
	return tok, nil
}

// Read splits a bundle back into the weights blob and the tokenizer GGUF. The
// returned slices alias data (zero-copy), so data must outlive their use — which
// is the point: the weights half is aliased all the way down to the int8 arrays.
// Reads both v1 (u32 weights length) and v2 (u64). Returns an error on a bad
// magic/version or truncation so the caller can fall back to a GGUF.
func Read(data []byte) (weights, tok []byte, err error) {
	c := &cur{b: data}
	if got := c.take(len(bundleMagic)); string(got) != bundleMagic {
		return nil, nil, fmt.Errorf("giw: bad bundle magic %q (want %q)", got, bundleMagic)
	}
	switch v := c.u32(); v {
	case 1:
		weights = c.take(int(c.u32())) // v1: u32 weights length (≤ 4 GiB)
	case 2:
		weights = c.take(int(c.u64())) // v2: u64 weights length
	default:
		return nil, nil, fmt.Errorf("giw: bundle version %d, this build reads 1–2", v)
	}
	tok = c.take(int(c.u32()))
	if c.err != nil {
		return nil, nil, fmt.Errorf("giw: truncated bundle")
	}
	return weights, tok, nil
}

type cur struct {
	b   []byte
	off int
	err error
}

func (c *cur) u32() uint32 {
	b := c.take(4)
	if b == nil {
		return 0
	}
	return binary.LittleEndian.Uint32(b)
}

func (c *cur) u64() uint64 {
	b := c.take(8)
	if b == nil {
		return 0
	}
	return binary.LittleEndian.Uint64(b)
}

func (c *cur) take(n int) []byte {
	// `n > len(c.b)-c.off` rather than `c.off+n > len(c.b)`: a hostile v2 length
	// (int(u64) near maxint64) makes c.off+n overflow to a negative value that
	// slips past the bound check and panics the slice. c.off <= len(c.b) is an
	// invariant (take only advances on a valid read), so len(c.b)-c.off is a safe,
	// non-negative right-hand side.
	if c.err != nil || n < 0 || n > len(c.b)-c.off {
		if c.err == nil {
			c.err = fmt.Errorf("giw: short read")
		}
		return nil
	}
	b := c.b[c.off : c.off+n]
	c.off += n
	return b
}
