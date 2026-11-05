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
	// bundleVersion is the one frame version this build writes and reads. The weights length is a u64 (a 7B int4
	// blob is over 4 GiB); tok stays u32, a metadata-only GGUF. The header is padded with zeros to
	// bundleBlobOffsetV3 bytes so the weights blob starts on a 16-byte boundary of the FILE (and so of an mmap of
	// it, whose base is page-aligned), which is what lets the decoder alias the arrays inside it.
	bundleVersion = 3

	// bundleBlobOffsetV3 is where the weights blob begins. A multiple of 16 (and a divisor of any page size),
	// larger than the 17 bytes of magic, version and length.
	bundleBlobOffsetV3 = 64
)

// Write frames the weights blob + tokenizer GGUF into a single bundle:
//
//	magic "GINFB" | u32 version=3 | u64 len(weights) | zero pad to offset 64 | weights | u32 len(tok) | tok
func Write(weights, tok []byte) []byte {
	out := make([]byte, 0, bundleBlobOffsetV3+len(weights)+len(tok)+4)
	out = append(out, bundleMagic...)
	out = binary.LittleEndian.AppendUint32(out, bundleVersion)
	out = binary.LittleEndian.AppendUint64(out, uint64(len(weights)))
	out = append(out, make([]byte, bundleBlobOffsetV3-len(out))...)
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
	var hdr [bundleBlobOffsetV3 - len(bundleMagic)]byte // u32 version + u64 placeholder length + zero pad
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

// weightsMagic opens the weights blob inside a bundle. decoder/serialize.go owns it (giwMagic); it is
// repeated here only so WeightsVersionFile can name the blob's version without loading it, and
// TestWeightsVersionFile_currentWriter holds the two together.
const weightsMagic = "GINFW"

// WeightsVersionFile returns the format version of the weights blob inside the bundle at path (the u32
// after the blob's "GINFW" magic — decoder's giwVersion as it stood when the file was written), reading
// only the two headers. For a cache that must decide whether an existing file is worth keeping before
// paying a full load (decoder's cacheFresh).
func WeightsVersionFile(path string) (uint32, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	var hdr [9]byte // magic(5) + u32 bundle version
	if _, err := f.ReadAt(hdr[:], 0); err != nil {
		return 0, fmt.Errorf("giw: read header: %w", err)
	}
	if string(hdr[:len(bundleMagic)]) != bundleMagic {
		return 0, fmt.Errorf("giw: bad bundle magic %q", hdr[:len(bundleMagic)])
	}
	if ver := binary.LittleEndian.Uint32(hdr[5:9]); ver != bundleVersion {
		return 0, errBundleVersion(ver)
	}
	var wh [len(weightsMagic) + 4]byte
	if _, err := f.ReadAt(wh[:], bundleBlobOffsetV3); err != nil {
		return 0, fmt.Errorf("giw: read weights header: %w", err)
	}
	if string(wh[:len(weightsMagic)]) != weightsMagic {
		return 0, fmt.Errorf("giw: bad weights magic %q", wh[:len(weightsMagic)])
	}
	return binary.LittleEndian.Uint32(wh[len(weightsMagic):]), nil
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
	var hdr [17]byte // magic(5) + u32 version + u64 weights-len
	if _, err := f.ReadAt(hdr[:], 0); err != nil {
		return nil, fmt.Errorf("giw: read header: %w", err)
	}
	if string(hdr[:len(bundleMagic)]) != bundleMagic {
		return nil, fmt.Errorf("giw: bad bundle magic %q", hdr[:len(bundleMagic)])
	}
	if ver := binary.LittleEndian.Uint32(hdr[5:9]); ver != bundleVersion {
		return nil, errBundleVersion(ver)
	}
	tokOff := bundleBlobOffsetV3 + int64(binary.LittleEndian.Uint64(hdr[9:17]))
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
// Returns an error on a bad magic, a frame version other than bundleVersion, or truncation.
func Read(data []byte) (weights, tok []byte, err error) {
	c := &cur{b: data}
	if got := c.take(len(bundleMagic)); string(got) != bundleMagic {
		return nil, nil, fmt.Errorf("giw: bad bundle magic %q (want %q)", got, bundleMagic)
	}
	if v := c.u32(); c.err == nil && v != bundleVersion {
		return nil, nil, errBundleVersion(v)
	}
	n := int(c.u64()) // then zero pad so the blob starts at bundleBlobOffsetV3
	c.take(bundleBlobOffsetV3 - c.off)
	weights = c.take(n)
	tok = c.take(int(c.u32()))
	if c.err != nil {
		return nil, nil, fmt.Errorf("giw: truncated bundle")
	}
	return weights, tok, nil
}

// errBundleVersion is the refusal for a frame this build does not read. A sidecar is rebuilt from its source; a
// hand-built or embedded bundle has to be rebuilt with cmd/prequant.
func errBundleVersion(v uint32) error {
	return fmt.Errorf("giw: bundle frame v%d, this build reads v%d — delete the sidecar (it rebuilds) or rebuild with `go run ./cmd/prequant …`", v, bundleVersion)
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
	// `n > len(c.b)-c.off` rather than `c.off+n > len(c.b)`: a hostile weights length
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
