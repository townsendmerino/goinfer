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
)

const (
	bundleMagic   = "GINFB"
	bundleVersion = 1
)

// Write frames the weights blob + tokenizer GGUF into a single bundle:
//
//	magic "GINFB" | u32 version | u32 len(weights) | weights | u32 len(tok) | tok
func Write(weights, tok []byte) []byte {
	out := make([]byte, 0, len(bundleMagic)+12+len(weights)+len(tok))
	out = append(out, bundleMagic...)
	out = binary.LittleEndian.AppendUint32(out, bundleVersion)
	out = binary.LittleEndian.AppendUint32(out, uint32(len(weights)))
	out = append(out, weights...)
	out = binary.LittleEndian.AppendUint32(out, uint32(len(tok)))
	out = append(out, tok...)
	return out
}

// Read splits a bundle back into the weights blob and the tokenizer GGUF. The
// returned slices alias data (zero-copy), so data must outlive their use — which
// is the point: the weights half is aliased all the way down to the int8 arrays.
// Returns an error on a bad magic/version or truncation so the caller can fall
// back to a GGUF.
func Read(data []byte) (weights, tok []byte, err error) {
	c := &cur{b: data}
	if got := c.take(len(bundleMagic)); string(got) != bundleMagic {
		return nil, nil, fmt.Errorf("giw: bad bundle magic %q (want %q)", got, bundleMagic)
	}
	if v := c.u32(); v != bundleVersion {
		return nil, nil, fmt.Errorf("giw: bundle version %d, this build reads %d", v, bundleVersion)
	}
	weights = c.take(int(c.u32()))
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

func (c *cur) take(n int) []byte {
	if c.err != nil || n < 0 || c.off+n > len(c.b) {
		if c.err == nil {
			c.err = fmt.Errorf("giw: short read")
		}
		return nil
	}
	b := c.b[c.off : c.off+n]
	c.off += n
	return b
}
