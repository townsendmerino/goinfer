package giw

import (
	"bytes"
	"encoding/binary"
	"testing"
)

// Track 2.3 (testing campaign): the .giw bundle frame is goinfer's own untrusted
// binary input (cmd/prequant writes it, the demo embeds it). Read's magic /
// version / CRC-style guards exist; these targets fuzz PAST them — valid header,
// hostile body, truncation at every boundary, and the u64 v2 length path — to
// hold the "typed error, never a panic" bar.

// validBundle frames a v2 bundle the way Write does, for seeding.
func validBundle(weights, tok []byte) []byte { return Write(weights, tok) }

// v1Bundle frames a legacy v1 (u32 weights length) bundle for seeding.
func v1Bundle(weights, tok []byte) []byte {
	var b []byte
	b = append(b, bundleMagic...)
	b = binary.LittleEndian.AppendUint32(b, 1)
	b = binary.LittleEndian.AppendUint32(b, uint32(len(weights)))
	b = append(b, weights...)
	b = binary.LittleEndian.AppendUint32(b, uint32(len(tok)))
	b = append(b, tok...)
	return b
}

// FuzzGIWRead feeds arbitrary bytes to Read: it must always return a typed error
// or a clean (weights, tok) split, never panic — including on a hostile v2 length
// field engineered to overflow the bounds arithmetic in cur.take.
func FuzzGIWRead(f *testing.F) {
	f.Add(validBundle([]byte("weights"), []byte("tok")))
	f.Add(v1Bundle([]byte("w"), []byte("t")))
	f.Add([]byte(bundleMagic))
	f.Add([]byte(bundleMagic + "\x02\x00\x00\x00")) // magic + version, truncated
	// magic + v2 + a near-maxint64 weights length (the take() overflow vector).
	var hostile []byte
	hostile = append(hostile, bundleMagic...)
	hostile = binary.LittleEndian.AppendUint32(hostile, 2)
	hostile = binary.LittleEndian.AppendUint64(hostile, 0x7fffffffffffffff)
	f.Add(hostile)

	f.Fuzz(func(t *testing.T, data []byte) {
		w, tok, err := Read(data)
		if err != nil {
			if w != nil || tok != nil {
				t.Fatalf("Read returned (%v,%v) alongside error %v", w != nil, tok != nil, err)
			}
			return
		}
		// On success the two slices must lie within data (Read is zero-copy) — a
		// returned slice outside data would mean the length math went wrong.
		if len(w) > len(data) || len(tok) > len(data) {
			t.Fatalf("Read returned slice longer than input: len(w)=%d len(tok)=%d len(data)=%d", len(w), len(tok), len(data))
		}
	})
}

// FuzzGIWRoundTrip is the inverse property: anything Write frames, Read recovers
// byte-for-byte. (weights and tok are independent fuzz inputs.)
func FuzzGIWRoundTrip(f *testing.F) {
	f.Add([]byte("the-quantized-weights-blob"), []byte("metadata-gguf"))
	f.Add([]byte{}, []byte{})
	f.Add([]byte{0}, []byte("\xff\x00\xfe"))
	f.Fuzz(func(t *testing.T, weights, tok []byte) {
		w, tk, err := Read(Write(weights, tok))
		if err != nil {
			t.Fatalf("Read(Write(...)) failed: %v", err)
		}
		if !bytes.Equal(w, weights) || !bytes.Equal(tk, tok) {
			t.Fatalf("round-trip mismatch: got weights=%q tok=%q", w, tk)
		}
	})
}
