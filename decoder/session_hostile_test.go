package decoder

import (
	"encoding/binary"
	"errors"
	"hash/crc32"
	"os"
	"testing"
)

// M-04: LoadSession trusted blob-controlled lengths, and the M17 allocation bound was bypassed
// by a header declaring numLayers = 0.
//
// TestLoadSession_rejectsCorrupt only flipped the CRC, which proves the CRC works and nothing
// else — the CRC covers the ATTACKER'S bytes, so every mutation below is CRC-VALID by
// construction. That is the whole point: a hostile or corrupt session file is well-formed by
// every check that ran on it.
//
// Table-driven "mutate one header field, recompute the CRC, expect a *SnapshotError", as the
// audit's fix text specifies.
func TestLoadSession_hostileHeaders(t *testing.T) {
	// A COMMITTED fixture, not loadTestModel's gemma-3-270m: that one is untracked, so this
	// test would SKIP on a clean checkout and in CI — and a skip is not a pass. llama-tiny is
	// in git and loads on any machine.
	const fixture = "../testdata/llama-tiny"
	if _, err := os.Stat(fixture); err != nil {
		t.Skipf("no fixture at %s: %v", fixture, err)
	}
	m, err := Load(fixture, Options{})
	if err != nil {
		t.Fatalf("Load(%s): %v", fixture, err)
	}
	defer m.Close()

	sess := m.NewSession(0)
	sess.tokens = nil // pos 0 ⇒ zero tokens; the header guards are what this exercises
	sess.cache.pos = 0
	clean := sess.Snapshot("model-A")

	// The header layout, from Snapshot's writer: magic, ver(4), id (u32 length + bytes),
	// numLayers(4) kvDim(4) window(4) headDim(4) manualPos(1) quant(1) pos(4).
	// Offsets derived from kvSnapMagic rather than hardcoded — the first draft assumed a
	// 4-byte magic and indexed into the middle of the id string.
	idLenOff := len(kvSnapMagic) + 4
	idLen := int(binary.LittleEndian.Uint32(clean[idLenOff:]))
	base := idLenOff + 4 + idLen
	off := map[string]int{
		"numLayers": base, "kvDim": base + 4, "window": base + 8, "headDim": base + 12,
		"pos": base + 18,
	}
	// The premise: those offsets really are the fields. If this drifts, every case below
	// mutates something else and passes for the wrong reason.
	if got := int(binary.LittleEndian.Uint32(clean[off["numLayers"]:])); got != m.w.arch.NumLayers {
		t.Fatalf("header layout drifted: numLayers reads %d, model has %d", got, m.w.arch.NumLayers)
	}

	reseal := func(b []byte) []byte {
		binary.LittleEndian.PutUint32(b[len(b)-4:], crc32.ChecksumIEEE(b[:len(b)-4]))
		return b
	}
	mutate := func(field string, v uint32) []byte {
		b := append([]byte(nil), clean...)
		binary.LittleEndian.PutUint32(b[off[field]:], v)
		return reseal(b)
	}

	// The clean blob must load, or every rejection below is meaningless.
	if _, err := m.LoadSession(clean, ""); err != nil {
		t.Fatalf("the clean blob does not load: %v", err)
	}

	var se *SnapshotError
	for name, blob := range map[string][]byte{
		// THE FATAL ONE. numLayers=0 passed the "implausible dims" check (only negatives were
		// rejected), which made perPos 0, which skipped the `pos` bound entirely, and
		// m.NewCache(pos) then allocated with the MODEL's geometry and this pos. A 20-byte
		// header in -session-dir was `runtime: out of memory` at server boot.
		"numLayers=0 with a huge pos": func() []byte {
			b := append([]byte(nil), clean...)
			binary.LittleEndian.PutUint32(b[off["numLayers"]:], 0)
			binary.LittleEndian.PutUint32(b[off["pos"]:], 0x7FFFFFFF)
			return reseal(b)
		}(),
		"kvDim=0 with a huge pos": func() []byte {
			b := append([]byte(nil), clean...)
			binary.LittleEndian.PutUint32(b[off["kvDim"]:], 0)
			binary.LittleEndian.PutUint32(b[off["pos"]:], 0x7FFFFFFF)
			return reseal(b)
		}(),
		"headDim=0":            mutate("headDim", 0),
		"pos beyond the body":  mutate("pos", 0x7FFFFFFF),
		"numLayers mismatched": mutate("numLayers", 999),
		"kvDim mismatched":     mutate("kvDim", 999),
	} {
		t.Run(name, func(t *testing.T) {
			got, err := m.LoadSession(blob, "")
			if !errors.As(err, &se) {
				t.Fatalf("accepted (or wrong error type): session=%v err=%v — this blob is "+
					"CRC-valid, so nothing but an explicit check can reject it (M-04)", got != nil, err)
			}
		})
	}
}

// The BODY half of M-04, which the header cases above do not reach: with pos == 0 every
// blob-controlled length is legally 0, so those mutations exercise the header guards only.
// Measured — reverting the length checks left the header test green.
func TestLoadSession_bodyLengthsMustAgreeWithPos(t *testing.T) {
	const fixture = "../testdata/llama-tiny"
	if _, err := os.Stat(fixture); err != nil {
		t.Skipf("no fixture at %s: %v", fixture, err)
	}
	m, err := Load(fixture, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()

	// tokens longer than pos is the QUIET failure, and it needs no byte surgery: Snapshot
	// writes sess.tokens verbatim. rewindForReuse then computes matched > c.pos, TruncateTo
	// treats an out-of-range target as a no-op, and the reuse reports an EXACT match on a
	// cache it never rewound — no panic, no error, a session continuing from the wrong KV.
	sess := m.NewSession(0)
	sess.tokens = []int{1, 2, 3}
	sess.cache.pos = 0
	blob := sess.Snapshot("m")
	if blob == nil {
		t.Fatal("Snapshot returned nil for a non-recurrent family")
	}
	var se *SnapshotError
	if got, err := m.LoadSession(blob, ""); !errors.As(err, &se) {
		t.Errorf("3 tokens at pos 0 accepted (session=%v, err=%v) — len(tokens) > pos makes "+
			"rewindForReuse claim an exact match it never performed (M-04)", got != nil, err)
	}
}

// checkGlobalLen is the predicate both storage arms share. Unit-tested directly because
// reaching it through a snapshot needs a populated KV and byte surgery on the payload; what
// matters is that 0 stays legal (KV-shared layers) and any other mismatch is refused.
func TestCheckGlobalLen(t *testing.T) {
	if e := checkGlobalLen(0, "keys", 0, 128); e != nil {
		t.Errorf("0 rejected: %v — a KV-shared layer stores nothing of its own", e)
	}
	if e := checkGlobalLen(0, "keys", 128, 128); e != nil {
		t.Errorf("an exact length rejected: %v", e)
	}
	for _, got := range []int{1, 127, 129, 1 << 20} {
		if e := checkGlobalLen(3, "vals", got, 128); e == nil {
			t.Errorf("len %d accepted against an expected 128 — the forward derives its key "+
				"count from `keys` and indexes `vals` at the same positions, so a short one is "+
				"an out-of-range read in the generation goroutine (M-04)", got)
		}
	}
}
