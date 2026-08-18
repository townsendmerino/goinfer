package decoder

import (
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"testing"
)

// TestMXFP4_referenceValues pins the format constants and the e8m0 scale against hand-derived
// values from the OCP MX spec / gguf reference — asset-free, so it runs in CI regardless of the
// golden fixture. It is the cheap guard that the table and the (subnormal-sensitive) scale bit
// formula are right; the golden test below is the bit-exact proof on real tensor bytes.
func TestMXFP4_referenceValues(t *testing.T) {
	// e8m0_to_fp32_half (ggml_e8m0_to_fp32_half): x<2 subnormal, else exponent = x-1.
	cases := []struct {
		x    uint8
		bits uint32 // expected float32 bit pattern
	}{
		{0, 0x00200000},          // 2^-128 (subnormal)
		{1, 0x00400000},          // 2^-127 (subnormal)
		{2, 0x00800000},          // 2^-126 (smallest normal)
		{127, uint32(126) << 23}, // 2^-1 = 0.5
		{128, uint32(127) << 23}, // 2^0 = 1.0
		{255, uint32(254) << 23}, // 2^127
	}
	for _, c := range cases {
		got := math.Float32bits(e8m0ToF32Half(c.x))
		if got != c.bits {
			t.Errorf("e8m0ToF32Half(%d) bits=%#08x, want %#08x", c.x, got, c.bits)
		}
	}
	// kvalues: e2m1 doubled (gguf/quants.py MXFP4.kvalues).
	want := [16]int8{0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}
	if mxfp4KValues != want {
		t.Errorf("mxfp4KValues = %v, want %v", mxfp4KValues, want)
	}
	// A synthetic block: scale byte 128 (d=1.0) so element = kvalues[idx] exactly. Byte j packs
	// element j (low nibble) and element j+16 (high nibble).
	blk := make([]byte, mxfp4BlockBytes)
	blk[0] = 128  // d = 1.0
	blk[1] = 0x07 // low=7 (→+12), high=0 (→0)  ⇒ elem 0 = 12, elem 16 = 0
	blk[2] = 0x9F // low=0xF (→-12), high=9 (→-1) ⇒ elem 1 = -12, elem 17 = -1
	var dst [mxfp4BlockElems]float32
	mxfp4DequantBlock(blk, dst[:])
	for idx, exp := range map[int]float32{0: 12, 16: 0, 1: -12, 17: -1} {
		if dst[idx] != exp {
			t.Errorf("block elem %d = %v, want %v", idx, dst[idx], exp)
		}
	}
}

// mxfp4Golden is the fixture produced by scripts/extract_mxfp4_golden.py from a real gpt-oss:20b
// MXFP4 tensor: the raw block bytes plus the reference `gguf`-library dequant as float32 bits.
type mxfp4Golden struct {
	Tensor   string   `json:"tensor"`
	NBlocks  int      `json:"n_blocks"`
	RawHex   string   `json:"raw_hex"`
	WantBits []uint32 `json:"want_bits"`
}

// TestMXFP4_bitExactGolden is THE first-move gate (§8): the Go unpacker must reproduce the
// reference `gguf` library's dequant BIT-FOR-BIT on real gpt-oss:20b tensor bytes. Verifying the
// unpack against the reference before writing the forward is what let cpubrrr's forward work on
// the first run; this test is that check. The fixture is committed so it runs in CI (a skip for a
// missing asset is a blocker, not a pass — regenerate with scripts/extract_mxfp4_golden.py).
func TestMXFP4_bitExactGolden(t *testing.T) {
	const path = "testdata/mxfp4_golden.json"
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("no golden at %s — regenerate: python3 scripts/extract_mxfp4_golden.py > %s", path, path)
	}
	var g mxfp4Golden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	blocks, err := hex.DecodeString(g.RawHex)
	if err != nil {
		t.Fatalf("decode raw_hex: %v", err)
	}
	if len(blocks) != g.NBlocks*mxfp4BlockBytes {
		t.Fatalf("raw is %d bytes, want %d (%d blocks × %d)", len(blocks), g.NBlocks*mxfp4BlockBytes, g.NBlocks, mxfp4BlockBytes)
	}
	if len(g.WantBits) != g.NBlocks*mxfp4BlockElems {
		t.Fatalf("golden has %d values, want %d", len(g.WantBits), g.NBlocks*mxfp4BlockElems)
	}

	got, err := mxfp4Dequant(blocks, g.NBlocks)
	if err != nil {
		t.Fatalf("mxfp4Dequant: %v", err)
	}
	mism := 0
	for i, w := range g.WantBits {
		if b := math.Float32bits(got[i]); b != w {
			if mism < 8 {
				t.Errorf("value %d (block %d, elem %d): got bits %#08x (%v), want %#08x (%v)",
					i, i/mxfp4BlockElems, i%mxfp4BlockElems, b, got[i], w, math.Float32frombits(w))
			}
			mism++
		}
	}
	if mism > 0 {
		t.Fatalf("%d/%d values differ from the gguf reference — unpacker is NOT bit-exact", mism, len(g.WantBits))
	}
	t.Logf("bit-exact vs gguf on %q: %d blocks / %d values, 0 mismatches", g.Tensor, g.NBlocks, len(g.WantBits))
}

// TestMXFP4_splitIsSequentialNotGGML pins the intra-block nibble order of the SAFETENSORS
// layout, which is NOT GGML's.
//
// This test exists because the opposite was assumed and written down. Phase 0 recorded that
// safetensors MXFP4 differed from GGUF "only in addressing — no new numerics", and the first
// implementation reused the GGML core. Dequantizing a real gpt-oss expert both ways and
// diffing against the same weight through the already-validated GGUF reader settled it:
// cosine 0.081 for GGML order, 1.000000 for sequential. The bug it would have shipped is the
// worst kind — finite values, correct shapes, plausible magnitudes, entirely wrong weights.
//
// So this asserts the two orders DISAGREE in exactly the documented way, rather than
// asserting they agree (which the earlier version of this test did, and passed, because the
// implementation shared their shared mistake).
func TestMXFP4_splitIsSequentialNotGGML(t *testing.T) {
	// One block: byte j holds low nibble = code (j % 15), high nibble = code ((j + 7) % 15).
	// Codes stay < 15 so every value is distinct and non-zero under mxfp4KValues.
	blocks := make([]byte, 16)
	for j := range 16 {
		blocks[j] = byte((j%15)&0x0F | (((j+7)%15)&0x0F)<<4)
	}
	scales := []byte{127} // e8m0 127 -> a clean power of two, so the codes are readable

	got, err := mxfp4DequantSplit(blocks, scales, 1)
	if err != nil {
		t.Fatalf("split: %v", err)
	}
	d := e8m0ToF32Half(127)
	for j := range 16 {
		wantLo := d * float32(mxfp4KValues[blocks[j]&0x0F])
		wantHi := d * float32(mxfp4KValues[blocks[j]>>4])
		// SEQUENTIAL: byte j carries elements 2j and 2j+1.
		if got[2*j] != wantLo || got[2*j+1] != wantHi {
			t.Fatalf("byte %d: got (%v,%v) at [2j,2j+1], want (%v,%v) — safetensors packs "+
				"elements 2j/2j+1, not GGML's j/j+16", j, got[2*j], got[2*j+1], wantLo, wantHi)
		}
	}

	// And the GGML core on the same bytes must produce a DIFFERENT vector. If these ever
	// agree, one of the two orders has been silently changed to match the other.
	ggml := make([]float32, mxfp4BlockElems)
	mxfp4Block(scales[0], blocks, ggml)
	same := true
	for i := range ggml {
		if ggml[i] != got[i] {
			same = false
			break
		}
	}
	if same {
		t.Error("GGML and safetensors orders produced identical output — they must differ; " +
			"the real checkpoint measurement says 0.081 vs 1.000000 cosine")
	}

	// Mismatched pairs are corruption, not something to interpret: these are two
	// independently-shaped tensors, so a length disagreement between them must be refused.
	if _, err := mxfp4DequantSplit(blocks, nil, 1); err == nil {
		t.Error("missing scales accepted; want an error")
	}
	if _, err := mxfp4DequantSplit(blocks[:8], scales, 1); err == nil {
		t.Error("short blocks accepted; want an error")
	}
}
