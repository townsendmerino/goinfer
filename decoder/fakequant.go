package decoder

import "os"

// fakeQuantScheme (GOINFER_FAKEQUANT=affine|sym|symmse) is a diagnostic, not for production: when set, the int4
// quantization path (quantizeWM's quantInt4 case) quantize→dequantizes each weight with the named 4-bit scheme and
// stores the result at int8 (W8A8). That isolates the 4-bit weight-scheme quality without any packed-nibble format,
// dequant routine, W4A8 kernel or .giw change: the int8 path is nearly transparent (0.99995 reconstruction), so the
// fake-int4 error survives to the forward. "sym" is the control: it must reproduce the real-int4 output, or the
// harness is lying (asserted bit-for-bit by TestFakeQuantSymMatchesRuntimeInt4). The env is read once at package load;
// unset is a strict no-op, so every int4 consumer stays bit-identical (TestFakeQuantOffBitIdentical).
var (
	fakeQuantScheme      = os.Getenv("GOINFER_FAKEQUANT")
	fakeQuantF32Act      = os.Getenv("GOINFER_FAKEQUANT_ACT") == "f32"
	fakeQuantExpertsOnly = os.Getenv("GOINFER_FAKEQUANT_EXPERTS") == "1"
	fakeQuantPerRow      bool // §7 Phase 0b: one scale per row; a test seam (tests set it; env read retired 2026-09-24)
)

// fakeQuantF32Act (GOINFER_FAKEQUANT_ACT=f32) stores the fake-quant reconstruction at weight-only int8 (Q8, f32
// activations) instead of W8A8 (int8 activations), so a scheme can be tested against both activation precisions.
//
// fakeQuantExpertsOnly (GOINFER_FAKEQUANT_EXPERTS=1) restricts the fake-quant to the MoE experts (streamExperts): load
// at int8int8 so attention/dense/embed stay int8 and only the experts take the 4-bit scheme. It tests whether the
// experts alone tolerate 4-bit when the router's input hidden state is kept accurate.
