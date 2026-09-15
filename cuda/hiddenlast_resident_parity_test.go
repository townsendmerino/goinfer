//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// hiddenLastResidentBarCUDA mirrors metal/hiddenlast_resident_parity_test.go's own bar exactly —
// measured directly on THIS backend's kernels, not assumed to match Metal's: that file's own
// comment explains why 0.998 (not 0.9999) is the right shape of bar for a hidden-state-only,
// pre-LM-head comparison of CPU vs GPU int8 execution of the same weights. If this fails on a real
// run, replace the bar with what was actually measured (same discipline that file used) rather than
// loosening it to make the number go away.
const hiddenLastResidentBarCUDA = 0.998

// TestHiddenLastResidentParityCUDA is M-10's own gate (docs/audit-2026-09-10.md) — the CUDA twin of
// metal/hiddenlast_resident_parity_test.go, which the finding named as missing ("Only Metal has a
// hiddenlast_resident_parity_test.go"). Same shape: resident HiddenLast (cudaResident.HiddenLast /
// prefillChunked's tailHiddenLast) vs CPU HiddenLast, same int8int8 weights, called directly via
// ResidentForwardForTest (bypassing decoder.Model.HiddenLast's own resBusy/fallback dispatch,
// already covered by the fake-backend seam tests) so this isolates the KERNEL correctness question.
//
// This is also forceExactKernels' own integration proof (M-09/M-10/M-11): before that fix,
// prefillCore's tailHiddenLast pass could silently engage useAttnFused/useGemmMMA whenever M/K/
// position crossed their shape thresholds — this fixture's own M may or may not cross them, so a
// failing run here without the fix would only be a coincidence; the real proof that forceExactKernels
// is wired correctly is decoder/spec_verify_guard_test.go's pure unit coverage plus this test
// passing at whatever cosine the real kernels produce, unaffected by prompt length.
func TestHiddenLastResidentParityCUDA(t *testing.T) {
	const ckpt = "../testdata/llama-tiny"
	requireDeviceAndFixture(t, ckpt)

	mRes, err := decoder.Load(ckpt, decoder.Options{Backend: "cuda", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load cuda: %v", err)
	}
	defer mRes.Close()
	rf := mRes.ResidentForwardForTest()
	if rf == nil {
		t.Fatalf("llama-tiny did not go CUDA-resident — decode path %q; decline: %s", mRes.DecodePath(), mRes.ResidentDecline())
	}
	rh, ok := rf.(decoder.ResidentHiddenLast)
	if !ok {
		t.Fatalf("cuda resident runner (%T) does not implement decoder.ResidentHiddenLast", rf)
	}

	mCPU, err := decoder.Load(ckpt, decoder.Options{Backend: "cpu", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load cpu: %v", err)
	}
	defer mCPU.Close()

	_, _, _, _, _, _, vocab := mCPU.Dims()
	const ntok = 24
	ids := make([]int, ntok)
	for i := range ids {
		ids[i] = (i*131 + 7) % vocab
	}

	want, err := mCPU.HiddenLast(ids)
	if err != nil {
		t.Fatalf("cpu HiddenLast: %v", err)
	}

	embs := make([][]float32, len(ids))
	for i, id := range ids {
		embs[i] = mRes.EmbedResidentForTest(id)
	}
	got, err := rh.HiddenLast(context.Background(), embs, 0)
	if err != nil {
		t.Fatalf("resident HiddenLast: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("resident HiddenLast returned %d dims, want %d", len(got), len(want))
	}
	cos, maxAbs := cosF32(want, got)
	t.Logf("HiddenLast over %d tokens: cosine=%.8f maxAbs=%.6g", ntok, cos, maxAbs)
	if cos < hiddenLastResidentBarCUDA {
		t.Errorf("resident HiddenLast cosine %.8f < %.4f bar (M-10 gate) against the CPU reference", cos, hiddenLastResidentBarCUDA)
	}

	// A second, shorter sequence after Reset must ALSO match — proves the KV writes from a prior
	// HiddenLast call do not leak into the next one (mirrors Metal's identical assertion).
	rf.Reset()
	ids2 := ids[:8]
	want2, err := mCPU.HiddenLast(ids2)
	if err != nil {
		t.Fatalf("cpu HiddenLast (2nd, shorter): %v", err)
	}
	got2, err := rh.HiddenLast(context.Background(), embs[:8], 0)
	if err != nil {
		t.Fatalf("resident HiddenLast (2nd, shorter): %v", err)
	}
	cos2, maxAbs2 := cosF32(want2, got2)
	t.Logf("HiddenLast over %d tokens (2nd, after Reset): cosine=%.8f maxAbs=%.6g", len(ids2), cos2, maxAbs2)
	if cos2 < hiddenLastResidentBarCUDA {
		t.Errorf("resident HiddenLast (2nd sequence, after Reset) cosine %.8f < %.4f — a fresh "+
			"HiddenLast sequence must not see KV left over from the previous one", cos2, hiddenLastResidentBarCUDA)
	}
}
