//go:build darwin && goinfer_testhooks

package metal

import (
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestMoEPaging_matchesNonPaged proves the property a cache has to have — generalizing the same
// check gemma4_moe.go's paging carries ("the continuation is BYTE-IDENTICAL across all three slot
// counts"). Builds the resident THREE times against qwen3_5_moe-tiny (nE=4, topK=2: the paged
// configs are slots=2, forcing eviction every token since it equals topK, and slots=3, one spare)
// under GOINFER_METAL_MOE_SLOTS unset/2/3, drives the SAME token sequence through each, and
// requires EXACT logit equality — not a cosine floor. Reuse must not be observable in the output.
func TestMoEPaging_matchesNonPaged(t *testing.T) {
	const ckpt = "../decoder/testdata/qwen3_5_moe-tiny"
	if _, err := os.Stat(ckpt + "/model.safetensors"); err != nil {
		t.Skipf("no qwen3_5_moe fixture at %s (run scripts/pin_qwen3_5_forward.py --moe): %v", ckpt, err)
	}

	run := func(t *testing.T, slotsEnv string) [][]float32 {
		if slotsEnv == "" {
			os.Unsetenv("GOINFER_METAL_MOE_SLOTS")
		} else {
			t.Setenv("GOINFER_METAL_MOE_SLOTS", slotsEnv)
		}
		mRes, err := decoder.Load(ckpt, decoder.Options{Backend: "metal", Quant: "int4"})
		if err != nil {
			t.Fatalf("load metal (slots=%q): %v", slotsEnv, err)
		}
		defer mRes.Close()
		rf := mRes.ResidentForwardForTest()
		if rf == nil {
			t.Fatalf("not resident (slots=%q): %s", slotsEnv, mRes.ResidentDecline())
		}
		_, _, _, _, _, _, vocab := mRes.Dims()
		rf.Reset()
		const ntok = 32
		out := make([][]float32, ntok)
		for i := range ntok {
			lr, err := rf.Forward(mRes.EmbedResidentForTest((i*131+7)%vocab), i)
			if err != nil {
				t.Fatalf("forward[%d] (slots=%q): %v", i, slotsEnv, err)
			}
			out[i] = append([]float32(nil), lr...)
		}
		return out
	}
	// t.Setenv forbids parallel subtests sharing the var, but these run sequentially anyway (each
	// needs the FULL resident build to see the env var it was set before).
	base := run(t, "") // non-paged: all 4 experts stacked resident, today's existing behavior
	for _, slots := range []string{"2", "3"} {
		t.Run("slots="+slots, func(t *testing.T) {
			got := run(t, slots)
			for i := range base {
				if len(got[i]) != len(base[i]) {
					t.Fatalf("tok %d: length %d != %d", i, len(got[i]), len(base[i]))
				}
				for j := range base[i] {
					if got[i][j] != base[i][j] {
						t.Fatalf("tok %d elem %d: paged(slots=%s)=%v != non-paged=%v — "+
							"reuse is observable in the output, which a cache must never allow",
							i, j, slots, got[i][j], base[i][j])
					}
				}
			}
			t.Logf("slots=%s: %d tokens, exact logit match against the non-paged (all-experts-resident) run", slots, len(base))
		})
	}
}
