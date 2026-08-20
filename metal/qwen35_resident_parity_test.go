//go:build darwin && goinfer_testhooks

package metal

import (
	"os"
	"strconv"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestQwen35ResidentParityMetal is the Metal twin of cuda's TestQwen35ResidentParityCUDA — the
// whole-model gate the "Done means" bar in the porting task requires before FeatDeltaNet is real,
// not just declared. Read that file's header comment; this mirrors its structure exactly: the
// resident runner vs the CPU runLayersQwen35, fed the same fixed token sequence, run LONG (three
// of every four layers carry compounding recurrent state, so a wiring bug shows as a decaying
// cosine, not a bad first token), then a REPLAY after Reset to prove the state actually resets
// rather than merely starting zeroed.
func TestQwen35ResidentParityMetal(t *testing.T) {
	for _, fx := range []struct{ name, quant string }{
		{"qwen3_5-tiny", "int8int8"},
		{"qwen3_5_moe-tiny", "int4"},
	} {
		t.Run(fx.name, func(t *testing.T) {
			qwen35ResidentParityMetal(t, "../decoder/testdata/"+fx.name, fx.quant)
		})
	}
	if ck := os.Getenv("GOINFER_DNET_CKPT_METAL"); ck != "" { // a bigger/real checkpoint, opt-in
		t.Run("env", func(t *testing.T) { qwen35ResidentParityMetal(t, ck, "int8int8") })
	}
}

func qwen35ResidentParityMetal(t *testing.T, ckpt, quant string) {
	if _, err := os.Stat(ckpt + "/model.safetensors"); err != nil {
		// stat the WEIGHTS, not the directory: the fixture's config.json is committed while
		// model.safetensors is gitignored, so a dir-existence guard would flip this skip into a
		// hard failure on a fresh clone.
		t.Skipf("no qwen3_5 fixture at %s (run scripts/pin_qwen3_5_forward.py [--moe]): %v", ckpt, err)
	}
	if v := os.Getenv("GOINFER_DNET_QUANT_METAL"); v != "" {
		quant = v
	}

	mRes, err := decoder.Load(ckpt, decoder.Options{Backend: "metal", Quant: quant})
	if err != nil {
		t.Fatalf("load metal: %v", err)
	}
	defer mRes.Close()
	rf := mRes.ResidentForwardForTest()
	if rf == nil {
		// Non-vacuity: this test is worthless if it silently measures CPU-vs-CPU, and a declined
		// residency is exactly how that happens.
		t.Fatalf("qwen3.5 did not go resident (BuildResident refused) — decode path %q; decline: %s", mRes.DecodePath(), mRes.ResidentDecline())
	}
	t.Logf("resident decode path: %s", mRes.DecodePath())

	// The CPU arm runs at the SAME weight quant, so what is being measured is WIRING, not the
	// int8 gap. The resident path still quantizes activations the CPU keeps in f32, which is the
	// floor this cosine sits on — it should be flat, not decaying.
	mCPU, err := decoder.Load(ckpt, decoder.Options{Backend: "cpu", Quant: quant})
	if err != nil {
		t.Fatalf("load cpu: %v", err)
	}
	defer mCPU.Close()

	ntok := 128
	if v := os.Getenv("GOINFER_DNET_NTOK_METAL"); v != "" {
		if n, e := strconv.Atoi(v); e == nil {
			ntok = n
		}
	}
	_, _, _, _, _, _, vocab := mCPU.Dims()
	checks := map[int]bool{1: true, 2: true, 4: true, 8: true, 16: true, 64: true, 128: true, 512: true}

	cache := mCPU.NewCache(ntok + 1)
	rf.Reset()
	worst, first, agree := 1.0, 1.0, 0
	var firstRun [][]float32 // the resident's own opening logits, for the replay check below
	for i := range ntok {
		tok := (i*131 + 7) % vocab
		lc, err := mCPU.ForwardForTest(tok, cache)
		if err != nil {
			t.Fatalf("cpu forward[%d]: %v", i, err)
		}
		lr, err := rf.Forward(mRes.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("resident forward[%d]: %v", i, err)
		}
		if len(firstRun) < 16 {
			firstRun = append(firstRun, append([]float32(nil), lr...))
		}
		if argmaxF32(lc) == argmaxF32(lr) {
			agree++
		}
		cos, maxAbs := cosF32(lc, lr)
		if i == 0 {
			first = cos
		}
		if cos < worst {
			worst = cos
		}
		if checks[i+1] || i+1 == ntok {
			t.Logf("  tok %4d: cosine=%.6f maxAbs=%.4g argmax_match=%v", i+1, cos, maxAbs, argmaxF32(lc) == argmaxF32(lr))
		}
	}
	t.Logf("  worst cosine over %d tokens: %.6f (tok1 %.6f, drift %.4f); argmax agree %d/%d",
		ntok, worst, first, first-worst, agree, ntok)

	// DRIFT is the wiring signal, checked separately from the absolute floor on purpose: a
	// recurrence that is merely coarse sits flat and low, while one that is wired wrong decays.
	if first-worst > 0.02 {
		t.Errorf("cosine DRIFTS over the run: tok1 %.6f -> worst %.6f (delta %.4f). The recurrent state "+
			"compounds; a decaying cosine is a wiring error, not quantization noise", first, worst, first-worst)
	}
	if worst < 0.95 {
		t.Errorf("worst cosine %.6f < 0.95 — below the resident-path floor the other families hold", worst)
	}

	// SECOND GENERATION on the same model — a REPLAY, not another CPU comparison. The recurrent
	// state compounds and is NOT positional, so unlike a KV cache the next sequence cannot simply
	// overwrite it; it has to be zeroed (audit C-01's CUDA analogue). Comparing generation 2
	// against the CPU would not catch a missing reset either — the decay gate shrinks stale state
	// faster than a short comparison notices (measured on CUDA). So this replays the first run's
	// opening tokens and requires the resident to reproduce ITS OWN logits: same inputs + same
	// (reset) state => the same output, to f32 determinism.
	const replay = 16
	rf.Reset()
	worstReplay := 1.0
	for i := range replay {
		lr, err := rf.Forward(mRes.EmbedResidentForTest((i*131+7)%vocab), i)
		if err != nil {
			t.Fatalf("resident replay[%d]: %v", i, err)
		}
		if cos, _ := cosF32(firstRun[i], lr); cos < worstReplay {
			worstReplay = cos
		}
	}
	t.Logf("  replay of the first %d tokens after Reset: worst self-cosine %.9f", replay, worstReplay)
	if worstReplay < 0.9999999 {
		t.Errorf("replaying the same tokens after Reset gives DIFFERENT logits (worst self-cosine "+
			"%.9f): the recurrent state carried over from the first sequence. metalResident.Reset "+
			"must zero the DeltaNet {win, state}", worstReplay)
	}
}
