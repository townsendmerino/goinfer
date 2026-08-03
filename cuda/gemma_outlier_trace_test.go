//go:build cuda

package cuda

import (
	"math"
	"os"
	"sort"
	"testing"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/goinfer/decoder"
)

// captureResidualForTest returns the PRE-FINAL-NORM residual stream (r.x, [hidden]) after one
// resident token. launchToken's tail — final rmsnorm then LM head — reads r.x into r.aq and
// reads THAT into r.logits, so neither overwrites r.x; after launchToken it still holds the
// residual entering the final norm. This is the exact tap the Metal bisect uses on the CPU side
// (ForwardCapture hidden[nL-1]).
func (r *cudaResident) captureResidualForTest(emb []float32, pos int) ([]float32, error) {
	var out []float32
	err := r.do(func() error {
		if e := r.launchToken(emb, pos, true); e != nil {
			return e
		}
		if e := r.stream.Sync(); e != nil {
			return e
		}
		h := make([]float32, r.hidden)
		if e := gpu.Download(r.x, h); e != nil {
			return e
		}
		out = h
		return nil
	})
	return out, err
}

// TestGemmaOutlierTrace is the CUDA-side answer to the Metal Gemma bisect: does the resident
// int4 forward stay clean on Gemma's massive-activation channels, or diverge there the way
// Metal does?
//
// Coordinates confirmed by the Metal harness (bisect_test.go):
//   - tap  : pre-final-norm residual hidden state, H=2560, hidden_size indices (not head-local)
//   - prompt: "The capital of France is" -> [2 669 5279 529 7001 563]
//   - probe: pos 5 (" is", id 563) — deliberately NOT BOS; at pos 0 attn output = v0 and RoPE
//     is identity, so the suspect ops don't even run.
//
// Three-way so the mechanism is separable, exactly like the Metal-side control:
//   - CUDA-int4 vs CPU-int4 : is the RESIDENT path clean? (the actual question)
//   - CPU-int8  vs CPU-int4 : how much does weight precision alone move these channels?
//
// This is a DIAGNOSTIC: it asserts nothing and prints the table. The massive-activation
// channels are identified here by magnitude (top-|value| in the CPU-int4 residual) rather than
// taken on faith from the relayed index list, which doubles as a cross-check — if my top-K by
// magnitude matches the Mac's {1698,1730,...}, the two harnesses agree on WHICH channels; if
// not, one of us is tapping the wrong thing.
func TestGemmaOutlierTrace(t *testing.T) {
	requireHeavyModel(t)
	path := os.ExpandEnv("$HOME/models/gemma-3-4b-it")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s", path)
	}
	// [2 669 5279 529 7001 563] = "<bos> The capital of France is" in Gemma's own tokenizer;
	// pinned here so the trace does not depend on a tokenizer load and matches the Metal probe
	// token-for-token. Probe is the LAST position.
	prompt := []int{2, 669, 5279, 529, 7001, 563}
	const probe = 5

	mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cuda int4): %v", err)
	}
	defer mc.Close()
	rfIface := mc.ResidentForwardForTest()
	if rfIface == nil {
		t.Fatal("cuda resident DECLINED gemma-3 — the trace needs the resident path")
	}
	rf, ok := rfIface.(*cudaResident)
	if !ok {
		t.Fatalf("resident forward is %T, not *cudaResident", rfIface)
	}
	H, nL, _, _, _, _, _ := mc.Dims()

	mc4, err := decoder.Load(path, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cpu int4): %v", err)
	}
	defer mc4.Close()
	mc8, err := decoder.Load(path, decoder.Options{Quant: "int8"})
	if err != nil {
		t.Fatalf("load (cpu int8): %v", err)
	}
	defer mc8.Close()
	// f32 is the GROUND TRUTH — the reference the Metal bisect is (probably) tapping. Without it,
	// every "sign flip" is measured against CPU-int4, which is itself lossy: if int4 quant flips a
	// channel, both CPU-int4 and CUDA-int4 flip TOGETHER and the resident path looks clean when the
	// real story is "int4 flips this channel on every backend." f32 separates int4-inherent flips
	// from backend-specific ones.
	mcF, err := decoder.Load(path, decoder.Options{Quant: ""})
	if err != nil {
		t.Fatalf("load (cpu f32): %v", err)
	}
	defer mcF.Close()

	// CPU pre-final-norm residual at the probe: ForwardCapture layer nL-1, fed the whole prompt.
	cpuResidual := func(m *decoder.Model) []float32 {
		cache := m.NewCache(len(prompt) + 1)
		var got []float32
		for i, tok := range prompt {
			_, hid, err := m.ForwardCapture(tok, cache, []int{nL - 1})
			if err != nil {
				t.Fatalf("ForwardCapture pos %d: %v", i, err)
			}
			if i == probe {
				got = append([]float32(nil), hid[0]...)
			}
		}
		return got
	}
	cpu4 := cpuResidual(mc4)
	cpu8 := cpuResidual(mc8)
	cpuF := cpuResidual(mcF)

	// CUDA resident residual at the probe: same prompt, same positions.
	var cuda4 []float32
	for i, tok := range prompt {
		emb := mc.EmbedResidentForTest(tok)
		if i == probe {
			h, err := rf.captureResidualForTest(emb, i)
			if err != nil {
				t.Fatalf("cuda capture pos %d: %v", i, err)
			}
			cuda4 = h
		} else if _, err := rf.Forward(emb, i); err != nil {
			t.Fatalf("cuda Forward pos %d: %v", i, err)
		}
	}
	if len(cpu4) != H || len(cpu8) != H || len(cuda4) != H {
		t.Fatalf("residual widths: cpu4=%d cpu8=%d cuda4=%d, want H=%d", len(cpu4), len(cpu8), len(cuda4), H)
	}

	// Whole-vector cosines: the aggregate signal, so a per-channel table is read in context.
	t.Logf("probe = pos %d (id %d), H=%d, pre-final-norm residual", probe, prompt[probe], H)
	t.Logf("cosine  CUDA-int4 vs CPU-int4 = %.6f", cosine(cuda4, cpu4))
	t.Logf("cosine  CPU-int8  vs CPU-int4 = %.6f", cosine(cpu8, cpu4))

	// Massive-activation channels: the largest |CPU-int4| dims. These are the ones the final
	// RMSNorm amplifies into the head input, so a divergence here is the one that matters.
	idx := make([]int, H)
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return math.Abs(float64(cpu4[idx[a]])) > math.Abs(float64(cpu4[idx[b]])) })
	const topK = 16
	t.Logf("top-%d massive-activation channels (by |CPU-int4|):", topK)
	t.Logf("  %-6s %14s %14s %14s   %10s %10s", "chan", "CPU-int4", "CPU-int8", "CUDA-int4", "d(cuda)%", "d(int8)%")
	for r := 0; r < topK && r < H; r++ {
		c := idx[r]
		v4, v8, vc := cpu4[c], cpu8[c], cuda4[c]
		dc := relPct(vc, v4)
		d8 := relPct(v8, v4)
		flag := ""
		if (v4 > 0) != (vc > 0) && math.Abs(float64(v4)) > 1 {
			flag = "  <- SIGN FLIP (cuda)"
		} else if math.Abs(float64(vc)) < 0.1*math.Abs(float64(v4)) && math.Abs(float64(v4)) > 1 {
			flag = "  <- ZEROED (cuda)"
		}
		t.Logf("  %-6d %14.4f %14.4f %14.4f   %9.2f%% %9.2f%%%s", c, v4, v8, vc, dc, d8, flag)
	}

	// The channels the Metal bisect fingered as the ACTUAL failure — mid-magnitude trunk
	// channels Metal sign-flips or zeroes, which the final (1+w) norm (~16-20x) then amplifies
	// into the head. The massive channel 443 both backends track; the catastrophe is here. This
	// is the definitive oracle question: does CUDA keep these channels' SIGN, where Metal loses
	// it? Neighbors +/-1 printed too, to catch any index-offset between the two harnesses (my
	// drift table had 1697 with the opposite CPU sign the Mac reports at 1698).
	// HANDSHAKE ANCHORS: the values the Mac must match to prove we run the same input. If f32@443
	// differs across boxes on identical code, the inputs differ (checkpoint / aikit) and no oracle
	// is valid yet. If f32 matches but their "CPU" == my f32, the earlier int4-vs-int4 mismatch was
	// a precision-label confusion, not a bug.
	for _, c := range []int{443, 19, 172, 1698} {
		t.Logf("ANCHOR chan %-5d  f32=%14.4f  int4=%14.4f  int8=%14.4f  cuda-int4=%14.4f",
			c, cpuF[c], cpu4[c], cpu8[c], cuda4[c])
	}

	metalFlagged := []int{1698, 1730, 2482, 1723, 227}
	t.Logf("Metal-flagged channels vs f32 GROUND TRUTH (does each int path keep f32's sign?):")
	t.Logf("  %-6s %14s %14s %14s %14s   %s", "chan", "f32(truth)", "CPU-int4", "CUDA-int4", "CPU-int8", "verdict")
	sgn := func(x float32) bool { return x > 0 }
	for _, c := range metalFlagged {
		for _, cc := range []int{c - 1, c, c + 1} {
			if cc < 0 || cc >= H {
				continue
			}
			vf, v4, vc, v8 := cpuF[cc], cpu4[cc], cuda4[cc], cpu8[cc]
			// Verdict is measured against f32, and it separates the two causes: if int4 (CPU) and
			// CUDA both flip vs f32, it is int4-inherent; if only CUDA flips, it is resident-specific.
			verdict := "sign OK"
			big := math.Abs(float64(vf)) > 10
			cudaFlip := sgn(vf) != sgn(vc) && big
			int4Flip := sgn(vf) != sgn(v4) && big
			switch {
			case cudaFlip && int4Flip:
				verdict = "int4-inherent flip (CPU+CUDA both, vs f32)"
			case cudaFlip:
				verdict = "*** CUDA-SPECIFIC flip vs f32 ***"
			case int4Flip:
				verdict = "CPU-int4 flips but CUDA holds f32"
			}
			mark := "  "
			if cc == c {
				mark = "=>"
			}
			t.Logf("%s%-6d %14.4f %14.4f %14.4f %14.4f   %s", mark, cc, vf, v4, vc, v8, verdict)
		}
	}

	// Also surface the channels where CUDA-int4 diverges MOST from CPU-int4 in absolute terms —
	// the outlier need not be the same set as the largest magnitudes, and if it is, that IS the
	// finding.
	sort.Slice(idx, func(a, b int) bool {
		return math.Abs(float64(cuda4[idx[a]]-cpu4[idx[a]])) > math.Abs(float64(cuda4[idx[b]]-cpu4[idx[b]]))
	})
	t.Logf("top-%d channels by |CUDA-int4 - CPU-int4| (where the resident path drifts most):", topK)
	t.Logf("  %-6s %14s %14s %14s   %12s", "chan", "CPU-int4", "CUDA-int4", "CPU-int8", "abs.drift")
	for r := 0; r < topK && r < H; r++ {
		c := idx[r]
		t.Logf("  %-6d %14.4f %14.4f %14.4f   %12.4f", c, cpu4[c], cuda4[c], cpu8[c], math.Abs(float64(cuda4[c]-cpu4[c])))
	}
}

func cosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return math.NaN()
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

// relPct is the divergence of v from ref as a percentage of |ref| (0 when ref is ~0).
func relPct(v, ref float32) float64 {
	if math.Abs(float64(ref)) < 1e-6 {
		return 0
	}
	return float64(v-ref) / math.Abs(float64(ref)) * 100
}
