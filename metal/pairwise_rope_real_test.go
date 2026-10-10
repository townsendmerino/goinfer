//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime/debug"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// G-PR4 (per position) and G-PR5 (docs/tasks/task-metal-pairwise-rope-2026-10.md): the pairwise families on the Metal
// resident against the CPU at int4, on the real checkpoints. 64 tokens of real text (README.md) and a 16-token greedy
// continuation the CPU chose, teacher-forced on the resident. Bars, CUDA's real-checkpoint tier
// (docs/measurements/cuda-pairwise-rope-2026-10-01.md): mean per-position cosine >= 0.99 and worst >= 0.90, and every
// continuation argmax difference a near-tie (the resident's token within 3% of the CPU's logit range of the CPU's top).
// The NeoX kernels rebound into the same resident must read mean < 0.99 or worst < 0.90.
//
// The CPU reference is computed first and its model closed before the Metal load, so an 8B model is never held twice.
// GOINFER_HEAVY_TESTS=1; a model missing from ~/models is skipped by name.
func TestPairwiseRoPERealMetal(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1")
	}
	home, _ := os.UserHomeDir()
	readme, err := os.ReadFile("../README.md")
	if err != nil {
		t.Fatal(err)
	}
	const nPrompt, nCont = 64, 16
	for _, name := range []string{"glm-ocr", "command-r7b", "aya-expanse-8b"} {
		t.Run(name, func(t *testing.T) {
			dir := filepath.Join(home, "models", name)
			if _, err := os.Stat(filepath.Join(dir, "config.json")); err != nil {
				t.Skipf("no %s", dir)
			}
			tk, err := tokenizer.Load(filepath.Join(dir, "tokenizer.json"))
			if err != nil {
				t.Fatal(err)
			}
			ids, err := tk.Encode(string(readme), false)
			if err != nil || len(ids) < nPrompt {
				t.Fatalf("encode README: %d ids, %v", len(ids), err)
			}
			ids = ids[:nPrompt]
			t0 := time.Now()
			mc, err := decoder.Load(dir, decoder.Options{Backend: "cpu", Quant: "int4"})
			if err != nil {
				t.Fatalf("load cpu: %v", err)
			}
			fmt.Fprintf(os.Stderr, "[%s] cpu int4 loaded in %s\n", name, time.Since(t0).Round(time.Second))
			c := mc.NewCache(nPrompt + nCont)
			var ref [][]float32
			toks := append([]int(nil), ids...)
			for i := 0; i < nPrompt+nCont; i++ {
				l, err := mc.ForwardForTest(toks[i], c)
				if err != nil {
					t.Fatal(err)
				}
				ref = append(ref, append([]float32(nil), l...))
				if i >= nPrompt-1 && len(toks) < nPrompt+nCont {
					toks = append(toks, argmaxF(l))
				}
			}
			mc.Close()
			// Hand the CPU model's heap back to the OS before the Metal load: Metal's resident guard prices against the
			// machine's live-available memory, which a closed-but-unreturned 8 GB heap would still occupy.
			mc = nil
			debug.FreeOSMemory()
			fmt.Fprintf(os.Stderr, "[%s] cpu reference: %d positions in %s\n", name, len(ref), time.Since(t0).Round(time.Second))
			// Through the Metal sidecar when there is one (its int4 weights are mapped, not held twice, so an 8B fits
			// Metal's resident guard on a 16 GB Mac); it generates exactly what the directory load does.
			msrc := dir
			if _, err := os.Stat(dir + ".int4.metal.giw"); err == nil {
				msrc = dir + ".int4.metal.giw"
			}
			mr, err := decoder.Load(msrc, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 512})
			if err != nil {
				t.Fatalf("load metal: %v", err)
			}
			defer mr.Close()
			a, ok := mr.ResidentForwardForTest().(*metalResident)
			if !ok {
				t.Fatalf("not Metal-resident: %s", mr.ResidentDecline())
			}
			if !a.r.pairwiseRoPE {
				t.Fatal("pairwise twins not bound")
			}
			measure := func() (mean, worst float64, worstPos, real, ties int) {
				a.Reset()
				worst = 1
				for i, tok := range toks {
					l, err := a.Forward(mr.EmbedResidentForTest(tok), i)
					if err != nil {
						t.Fatal(err)
					}
					cs := cosF(l, ref[i])
					mean += cs
					if cs < worst {
						worst, worstPos = cs, i
					}
					if i >= nPrompt-1 && i < len(toks)-1 {
						if want, got := argmaxF(ref[i]), argmaxF(l); want != got {
							lo, hi := ref[i][0], ref[i][0]
							for _, v := range ref[i] {
								lo, hi = min(lo, v), max(hi, v)
							}
							if gap := float64(ref[i][want]-ref[i][got]) / float64(hi-lo); gap <= 0.03 {
								ties++
							} else {
								real++
							}
						}
					}
				}
				return mean / float64(len(toks)), worst, worstPos, real, ties
			}
			mean, worst, wp, real, ties := measure()
			fmt.Fprintf(os.Stderr, "[%s] PAIRWISE %d positions: mean cos %.6f, worst %.6f (pos %d); continuation flips: %d near-tie, %d real\n", name, len(toks), mean, worst, wp, ties, real)
			// The mean bar is 0.99 for every model but Command-R7B, whose 0.989242 the owner accepted (docs/tasks/task-metal-pairwise-rope-2026-10.md
			// G-PR5). Part C of docs/tasks/task-metal-pairwise-followups-2026-10.md localizes that gap; this floor records the decision, it is
			// not a measured tolerance.
			meanBar := 0.99
			if name == "command-r7b" {
				meanBar = 0.989
			}
			if mean < meanBar || worst < 0.90 || real > 0 {
				t.Errorf("%s: mean %.6f (>= %.3f), worst %.6f (>= 0.90), %d non-tie continuation flips (0)", name, mean, meanBar, worst, real)
			}
			// The batched pass (docs/tasks/task-metal-pairwise-followups-2026-10.md, G-A2/G-B3): the prompt through PrefillLast,
			// its last-token logits against the CPU's (>= 0.98, CUDA's real tier), then the continuation decoded after it,
			// teacher-forced, against the CPU per position (the decode bars) with no non-tie flip.
			embs := make([][]float32, nPrompt)
			for i := range nPrompt {
				embs[i] = mr.EmbedResidentForTest(toks[i])
			}
			a.Reset()
			if last, perr := a.PrefillLast(context.Background(), embs, 0); perr != nil {
				fmt.Fprintf(os.Stderr, "[%s] batched prefill declined, NOT graded: %v\n", name, perr)
			} else {
				pc := cosF(last, ref[nPrompt-1])
				var bmean, bworst float64 = 0, 1
				breal := 0
				for k := nPrompt; k < len(toks); k++ {
					l, err := a.Forward(mr.EmbedResidentForTest(toks[k]), k)
					if err != nil {
						t.Fatal(err)
					}
					cs := cosF(l, ref[k])
					bmean += cs
					bworst = min(bworst, cs)
					if k < len(toks)-1 {
						if want, got := argmaxF(ref[k]), argmaxF(l); want != got {
							lo, hi := ref[k][0], ref[k][0]
							for _, v := range ref[k] {
								lo, hi = min(lo, v), max(hi, v)
							}
							if float64(ref[k][want]-ref[k][got])/float64(hi-lo) > 0.03 {
								breal++
							}
						}
					}
				}
				bmean /= float64(len(toks) - nPrompt)
				fmt.Fprintf(os.Stderr, "[%s] BATCHED prefill last-token cos %.6f; decode after it: mean %.6f, worst %.6f, %d non-tie flips\n", name, pc, bmean, bworst, breal)
				// G-B3 as registered: the prefill's last token >= 0.98 and no non-tie flip in the decode after it. The decode
				// mean and worst after a batched prefill are logged, not graded: they carry the same Metal decode gap the
				// plain decode path shows (Part C).
				if pc < 0.98 || breal > 0 {
					t.Errorf("%s batched: prefill cos %.6f (>= 0.98), %d non-tie flips in the decode after it (0)", name, pc, breal)
				}
			}
			restore := pwForceNeoX(t, a)
			nm, nw, nwp, _, _ := measure()
			restore()
			fmt.Fprintf(os.Stderr, "[%s] NeoX     %d positions: mean cos %.6f, worst %.6f (pos %d)\n", name, len(toks), nm, nw, nwp)
			if nm >= 0.99 && nw >= 0.90 {
				t.Errorf("%s: the gate is BLIND: with the NeoX kernels it reads mean %.6f, worst %.6f", name, nm, nw)
			}
		})
	}
}
