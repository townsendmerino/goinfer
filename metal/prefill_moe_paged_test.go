//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"math"
	"os"
	"os/exec"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestMoELayerMajor_matchesSequential is prefillMoEPaged's gate, the generic twin of TestG4LayerMajor_matchesSequential:
// on Mixtral, two Qwen3-MoE shapes and the Qwen3.5 DeltaNet MoE hybrid, paged at 2 and 3 slots, prompts of 1, 5 and 9
// rows from position 0 and 6 rows continuing a 3-row prefix, through the layer-major path and through the sequential
// loop. The last row's logits and 3 decode steps after it (which read every layer's K/V and DeltaNet state) must be
// equal bit for bit, and every attention layer's K/V. Each arm starts from Reset (a DeltaNet's state zeroed). The layer-major
// arm runs KS-D9's batched phase 2, which every fixture here admits and every prompt shape must take (D9-G1), and a third arm,
// layer-major with the per-row phase 2 (moeExpertBatchOn off), must equal it bit for bit too. A resident the backend declines
// fails the test; only a slot count under the model's top-k (which the pager refuses) skips. Under GOINFER_KSD9_PLANT_CHILD=1
// (set by the plant check, TestKSD9_plantSeenByTheGate, in its own process) the startup self-test is skipped, so a planted
// defect reaches these cells.
func TestMoELayerMajor_matchesSequential(t *testing.T) {
	if os.Getenv("GOINFER_KSD9_PLANT_CHILD") == "1" {
		decoder.SkipSelfTests()
	}
	for _, fx := range []string{"../testdata/mixtral-tiny", "../testdata/qwen3moe-tiny", "../testdata/qwen3moe-tiny-k3", "../decoder/testdata/qwen3_5_moe-tiny"} {
		for _, slots := range []int{2, 3} {
			t.Run(fx+"/slots="+strconv.Itoa(slots), func(t *testing.T) {
				if _, err := os.Stat(fx + "/model.safetensors"); err != nil {
					t.Skipf("no fixture: %v", err)
				}
				t.Setenv("GOINFER_METAL_MOE_SLOTS", strconv.Itoa(slots))
				m, err := decoder.Load(fx, decoder.Options{Backend: "metal", Quant: "int4"})
				if err != nil {
					t.Fatalf("load: %v", err)
				}
				defer m.Close()
				mr, ok := m.ResidentForwardForTest().(*metalResident)
				if !ok || mr.r.moe == nil || !mr.r.moe.paged {
					if why := m.ResidentDecline(); strings.Contains(why, "need integer >= topK") {
						t.Skipf("%d slots is under the model's top-k: %s", slots, why)
					}
					t.Fatalf("no paged Metal MoE resident at %d slots: %s", slots, m.ResidentDecline())
				}
				r := mr.r
				if !r.moeExpertBatchOK() {
					t.Fatalf("the batched phase 2 does not admit this fixture")
				}
				_, _, _, _, _, _, vocab := m.Dims()
				toks := make([]int, 16)
				for i := range toks {
					toks[i] = (i*29 + 7) % vocab
				}
				emb := func(i int) []float32 { return m.EmbedResidentForTest(toks[i]) }
				kv := func(n int) [][]uint16 {
					var out [][]uint16
					for l := range r.layers {
						if r.layers[l].delta != nil || r.kc[l] == (Buffer{}) {
							continue
						}
						d := r.layers[l].geom.kvDim
						o := r.kvHostOff(l, 2)
						out = append(out, append([]uint16(nil), r.kc[l].U16s()[o:o+n*d]...), append([]uint16(nil), r.vc[l].U16s()[o:o+n*d]...))
					}
					return out
				}
				for _, c := range []struct{ prefix, n int }{{0, 1}, {0, 5}, {0, 9}, {3, 6}} {
					total := c.prefix + c.n
					run := func(layerMajor, batched bool) (last []float32, kvs [][]uint16, dec [][]float32) {
						prevOn := moeExpertBatchOn
						moeExpertBatchOn = batched
						defer func() { moeExpertBatchOn = prevOn }()
						mr.Reset()
						for i := range c.prefix {
							if err := mr.ForwardNoLogits(emb(i), i); err != nil {
								t.Fatal(err)
							}
						}
						if layerMajor {
							embs := make([][]float32, c.n)
							for i := range embs {
								embs[i] = emb(c.prefix + i)
							}
							groups := r.moeBatchGroups
							last = append([]float32(nil), r.prefillMoEPaged(embs, c.prefix, true)...)
							if err := r.takeExecErr(); err != nil {
								t.Fatal(err)
							}
							if took := r.moeBatchGroups > groups; took != batched {
								t.Fatalf("prefix %d, %d rows: batched phase 2 taken %v, want %v", c.prefix, c.n, took, batched)
							}
						} else {
							for i := c.prefix; i < total; i++ {
								if i < total-1 {
									if err := mr.ForwardNoLogits(emb(i), i); err != nil {
										t.Fatal(err)
									}
									continue
								}
								lg, err := mr.Forward(emb(i), i)
								if err != nil {
									t.Fatal(err)
								}
								last = append([]float32(nil), lg...)
							}
						}
						kvs = kv(total)
						for s := range 3 {
							lg, err := mr.Forward(emb(total+s), total+s)
							if err != nil {
								t.Fatal(err)
							}
							dec = append(dec, append([]float32(nil), lg...))
						}
						return
					}
					sLast, sKV, sDec := run(false, true)
					lLast, lKV, lDec := run(true, true)
					pLast, pKV, pDec := run(true, false)
					label := "prefix " + strconv.Itoa(c.prefix) + ", " + strconv.Itoa(c.n) + " rows"
					same := func(what string, a, b []float32) {
						if len(a) != len(b) {
							t.Fatalf("%s: %s: %d logits against %d", label, what, len(a), len(b))
						}
						for j := range a {
							if math.Float32bits(a[j]) != math.Float32bits(b[j]) {
								t.Fatalf("%s: %s: logit %d %v against %v", label, what, j, a[j], b[j])
							}
						}
					}
					same("batched against per-row phase 2, last row", lLast, pLast)
					for s := range lDec {
						same("batched against per-row phase 2, decode step "+strconv.Itoa(s), lDec[s], pDec[s])
					}
					for i := range lKV {
						if !slices.Equal(lKV[i], pKV[i]) {
							t.Fatalf("%s: batched against per-row phase 2: K/V differ", label)
						}
					}
					if len(lLast) != len(sLast) {
						t.Fatalf("%s: %d logits, sequential %d", label, len(lLast), len(sLast))
					}
					for j := range sLast {
						if math.Float32bits(sLast[j]) != math.Float32bits(lLast[j]) {
							t.Fatalf("%s: last-row logit %d: layer-major %v, sequential %v", label, j, lLast[j], sLast[j])
						}
					}
					for i := range sKV {
						for j := range sKV[i] {
							if sKV[i][j] != lKV[i][j] {
								t.Fatalf("%s: K/V differ", label)
							}
						}
					}
					for s := range sDec {
						for j := range sDec[s] {
							if math.Float32bits(sDec[s][j]) != math.Float32bits(lDec[s][j]) {
								t.Fatalf("%s: decode step %d logit %d differs", label, s, j)
							}
						}
					}
				}
				t.Logf("4 prompt shapes: logits, K/V (%d attention layers) and 3 decode steps bit-identical, sequential, layer-major batched and per-row; %d batched slot groups",
					len(kv(1))/2, r.moeBatchGroups)
			})
		}
	}
}

// TestMoELayerMajor_generateThroughDecoder tests prefillMoEPaged through its caller: decoder.Generate on a paged Mixtral
// and the paged Qwen3.5 DeltaNet MoE hybrid, a 12-token prompt and 16 greedy tokens, with moeLayerMajorOn and without:
// equal tokens, and the layer-major arm took the route (PrefillLast is offered suffixes of 8 or more).
func TestMoELayerMajor_generateThroughDecoder(t *testing.T) {
	for _, fx := range []string{"../testdata/mixtral-tiny", "../decoder/testdata/qwen3_5_moe-tiny"} {
		t.Run(fx, func(t *testing.T) {
			if _, err := os.Stat(fx + "/model.safetensors"); err != nil {
				t.Skipf("no fixture: %v", err)
			}
			t.Setenv("GOINFER_METAL_MOE_SLOTS", "3")
			run := func(on bool) ([]int, int) {
				prev := moeLayerMajorOn
				moeLayerMajorOn = on
				defer func() { moeLayerMajorOn = prev }()
				m, err := decoder.Load(fx, decoder.Options{Backend: "metal", Quant: "int4"})
				if err != nil {
					t.Fatal(err)
				}
				defer m.Close()
				mr, ok := m.ResidentForwardForTest().(*metalResident)
				if !ok || mr.r.moe == nil || !mr.r.moe.paged {
					t.Fatalf("not a paged generic MoE (%s)", m.ResidentDecline())
				}
				_, _, _, _, _, _, vocab := m.Dims()
				prompt := make([]int, 12)
				for i := range prompt {
					prompt[i] = (i*41 + 5) % vocab
				}
				ch, g := m.Generate(context.Background(), prompt, 16, decoder.SamplingParams{})
				var ids []int
				for id := range ch {
					ids = append(ids, id)
				}
				if err := g.Err(); err != nil {
					t.Fatal(err)
				}
				return ids, mr.r.moeLayerMajorRuns
			}
			off, offRuns := run(false)
			on, onRuns := run(true)
			if offRuns != 0 || onRuns == 0 {
				t.Fatalf("layer-major runs: %d off, %d on", offRuns, onRuns)
			}
			if len(on) != len(off) {
				t.Fatalf("%d tokens on, %d off", len(on), len(off))
			}
			for i := range off {
				if on[i] != off[i] {
					t.Fatalf("token %d: layer-major %v, sequential %v", i, on, off)
				}
			}
			t.Logf("%d tokens equal with and without layer-major (%d layer-major prefill)", len(on), onRuns)
		})
	}
}

// TestKSD9_plantSeenByTheGate is KS-D9's planted-defect check: with a defect planted in the batched phase 2 by hand, the startup
// self-test declines the backend first, so it re-runs TestMoELayerMajor_matchesSequential in a child process with the self-test
// skipped (GOINFER_KSD9_PLANT_CHILD=1, which skips it for that process only) and passes iff the child fails. Armed by
// GOINFER_KSD9_PLANT=1 only; unarmed, it skips.
func TestKSD9_plantSeenByTheGate(t *testing.T) {
	if os.Getenv("GOINFER_KSD9_PLANT") != "1" || os.Getenv("GOINFER_KSD9_PLANT_CHILD") == "1" {
		t.Skip("set GOINFER_KSD9_PLANT=1 with a defect planted in the batched phase 2")
	}
	cmd := exec.Command(os.Args[0], "-test.run", "^TestMoELayerMajor_matchesSequential$", "-test.count=1")
	cmd.Env = append(os.Environ(), "GOINFER_KSD9_PLANT_CHILD=1")
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatalf("the gate passed with the plant in place:\n%s", out)
	}
	t.Logf("the gate failed with the plant in place, as it must:\n%s", out)
}

// TestKSD9_scratchWithinPlan: the batched phase 2's scratch is allocated once per layer-major call, sized for its rows, and
// priced in the plan. On paged Mixtral (2 slots), one 9-row prefillMoEPaged counts exactly moeBatchBytes(9) in
// prefillScratchSeen, and prefillScratchBytes grows by moeBatchBytes of min(M, g4LayerMajorChunk) when the batched phase 2 is on.
func TestKSD9_scratchWithinPlan(t *testing.T) {
	const fx = "../testdata/mixtral-tiny"
	if _, err := os.Stat(fx + "/model.safetensors"); err != nil {
		t.Skipf("no fixture: %v", err)
	}
	t.Setenv("GOINFER_METAL_MOE_SLOTS", "2")
	m, err := decoder.Load(fx, decoder.Options{Backend: "metal", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	mr, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok || mr.r.moe == nil || !mr.r.moe.paged || !mr.r.moeExpertBatchOK() {
		t.Fatalf("not a paged generic MoE the batched phase 2 admits: %s", m.ResidentDecline())
	}
	r := mr.r
	embs := make([][]float32, 9)
	for i := range embs {
		embs[i] = m.EmbedResidentForTest(i*31 + 7)
	}
	before, groups := prefillScratchSeen.Load(), r.moeBatchGroups
	r.prefillMoEPaged(embs, 0, true)
	if err := r.takeExecErr(); err != nil {
		t.Fatal(err)
	}
	if r.moeBatchGroups == groups {
		t.Fatal("the batched phase 2 did not run")
	}
	if seen, want := prefillScratchSeen.Load()-before, r.moeBatchBytes(9); seen != want {
		t.Errorf("9 rows: the batch scratch counted %d bytes, want moeBatchBytes(9) = %d (one allocation per call)", seen, want)
	}
	for _, M := range []int{9, 300} {
		on := r.prefillScratchBytes(M)
		moeExpertBatchOn = false
		off := r.prefillScratchBytes(M)
		moeExpertBatchOn = true
		if want := r.moeBatchBytes(min(M, g4LayerMajorChunk)); on-off != want {
			t.Errorf("M=%d: the plan adds %d bytes for the batched phase 2, want %d", M, on-off, want)
		}
	}
}
