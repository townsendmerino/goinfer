//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"errors"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestCUDAStepBatch_matchesForward is MC3-on-CUDA's step-identity gate (docs/tasks/task-concurrency-2026-09.md, "MC3 on
// CUDA", correctness 1). For B = 2, 3 and 4 sequences, each on its own KV slot at its own depth, one StepBatch row must
// equal that sequence's own Forward on its slot bit for bit — or ForwardSample's id for a row that carries a Draw (the
// last row of each step does). Four teacher-forced steps per B.
//
// The reference runs AFTER the step at the same positions: Forward's rope_kv rewrites position pos before its attention
// reads it, so it does not depend on what the step wrote there. On a real checkpoint (GOINFER_CUDA_STEP_MODEL) one
// sequence sits past 2048 keys, so decode's flash-decode lane runs inside the step.
//
// Two fixtures carry Forward's host tail, which the bit comparison of a logits row checks was applied exactly once:
// gemma4-dense-twogeom-tiny has a final-logit softcap and granite-dense-tiny a logit scale. The subtest asserts the
// transform is armed on the loaded model, so a fixture that lost it fails rather than passing on the identity. The Gemma 4
// fixture is generated (scripts/pin_gemma4_dense_twogeom.py), so that subtest skips where it has not been pinned. The
// Cohere fixtures are not here: their batched LayerNorm (layernorm_quant_batched) rounds in a different order from
// decode's, so their step is not bit-identical to Forward for a reason that is not the host tail
// (docs/tasks/task-forward-architecture-2026-10.md §4.5).
func TestCUDAStepBatch_matchesForward(t *testing.T) {
	type cfg struct {
		path   string
		depths []int
		tail   string // "softcap" or "logitScale": the host transform the fixture must arm; "" for none
		kv     string // the KV precision asked ("" for the default)
	}
	cfgs := []cfg{
		{filepath.Join("../../testdata/llama-tiny"), []int{5, 23, 40, 11}, "", ""},
		{filepath.Join("../../testdata/llama-tiny"), []int{5, 23, 40, 11}, "", "i8"},         // the int8 cache: each row's store and scales through its own slot
		{filepath.Join("../../testdata/mistral-tiny-window"), []int{9, 70, 200, 33}, "", ""}, // window: some rows past it
		{filepath.Join("../../testdata/gemma4-dense-twogeom-tiny"), []int{5, 23, 40, 11}, "softcap", ""},
		{filepath.Join("../../testdata/granite-dense-tiny"), []int{5, 23, 40, 11}, "logitScale", ""},
	}
	if p := os.Getenv("GOINFER_CUDA_STEP_MODEL"); p != "" {
		cfgs = []cfg{{p, []int{120, 2200, 700, 64}, "", ""}, {p, []int{120, 2200, 700, 64}, "", "i8"}}
	}
	for _, c := range cfgs {
		t.Run(filepath.Base(c.path)+"/"+c.kv, func(t *testing.T) {
			requireCUDADevice(t)
			if _, err := os.Stat(c.path); err != nil {
				t.Skipf("no fixture at %s", c.path)
			}
			m, err := decoder.Load(c.path, decoder.Options{Backend: "cuda", Quant: "int4", ResidentKVSlots: 4, KVPrecision: c.kv})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer m.Close()
			r, ok := m.ResidentForwardForTest().(*cudaResident)
			if !ok {
				t.Fatalf("not CUDA-resident: %s", m.ResidentDecline())
			}
			if r.kvI8 != (c.kv == "i8") {
				t.Fatalf("-kv %q built an int8 cache: %v", c.kv, r.kvI8)
			}
			if lo, hi := r.BatchStepRange(); lo != 2 || hi != 4 {
				t.Fatalf("BatchStepRange = (%d, %d), want (2, 4): the step does not engage here", lo, hi)
			}
			switch c.tail {
			case "softcap":
				if r.finalSoftcap <= 0 {
					t.Fatalf("finalSoftcap = %v: the fixture no longer arms the final-logit softcap this case exists for", r.finalSoftcap)
				}
			case "logitScale":
				if r.logitScale == 0 || r.logitScale == 1 {
					t.Fatalf("logitScale = %v: the fixture no longer arms the logit scale this case exists for", r.logitScale)
				}
			}
			_, _, _, _, _, _, vocab := m.Dims()
			emb := func(id int) []float32 { return m.EmbedResidentForTest(id % vocab) }
			for _, B := range []int{2, 3, 4} {
				// Prefill each sequence into its own slot.
				ids := make([]int, B)
				pos := make([]int, B)
				for b := range B {
					if err := r.UseKVSlot(b); err != nil {
						t.Fatal(err)
					}
					d := c.depths[b]
					prompt := make([][]float32, d)
					for i := range prompt {
						prompt[i] = emb(i*37 + b*11 + 3)
					}
					if _, err := r.PrefillLast(context.Background(), prompt, 0); err != nil {
						t.Fatalf("prefill slot %d: %v", b, err)
					}
					ids[b], pos[b] = (b*53+7)%vocab, d
				}
				for step := range 4 {
					seqs := make([]decoder.ResidentBatchSeq, B)
					for b := range B {
						seqs[b] = decoder.ResidentBatchSeq{Slot: b, Pos: pos[b], Emb: emb(ids[b])}
					}
					draw := &decoder.ResidentBatchDraw{Temperature: 0.8, Seed: uint64(1000 + step), Draw: uint64(step)}
					seqs[B-1].Draw = draw
					out, err := r.StepBatch(seqs)
					if err != nil {
						t.Fatalf("B=%d step %d: %v", B, step, err)
					}
					for b := range B {
						if err := r.UseKVSlot(b); err != nil {
							t.Fatal(err)
						}
						if seqs[b].Draw != nil {
							want, err := r.ForwardSample(seqs[b].Emb, pos[b], draw.Temperature, draw.Seed, draw.Draw)
							if err != nil {
								t.Fatal(err)
							}
							if out[b].ID != want {
								t.Fatalf("B=%d step %d seq %d: step drew %d, ForwardSample %d", B, step, b, out[b].ID, want)
							}
							ids[b] = want
						} else {
							want, err := r.Forward(seqs[b].Emb, pos[b])
							if err != nil {
								t.Fatal(err)
							}
							got := out[b].Logits
							if len(got) != len(want) {
								t.Fatalf("B=%d step %d seq %d: %d logits, Forward %d", B, step, b, len(got), len(want))
							}
							for j := range want {
								if math.Float32bits(got[j]) != math.Float32bits(want[j]) {
									t.Fatalf("B=%d step %d seq %d (slot %d, pos %d): logit %d = %v in the step, %v from Forward — not bit-identical",
										B, step, b, b, pos[b], j, got[j], want[j])
								}
							}
							ids[b] = argmaxF32(want)
						}
						pos[b]++
					}
				}
				t.Logf("B=%d: 4 steps bit-identical to each sequence's own Forward / ForwardSample (depths %v)", B, c.depths[:B])
			}
		})
	}
}

// TestCUDAStepBatch_concurrentMatchesAlone is MC3-on-CUDA's production-path gate (docs/tasks/task-concurrency-2026-09.md,
// "MC3 on CUDA", correctness 2). 4 conversations run through Model.Generate at once on a CUDA model with
// EnableResidentConcurrency(4): the decoder's MC3 batcher joins their decode tokens into CUDA StepBatch steps, each on its
// own KV slot. Every turn's ids must equal what the same conversation emits alone on a model where concurrency is never
// enabled — greedy, and sampled with a fixed seed per conversation (on-device draws). Conversation 0 opens with a long
// prompt, so chunked prefill (ResidentPrefillChunk) runs between steps while the others decode. A control reads
// ResidentBatchStats: steps of at least two sequences must have run.
//
// GOINFER_CUDA_STEP_MODEL=<checkpoint> runs it on a real model (int4, 24 tokens per turn) instead of llama-tiny.
func TestCUDAStepBatch_concurrentMatchesAlone(t *testing.T) {
	requireCUDADevice(t)
	path, maxTok, longLen := filepath.Join("../../testdata/llama-tiny"), 8, 90
	if p := os.Getenv("GOINFER_CUDA_STEP_MODEL"); p != "" {
		path, maxTok, longLen = p, 24, 700
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	const nConv, turns = 4, 3
	opts := decoder.Options{Backend: "cuda", Quant: "int4", ResidentKVSlots: 4, ResidentPrefillChunk: 32}
	if os.Getenv("GOINFER_CUDA_STEP_MODEL") != "" {
		opts.ResidentPrefillChunk = 512
	}
	for _, sampled := range []bool{false, true} {
		name := map[bool]string{false: "greedy", true: "sampled"}[sampled]
		t.Run(name, func(t *testing.T) {
			sp := func(c, tn int) decoder.SamplingParams {
				if !sampled {
					return decoder.SamplingParams{}
				}
				return decoder.SamplingParams{Temperature: 0.8, Seed: int64(500 + 10*c + tn)}
			}
			play := func(m *decoder.Model, concurrent bool) ([][][]int, [][]int) {
				_, _, _, _, _, _, vocab := m.Dims()
				ids, reused := make([][][]int, nConv), make([][]int, nConv)
				run := func(c int) {
					prompt := []int{1, 2, 3, (c*17 + 5) % vocab, (c*29 + 9) % vocab}
					if c == 0 {
						for i := range longLen {
							prompt = append(prompt, (i*31+7)%vocab)
						}
					}
					for tn := range turns {
						ch, gen := m.Generate(context.Background(), prompt, maxTok, sp(c, tn))
						var out []int
						for id := range ch {
							out = append(out, id)
						}
						if err := gen.Err(); err != nil {
							t.Errorf("conversation %d turn %d: %v", c, tn, err)
							return
						}
						ids[c], reused[c] = append(ids[c], out), append(reused[c], gen.PrefillReused)
						prompt = append(append(append([]int(nil), prompt...), out...), (c*7+tn)%vocab, (c*11+tn)%vocab)
					}
				}
				if !concurrent {
					for c := range nConv {
						run(c)
					}
					return ids, reused
				}
				done := make(chan struct{}, nConv)
				for c := range nConv {
					go func() { run(c); done <- struct{}{} }()
				}
				for range nConv {
					<-done
				}
				return ids, reused
			}
			ref, err := decoder.Load(path, opts)
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			want, wantReused := play(ref, false)
			ref.Close()

			m, err := decoder.Load(path, opts)
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer m.Close()
			if n := m.EnableResidentConcurrency(nConv); n != nConv {
				t.Fatalf("EnableResidentConcurrency(%d) = %d: the CUDA step did not engage", nConv, n)
			}
			got, gotReused := play(m, true)
			for c := range nConv {
				for tn := range turns {
					if tn >= len(got[c]) || !slicesEqual(got[c][tn], want[c][tn]) {
						t.Errorf("conversation %d turn %d: concurrent %v, alone %v", c, tn, at(got[c], tn), want[c][tn])
					} else if gotReused[c][tn] != wantReused[c][tn] {
						t.Errorf("conversation %d turn %d: reused %d concurrent, %d alone", c, tn, gotReused[c][tn], wantReused[c][tn])
					}
				}
			}
			st := m.ResidentBatchStats()
			t.Logf("runs %d, steps %d (tokens %d), solo tokens %d, straggler runs %d, step sizes %v",
				st.Runs, st.Steps, st.StepTokens, st.SoloTokens, st.StragglerRuns, st.StepSizes[:nConv+1])
			if st.Steps == 0 || st.StepTokens < 2*st.Steps {
				t.Errorf("no step of two or more sequences ran (%d steps, %d tokens): the control shows batching did not happen", st.Steps, st.StepTokens)
			}
		})
	}
}

func slicesEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func at(v [][]int, i int) []int {
	if i < len(v) {
		return v[i]
	}
	return nil
}

// TestCUDAStepRagged_rowsMatchTheirShapes is G-RF2-I's batch-invariance gate for a ragged step of decode and verify rows
// (docs/tasks/task-forward-and-memory-2026-11.md, "RF2: registration" and "RF2: addendum 1"): four sequences on their own KV
// slots in one StepRagged, one a single row with an on-device draw, one a single row of logits, one 2 verify rows and one 4,
// for four teacher-forced steps. Every single row must equal its sequence's Forward (or ForwardSample's id) and every
// sequence's rows its ForwardN at the same positions, bit for bit. The references run after the step at the same positions:
// rope_kv rewrites a position before attention reads it. Two plants must each make a row differ: a sequence's second row
// bound to another sequence's slot, and every row of a sequence given its first row's position. On the fixtures every row's decode
// takes attn_batched, so each step must make its attention one attn_ragged launch per layer, and a third plant (every ragged row's
// table read one block entry on) must make a row differ too.
// GOINFER_CUDA_STEP_MODEL=<checkpoint> runs it on a real model, one sequence past 2048 keys (the flash-decode lane), where the
// rows' launches differ and the step takes each row's own.
func TestCUDAStepRagged_rowsMatchTheirShapes(t *testing.T) {
	type cfg struct {
		path   string
		depths []int
		kv     string
	}
	cfgs := []cfg{
		{"../../testdata/llama-tiny", []int{5, 23, 40, 11}, ""},
		{"../../testdata/llama-tiny", []int{5, 23, 40, 11}, "i8"},
		{"../../testdata/mistral-tiny-window", []int{9, 70, 200, 33}, ""},
		{"../../testdata/gemma4-dense-twogeom-tiny", []int{5, 23, 40, 11}, ""},
		{"../../testdata/granite-dense-tiny", []int{5, 23, 40, 11}, ""},
	}
	ragged := true
	if p := os.Getenv("GOINFER_CUDA_STEP_MODEL"); p != "" {
		cfgs = []cfg{{p, []int{120, 2200, 700, 64}, ""}, {p, []int{120, 2200, 700, 64}, "i8"}}
		ragged = false
	}
	nRows := []int{1, 1, 2, 4} // per sequence: a drawn decode row, a decode row, verify k = 2 and 4
	for _, c := range cfgs {
		t.Run(filepath.Base(c.path)+"/"+c.kv, func(t *testing.T) {
			requireCUDADevice(t)
			if _, err := os.Stat(c.path); err != nil {
				t.Skipf("no fixture at %s", c.path)
			}
			m, err := decoder.Load(c.path, decoder.Options{Backend: "cuda", Quant: "int4", ResidentKVSlots: 4, KVPrecision: c.kv})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer m.Close()
			r, ok := m.ResidentForwardForTest().(*cudaResident)
			if !ok {
				t.Fatalf("not CUDA-resident: %s", m.ResidentDecline())
			}
			if _, hi := r.BatchStepRange(); hi != 4 {
				t.Fatalf("BatchStepRange hi = %d, want 4: the step does not engage here", hi)
			}
			_, _, _, _, _, _, vocab := m.Dims()
			emb := func(id int) []float32 { return m.EmbedResidentForTest(id % vocab) }
			pos := make([]int, 4)
			for b := range 4 {
				if err := r.UseKVSlot(b); err != nil {
					t.Fatal(err)
				}
				prompt := make([][]float32, c.depths[b])
				for i := range prompt {
					prompt[i] = emb(i*37 + b*11 + 3)
				}
				if _, err := r.PrefillLast(context.Background(), prompt, 0); err != nil {
					t.Fatalf("prefill slot %d: %v", b, err)
				}
				pos[b] = c.depths[b]
			}
			// step runs one ragged step and its references; it returns how many rows differ from their references.
			step := func(n int) int {
				seqs := make([]decoder.ResidentRaggedSeq, 4)
				for b := range 4 {
					embs := make([][]float32, nRows[b])
					for j := range embs {
						embs[j] = emb(n*101 + b*53 + j*7 + 1)
					}
					seqs[b] = decoder.ResidentRaggedSeq{Slot: b, Pos: pos[b], Embs: embs}
				}
				draw := &decoder.ResidentBatchDraw{Temperature: 0.8, Seed: uint64(1000 + n), Draw: uint64(n)}
				seqs[0].Draw = draw
				out, err := r.StepRagged(seqs)
				if err != nil {
					t.Fatalf("step %d: %v", n, err)
				}
				differ := 0
				for b, s := range seqs {
					if err := r.UseKVSlot(b); err != nil {
						t.Fatal(err)
					}
					var want [][]float32
					switch {
					case s.Draw != nil:
						id, err := r.ForwardSample(s.Embs[0], s.Pos, draw.Temperature, draw.Seed, draw.Draw)
						if err != nil {
							t.Fatal(err)
						}
						if out[b].ID != id {
							differ++
						}
					case len(s.Embs) == 1:
						l, err := r.Forward(s.Embs[0], s.Pos)
						if err != nil {
							t.Fatal(err)
						}
						want = [][]float32{l}
					default:
						if want, err = r.ForwardN(s.Embs, s.Pos); err != nil {
							t.Fatal(err)
						}
					}
					for j := range want {
						if !bitsEqual(out[b].Logits[j], want[j]) {
							differ++
							t.Logf("step %d seq %d (slot %d) row %d at position %d: not bit-identical to its own shape", n, b, b, j, s.Pos+j)
						}
					}
					pos[b] += len(s.Embs)
				}
				return differ
			}
			for n := range 4 {
				launches := r.raggedAttnLaunches
				if d := step(n); d != 0 {
					t.Fatalf("step %d: %d rows differ from their own shapes", n, d)
				}
				if ragged && r.raggedAttnLaunches == launches {
					t.Fatalf("step %d: no attn_ragged launch; the rows took their own attention", n)
				}
			}
			t.Cleanup(func() { raggedPlantForTest = 0 })
			plants := []int{1, 2}
			if ragged {
				plants = append(plants, 4)
			}
			for _, plant := range plants {
				raggedPlantForTest = plant
				d := step(10 + plant)
				raggedPlantForTest = 0
				if d == 0 {
					t.Fatalf("plant %d: every row still matched its shape; the gate cannot see that defect", plant)
				}
				t.Logf("plant %d: %d rows differ (red, as it must be)", plant, d)
			}
		})
	}
}

// TestCUDAStepRagged_prefillChunkMatchesItsPass is RF2-2's gate (docs/tasks/task-forward-and-memory-2026-11.md, "RF2: steps",
// and addendum 1's mixed batch): a 64-row prompt prefilled as two 32-row passes, each inside a ragged step beside two MC3 decode
// rows (one drawn) and verifies at k = 2 and 4, the chunk first in the first step and third in the second (StepRagged reorders
// it), against the same two passes run alone (prefillChunkedRows' chunkOrdinary and
// chunkPromptLen) on a spare slot. The final pass's last-row logits see every K/V row both passes wrote, and must match bit
// for bit, as must every other row its own shape. The fast-prefill floor is 0, so the passes take the fast kernels: the stepped
// passes must launch fast kernels, and under a plant that runs the chunk on the exact kernels none, with the logits then
// differing. The plant is read only where the fused attention runs (head dim 64 or 128): at llama-tiny's head dim 16 the only
// fast kernel is the MMA GEMM, whose int32 group sums can equal the exact GEMV's bit for bit, so there the case is identity only.
func TestCUDAStepRagged_prefillChunkMatchesItsPass(t *testing.T) {
	for _, c := range []struct {
		path, kv string
		plant    bool
	}{
		{"../../testdata/llama-attnfa-tiny", "", true},
		{"../../testdata/llama-attnfa-tiny", "i8", true},
		{"../../testdata/llama-tiny", "", false},
	} {
		t.Run(filepath.Base(c.path)+"/"+c.kv, func(t *testing.T) {
			requireCUDADevice(t)
			if _, err := os.Stat(c.path); err != nil {
				t.Skipf("no fixture at %s", c.path)
			}
			m, err := decoder.Load(c.path, decoder.Options{Backend: "cuda", Quant: "int4", ResidentKVSlots: 6, KVPrecision: c.kv})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer m.Close()
			decoder.SetKnobEnvForTest(t, m, "GOINFER_CUDA_FAST_PREFILL_FLOOR", "0")
			r, ok := m.ResidentForwardForTest().(*cudaResident)
			if !ok {
				t.Fatalf("not CUDA-resident: %s", m.ResidentDecline())
			}
			_, _, _, _, _, _, vocab := m.Dims()
			emb := func(id int) []float32 { return m.EmbedResidentForTest(id % vocab) }
			const passRows, promptLen, refSlot = 32, 64, 5
			prompt := make([][]float32, promptLen)
			for i := range prompt {
				prompt[i] = emb(i*29 + 5)
			}
			// ownPasses runs the prompt's two passes alone on refSlot and returns the last row's logits and the fused attention
			// launches made.
			ownPasses := func() ([]float32, int) {
				if err := r.UseKVSlot(refSlot); err != nil {
					t.Fatal(err)
				}
				fast := r.fastAttnLaunches
				r.chunkOrdinary, r.chunkPromptLen = true, promptLen
				defer func() { r.chunkOrdinary, r.chunkPromptLen = false, 0 }()
				if _, _, err := r.prefillCore(context.Background(), prompt[:passRows], 0, tailKVOnly, nil, nil, nil); err != nil {
					t.Fatalf("own pass 1: %v", err)
				}
				outs, _, err := r.prefillCore(context.Background(), prompt[passRows:], passRows, tailLastLogits, nil, nil, nil)
				if err != nil {
					t.Fatalf("own pass 2: %v", err)
				}
				return outs[len(outs)-1], r.fastAttnLaunches - fast
			}
			want, fast := ownPasses()
			if c.plant && fast == 0 {
				t.Fatalf("the prompt's own passes launched no fused attention: the chunk's class is not visible here")
			}
			depths := []int{0, 7, 23, 11, 17}
			ks := []int{0, 1, 1, 2, 4} // rows per slot: slot 0 is the prompt's; a drawn decode row, a decode row, verify k = 2 and 4
			for b := 1; b < len(depths); b++ {
				if err := r.UseKVSlot(b); err != nil {
					t.Fatal(err)
				}
				pre := make([][]float32, depths[b])
				for i := range pre {
					pre[i] = emb(i*37 + b*11 + 3)
				}
				if _, err := r.PrefillLast(context.Background(), pre, 0); err != nil {
					t.Fatalf("prefill slot %d: %v", b, err)
				}
			}
			// run prefills the prompt through two steps and returns its last-row logits, how many other rows differ, and the fast
			// launches (fused attention and MMA GEMM) the steps made.
			run := func(n int) ([]float32, int, int) {
				fast0 := r.fastAttnLaunches + r.fastGemmLaunches
				var fast int
				pos := append([]int(nil), depths...)
				var last []float32
				differ := 0
				for pass := range 2 {
					var seqs []decoder.ResidentRaggedSeq
					for b := 1; b < len(ks); b++ {
						embs := make([][]float32, ks[b])
						for j := range embs {
							embs[j] = emb(n*101 + pass*71 + b*53 + j*7 + 1)
						}
						seqs = append(seqs, decoder.ResidentRaggedSeq{Slot: b, Pos: pos[b], Embs: embs})
					}
					draw := &decoder.ResidentBatchDraw{Temperature: 0.8, Seed: uint64(2000 + n), Draw: uint64(pass)}
					seqs[0].Draw = draw
					ci := 2 * pass // the chunk's index: first, then third
					seqs = slices.Insert(seqs, ci, decoder.ResidentRaggedSeq{Slot: 0, Pos: pass * passRows, Embs: prompt[pass*passRows : (pass+1)*passRows],
						Prefill: &decoder.ResidentRaggedPrefill{PromptLen: promptLen, Last: pass == 1}})
					out, err := r.StepRagged(seqs)
					if err != nil {
						t.Fatalf("step %d pass %d: %v", n, pass, err)
					}
					fast = r.fastAttnLaunches + r.fastGemmLaunches - fast0 // the references below are decode and verify: exact
					for j, l := range out[ci].Logits {
						if (l != nil) != (pass == 1 && j == passRows-1) {
							t.Fatalf("step %d pass %d: the chunk's row %d has logits %v; only the final pass's last row has them", n, pass, j, l != nil)
						}
					}
					if pass == 1 {
						last = out[ci].Logits[passRows-1]
					}
					for b, s := range seqs {
						if b == ci {
							continue
						}
						if err := r.UseKVSlot(s.Slot); err != nil {
							t.Fatal(err)
						}
						switch {
						case s.Draw != nil:
							id, err := r.ForwardSample(s.Embs[0], s.Pos, draw.Temperature, draw.Seed, draw.Draw)
							if err != nil {
								t.Fatal(err)
							}
							if out[b].ID != id {
								differ++
							}
						default:
							var ref [][]float32
							if len(s.Embs) == 1 {
								l, err := r.Forward(s.Embs[0], s.Pos)
								if err != nil {
									t.Fatal(err)
								}
								ref = [][]float32{l}
							} else if ref, err = r.ForwardN(s.Embs, s.Pos); err != nil {
								t.Fatal(err)
							}
							for j := range ref {
								if !bitsEqual(out[b].Logits[j], ref[j]) {
									differ++
									t.Logf("step %d pass %d seq %d row %d at position %d: not bit-identical to its own shape", n, pass, b, j, s.Pos+j)
								}
							}
						}
						pos[s.Slot] += len(s.Embs)
					}
				}
				for b := 1; b < len(depths); b++ {
					depths[b] = pos[b]
				}
				return last, differ, fast
			}
			got, differ, stepFast := run(0)
			if differ != 0 {
				t.Fatalf("%d decode and verify rows beside the chunk differ from their own shapes", differ)
			}
			if !bitsEqual(got, want) {
				t.Fatalf("the prompt prefilled in two steps is not bit-identical to its own two passes")
			}
			if stepFast == 0 {
				t.Fatalf("the stepped passes launched no fast kernel: the chunk did not take its class")
			}
			if !c.plant {
				t.Logf("identity only: the stepped passes made %d fast launches; no fused attention at this head dim", stepFast)
				return
			}
			t.Cleanup(func() { raggedPlantForTest = 0 })
			raggedPlantForTest = 3
			got, _, plantFast := run(1)
			raggedPlantForTest = 0
			if plantFast != 0 {
				t.Fatalf("plant 3: the chunk on the exact kernels still made %d fast launches; the plant does not reach them", plantFast)
			}
			if bitsEqual(got, want) {
				t.Fatalf("plant 3: the chunk on the exact kernels still matched its fast passes; the gate cannot see the class")
			}
			t.Logf("plant 3: the chunk out of its class differs (red, as it must be); fused attention launches: own passes %d", fast)
		})
	}
}

func bitsEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return false
		}
	}
	return true
}

// TestRaggedOrder_checksAndOrder holds StepRagged's checks on the sequences alone, which need no device: an empty step is an error,
// a step takes at most one prefill chunk and no draw on it, a chunk's rows end inside its prompt and end it exactly when (and only
// when) the chunk is the prompt's Last pass, and the chunk goes first wherever it stands, the others keeping their order.
func TestRaggedOrder_checksAndOrder(t *testing.T) {
	if _, err := (&cudaResident{}).StepRagged(nil); err == nil {
		t.Error("an empty step returned no error")
	}
	rows := func(n int) [][]float32 { return make([][]float32, n) }
	chunk := func(pos, n, promptLen int, last bool) decoder.ResidentRaggedSeq {
		return decoder.ResidentRaggedSeq{Slot: 9, Pos: pos, Embs: rows(n), Prefill: &decoder.ResidentRaggedPrefill{PromptLen: promptLen, Last: last}}
	}
	dec := func(slot int) decoder.ResidentRaggedSeq {
		return decoder.ResidentRaggedSeq{Slot: slot, Pos: 5, Embs: rows(1)}
	}
	drawn := chunk(0, 32, 64, false)
	drawn.Draw = &decoder.ResidentBatchDraw{Temperature: 1}
	for _, c := range []struct {
		name string
		seqs []decoder.ResidentRaggedSeq
	}{
		{"two chunks", []decoder.ResidentRaggedSeq{chunk(0, 32, 64, false), dec(1), chunk(32, 32, 64, true)}},
		{"a chunk with a draw", []decoder.ResidentRaggedSeq{drawn, dec(1)}},
		{"a chunk past its prompt", []decoder.ResidentRaggedSeq{chunk(48, 32, 64, true), dec(1)}},
		{"a Last chunk short of its prompt", []decoder.ResidentRaggedSeq{chunk(0, 32, 64, true), dec(1)}},
		{"a chunk that ends its prompt, not Last", []decoder.ResidentRaggedSeq{chunk(32, 32, 64, false), dec(1)}},
	} {
		if _, err := raggedOrder(c.seqs); err == nil {
			t.Errorf("%s: no error", c.name)
		}
	}
	for _, c := range []struct {
		name string
		seqs []decoder.ResidentRaggedSeq
		want []int
	}{
		{"no chunk", []decoder.ResidentRaggedSeq{dec(0), dec(1), dec(2)}, []int{0, 1, 2}},
		{"a chunk first", []decoder.ResidentRaggedSeq{chunk(0, 32, 64, false), dec(1), dec(2)}, []int{0, 1, 2}},
		{"a chunk third", []decoder.ResidentRaggedSeq{dec(0), dec(1), chunk(32, 32, 64, true), dec(3)}, []int{2, 0, 1, 3}},
	} {
		got, err := raggedOrder(c.seqs)
		if err != nil || !slices.Equal(got, c.want) {
			t.Errorf("%s: order %v (%v), want %v", c.name, got, err, c.want)
		}
	}
}

// TestCUDAStepRagged_prefillChunkDeclines holds raggedChunkDecline through StepRagged: a prefill chunk on a recurrent model, a
// routed-expert model and a windowed-KV build is refused with errPrefillDeclined, where the model offers the step at all (a model
// that does not is refused earlier, by the step range, and the test says which).
func TestCUDAStepRagged_prefillChunkDeclines(t *testing.T) {
	step := func(t *testing.T, r *cudaResident, m *decoder.Model) error {
		_, _, _, _, _, _, vocab := m.Dims()
		emb := func(id int) []float32 { return m.EmbedResidentForTest(id % vocab) }
		chunk := make([][]float32, 16)
		for i := range chunk {
			chunk[i] = emb(i + 3)
		}
		_, err := r.StepRagged([]decoder.ResidentRaggedSeq{
			{Slot: 1, Pos: 0, Embs: [][]float32{emb(7)}},
			{Slot: 0, Pos: 0, Embs: chunk, Prefill: &decoder.ResidentRaggedPrefill{PromptLen: 16, Last: true}},
		})
		return err
	}
	check := func(t *testing.T, r *cudaResident, err error) {
		if _, hi := r.BatchStepRange(); hi < 2 {
			if err == nil {
				t.Fatal("a model that offers no step ran one")
			}
			t.Logf("the model offers no step; refused by the step range: %v", err)
			return
		}
		if !errors.Is(err, errPrefillDeclined) {
			t.Fatalf("a prefill chunk was not declined: %v", err)
		}
	}
	for _, fx := range []string{"qwen35-tiny", "qwen3moe-tiny"} {
		t.Run(fx, func(t *testing.T) {
			requireCUDADevice(t)
			m, err := decoder.Load(filepath.Join("../../testdata", fx), decoder.Options{Backend: "cuda", Quant: "int4", ResidentKVSlots: 2})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer m.Close()
			r, ok := m.ResidentForwardForTest().(*cudaResident)
			if !ok {
				t.Fatalf("not CUDA-resident: %s", m.ResidentDecline())
			}
			check(t, r, step(t, r, m))
		})
	}
	t.Run("mistral-tiny-window/windowed", func(t *testing.T) {
		_, win, _, mw := loadWindowedPair(t, "mistral-tiny-window", 2)
		check(t, win, step(t, win, mw))
	})
}
