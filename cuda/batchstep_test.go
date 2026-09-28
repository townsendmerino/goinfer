//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"math"
	"os"
	"path/filepath"
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
func TestCUDAStepBatch_matchesForward(t *testing.T) {
	type cfg struct {
		path   string
		depths []int
	}
	cfgs := []cfg{
		{filepath.Join("..", "testdata", "llama-tiny"), []int{5, 23, 40, 11}},
		{filepath.Join("..", "testdata", "mistral-tiny-window"), []int{9, 70, 200, 33}}, // window: some rows past it
	}
	if p := os.Getenv("GOINFER_CUDA_STEP_MODEL"); p != "" {
		cfgs = []cfg{{p, []int{120, 2200, 700, 64}}}
	}
	for _, c := range cfgs {
		t.Run(filepath.Base(c.path), func(t *testing.T) {
			requireCUDADevice(t)
			if _, err := os.Stat(c.path); err != nil {
				t.Skipf("no fixture at %s", c.path)
			}
			m, err := decoder.Load(c.path, decoder.Options{Backend: "cuda", Quant: "int4", ResidentKVSlots: 4})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer m.Close()
			r, ok := m.ResidentForwardForTest().(*cudaResident)
			if !ok {
				t.Fatalf("not CUDA-resident: %s", m.ResidentDecline())
			}
			if lo, hi := r.BatchStepRange(); lo != 2 || hi != 4 {
				t.Fatalf("BatchStepRange = (%d, %d), want (2, 4): the step does not engage here", lo, hi)
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
	path, maxTok, longLen := filepath.Join("..", "testdata", "llama-tiny"), 8, 90
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
