package decoder

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"
)

// assertBatchedStepBitIdentical steps len(depths) sequences together through decodeMultiStep for steps tokens,
// teacher-forced, and requires every row's logits to be bit-identical to the production single-token forward over a
// copy of the same cache (MC3c step 2's identity contract; MC2's gate before it).
func assertBatchedStepBitIdentical(t *testing.T, m *Model, depths []int, steps int) {
	t.Helper()
	vocab := m.w.arch.VocabSize
	ctx := context.Background()
	multi, ref := make([]*KVCache, len(depths)), make([]*KVCache, len(depths))
	ids := make([]int, len(depths))
	for b, d := range depths {
		prompt := make([]int, d)
		for i := range prompt {
			prompt[i] = (i*37 + b*11 + 3) % vocab
		}
		for _, cp := range []**KVCache{&multi[b], &ref[b]} {
			*cp = m.NewCache(d + steps + 8)
			if _, err := m.prefillLogits(ctx, prompt, *cp); err != nil {
				t.Fatalf("prefill: %v", err)
			}
		}
		ids[b] = (b*53 + 7) % vocab
	}
	for step := range steps {
		got, err := m.decodeMultiStep(ids, multi)
		if err != nil {
			t.Fatalf("step %d: %v", step, err)
		}
		for b := range ids {
			want, err := m.forward(ids[b], ref[b])
			if err != nil {
				t.Fatalf("step %d seq %d: forward: %v", step, b, err)
			}
			for j := range want {
				if math.Float32bits(got[b][j]) != math.Float32bits(want[j]) {
					t.Fatalf("step %d seq %d (pos %d): logit %d = %v batched vs %v alone — not bit-identical",
						step, b, ref[b].Pos()-1, j, got[b][j], want[j])
				}
			}
			ids[b] = argmaxF32(want) // teacher-forced: both arms take the same next id
		}
	}
}

// TestCPUBatch_everyEligibleFixtureBitIdentical sweeps every committed fixture under testdata/: each one whose family
// cpuBatchModelEligible admits, with caches cpuBatchCacheEligible admits, must step bit-identically to the production
// forward. The eligibility predicate was written against the generic forward's features, and this is what makes that
// claim checked rather than asserted: a family whose forward does something decodeMultiStep does not fails here, and a
// family the predicate wrongly admits cannot pass silently. It also requires the sweep to have covered at least a few
// families, so a predicate that admits nothing is visible.
func TestCPUBatch_everyEligibleFixtureBitIdentical(t *testing.T) {
	dirs, err := filepath.Glob("../testdata/*/config.json")
	if err != nil || len(dirs) == 0 {
		t.Skipf("no fixtures: %v", err)
	}
	var admitted, declined []string
	for _, cfgPath := range dirs {
		dir := filepath.Dir(cfgPath)
		for _, quant := range []string{"", "int8int8"} {
			m, err := Load(dir, Options{Quant: quant})
			if err != nil {
				continue // not a text model this loader serves standalone (vision towers, drafters, ...)
			}
			if err := m.cpuBatchModelEligible(); err != nil {
				declined = append(declined, filepath.Base(dir))
				break
			}
			if c := m.NewCache(64); cpuBatchCacheEligible(c) != nil {
				declined = append(declined, filepath.Base(dir)+" (cache)")
				break
			}
			name := filepath.Base(dir) + "/" + quant
			t.Run(name, func(t *testing.T) { assertBatchedStepBitIdentical(t, m, []int{3, 17, 9}, 8) })
			admitted = append(admitted, name)
		}
	}
	t.Logf("admitted and bit-identical: %v", admitted)
	t.Logf("declined: %v", declined)
	if len(admitted) < 3 {
		t.Errorf("only %d fixture/quant cells were eligible — the sweep covers too little to vouch for the predicate", len(admitted))
	}
}

// cpuBatchConversations plays nConv conversations of turns turns each on their own sessions of m, all at once when
// concurrent is true (each conversation's turns in order), else one conversation after another; it returns every
// turn's ids and PrefillReused.
func cpuBatchConversations(t *testing.T, m *Model, nConv, turns, maxTok int, sp func(c, turn int) SamplingParams, concurrent bool) ([][][]int, [][]int) {
	t.Helper()
	vocab := m.w.arch.VocabSize
	ids, reused := make([][][]int, nConv), make([][]int, nConv)
	run := func(c int) {
		s := m.NewSession(512)
		prompt := []int{1, 2, 3, (c*17 + 5) % vocab, (c*29 + 9) % vocab}
		for tn := range turns {
			ch, gen := s.Generate(context.Background(), prompt, maxTok, sp(c, tn))
			var out []int
			for id := range ch {
				out = append(out, id)
			}
			if err := gen.Err(); err != nil {
				t.Errorf("conversation %d turn %d: %v", c, tn, err)
				return
			}
			ids[c], reused[c] = append(ids[c], out), append(reused[c], gen.PrefillReused)
			prompt = append(append(slices.Clone(prompt), out...), (c*7+tn)%vocab, (c*11+tn)%vocab)
		}
	}
	if !concurrent {
		for c := range nConv {
			run(c)
		}
		return ids, reused
	}
	var wg sync.WaitGroup
	for c := range nConv {
		wg.Go(func() { ; run(c) })
	}
	wg.Wait()
	return ids, reused
}

// TestCPUBatch_concurrentMatchesAlone is MC3c step 2's correctness gate through the production path: 4 conversations
// decoding at once on one model, their tokens joined into batched steps, each emit exactly the ids and reuse they do
// alone on a model with batching off — greedy, and sampled with a fixed seed per conversation (each sampler draws from
// the same logits). A control reads the batcher's stats: steps of at least two sequences must have run, or the test
// could pass on solo tokens alone.
//
// GOINFER_CPUBATCH_MODEL=<checkpoint> runs it on a real model at int4 (GOINFER_CPUBATCH_MODE=auto tests the default
// policy, which must then engage on its own).
func TestCPUBatch_concurrentMatchesAlone(t *testing.T) {
	path, quant, mode, maxTok := "../testdata/llama-tiny", "", CPUBatchOn, 10
	if p := os.Getenv("GOINFER_CPUBATCH_MODEL"); p != "" {
		path, quant, maxTok = p, "int4", 24
		if os.Getenv("GOINFER_CPUBATCH_MODE") == "auto" {
			mode = CPUBatchAuto
		}
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s: %v", path, err)
	}
	const nConv, turns = 4, 3
	for _, c := range []struct {
		name string
		sp   func(c, turn int) SamplingParams
	}{
		{"greedy", func(int, int) SamplingParams { return SamplingParams{} }},
		{"sampled", func(c, tn int) SamplingParams {
			return SamplingParams{Temperature: 0.8, Seed: int64(1000 + 10*c + tn)}
		}},
	} {
		t.Run(c.name, func(t *testing.T) {
			ref, err := Load(path, Options{Quant: quant, CPUBatchDecode: CPUBatchOff})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			if ref.resident != nil {
				t.Skip("MC3c covers the CPU path; this model went GPU-resident")
			}
			wantIDs, wantReused := cpuBatchConversations(t, ref, nConv, turns, maxTok, c.sp, false)
			ref.Close()

			m, err := Load(path, Options{Quant: quant, CPUBatchDecode: mode})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer m.Close()
			if !m.EnableCPUBatch(nConv) {
				t.Fatalf("EnableCPUBatch(%d) declined (mode %d, %d dense bytes, eligible: %v)", nConv, mode,
					m.ResidentDenseWeightBytes(), m.cpuBatchModelEligible())
			}
			gotIDs, gotReused := cpuBatchConversations(t, m, nConv, turns, maxTok, c.sp, true)
			for cv := range nConv {
				for tn := range turns {
					if tn >= len(gotIDs[cv]) || !slices.Equal(gotIDs[cv][tn], wantIDs[cv][tn]) {
						t.Errorf("conversation %d turn %d: batched %v, alone %v", cv, tn, at(gotIDs[cv], tn), wantIDs[cv][tn])
					} else if gotReused[cv][tn] != wantReused[cv][tn] {
						t.Errorf("conversation %d turn %d: reused %d batched, %d alone", cv, tn, gotReused[cv][tn], wantReused[cv][tn])
					}
				}
			}
			st := m.CPUBatchStats()
			t.Logf("runs %d, steps %d (tokens %d), solo tokens %d, straggler runs %d, step sizes %v",
				st.Runs, st.Steps, st.StepTokens, st.SoloTokens, st.StragglerRuns, st.StepSizes[:nConv+1])
			if st.Steps == 0 || st.StepTokens < 2*st.Steps {
				t.Errorf("no batched step of two or more sequences ran (%d steps, %d tokens) — the control shows the "+
					"test did not exercise batching", st.Steps, st.StepTokens)
			}
		})
	}
}

func at(v [][]int, i int) []int {
	if i < len(v) {
		return v[i]
	}
	return nil
}

// TestCPUBatch_ineligibleCachesBypass: a model whose family batches but whose caches do not (int8 KV) runs its
// concurrent generations as step 1's workers — nothing is submitted to the batcher — and still matches alone.
func TestCPUBatch_ineligibleCachesBypass(t *testing.T) {
	path := "../testdata/llama-tiny"
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture: %v", err)
	}
	greedy := func(int, int) SamplingParams { return SamplingParams{} }
	ref, err := Load(path, Options{KVQuant: "i8", CPUBatchDecode: CPUBatchOff})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	want, _ := cpuBatchConversations(t, ref, 3, 2, 8, greedy, false)
	m, err := Load(path, Options{KVQuant: "i8", CPUBatchDecode: CPUBatchOn})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if cpuBatchCacheEligible(m.NewCache(16)) == nil {
		t.Fatal("an int8-KV cache is eligible — this test no longer covers a declined cache")
	}
	if !m.EnableCPUBatch(3) {
		t.Fatal("EnableCPUBatch declined a model whose family is eligible")
	}
	got, _ := cpuBatchConversations(t, m, 3, 2, 8, greedy, true)
	for c := range want {
		for tn := range want[c] {
			if !slices.Equal(at(got[c], tn), want[c][tn]) {
				t.Errorf("conversation %d turn %d: %v concurrent, %v alone", c, tn, at(got[c], tn), want[c][tn])
			}
		}
	}
	if st := m.CPUBatchStats(); st.Runs != 0 {
		t.Errorf("int8-KV generations were submitted to the batcher (%d runs)", st.Runs)
	}
}

// TestEnableCPUBatch_policy pins Options.CPUBatchDecode: auto engages only at cpuBatchAutoMinBytes, on engages any
// eligible family, off never engages, n < 2 never engages, and an ineligible family never does.
func TestEnableCPUBatch_policy(t *testing.T) {
	tiny := "../testdata/llama-tiny"
	if _, err := os.Stat(tiny); err != nil {
		t.Skipf("no fixture: %v", err)
	}
	load := func(dir string, mode int) *Model {
		m, err := Load(dir, Options{CPUBatchDecode: mode})
		if err != nil {
			t.Fatalf("load %s: %v", dir, err)
		}
		return m
	}
	for _, c := range []struct {
		name string
		mode int
		n    int
		want bool
	}{
		{"auto: a tiny model stays on the workers", CPUBatchAuto, 4, false},
		{"on: batches", CPUBatchOn, 4, true},
		{"off: never", CPUBatchOff, 4, false},
		{"on, but one generation: nothing to batch", CPUBatchOn, 1, false},
	} {
		if got := load(tiny, c.mode).EnableCPUBatch(c.n); got != c.want {
			t.Errorf("%s: EnableCPUBatch = %v, want %v", c.name, got, c.want)
		}
	}
	if moe := "../testdata/mixtral-tiny"; func() bool { _, err := os.Stat(moe); return err == nil }() {
		if load(moe, CPUBatchOn).EnableCPUBatch(4) {
			t.Error("on: a MoE family engaged batching, which decodeMultiStep does not implement")
		}
	}
}
