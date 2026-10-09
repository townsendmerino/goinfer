package decoder

import (
	"context"
	"slices"
	"sync"
	"testing"
)

// The second round of option-grid cells (optiongrid.go), on the same pattern as
// optiongrid_paths_test.go: the option set through Options, the committed llama-tiny loaded with
// Load, the path run through the Session entry point, and K/V compared as well as tokens.

// TestOptionPath_cpuBatchedDecode fills the "CPU batched decode" column (MC3c): with
// CPUBatchDecode on, concurrent generations whose decode tokens are joined into batched steps must
// each emit the tokens and leave the K/V they do alone on a model with batching off. A control
// reads the batcher's stats, so the test cannot pass on solo tokens. TestCPUBatch_concurrentMatchesAlone
// is the same contract at f32; this runs it under each option. (KVQuant is not here: an int8 cache is
// declined by cpuBatchCacheEligible, TestCPUBatch_ineligibleCachesBypass.)
func TestOptionPath_cpuBatchedDecode(t *testing.T) {
	cases := []ogCase{
		{name: "Quant int8", opts: Options{Quant: "int8"}, effect: ogDiffersFromPlain},
		{name: "Quant int8int8", opts: Options{Quant: "int8int8"}, effect: ogDiffersFromPlain},
		{name: "Quant int4", opts: Options{Quant: "int4"}, effect: ogDiffersFromPlain},
		{name: "ActQuantGroup 32 (int8int8)", opts: Options{Quant: "int8int8", ActQuantGroup: 32}, effect: func(t *testing.T, m *Model) {
			if m.actGroup != 32 {
				t.Fatalf("actGroup %d, want 32", m.actGroup)
			}
		}},
		{name: "EmbedInt4 (int4)", opts: Options{Quant: "int4", EmbedInt4: true}, effect: ogDiffersFromPlain},
	}
	const nConv, maxTok = 4, 32
	prompts := make([][]int, nConv)
	for c := range prompts {
		prompts[c] = []int{1, 2, 3, (c*17 + 5) % 256, (c*29 + 9) % 256}
	}
	// run generates every conversation on m, concurrently or one after another, and returns each
	// session's emitted tokens and final K/V. Concurrent runs start together behind a barrier: the
	// batcher joins only generations decoding at the same moment, and on a model this small one can
	// otherwise finish before the next has started (it did on CI's linux-arm64 runner).
	run := func(t *testing.T, m *Model, concurrent bool) ([][]int, [][]float32) {
		toks, kv := make([][]int, nConv), make([][]float32, nConv)
		one := func(c int) {
			s := m.NewSession(64)
			ch, g := s.Generate(context.Background(), prompts[c], maxTok, SamplingParams{})
			for id := range ch {
				toks[c] = append(toks[c], id)
			}
			if err := g.Err(); err != nil {
				t.Errorf("conversation %d: %v", c, err)
			}
			kv[c] = ogKV(s.cache, s.cache.Pos())
		}
		if !concurrent {
			for c := range nConv {
				one(c)
			}
			return toks, kv
		}
		var wg sync.WaitGroup
		start := make(chan struct{})
		for c := range nConv {
			wg.Go(func() { ; <-start; one(c) })
		}
		close(start)
		wg.Wait()
		return toks, kv
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			offOpts := c.opts
			offOpts.CPUBatchDecode = CPUBatchOff
			ref := ogLoad(t, ogCase{opts: offOpts, effect: c.effect})
			wantToks, wantKV := run(t, ref, false)

			onOpts := c.opts
			onOpts.CPUBatchDecode = CPUBatchOn
			m := ogLoad(t, ogCase{opts: onOpts})
			if !m.EnableCPUBatch(nConv) {
				t.Fatalf("EnableCPUBatch declined: %v", m.cpuBatchModelEligible())
			}
			// Every attempt is checked for correctness; attempts repeat only until the control shows a
			// batched step ran, since whether the generations overlap is up to the scheduler.
			const attempts = 5
			for a := 1; ; a++ {
				gotToks, gotKV := run(t, m, true)
				for cv := range nConv {
					if !slices.Equal(gotToks[cv], wantToks[cv]) {
						t.Errorf("attempt %d conversation %d: batched %v, alone %v", a, cv, gotToks[cv], wantToks[cv])
					}
					if d := ogMaxDiff(gotKV[cv], wantKV[cv]); d != 0 {
						t.Errorf("attempt %d conversation %d: batched K/V differs from alone by %g", a, cv, d)
					}
				}
				st := m.CPUBatchStats()
				if st.Steps > 0 && st.StepTokens >= 2*st.Steps {
					break
				}
				if a == attempts {
					t.Fatalf("no batched step of two or more sequences ran in %d attempts (%d steps, %d tokens)",
						attempts, st.Steps, st.StepTokens)
				}
			}
		})
	}
}

// TestOptionPath_sessionSnapshot fills the "session reuse and snapshot" column for Quant and
// CPUBatchDecode: a session snapshotted after its first turn, restored with LoadSession and
// continued, must emit the tokens and hold the K/V of a session that ran both turns without a
// snapshot, and the restored session must reuse its whole stored prefix rather than prefill cold.
func TestOptionPath_sessionSnapshot(t *testing.T) {
	cases := []ogCase{
		{name: "Quant int8", opts: Options{Quant: "int8"}, effect: ogDiffersFromPlain},
		{name: "Quant int8int8", opts: Options{Quant: "int8int8"}, effect: ogDiffersFromPlain},
		{name: "Quant int4", opts: Options{Quant: "int4"}, effect: ogDiffersFromPlain},
		{name: "CPUBatchDecode on", opts: Options{CPUBatchDecode: CPUBatchOn}, effect: func(t *testing.T, m *Model) {
			if !m.EnableCPUBatch(2) || !m.CPUBatchActive() {
				t.Fatal("CPUBatchDecode on did not enable the CPU batch")
			}
		}},
	}
	ctx := context.Background()
	greedy := SamplingParams{}
	prompt := ogPrompt()
	const n = 10
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			m := ogLoad(t, c)

			// Uninterrupted: both turns on one session.
			whole := m.NewSession(0)
			gen(t, func() (<-chan int, *Generation) { return whole.Generate(ctx, prompt, n, greedy) })
			turn2 := append(append([]int(nil), whole.Tokens()...), 7, 11, 13)
			want := gen(t, func() (<-chan int, *Generation) { return whole.Generate(ctx, turn2, n, greedy) })

			// Snapshotted after the first turn, restored, continued.
			first := m.NewSession(0)
			gen(t, func() (<-chan int, *Generation) { return first.Generate(ctx, prompt, n, greedy) })
			stored := first.cache.Pos()
			blob := first.Snapshot("id")
			if blob == nil {
				t.Fatal("Snapshot refused the session")
			}
			r, err := m.LoadSession(blob, "id")
			if err != nil {
				t.Fatalf("LoadSession: %v", err)
			}
			if reused := r.rewindForReuse(turn2); reused != stored {
				t.Fatalf("restored session reuses %d positions of %d stored", reused, stored)
			}
			r2, err := m.LoadSession(blob, "id") // rewindForReuse above consumed r's state; continue a fresh restore
			if err != nil {
				t.Fatal(err)
			}
			got := gen(t, func() (<-chan int, *Generation) { return r2.Generate(ctx, turn2, n, greedy) })
			if !slices.Equal(got, want) {
				t.Errorf("restored continuation %v, uninterrupted %v", got, want)
			}
			if p, q := r2.cache.Pos(), whole.cache.Pos(); p != q {
				t.Fatalf("restored cache at %d, uninterrupted at %d", p, q)
			}
			if d := ogMaxDiff(ogKV(r2.cache, r2.cache.Pos()), ogKV(whole.cache, whole.cache.Pos())); d != 0 {
				t.Errorf("restored K/V differs from uninterrupted by %g", d)
			}
		})
	}
}
