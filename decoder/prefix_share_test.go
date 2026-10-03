package decoder

import (
	"context"
	"math"
	"testing"
)

// D8's gate (docs/tasks/task-constrained-confidence.md): sharing a prompt prefix across questions must give the same answers as prefilling each prompt alone, on the
// Qwen3.5 checkpoints (dense, MoE), f32 and int4 and with an int8 KV cache, and the checkpoint must be a COPY: resuming from it any number of times, in any order,
// leaves it as it was. Like the batched-versus-sequential gates beside it the difference from the unshared answer is bounded, not claimed zero.

// sharedPrompts is a "state" prefix of the given length followed by suffixes of different lengths, as a decision request's questions are.
func sharedPrompts(vocab, prefix int, suffixes []int) [][]int {
	base := make([]int, prefix)
	for i := range base {
		base[i] = (i*37 + 11) % vocab
	}
	out := make([][]int, len(suffixes))
	for k, n := range suffixes {
		p := append([]int(nil), base...)
		for i := 0; i < n; i++ {
			p = append(p, (k*101+i*53+7)%vocab)
		}
		out[k] = p
	}
	return out
}

func relL2(a, b []float32) float64 {
	var ne, nb float64
	for i := range a {
		d := float64(a[i]) - float64(b[i])
		ne, nb = ne+d*d, nb+float64(b[i])*float64(b[i])
	}
	return math.Sqrt(ne / nb)
}

func TestPromptHiddenMany_matchesSeparate(t *testing.T) {
	var shared, members int
	prefixShareHook = func(p, n int) { shared++; members += n }
	defer func() { prefixShareHook = nil }()
	for _, tc := range []struct{ ckpt, quant string }{
		{"qwen3_5-tiny-normw", ""}, {"qwen3_5_moe-tiny", ""}, {"qwen3_5-tiny-normw", "int4"}, {"qwen3_5_moe-tiny", "int4"},
	} {
		t.Run(tc.ckpt+"/"+tc.quant, func(t *testing.T) {
			m := loadFixtureModel(t, tc.ckpt, tc.quant)
			if !m.CanSharePrefix() {
				t.Fatalf("%s should take the checkpoint path", tc.ckpt)
			}
			prompts := sharedPrompts(m.w.arch.VocabSize, 40, []int{1, 3, 8, 20, 2})
			before := shared
			got, err := m.PromptHiddenMany(context.Background(), prompts)
			if err != nil {
				t.Fatal(err)
			}
			if shared != before+1 {
				t.Fatalf("%d checkpoints taken, want exactly 1 for one shared prefix", shared-before)
			}
			worst := 0.0
			for i, p := range prompts {
				want, err := m.PromptHidden(context.Background(), p)
				if err != nil {
					t.Fatal(err)
				}
				worst = math.Max(worst, relL2(got[i], want))
			}
			t.Logf("5 prompts sharing 40 tokens: worst relative L2, shared vs alone %.3g", worst)
			if worst > 1e-6 {
				t.Errorf("worst relative L2 %.3g > 1e-6", worst)
			}
		})
	}
	if members != 20 {
		t.Errorf("%d prompts resumed from a checkpoint across the four subtests, want 20 (5 each)", members)
	}
}

// With an int8 KV cache the checkpoint copies the int8 rows and their scales.
func TestPromptHiddenMany_int8KV(t *testing.T) {
	m, err := Load("testdata/qwen3_5-tiny-normw", Options{KVQuant: "i8"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	prompts := sharedPrompts(m.w.arch.VocabSize, 33, []int{4, 9, 2})
	got, err := m.PromptHiddenMany(context.Background(), prompts)
	if err != nil {
		t.Fatal(err)
	}
	for i, p := range prompts {
		want, _ := m.PromptHidden(context.Background(), p)
		if r := relL2(got[i], want); r > 1e-6 {
			t.Errorf("prompt %d: relative L2 %.3g > 1e-6 with an int8 KV cache", i, r)
		}
	}
}

// The checkpoint is read-only: the same suffix resumed twice, and in a different order around another suffix, gives the identical hidden state every time. A
// checkpoint that shared memory with the caches resumed from it (the Gated DeltaNet state is updated in place per token) would drift with each resume.
func TestPrefixCheckpoint_isNotMutatedByResuming(t *testing.T) {
	m := loadFixtureModel(t, "qwen3_5-tiny-normw", "")
	prompts := sharedPrompts(m.w.arch.VocabSize, 24, []int{6, 11})
	cp, err := m.checkpointPrefix(context.Background(), prompts[0][:24])
	if err != nil {
		t.Fatal(err)
	}
	a1, _ := m.resumeHidden(context.Background(), cp, prompts[0][24:])
	b1, _ := m.resumeHidden(context.Background(), cp, prompts[1][24:])
	a2, _ := m.resumeHidden(context.Background(), cp, prompts[0][24:])
	b2, _ := m.resumeHidden(context.Background(), cp, prompts[1][24:])
	for i := range a1 {
		if a1[i] != a2[i] || b1[i] != b2[i] {
			t.Fatalf("resuming the checkpoint again changed the result at %d: the checkpoint was mutated", i)
		}
	}
	if cp.cache.pos != 24 {
		t.Errorf("the checkpoint's position moved to %d", cp.cache.pos)
	}
}

// A family that cannot take the path (a dense llama, with no recurrent state) answers exactly as PromptHidden does: same bits, no checkpoint.
func TestPromptHiddenMany_fallsBackWhereItCannotShare(t *testing.T) {
	taken := 0
	prefixShareHook = func(int, int) { taken++ }
	defer func() { prefixShareHook = nil }()
	m, err := Load("../testdata/llama-tiny", Options{})
	if err != nil {
		t.Skipf("no llama-tiny: %v", err)
	}
	defer m.Close()
	if m.CanSharePrefix() {
		t.Fatal("a dense llama should not take the Qwen3.5 checkpoint path")
	}
	prompts := sharedPrompts(m.w.arch.VocabSize, 30, []int{3, 5})
	got, err := m.PromptHiddenMany(context.Background(), prompts)
	if err != nil {
		t.Fatal(err)
	}
	for i, p := range prompts {
		want, _ := m.PromptHidden(context.Background(), p)
		for j := range want {
			if got[i][j] != want[j] {
				t.Fatalf("prompt %d differs from PromptHidden at %d on a model that cannot share", i, j)
			}
		}
	}
	if taken != 0 {
		t.Errorf("%d checkpoints taken on a model that cannot share", taken)
	}
}

func TestShareGroups(t *testing.T) {
	mk := func(prefix []int, tail ...int) []int { return append(append([]int(nil), prefix...), tail...) }
	long := sharedPrompts(100, 20, []int{2})[0][:20]
	other := []int{9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9}
	prompts := [][]int{mk(long, 1, 2), other, mk(long, 3), mk(long, 1, 5)}
	members, prefix := shareGroups(prompts, 16)
	if len(members) != 2 || len(members[0]) != 3 || len(members[1]) != 1 {
		t.Fatalf("groups %v, want {0,2,3} and {1}", members)
	}
	if prefix[0] != 20 || prefix[1] != 0 {
		t.Errorf("prefixes %v, want 20 for the group and 0 for the loner", prefix)
	}
	// A prompt that IS the shared prefix keeps one token of its own: the last position's hidden state must come from a suffix run.
	members, prefix = shareGroups([][]int{long, mk(long, 7, 8)}, 16)
	if len(members) != 1 || prefix[0] != 19 {
		t.Errorf("groups %v prefixes %v, want one group with the prefix cut to 19", members, prefix)
	}
	// Below the minimum nothing is shared.
	members, _ = shareGroups([][]int{mk(long[:10], 1), mk(long[:10], 2)}, 16)
	if len(members) != 2 {
		t.Errorf("a 10-token common prefix was shared (groups %v)", members)
	}
}

func TestPromptHiddenMany_cancelled(t *testing.T) {
	m := loadFixtureModel(t, "qwen3_5-tiny-normw", "")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := m.PromptHiddenMany(ctx, sharedPrompts(m.w.arch.VocabSize, 30, []int{3, 4})); err == nil {
		t.Error("a cancelled context did not stop PromptHiddenMany")
	}
}

func TestPromptLogitsMany_isTheLMHeadOfTheHidden(t *testing.T) {
	m := loadFixtureModel(t, "qwen3_5-tiny-normw", "")
	prompts := sharedPrompts(m.w.arch.VocabSize, 28, []int{5, 9})
	lg, err := m.PromptLogitsMany(context.Background(), prompts)
	if err != nil {
		t.Fatal(err)
	}
	hs, _ := m.PromptHiddenMany(context.Background(), prompts)
	for i := range prompts {
		want := m.lmHeadN(hs[i], 1)
		if len(lg[i]) != m.w.arch.VocabSize {
			t.Fatalf("logits are %d wide, want %d", len(lg[i]), m.w.arch.VocabSize)
		}
		for j := range want {
			if lg[i][j] != want[j] {
				t.Fatalf("prompt %d: logits differ from lmHead(hidden) at %d", i, j)
			}
		}
	}
}
