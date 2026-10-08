package decoder

import (
	"context"
	"math"
	"slices"
	"testing"
)

// The option grid's cells (optiongrid.go) are moved from "admitted, untested" to "tested" by the
// tests in this file. Each sets the option through Options, loads a committed fixture with Load (the
// entry point a library caller uses), and runs the path through the function production calls, so
// the option reaches the path the way it does for a user. They run in CI: llama-tiny is in git.

const ogFixture = "../testdata/llama-tiny"

// ogCase is one option setting. effect checks that the option took hold on the loaded model, so a
// case whose option is silently dropped by Load fails instead of passing as a plain load.
type ogCase struct {
	name   string
	opts   Options
	effect func(t *testing.T, m *Model)
}

func ogLoad(t *testing.T, c ogCase) *Model {
	t.Helper()
	m, err := Load(ogFixture, c.opts)
	if err != nil {
		t.Fatalf("Load(%s, %+v): %v", ogFixture, c.opts, err)
	}
	t.Cleanup(func() { m.Close() })
	if c.effect != nil {
		c.effect(t, m)
	}
	return m
}

// ogDiffersFromPlain is an effect check for options that change the numbers: the option's first
// prefill logits must differ from a plain f32 load's, or the case is not exercising the option.
func ogDiffersFromPlain(t *testing.T, m *Model) {
	t.Helper()
	plain, err := Load(ogFixture, Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer plain.Close()
	p := ogPrompt()
	a, err := m.prefillLogits(context.Background(), p, m.NewCache(len(p)))
	if err != nil {
		t.Fatal(err)
	}
	b, err := plain.prefillLogits(context.Background(), p, plain.NewCache(len(p)))
	if err != nil {
		t.Fatal(err)
	}
	if ogMaxDiff(a, b) == 0 {
		t.Fatal("logits identical to a plain f32 load: the option did not take hold")
	}
}

// ogPrompt repeats a phrase so the n-gram drafter has something to propose during decode.
func ogPrompt() []int {
	return []int{1, 17, 42, 99, 5, 17, 42, 99, 5, 17, 42, 99, 5, 23, 61, 17, 42}
}

func ogMaxDiff(a, b []float32) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}
	d := 0.0
	for i := range a {
		d = math.Max(d, math.Abs(float64(a[i]-b[i])))
	}
	return d
}

// ogKV flattens the first n positions of every layer's K and V, in whichever precision the cache
// stores: f32 rows, or int8 rows with their per-head scales.
func ogKV(c *KVCache, n int) []float32 {
	var out []float32
	for l := 0; l < c.numLayers; l++ {
		if c.quant == kvI8 {
			nKV := c.kvDim / c.headDim
			for _, q := range [][]int8{c.keysQ[l][:n*c.kvDim], c.valsQ[l][:n*c.kvDim]} {
				for _, v := range q {
					out = append(out, float32(v))
				}
			}
			out = append(out, c.keyScale[l][:n*nKV]...)
			out = append(out, c.valScale[l][:n*nKV]...)
			continue
		}
		out = append(out, c.keys[l][:n*c.kvDim]...)
		out = append(out, c.vals[l][:n*c.kvDim]...)
	}
	return out
}

// TestOptionPath_cpuBatchedPrefill fills the "CPU batched prefill" column: for each option, a
// Session's prefill (Session.Generate → prefillLogits → forwardLayersN) must leave the same K/V and
// pick the same first token as the sequential per-token prefill on the same model. prefillLogits
// documents the batched prefill as bit-identical to the sequential one.
func TestOptionPath_cpuBatchedPrefill(t *testing.T) {
	cases := []ogCase{
		{name: "Quant int8", opts: Options{Quant: "int8"}, effect: ogDiffersFromPlain},
		{name: "Quant int8int8", opts: Options{Quant: "int8int8"}, effect: ogDiffersFromPlain},
		{name: "Quant int4", opts: Options{Quant: "int4"}, effect: ogDiffersFromPlain},
		{name: "KVQuant i8", opts: Options{KVQuant: "i8"}, effect: func(t *testing.T, m *Model) {
			if m.NewCache(1).quant != kvI8 {
				t.Fatal("KVQuant i8 did not give an int8 cache")
			}
		}},
		{name: "ActQuantGroup 32 (int8int8)", opts: Options{Quant: "int8int8", ActQuantGroup: 32}, effect: func(t *testing.T, m *Model) {
			if m.actGroup != 32 {
				t.Fatalf("actGroup %d, want 32", m.actGroup)
			}
		}},
		{name: "ExactPrefill", opts: Options{ExactPrefill: true}, effect: func(t *testing.T, m *Model) {
			if !m.ExactPrefill() || m.cpuFastAttention() {
				t.Fatal("ExactPrefill did not take hold")
			}
		}},
		{name: "EmbedInt4 (int4)", opts: Options{Quant: "int4", EmbedInt4: true}, effect: func(t *testing.T, m *Model) {
			ref, err := Load(ogFixture, Options{Quant: "int4"})
			if err != nil {
				t.Fatal(err)
			}
			defer ref.Close()
			p := ogPrompt()
			a, _ := m.prefillLogits(context.Background(), p, m.NewCache(len(p)))
			b, _ := ref.prefillLogits(context.Background(), p, ref.NewCache(len(p)))
			if ogMaxDiff(a, b) == 0 {
				t.Fatal("logits identical to int4 without EmbedInt4: the option did not take hold")
			}
		}},
	}
	ctx := context.Background()
	prompt := ogPrompt()
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			m := ogLoad(t, c)
			if !m.canBatchN(len(prompt)) {
				t.Fatal("canBatchN is false: the batched path would not run")
			}

			// Sequential reference: the per-token fallback prefillLogits takes when it cannot batch.
			seq := m.NewCache(len(prompt) + 1)
			for _, id := range prompt[:len(prompt)-1] {
				if _, err := m.runLayers(id, seq); err != nil {
					t.Fatal(err)
				}
			}
			seqLogits, err := m.forward(prompt[len(prompt)-1], seq)
			if err != nil {
				t.Fatal(err)
			}

			s := m.NewSession(len(prompt) + 1)
			got := gen(t, func() (<-chan int, *Generation) { return s.Generate(ctx, prompt, 1, SamplingParams{}) })
			if len(got) != 1 || got[0] != argmax(seqLogits) {
				t.Errorf("batched prefill picked %v, sequential picks %d", got, argmax(seqLogits))
			}
			if d := ogMaxDiff(ogKV(s.cache, len(prompt)), ogKV(seq, len(prompt))); d != 0 {
				t.Errorf("batched prefill K/V differs from sequential by %g; documented bit-identical", d)
			}
		})
	}
}

// ogCountingDrafter records how many rounds the drafter proposed something, so the speculative test
// can show its verify step ran on real drafts rather than degenerating to plain decode.
type ogCountingDrafter struct {
	d      NgramDrafter
	drafts int
}

func (c *ogCountingDrafter) Draft(ctx []int, k int) []int {
	out := c.d.Draft(ctx, k)
	if len(out) > 0 {
		c.drafts++
	}
	return out
}

// ogRejectingDrafter knows the plain greedy continuation and proposes it with the block's last token
// changed, so every verify accepts all but one drafted token and must roll the rejected one back.
// An n-gram drafter on a model this small mostly guesses right, and a fully accepted block never
// exercises the rollback.
type ogRejectingDrafter struct {
	full   []int // prompt followed by the plain greedy tokens
	blocks int
}

func (r *ogRejectingDrafter) Draft(ctx []int, k int) []int {
	i := len(ctx)
	if k < 2 || i >= len(r.full) {
		return nil
	}
	out := append([]int(nil), r.full[i:min(i+k, len(r.full))]...)
	if len(out) < 2 {
		return nil
	}
	out[len(out)-1] ^= 1 // a different token id, so the verify rejects it
	r.blocks++
	return out
}

// TestOptionPath_specVerify fills the "speculative verify" column: for each option, greedy n-gram
// speculative decoding through a Session (Session.GenerateNgramSpeculative, whose verify runs the
// drafted block through one batched forward and rolls back what it rejects) must emit exactly the
// tokens plain greedy decoding does and leave exactly the same K/V. Tokens alone are too coarse on
// this model: greedy picks rarely move under a small numeric error, so the cache is compared too.
// Two drafters: the n-gram one production uses, and one built to have every block partly rejected.
func TestOptionPath_specVerify(t *testing.T) {
	cases := []ogCase{
		{name: "ActQuantGroup 32 (int8int8)", opts: Options{Quant: "int8int8", ActQuantGroup: 32}, effect: func(t *testing.T, m *Model) {
			if m.actGroup != 32 {
				t.Fatalf("actGroup %d, want 32", m.actGroup)
			}
		}},
		{name: "CPUBatchDecode on", opts: Options{CPUBatchDecode: CPUBatchOn}, effect: func(t *testing.T, m *Model) {
			if !m.EnableCPUBatch(2) || !m.CPUBatchActive() {
				t.Fatal("CPUBatchDecode on did not enable the CPU batch")
			}
		}},
		{name: "EmbedInt4 (int4)", opts: Options{Quant: "int4", EmbedInt4: true}, effect: ogDiffersFromPlain},
		{name: "ExactPrefill", opts: Options{ExactPrefill: true}, effect: func(t *testing.T, m *Model) {
			if !m.ExactPrefill() {
				t.Fatal("ExactPrefill did not take hold")
			}
		}},
		{name: "KVQuant i8", opts: Options{KVQuant: "i8"}, effect: func(t *testing.T, m *Model) {
			if m.NewCache(1).quant != kvI8 {
				t.Fatal("KVQuant i8 did not give an int8 cache")
			}
		}},
	}
	ctx := context.Background()
	greedy := SamplingParams{}
	prompt := ogPrompt()
	const n = 24
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			m := ogLoad(t, c)
			if !m.specRollbackSafe() {
				t.Fatal("specRollbackSafe is false: the speculative path would refuse")
			}
			plain := m.NewSession(0)
			ref := gen(t, func() (<-chan int, *Generation) { return plain.Generate(ctx, prompt, n, greedy) })
			full := append(append([]int(nil), prompt...), ref...)

			for _, K := range []int{2, 4} {
				ngram := &ogCountingDrafter{}
				reject := &ogRejectingDrafter{full: full}
				for _, d := range []struct {
					name string
					d    Drafter
					ran  func() int
				}{
					{"ngram", ngram, func() int { return ngram.drafts }},
					{"rejecting", reject, func() int { return reject.blocks }},
				} {
					s := m.NewSession(0)
					got := gen(t, func() (<-chan int, *Generation) {
						ch, g, err := s.GenerateNgramSpeculative(ctx, prompt, n, d.d, K, greedy)
						if err != nil {
							t.Fatalf("K=%d %s: %v", K, d.name, err)
						}
						return ch, g
					})
					if !slices.Equal(got, ref) {
						t.Errorf("K=%d %s: speculative %v, plain greedy %v", K, d.name, got, ref)
					}
					if d.ran() == 0 {
						t.Errorf("K=%d %s: the drafter never proposed a block, so verify never ran", K, d.name)
					}
					// A block accepted past maxTokens stays committed: the session can end a few positions
					// beyond plain decode, holding tokens it never emitted. They must be the greedy
					// continuation and agree with the cache; the shared positions must match exactly.
					q := plain.cache.Pos()
					if s.cache.Pos() < q || len(s.Tokens()) != s.cache.Pos() || !slices.Equal(s.Tokens()[:q], plain.Tokens()) {
						t.Errorf("K=%d %s: session holds %d tokens at cache position %d; plain decode holds %d at %d",
							K, d.name, len(s.Tokens()), s.cache.Pos(), len(plain.Tokens()), q)
						continue
					}
					if dd := ogMaxDiff(ogKV(s.cache, q), ogKV(plain.cache, q)); dd != 0 {
						t.Errorf("K=%d %s: K/V differs from plain decode's by %g", K, d.name, dd)
					}
				}
			}
		})
	}
}
