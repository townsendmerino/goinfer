package decoder

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"

	"github.com/townsendmerino/goinfer/internal/giw"
)

// ogRand fills n values from a fixed linear congruential sequence in [-scale, scale): varied enough
// that greedy decoding does not collapse onto one token, and the same on every run.
func ogRand(n int, seed uint32, scale float32) []float32 {
	d := make([]float32, n)
	x := seed*2654435761 + 1
	for i := range d {
		x = x*1664525 + 1013904223
		d[i] = (float32(x>>8)/float32(1<<24)*2 - 1) * scale
	}
	return d
}

// ogMoEFixture writes a synthetic Mixtral-shaped checkpoint whose experts are large enough to page:
// the expert pager registers only spans that survive page rounding, and the committed mixtral-tiny's
// experts (8 KB at int8) do not, so a Load of it builds no pager. Here each int8 expert projection is
// 64 KB, four 16 KB pages.
func ogMoEFixture(t *testing.T) string {
	t.Helper()
	const hidden, heads, headDim, inter, vocab, layers, experts = 128, 4, 32, 512, 256, 2, 4
	dir := t.TempDir()
	cfg := fmt.Sprintf(`{"architectures":["MixtralForCausalLM"],"model_type":"mixtral","hidden_size":%d,
		"intermediate_size":%d,"num_hidden_layers":%d,"num_attention_heads":%d,"num_key_value_heads":%d,
		"head_dim":%d,"num_local_experts":%d,"num_experts_per_tok":2,"vocab_size":%d,"max_position_embeddings":256,
		"rms_norm_eps":1e-6,"rope_theta":10000,"tie_word_embeddings":false,"bos_token_id":1,"eos_token_id":2}`,
		hidden, inter, layers, heads, heads, headDim, experts, vocab)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfg), 0o644); err != nil {
		t.Fatal(err)
	}
	seed := uint32(1)
	r := func(n int, scale float32) []float32 { seed++; return ogRand(n, seed, scale) }
	ones := func(n int) []float32 {
		d := make([]float32, n)
		for i := range d {
			d[i] = 1
		}
		return d
	}
	ts := map[string]stTensor{
		"model.embed_tokens.weight": {[]int{vocab, hidden}, r(vocab*hidden, 1)},
		"model.norm.weight":         {[]int{hidden}, ones(hidden)},
		"lm_head.weight":            {[]int{vocab, hidden}, r(vocab*hidden, 0.2)},
	}
	q := heads * headDim
	for l := range layers {
		p := func(s string) string { return fmt.Sprintf("model.layers.%d%s", l, s) }
		ts[p(".self_attn.q_proj.weight")] = stTensor{[]int{q, hidden}, r(q*hidden, 0.1)}
		ts[p(".self_attn.k_proj.weight")] = stTensor{[]int{q, hidden}, r(q*hidden, 0.1)}
		ts[p(".self_attn.v_proj.weight")] = stTensor{[]int{q, hidden}, r(q*hidden, 0.1)}
		ts[p(".self_attn.o_proj.weight")] = stTensor{[]int{hidden, q}, r(hidden*q, 0.1)}
		ts[p(".input_layernorm.weight")] = stTensor{[]int{hidden}, ones(hidden)}
		ts[p(".post_attention_layernorm.weight")] = stTensor{[]int{hidden}, ones(hidden)}
		ts[p(".block_sparse_moe.gate.weight")] = stTensor{[]int{experts, hidden}, r(experts*hidden, 0.5)}
		for e := range experts {
			ep := func(s string) string { return p(fmt.Sprintf(".block_sparse_moe.experts.%d.%s.weight", e, s)) }
			ts[ep("w1")] = stTensor{[]int{inter, hidden}, r(inter*hidden, 0.1)}
			ts[ep("w3")] = stTensor{[]int{inter, hidden}, r(inter*hidden, 0.1)}
			ts[ep("w2")] = stTensor{[]int{hidden, inter}, r(hidden*inter, 0.1)}
		}
	}
	writeSafetensors(t, filepath.Join(dir, "model.safetensors"), ts)
	return dir
}

// ogGIW loads dir at quant and writes it as a .giw, the mmap-backed format the paging options read.
func ogGIW(t *testing.T, dir, quant string) string {
	t.Helper()
	m, err := Load(dir, Options{Quant: quant})
	if err != nil {
		t.Fatalf("Load(%s, %s): %v", dir, quant, err)
	}
	blob, err := SerializeWeights(m.w, "option-grid")
	m.Close()
	if err != nil {
		t.Fatalf("SerializeWeights: %v", err)
	}
	p := filepath.Join(t.TempDir(), "model.giw")
	if err := os.WriteFile(p, giw.Write(blob, nil), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// ogPagingCase is one paging configuration of the same .giw; check confirms the option took hold.
type ogPagingCase struct {
	name  string
	opts  Options
	check func(t *testing.T, m *Model)
}

// ogPagingCases lists the configurations. autoBudget is the pager budget of a StreamWeights load with
// WeightCacheBytes 0 (auto), which holds every expert of this small model: a 1-byte request must give
// a smaller budget, and the pager must have evicted. Evictions alone do not show the budget took hold:
// the pread pool rounds its slots and evicts even at the auto budget.
func ogPagingCases(autoBudget int64) []ogPagingCase {
	evicts := func(t *testing.T, m *Model) {
		t.Helper()
		if b := m.pager.budget(); b >= autoBudget {
			t.Errorf("pager budget %d at WeightCacheBytes 1, %d at auto: WeightCacheBytes did not take hold", b, autoBudget)
		}
		if _, _, ev := m.pager.stats(); ev == 0 {
			t.Errorf("the pager evicted nothing at a 1-byte budget")
		}
	}
	return []ogPagingCase{
		{"StreamWeights", Options{StreamWeights: true, AcceptSlowMoE: true}, nil},
		{"WeightCacheBytes 1", Options{StreamWeights: true, WeightCacheBytes: 1, AcceptSlowMoE: true}, evicts},
		{"MoEPager mmap", Options{StreamWeights: true, WeightCacheBytes: 1, MoEPager: "mmap", AcceptSlowMoE: true},
			func(t *testing.T, m *Model) {
				t.Helper()
				if m.pager.pool != nil {
					t.Error("MoEPager mmap built a pread pool")
				}
				evicts(t, m)
			}},
		{"MoEPager pool", Options{StreamWeights: true, WeightCacheBytes: 1, MoEPager: "pool", AcceptSlowMoE: true},
			func(t *testing.T, m *Model) {
				t.Helper()
				if m.pager.pool == nil {
					t.Error("MoEPager pool built no pread pool")
				}
				evicts(t, m)
			}},
	}
}

// ogAutoBudget is the pager budget of a StreamWeights load of giwPath with the automatic budget.
func ogAutoBudget(t *testing.T, giwPath, pager string) int64 {
	t.Helper()
	m := ogLoadPaged(t, giwPath, ogPagingCase{opts: Options{StreamWeights: true, MoEPager: pager, AcceptSlowMoE: true}})
	return m.pager.budget()
}

// ogLoadPaged loads giwPath under c and fails unless an expert pager was built.
func ogLoadPaged(t *testing.T, giwPath string, c ogPagingCase) *Model {
	t.Helper()
	m, err := Load(giwPath, c.opts)
	if err != nil {
		t.Fatalf("Load(%+v): %v", c.opts, err)
	}
	t.Cleanup(func() { m.Close() })
	if m.pager == nil {
		t.Fatal("no expert pager was built: StreamWeights did not take hold")
	}
	return m
}

// TestOptionPath_moePaging fills the CPU decode, CPU batched prefill and speculative verify cells of
// StreamWeights, WeightCacheBytes and MoEPager. Expert paging is documented bit-exact (a read-only
// re-fault, or a pread of the same bytes), so each path on a paged load of a .giw must emit the
// tokens and leave the K/V of the same path on the same .giw fully resident. At a 1-byte budget the
// pager evicts and re-reads experts throughout, which the cases check happened. The MoE is
// synthetic (ogMoEFixture) because the committed one's experts are too small to page.
func TestOptionPath_moePaging(t *testing.T) {
	giwPath := ogGIW(t, ogMoEFixture(t), "int8")
	ref, err := Load(giwPath, Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer ref.Close()
	if ref.pager != nil {
		t.Fatal("the resident reference built a pager")
	}
	ctx := context.Background()
	greedy := SamplingParams{}

	// runSession generates on a fresh session of m and returns its tokens and its first n positions' K/V
	// (n = 0: all of them).
	runSession := func(t *testing.T, m *Model, prompt []int, max, n int) ([]int, []float32, *Session) {
		s := m.NewSession(0)
		toks := gen(t, func() (<-chan int, *Generation) { return s.Generate(ctx, prompt, max, greedy) })
		if n == 0 {
			n = s.cache.Pos()
		}
		return toks, ogKV(s.cache, n), s
	}

	auto := min(ogAutoBudget(t, giwPath, "mmap"), ogAutoBudget(t, giwPath, "pool"))
	for _, c := range ogPagingCases(auto) {
		t.Run(c.name, func(t *testing.T) {
			m := ogLoadPaged(t, giwPath, c)

			t.Run("CPU decode", func(t *testing.T) {
				p := []int{3} // one prompt token: everything after it is per-token decode
				wt, wk, _ := runSession(t, ref, p, 16, 0)
				gt, gk, _ := runSession(t, m, p, 16, 0)
				if !slices.Equal(gt, wt) {
					t.Errorf("paged %v, resident %v", gt, wt)
				}
				if d := ogMaxDiff(gk, wk); d != 0 {
					t.Errorf("paged K/V differs from resident by %g", d)
				}
			})
			t.Run("CPU batched prefill", func(t *testing.T) {
				p := ogPrompt()
				if !m.canBatchN(len(p)) {
					t.Fatal("canBatchN is false: the batched prefill would not run")
				}
				wt, wk, _ := runSession(t, ref, p, 1, len(p))
				gt, gk, _ := runSession(t, m, p, 1, len(p))
				if !slices.Equal(gt, wt) {
					t.Errorf("paged first token %v, resident %v", gt, wt)
				}
				if d := ogMaxDiff(gk, wk); d != 0 {
					t.Errorf("paged prefill K/V differs from resident by %g", d)
				}
			})
			t.Run("speculative verify", func(t *testing.T) {
				if !m.specRollbackSafe() {
					t.Fatal("specRollbackSafe is false: the speculative path would refuse")
				}
				p := ogPrompt()
				const n = 20
				want, _, plain := runSession(t, ref, p, n, 0)
				full := append(append([]int(nil), p...), want...)
				for _, d := range []Drafter{&ogCountingDrafter{}, &ogRejectingDrafter{full: full}} {
					s := m.NewSession(0)
					got := gen(t, func() (<-chan int, *Generation) {
						ch, g, err := s.GenerateNgramSpeculative(ctx, p, n, d, 4, greedy)
						if err != nil {
							t.Fatalf("%T: %v", d, err)
						}
						return ch, g
					})
					if !slices.Equal(got, want) {
						t.Errorf("%T: paged speculative %v, resident plain %v", d, got, want)
					}
					q := plain.cache.Pos()
					if s.cache.Pos() < q {
						t.Errorf("%T: cache at %d, resident plain at %d", d, s.cache.Pos(), q)
						continue
					}
					if dd := ogMaxDiff(ogKV(s.cache, q), ogKV(plain.cache, q)); dd != 0 {
						t.Errorf("%T: paged speculative K/V differs from resident plain by %g", d, dd)
					}
				}
			})
			if c.check != nil {
				c.check(t, m)
			}
		})
	}
}

// TestOptionPath_moePagingDeclinesCPUBatch is the CPU batched decode cell of the same three options: a
// paged load is not offered to the MC3c batcher (cpuBatchModelEligible refuses an MoE, and a
// StreamWeights dense load through its layer pager), so EnableCPUBatch must decline, and concurrent
// generations on the paged model, each faulting experts the others evict, must still match the same
// generations run one at a time on the resident model.
func TestOptionPath_moePagingDeclinesCPUBatch(t *testing.T) {
	giwPath := ogGIW(t, ogMoEFixture(t), "int8")
	ref, err := Load(giwPath, Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer ref.Close()
	const nConv, maxTok = 4, 16
	prompts := make([][]int, nConv)
	for c := range prompts {
		prompts[c] = []int{1, (c*37 + 11) % 256, (c*53 + 7) % 256, 9}
	}
	run := func(t *testing.T, m *Model, concurrent bool) [][]int {
		out := make([][]int, nConv)
		one := func(c int) {
			s := m.NewSession(0)
			ch, g := s.Generate(context.Background(), prompts[c], maxTok, SamplingParams{})
			for id := range ch {
				out[c] = append(out[c], id)
			}
			if err := g.Err(); err != nil {
				t.Errorf("conversation %d: %v", c, err)
			}
		}
		if !concurrent {
			for c := range nConv {
				one(c)
			}
			return out
		}
		var wg sync.WaitGroup
		start := make(chan struct{})
		for c := range nConv {
			wg.Add(1)
			go func() { defer wg.Done(); <-start; one(c) }()
		}
		close(start)
		wg.Wait()
		return out
	}
	want := run(t, ref, false)
	for _, c := range ogPagingCases(0) {
		t.Run(c.name, func(t *testing.T) {
			opts := c.opts
			opts.CPUBatchDecode = CPUBatchOn
			m := ogLoadPaged(t, giwPath, ogPagingCase{opts: opts})
			if m.cpuBatchModelEligible() == nil {
				t.Fatal("cpuBatchModelEligible admits a paged MoE load")
			}
			if m.EnableCPUBatch(nConv) {
				t.Fatal("EnableCPUBatch accepted a paged MoE load")
			}
			got := run(t, m, true)
			for cv := range nConv {
				if !slices.Equal(got[cv], want[cv]) {
					t.Errorf("conversation %d: concurrent on the paged model %v, alone resident %v", cv, got[cv], want[cv])
				}
			}
		})
	}
}
