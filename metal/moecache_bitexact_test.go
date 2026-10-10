//go:build darwin && goinfer_testhooks

package metal

import (
	"fmt"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestMoECacheExperts_bitExactMetal is G-1 (docs/tasks/task-option-path-admission-2026-10.md §4.3, finding 2): Metal's expert
// cache (MoECacheExperts with fewer slots than experts) is bit-identical to the fully resident load, through the real caller
// (decoder.Load, then Generate): the generated tokens, their top-5 log-probabilities, and every layer's K/V.
//
// The reference is the fully resident load with ExactPrefill. A paged model never takes the batched f16 prefill (paging is never
// used by prefill), so against the default resident load a prompt over the batched-prefill floor differs by the batched
// prefill's own tolerance, which is the gap finding 2 saw; --exact-prefill alone reproduces it to the last bit and paging adds
// nothing. Two prompts, under and over the floor; at top-k, top-k+1 and top-k+2 slots and at all but one, each asserted to have
// evicted. The generic MoE path (mixtral-tiny: 8 experts, top-2) and Gemma 4's (gemma4-moe-tiny: 4 experts, top-2) page
// separately. Each planted defect (expertPoolDefect) must go red: an evicted slot not restaged, the GPU told the next slot, a
// neighbour's scales.
func TestMoECacheExperts_bitExactMetal(t *testing.T) {
	for _, fx := range []struct {
		dir        string
		nE, topK   int
		gemma4Path bool
	}{{"../testdata/mixtral-tiny", 8, 2, false}, {"../testdata/gemma4-moe-tiny", 4, 2, true}} {
		t.Run(fx.dir[len("../testdata/"):], func(t *testing.T) {
			if _, err := os.Stat(fx.dir); err != nil {
				t.Skipf("no fixture: %v", err)
			}
			base := decoder.Options{Quant: "int4", Backend: "metal", ExactPrefill: true}
			type fp struct {
				toks []int
				lp   []float64
				kv   [][]uint16
			}
			run := func(opts decoder.Options) (fp, int) {
				t.Helper()
				m, err := decoder.Load(fx.dir, opts)
				if err != nil {
					t.Fatal(err)
				}
				defer m.Close()
				a, ok := m.ResidentForwardForTest().(*metalResident)
				if !ok {
					t.Fatalf("not Metal-resident: %s", m.ResidentDecline())
				}
				var f fp
				for _, p := range [][]int{mgShort(), mgLong()} {
					toks, lp := mgGen(t, m, p, 12)
					f.toks, f.lp = append(f.toks, toks...), append(f.lp, lp...)
				}
				for l := range a.r.kc {
					f.kv = append(f.kv, slices.Clone(a.r.kc[l].U16s()[:a.r.kc[l].Len()/2]), slices.Clone(a.r.vc[l].U16s()[:a.r.vc[l].Len()/2])) // f16 K/V, read as decode_attn_r17_test does
				}
				ev := 0
				for _, L := range a.r.layers {
					switch {
					case L.moe != nil && L.moe.pool != nil:
						ev += L.moe.pool.evictions
					case L.g4moe != nil && L.g4moe.pool != nil:
						ev += L.g4moe.pool.evictions
					}
				}
				return f, ev
			}
			same := func(a, b fp) (bool, string) {
				if !slices.Equal(a.toks, b.toks) {
					return false, "tokens"
				}
				if !slices.Equal(a.lp, b.lp) {
					return false, fmt.Sprintf("log-probabilities (max |d| %.3g)", mgMaxDiff(a.lp, b.lp))
				}
				for i := range a.kv {
					if !slices.Equal(a.kv[i], b.kv[i]) {
						return false, fmt.Sprintf("K/V buffer %d", i)
					}
				}
				return true, ""
			}
			ref, _ := run(base)
			slotSet := []int{fx.topK, fx.topK + 1, fx.topK + 2, fx.nE - 1}
			slotSet = slices.Compact(slices.Sorted(slices.Values(slotSet)))
			for _, n := range slotSet {
				if n >= fx.nE {
					continue
				}
				o := base
				o.MoECacheExperts, o.MoECacheSlots = true, n
				got, ev := run(o)
				ok, what := same(ref, got)
				fmt.Fprintf(os.Stderr, "[G-1] %s: %d of %d slots, %d evictions: bit-identical %v %s\n", fx.dir, n, fx.nE, ev, ok, what)
				if ev == 0 {
					t.Errorf("%d slots: no eviction happened, so the comparison proves nothing about eviction", n)
				}
				if !ok {
					t.Errorf("%d slots: differs from the fully resident load in its %s", n, what)
				}
			}
			for d, name := range map[int]string{1: "an evicted slot not restaged", 2: "the GPU told the next slot", 3: "a neighbour's scales"} {
				expertPoolDefect = d
				o := base
				o.MoECacheExperts, o.MoECacheSlots = true, fx.topK
				got, _ := run(o)
				expertPoolDefect = 0
				ok, what := same(ref, got)
				fmt.Fprintf(os.Stderr, "[G-1] %s planted (%d) %s: differs in %s\n", fx.dir, d, name, what)
				if ok {
					t.Errorf("planted defect (%d) %s left the gate green", d, name)
				}
			}
		})
	}
}

// TestMoECacheExperts_int8DeclinesByName is G-2 (§4.3 finding 2): with int8 experts, the expert cache declines the
// Metal resident with a named reason, checked before the build, not by recovering the stage-setup panic. Without the
// check (skipPagedInt4CheckForTest), the decline is the recovered panic again, so the gate goes red.
func TestMoECacheExperts_int8DeclinesByName(t *testing.T) {
	for _, dir := range []string{"../testdata/mixtral-tiny", "../testdata/gemma4-moe-tiny"} {
		if _, err := os.Stat(dir); err != nil {
			t.Skipf("no fixture: %v", err)
		}
		decline := func(skipCheck bool) string {
			skipPagedInt4CheckForTest = skipCheck
			defer func() { skipPagedInt4CheckForTest = false }()
			m, err := decoder.Load(dir, decoder.Options{Quant: "int8int8", Backend: "metal", MoECacheExperts: true, MoECacheSlots: 2})
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			if _, ok := m.ResidentForwardForTest().(*metalResident); ok {
				t.Fatalf("%s: int8 experts went resident under the expert cache", dir)
			}
			return m.ResidentDecline()
		}
		got := decline(false)
		fmt.Fprintf(os.Stderr, "[G-2] %s: %s\n", dir, got)
		if !strings.Contains(got, "pages int4 experts only") || strings.Contains(got, "panicked") {
			t.Errorf("%s: decline %q does not name the reason (or came from a recovered panic)", dir, got)
		}
		without := decline(true)
		fmt.Fprintf(os.Stderr, "[G-2] %s without the check: %s\n", dir, without)
		if !strings.Contains(without, "panicked") {
			t.Errorf("%s: without the check the decline is %q, not the recovered panic the gate exists to replace", dir, without)
		}
	}
}

// TestMoECacheExperts_specAndSessionMetal holds the expert cache's last two option-grid cells on Metal (§4.3 finding
// 2): with 2 of 8 experts' slots on mixtral-tiny int4, speculative verify (both drafters, mgSpecMatchesPlain) emits what
// plain decode does, and a two-turn Session matches the fully resident ExactPrefill load's bit for bit, reuse included.
func TestMoECacheExperts_specAndSessionMetal(t *testing.T) {
	const dir = "../testdata/mixtral-tiny"
	if _, err := os.Stat(dir); err != nil {
		t.Skipf("no fixture: %v", err)
	}
	load := func(o decoder.Options) *decoder.Model {
		m, err := decoder.Load(dir, o)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { m.Close() })
		return m
	}
	ref := load(decoder.Options{Quant: "int4", Backend: "metal", ExactPrefill: true})
	paged := load(decoder.Options{Quant: "int4", Backend: "metal", ExactPrefill: true, MoECacheExperts: true, MoECacheSlots: 2})
	if a, ok := paged.ResidentForwardForTest().(*metalResident); !ok || !a.r.moe.paged {
		t.Fatalf("not a paged Metal resident: %s", paged.ResidentDecline())
	}
	mgSpecMatchesPlain(t, paged)
	rt, rl, rr := mgTwoTurns(t, ref)
	pt, pl, pr := mgTwoTurns(t, paged)
	fmt.Fprintf(os.Stderr, "[G-1 session] reused %d (resident) and %d (paged) positions; tokens equal %v; max |dlogprob| %.3g\n",
		rr, pr, slices.Equal(rt, pt), mgMaxDiff(rl, pl))
	if !slices.Equal(rt, pt) || !slices.Equal(rl, pl) || rr != pr {
		t.Errorf("the paged session's second turn differs from the resident one's")
	}
}
