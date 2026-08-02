package decoder

import (
	"container/list"
	"context"
	"fmt"
	"os"
	"sort"
	"testing"
)

// SPIKE (docs/ideas-weight-memory.md, idea #2 — MoE expert demand-paging viability).
// The make-or-break number: on a real large MoE, do a minority of experts dominate
// (so an LRU is warm), and what's the cold-miss tail latency a token eats when a
// routed expert isn't resident? We run the real Qwen3.6-35B-A3B, record every
// router top-k selection in forward order (moeSelTrace, set here), de-interleave by
// layer, and simulate an LRU of whole-expert pages at several RAM budgets — reporting
// hit rate + estimated added ms/token. Asset-gated; set GOINFER_MOE35 to the GGUF.
//
// This decides whether "35B-A3B on a 24 GB box, interactively" is real or just "it
//
//	loads." Run: GOINFER_MOE35=~/models/qwen3.6-35b-a3b-Q8_0.gguf go test ./decoder/ \
//	  -run TestMoEPagingSpike -v -timeout 3600s
func TestMoEPagingSpike(t *testing.T) {
	requireHeavyModel(t)
	path := os.Getenv("GOINFER_MOE35")
	if path == "" {
		path = os.Getenv("HOME") + "/models/qwen3.6-35b-a3b-Q8_0.gguf"
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no 35B MoE GGUF at %s (set GOINFER_MOE35): %v", path, err)
	}

	t.Logf("loading %s (int8int8)…", path)
	m, err := Load(path, Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	a := m.w.arch
	if a.MoE == nil {
		t.Fatalf("not a MoE model")
	}
	E, K, L := a.MoE.NumExperts, a.MoE.TopK, a.NumLayers
	// One routed expert's resident bytes at int8: gate+up+down = 3 · inter · hidden.
	expBytes := 3 * a.MoE.IntermediateDim * a.HiddenDim // +per-row scales ≈ negligible
	t.Logf("arch: %d layers, %d experts/layer, top-%d, expert≈%.1f MB int8, all-experts≈%.1f GB",
		L, E, K, float64(expBytes)/1e6, float64(L*E*expBytes)/1e9)

	// Record every router selection over a real generated stream.
	moeSelTrace = make([][]int, 0, 1<<16)
	defer func() { moeSelTrace = nil }()
	prompt := []int{785, 264, 6573, 311, 1438, 279, 2038, 25} // valid Qwen ids; the 200 generated tokens drive realistic decode-time routing
	const gen = 200
	ch, g := m.Generate(context.Background(), prompt, gen, SamplingParams{Temperature: 0})
	n := 0
	for range ch {
		n++
	}
	if g.Err() != nil {
		t.Fatalf("generate: %v", g.Err())
	}
	calls := len(moeSelTrace)
	tokens := calls / L
	t.Logf("generated %d tokens; %d moeMLP calls (%d tokens × %d layers)", n, calls, tokens, L)
	if calls == 0 {
		t.Fatalf("no MoE calls recorded — wrong forward path?")
	}

	// --- de-interleave: page key = layer*E + expert; build the access stream ---
	access := make([]int, 0, calls*K)  // page keys in forward order
	freq := make(map[int]int, L*E)     // per-(layer,expert) count
	perLayer := make([]map[int]int, L) // per-layer expert counts (skew)
	for i := range perLayer {
		perLayer[i] = map[int]int{}
	}
	for c, sel := range moeSelTrace {
		layer := c % L
		for _, e := range sel {
			key := layer*E + e
			access = append(access, key)
			freq[key]++
			perLayer[layer][e]++
		}
	}

	// --- skew: across all layers, what fraction of traffic hits the hottest experts ---
	type kv struct{ k, c int }
	var fl []kv
	for k, c := range freq {
		fl = append(fl, kv{k, c})
	}
	sort.Slice(fl, func(i, j int) bool { return fl[i].c > fl[j].c })
	total := len(access)
	cum, distinct := 0, len(freq)
	t.Logf("distinct (layer,expert) pages touched: %d of %d (%.0f%% of the universe)",
		distinct, L*E, 100*float64(distinct)/float64(L*E))
	for _, pct := range []float64{0.05, 0.10, 0.25, 0.50} {
		want := int(pct * float64(L*E))
		s := 0
		for i := 0; i < want && i < len(fl); i++ {
			s += fl[i].c
		}
		t.Logf("  hottest %4.0f%% of all experts (%d pages) absorb %5.1f%% of accesses",
			100*pct, want, 100*float64(s)/float64(total))
	}
	_ = cum

	// --- LRU hit-rate at several RAM budgets (whole-expert pages) ---
	// per-token accesses = K·L (the active working set is at least this many pages).
	activeSet := K * L
	t.Logf("active set (min resident) = K·L = %d experts ≈ %.1f GB", activeSet, float64(activeSet*expBytes)/1e9)
	t.Logf("=== hit rate vs RAM budget (whole-expert pages): B0 two-policy comparison ===")
	t.Logf("  LRU-tail = aikit ≤v1.14 (evict LEAST-recent); MRU = v1.15 6c0483f scan-resistant (evict MOST-recent OTHER)")
	t.Logf("  %-10s %-9s %-12s %-12s %-10s", "cacheGB", "pages", "LRU-tail hit%", "MRU hit%", "Δ (MRU-LRU)")
	// NVMe random-read model: ~20µs seek + bytes/3GBps. Conservative for a consumer SSD.
	const nvmeBps = 3.0e9
	const seekS = 20e-6
	perMissS := seekS + float64(expBytes)/nvmeBps
	sizes := []int{activeSet, activeSet * 2, activeSet * 3, activeSet * 4, activeSet * 6,
		int(0.25 * float64(L*E)), int(0.5 * float64(L*E)), distinct}
	seen := map[int]bool{}
	var worstDelta float64 = 1
	for _, C := range sizes {
		if C <= 0 || C > L*E || seen[C] {
			continue
		}
		seen[C] = true
		lh, lm := lruSim(access, C)
		mh, mm := mruSim(access, C)
		lrate := float64(lh) / float64(lh+lm)
		mrate := float64(mh) / float64(mh+mm)
		worstDelta = min(worstDelta, mrate-lrate)
		t.Logf("  %-10.1f %-9d %-12.1f %-12.1f %+-10.1f",
			float64(C*expBytes)/1e9, C, 100*lrate, 100*mrate, 100*(mrate-lrate))
		_ = perMissS
	}
	t.Logf("worst MRU−LRU hit-rate delta across budgets: %+.1f pp (negative ⇒ the v1.15 policy REGRESSES the expert pager — B0)", 100*worstDelta)

	// Steady-state (drop the first ~20% as cache warmup) at a mid budget.
	warm := len(access) / 5
	if warm < len(access) {
		C := min(activeSet*3, L*E)
		h, mi := lruSim(access[warm:], C)
		t.Logf("steady-state (post-warmup) @ %.1f GB: hit %.1f%%, miss/tok %.1f",
			float64(C*expBytes)/1e9, 100*float64(h)/float64(h+mi), float64(mi)/float64(tokens))
	}
}

// lruSim runs an exact LRU of capacity C pages over the access stream, returning
// (hits, misses). O(1)/access via container/list.
func lruSim(access []int, C int) (hits, misses int) {
	ll := list.New()
	pos := make(map[int]*list.Element, C)
	for _, key := range access {
		if el, ok := pos[key]; ok {
			ll.MoveToFront(el)
			hits++
			continue
		}
		misses++
		if ll.Len() >= C {
			back := ll.Back()
			delete(pos, back.Value.(int))
			ll.Remove(back)
		}
		pos[key] = ll.PushFront(key)
	}
	return
}

// mruSim mirrors aikit v1.15's scan-resistant SpanCache eviction (6c0483f): on a
// miss over budget it releases the MOST-recently-touched OTHER member (Front().Next()
// after pushing the new key to Front), not the LRU tail. For a frequency-skewed
// demand signal — which the MoE router is — this evicts the hot set and keeps the
// cold prefix, the opposite of what the expert pager wants (task doc B0).
func mruSim(access []int, C int) (hits, misses int) {
	ll := list.New()
	pos := make(map[int]*list.Element, C)
	for _, key := range access {
		if el, ok := pos[key]; ok {
			ll.MoveToFront(el)
			hits++
			continue
		}
		misses++
		el := ll.PushFront(key)
		pos[key] = el
		if ll.Len() > C { // evict the most-recent OTHER member (Front().Next())
			if victim := el.Next(); victim != nil {
				delete(pos, victim.Value.(int))
				ll.Remove(victim)
			}
		}
	}
	return
}

var _ = fmt.Sprint
