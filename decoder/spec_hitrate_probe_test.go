//go:build darwin && goinfer_testhooks

package decoder

import (
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"
)

// TestSpecHitRate is Step-6 Step-0b: measure the SPECULATIVE hit rate for Metal MoE expert paging
// directly, on the CPU MoE path — the business case for building speculation at all. It does NOT
// rest on CUDA's 38-slot VRAM hit rate (81.6%, different hardware/ratio); it measures the strict
// quantity speculation needs: given the previous token's expert set plus a bounded per-layer LRU
// pool, is every expert in THIS token's top-8 already resident (no stall)?
//
// Model: gemma4-26b-A4B (the real target: 30 MoE layers, nE=128, top-8). Greedy autoregressive
// generation from a diverse seed (deterministic + repo-reproducible; a repetition guard flags the
// case where greedy loops, which would inflate locality). Router top-8 per MoE layer per token is
// captured via SetRouterCaptureForTest. Reports, per layer and aggregated:
//  1. exact-set match  P(top8_t == top8_{t-1})   — strictest "speculate from previous token".
//  2. coverage(N)      P(top8_t ⊆ per-layer LRU-N pool, which includes last token's set for N>=8).
//  3. stalls/token(N) = Σ_layers (1 - coverage_L(N)); implied ms/token at 0.230 ms/stall.
//
// GATE: compare (3) at an affordable N against 6.9 ms/token (full synchronous paging = a stall at
// every one of the 30 MoE layers). Recovers most → build speculation. Recovers < half → synchronous
// paging ships and speculation is dropped.
func TestSpecHitRate(t *testing.T) {
	if os.Getenv("GOINFER_SPEC_PROBE") == "" {
		t.Skip("set GOINFER_SPEC_PROBE=1 to run the speculative hit-rate probe (loads the 26B, ~minutes)")
	}
	// N-41: was a hardcoded /Users/<me>/ path — the last surviving dev-home path after G-06,
	// so this probe could only ever run on one machine and silently did nothing anywhere else.
	giw := os.Getenv("GOINFER_SPEC_PROBE_GIW")
	if giw == "" {
		giw = filepath.Join(os.Getenv("HOME"), "models", "gemma4-26b-int4.giw")
	}
	if _, err := os.Stat(giw); err != nil {
		t.Skip("no .giw (26B) — this measurement needs the real MoE target")
	}
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")
	m, err := Load(giw, Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()

	var moeLayers []int
	var nE, topK int
	for l := 0; l < 64; l++ {
		if _, _, e, k, _, ok := m.Gemma4MoERouterForTest(l); ok {
			moeLayers = append(moeLayers, l)
			nE, topK = e, k
		}
	}
	nMoE := len(moeLayers)
	if nMoE == 0 {
		t.Fatal("no MoE layers")
	}

	const nGen = 256
	SetRouterCaptureForTest(true)
	defer SetRouterCaptureForTest(false)

	// TRUSTWORTHY input needs a REAL, diverse token stream. This box has no in-package tokenizer, and
	// self-generated text from the 26B BASE model (greedy OR temp=1.0 top-k=40 sampling) degenerates
	// into a low-entropy loop (4–12% distinct tokens) whose expert-locality is NOT representative of
	// real generation. So the primary path is GOINFER_SPEC_TOKENS=<file of whitespace-separated real
	// token ids> (e.g. a corpus tokenized elsewhere / on the box), teacher-forced. Absent that, it
	// falls back to seeded sampling and LOUDLY flags the result as unrepresentative.
	cache := m.NewCache(4 + nGen + 4096)
	pos := 0
	var logits []float32
	feed := func(id int) {
		logits, err = m.ForwardForTest(id, cache)
		if err != nil {
			t.Fatalf("forward pos %d: %v", pos, err)
		}
		pos++
	}
	st := time.Now()
	var gen []int
	if f := os.Getenv("GOINFER_SPEC_TOKENS"); f != "" {
		gen = readTokenFile(t, f) // real tokenized corpus — the trustworthy path
		for _, id := range gen {
			feed(id)
		}
		t.Logf("teacher-forced %d REAL tokens from %s in %v (%.2f s/tok)", len(gen), f, time.Since(st), time.Since(st).Seconds()/float64(len(gen)))
	} else {
		for _, id := range []int{2, 108, 4368, 235, 9020, 1596, 774, 12} {
			feed(id)
		}
		rng := rand.New(rand.NewSource(1))
		for i := 0; i < nGen; i++ {
			nxt := sampleTopK(logits, 40, 1.0, rng)
			gen = append(gen, nxt)
			feed(nxt)
		}
		t.Logf("generated %d tokens (temp=1.0 top-40 sampling) in %v (%.2f s/tok)", nGen, time.Since(st), time.Since(st).Seconds()/float64(pos))
	}
	t.Logf("%d MoE layers (nE=%d top-%d)", nMoE, nE, topK)

	// Diversity guard: low distinct-token % ⇒ the stream is a loop ⇒ locality is UNREPRESENTATIVE
	// (a loop touches few distinct experts, so LRU coverage reads OPTIMISTICALLY high vs real text).
	uniq := map[int]bool{}
	for _, g := range gen {
		uniq[g] = true
	}
	distinctPct := 100 * float64(len(uniq)) / float64(len(gen))
	warn := ""
	if distinctPct < 40 {
		warn = "  *** UNREPRESENTATIVE (loop) — coverage below is OPTIMISTIC; use GOINFER_SPEC_TOKENS with a real corpus ***"
	}
	t.Logf("token diversity: %d distinct / %d (%.0f%%)%s", len(uniq), len(gen), distinctPct, warn)

	// Router capture → sets[layerPos][token] = top-k set. Decisions are token-outer, layer-inner
	// (routercapture.go): decision d → token d/nMoE, layerPos d%nMoE.
	idxAll, _ := RouterCaptureForTest()
	if len(idxAll)%nMoE != 0 {
		t.Fatalf("%d decisions not a multiple of %d MoE layers", len(idxAll), nMoE)
	}
	nTokCap := len(idxAll) / nMoE
	sets := make([][]map[int]bool, nMoE)
	for lp := 0; lp < nMoE; lp++ {
		sets[lp] = make([]map[int]bool, nTokCap)
	}
	for d, sel := range idxAll {
		tok, lp := d/nMoE, d%nMoE
		s := make(map[int]bool, len(sel))
		for _, e := range sel {
			s[e] = true
		}
		sets[lp][tok] = s
	}

	// (1) exact-set match rate, aggregated over layers.
	exactAgg := 0.0
	for lp := 0; lp < nMoE; lp++ {
		match, tot := 0, 0
		for tk := 1; tk < nTokCap; tk++ {
			if setEq(sets[lp][tk], sets[lp][tk-1]) {
				match++
			}
			tot++
		}
		exactAgg += float64(match) / float64(tot)
	}
	exactAgg /= float64(nMoE)

	// (2)+(3) coverage under per-layer LRU-N (move-to-front; pool = N most-recently-used, which
	// includes last token's set for N>=topK), swept over affordable slot counts.
	Ns := []int{8, 16, 24, 32, 38, 48, 64}
	full := float64(nMoE) * 0.230
	t.Logf("=== per-layer LRU-N coverage + stalls/token (0.230 ms/stall; full-sync = %d layers = %.1f ms/tok) ===", nMoE, full)
	t.Logf("exact-set match P(top%d_t == top%d_t-1), aggregated: %.1f%%", topK, topK, exactAgg*100)
	for _, N := range Ns {
		covAgg, stalls := 0.0, 0.0
		for lp := 0; lp < nMoE; lp++ {
			lru := []int{}
			hit, tot := 0, 0
			for tk := 0; tk < nTokCap; tk++ {
				if tk > 0 {
					pool := map[int]bool{}
					for _, e := range lru {
						if len(pool) >= N {
							break
						}
						pool[e] = true
					}
					covered := true
					for e := range sets[lp][tk] {
						if !pool[e] {
							covered = false
							break
						}
					}
					if covered {
						hit++
					}
					tot++
				}
				for e := range sets[lp][tk] {
					lru = moveToFront(lru, e)
				}
			}
			cov := float64(hit) / float64(tot)
			covAgg += cov
			stalls += 1 - cov
		}
		covAgg /= float64(nMoE)
		ms := stalls * 0.230
		t.Logf("  N=%-3d coverage=%5.1f%%  stalls/token=%5.2f  => %5.2f ms/token  (%.0f%% of full-sync recovered)",
			N, covAgg*100, stalls, ms, (1-ms/full)*100)
	}
	t.Logf("VERDICT INPUT: at affordable N≈38, ms/token << 6.9 ⇒ build speculation; recovers < half (>~3.5 ms) ⇒ "+
		"synchronous paging ships, speculation dropped. distinctPct=%.0f%% (low ⇒ this locality is optimistic).", distinctPct)
}

// sampleTopK draws from the top-k logits after temperature scaling — a representative diverse token
// stream (seeded rng → reproducible). Keeps only the k highest logits, softmaxes them, samples.
func sampleTopK(logits []float32, k int, temp float64, rng *rand.Rand) int {
	type le struct {
		id int
		v  float32
	}
	xs := make([]le, len(logits))
	for i, v := range logits {
		xs[i] = le{i, v}
	}
	sort.Slice(xs, func(a, b int) bool { return xs[a].v > xs[b].v })
	if k > len(xs) {
		k = len(xs)
	}
	xs = xs[:k]
	mx := xs[0].v
	var sum float64
	ps := make([]float64, k)
	for i, x := range xs {
		p := math.Exp(float64(x.v-mx) / temp)
		ps[i] = p
		sum += p
	}
	r := rng.Float64() * sum
	for i, p := range ps {
		r -= p
		if r <= 0 {
			return xs[i].id
		}
	}
	return xs[k-1].id
}

// readTokenFile reads whitespace-separated integer token ids from a file — the real-corpus input
// path (GOINFER_SPEC_TOKENS) that makes the locality number trustworthy.
func readTokenFile(t *testing.T, path string) []int {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read token file %s: %v", path, err)
	}
	var ids []int
	for _, f := range strings.Fields(string(b)) {
		n, err := strconv.Atoi(f)
		if err != nil {
			t.Fatalf("token file %s: %q is not an int: %v", path, f, err)
		}
		ids = append(ids, n)
	}
	if len(ids) < 2 {
		t.Fatalf("token file %s: need >=2 ids, got %d", path, len(ids))
	}
	return ids
}

func setEq(a, b map[int]bool) bool {
	if len(a) != len(b) {
		return false
	}
	for k := range a {
		if !b[k] {
			return false
		}
	}
	return true
}

func moveToFront(lru []int, e int) []int {
	out := make([]int, 0, len(lru)+1)
	out = append(out, e)
	for _, x := range lru {
		if x != e {
			out = append(out, x)
		}
	}
	return out
}
