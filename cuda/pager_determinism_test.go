//go:build cuda && goinfer_testhooks

// Does repeated greedy generation on a PAGED MoE return the same tokens every time?
//
// FOUND WHILE MEASURING SOMETHING ELSE. In the spec-x-pager run the off arm produced 64 tokens on
// the first repeat of a prompt, 1 on the second and 0 on the third — same prompt, temperature 0,
// same process, `Generate` returning a nil error each time, while the two neighbouring prompts gave
// 64/64/64 around it. Greedy decode is deterministic by construction, so identical inputs returning
// different outputs means state is carrying between generations.
//
// THE HYPOTHESIS THIS TEST EXISTS TO CHECK, and the reason it belongs next to the pager rather than
// in a general decode test: the ONE piece of state deliberately kept across generations here is the
// C′ expert slot cache. Everything else is reset or positional — Forward and ForwardNoLogits both
// call Reset() at pos 0 (resident.go), so a Gated-DeltaNet's conv ring and matrix state are
// re-zeroed per sequence. The LRU is not, by design: it is a cache, and C′ documents itself as
// BIT-IDENTICAL to the fully-resident path. If that identity holds, generation N and generation 1
// must emit the same token ids no matter what the cache happens to hold. If they do not, a slot is
// being read while holding an expert other than the one the router asked for, and "bit-identical"
// is false in the configuration that only appears after the cache has been warmed by a previous
// generation — invisible to every single-generation test, which is what the existing 26B/35B cache
// tests are.
//
// It reports rather than diagnoses. Two outcomes are worth separating and both are recorded: ids
// that diverge at some position (a wrong-weights read) and ids that are a strict PREFIX of the
// first run (an early stop), because they point at different faults.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags "cuda goinfer_testhooks" ./cuda/ \
//	  -run TestPagerDeterminism -v -timeout 60m
package cuda

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

func TestPagerDeterminism(t *testing.T) {
	requireHeavyModel(t)
	path := os.Getenv("GOINFER_SPECPAGER_MODEL")
	if path == "" {
		home, _ := os.UserHomeDir()
		path = filepath.Join(home, "models", "qwen3.6-35b-a3b-int4.giw")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s: %v", path, err)
	}
	t.Setenv("GOINFER_MOE_CACHE_EXPERTS", "1")
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")
	hb := func(f string, a ...any) {
		fmt.Fprintf(os.Stderr, "[pager-det %s] %s\n", time.Now().Format("15:04:05"), fmt.Sprintf(f, a...))
	}

	hb("loading %s", filepath.Base(path))
	m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	rf := m.ResidentForwardForTest()
	if rf == nil {
		t.Fatalf("resident declined: %s", m.ResidentDecline())
	}
	r := rf.(*cudaResident)
	if !r.cacheExperts {
		t.Fatal("C′ staging off — nothing here would exercise the pager")
	}
	hb("resident + C′, %d slots/layer, topK %d", r.CacheSlotsForTest(), r.topK)

	tk, err := specPagerTokenizer(path)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	tmpl, err := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has})
	if err != nil {
		t.Fatalf("chat template: %v", err)
	}

	// A/B ON THE ONE SUSPECT, because a repro that only shows the symptom cannot name the cause.
	// decoder/resident_reuse.go (commit 3358e6ba, today) added prefix reuse on the resident KV and
	// gates it on GOINFER_NO_RESIDENT_REUSE alone — there is no recurrent-state exclusion in
	// residentReuseLen. For a Gated-DeltaNet family that is the exact hazard the rest of the tree
	// already refuses: the conv ring and matrix state are NOT position-truncatable
	// (decoder/deltanet.go: "why qwen3_5_moe falls back from prefix reuse / speculative"), and
	// cudaResident.Forward re-zeroes them only at pos == 0 — which a reused prefix never reaches.
	// So a second generation would decode from the PREVIOUS generation's tail state.
	//
	// If disabling reuse makes the symptom vanish, that is the cause. If it survives, the reuse
	// path is exonerated and the hunt moves to the pager, which is why both arms run here rather
	// than just the one that confirms the guess.
	for _, mode := range []string{"reuse-on", "reuse-off"} {
		t.Run(mode, func(t *testing.T) {
			if mode == "reuse-off" {
				t.Setenv("GOINFER_NO_RESIDENT_REUSE", "1")
			} else {
				t.Setenv("GOINFER_NO_RESIDENT_REUSE", "")
			}
			runDeterminismArm(t, m, r, tk, tmpl, hb)
		})
	}
}

func runDeterminismArm(t *testing.T, m *decoder.Model, r *cudaResident, tk *tokenizer.Tokenizer,
	tmpl *chat.Template, hb func(string, ...any)) {
	for _, text := range []string{
		"Write a Go function that merges two sorted int slices into one sorted slice.",
		"Explain what a hash table is and why its lookups are fast, in three sentences.",
	} {
		ids, e := tk.EncodeSegments(tmpl.RenderSegments("", []chat.Turn{{Role: "user", Content: text}}), false)
		if e != nil {
			t.Fatalf("encode: %v", e)
		}
		const runs, nNew = 5, 48
		var first []int
		for run := 0; run < runs; run++ {
			var got []int
			ch, gen := m.Generate(context.Background(), ids, nNew, decoder.SamplingParams{Temperature: 0})
			for tok := range ch {
				got = append(got, tok)
			}
			if gen != nil {
				if ge := gen.Err(); ge != nil {
					t.Errorf("%.40s… run %d: Generate error: %v", text, run, ge)
					continue
				}
			}
			// STAGE COUNT IS THE DIAGNOSTIC, not decoration. One staging event happens per routed
			// MoE layer per forward POSITION, so stages/layers is the number of positions the run
			// actually pushed through the model — prompt prefill included. In the run that raised
			// this, the first repeat of a prompt showed 88 positions (24 prompt + 64 generated) and
			// the second showed 65, i.e. the prompt was not prefilled the second time. Whether that
			// is KV being reused across a supposedly stateless Generate, or the prefill being
			// skipped for another reason, the position count distinguishes it from a sampling or
			// stop-token explanation, which would leave prefill untouched.
			hits, misses := r.CacheStatsForTest()
			st, _ := r.PagerStageStatsForTest()
			hb("%.40s… run %d: %d tok | %d stages (~%d positions) | hit %.1f%%",
				text, run, len(got), st, int(st)/max(1, nMoELayers(r)),
				100*float64(hits)/float64(max(1, int(hits+misses))))
			r.ResetPagerStatsForTest()
			if run == 0 {
				first = got
				continue
			}
			if len(got) == len(first) {
				same := true
				for i := range got {
					if got[i] != first[i] {
						same = false
						t.Errorf("%.40s… run %d DIVERGES from run 0 at token %d (%d != %d) — "+
							"greedy decode is deterministic, so this is state carried across generations",
							text, run, i, got[i], first[i])
						break
					}
				}
				if same {
					continue
				}
			} else {
				prefix := len(got) < len(first)
				for i := 0; i < len(got) && prefix; i++ {
					prefix = got[i] == first[i]
				}
				t.Errorf("%.40s… run %d emitted %d tokens against run 0's %d (strict prefix: %v) — "+
					"an early stop, not a divergent continuation", text, run, len(got), len(first), prefix)
			}
		}
	}
}

// nMoELayers counts the routed MoE layers, the divisor that turns staging events into forward
// positions. Counted from the built caches rather than assumed to be every layer: families with a
// dense prefix (first_k_dense_replace) route in only some of them.
func nMoELayers(r *cudaResident) int {
	n := 0
	for i := range r.layers {
		if r.layers[i].expCache != nil {
			n++
		}
	}
	return n
}
