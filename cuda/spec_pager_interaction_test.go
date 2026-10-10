//go:build cuda && goinfer_testhooks

// Does a speculative verify break the expert pager?
//
// The hypothesis is that a width-K verify forces expert re-fetches on a paged MoE. It is generalized off a field report's
// CPU/GPU hybrid framing (which this tree cannot run) onto the path that does exist: a width-K verify against a model whose
// routed experts are paged host->VRAM by C′. Only the mechanism is under test; no number from that report appears here.
//
// TWO HYPOTHESES, AND WALL-CLOCK CANNOT SEPARATE THEM, which is why this test has the shape it has:
//
//	H-paging    a width-K verify presents K positions' routing at once, so it asks for several
//	            times the distinct experts a decode step does, overflowing a slot budget that was
//	            tuned on decode traffic.
//	H-noamort   the batched weight-stationary path declines MoE, so a verify walks position by
//	            position: the pager sees ordinary decode traffic and is never stressed, but the
//	            verify pays K full decode steps to commit at most K tokens and the amortization
//	            speculation depends on is simply absent.
//
// Both predict a regression, with the same sign and a similar size. They are told apart by PagerStageStatsForTest's
// distinct-experts-per-staging-event, which rises with K under H-paging and stays pinned at topK under H-noamort.
// Pre-registration, with the thresholds and the ambiguous→parked bands fixed before any arm ran:
// docs/measurements/spec-x-pager-prereg-2026-09-02.md
//
// H-noamort's premise is as of the experiment: prefillStaticDecline admits MoE now (the per-row MoE FFN,
// TestPrefillMoE_bitIdentical).
//
// ONE SLOT RUNG PER PROCESS. The slot depth is read at Load, so a ladder inside one process would mean reloading a 20 GB
// pinned allocation per rung; the runner drives the rungs by re-invoking with GOINFER_MOE_CACHE_SLOTS set, and each
// process reports the depth actually BUILT (capSlots caps a request to free VRAM, so the request is not the depth).
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_MOE_CACHE_SLOTS=16 \
//	  go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestSpecPagerInteraction -v -timeout 3h
package cuda

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/giw"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// specPagerPrompts is realistic traffic, spanning prose / code / math.
//
// scripts/prompts.json is deliberately NOT used. It is four-unique-word filler ("the the the …"), and on a MoE every such
// position routes to the same experts: the pager never has to stage anything, so the effect under test cannot appear and
// the run would manufacture a null.
var specPagerPrompts = []string{
	"Explain what a hash table is and why its lookups are fast, in three sentences.",
	"Write a Go function that merges two sorted int slices into one sorted slice.",
	"A train travels 120 km in 1.5 hours, then 90 km in 45 minutes. What is its average speed over the whole journey? Show your working.",
}

// specPagerArm is one measured configuration. Times are per (prompt, repeat) and kept RAW rather than pre-averaged, so
// the paired ratios R1 asks for can be formed per round: a ratio of medians can disagree with the paired form, and that
// is uncorrectable after the fact.
type specPagerArm struct {
	label string
	kind  string // "off" | "block" | "ngram"
	width int    // verify width; 0 on the off arm

	// rounds/drafted/accepted come from SpecStats on the n-gram path, which has no OnRound seam.
	rounds, drafted, accepted int
	short                     int // generations that stopped before nNew, excluded from timings

	// secs is wall-clock NORMALIZED to a fixed token count: a round commits a whole block, so a speculative arm overshoots or
	// undershoots MaxTokens by a few tokens while the off arm lands exactly on it, and raw durations would charge an arm for
	// tokens it produced as a bonus. The divisor is the arm's OWN emitted count, never another arm's decode step.
	secs   [][]float64 // [prompt][repeat] seconds to emit nNew tokens, normalized
	tokens [][]int     // [prompt][repeat] tokens actually produced (the normalization divisor)

	// Pager demand + reuse, summed over the arm.
	stages, distinct, hits, misses uint64
	// Rounds, for realized alpha. Empty on the off arm, which has no rounds.
	committed []int

	// declined records a per-request refusal from the speculative path, which is an OUTCOME and
	// not an error to abort on: "the drafter cannot run here" answers the question this test asks
	// just as much as "the drafter runs here and is slow", and aborting would throw away the off
	// arm and the instrument validation that go with it.
	declined string
}

func specPagerMean(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := 0.0
	for _, x := range xs {
		s += x
	}
	return s / float64(len(xs))
}

func specPagerMedian(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	c := append([]float64(nil), xs...)
	sort.Float64s(c)
	if n := len(c); n%2 == 1 {
		return c[n/2]
	}
	return (c[len(c)/2-1] + c[len(c)/2]) / 2
}

// specPagerSpread is (max-min)/mean, the form the Metal slot sweep reported its thrashing signature in. Reported because a
// pager under pressure announces itself in VARIANCE before it does in the mean.
//
// IT IS APPLIED WITHIN A PROMPT, NEVER POOLED ACROSS THEM. Pooling would fold between-prompt variance (different lengths,
// different routing) into a number read as run-to-run noise, and the prompts here differ by design. Same rule as
// differencing matched observations instead of pooling them (CLAUDE.md, Measurement discipline).
func specPagerSpread(xs []float64) float64 {
	if len(xs) < 2 {
		return 0
	}
	lo, hi := xs[0], xs[0]
	for _, x := range xs {
		lo, hi = math.Min(lo, x), math.Max(hi, x)
	}
	if m := specPagerMean(xs); m > 0 {
		return (hi - lo) / m
	}
	return 0
}

// specPagerAlpha is committed tokens per verify round, the realized acceptance the speedup rides
// on. The two paths expose it through different seams and neither is optional here: the block path
// reports each round through OnRound, while the n-gram path has no such callback and reports
// SpecStats totals instead. Committed-per-round is (accepted drafts + the target's own token), so
// the n-gram form adds one per round — matching what OnRound's `committed` already counts.
func (a *specPagerArm) alpha() float64 {
	if len(a.committed) > 0 {
		s := 0
		for _, c := range a.committed {
			s += c
		}
		return float64(s) / float64(len(a.committed))
	}
	if a.rounds > 0 {
		return float64(a.accepted+a.rounds) / float64(a.rounds)
	}
	return 0
}

func TestSpecPagerInteraction(t *testing.T) {
	needsFreshProcess(t, "NewBlockSpec hit CUDA_ERROR_OUT_OF_MEMORY in the one-process heavy tier; passes alone")
	requireHeavyModel(t)

	// PARAMETERIZED OVER THE VENUE, because one paged MoE cannot answer the whole question. A Gated-DeltaNet MoE
	// (qwen3.6-35B-A3B) refuses both speculative arms for two unrelated reasons (the block path on the MoE batched-verify check
	// as of the experiment, the n-gram path on the recurrent-rollback check), so no throughput number exists to be had there.
	// gemma-4-26B-A4B is a paged MoE with NO recurrent state, so its n-gram arm actually runs: the venue where the throughput
	// claim can be tested rather than sidestepped.
	path := os.Getenv("GOINFER_SPECPAGER_MODEL")
	if path == "" {
		path = os.Getenv("GOINFER_QWEN36_35B")
	}
	if path == "" {
		home, _ := os.UserHomeDir()
		path = filepath.Join(home, "models", "qwen3.6-35b-a3b-int4.giw")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no 35B checkpoint at %s: %v", path, err)
	}
	// The archive is storage, never a read path for a timed row: a run off /srv/models measures a
	// 5400 rpm SMR disk instead of the engine, and it does not error — it returns a plausible,
	// wrong number. Refuse rather than produce a void row.
	if strings.HasPrefix(path, "/srv/models") || strings.HasPrefix(path, "/Volumes/") {
		t.Fatalf("checkpoint is on the archive (%s) — every timed row must read from local disk; run models-pull first", path)
	}
	t.Setenv("GOINFER_MOE_CACHE_EXPERTS", "1")
	// Gemma-4's resident path is env-gated; harmless on every other family.
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")

	// PROGRESS. The 20 GB pinned-host allocation inside Load runs for minutes with nothing to show
	// for it, and a passing Go package discards t.Logf and os.Stderr alike without -v; under -v
	// these stream as they happen while t.Logf is held until the function returns, which is the
	// difference between seeing a slow test and staring at a hung one.
	hb := func(f string, a ...any) {
		fmt.Fprintf(os.Stderr, "[spec-pager %s] %s\n", time.Now().Format("15:04:05"), fmt.Sprintf(f, a...))
	}

	hb("loading %s (pinned ~20 GB, expect ~5 min)", filepath.Base(path))
	t0 := time.Now()
	m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("Load(35B, cuda int4): %v", err)
	}
	defer m.Close()
	hb("loaded in %s (decode path %s)", time.Since(t0).Round(time.Second), m.DecodePath())

	rf := m.ResidentForwardForTest()
	if rf == nil {
		// Non-vacuity. Without this the test measures the CPU path and reports it as a paged-MoE
		// result — there would be no pager at all, and every demand number would be zero while the
		// wall-clock numbers stayed superficially reasonable.
		t.Fatalf("cuda resident DECLINED the 35B with C′ on — decode path %q; decline: %s",
			m.DecodePath(), m.ResidentDecline())
	}
	r := rf.(*cudaResident)
	if !r.cacheExperts {
		t.Fatal("resident built but C′ expert staging is OFF — nothing here would be measuring a pager")
	}
	slots := r.CacheSlotsForTest()
	req := os.Getenv("GOINFER_MOE_CACHE_SLOTS")
	hb("pager: %d slots/layer BUILT (requested %q), topK=%d", slots, req, r.topK)
	// A request BELOW topK is a HARD ERROR at Load time (the `t.Fatalf("Load(35B, cuda int4): %v", err)` above already ends
	// the test for one), so this line is never reached with n < r.topK. A request AT OR ABOVE topK is honoured UNIFORMLY, then
	// possibly capped: first to the model's own expert count, then to measured free VRAM by capSlots (cuda/resident.go). Either
	// cap is a real rung, just not the requested one; report the depth built (V-24, docs/review-2026-09-04.md).
	if n, e := strconv.Atoi(req); e == nil && n != slots {
		hb("NOTE: requested %d but BUILT %d — honoured (n=%d is at or above topK=%d) and then "+
			"capped, either to the model's expert count or to free VRAM by capSlots; this rung "+
			"is a real depth, just not the requested one", n, slots, n, r.topK)
	}

	tk, err := specPagerTokenizer(path)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	tmpl, err := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has})
	if err != nil {
		t.Fatalf("chat template: %v", err)
	}
	prompts := make([][]int, len(specPagerPrompts))
	for i, p := range specPagerPrompts {
		ids, e := tk.EncodeSegments(tmpl.RenderSegments("", []chat.Turn{{Role: "user", Content: p}}), false)
		if e != nil {
			t.Fatalf("encode %d: %v", i, e)
		}
		prompts[i] = ids
	}

	ddir := os.Getenv("GOINFER_SPECPAGER_DRAFTER")
	if ddir == "" {
		ddir = os.Getenv("GOINFER_QWEN36_DFLASH")
	}
	if ddir == "" {
		home, _ := os.UserHomeDir()
		ddir = filepath.Join(home, "models", "qwen36-35b-dflash")
	}
	// THE DRAFTER IS OPTIONAL, and only the BLOCK arms need it. A pretrained block drafter is
	// paired to one target (vocab, hidden, tap layers), so a venue with no matching drafter can
	// still run the off and n-gram arms — n-gram drafts from the prompt itself and is
	// model-agnostic. Requiring a drafter here would have excluded exactly the venues chosen for
	// being able to answer the throughput question, which is the opposite of what this measures.
	var spec *decoder.BlockSpec
	dr, derr := decoder.LoadDFlashDrafter(ddir)
	if derr != nil {
		hb("no block drafter at %s (%v) — block arms will be marked unavailable; n-gram arms still run", ddir, derr)
	} else {
		defer dr.Close()
		if !m.BlockSpecCapable() {
			t.Fatal("BlockSpecCapable() = false on a cuda resident")
		}
		if spec, err = m.NewBlockSpec(dr, dr.TargetLayerIDs()); err != nil {
			t.Fatalf("NewBlockSpec: %v", err)
		}
		hb("drafter attached: block %d, taps %v", dr.BlockSize(), dr.TargetLayerIDs())
	}

	const nNew, repeats = 64, 3
	// BOTH speculative paths, because they reach the backend differently and only one of them can
	// answer the throughput question.
	//
	//	block  BlockSpec verifies via PrefillLastNArgmax -> prefillCore, which has NO sequential
	//	       fallback, so a MoE decline is terminal for the request.
	//	ngram  GenerateNgramSpeculative verifies via ForwardN, which DOES fall back to a per-token
	//	       loop on an arch decline. So it RUNS on a MoE — at K sequential positions per round —
	//	       and is therefore the only arm on which the field report's throughput claim can be
	//	       given a number here at all. Omitting it would have closed the item on "speculation
	//	       cannot run", which is true of one path and false of the other.
	arms := []*specPagerArm{
		{label: "off", kind: "off", width: 0},
		{label: "blk4", kind: "block", width: 4},
		{label: "blk7", kind: "block", width: 7},
		{label: "ng4", kind: "ngram", width: 4},
		{label: "ng7", kind: "ngram", width: 7},
	}

	// WARM UP EVERY PROMPT, not just the first. A single global warm-up leaves the first repeat of each prompt paying a cold
	// pager (slowest every time, and a within-prompt spread far too noisy to resolve R1's 10% threshold against). The cold cost
	// is real but belongs to neither arm, and whichever arm runs first would otherwise absorb it.
	for pi := range prompts {
		hb("warm-up prompt %d (untimed)", pi)
		wch, _ := m.Generate(context.Background(), prompts[pi], 24, decoder.SamplingParams{Temperature: 0})
		for range wch { //nolint:revive // drain
		}
	}

	// PROBE FIRST. BlockSpecCapable() checks only that the resident implements ResidentDrafterHost;
	// it does not check that a batched verify exists for this ARCH. A four-token probe finds a
	// per-request decline in seconds instead of after the first timed arm, and lets the run record
	// it while still measuring everything a decline does not invalidate.
	var specDecline string
	if spec == nil {
		specDecline = "no block drafter paired with this target on this box (block arms not measurable here)"
		hb("block arms unavailable: %s", specDecline)
	} else if _, _, e := spec.Generate(prompts[0], decoder.BlockSpecOptions{VerifyWidth: 4, MaxTokens: 4}); e != nil {
		specDecline = e.Error()
		hb("SPEC PROBE DECLINED: %v", e)
		hb("recording the decline and measuring the off arm only (the ladder still runs, so the " +
			"decline can be shown slot-INDEPENDENT rather than assumed to be)")
	} else {
		hb("spec probe ok — the speculative arms are live")
	}

	total := len(arms) * len(prompts) * repeats
	done := 0
	runStart := time.Now()
	for _, a := range arms {
		a.secs = make([][]float64, len(prompts))
		a.tokens = make([][]int, len(prompts))
		if a.kind == "block" && specDecline != "" {
			a.declined = specDecline
			done += len(prompts) * repeats
			continue
		}
		for pi := range prompts {
			for rep := range repeats {
				r.ResetPagerStatsForTest()
				var got []int
				var dur time.Duration

				switch a.kind {
				case "off":
					// CHECK gen.Err(). Discarding it presents a real failure as "produced no tokens", indistinguishable from an empty
					// generation; an error swallowed here is an error attributed to the wrong mechanism.
					s := time.Now()
					ch, gen := m.Generate(context.Background(), prompts[pi], nNew, decoder.SamplingParams{Temperature: 0})
					for tok := range ch {
						got = append(got, tok)
					}
					dur = time.Since(s)
					if gen != nil {
						if e := gen.Err(); e != nil {
							a.declined = e.Error()
							hb("%s p%d r%d ERR after %d tok: %v", a.label, pi, rep, len(got), e)
							break
						}
					}
				case "ngram":
					s := time.Now()
					ch, gen, e := m.GenerateNgramSpeculative(context.Background(), prompts[pi], nNew,
						&decoder.NgramDrafter{}, a.width, decoder.SamplingParams{Temperature: 0})
					if e != nil {
						a.declined = e.Error()
						hb("%s p%d r%d DECLINED: %v", a.label, pi, rep, e)
						break
					}
					for tok := range ch {
						got = append(got, tok)
					}
					dur = time.Since(s)
					if gen != nil {
						if er := gen.Err(); er != nil {
							a.declined = er.Error()
							hb("%s p%d r%d ERR after %d tok: %v", a.label, pi, rep, len(got), er)
							break
						}
					}
					if gen != nil && gen.Spec != nil {
						a.rounds += gen.Spec.Rounds
						a.drafted += gen.Spec.Drafted
						a.accepted += gen.Spec.Accepted
					}
				default:
					opt := decoder.BlockSpecOptions{VerifyWidth: a.width, MaxTokens: nNew}
					opt.OnRound = func(w, c int) { a.committed = append(a.committed, c) }
					s := time.Now()
					ids, _, e := spec.Generate(prompts[pi], opt)
					dur = time.Since(s)
					if e != nil {
						// Not fatal: a refusal here is a result. Record it and move on so the
						// arms that CAN run still report.
						a.declined = e.Error()
						hb("%s p%d r%d DECLINED: %v", a.label, pi, rep, e)
						break
					}
					got = ids
				}
				if a.declined != "" {
					break
				}

				st, di := r.PagerStageStatsForTest()
				hi, mi := r.CacheStatsForTest()
				a.stages, a.distinct, a.hits, a.misses = a.stages+st, a.distinct+di, a.hits+hi, a.misses+mi
				if len(got) < nNew {
					// Record and carry on rather than aborting a 7-minute rung: a short run is
					// itself data, and the arms that DO complete still report. It is excluded from
					// the timing series because normalizing a truncated run to nNew would
					// extrapolate a rate from a generation that stopped for an unknown reason.
					hb("%s p%d r%d SHORT: %d/%d tokens — excluded from timings", a.label, pi, rep, len(got), nNew)
					a.short++
					continue
				}
				a.secs[pi] = append(a.secs[pi], dur.Seconds()/float64(len(got))*float64(nNew))
				a.tokens[pi] = append(a.tokens[pi], len(got))

				done++
				el := time.Since(runStart)
				hb("%s p%d r%d: %d tok in %.2fs (%.2f tok/s) | stages %d distinct/stage %.3f | %d/%d done, elapsed %s",
					a.label, pi, rep, len(got), dur.Seconds(), float64(len(got))/dur.Seconds(),
					st, float64(di)/math.Max(1, float64(st)), done, total, el.Round(time.Second))
			}
		}
	}

	specPagerReport(t, hb, arms, filepath.Base(path), slots, r.topK, nNew)
}

// specPagerTokenizer picks the loader for the container: three containers, three loaders, and the
// .giw arm is the default path here — handing a bundle to the HF JSON parser fails as
// `invalid character 'G'`, which reads like a corrupt checkpoint rather than a missing case.
func specPagerTokenizer(path string) (*tokenizer.Tokenizer, error) {
	switch {
	case strings.HasSuffix(path, ".giw"):
		// A .giw carries the tokenizer half of whatever it was BUILT FROM, which is not always a
		// GGUF: the 35B bundle was transcoded from GGUF and carries GGUF metadata, while the
		// gemma-4-26B bundle came from safetensors and carries HF JSON. Assuming one produced
		// `gguf: bad magic (not a GGUF file)` — which reads like a corrupt bundle rather than the
		// other valid format. Try both, and report both failures if neither parses.
		b, err := giw.ReadTokFile(path)
		if err != nil {
			return nil, err
		}
		tk, gerr := tokenizer.LoadGGUFBytes(b)
		if gerr == nil {
			return tk, nil
		}
		tk, jerr := tokenizer.LoadJSONBytes(b)
		if jerr == nil {
			return tk, nil
		}
		return nil, fmt.Errorf("giw tokenizer parsed as neither GGUF (%v) nor HF JSON (%v)", gerr, jerr)
	case strings.HasSuffix(path, ".gguf"):
		return tokenizer.LoadGGUF(path)
	default:
		return tokenizer.Load(path)
	}
}

// specPagerReport scores the arms against the pre-registered rules. It reports; it does not decide
// what the thresholds are — those were fixed before the run.
func specPagerReport(t *testing.T, hb func(string, ...any), arms []*specPagerArm, venue string, slots, topK, nNew int) {
	t.Helper()
	byLabel := map[string]*specPagerArm{}
	for _, a := range arms {
		byLabel[a.label] = a
	}
	off := byLabel["off"]

	t.Logf("=== spec x pager [%s], %d slots/layer, topK %d, %d new tokens/generation ===", venue, slots, topK, nNew)
	t.Logf("seconds are normalized to %d emitted tokens; per-arm raw token counts logged below", nNew)
	t.Logf("%-5s | %8s | %8s | %7s | %10s | %8s | %8s | %6s",
		"arm", "mean s", "med s", "wspread", "dist/stage", "hit%", "miss/stg", "alpha")
	for _, a := range arms {
		if a.declined != "" {
			t.Logf("%-5s | DECLINED — %s", a.label, a.declined)
			continue
		}
		var flat []float64
		var withinSpread []float64
		for _, ps := range a.secs {
			flat = append(flat, ps...)
			withinSpread = append(withinSpread, specPagerSpread(ps))
		}
		dps := float64(a.distinct) / math.Max(1, float64(a.stages))
		hitPct := 100 * float64(a.hits) / math.Max(1, float64(a.hits+a.misses))
		mps := float64(a.misses) / math.Max(1, float64(a.stages))
		alpha := a.alpha()
		t.Logf("%-5s | %8.3f | %8.3f | %6.1f%% | %10.4f | %7.1f%% | %8.3f | %6.3f",
			a.label, specPagerMean(flat), specPagerMedian(flat), 100*specPagerMean(withinSpread), dps, hitPct, mps, alpha)
		t.Logf("      raw token counts %v (short/excluded: %d)", a.tokens, a.short)
	}

	// R1 — PAIRED per (prompt, repeat), never a ratio of medians.
	for _, a := range arms {
		if a.kind == "off" || a.declined != "" {
			continue
		}
		var ratios []float64
		signAgree := true
		for pi := range a.secs {
			for rep := range a.secs[pi] {
				// PAIR ONLY WHERE BOTH ARMS HAVE THE ROUND. Short generations are dropped from an
				// arm's series, so the arms can differ in length per prompt; indexing off by this
				// arm's rep would pair round r of one arm against a DIFFERENT round of the other,
				// which is worse than dropping it — it silently reintroduces the unpaired
				// comparison the paired form exists to avoid.
				if pi >= len(off.secs) || rep >= len(off.secs[pi]) {
					continue
				}
				ratios = append(ratios, a.secs[pi][rep]/off.secs[pi][rep])
			}
		}
		if len(ratios) == 0 {
			t.Logf("R1 %s vs off: no paired rounds survived — not evaluable", a.label)
			continue
		}
		// Sign agreement across repeats, per the pre-registration: a verdict other than PARKED
		// requires every repeat to point the same way.
		med := specPagerMedian(ratios)
		for _, q := range ratios {
			if (q > 1) != (med > 1) {
				signAgree = false
			}
		}
		verdict := "AMBIGUOUS -> PARKED"
		switch {
		case med >= 1.10 && signAgree:
			verdict = "REAL REGRESSION"
		case med <= 1.02 && signAgree:
			verdict = "no regression"
		}
		alpha := a.alpha()
		// R3's prediction: if verify walks position by position, a width-K round costs K positions
		// and commits `alpha` tokens, so the wall-clock ratio should land at K/alpha.
		pred := float64(a.width) / math.Max(1e-9, alpha)
		t.Logf("R1 %s vs off: paired median %.4f (n=%d, sign agrees %v) -> %s", a.label, med, len(ratios), signAgree, verdict)
		t.Logf("R3 %s: alpha %.3f committed/round, H-noamort predicts K/alpha = %.4f; measured/predicted = %.4f (closes at 1.00 +/- 0.15)",
			a.label, alpha, pred, med/math.Max(1e-9, pred))
	}

	// R2 — the number that separates the two hypotheses.
	offD := float64(off.distinct) / math.Max(1, float64(off.stages))
	if offD < 0.98*float64(topK) || offD > 1.02*float64(topK) {
		t.Errorf("R2 INSTRUMENT INVALID: off arm reads %.4f distinct/stage against topK %d — the router picks topK "+
			"distinct experts, so the off arm must read topK. The instrument is wrong, not the hypothesis.", offD, topK)
	}
	for _, a := range arms {
		if a.kind == "off" || a.declined != "" {
			t.Logf("R2 %s: not evaluable (%s)", a.label,
				map[bool]string{true: "off arm is the baseline", false: "arm declined"}[a.kind == "off"])
			continue
		}
		d := float64(a.distinct) / math.Max(1, float64(a.stages))
		verdict := "AMBIGUOUS -> PARKED"
		switch {
		case d >= 10.0:
			verdict = "H-paging CONFIRMED"
		case d >= 0.98*float64(topK) && d <= 1.02*float64(topK):
			verdict = "H-paging REFUTED (pager sees decode-shaped traffic)"
		}
		t.Logf("R2 %s: %.4f distinct/stage vs off %.4f -> %s", a.label, d, offD, verdict)
	}
	hb("arms complete")
}
