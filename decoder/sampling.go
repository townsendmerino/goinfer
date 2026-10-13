package decoder

import "github.com/townsendmerino/goinfer/internal/sampling"

// The sampler's implementation lives in internal/sampling, a package that imports nothing from decoder, so its
// statistical tests are cached apart from decoder's and a decoder edit does not re-run them
// (docs/measurements/test-wait-2026-10.md §2). The public types stay defined here, so `go doc` and pkg.go.dev show
// their fields and methods. Each has a twin in internal/sampling with the same fields: the conversions below stop
// compiling if the two drift.

// SamplingParams controls next-token selection. Zero value (Temperature 0)
// is greedy/argmax — deterministic and the right default for parity tests.
type SamplingParams struct {
	Temperature float64 // 0 = greedy; >0 scales logits before softmax
	TopK        int     // 0 = disabled; keep the K highest-prob tokens
	TopP        float64 // 0 = disabled; nucleus — smallest set with cumprob ≥ TopP
	MinP        float64 // 0 = disabled; keep tokens with prob ≥ MinP·maxProb (a relative floor — the now-standard filter)
	Seed        int64   // RNG seed for reproducible sampling

	// Repetition controls discourage repeats by adjusting the logits of tokens
	// seen in the last RepeatLastN tokens (prompt + output — the generation loop
	// seeds the prompt). All three compose.
	//   RepeatPenalty    llama.cpp style: a positive logit is divided by it
	//                    (negative logits multiplied). 0 or 1 = disabled.
	//   PresencePenalty  OpenAI style: flat subtraction if the token is present.
	//   FrequencyPenalty OpenAI style: subtraction proportional to its count.
	RepeatPenalty    float64
	PresencePenalty  float64
	FrequencyPenalty float64
	RepeatLastN      int // penalty window: 0 (or <0) = whole history, n = last n tokens

	// LogitBias adds a per-token offset to that token's logit before sampling
	// (large positive to force, large negative to ban). Keyed by token id.
	LogitBias map[int]float32

	// Logprobs makes SampleWithInfo / Generate report the chosen token's
	// log-probability; TopLogprobs additionally reports that many
	// highest-probability alternatives. Reported over the full-vocab softmax
	// after bias+penalties, at the sampling temperature (1 when greedy).
	Logprobs    bool
	TopLogprobs int

	StopIDs []int // extra ids that end generation (besides config EOS), e.g. <end_of_turn>
	// LogitProcessor, if non-nil, is called each decode step with the ids
	// generated so far and that step's logits, which it may modify in place
	// before the sampler runs — e.g. masking disallowed tokens to -inf for
	// constrained/structured decoding (see the constrain package). It runs after
	// the forward pass and before sampling and the stop check, so a constraint
	// can also gate EOS (mask it until the output is a complete document).
	LogitProcessor func(generated []int, logits []float32)
	// LogitProcessorGate, when set with LogitProcessor, makes the processor lazy: it is asked after every emitted token
	// (with the ids generated so far) whether the next step needs the processor. While it answers false the processor is
	// not called and the step keeps every on-device fast path (greedy argmax, device sampling, device top-K) as if there
	// were no processor; a step it answers true for reads the full logits and runs the processor. It is for a constraint
	// that switches on partway through a turn (a tool call under tool_choice auto, constrain.LazyMasker): an ungated
	// processor costs every token of the turn its fast path, prose included
	// (docs/measurements/tool-union-2026-09-24.md). The gate must be a pure function of generated ids. It is consulted
	// only on the decode loop in model.go; the speculative paths still refuse any LogitProcessor, gated or not, and
	// callers fall back to Generate.
	LogitProcessorGate func(generated []int) bool
}

// TokenLogprob is a token id paired with its log-probability.
type TokenLogprob struct {
	ID      int
	Logprob float64
}

// SampleInfo is a sample's result: the chosen id, plus — when
// SamplingParams.Logprobs is set — its log-probability and the top alternatives.
type SampleInfo struct {
	ID      int
	Logprob float64        // log P(ID); 0 unless Logprobs is set
	Top     []TokenLogprob // up to TopLogprobs entries, prob-descending; nil unless requested
}

// HistoryDependent reports whether any history-dependent logit transform is configured: repetition, presence or
// frequency penalties or LogitBias. Greedy speculative decoding verifies with argmax over the target's raw logits and
// cannot honor these, so its validators reject them to keep "token-identical to plain greedy" true. The sampled
// speculative path threads them per position instead (see needsHistory, which mirrors this check).
func (p SamplingParams) HistoryDependent() bool { return sampling.SamplingParams(p).HistoryDependent() }

// Sampler turns a logit vector into a token id. It owns its RNG so a run is
// reproducible from Seed, and tracks the token history the penalties read.
type Sampler struct {
	impl *sampling.Sampler
}

// NewSampler builds a sampler from params.
func NewSampler(p SamplingParams) *Sampler {
	return &Sampler{impl: sampling.NewSampler(sampling.SamplingParams(p))}
}

func sampleInfoFrom(si sampling.SampleInfo) SampleInfo {
	out := SampleInfo{ID: si.ID, Logprob: si.Logprob}
	if si.Top != nil {
		out.Top = make([]TokenLogprob, len(si.Top))
		for i, t := range si.Top {
			out.Top[i] = TokenLogprob(t)
		}
	}
	return out
}

// Observe seeds the penalty history with already-seen tokens (the generation
// loop calls it once with the prompt) so repetition penalties consider them.
// Each Sample then records the token it draws.
func (s *Sampler) Observe(ids ...int) {
	s.impl.Observe(ids...)
}

// Sample returns the chosen token id for the given logits ([VocabSize]).
//
//   - Temperature ≤ 0 is greedy (argmax) — deterministic, ignores the filters.
//   - Temperature > 0: softmax at that temperature, optionally restricted to
//     top-k ∩ top-p ∩ min-p, then a multinomial draw from the seeded RNG.
//
// LogitBias and the repetition/presence/frequency penalties adjust the logits
// first (on a copy — the caller's slice is untouched when any apply).
func (s *Sampler) Sample(logits []float32) (int, error) {
	return s.impl.Sample(logits)
}

// ArgmaxEquivalent reports whether this sampler's pick is exactly Argmax(raw logits): no logit bias, no penalties, no
// temperature or filtering, and no logprob reporting that would need the distribution. When true (and
// SamplingParams.LogitProcessor is nil), a backend that can compute the argmax on-device may return just the id and
// skip the full-logits readback; the emitted tokens are identical.
//
// This and GreedyEquivalent live next to every logit-touching parameter on purpose: a new parameter that mutates or
// reads logits must be reflected in both, so a fast path can never silently diverge. They are separate predicates,
// not one widened one, so call sites choose which notion of "deterministic" they need (see GreedyEquivalent).
func (s *Sampler) ArgmaxEquivalent() bool {
	return s.impl.ArgmaxEquivalent()
}

// GreedyEquivalent reports whether this sampler's pick is deterministically the argmax token even though a
// temperature is set: the top_k=1 shape. It is true at any temperature: scaling is strictly monotone and preserves
// the logit ordering, and a distribution restricted to one token is deterministic. top_p and min_p at any value are
// safe alongside it, because both cuts keep at least one token (topFilterLogits), so the retained set is exactly the
// top-1. The tie-break agrees: topFilterLogits orders ties by ascending id and argmax uses a strict >, both selecting
// the lowest tied index, which is the contract the device argmax is aligned to (cuda.TestArgmaxTieBreak).
//
// It uses HistoryDependent, not penaltiesActive: penaltiesActive depends on the observed history, so it is false
// before the first Observe and true after, and a backend consults this predicate once per request, before any token
// is observed, and would latch the wrong answer. HistoryDependent asks whether penalties or bias are configured,
// which is stable. Logprobs is excluded as in ArgmaxEquivalent: reporting a distribution needs the logits the fast
// path skips.
func (s *Sampler) GreedyEquivalent() bool {
	return s.impl.GreedyEquivalent()
}

// SampleWithInfo is Sample plus log-probability reporting (when SamplingParams.Logprobs is set): the chosen token's
// logprob and, if TopLogprobs > 0, that many highest-probability alternatives.
func (s *Sampler) SampleWithInfo(logits []float32) (SampleInfo, error) {
	si, err := s.impl.SampleWithInfo(logits)
	return sampleInfoFrom(si), err
}

// NextDraw returns the (seed, draw index) of the next temperature-only draw and advances it. It is the
// hook a device sampler uses so its draw i is the SAME draw the host would have made: whichever path
// serves a step, the stream is a function of (seed, step, logits) only.
func (s *Sampler) NextDraw() (seed, draw uint64) {
	return s.impl.NextDraw()
}

// SampleEligible reports whether this sampler's configuration can be drawn on-device by ResidentSample:
// temperature-only sampling with nothing that needs or rewrites the full row (logprobs, bias, penalties). The
// decode loop additionally requires no LogitProcessor.
func (s *Sampler) SampleEligible() bool {
	return s.impl.SampleEligible()
}

// TopKEligible reports whether this sampler's configuration can be served from a device top-K: a
// filtered sampler (top-k, min-p or a real top-p) at temperature > 0, with nothing that needs or
// rewrites the full row. The decode loop additionally requires no LogitProcessor.
func (s *Sampler) TopKEligible() bool {
	return s.impl.TopKEligible()
}

// TopKWidth returns how many candidates to request for a vocab of the given size, and false when the
// configuration cannot be served (an explicit top-k wider than the kernel's maximum).
func (s *Sampler) TopKWidth(vocab int) (int, bool) {
	return s.impl.TopKWidth(vocab)
}

// SampleFromTopK draws a token from a device top-K row exactly as SampleWithInfo would from the full
// row, or reports ok=false — without consuming any randomness — when the K candidates cannot prove
// they hold the whole retained set. The caller then reads the full row and calls SampleWithInfo.
// vocab is the full row's length.
func (s *Sampler) SampleFromTopK(row TopKRow, vocab int) (SampleInfo, bool) {
	si, ok := s.impl.SampleFromTopK(sampling.TopKRow(row), vocab)
	return sampleInfoFrom(si), ok
}

func (s *Sampler) distVector(logits []float32) []float64 { return s.impl.DistVector(logits) }

func (s *Sampler) distVectorHist(logits []float32, history []int) []float64 {
	return s.impl.DistVectorHist(logits, history)
}

func (s *Sampler) drawTarget(logits []float32, history []int) int {
	return s.impl.DrawTarget(logits, history)
}

func (s *Sampler) needsHistory() bool { return s.impl.NeedsHistory() }

func (s *Sampler) specStep(p []float64, x int) (int, bool) { return s.impl.SpecStep(p, x) }

func (s *Sampler) topPActive() bool { return s.impl.TopPActive() }

func argmax(logits []float32) int { return sampling.Argmax(logits) }

func softmaxStable(logits []float32, temperature float64) []float64 {
	return sampling.SoftmaxStable(logits, temperature)
}
