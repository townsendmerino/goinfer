package decoder

import (
	"context"
	"errors"
	"regexp"
	"slices"
	"sync"
	"sync/atomic"

	"github.com/townsendmerino/goinfer/internal/runtime/logs"
)

// prefillDeclineDigitsRE normalizes a decline error's varying numbers (prompt length, floor, byte counts) out of the dedup
// key below, so every below-floor prompt collapses to the same reason instead of re-triggering the warning.
var prefillDeclineDigitsRE = regexp.MustCompile(`\d+`)

// prefillDeclineMu guards prefillDeclineSeen.
var prefillDeclineMu sync.Mutex

// prefillDeclineSeen is the per-reason dedup set warnPrefillDeclined reports through. It is keyed per normalized reason, not
// a process-lifetime sync.Once: a Once let the first decline of any kind (on Metal the routine below-floor case, which fires
// on nearly every short prompt) silence every later, different one an operator would want to see.
var prefillDeclineSeen = map[string]bool{}

// resetPrefillDeclineDedup clears the per-reason dedup state; test-only.
func resetPrefillDeclineDedup() {
	prefillDeclineMu.Lock()
	defer prefillDeclineMu.Unlock()
	prefillDeclineSeen = map[string]bool{}
}

// warnPrefillDeclined reports, once per distinct reason, that a backend's batched prefill refused a prompt at call time and
// the prompt (and every later one with the same reason) is being ingested one token at a time. It exists because
// PrefillPath() answers from static model properties, so the banner and /v1/models can say "batched" while a long prompt hits
// an M-dependent decline. Stderr and not an error: the fallback is correct, just slow, and failing the request over a
// performance decline would be worse than serving it.
func warnPrefillDeclined(n int, err error) {
	key := prefillDeclineDigitsRE.ReplaceAllString(err.Error(), "#")
	prefillDeclineMu.Lock()
	seen := prefillDeclineSeen[key]
	prefillDeclineSeen[key] = true
	prefillDeclineMu.Unlock()
	if seen {
		return
	}
	logs.Warnf("goinfer: batched prefill declined for a %d-token prompt, falling back to "+
		"the per-token path (slower TTFT; each distinct reason is reported once): %v", n, err)
}

// residentPrefillSeed ingests the prompt into the resident KV and returns the last token's logits, the seed for decode. Both
// resident generation paths call it (generateInto and genNgramInto); keep it one function, so an optimisation cannot land on
// one path and miss the other (docs/code-notes/decoder.md#Model.residentPrefillSeed.shared).
//
// from is the first position to compute: prompt[:from] is already committed to the resident KV (prefix reuse,
// resident_reuse.go) and positions carry through unchanged because the cache is positional. from == 0 is the cold path.
//
// Batched prefill (optional Prefiller) ingests the prompt in one pass, much faster TTFT for long prompts. It declines (falls
// back) past the backend's cap, is absent for a backend without a batched forward, and is skipped for tiny prompts. It is on
// by default and must stay bit-identical to sequential decode: every float MAC in both paths is an explicit __fmaf_rn,
// enforced by cuda.TestKernelFMALint and guarded by TestPrefillDivergenceRate. GOINFER_BATCHED_PREFILL=0 force-disables.
//
// hasAdapter must be true whenever this call runs under a bound resident adapter (cache.lora != nil at generateInto, the only
// caller where that is possible; GenerateVL, the n-gram spec target and GenerateSpeculative's target and draft never bind
// one). The batched Prefiller path is a separate encoded launch per backend that never reads the bound delta, so an adapter
// session that reached it would prefill the prompt's K/V from the base weights and only start applying the adapter at
// decode: plausible, wrong, HTTP 200. This mirrors the CPU rule: the canBatchN caller in forwardn.go declines batched prefill
// whenever cache.lora != nil, because compute-time LoRA is wired only into the sequential forward.
func (m *Model) residentPrefillSeed(ctx context.Context, prompt []int, from int, hasAdapter bool) ([]float32, error) {
	return m.residentPrefillSeedMarked(ctx, prompt, from, hasAdapter, nil)
}

// prefillSeedFrom is where a prefill seed of prompt starts: from, or 0 when from is outside the prompt (the seed token, whose logits
// start decode, is never skipped); a seed from 0 rewrites the resident's KV from the start (residentWritesFromZero).
func (m *Model) prefillSeedFrom(prompt []int, from int) int {
	if from < 0 || from >= len(prompt) {
		from = 0
	}
	if from == 0 {
		m.residentWritesFromZero()
	}
	return from
}

// cutMarks is the marks of mk's chain that split the prefill, earliest first: mk's earlier marks, and mk itself when it is a cut.
func cutMarks(mk *prefillMark) []*prefillMark {
	var out []*prefillMark
	for ; mk != nil; mk = mk.pre {
		if mk.cut {
			out = append([]*prefillMark{mk}, out...)
		}
	}
	return out
}

// residentPrefillSeedMarked is residentPrefillSeed that, given a mark, has a resident that can (PrefillMarkedAvailable) call it back between two
// passes of the batched prefill, at the mark's position: the prefix snapshot of a prompt whose tail after the stable section is long (kvsnap.go).
// A resident that cannot, a prefill that does not reach the position at a pass boundary, or one that falls to the sequential loop, calls nothing.
func (m *Model) residentPrefillSeedMarked(ctx context.Context, prompt []int, from int, hasAdapter bool, mk *prefillMark) ([]float32, error) {
	from = m.prefillSeedFrom(prompt, from)
	for _, c := range cutMarks(mk) {
		// A cut: prefill up to the point as a prompt of its own length, call the mark, and go on from it. Its rows are written between passes,
		// which is where a recurrent state exists.
		if _, err := m.residentPrefillSeedMarked(ctx, prompt[:c.q], from, hasAdapter, nil); err != nil {
			return nil, err
		}
		c.at(c.q)
		from = c.q
	}
	if mk != nil && mk.cut {
		mk = nil
	}
	suffix := prompt[from:]
	// Offer the batched path suffixes of 8 tokens or more (a shorter one is cheaper token by token) -- and, when
	// the backend's short continuation is exact (PrefillTailExact), any suffix of a prompt the cold run would have
	// offered whole, so a reused prompt takes the cold run's route and returns its bits.
	offer := len(suffix) >= 8
	if !offer && from > 0 && len(prompt) >= 8 && m.resPrefill().PrefillTailExact() {
		offer = true
	}
	if m.knobs.get(knobBatchedPrefill) != "0" && offer && !hasAdapter {
		if canStep(m.resident, stepBatch{Pass: passRows, Emb: stepProbe}) {
			embed := func(ids []int) [][]float32 {
				if embedRowsObserver != nil {
					embedRowsObserver(len(ids))
				}
				embs := make([][]float32, len(ids))
				for i, id := range ids {
					embs[i] = m.embedResident(id)
				}
				return embs
			}
			// A resident whose pass allocates scratch per row takes a long suffix in blocks (prefillBlockRows), each
			// continuing the last at its position, so the scratch and the host embeddings are a block's and not the
			// prompt's. The last pass keeps at least prefillTailMinRows rows. A block the pass declines leaves `from` at
			// its first row, or where its PrefillPartial says: everything before it is written, and the rest takes the sequential loop below.
			var perr error
			for B := m.prefillBlockRows(len(prompt)); B > 0 && perr == nil && len(prompt)-from >= B+m.prefillTailMinRows(); {
				if perr = ctx.Err(); perr != nil {
					break
				}
				if _, perr = m.step(stepBatch{Pass: passRows, Ctx: ctx, Emb: embed(prompt[from : from+B]), Pos: from}); perr == nil {
					from += B
				}
			}
			// startPos is why the batched path needs no change for reuse: it already
			// places the run at an offset.
			if perr == nil {
				var sr stepResult
				if mk != nil && canStep(m.resident, stepBatch{Pass: passRows, Emb: stepProbe, At: stepProbeAt}) {
					sr, perr = m.step(stepBatch{Pass: passRows, Ctx: ctx, Emb: embed(prompt[from:]), Pos: from, Mark: mk.q, At: mk.at})
				} else if canStep(m.resident, stepBatch{Pass: passRows, In: inRowsFn, N: 1, Rows: stepProbeRows}) {
					// The resident reads the suffix a chunk at a time.
					sr, perr = m.step(stepBatch{Pass: passRows, Ctx: ctx, In: inRowsFn, N: len(prompt) - from, Pos: from, Rows: m.suffixRows(prompt[from:])})
				} else {
					sr, perr = m.step(stepBatch{Pass: passRows, Ctx: ctx, Emb: embed(prompt[from:]), Pos: from})
				}
				if perr == nil {
					return sr.Logits, nil
				}
			}
			// A CANCELLED PREFILL IS NOT A DECLINE. Falling through to the sequential loop would
			// re-ingest the whole prompt the caller just abandoned — and that loop's own ctx check
			// would abort it one token later anyway, after another forward. Return it.
			if errors.Is(perr, context.Canceled) || errors.Is(perr, context.DeadlineExceeded) || errors.Is(perr, ErrKVPoolFull) {
				return nil, perr
			}
			from = prefillResume(from, perr) // the passes before the one that declined are written (PrefillPartial)
			warnPrefillDeclined(len(prompt)-from, perr)
			NotePrefillFallback(ctx, "sequential", perr.Error())
			// The rest runs on decode's kernels, which is not the class the KV manager was told this pass writes (and
			// some blocks may have run batched first): its rows name no producer, so no other request takes them.
			if m.kv != nil {
				m.kv.Unrecorded(m.kv.Bound())
			}
		}
	}
	// KV-only prefill: prompt[:-1] tokens need only their K/V in the cache — skip the LM head
	// (a big-vocab matmul + readback + softcap) on every prefill token but the last, whose
	// logits seed decode. Byte-identical (same layer chain → same KV → same last-token logits);
	// GOINFER_NO_KVONLY_PREFILL forces the full-logits prefill (A/B / escape hatch).
	useKV := m.canDecode(outNone) && m.knobs.get(knobNoKVOnlyPrefill) == ""
	var row [1][]float32 // the token's one-row batch, reused across the prompt
	var logits []float32
	var err error
	for i, id := range prompt {
		if i < from {
			continue // already in the cache at position i
		}
		// The resident prefill loop is the GPU-side twin of the batched CPU path's per-layer ctx check:
		// without it an abandoned client leaves the whole prompt streaming through the device.
		if err = ctx.Err(); err != nil {
			return nil, err
		}
		emb := m.embedResident(id)
		if useKV && i < len(prompt)-1 {
			row[0] = emb
			if _, err = m.step(stepBatch{Pass: passToken, Emb: row[:], Pos: i, Out: outNone}); err != nil {
				return nil, err
			}
			continue
		}
		if logits, err = m.resident.Forward(emb, i); err != nil {
			return nil, err
		}
	}
	return logits, nil
}

// residentPrefillSeedMRoPE is residentPrefillSeed's m-RoPE-aware sibling, used by GenerateQwenVL's image-reuse fast path
// (docs/multimodal.md), where decode past the (already-resident) image block needs pos+mropeDelta for the rotation, not plain
// pos. Deliberately simple (no batched or KV-only prefill): the reused-suffix case is an agent turn's short trailing
// extension, not a long cold prompt worth the batched machinery.
func (m *Model) residentPrefillSeedMRoPE(ctx context.Context, prompt []int, from, mropeDelta int) ([]float32, error) {
	from = m.prefillSeedFrom(prompt, from)
	var sr stepResult
	var err error
	var row [1][]float32
	for i := from; i < len(prompt); i++ {
		if err = ctx.Err(); err != nil {
			return nil, err
		}
		row[0] = m.embedResident(prompt[i])
		if sr, err = m.step(stepBatch{Pass: passToken, In: inMRoPE, Emb: row[:], Pos: i, RopePos: i + mropeDelta}); err != nil {
			return nil, err
		}
	}
	return sr.Logits, nil
}

// mc3Prefill prefills prompt[from:] into a batched generation's slot and returns the seed logits. A long suffix is taken
// in chunks while other generations are decoding (prefillChunkTokens; mc3_batch.go), each chunk in its own exclusive
// section with one decode step yielded between them; otherwise — a short suffix, nobody decoding, or a resident whose
// batched prefill declines — the rest goes through residentPrefillSeed in one section, exactly as before.
func (m *Model) mc3Prefill(ctx context.Context, mc3 *residentBatcher, slot int, prompt []int, from int) ([]float32, error) {
	if from < 0 || from >= len(prompt) {
		from = 0 // never skip the seed token (residentPrefillSeed's own rule)
	}
	ok := canStep(m.resident, stepBatch{Pass: passRows, Emb: stepProbe})
	C := m.opts.ResidentPrefillChunk
	// A chunk below the resident's kernel-class floor would put the first chunk on the other class from the
	// whole prompt's; raising the chunk to the floor keeps every pass on the class a whole prefill uses.
	if f := m.prefillKernelFloor(); f > 0 && len(prompt) >= f && C > 0 && C < f {
		C = f
	}
	chunk := ok && C > 0 && m.knobs.get(knobBatchedPrefill) != "0"
	// ride: each chunk is carried by a decode run, one ragged step with the decode rows pending then (rideChunk), instead of a
	// pass of its own followed by yieldToDecode.
	ride := chunk && mc3.ride != nil
	var logits []float32
	var err error
	// Cut a chunk only while a chunk plus prefillTailMinRows tokens remain, so the pass that ends the prefill has at least
	// that many tokens (residentPrefillSeed sends a suffix under 8 down the sequential path, whose numerics are
	// not the batched prefill's) and at most C + prefillTailMinRows - 1 — never the up-to-2C tail the first candidate
	// left, which was its longest pass (docs/measurements/chunked-prefill-2026-09-27.md).
	for chunk && len(prompt)-from >= C+m.prefillTailMinRows() && mc3.decoding() > 0 {
		if err = ctx.Err(); err != nil {
			return nil, err
		}
		end := from + C
		embs := make([][]float32, end-from)
		for i, id := range prompt[from:end] {
			embs[i] = m.embedResident(id)
		}
		if ride {
			c := &chunkReq{seq: ResidentRaggedSeq{Slot: slot, Pos: from, Embs: embs, Prefill: &ResidentRaggedPrefill{PromptLen: len(prompt)}}}
			if err = mc3.rideChunk(c); err == nil {
				from = end
				continue
			}
			// The ragged step did not take the chunk (a family it declines, a full KV pool): this chunk and the rest alternate.
			ride, err = false, nil
		}
		mc3.prefillExclusive(func() {
			if err = m.residentBind(slot); err == nil {
				_, err = m.step(stepBatch{Pass: passRows, Ctx: ctx, Emb: embs, Pos: from})
			}
		})
		if err != nil {
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				return nil, err
			}
			// The batched prefill declined (or failed) on this chunk: everything before `from` is already written, and so is
			// the part of the chunk a PrefillPartial names, so the remainder goes through the ordinary path below.
			from = prefillResume(from, err)
			break
		}
		from = end
		mc3.yieldToDecode()
	}
	err = nil
	mc3.prefillExclusive(func() {
		if err = m.residentBind(slot); err == nil {
			logits, err = m.residentPrefillSeed(ctx, prompt, from, false)
			// The seed may be the resident's shared host buffer (a Forward, or a prefill path that returns it), and the generation reads
			// it after this exclusive section, when another generation's token may be rewriting it: copy it while the resident is still
			// ours (see generateInto's mc3Logits).
			if err == nil {
				logits = slices.Clone(logits)
			}
		}
	})
	return logits, err
}

// warnResidentCap reports, once per model, that a prompt passed the resident's context and ran on the CPU path, which is
// correct but far slower to the first token; raising the resident context (-ctx) keeps such prompts on the device.
func (m *Model) warnResidentCap(why string) {
	if atomic.CompareAndSwapInt32(&m.capWarned, 0, 1) {
		logs.Warnf("decoder: %s, so it runs on the CPU (first time only; a larger resident context keeps it on the device)", why)
	}
}

// tryClaimResident attempts the exclusive claim on the model's single shared resident KV (the resBusy CAS), the one CAS site
// every resident-touching caller shares. It returns false when there is no resident backend or another generation holds the
// claim; the caller must then fall back to the CPU/staged path (distinct sequences still complete correctly, only resident
// speed is lost). On true the caller owns the claim and must release it with atomic.StoreInt32(&m.resBusy, 0), typically
// deferred, once its resident-touching work is done.
func (m *Model) tryClaimResident() bool {
	if m.resident == nil {
		return false
	}
	if m.batcher != nil { // Batched decode: the exclusive claim waits out no one — it simply fails while a holder has a slot
		return m.batcher.claimExclusive(&m.resBusy)
	}
	return atomic.CompareAndSwapInt32(&m.resBusy, 0, 1)
}

// generateInto is the shared prefill+decode loop behind Model.Generate and Session.Generate. It assumes cache already holds
// prompt[:prefillFrom] (0 for a fresh generation), prefills prompt[prefillFrom:] (always at least one token: the seed, whose
// last position's logits start the decode), then decodes up to maxTokens, streaming each id to out. It does not close out:
// the caller owns the channel so it can run post-generation bookkeeping (a Session reconciling its token list) before the
// consumer observes the close. commit, if non-nil, is called with each id once its forward has committed that position to
// the cache, the seam Session uses to track exactly what the cache holds. Terminal status lands on g.err.
// kvRowCeiling is the context size a CPU KV cache may grow to: the pinned or requested context when there is one, else the
// model's maximum positions (0 when neither is known).
func (m *Model) kvRowCeiling() int {
	if m.resCtxReq > 0 {
		return m.resCtxReq
	}
	return m.w.Cfg.MaxPositions
}
