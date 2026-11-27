package decoder

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"

	"github.com/townsendmerino/goinfer/internal/runtime/vlprep"
)

func (m *Model) prefillLogitsGemma4VL(ctx context.Context, ids []int, imageEmbeds []float32, imgPos, imgLen int, cache *KVCache) ([]float32, error) {
	return m.prefillLogitsGemma4VLSpans(ctx, ids, imageEmbeds, []ImageSpan{{Pos: imgPos, Len: imgLen}}, cache)
}

func (m *Model) prefillLogitsGemma4VLSpans(ctx context.Context, ids []int, imageEmbeds []float32, spans []ImageSpan, cache *KVCache) ([]float32, error) {
	e := m.cpuEngine()
	return vlprep.PrefillGemma4VLSpans(ctx, &e, ids, imageEmbeds, spans, unwrapCache(cache))
}

func (m *Model) prefillLogitsGemma4VLBidirectional(ctx context.Context, ids []int, imageEmbeds []float32, imgPos, imgLen int, cache *KVCache) ([]float32, error) {
	return m.prefillLogitsGemma4VLBidirectionalSpans(ctx, ids, imageEmbeds, []ImageSpan{{Pos: imgPos, Len: imgLen}}, cache)
}

func (m *Model) prefillLogitsGemma4VLBidirectionalSpans(ctx context.Context, ids []int, imageEmbeds []float32, spans []ImageSpan, cache *KVCache) ([]float32, error) {
	e := m.cpuEngine()
	return vlprep.PrefillGemma4VLBidirectionalSpans(ctx, &e, ids, imageEmbeds, spans, unwrapCache(cache))
}

// GenerateGemma4VL streams a continuation for a Gemma 4 multimodal prompt, in the shape of GenerateVL and
// GenerateQwenVL.
//
// Prefill dispatches between two forwards depending on the checkpoint: sequential (prefillLogitsGemma4VL, causal) or
// batched (prefillLogitsGemma4VLBidirectional, use_bidirectional_attention: "vision"). Decode is always the unchanged
// per-token path: a decode token is never inside the image block again, so no masking distinction applies after prefill.
//
// Resident GPU decode (docs/multimodal.md): when the model has a resident, the CPU prefill's KV is uploaded through the
// generic residentUploadPrefill bridge (the one GenerateVL uses) and decode runs resident. The bridge skips KV-shared
// layers, which own no KV, and embedResident carries the PLE inputs, so the E-model class is admitted wherever a backend
// can run it (gemma4ImageDecodeResident holds the rule); a backend that declines the E-model shape at load keeps decoding
// on the CPU after an image. A causal, non-E Gemma 4 also keeps decoding on the CPU after an image: widening to it is
// untested on CUDA.
//
// imgHash, imgPos and imgLen are also what the resident commit records as this turn's residentImageBlock, so a later
// turn's prefix scan (residentReuseLen) can tell this image's span from ordinary text: Gemma 4's soft-token placeholder
// ids are content-independent, so two different images of the same size produce identical ids, and without a block
// record the scan cannot tell them apart.
func (m *Model) GenerateGemma4VL(ctx context.Context, ids []int, imgPos, imgLen int, imgHash uint64, features func() ([]float32, error), maxTokens int, sp SamplingParams) (<-chan int, *Generation) {
	return m.GenerateGemma4VLSpans(ctx, ids, []ImageSpan{{Pos: imgPos, Len: imgLen, Hash: imgHash}}, features, maxTokens, sp)
}

// GenerateGemma4VLSpans is GenerateGemma4VL for several images: spans in prompt order, features every span's rows
// concatenated in span order. A bidirectional checkpoint gives each span its own block; an E-model is causal, so its
// resident prefill takes several images unchanged.
func (m *Model) GenerateGemma4VLSpans(ctx context.Context, ids []int, spans []ImageSpan, features func() ([]float32, error), maxTokens int, sp SamplingParams) (<-chan int, *Generation) {
	out := make(chan int)
	g := &Generation{}
	m.runGen(ctx, out, g, nil, func(ctx context.Context) {
		if len(spans) == 0 {
			g.err = fmt.Errorf("decoder: GenerateGemma4VLSpans needs at least one image span")
			return
		}
		feats, err := features()
		if err != nil {
			g.err = err
			return
		}
		bidirectional := m.w.Cfg.UseBidirectionalAttention != ""
		eModel := gemma4IsEModel(m.w.Arch)
		// An E-model's or a bidirectional checkpoint's image turn prefills on the resident when the backend takes it, from the
		// rows the CPU path would run (gemma4VLResidentPrefill). A decline falls through to the CPU prefill and upload below,
		// unchanged.
		if eModel != bidirectional {
			if done := m.gemma4VLResidentPrefill(ctx, out, g, ids, feats, spans, maxTokens, sp); done {
				return
			}
		}
		cache := m.NewCache(len(ids) + maxTokens)
		var logits []float32
		if bidirectional {
			logits, err = m.prefillLogitsGemma4VLBidirectionalSpans(ctx, ids, feats, spans, cache)
		} else {
			logits, err = m.prefillLogitsGemma4VLSpans(ctx, ids, feats, spans, cache)
		}
		if err != nil {
			g.err = err
			return
		}

		useGPU := false
		gpuPos := 0
		committed := false
		if gemma4ImageDecodeResident(bidirectional, eModel) && m.tryClaimResident() { // support_table.go reads the same rule
			// The resident cache is about to hold this turn's content. If decode completes naturally, residentCommitIDs below
			// records it and this defer's forget is skipped (committed=true); any other exit (error, cancel, or the upload never
			// engaging) forgets unconditionally, as in GenerateVL's upload bridge.
			defer func() {
				if !committed {
					m.resHost().ForgetIDsIn(kvScopeOf(ctx))
				}
				atomic.StoreInt32(&m.resBusy, 0)
			}()
			if uerr := m.residentUploadPrefill(cache); uerr == nil {
				useGPU = true
				g.DecodeResident = true
				gpuPos = len(ids)
				if c, clamped := m.clampToResidentCap(gpuPos, maxTokens); clamped {
					maxTokens, g.BudgetClamped = c, true
				}
			}
		}

		g.Budget = maxTokens // publish the effective (possibly clamped) budget so a cap-truncated turn reports finish_reason "length"
		sampler := NewSampler(sp)
		sampler.Observe(ids...)
		generated := m.vlDecodeLoop(ctx, out, g, sampler, maxTokens, sp, logits, func(next int) ([]float32, error) {
			if useGPU {
				l, err := m.resident.Forward(m.embedResident(next), gpuPos)
				gpuPos++
				return l, err
			}
			// Decode reuses the plain per-token path unmodified: m.forward already
			// runs the Gemma 4 program walk, so no
			// gemma4-specific decode step is needed once the image is prefilled.
			return m.forward(next, cache)
		})
		if useGPU && g.err == nil {
			// Commit the image block the way GenerateVL and GenerateQwenVL do, so residentReuseLen's prefix scan has something to
			// key a later turn's reuse check on, instead of a plain id comparison that Gemma 4's content-independent soft-token ids
			// can satisfy by coincidence.
			m.resHost().CommitIDs(ids, generated, residentImageBlocksOf(spans), 0)
			committed = true
		}
	})
	return out, g
}

// gemma4VLResidentPrefill is GenerateGemma4VL's resident image prefill, then resident decode. An E-model (causal) runs
// the whole prompt through the backend's Prefiller; a bidirectional non-E checkpoint through its SpanPrefillAvailable,
// with each span its own block. It reports whether it ran the turn (true: the generation is done, g holds its result);
// false means the backend has no such prefill, the resident is busy, or the prefill declined, and the caller runs the
// CPU path. A cancelled prefill is not a decline: it ends the turn.
func (m *Model) gemma4VLResidentPrefill(ctx context.Context, out chan<- int, g *Generation, ids []int, feats []float32, spans []ImageSpan, maxTokens int, sp SamplingParams) bool {
	if m.knobs.get(knobBatchedPrefill) == "0" {
		return false
	}
	var prefill func(rows [][]float32) ([]float32, error)
	build := func(ids []int, feats []float32, spans []ImageSpan) ([][]float32, error) {
		e := m.cpuEngine()
		return vlprep.Gemma4EModelImageRows(m.w.Arch, m.BackendView().EmbedLen(), e.Gemma4PLEInputs, m.embedResident, ids, feats, spans, gemma4VLRowsDefectForTest)
	}
	if m.w.Cfg.UseBidirectionalAttention != "" {
		if !canStep(m.resident, stepBatch{Pass: passRows, In: inSpans, Emb: stepProbe}) || gemma4IsEModel(m.w.Arch) {
			return false
		}
		blocks := imageSpanBlocks(spans)
		prefill = func(rows [][]float32) ([]float32, error) {
			sr, err := m.step(stepBatch{Pass: passRows, Ctx: ctx, In: inSpans, Emb: rows, Spans: blocks})
			return sr.Logits, err
		}
		build = func(ids []int, feats []float32, spans []ImageSpan) ([][]float32, error) {
			return vlprep.Gemma4BidirImageRows(m.w.Arch, m.program().Embed.Scale, m.embedN, ids, feats, spans, gemma4VLRowsDefectForTest)
		}
	} else {
		if !canStep(m.resident, stepBatch{Pass: passRows, Emb: stepProbe}) {
			return false
		}
		prefill = func(rows [][]float32) ([]float32, error) {
			sr, err := m.step(stepBatch{Pass: passRows, Ctx: ctx, Emb: rows})
			return sr.Logits, err
		}
	}
	if !m.tryClaimResident() {
		return false
	}
	rows, err := build(ids, feats, spans)
	if err != nil { // a malformed image run is the CPU path's to report; it is not this one's to panic on
		atomic.StoreInt32(&m.resBusy, 0)
		return false
	}
	m.resHost().ForgetIDsIn(kvScopeOf(ctx)) // the prefill overwrites the resident cache from position 0
	m.resHost().WritesFromZero()
	logits, err := prefill(rows)
	if err != nil {
		atomic.StoreInt32(&m.resBusy, 0)
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			g.err = err
			return true
		}
		return false
	}
	g.ImgPrefillResident, g.DecodeResident = true, true
	gpuPos := len(ids)
	if c, clamped := m.clampToResidentCap(gpuPos, maxTokens); clamped {
		maxTokens, g.BudgetClamped = c, true
	}
	g.Budget = maxTokens
	sampler := NewSampler(sp)
	sampler.Observe(ids...)
	generated := m.vlDecodeLoop(ctx, out, g, sampler, maxTokens, sp, logits, func(next int) ([]float32, error) {
		l, err := m.resident.Forward(m.embedResident(next), gpuPos)
		gpuPos++
		return l, err
	})
	if g.err == nil {
		m.resHost().CommitIDs(ids, generated, residentImageBlocksOf(spans), 0)
	} else {
		m.resHost().ForgetIDsIn(kvScopeOf(ctx))
	}
	atomic.StoreInt32(&m.resBusy, 0)
	return true
}

// gemma4VLRowsDefectForTest is a planted-defect selector for Gemma 4 resident row builders (0 = none; 1: placeholder ID
// for PLE, 2: feature scaled by embed scale, 3: zero PLE tail, 4: image positions built as text rows). Set only through
// SetGemma4VLRowsDefectForTest (goinfer_testhooks).
var gemma4VLRowsDefectForTest int
