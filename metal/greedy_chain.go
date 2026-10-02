//go:build darwin

package metal

import (
	"fmt"
	"math"
	"os"
	"runtime"

	"golang.org/x/sys/unix"
)

// The greedy chain: C-B01 of docs/audit-metal-2026-09-30.md (Phase 3 item 5 of docs/tasks/task-metal-audit-2026-10.md).
//
// The pipelined executor (execLoop) pre-encodes token t+1 while t runs, but cannot commit it until t finishes: t+1's
// input is the embedding of the token t's logits pick, and the host picks it. So between t's last kernel and t+1's
// first the GPU waits for the host: completion, 608 KB of logits, the decoder's argmax, the embedding, the commit
// (T1.3 measured that idle gap at 0.58-0.66 ms per token on the 0.5B and 1.5B). For greedy decoding the pick needs no
// host: each chained command buffer ends with the fused argmax head ForwardArgmax uses (gemv_w8a8_amax then
// argmax_finish, writing a 4-byte id) and starts with embed_gather_i8, which writes that id's embedding into r.x from
// the int8 embedding table on the device (chainEmbedTable: the LM-head buffers for a tied head, else a copy made on
// first use). So t+1 can be committed before t finishes, and the queue runs it right after.
//
// In flight: chainNext commits the buffer after the newest before it waits for the oldest, so the next token is always
// queued when the host waits, and one buffer is outstanding between calls. Token t binds uniform set t%2 and writes its
// id to chainTok[t%2]; t+1 reads chainTok[t%2] and writes chainTok[(t+1)%2]. Buffers are hazard-tracked (aikit's
// default) and the encoder is serial, so Metal orders a buffer's reads behind the previous buffer's writes; the host
// rewrites set k only for token t+2, after it has waited for token t, the set's last user. Every chainDrainEvery
// buffers the chain waits with nothing queued and drains its autorelease pool (one gap), as execLoop does.
//
// Bit-identical to the full-logits path by construction: the same trunk dispatches with the same values per token, the
// argmax ForwardArgmax uses (first maximum wins, as the decoder's), and the host embedding's arithmetic (Embed.Row's
// float32(q)*scale). TestGreedyChain_bitIdentical checks it. A chain stopped after an EOS has already run one forward
// past it; the K/V it wrote sits past the generation's end and is overwritten before anything reads it.
const chainDrainEvery = 64

const (
	chainOpStart = iota
	chainOpNext
	chainOpStop
)

type chainReq struct {
	op, id, pos int
	sampled     bool    // a temperature-only chain (chainStartSampled), or a sampled chainNext
	invT        float32 // start, sampled: 1/temperature
	seed, draw  uint64  // start: the first forward's draw; next: the decoder's draw for the forward it asks for
}

type chainResp struct {
	id  int
	err error
}

// greedyChainOff, when true, keeps the chain unavailable. FALSE in production: set only by tests that compare the chain
// with the path it replaces.
var greedyChainOff = false

// greedyChainWhyNot is why this resident cannot run the greedy chain ("" if it can). It covers what the GPU side needs
// apart from the gather table, which chainEmbedTable prices and makes; the decoder adds what only it knows (its
// embedding lookup has no multiplier, no adapter, no MC3).
func (r *resident) greedyChainWhyNot() string {
	switch {
	case greedyChainOff:
		return "turned off (test)"
	case r.embedScale > 1:
		return "an embedding scale (embed_gather_i8 applies none)"
	case r.learnedPos:
		return "learned position embeddings (added on the host)"
	case r.V%8 != 0:
		return "a vocabulary that is not a multiple of 8 (gemv_w8a8_amax's grid; ForwardArgmax's own exclusion)"
	case (r.g4moe != nil && r.g4moe.paged) || (r.moe != nil && r.moe.paged):
		return "paged experts (each MoE layer is its own submit, with a host readback between)"
	case r.loraLayers != nil:
		return "an adapter bound"
	case r.dnet != nil:
		return "recurrent state: a forward past EOS cannot be undone"
	}
	for _, L := range r.layers {
		if L.delta != nil {
			return "recurrent state: a forward past EOS cannot be undone"
		}
	}
	return ""
}

// chainEmbedTable is the int8 table and per-row scales the chain gathers the next token's embedding from, or why there
// is none. A tied head's LM-head buffers are the embedding table. Otherwise the table is put on the device the first
// time a chain is asked for (never for a resident that only samples) and kept until Close. It is priced first against
// the memory guard's budget, on top of the build's own price, and against the memory available now. The price is the
// whole table even where it aliases the .giw mapping, because Metal wires the pages a command buffer touches, which is
// why the guard counts every weight. A table that does not fit declines the chain, and decode keeps the full-logits path.
func (r *resident) chainEmbedTable() (Buffer, Buffer, string) {
	if r.lmTied {
		return r.lmW, r.lmS, ""
	}
	r.chainEmbMu.Lock()
	defer r.chainEmbMu.Unlock()
	if !r.chainEmbTried {
		r.chainEmbTried = true
		r.chainEmbW, r.chainEmbS, r.chainEmbWhy = r.buildChainEmbed()
		if r.chainEmbWhy != "" {
			fmt.Fprintf(os.Stderr, "metal: greedy chain off for this model: %s\n", r.chainEmbWhy)
		}
	}
	return r.chainEmbW, r.chainEmbS, r.chainEmbWhy
}

func (r *resident) buildChainEmbed() (w, s Buffer, why string) {
	q8, sc, _, ok := r.embed.Int8()
	if !ok {
		return Buffer{}, Buffer{}, fmt.Sprintf("an embedding table of kind %q (embed_gather_i8 reads int8)", r.embed.Kind())
	}
	need := int64(len(q8) + 4*len(sc))
	const mb = 1 << 20
	if !r.chainEmbGuardOff {
		if ram, err := unix.SysctlUint64("hw.memsize"); err == nil && ram > 0 {
			if ceil := metalStaticCeiling(ram); r.chainEmbBase+need > ceil {
				return Buffer{}, Buffer{}, fmt.Sprintf("the untied head's device embedding table (%.0f MB) would take this resident (%.0f MB) over the memory guard's %.0f MB",
					float64(need)/mb, float64(r.chainEmbBase)/mb, float64(ceil)/mb)
			}
		}
		if live := metalLiveAvailable(); live > 0 && need > live {
			return Buffer{}, Buffer{}, fmt.Sprintf("the untied head's device embedding table (%.0f MB) is more than the %.0f MB available now",
				float64(need)/mb, float64(live)/mb)
		}
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	pool := NewARPool()
	defer pool.Drain()
	defer func() {
		if p := recover(); p != nil { // mustBuf panics on a failed allocation, as buildResident's callers recover
			w, s, why = Buffer{}, Buffer{}, fmt.Sprintf("allocating the untied head's device embedding table failed: %v", p)
		}
	}()
	var err error
	if w, s, err = int8BufA(r.d, r.alias, r.embed); err != nil {
		return Buffer{}, Buffer{}, err.Error()
	}
	return w, s, ""
}

// chainStart opens a greedy chain whose first input is token id at position pos, and commits that token's buffer.
func (r *resident) chainStart(id, pos int) error {
	return r.chainOpen(chainReq{op: chainOpStart, id: id, pos: pos})
}

// chainStartSampled opens a temperature-only chain (C-P02): each buffer ends with ForwardSample's draw (the full
// LM-head row, then gumbel_stage1 and gumbel_stage2) instead of the argmax head. The forward at position pos draws with
// (seed, draw), and each later one with the next draw, which is how decoder.Sampler.NextDraw advances (the seed fixed,
// the draw counting up). chainNextSampled checks that against the decoder's own draw for every token.
func (r *resident) chainStartSampled(id, pos int, temperature float64, seed, draw uint64) error {
	invT := float32(1 / temperature)
	if math.IsInf(float64(invT), 0) || math.IsNaN(float64(invT)) {
		return fmt.Errorf("metal: no sampled chain at temperature %g (ForwardSample takes the argmax there)", temperature)
	}
	if !r.SampleAvailable() {
		return fmt.Errorf("metal: no sampled chain on this resident: no device draw (SampleAvailable)")
	}
	return r.chainOpen(chainReq{op: chainOpStart, id: id, pos: pos, sampled: true, invT: invT, seed: seed, draw: draw})
}

// chainOpen starts the chain goroutine and its first buffer. Every other resident entry point stops the chain first
// (stopExec calls stopChain), so while one is open nothing else uses r.q.
func (r *resident) chainOpen(q chainReq) error {
	if why := r.greedyChainWhyNot(); why != "" {
		return fmt.Errorf("metal: no greedy chain on this resident: %s", why)
	}
	gw, gs, why := r.chainEmbedTable()
	if why != "" {
		return fmt.Errorf("metal: no greedy chain on this resident: %s", why)
	}
	r.stopExec() // the encode-ahead executor and any earlier chain
	r.chainGW, r.chainGS = gw, gs
	if r.chainTok[0] == (Buffer{}) {
		for k := range r.chainSets {
			r.chainTok[k] = NewBufferU32(r.d, 0)
			r.chainSets[k] = posUniforms{NewBufferU32(r.d, 0), NewBufferU32(r.d, 1), NewBufferU32(r.d, 0),
				NewBufferFloats(r.d, []float32{1}), NewBufferU32(r.d, 0)}
		}
	}
	if q.sampled && r.chainLogits[0] == (Buffer{}) {
		r.chainInvT, r.chainK0, r.chainK1 = NewBufferFloats(r.d, []float32{1}), NewBufferU32(r.d, 0), NewBufferU32(r.d, 0)
		for k := range r.chainLogits {
			r.chainLogits[k] = r.d.NewBufferLen(r.V)
			r.chainD0[k], r.chainD1[k] = NewBufferU32(r.d, 0), NewBufferU32(r.d, 0)
		}
	}
	r.chainReq, r.chainResp, r.chainDone = make(chan chainReq), make(chan chainResp), make(chan struct{})
	go r.chainLoop()
	r.chainReq <- q
	return (<-r.chainResp).err
}

// chainNext returns the argmax of the oldest committed forward, the next greedy token.
func (r *resident) chainNext() (int, error) { return r.chainAsk(chainReq{op: chainOpNext}) }

// chainNextSampled returns the oldest committed forward's draw, the next sampled token. (seed, draw) are the decoder's
// draw for that forward; the chain refuses one that is not the draw it used.
func (r *resident) chainNextSampled(seed, draw uint64) (int, error) {
	return r.chainAsk(chainReq{op: chainOpNext, sampled: true, seed: seed, draw: draw})
}

func (r *resident) chainAsk(q chainReq) (int, error) {
	if r.chainReq == nil {
		return 0, fmt.Errorf("metal: greedy chain not open")
	}
	r.chainReq <- q
	a := <-r.chainResp
	if a.err == nil {
		r.chainServed++
	}
	return a.id, a.err
}

// stopChain closes the chain, if one is open, and blocks until no buffer of it is in flight.
func (r *resident) stopChain() {
	if r.chainReq == nil {
		return
	}
	close(r.chainReq)
	r.chainReq = nil
	<-r.chainDone
}

// chainFallbackAtForTest, when >= 0, makes the sampled chain's draw at that position read a row of -inf instead of the
// forward's logits, so gumbel_stage2 really writes the -1 of a row with nothing comparable: the next buffer then gathers
// past the vocabulary, and a test can drive the recovery below on the device. -1 in production.
var chainFallbackAtForTest = -1

// chainNegInfForTest is the all -inf row chainFallbackAtForTest's draw reads, made on first use.
func (r *resident) chainNegInfForTest() Buffer {
	if r.chainNegInf == (Buffer{}) {
		row := make([]float32, r.V)
		for i := range row {
			row[i] = float32(math.Inf(-1))
		}
		r.chainNegInf = NewBufferFloats(r.d, row)
	}
	return r.chainNegInf
}

// encodeChainCB encodes token pos's chained buffer with uniform set k: gather the embedding of the id in in, run the
// trunk, and write the next token to out: the argmax head, or for a sampled chain the full LM-head row (into
// chainLogits[k]) and ForwardSample's two gumbel dispatches.
func (r *resident) encodeChainCB(pos, k int, sampled bool, in, out Buffer) *Encoder {
	u := r.chainSets[k]
	r.encNKeys, r.encFANSplit = pos+1, u.uFANSplit
	defer func() { r.encNKeys, r.encFANSplit = 0, Buffer{} }()
	e := r.q.BeginNP()
	e.Dispatch(r.pEmbedGather, r.H, min(256, r.H), r.chainGW, r.chainGS, in, r.x, r.uH, r.uGumbelV)
	r.encodeTrunkWith(e, u.uPos, u.uNKeys, u.uQTempScale, u.uRopePos)
	if !sampled {
		e.Dispatch(r.pGemvW8Amax, r.V*32, 256, r.aq, r.aSc, r.lmW, r.lmS, r.part, r.uH)
		e.Dispatch(r.pArgFinish, 256, 256, r.part, out, r.uP)
	} else {
		e.Dispatch(r.pGemvW8, r.V*32, 32, r.aq, r.aSc, r.lmW, r.lmS, r.chainLogits[k], r.uH) // ForwardSample's head
		drawRow := r.chainLogits[k]
		if pos == chainFallbackAtForTest {
			drawRow = r.chainNegInfForTest()
		}
		const gbThreads = 256
		const gbShmBytes = gbThreads * 2 * 4
		e.DispatchTG(r.pGumbel1, r.gumbelNB*gbThreads, gbThreads, gbShmBytes,
			drawRow, r.uGumbelV, r.chainInvT, r.chainK0, r.chainK1, r.chainD0[k], r.chainD1[k],
			r.gumbelBKey, r.gumbelBIdx)
		e.DispatchTG(r.pGumbel2, gbThreads, gbThreads, gbShmBytes, r.gumbelBKey, r.gumbelBIdx, r.uGumbelNB, out)
	}
	e.FinishEncoding()
	return e
}

// chainLoop is the chain's goroutine: it pins one OS thread for the autorelease pool, as execLoop does.
func (r *resident) chainLoop() {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	defer close(r.chainDone)
	pool := NewARPool()
	type inflight struct {
		e    *Encoder
		pos  int
		draw uint64
	}
	var fifo []inflight
	next, sinceDrain := 0, 0
	sampled, pos0, draw0 := false, 0, uint64(0)
	commit := func() {
		if next >= r.ctxCap { // nothing to speculate into: the next position is past the resident KV
			return
		}
		k := next & 1
		r.writePosUniforms(r.chainSets[k], next, next)
		d := draw0 + uint64(next-pos0)
		if sampled {
			r.chainD0[k].SetU32(uint32(d))
			r.chainD1[k].SetU32(uint32(d >> 32))
		}
		e := r.encodeChainCB(next, k, sampled, r.chainTok[k^1], r.chainTok[k])
		e.Commit()
		fifo = append(fifo, inflight{e, next, d})
		next++
		sinceDrain++
	}
	oldest := func() (inflight, error) {
		f := fifo[0]
		fifo = fifo[1:]
		f.e.WaitDone()
		r.gpuStart, r.gpuEnd, r.kernStart, r.kernEnd = f.e.GPUStart(), f.e.GPUEnd(), f.e.KernStart(), f.e.KernEnd()
		return f, f.e.Err()
	}
	drain := func() {
		for len(fifo) > 0 {
			_, _ = oldest()
		}
		pool.Drain()
		pool = NewARPool()
		sinceDrain = 0
	}
	// pick is forward f's token. A sampled forward whose row had nothing comparable wrote -1, and ForwardSample takes the
	// argmax of its row then; the buffers queued after it gathered from the -1 (embed_gather_i8 writes zeros past the
	// vocabulary), so they are waited out and re-committed from the argmax. They overwrite the K/V the discarded ones
	// wrote, at the same positions, before anything reads it.
	pick := func(f inflight) int {
		id := int32(r.chainTok[f.pos&1].U32())
		if !sampled || id >= 0 {
			return int(id)
		}
		am := argmaxF32(r.chainLogits[f.pos&1].Floats()[:r.V])
		redo := len(fifo) > 0
		drain()
		r.chainTok[f.pos&1].SetU32(uint32(am))
		next = f.pos + 1
		if redo {
			commit()
		}
		return am
	}
	defer func() { drain(); pool.Drain() }()
	for q := range r.chainReq {
		switch q.op {
		case chainOpStart:
			drain()
			next, sampled, pos0, draw0 = q.pos, q.sampled, q.pos, q.draw
			if sampled {
				r.chainInvT.Floats()[0] = q.invT
				r.chainK0.SetU32(uint32(q.seed))
				r.chainK1.SetU32(uint32(q.seed >> 32))
			}
			r.chainTok[(q.pos+1)&1].SetU32(uint32(q.id)) // token pos's buffer reads the id the "previous" one would have written
			if r.attnFANKV > 0 {
				r.uAttnFAG.SetU32(uint32(r.nH / r.attnFANKV))
			}
			commit()
			r.chainResp <- chainResp{}
		case chainOpNext:
			if len(fifo) == 0 {
				r.chainResp <- chainResp{err: fmt.Errorf("metal: greedy chain: no forward in flight")}
				continue
			}
			if q.sampled != sampled {
				r.chainResp <- chainResp{err: fmt.Errorf("metal: greedy chain: a sampled request on a greedy chain, or the reverse")}
				continue
			}
			if want := draw0 + uint64(fifo[0].pos-pos0); sampled && (q.seed != draw0Seed(r) || q.draw != want) {
				r.chainResp <- chainResp{err: fmt.Errorf("metal: sampled chain: the decoder's draw (%d, %d) for position %d is not the chain's (%d, %d)",
					q.seed, q.draw, fifo[0].pos, draw0Seed(r), want)}
				continue
			}
			if sinceDrain >= chainDrainEvery { // the one gap per chainDrainEvery tokens
				f, err := oldest()
				drain()
				id := 0
				if err == nil {
					id = pick(f)
					if len(fifo) == 0 {
						commit()
					}
				}
				r.chainResp <- chainResp{id: id, err: err}
				continue
			}
			commit() // the token after the newest, queued before the host waits
			f, err := oldest()
			id := 0
			if err == nil {
				id = pick(f)
			}
			r.chainResp <- chainResp{id: id, err: err}
		case chainOpStop:
			drain()
			r.chainResp <- chainResp{}
		}
	}
}

// draw0Seed is the seed the open sampled chain draws with (chainK1:chainK0).
func draw0Seed(r *resident) uint64 { return uint64(r.chainK1.U32())<<32 | uint64(r.chainK0.U32()) }
