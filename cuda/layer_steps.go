//go:build cuda

package cuda

import (
	"fmt"
	"slices"

	gpu "github.com/townsendmerino/goinfer/cuda/internal/gpudev"
	"github.com/townsendmerino/goinfer/internal/program"
)

// cudaStep is one entry of CUDA's fusion table (OP4cu-b2, docs/tasks/task-forward-architecture-2026-10.md §4.9a): one of
// today's decode segments, emitting exactly its launches. A static step is a graph-capturable launch run (captureGraphs
// records it); a dynamic one binds per-token state (the position, the key count, a host readback).
type cudaStep uint8

const (
	stepQKV      cudaStep = iota + 1 // static: segA, the input norm through the QK-norm and v-norm
	stepAttnGap                      // dynamic: decodeAttnGap, rope, the KV store into the bound slot, attention by lane
	stepMLA                          // dynamic: segAMLA
	stepOutFFN                       // static: segB, the output projection and its residual, then the FFN up to the router readback
	stepDeltaNet                     // static: deltaNetMixer then segBFFN, one captured segment
	stepMamba                        // static: mambaMixer
	stepFFNOnly                      // static: nonGatedMLP
	stepTail                         // dynamic, then static: layerTail, the router readback and expert DMA, then segC
)

var cudaStepNames = [...]string{stepQKV: "qkv", stepAttnGap: "attnGap", stepMLA: "mla", stepOutFFN: "outFFN",
	stepDeltaNet: "deltaNet", stepMamba: "mamba", stepFFNOnly: "ffnOnly", stepTail: "tail"}

func (s cudaStep) String() string {
	if int(s) < len(cudaStepNames) && cudaStepNames[s] != "" {
		return cudaStepNames[s]
	}
	return fmt.Sprintf("cudaStep(%d)", uint8(s))
}

// cudaSublayerSteps is CUDA's fusion table: each layer sublayer pattern a CUDA resident runs (program.SublayersOf's kinds, in
// order, and whether they are a parallel block's two halves) and the segment steps that run it. PLE and the layer scale appear
// only after a sequential dense FFN, where segBFFN runs them; Gemma 4's MoE block scales inside segC; segB runs a parallel block.
var cudaSublayerSteps = []struct {
	pattern  []program.Sublayer
	parallel bool
	steps    []cudaStep
}{
	{[]program.Sublayer{program.SublayerAttention, program.SublayerDenseFFN}, false, attnSteps},
	{[]program.Sublayer{program.SublayerAttention, program.SublayerDenseFFN}, true, attnSteps},                                                   // a parallel block
	{[]program.Sublayer{program.SublayerAttention, program.SublayerDenseFFN, program.SublayerLayerScale}, false, attnSteps},                      // Gemma 4's dense layer
	{[]program.Sublayer{program.SublayerAttention, program.SublayerDenseFFN, program.SublayerPLE, program.SublayerLayerScale}, false, attnSteps}, // Gemma 4's E-model layer
	{[]program.Sublayer{program.SublayerAttention, program.SublayerMoE}, false, attnSteps},
	{[]program.Sublayer{program.SublayerAttention, program.SublayerGemma4MoE}, false, attnSteps},
	{[]program.Sublayer{program.SublayerAttention}, false, attnSteps}, // Nemotron-H's attention block
	{[]program.Sublayer{program.SublayerMLA, program.SublayerDenseFFN}, false, mlaSteps},
	{[]program.Sublayer{program.SublayerMLA, program.SublayerMoE}, false, mlaSteps},
	{[]program.Sublayer{program.SublayerDeltaNet, program.SublayerDenseFFN}, false, deltaNetSteps},
	{[]program.Sublayer{program.SublayerDeltaNet, program.SublayerMoE}, false, deltaNetSteps},
	{[]program.Sublayer{program.SublayerMamba}, false, mambaSteps},         // Nemotron-H's Mamba-2 block
	{[]program.Sublayer{program.SublayerNonGatedFFN}, false, ffnOnlySteps}, // Nemotron-H's MLP block
}

var (
	attnSteps     = []cudaStep{stepQKV, stepAttnGap, stepOutFFN, stepTail}
	mlaSteps      = []cudaStep{stepMLA, stepOutFFN, stepTail}
	deltaNetSteps = []cudaStep{stepDeltaNet, stepTail}
	mambaSteps    = []cudaStep{stepMamba, stepTail}
	ffnOnlySteps  = []cudaStep{stepFFNOnly, stepTail}
)

// cudaSublayersOf is a layer's sublayer pattern, program.SublayersOf's kinds checked by program.CheckSublayerClaims, and whether
// they hold a parallel block.
func cudaSublayersOf(ops []program.Op) (pattern []program.Sublayer, parallel bool, err error) {
	spans, err := program.SublayersOf(ops, "CUDA")
	if err == nil {
		err = program.CheckSublayerClaims(ops, spans)
	}
	if err != nil {
		return nil, false, err
	}
	pattern = make([]program.Sublayer, len(spans))
	for i, sp := range spans {
		pattern[i] = sp.Kind
		parallel = parallel || sp.Parallel
	}
	return pattern, parallel, nil
}

// cudaStepsOf is a layer's steps from its operations: its sublayer pattern and parallel flag (cudaSublayersOf) looked up in
// cudaSublayerSteps. A layer whose operations match no sublayers, whose sublayers claim an operation their blocks do not emit, or
// whose pattern has no entry, is an error naming it.
func cudaStepsOf(ops []program.Op) ([]cudaStep, error) {
	pattern, parallel, err := cudaSublayersOf(ops)
	if err != nil {
		return nil, err
	}
	for _, e := range cudaSublayerSteps {
		if e.parallel == parallel && slices.Equal(e.pattern, pattern) {
			return slices.Clone(e.steps), nil
		}
	}
	return nil, fmt.Errorf("its sublayers %v (a parallel block: %v) have no entry in CUDA's fusion table", pattern, parallel)
}

// loadedSublayers is the sublayer pattern of what a built layer's upload holds, read from its buffers: the mixer's weights
// (DeltaNet's, Mamba-2's, MLA's, or attention's q), the FFN's (a router, with Gemma 4's MoE norms or not; an up projection
// without a gate, which is Nemotron-H's non-gated MLP; or a gate), then, after a dense FFN, the PLE gate and a non-zero layer
// scalar. parallel is the resident's parallel-block placement, which is model-level on CUDA. The buffers do not tell a gated
// dense FFN's activation, nor attention's form (plain, gated or Gemma 4's): the build checks those where it reads them.
func loadedSublayers(L *cudaLayer) []program.Sublayer {
	var s []program.Sublayer
	switch {
	case L.dnQKV.N > 0:
		s = append(s, program.SublayerDeltaNet)
	case L.mbIn.N > 0:
		s = append(s, program.SublayerMamba)
	case L.mlaQA.N > 0 || L.mlaQ.N > 0:
		s = append(s, program.SublayerMLA)
	case L.q.N > 0:
		s = append(s, program.SublayerAttention)
	}
	switch {
	case L.routerW != (Buffer{}) && L.g4preFFN2 != (Buffer{}):
		return append(s, program.SublayerGemma4MoE)
	case L.routerW != (Buffer{}):
		return append(s, program.SublayerMoE)
	case L.u.N > 0 && L.g.N == 0:
		return append(s, program.SublayerNonGatedFFN)
	case L.g.N == 0:
		return s
	}
	s = append(s, program.SublayerDenseFFN)
	if L.pleGate.N > 0 {
		s = append(s, program.SublayerPLE)
	}
	if L.layerScalar != 0 {
		s = append(s, program.SublayerLayerScale)
	}
	return s
}

// checkLoadedSublayers is an error unless a layer's program sublayers and parallel flag are what its upload holds.
func checkLoadedSublayers(prog []program.Sublayer, progParallel bool, loaded []program.Sublayer, loadedParallel bool) error {
	if !slices.Equal(prog, loaded) || progParallel != loadedParallel {
		return fmt.Errorf("the program's sublayers are %v (a parallel block: %v) and the layer's upload holds %v (parallel: %v)",
			prog, progParallel, loaded, loadedParallel)
	}
	return nil
}

// walkPlant plants one defect in the walk, so B2-2 can show B2-1 goes red; zero in production (set by tests only, through
// setWalkPlantForTest). The plants act in runStep's live path, so they are run with CUDA graphs off: a captured segment replays
// without passing them.
var walkPlant walkPlantKind

type walkPlantKind uint8

const (
	walkPlantNone       walkPlantKind = iota
	walkPlantDropGap                  // the walk skips attnGap
	walkPlantSwap                     // the walk runs outFFN before attnGap
	walkPlantQDimMinus1               // outFFN reads the layer's qDim as qDim-1
)

// legacyLayerForTest, when set, runs a layer with the loop launchToken had before the walk, for B2-1's comparison and B2-6's
// A/B. A build made while it is set also skips the walk's build checks, as the pre-walk build did. Nil in production; tests set
// it directly and restore it.
var legacyLayerForTest func(r *cudaResident, Ly *cudaLayer, l, pos, ropePos int, qTempScale float32, gA, gB, gC bool) error

// walkLayer runs layer l's steps (OP4cu-b2). gA, gB and gC say whether the static segments replay their captured graphs.
func (r *cudaResident) walkLayer(Ly *cudaLayer, l, pos, ropePos int, qTempScale float32, gA, gB, gC bool) error {
	steps := Ly.steps
	if walkPlant == walkPlantSwap && len(steps) == 4 {
		steps = []cudaStep{steps[0], steps[2], steps[1], steps[3]}
	}
	for _, st := range steps {
		if walkPlant == walkPlantDropGap && st == stepAttnGap {
			continue
		}
		if e := r.runStep(st, Ly, l, pos, ropePos, qTempScale, gA, gB, gC); e != nil {
			return e
		}
	}
	return nil
}

// runStep emits one step's launches, replaying a static step's captured graph when graphs are on.
func (r *cudaResident) runStep(st cudaStep, Ly *cudaLayer, l, pos, ropePos int, qTempScale float32, gA, gB, gC bool) error {
	switch st {
	case stepQKV:
		if gA {
			return r.replay(Ly.gSegA)
		}
		return r.segA(Ly, l)
	case stepAttnGap:
		return r.decodeAttnGap(Ly, l, pos, ropePos, qTempScale)
	case stepMLA:
		return r.segAMLA(Ly, l, pos, ropePos)
	case stepOutFFN:
		if r.subCap { // pre-o-proj attention context (qDim), before quant — the cross-box discriminator (live path only)
			r.capVec(r.cctx, r.subCtxC, l, Ly.qDim)
		}
		if gB {
			return r.replay(Ly.gSegB)
		}
		if walkPlant == walkPlantQDimMinus1 {
			p := *Ly
			p.qDim--
			return r.segB(&p, l, r.x)
		}
		return r.segB(Ly, l, r.x)
	case stepDeltaNet:
		// The mixer and its FFN half are captured as ONE segment (gSegA): with no rope or attention gap there is nothing
		// dynamic to split around.
		if gA {
			return r.replay(Ly.gSegA)
		}
		if e := r.deltaNetMixer(Ly, l); e != nil {
			return e
		}
		return r.segBFFN(Ly, l, r.x)
	case stepMamba:
		return r.mambaMixer(Ly)
	case stepFFNOnly:
		return r.nonGatedMLP(Ly)
	case stepTail:
		return r.layerTail(Ly, l, gC, r.x)
	}
	return fmt.Errorf("cuda: layer %d: step %v has no launches", l, st)
}

// replay runs a captured segment, then syncs the stream when graphsSync asks for it.
func (r *cudaResident) replay(g *gpu.Graph) error {
	if e := g.Replay(); e != nil {
		return e
	}
	if r.graphsSync {
		return r.stream.Sync()
	}
	return nil
}
