//go:build cuda

package cuda

import (
	"fmt"
	"slices"

	gpu "github.com/townsendmerino/goinfer/cuda/internal/gpudev"
	"github.com/townsendmerino/goinfer/decoder"
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

// layerStepsFor is the steps a layer whose build loaded mixer runs, in the order launchToken runs them: the build's own view,
// which must equal cudaStepsOf's from the program. A mixer the table has no entries for is an error naming it.
func layerStepsFor(mixer decoder.ResidentMixer) ([]cudaStep, error) {
	switch mixer {
	case decoder.MixerAttention:
		return slices.Clone(attnSteps), nil
	case decoder.MixerMLA:
		return slices.Clone(mlaSteps), nil
	case decoder.MixerDeltaNet:
		return slices.Clone(deltaNetSteps), nil
	case decoder.MixerMamba:
		return slices.Clone(mambaSteps), nil
	case decoder.MixerNone:
		return slices.Clone(ffnOnlySteps), nil
	}
	return nil, fmt.Errorf("the %v mixer has no steps on CUDA", mixer)
}

// cudaSublayerSteps is CUDA's fusion table: each layer sublayer pattern (program.SublayersOf's kinds, PLE and the layer scale
// aside, which segBFFN and segC run inside the FFN steps) and the segment steps that run it. A parallel block's two halves are
// the attention and dense FFN pattern; segB runs it.
var cudaSublayerSteps = []struct {
	pattern []program.Sublayer
	steps   []cudaStep
}{
	{[]program.Sublayer{program.SublayerAttention, program.SublayerDenseFFN}, attnSteps},
	{[]program.Sublayer{program.SublayerAttention, program.SublayerMoE}, attnSteps},
	{[]program.Sublayer{program.SublayerAttention, program.SublayerGemma4MoE}, attnSteps},
	{[]program.Sublayer{program.SublayerAttention}, attnSteps}, // Nemotron-H's attention block
	{[]program.Sublayer{program.SublayerMLA, program.SublayerDenseFFN}, mlaSteps},
	{[]program.Sublayer{program.SublayerMLA, program.SublayerMoE}, mlaSteps},
	{[]program.Sublayer{program.SublayerDeltaNet, program.SublayerDenseFFN}, deltaNetSteps},
	{[]program.Sublayer{program.SublayerDeltaNet, program.SublayerMoE}, deltaNetSteps},
	{[]program.Sublayer{program.SublayerMamba}, mambaSteps},          // Nemotron-H's Mamba-2 block
	{[]program.Sublayer{program.SublayerNonGatedFFN}, ffnOnlySteps}, // Nemotron-H's MLP block
}

var (
	attnSteps     = []cudaStep{stepQKV, stepAttnGap, stepOutFFN, stepTail}
	mlaSteps      = []cudaStep{stepMLA, stepOutFFN, stepTail}
	deltaNetSteps = []cudaStep{stepDeltaNet, stepTail}
	mambaSteps    = []cudaStep{stepMamba, stepTail}
	ffnOnlySteps  = []cudaStep{stepFFNOnly, stepTail}
)

// cudaStepsOf is a layer's steps from its operations: its sublayers (program.SublayersOf, checked by
// program.CheckSublayerClaims), looked up in cudaSublayerSteps. A layer whose operations match no sublayers, whose sublayers
// claim an operation their blocks do not emit, or whose pattern has no entry, is an error naming it.
func cudaStepsOf(ops []program.Op) ([]cudaStep, error) {
	spans, err := program.SublayersOf(ops, "CUDA")
	if err == nil {
		err = program.CheckSublayerClaims(ops, spans)
	}
	if err != nil {
		return nil, err
	}
	var pattern []program.Sublayer
	for _, sp := range spans {
		if sp.Kind != program.SublayerPLE && sp.Kind != program.SublayerLayerScale {
			pattern = append(pattern, sp.Kind)
		}
	}
	for _, e := range cudaSublayerSteps {
		if slices.Equal(e.pattern, pattern) {
			return slices.Clone(e.steps), nil
		}
	}
	return nil, fmt.Errorf("its sublayers %v have no entry in CUDA's fusion table", pattern)
}

// walkPlant plants one defect in the walk, so B2-2 can show B2-1 goes red; zero in production (set by tests only, through
// setWalkPlantForTest).
var walkPlant walkPlantKind

type walkPlantKind uint8

const (
	walkPlantNone       walkPlantKind = iota
	walkPlantDropGap                  // the walk skips attnGap
	walkPlantSwap                     // the walk runs outFFN before attnGap
	walkPlantQDimMinus1               // outFFN reads the layer's qDim as qDim-1
)

// legacyLayerForTest, when set, runs a layer with the loop launchToken had before the walk, for B2-1's comparison and B2-6's
// A/B. Nil in production; set by tests only, through setLegacyLayerForTest.
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
