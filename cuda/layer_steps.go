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

// cudaSublayerSteps is CUDA's fusion table: each layer sublayer pattern a CUDA resident runs (program.SublayersOf's kinds, in
// order) and the segment steps that run it. PLE and the layer scale appear only after a dense FFN, where segBFFN runs them; Gemma
// 4's MoE block scales inside segC. A parallel block's two halves are the attention and dense FFN pattern; segB runs it.
var cudaSublayerSteps = []struct {
	pattern []program.Sublayer
	steps   []cudaStep
}{
	{[]program.Sublayer{program.SublayerAttention, program.SublayerDenseFFN}, attnSteps},
	{[]program.Sublayer{program.SublayerAttention, program.SublayerDenseFFN, program.SublayerLayerScale}, attnSteps},       // Gemma 4's dense layer
	{[]program.Sublayer{program.SublayerAttention, program.SublayerDenseFFN, program.SublayerPLE, program.SublayerLayerScale}, attnSteps}, // Gemma 4's E-model layer
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

// cudaSublayersOf is a layer's sublayer pattern: program.SublayersOf's kinds, checked by program.CheckSublayerClaims.
func cudaSublayersOf(ops []program.Op) ([]program.Sublayer, error) {
	spans, err := program.SublayersOf(ops, "CUDA")
	if err == nil {
		err = program.CheckSublayerClaims(ops, spans)
	}
	if err != nil {
		return nil, err
	}
	pattern := make([]program.Sublayer, len(spans))
	for i, sp := range spans {
		pattern[i] = sp.Kind
	}
	return pattern, nil
}

// cudaStepsOf is a layer's steps from its operations: its sublayer pattern (cudaSublayersOf) looked up in cudaSublayerSteps. A
// layer whose operations match no sublayers, whose sublayers claim an operation their blocks do not emit, or whose pattern has
// no entry, is an error naming it.
func cudaStepsOf(ops []program.Op) ([]cudaStep, error) {
	pattern, err := cudaSublayersOf(ops)
	if err != nil {
		return nil, err
	}
	for _, e := range cudaSublayerSteps {
		if slices.Equal(e.pattern, pattern) {
			return slices.Clone(e.steps), nil
		}
	}
	return nil, fmt.Errorf("its sublayers %v have no entry in CUDA's fusion table", pattern)
}

// loadedSublayers is the sublayer pattern of the blocks a layer's build loaded: its mixer's, then its FFN's (Gemma 4's MoE
// block, a MoE FFN, Nemotron-H's MLP, or a dense FFN), then, after a dense FFN, the PLE branch when the model has one and the
// layer scale when its scalar is read.
func loadedSublayers(mixer decoder.ResidentMixer, ffn, g4moe, moe, ple, layerScale bool) []program.Sublayer {
	var s []program.Sublayer
	switch mixer {
	case decoder.MixerAttention:
		s = append(s, program.SublayerAttention)
	case decoder.MixerMLA:
		s = append(s, program.SublayerMLA)
	case decoder.MixerDeltaNet:
		s = append(s, program.SublayerDeltaNet)
	case decoder.MixerMamba:
		s = append(s, program.SublayerMamba)
	}
	if !ffn {
		return s
	}
	switch {
	case g4moe:
		return append(s, program.SublayerGemma4MoE)
	case moe:
		return append(s, program.SublayerMoE)
	case mixer == decoder.MixerNone:
		return append(s, program.SublayerNonGatedFFN)
	}
	s = append(s, program.SublayerDenseFFN)
	if ple {
		s = append(s, program.SublayerPLE)
	}
	if layerScale {
		s = append(s, program.SublayerLayerScale)
	}
	return s
}

// checkLoadedSublayers is an error unless the program's sublayer pattern for a layer is the one its build loaded.
func checkLoadedSublayers(prog, loaded []program.Sublayer) error {
	if !slices.Equal(prog, loaded) {
		return fmt.Errorf("the program's sublayers are %v and the layer's build loaded %v", prog, loaded)
	}
	return nil
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
