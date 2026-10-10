package decoder

import "runtime"

// streamLayerBudgetFallback is the window's memory budget on a host with no available-memory probe.
const streamLayerBudgetFallback = 1 << 30

// streamLayerBudget is the memory a streaming transcode may hold in built, unwritten layers: a quarter of the RAM
// available when the transcode starts (docs/tasks/task-giw-v16-transcode-2026-10.md, Decision 3).
func streamLayerBudget() int64 {
	if a := HostRAMAvailableBytes(); a > 0 {
		return a / 4
	}
	return streamLayerBudgetFallback
}

// streamLayerWindowForTest, when positive, is the window streamLayers uses after layer 0, whatever the budget says.
var streamLayerWindowForTest int

// streamLayers builds a model's layers for a sink: a window of layers in parallel, written in layer order and released,
// then the next window. A layer's bytes depend on no other layer and the write order does not change, so the bundle is
// byte for byte the one a layer-at-a-time build writes. load must be safe to call concurrently for distinct layers, as
// the resident build already requires of it.
//
// The window. A MoE builds one layer at a time: its experts run in parallel inside the layer, and nesting both would
// multiply the transients of the largest layers there are. Any other model builds and writes layer 0 alone, and from then
// on the window is the smaller of GOMAXPROCS and sink.layerBudget divided by the largest layer written so far, never
// under one. A writer with no budget set therefore builds one layer at a time.
func streamLayers(w *Weights, arch *Architecture, sink *giwWriter, load func(i int) error) error {
	n := len(w.Layers)
	var largest int64 // bytes of the largest layer written so far, as the sink counted them
	for p := 0; p < n; {
		k := 1
		if arch.MoE == nil && p > 0 && largest > 0 {
			k = int(min(int64(runtime.GOMAXPROCS(0)), max(1, sink.layerBudget/largest)))
			if streamLayerWindowForTest > 0 {
				k = streamLayerWindowForTest
			}
			k = min(k, n-p)
		}
		at := func(j int) int { return streamLayerAt(p+j, n) }
		if err := parallelLayers(k, nil, func(j int) error { return load(at(j)) }); err != nil {
			return err
		}
		for j := range k {
			i := at(j)
			before := sink.n
			sink.layer(&w.Layers[i])
			if sink.err != nil {
				return sink.err
			}
			largest = max(largest, sink.n-before)
			w.Layers[i] = LayerWeights{} // release before the next window is built
		}
		p += k
	}
	return nil
}

// streamLayerAt is the layer written at position pos of n: pos itself, unless streamDirSwapForTest plants its defect
// (layers 0 and 1 in each other's place).
func streamLayerAt(pos, n int) int {
	if streamDirSwapForTest && n > 1 && pos < 2 {
		return 1 - pos
	}
	return pos
}
