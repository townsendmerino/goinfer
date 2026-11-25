package decoder

import (
	"context"
	"runtime"
	"sync/atomic"
)

type streamHooksKey struct{}

// streamTestHooks holds optional test seams attached to a streaming transcode context and copied onto giwWriter.
type streamTestHooks struct {
	layerWindow    int
	expertsInOrder bool
	expertFanouts  *atomic.Int64
	embedWorkers   int
}

func withStreamTestHooks(ctx context.Context, h streamTestHooks) context.Context {
	return context.WithValue(ctx, streamHooksKey{}, h)
}

func streamTestHooksOf(ctx context.Context) streamTestHooks {
	if ctx == nil {
		return streamTestHooks{}
	}
	h, _ := ctx.Value(streamHooksKey{}).(streamTestHooks)
	return h
}

// streamLayerBudgetFallback is the window's memory budget on a host with no available-memory probe.
const streamLayerBudgetFallback = 1 << 30

// streamLayerBudget is the memory a streaming transcode may hold in built, unwritten layers: a quarter of the RAM
// available when the transcode starts.
func streamLayerBudget() int64 {
	if a := HostRAMAvailableBytes(); a > 0 {
		return a / 4
	}
	return streamLayerBudgetFallback
}

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
			if sink.testHooks.layerWindow > 0 {
				k = sink.testHooks.layerWindow
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

// forExperts runs one(e) for each of a MoE layer's n experts. Building for a sink it fans them out across cores:
// streamLayers builds a MoE one layer at a time, and the experts are most of that layer. With no sink it runs them in
// order, because a resident build already runs one layer per core and nesting the two would multiply the transients.
//
// one must write only its own expert's slot and read its source without shared state, as the resident build already
// requires across layers. The result is then the one the in-order run gives.
func forExperts(n int, sink *giwWriter, one func(e int) error) error {
	if sink != nil && !sink.testHooks.expertsInOrder {
		if sink.testHooks.expertFanouts != nil {
			sink.testHooks.expertFanouts.Add(1)
		}
		return parallelLayers(n, nil, one)
	}
	for e := range n {
		if err := one(e); err != nil {
			return err
		}
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
