//go:build cuda

package cuda

import (
	"math"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/goinfer/decoder"
)

// gumbelThreads is GB_THREADS in gumbel.cu: stage 1 covers 4 vocabulary entries per thread.
const gumbelThreads = 256

// gumbelBlocks is the stage-1 grid for a vocabulary of v entries.
func gumbelBlocks(v int) int { return (v + 4*gumbelThreads - 1) / (4 * gumbelThreads) }

// SampleAvailable reports whether this resident can draw a temperature-only sample on-device: the same
// condition as TopKAvailable (the device row must be the raw LM-head output the host sampler would see).
func (r *cudaResident) SampleAvailable() bool { return r.TopKAvailable() }

// ForwardSample runs one token's forward and draws the NEXT token on-device by Gumbel-max
// (decoder.ResidentSample), returning just the id — no logits readback, no host normalisation. (seed, draw)
// are the sampler's own (Sampler.NextDraw), so this is the draw the host would have made.
func (r *cudaResident) ForwardSample(embedding []float32, pos int, temperature float64, seed, draw uint64) (int, error) {
	invT := float32(1 / temperature)
	if math.IsInf(float64(invT), 0) { // absurdly small temperature: greedy, exactly as the host does
		return r.ForwardArgmax(embedding, pos)
	}
	if e := r.checkCap(pos, 1); e != nil {
		return 0, e
	}
	var id int
	err := r.do(func() error {
		if e := r.launchToken(embedding, pos, pos, true); e != nil {
			return e
		}
		var e error
		id, e = r.gumbelPick(r.vocab, invT, seed, draw)
		return e
	})
	return id, err
}

// gumbelPick draws over r.logits[:v]. Executor thread only. Shared by ForwardSample and the test hook so
// the tests exercise exactly the launch and readback decode runs.
func (r *cudaResident) gumbelPick(v int, invT float32, seed, draw uint64) (int, error) {
	nb := gumbelBlocks(v)
	cfg1 := LaunchConfig{GridX: uint32(nb), GridY: 1, GridZ: 1, BlockX: gumbelThreads, BlockY: 1, BlockZ: 1}
	if e := r.launch(r.fGumbel1, cfg1, Arg(r.logits), gpu.ArgValue(int32(v)), gpu.ArgValue(invT),
		gpu.ArgValue(uint32(seed)), gpu.ArgValue(uint32(seed>>32)),
		gpu.ArgValue(uint32(draw)), gpu.ArgValue(uint32(draw>>32)), Arg(r.gbKey), Arg(r.gbIdx)); e != nil {
		return 0, e
	}
	if e := r.launch(r.fGumbel2, onecfg(gumbelThreads, 0), Arg(r.gbKey), Arg(r.gbIdx), gpu.ArgValue(int32(nb)), Arg(r.gbOut)); e != nil {
		return 0, e
	}
	if e := r.stream.Sync(); e != nil {
		return 0, e
	}
	out := make([]int32, 1)
	if e := gpu.Download(r.gbOut, out); e != nil {
		return 0, e
	}
	if out[0] >= 0 {
		return int(out[0]), nil
	}
	// Nothing comparable (an all -inf row): the host draws the argmax; so does the device.
	if e := r.launch(r.fArg, onecfg(256, 256*4+256*4), Arg(r.logits), gpu.ArgValue(int32(v)), Arg(r.argIdx), Arg(r.argVal)); e != nil {
		return 0, e
	}
	if e := r.stream.Sync(); e != nil {
		return 0, e
	}
	if e := gpu.Download(r.argIdx, out); e != nil {
		return 0, e
	}
	return int(out[0]), nil
}

var _ decoder.ResidentSample = (*cudaResident)(nil)
