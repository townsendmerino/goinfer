//go:build cuda

package cuda

import (
	"fmt"

	gpu "github.com/townsendmerino/goinfer/backend/cuda/internal/gpudev"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
)

// stateSpecsOf is the list every recurrent buffer on this backend is sized from and priced from. A variable so a test can move the
// list and see which buffers follow it.
var stateSpecsOf = (*decoder.Model).StateSpecs

// recurrentFloats is one recurrent layer's conv window and state, in floats, as the list gives them.
type recurrentFloats struct{ win, state int }

// recurrentFloatsOf reads layer l's window and state from stateSpecsOf. Gated-DeltaNet and Mamba-2 both keep exactly those two.
func recurrentFloatsOf(m *decoder.Model, l int) (recurrentFloats, error) {
	specs := stateSpecsOf(m, l)
	if len(specs) != 2 || specs[0].Name != "window" || specs[1].Name != "state" {
		return recurrentFloats{}, fmt.Errorf("cuda: layer %d's recurrent state is %v, and the block holds a window and a state", l, specs)
	}
	return recurrentFloats{win: specs[0].Floats, state: specs[1].Floats}, nil
}

// setRecurrentFloats records layer l's window and state sizes from the list, for the upload and everything after it.
func (r *cudaResident) setRecurrentFloats(m *decoder.Model, l, nLayers int) (recurrentFloats, error) {
	rf, err := recurrentFloatsOf(m, l)
	if err != nil {
		return rf, err
	}
	if r.recFloats == nil {
		r.recFloats = make([]recurrentFloats, nLayers)
	}
	r.recFloats[l] = rf
	return rf, nil
}

// zeroFloats returns a function giving n zeroed floats from one buffer grown to the largest n asked for; each result is read, not kept.
func zeroFloats() func(n int) []float32 {
	var z []float32
	return func(n int) []float32 {
		if len(z) < n {
			z = make([]float32, n)
		}
		return z[:n]
	}
}

// recurrentPair is one KV slot's copy of a recurrent layer's two compounding buffers: the conv window and the matrix or SSM state.
type recurrentPair struct{ win, state Buffer }

func (L *cudaLayer) isRecurrent() bool { return L.isDeltaNet() || L.mixer == decoder.MixerMamba }

// recurrentLive is the pair the layer's launches read now: the bound slot's.
func (L *cudaLayer) recurrentLive() recurrentPair {
	if L.isDeltaNet() {
		return recurrentPair{L.dnWin, L.dnState}
	}
	return recurrentPair{L.mbWin, L.mbSSM}
}

func (L *cudaLayer) bindRecurrent(p recurrentPair) {
	if L.isDeltaNet() {
		L.dnWin, L.dnState = p.win, p.state
		return
	}
	L.mbWin, L.mbSSM = p.win, p.state
}

// recurrentOf is KV slot s's pair, bound or not.
func (L *cudaLayer) recurrentOf(s int) recurrentPair {
	if s < len(L.recSlots) {
		return L.recSlots[s]
	}
	return L.recurrentLive()
}

// recurrentResetEverySlotForTest, when set, makes a reset zero every slot's pair, not the bound one's: the plant of KV8's S-2. Only a test sets it.
var recurrentResetEverySlotForTest bool

// resetEverySlotForTest is that plant's body.
func (r *cudaResident) resetEverySlotForTest() error {
	for l := range r.layers {
		for _, p := range r.layers[l].recSlots {
			for _, b := range []Buffer{p.win, p.state} {
				if e := gpu.Upload(b, make([]float32, b.Len())); e != nil {
					return e
				}
			}
		}
	}
	return nil
}

// recurrentSlotAllocFailForTest, when set, fails allocRecurrentSlots's first state as a full device would. Only a test sets it.
var recurrentSlotAllocFailForTest bool

// allocRecurrentSlots gives every recurrent layer a zeroed conv window and state for each KV slot past the first, whose pair the
// upload allocated before the slot count was decided. Called with the KV pools, so a device allocation failure takes a slot away
// with its pools (kvTryAllocPools); kvChargedSlotsFit prices them.
func (r *cudaResident) allocRecurrentSlots(slots int) {
	if slots < 2 || !r.recurrent() {
		return
	}
	defer r.onLine(mem.LineState)()
	for l := range r.layers {
		L := &r.layers[l]
		if !L.isRecurrent() {
			continue
		}
		win, state := r.recFloats[l].win, r.recFloats[l].state
		L.recSlots = []recurrentPair{L.recurrentLive()}
		for range slots - 1 {
			L.recSlots = append(L.recSlots, recurrentPair{win: r.up32(make([]float32, win))})
			if recurrentSlotAllocFailForTest {
				panic("cuda: device allocation failed (test seam): cuMemAlloc_v2: CUDA_ERROR_OUT_OF_MEMORY")
			}
			L.recSlots[len(L.recSlots)-1].state = r.up32(make([]float32, state))
		}
	}
}

// releaseRecurrentSlots releases what allocRecurrentSlots made, leaving each layer on the pair the upload allocated.
func (r *cudaResident) releaseRecurrentSlots() {
	for l := range r.layers {
		L := &r.layers[l]
		for i, p := range L.recSlots {
			if i == 0 {
				L.bindRecurrent(p)
				continue
			}
			for _, b := range []Buffer{p.win, p.state} {
				if b.Len() > 0 {
					r.dev.ReleaseBuf(b)
				}
			}
		}
		L.recSlots = nil
	}
}

// recurrentSlotBytes is what the KV slots past the first cost in recurrent state, on top of the one the upload allocated.
func (r *cudaResident) recurrentSlotBytes(slots int) int64 {
	return int64(max(slots, 1)-1) * r.recurrentStateCharged()
}
