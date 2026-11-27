//go:build gpu

package towers

import (
	"github.com/townsendmerino/goinfer/backend/webgpu/internal/kernels"
	"testing"

	"github.com/townsendmerino/goinfer/internal/vision"
)

// refusesAll runs each misrouted op through walk; a walker with no device panics on anything it queues or allocates.
func refusesAll(t *testing.T, walk func(vision.TowerBlock) error) {
	t.Helper()
	for _, c := range vision.MisroutedTowerOpsForTest() {
		err := func() (err error) {
			defer func() {
				if recover() != nil {
					t.Errorf("%s: queued instead of refused", c.Name)
				}
			}()
			return walk(vision.TowerBlock{c.Op})
		}()
		if err == nil {
			t.Errorf("%s: accepted", c.Name)
		}
	}
}

// TestGridVTowerWalkBlock_refusesMisroutedOps: the grid towers' WebGPU walker refuses each misrouted op before it
// allocates or queues anything.
func TestGridVTowerWalkBlock_refusesMisroutedOps(t *testing.T) {
	const hidden, inter = 8, 32
	a := &gridVTower{towerOps: &towerOps{}, hidden: hidden, inter: inter, heads: 2, hd: hidden / 2}
	hp := gvProj{out: hidden, in: hidden}
	b := &gvBlock{q: hp, k: hp, v: hp, proj: hp, fc1: gvProj{out: inter, in: hidden}, fc2: gvProj{out: hidden, in: inter},
		gate: gvProj{out: inter, in: hidden}, up: gvProj{out: inter, in: hidden}, down: gvProj{out: hidden, in: inter}}
	refusesAll(t, func(p vision.TowerBlock) error { return a.walkBlock(p, b, nil, nil, nil, 4, []int{0, 4}, nil) })
}

// TestVisionEncoderWalkBlock_refusesMisroutedOps: the same for SigLIP's WebGPU walker, in its float32 form.
func TestVisionEncoderWalkBlock_refusesMisroutedOps(t *testing.T) {
	const hidden, inter = 8, 32
	ve := &VisionEncoder{f32: true, w: vision.GPUWeights{Hidden: hidden, Inter: inter, NumHeads: 2, HeadDim: hidden / 2, NumPatches: 4}}
	sq := kernels.NewDeviceBuffer(nil, hidden*hidden)
	L := &visLayer{qf: sq, kf: sq, vf: sq, of: sq, fc1f: kernels.NewDeviceBuffer(nil, inter*hidden), fc2f: kernels.NewDeviceBuffer(nil, hidden*inter)}
	proj := func(*DeviceBuffer, *ResidentW8A8, *DeviceBuffer, *DeviceBuffer, int, int, int) (*DeviceBuffer, error) {
		panic("queued")
	}
	refusesAll(t, func(p vision.TowerBlock) error {
		_, err := ve.walkBlock(p, L, nil, 0, proj, func(string) {})
		return err
	})
}

// TestVisionEncoderWalkBlock_refusesMisroutedOpsW8A8: the same in its int8 form, whose matrices carry their shape.
func TestVisionEncoderWalkBlock_refusesMisroutedOpsW8A8(t *testing.T) {
	const hidden, inter = 8, 32
	ve := &VisionEncoder{w: vision.GPUWeights{Hidden: hidden, Inter: inter, NumHeads: 2, HeadDim: hidden / 2, NumPatches: 4}}
	sq := kernels.NewResidentW8A8Shape(hidden, hidden)
	L := &visLayer{qw: sq, kw: sq, vw: sq, ow: sq, fc1w: kernels.NewResidentW8A8Shape(inter, hidden), fc2w: kernels.NewResidentW8A8Shape(hidden, inter)}
	proj := func(*DeviceBuffer, *ResidentW8A8, *DeviceBuffer, *DeviceBuffer, int, int, int) (*DeviceBuffer, error) {
		panic("queued")
	}
	refusesAll(t, func(p vision.TowerBlock) error {
		_, err := ve.walkBlock(p, L, nil, 0, proj, func(string) {})
		return err
	})
}

// TestG4VTowerWalkBlock_refusesMisroutedOps: the same for the Gemma 4 tower's WebGPU walker.
func TestG4VTowerWalkBlock_refusesMisroutedOps(t *testing.T) {
	const hidden, inter = 8, 32
	a := &g4vTower{towerOps: &towerOps{}, cfg: vision.Gemma4EncoderConfig{HiddenSize: hidden, IntermediateSize: inter, NumAttentionHeads: 2, HeadDim: hidden / 2}}
	hp := g4vProj{out: hidden, in: hidden}
	ly := &g4vLayer{q: hp, k: hp, v: hp, o: hp, gate: g4vProj{out: inter, in: hidden}, up: g4vProj{out: inter, in: hidden}, down: g4vProj{out: hidden, in: inter}}
	refusesAll(t, func(p vision.TowerBlock) error { return a.walkBlock(p, ly, nil, nil, nil, 4, func(string) {}) })
}
