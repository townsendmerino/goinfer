//go:build gpu

package kernels

import (
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
	sq := &DeviceBuffer{n: hidden * hidden}
	L := &visLayer{qf: sq, kf: sq, vf: sq, of: sq, fc1f: &DeviceBuffer{n: inter * hidden}, fc2f: &DeviceBuffer{n: hidden * inter}}
	proj := func(*DeviceBuffer, *ResidentW8A8, *DeviceBuffer, *DeviceBuffer, int, int, int) (*DeviceBuffer, error) {
		panic("queued")
	}
	refusesAll(t, func(p vision.TowerBlock) error {
		_, err := ve.walkBlock(p, L, nil, 0, proj, func(string) {})
		return err
	})
}
