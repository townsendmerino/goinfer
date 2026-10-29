//go:build cuda

package cuda

import (
	"testing"

	"github.com/townsendmerino/goinfer/internal/vision"
)

// TestGridTowerWalkBlock_refusesMisroutedOps: CUDA's tower walker refuses each misrouted op before it queues a kernel.
// The tower has no device, so an op that reached a dispatch would panic.
func TestGridTowerWalkBlock_refusesMisroutedOps(t *testing.T) {
	const hidden, inter = 8, 32
	a := &gridTower{hidden: hidden, inter: inter, heads: 2, hd: hidden / 2}
	hp := gridProj{out: hidden, in: hidden}
	b := &gridBlock{q: hp, k: hp, v: hp, proj: hp, fc1: gridProj{out: inter, in: hidden}, fc2: gridProj{out: hidden, in: inter},
		gate: gridProj{out: inter, in: hidden}, up: gridProj{out: inter, in: hidden}, down: gridProj{out: hidden, in: inter}}
	for _, c := range vision.MisroutedTowerOpsForTest() {
		err := func() (err error) {
			defer func() {
				if recover() != nil {
					t.Errorf("%s: queued instead of refused", c.Name)
				}
			}()
			return a.walkBlock(vision.TowerBlock{c.Op}, b, &towerRegs{}, 4, &towerWalkEnv{segs: []int{0, 4}})
		}()
		if err == nil {
			t.Errorf("%s: accepted", c.Name)
		}
	}
}
