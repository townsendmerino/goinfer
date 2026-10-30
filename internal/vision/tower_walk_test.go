package vision

import (
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

type shapeOnlyBlock map[TowerWeight][2]int // a linear's [out, in]

func (shapeOnlyBlock) towerNorm(TowerWeight) ([]float32, []float32, bool) {
	return make([]float32, 32), make([]float32, 32), true
}

func (b shapeOnlyBlock) towerLinear(w TowerWeight) (*linalg.WeightMat, []float32, bool) {
	s, ok := b[w]
	if !ok {
		return nil, nil, false
	}
	m := linalg.WrapF32(make([]float32, s[0]*s[1]), s[0], s[1])
	return &m, nil, true
}

// TestWalkTowerBlock_refusesMisroutedOps: the CPU walker refuses each misrouted op before its matmul or attention runs.
func TestWalkTowerBlock_refusesMisroutedOps(t *testing.T) {
	const rows, hidden, inter = 4, 8, 32
	b := shapeOnlyBlock{TowerWQKV: {3 * hidden, hidden}, TowerWO: {hidden, hidden}, TowerWFC1: {inter, hidden}, TowerWFC2: {hidden, inter},
		TowerWGate: {inter, hidden}, TowerWUp: {inter, hidden}, TowerWDown: {hidden, inter}}
	buf := func(n int) []float32 { return make([]float32, rows*n) }
	r := &towerRegs{x: buf(hidden), xn: buf(hidden), q: buf(hidden), k: buf(hidden), v: buf(hidden), ctx: buf(hidden),
		mid: buf(inter), mid2: buf(inter), qkv: buf(3 * hidden), tmp: buf(inter)}
	env := &towerEnv{rows: rows, hidden: hidden, heads: 2, inter: inter, cos: buf(hidden / 2), sin: buf(hidden / 2),
		matmul: func(*linalg.WeightMat, []float32, []float32, int) { panic("queued") },
		attend: func(_, _, _, _ []float32, _ float64) error { panic("queued") },
	}
	for _, c := range MisroutedTowerOpsForTest() {
		err := func() (err error) {
			defer func() {
				if recover() != nil {
					t.Errorf("%s: ran instead of refused", c.Name)
				}
			}()
			return walkTowerBlock(TowerBlock{c.Op}, b, r, env)
		}()
		if err == nil {
			t.Errorf("%s: accepted", c.Name)
		}
	}
}

// TestWalkTowerBlock_refusesShortRoPETables: RoPE and axial RoPE refuse tables shorter than rows·head_dim before they
// rotate anything, never a slice past the tables' end.
func TestWalkTowerBlock_refusesShortRoPETables(t *testing.T) {
	const rows, hidden, heads = 4, 8, 2
	short := make([]float32, rows*hidden/heads-1)
	env := &towerEnv{rows: rows, hidden: hidden, heads: heads, inter: 32, cos: short, sin: short}
	r := &towerRegs{q: make([]float32, rows*hidden)}
	for _, k := range []TowerOpKind{TowerRoPE, TowerRoPEAxial} {
		err := func() (err error) {
			defer func() {
				if recover() != nil {
					t.Errorf("%s: rotated past the tables instead of refusing", k)
				}
			}()
			return walkTowerBlock(TowerBlock{{Kind: k, Out: TowerQ}}, shapeOnlyBlock{}, r, env)
		}()
		if err == nil {
			t.Errorf("%s: accepted", k)
		}
	}
}
