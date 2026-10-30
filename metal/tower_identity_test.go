//go:build darwin

package metal

import (
	"flag"
	"fmt"
	"os"
	"syscall"
	"testing"

	"github.com/townsendmerino/goinfer/internal/towerid"
	"github.com/townsendmerino/goinfer/internal/vision"
)

// TestTowerIdentity_metal is AX2a-T on Metal (docs/tasks/task-subsystems-2026-10.md §7.8): every tower Metal declares
// attaches through its production registry and gives the same bits twice.
func TestTowerIdentity_metal(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no Metal device: %v", err)
	}
	d.ReleaseObjects()
	towerid.Check(t, "..", "metal", os.Getenv("GOINFER_TOWER_ID_OUT"))
}

var towerUpdate = flag.Bool("towerid.update", false, "rewrite this platform's Metal tower golden (review the diff)")

// metalAnchor names what a Metal record's bits depend on: the device, and the macOS version and build, since Metal's shader
// compiler and driver ship with the OS.
func metalAnchor(device string) (string, error) {
	ver, err := syscall.Sysctl("kern.osproductversion")
	if err != nil {
		return "", err
	}
	build, err := syscall.Sysctl("kern.osversion")
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("metal %s, macOS %s (%s)", device, ver, build), nil
}

// TestTowerGolden_metal is G-SB9-1M, G-SB9-2M and G-SB9-3M (docs/tasks/task-subsystems-2026-10.md, "SB9 step S on the
// Mac") and G-SB9M-2 on Metal: the record equals the committed golden recorded on this device and macOS build (another
// fails until it is re-anchored), each family's plant turns exactly its own cases red, each plant in each block program Metal
// walks turns exactly the families that walk it red, and each device case agrees with the CPU within the registered bound.
func TestTowerGolden_metal(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no Metal device: %v", err)
	}
	name := d.Name()
	d.ReleaseObjects()
	anchor, err := metalAnchor(name)
	if err != nil {
		t.Fatalf("the golden's anchor (macOS version and build) cannot be read: %v", err)
	}
	rec := towerid.Check(t, "..", "metal", "")
	golden := towerid.CheckGolden(t, "..", rec, anchor, towerid.FailOtherAnchor, *towerUpdate)
	towerid.CheckPlants(t, "..", "metal", golden)
	towerid.CheckProgramPlants(t, "..", "metal", golden, "siglip", "qwen3", "glm-ocr", "gemma4", "qwen2.5")
	towerid.CompareBackends(t, "..", "metal")
}

// TestWalkBlock_refusesMisroutedOps holds walkBlock to the registers its operations really use: an op routed elsewhere is
// an error before anything is queued, never a dispatch that silently reads or writes another buffer.
func TestWalkBlock_refusesMisroutedOps(t *testing.T) {
	const hidden, inter = 8, 32
	a := &gridVAccel{hidden: hidden, inter: inter}
	a.heads, a.hd = 2, hidden/2
	hp := gvProj{out: hidden, in: hidden}
	b := &gvBlock{q: hp, k: hp, v: hp, fc1: gvProj{out: inter, in: hidden}, fc2: gvProj{out: hidden, in: inter}}
	for _, c := range []struct {
		name string
		op   vision.TowerOp
	}{
		{"attention into the residual", vision.TowerOp{Kind: vision.TowerAttention, Out: vision.TowerX}},
		{"GELU from another register", vision.TowerOp{Kind: vision.TowerGELUTanh, In: vision.TowerX, Out: vision.TowerMid}},
		{"fc1 into a hidden-width register", vision.TowerOp{Kind: vision.TowerLinear, In: vision.TowerXN, Out: vision.TowerQ, W: vision.TowerWFC1}},
		{"fc2 from a hidden-width register", vision.TowerOp{Kind: vision.TowerLinear, In: vision.TowerXN, Out: vision.TowerX, W: vision.TowerWFC2}},
		{"fc1 added to the residual", vision.TowerOp{Kind: vision.TowerLinearAdd, In: vision.TowerXN, Out: vision.TowerX, W: vision.TowerWFC1}},
		{"fc1 added into mid", vision.TowerOp{Kind: vision.TowerLinearAdd, In: vision.TowerXN, Out: vision.TowerMid, W: vision.TowerWFC1}},
		{"a linear with no input register", vision.TowerOp{Kind: vision.TowerLinear, Out: vision.TowerQ, W: vision.TowerWQ}},
		{"a norm into mid", vision.TowerOp{Kind: vision.TowerNorm, In: vision.TowerX, Out: vision.TowerMid, W: vision.TowerNorm1, Form: vision.TowerLayerNorm}},
		{"a norm from mid", vision.TowerOp{Kind: vision.TowerNorm, In: vision.TowerMid, Out: vision.TowerXN, W: vision.TowerNorm1, Form: vision.TowerLayerNorm}},
		{"attention with an input register", vision.TowerOp{Kind: vision.TowerAttention, In: vision.TowerXN, Out: vision.TowerCtx}},
		{"a per-head norm on mid", vision.TowerOp{Kind: vision.TowerHeadNorm, Out: vision.TowerMid, W: vision.TowerQNorm, Form: vision.TowerRMSNorm}},
		{"a per-head norm on v", vision.TowerOp{Kind: vision.TowerHeadNorm, Out: vision.TowerV, W: vision.TowerQNorm, Form: vision.TowerRMSNorm}},
		{"RoPE on v", vision.TowerOp{Kind: vision.TowerRoPE, Out: vision.TowerV}},
		{"RoPE from another register", vision.TowerOp{Kind: vision.TowerRoPE, In: vision.TowerXN, Out: vision.TowerQ}},
		{"the fused qkv with an output register", vision.TowerOp{Kind: vision.TowerLinearQKV, In: vision.TowerXN, Out: vision.TowerX, W: vision.TowerWQKV}},
		{"the fused qkv from mid", vision.TowerOp{Kind: vision.TowerLinearQKV, In: vision.TowerMid, W: vision.TowerWQKV}},
		{"the SiLU product of mid with itself", vision.TowerOp{Kind: vision.TowerSiLUMul, In: vision.TowerMid, Out: vision.TowerMid}},
		{"the SiLU product from a hidden register", vision.TowerOp{Kind: vision.TowerSiLUMul, In: vision.TowerXN, Out: vision.TowerMid}},
	} {
		err := func() (err error) {
			defer func() {
				if recover() != nil {
					t.Errorf("%s: queued instead of refused", c.name)
				}
			}()
			mark := func(string) { panic("queued") } // every op marks its class before it dispatches
			return a.walkBlock(vision.TowerBlock{c.op}, b, 4, []int{0, 4}, nil, mark, nil)
		}()
		if err == nil {
			t.Errorf("%s: accepted", c.name)
		} else {
			t.Logf("%s: %v", c.name, err)
		}
	}
}

// TestRun_refusesEmptyProgram: a grid tower run with no block program is an error before anything is queued, never a
// tower that returns its patch embed as the hidden state.
func TestRun_refusesEmptyProgram(t *testing.T) {
	a := &gridVAccel{hidden: 8, blocks: make([]gvBlock, 2)}
	for _, prog := range []vision.TowerBlock{nil, {}} {
		err := func() (err error) {
			defer func() {
				if recover() != nil {
					t.Errorf("run with program %v: queued instead of refused", prog)
				}
			}()
			_, err = a.run(4, func(int) []int { return []int{0, 4} }, prog)
			return err
		}()
		if err == nil {
			t.Errorf("run with program %v: accepted", prog)
		}
	}
}
