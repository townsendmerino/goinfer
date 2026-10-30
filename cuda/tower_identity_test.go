//go:build cuda

package cuda

import (
	"errors"
	"flag"
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/towerid"
	"github.com/townsendmerino/goinfer/internal/vision"
)

// TestTowerIdentity_cuda is AX2a-T on CUDA (docs/tasks/task-subsystems-2026-10.md §7.8): every tower CUDA declares, and
// the Whisper encoders, attach through their production registries and give the same bits twice.
func TestTowerIdentity_cuda(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	d.ReleaseObjects()
	towerid.Check(t, "..", "cuda", os.Getenv("GOINFER_TOWER_ID_OUT"))
}

var towerUpdate = flag.Bool("towerid.update", false, "rewrite this platform's CUDA tower golden (review the diff)")

// TestTowerGolden_cuda is G-SB9-1, G-SB9-2, G-SB9-3 and G-SB9M-2 on CUDA: the record equals the committed golden recorded on this GPU
// and NVIDIA driver (another fails until it is re-anchored), each family's plant turns exactly its own cases red, each
// plant in SigLIP's block program turns exactly the families that walk it red, and each device case agrees with the CPU within the registered bound.
func TestTowerGolden_cuda(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	bus, err := d.PCIBusID()
	d.ReleaseObjects()
	if err != nil {
		t.Fatalf("the golden's anchor cannot name the device under test: %v", err)
	}
	// Scoped by PCI bus id, so on a host with several GPUs the anchor is the device under test's, whatever CUDA's ordinal order
	// and CUDA_VISIBLE_DEVICES make of it.
	smi, err := exec.Command("nvidia-smi", "-i", bus, "--query-gpu=name,driver_version", "--format=csv,noheader").Output()
	if err != nil {
		t.Fatalf("the golden's anchor (GPU and driver) cannot be read: nvidia-smi -i %s: %v", bus, err)
	}
	if n := strings.Count(strings.TrimSpace(string(smi)), "\n"); n != 0 {
		t.Fatalf("nvidia-smi -i %s named %d GPUs, want the one under test: %q", bus, n+1, smi)
	}
	anchor := "cuda " + strings.TrimSpace(string(smi))
	rec := towerid.Check(t, "..", "cuda", "")
	golden := towerid.CheckGolden(t, "..", rec, anchor, towerid.FailOtherAnchor, *towerUpdate)
	towerid.CheckPlants(t, "..", "cuda", golden)
	towerid.CheckProgramPlants(t, "..", "cuda", golden, "siglip", "qwen3", "glm-ocr", "gemma4", "qwen2.5", "whisper")
	towerid.CompareBackends(t, "..", "cuda")
}

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

// TestInt8SiglipWalkBlock_refusesMisroutedOps: the int8 SigLIP tower's walker refuses each misrouted op before it queues
// anything. Its queue is nil and its register allocation and attention panic, so an op that reached one would panic.
func TestInt8SiglipWalkBlock_refusesMisroutedOps(t *testing.T) {
	const hidden, inter = 8, 32
	r := &VisionEncoder{hidden: hidden, inter: inter, numHeads: 2, headDim: hidden / 2}
	hp := cudaWQ{N: hidden, K: hidden}
	L := &visionLayer{qw: hp, kw: hp, vw: hp, ow: hp, fc1w: cudaWQ{N: inter, K: hidden}, fc2w: cudaWQ{N: hidden, K: inter}}
	attend := func(_, _, _, _ Buffer, _ float32) error { panic("queued") }
	for _, c := range vision.MisroutedTowerOpsForTest() {
		s := &int8Regs{f: map[vision.TowerReg]Buffer{}, alloc: func(int) [2]Buffer { panic("queued") }}
		for n := vision.TowerX; n <= vision.TowerMid2; n++ {
			s.f[n] = Buffer{}
		}
		err := func() (err error) {
			defer func() {
				if recover() != nil {
					err = errQueued
				}
			}()
			return r.walkBlock(nil, vision.TowerBlock{c.Op}, L, s, 4, attend)
		}()
		switch {
		case errors.Is(err, errQueued):
			t.Errorf("%s: queued instead of refused", c.Name)
		case err == nil:
			t.Errorf("%s: accepted", c.Name)
		}
	}
}

// errQueued marks an op a walker queued (and panicked in) instead of refusing.
var errQueued = errors.New("queued")

// int8TestRegs are the int8 tower's registers as ForwardPatches builds them: float buffers for x, q, k, v, ctx and mid, none
// for xn or mid2, and an allocation that panics, so a walker that reached it fails as queued.
func int8TestRegs() *int8Regs {
	f := map[vision.TowerReg]Buffer{}
	for _, n := range []vision.TowerReg{vision.TowerX, vision.TowerQ, vision.TowerK, vision.TowerV, vision.TowerCtx, vision.TowerMid} {
		f[n] = Buffer{}
	}
	return &int8Regs{f: f, alloc: func(int) [2]Buffer { panic("queued") }}
}

// TestInt8SiglipWalkBlock_refusesInvalidBlocks: whole blocks the int8 tower cannot run are refused before anything is
// queued. Each case is refused by a different check: a quantized write no linear reads, x written in int8, a later op of
// the block (the whole block is checked before its first launch), a float read of a register an earlier block left int8
// only, and a float read of a register with no float buffer.
func TestInt8SiglipWalkBlock_refusesInvalidBlocks(t *testing.T) {
	const hidden, inter = 8, 32
	r := &VisionEncoder{hidden: hidden, inter: inter, numHeads: 2, headDim: hidden / 2}
	hp := cudaWQ{N: hidden, K: hidden}
	L := &visionLayer{qw: hp, kw: hp, vw: hp, ow: hp, fc1w: cudaWQ{N: inter, K: hidden}, fc2w: cudaWQ{N: hidden, K: inter}}
	attend := func(_, _, _, _ Buffer, _ float32) error { panic("queued") }
	norm := func(in, out vision.TowerReg) vision.TowerOp {
		return vision.TowerOp{Kind: vision.TowerNorm, In: in, Out: out, W: vision.TowerNorm1, Form: vision.TowerLayerNorm, Eps: 1e-6}
	}
	lin := func(in, out vision.TowerReg, w vision.TowerWeight) vision.TowerOp {
		return vision.TowerOp{Kind: vision.TowerLinear, In: in, Out: out, W: w}
	}
	gelu := func(out vision.TowerReg) vision.TowerOp { return vision.TowerOp{Kind: vision.TowerGELUTanh, Out: out} }
	for _, c := range []struct {
		name  string
		prog  vision.TowerBlock
		state func(*int8Regs)
	}{
		{"a norm whose output no linear reads", vision.TowerBlock{norm(vision.TowerX, vision.TowerXN)}, nil},
		{"the GELU written into x, which a linear then reads", vision.TowerBlock{gelu(vision.TowerX), lin(vision.TowerX, vision.TowerQ, vision.TowerWQ)}, nil},
		{"a valid linear, then a norm no linear reads", vision.TowerBlock{lin(vision.TowerX, vision.TowerQ, vision.TowerWQ), norm(vision.TowerX, vision.TowerXN)}, nil},
		{"the GELU on mid an earlier block left int8 only", vision.TowerBlock{gelu(vision.TowerMid), lin(vision.TowerMid, vision.TowerQ, vision.TowerWFC2)},
			func(s *int8Regs) {
				s.init()
				s.int8[vision.TowerMid], s.stale[vision.TowerMid] = true, true
			}},
		{"a norm from xn, which has no float buffer", vision.TowerBlock{norm(vision.TowerXN, vision.TowerQ), lin(vision.TowerQ, vision.TowerK, vision.TowerWK)}, nil},
	} {
		s := int8TestRegs()
		if c.state != nil {
			c.state(s)
		}
		err := func() (err error) {
			defer func() {
				if recover() != nil {
					err = errQueued
				}
			}()
			return r.walkBlock(nil, c.prog, L, s, 4, attend)
		}()
		switch {
		case errors.Is(err, errQueued):
			t.Errorf("%s: queued instead of refused", c.name)
		case err == nil:
			t.Errorf("%s: accepted", c.name)
		}
	}
}

// TestInt8SiglipPlanBlock_productionProgram: SigLIP's program plans on ForwardPatches' registers for two blocks running,
// the first block's forms carried into the second, and leaves x float with xn and mid held int8.
func TestInt8SiglipPlanBlock_productionProgram(t *testing.T) {
	const hidden, inter = 8, 32
	r := &VisionEncoder{hidden: hidden, inter: inter, numHeads: 2, headDim: hidden / 2}
	hp := cudaWQ{N: hidden, K: hidden}
	L := &visionLayer{qw: hp, kw: hp, vw: hp, ow: hp, fc1w: cudaWQ{N: inter, K: hidden}, fc2w: cudaWQ{N: hidden, K: inter}}
	prog := vision.SiglipProgram(vision.EncoderConfig{HiddenSize: hidden, IntermediateSize: inter, NumAttentionHeads: 2, LayerNormEps: 1e-6})
	s := int8TestRegs()
	s.init()
	for block := range 2 {
		_, int8, stale, err := r.planBlock(prog, L, s)
		if err != nil {
			t.Fatalf("block %d: %v", block, err)
		}
		if stale[vision.TowerX] || !int8[vision.TowerXN] || !int8[vision.TowerMid] || !stale[vision.TowerMid] {
			t.Fatalf("block %d leaves int8 %v, stale %v; want x float, xn and mid int8", block, int8, stale)
		}
		s.int8, s.stale = int8, stale
	}
}
