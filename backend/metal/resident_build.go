//go:build darwin

package metal

import (
	"fmt"
	"strings"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

// residentBuild is buildResident's state: what each build phase reads from the phases before it. A phase is a method that
// runs in build order and returns the error that declines the build; buildResident calls them in sequence, and the device's
// ledgers (buildResident's deferred release) own whatever a phase allocated when a later one fails.
type residentBuild struct {
	m            *decoder.Model
	bv           decoder.BackendModelView
	d            *Device
	r            *resident
	shape        program.Shape
	pipe         func(name string) Pipeline
	kvSlots      int
	stepHeadroom int64
	attnSoftcap  float32
}

// compileKernels compiles the kernel library and creates the resident, with its dimensions, context cap and the memory-guard readings
// the later phases size from. It sets b.bv, b.pipe and b.r.
func (b *residentBuild) compileKernels() error {
	m, d, kvSlots, stepHeadroom := b.m, b.d, b.kvSlots, b.stepHeadroom
	// preciseMathCompile (a test and measurement toggle) turns fast-math off, which removes the compiler's
	// contraction, reassociation and transcendental discretion: the axis that makes within-machine bit-identity
	// fragile to an OS toolchain update. The default is fast-math: precise costs a few percent of decode speed and
	// does not improve CPU parity (docs/ollama-chase.md §A2-Metal). GOINFER_PRECISE_MATH is a documented opt-in for
	// bits robust to a toolchain update; the snapshot golden otherwise detects such drift.
	compile := d.CompileLibrary
	preciseMath := preciseMathCompile || modelKnob(m, "GOINFER_PRECISE_MATH") != "" || w8PreciseMath(m)
	if preciseMath {
		compile = d.CompileLibraryPrecise
	}
	bv := m.BackendView()
	attnSoftcap := bv.AttnLogitSoftcap()
	if attnSoftcapDropForTest {
		attnSoftcap = 0
	}
	libSrc, err := withAttnSoftcap(allKernels+mc3RowsKernels+kvF32Kernels+kvRowsKernels, attnSoftcap) // + MC3 S3's multi-row forms (batch_rows.go), attention_f32 (kv_f32.go) and the step's int8 and f32 KV forms (kv_rows.go), derived from allKernels
	if ksc2OldDownForTest {
		libSrc = "#define KSC2_OLD_DOWN 1\n" + libSrc
	}
	if p := mambaSrcPlantForTest; err == nil && p[0] != "" {
		libSrc = strings.Replace(libSrc, p[0], p[1], 1)
	}
	if err != nil {
		return err
	}
	lib, err := compile(libSrc, MSL3_1)
	if err != nil {
		return err
	}
	pipe := func(name string) Pipeline {
		p, e := d.NewComputePipeline(lib, name)
		if e != nil {
			panic(fmt.Sprintf("metal pipeline %s: %v", name, e))
		}
		return p
	}
	H, nL, nH, _, _, I, V := bv.Dims() // model-level hd/nKV dropped — geometry is per-layer (geom.go)
	r := &resident{knob: m.Knob, prog: m.Program(), d: d, H: H, nL: nL, nH: nH, I: I, V: V, preciseMath: preciseMath, stepHeadroom: stepHeadroom,
		moeStep: m.ResidentMoEStep(), recurrentStep: m.ResidentRecurrentStepOn(kvSlots), gemma4Step: m.ResidentGemma4StepOn(kvSlots), ckptSlot: -1}
	r.chainEmbBase = residentNeedBytes(m) + int64(kvSlots-1)*metalSlotBytes(m) // every slot's KV and recurrent state
	r.chainEmbGuardOff = modelKnob(m, "GOINFER_NO_RESIDENT_MEM_GUARD") != ""
	if r.ctxCap, err = resolveMetalCtxCap(m); err != nil {
		return err
	}
	b.bv, b.pipe, b.r, b.attnSoftcap = bv, pipe, r, attnSoftcap
	return nil
}
