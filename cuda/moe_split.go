//go:build cuda

package cuda

import (
	"fmt"
	"runtime"
	"unsafe"

	gpu "github.com/townsendmerino/aikit/gpu"
)

// The host share of a Gemma 4 MoE layer's expert-cache misses (docs/tasks/task-multimodal-support-2026-10.md, S20 step 4).
// A miss the DMA path would copy over PCIe can instead be computed on the CPU from the pinned host stack the copy reads,
// while the GPU runs the layer's other experts. The CPU's products are bit-identical to the GPU's: the two GEMVs are host
// replicas of the kernels (moe_cpu.go) and the GELU between them runs on the device (glu_quant, whose tanhf the host cannot
// reproduce), on a second queue that reads and writes mapped host memory. The device experts store their products
// (gemv_w4a8_moe_res) and moe_combine accumulates every product in expert order, the chain gemv_w4a8_moe_wacc leaves.

// moeSplit is the resident's host-expert state, nil when the split is off.
type moeSplit struct {
	pool *moeCPUPool
	side Queue

	res, stage, combine Pipeline

	// device: every rank's down product (the combine's input) and the host experts' GELU scratch
	resD, scrD Buffer
	// mapped host memory: the expert input, the host experts' gate/up products, their GELU outputs, their down products
	hMq, hMs, hGU, hQ, hSc, hRes *gpu.MappedHostBuffer

	evMq, evGlu, evRes gpu.Event

	// cpu[j] marks the ranks this layer computes on the host; n counts them
	cpu []bool
	n   int

	// the plan's costs in microseconds: a copied miss, one host expert, and the fixed cost of a layer with any host expert
	// (its two round trips); and force, a test's override of the plan (nil: the plan decides)
	dmaUS, cpuUS, fixedUS float64
	force                 func(misses int) int

	// hostExperts counts the experts computed on the host since the build
	hostExperts uint64
}

// moeSplitDefaultCosts are the plan's costs, measured on the RTX 2070 SUPER and a Ryzen 7 3700X (G-S20a, G-S20j).
const (
	moeSplitDMAUS   = 230
	moeSplitCPUUS   = 135
	moeSplitFixedUS = 122
)

// moeSplitEligible says whether this resident can split its MoE layers: the Gemma 4 expert loop under C′ with the DMA overlap
// (the host share rides its events), per-vector activation scales (the host GEMVs read one), the AVX2 replica, and no graphs.
func (r *cudaResident) moeSplitEligible() bool {
	if !r.cacheExperts || !r.overlap || r.graphs || r.actG32 || !moeCPUAsm || r.gptOssSw != (Pipeline{}) {
		return false
	}
	for i := range r.layers {
		if r.layers[i].g4moe {
			return r.hidden%32 == 0 && r.moeInter%32 == 0
		}
	}
	return false
}

// moeSplitWorkers is the host pool's size: four workers read a cold expert fastest on the 3700X's two memory channels, and
// more contend (G-S20j: 128 µs an expert at four, about 200 at fifteen).
const moeSplitWorkers = 4

// initMoESplit builds the split's state on the executor.
func (r *cudaResident) initMoESplit() (err error) {
	s := &moeSplit{dmaUS: moeSplitDMAUS, cpuUS: moeSplitCPUUS, fixedUS: moeSplitFixedUS, cpu: make([]bool, r.topK)}
	mod, err := r.dev.CompileLibrary(moeSplitPTX)
	if err != nil {
		return fmt.Errorf("moe_split module: %w", err)
	}
	for _, f := range []struct {
		dst  *Pipeline
		name string
	}{{&s.res, "gemv_w4a8_moe_res"}, {&s.stage, "moe_act_to_host"}, {&s.combine, "moe_combine"}} {
		if *f.dst, err = r.dev.NewComputePipeline(mod, f.name); err != nil {
			return fmt.Errorf("moe_split %s: %w", f.name, err)
		}
	}
	k, h, in := r.topK, r.hidden, r.moeInter
	mapped := []struct {
		dst *(*gpu.MappedHostBuffer)
		n   int
	}{{&s.hMq, h}, {&s.hMs, 4}, {&s.hGU, k * 2 * in * 4}, {&s.hQ, k * in}, {&s.hSc, k * 4}, {&s.hRes, k * h * 4}}
	for _, m := range mapped {
		if *m.dst, err = r.dev.NewMappedHostBuffer(m.n); err != nil {
			s.closeMapped()
			return fmt.Errorf("moe split mapped buffer: %w", err)
		}
	}
	s.resD = r.af(k * h)
	s.scrD = r.af(k * in)
	s.side = r.dev.NewCommandQueue()
	for _, ev := range []*gpu.Event{&s.evMq, &s.evGlu, &s.evRes} {
		if *ev, err = r.dev.NewEvent(); err != nil {
			s.closeMapped()
			return err
		}
	}
	s.pool = newMoECPUPool(min(moeSplitWorkers, max(1, runtime.NumCPU()-1)))
	r.split = s
	return nil
}

// close frees the split's host resources: its workers and its page-locked memory (before the context, like the stacks).
func (s *moeSplit) close() {
	if s == nil {
		return
	}
	if s.pool != nil {
		s.pool.close()
	}
	s.closeMapped()
}

func (s *moeSplit) closeMapped() {
	for _, m := range []*gpu.MappedHostBuffer{s.hMq, s.hMs, s.hGU, s.hQ, s.hSc, s.hRes} {
		if m != nil {
			_ = m.Close()
		}
	}
}

// plan is how many of a layer's misses the host computes: the count that minimises the layer's miss time, the copies and the
// host's work running at once, with the host's fixed cost paid only when it takes any.
func (s *moeSplit) plan(misses int) int {
	if s.force != nil {
		return min(max(s.force(misses), 0), misses)
	}
	best, bestT := 0, s.dmaUS*float64(misses)
	for c := 1; c <= misses; c++ {
		t := max(s.dmaUS*float64(misses-c), s.fixedUS+s.cpuUS*float64(c))
		if t < bestT {
			best, bestT = c, t
		}
	}
	return best
}

// markHostRanks chooses this layer's host experts from the router's picks before any is admitted: the last c misses in rank
// order, so the copies of the first ones start earliest. It clears the marks when the split is off or takes none.
func (r *cudaResident) markHostRanks(c *expertCache) {
	s := r.split
	if s == nil {
		return
	}
	for j := range s.cpu {
		s.cpu[j] = false
	}
	s.n = 0
	misses := 0
	for j := 0; j < r.topK; j++ {
		if c.slotOf[r.hostIdx[j]] < 0 {
			misses++
		}
	}
	take := s.plan(misses)
	for j := r.topK - 1; j >= 0 && s.n < take; j-- {
		if c.slotOf[r.hostIdx[j]] < 0 {
			s.cpu[j] = true
			s.n++
		}
	}
}

// hostRows views w's pinned host stack as rows of expert e: its packed words and its f16 scales.
func hostRows(w *cudaWQ, e int) ([]uint32, []uint16) {
	wb, sb := w.srcW.Bytes(), w.srcS.Bytes()
	words := unsafe.Slice((*uint32)(unsafe.Pointer(&wb[0])), len(wb)/4)
	scales := unsafe.Slice((*uint16)(unsafe.Pointer(&sb[0])), len(sb)/2)
	return words[e*w.perExpertW : (e+1)*w.perExpertW], scales[e*w.perExpertS : (e+1)*w.perExpertS]
}

func f32View(m *gpu.MappedHostBuffer) []float32 {
	b := m.Bytes()
	return unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), len(b)/4)
}

func i32View(m *gpu.MappedHostBuffer) []int32 {
	b := m.Bytes()
	return unsafe.Slice((*int32)(unsafe.Pointer(&b[0])), len(b)/4)
}

// gemma4MoeMLPPostSplit is gemma4MoeMLPPost for a layer with host experts (markHostRanks). The device experts are queued
// first, so the GPU runs them while the host computes its own; the combine then waits, on the device, for the host's products.
func (r *cudaResident) gemma4MoeMLPPostSplit(Ly *cudaLayer, l int, x Buffer) error {
	s := r.split
	k, h, in, gu := r.topK, r.hidden, r.moeInter, 2*r.moeInter
	// the expert input to the host
	if e := r.launch(s.stage, g1cfg(h/4, 256), Arg(r.mq), Arg(r.mSc), gpu.ArgValue(int32(h/4)), Arg(s.hMq.Buffer()), Arg(s.hMs.Buffer())); e != nil {
		return e
	}
	if e := r.stream.Record(s.evMq); e != nil {
		return e
	}
	// the device experts: gate/up, GELU, down product into resD
	for j := 0; j < k; j++ {
		if s.cpu[j] {
			continue
		}
		if e := r.waitMiss(j); e != nil {
			return e
		}
		if e := r.launch(r.fMoEGemv, LaunchConfig{GridX: uint32((gu + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
			Arg(Ly.expGU.W), Arg(r.mq), Arg(Ly.expGU.ws16), Arg(r.mSc), Arg(r.expIdx()),
			gpu.ArgValue(int32(j)), gpu.ArgValue(int32(gu)), gpu.ArgValue(int32(gu)),
			gpu.ArgValue(int32(h/8)), gpu.ArgValue(int32(h/32)), Arg(r.moeGU)); e != nil {
			return e
		}
		if e := r.launchGluSplitExpert(r.moeGU, in, r.moeQ, r.moeSc, r.moeScr, r.expBiasArg(Ly), j); e != nil {
			return e
		}
		if e := r.launch(s.res, LaunchConfig{GridX: uint32((h + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
			Arg(Ly.expDown.W), Arg(r.moeQ), Arg(Ly.expDown.ws16), Arg(r.moeSc), Arg(r.expIdx()),
			gpu.ArgValue(int32(j)), gpu.ArgValue(int32(h)), gpu.ArgValue(int32(h)),
			gpu.ArgValue(int32(in/8)), gpu.ArgValue(int32(in/32)), Arg(s.resD)); e != nil {
			return e
		}
	}
	// the host experts' gate/up products
	if e := s.evMq.Sync(); e != nil {
		return e
	}
	aScale := f32View(s.hMs)[0]
	actGU := newMoEActTables(i32View(s.hMq)[:h/4])
	guOut := f32View(s.hGU)
	tasks := make([]moeCPUTask, 0, s.n)
	for j := 0; j < k; j++ {
		if s.cpu[j] {
			w, sc := hostRows(&Ly.expGU, int(r.hostIdx[j]))
			tasks = append(tasks, moeCPUTask{tab: &actGU, w: w, s: sc, out: guOut[j*gu : (j+1)*gu]})
		}
	}
	s.pool.run(tasks)
	for _, t := range tasks {
		for i := range t.out {
			t.out[i] = float32(t.out[i] * aScale)
		}
	}
	// their GELU on the second queue, from and into mapped memory
	for j := 0; j < k; j++ {
		if !s.cpu[j] {
			continue
		}
		g := s.hGU.Buffer().At(j * gu * 4)
		if e := s.side.Launch(r.fSw, onecfg(glueQuantThreads, glueQuantThreads*4), Arg(g), Arg(g), gpu.ArgValue(int32(0)),
			gpu.ArgValue(int32(in)), gpu.ArgValue(int32(in)), gpu.ArgValue(r.act), Arg(s.hQ.Buffer().At(j*in)),
			Arg(s.hSc.Buffer().At(j*4)), Arg(s.scrD.At(j*in*4))); e != nil {
			return e
		}
	}
	if e := s.side.Record(s.evGlu); e != nil {
		return e
	}
	if e := s.evGlu.Sync(); e != nil {
		return e
	}
	// their down products
	q, sc, resOut := i32View(s.hQ), f32View(s.hSc), f32View(s.hRes)
	acts := make([]moeActTables, k)
	tasks = tasks[:0]
	for j := 0; j < k; j++ {
		if s.cpu[j] {
			acts[j] = newMoEActTables(q[j*in/4 : (j+1)*in/4])
			w, scl := hostRows(&Ly.expDown, int(r.hostIdx[j]))
			tasks = append(tasks, moeCPUTask{tab: &acts[j], w: w, s: scl, out: resOut[j*h : (j+1)*h]})
		}
	}
	s.pool.run(tasks)
	for j := 0; j < k; j++ {
		if !s.cpu[j] {
			continue
		}
		o := resOut[j*h : (j+1)*h]
		for i := range o {
			o[i] = float32(sc[j] * o[i])
		}
		if e := s.side.UploadAsyncAtFrom(s.resD.At(j*h*4), s.hRes, j*h*4, h*4); e != nil {
			return e
		}
	}
	s.hostExperts += uint64(s.n)
	if e := s.side.Record(s.evRes); e != nil {
		return e
	}
	if e := r.stream.Wait(s.evRes); e != nil {
		return e
	}
	if e := r.launch(s.combine, g1cfg(h, 256), Arg(s.resD), Arg(r.rWgt), gpu.ArgValue(int32(k)), gpu.ArgValue(int32(h)), Arg(r.g4x2)); e != nil {
		return e
	}
	return r.gemma4MoeJoin(Ly, l, x)
}
