//go:build gpu && goinfer_testhooks

package gpu

import (
	"context"
	"math"
	"testing"

	"github.com/oliverbestmann/webgpu/wgpu"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/gpu/internal/kernels"
	"github.com/townsendmerino/goinfer/internal/mem"
	"github.com/townsendmerino/goinfer/internal/memgate"
	"github.com/townsendmerino/goinfer/internal/sb2census"
)

// webgpuRounding names Vulkan's block allocator. The ledger charges each buffer its requested size, so today it forgives nothing: the
// blocks are outside what a row reads (webgpuVkBlocks prices them in the slot census).
var webgpuRounding = memgate.PowerOfTwo("Vulkan's power-of-two blocks")

// webgpuDenseVectorFixtures are the fixtures whose dense line reads over plan: a tied head uploads the table the plan counts, so the
// host-kept table that hides the vectors on other fixtures is not there to hide them.
var webgpuDenseVectorFixtures = []string{
	"selftestdata/gemma3-vl-tiny", "testdata/gemma1-tiny", "testdata/gemma1-tiny.gguf", "testdata/gemma3-vl-tiny", "testdata/gptoss_tiny.gguf", "testdata/qwen3asr-tiny",
}

// webgpuMemGaps is WebGPU's table of known reds. It may only shrink: the fix that removes a row lowers webgpuMemGapsPin.
var webgpuMemGaps = func() []memgate.Gap {
	var out []memgate.Gap
	for _, f := range webgpuDenseVectorFixtures {
		out = append(out, memgate.Gap{Row: f + " *", Line: mem.LineDense, What: "the plan's dense figure sums weight matrices: the norms, biases, rotation tables and small constant buffers are leased on dense and priced nowhere",
			Item: "MM8, dense vectors", Bytes: func(f map[string]int64) int64 { return f["vectors"] + f["tables"] }})
	}
	for _, f := range []string{"testdata/gemma4-moe-kv-tiny", "testdata/gemma4-moe-kv-tiny32"} {
		out = append(out, memgate.Gap{Row: f + " *", Line: mem.LineDense, What: "Gemma 4's MoE block leases router and expert tables on dense beside the vectors, priced nowhere: not diagnosed",
			Item: "MM8, Gemma 4 MoE dense", Bytes: func(f map[string]int64) int64 { return f["vectors"] + f["tables"] + f["expertsFull"]/2 }})
	}
	for _, f := range []string{"testdata/gemma4-moe-kv-tiny", "testdata/qwen3vlmoe-tiny"} {
		out = append(out, memgate.Gap{Row: f + " *", Line: mem.LineExperts, What: "the stacked experts lease more than the plan's expert figure: not diagnosed",
			Item: "MM8 WebGPU, stacked experts", Bytes: func(f map[string]int64) int64 { return f["expertsFull"] / 3 }})
	}
	return out
}()

const webgpuMemGapsPin = 10

// webgpuGatePlantForTest, when set, runs on a row's resident after its generation and before the ledger is read; the function it returns
// runs after that read, so the plant is released before Close and the package's own ledger check stays clean.
var webgpuGatePlantForTest func(rd *residentDecoder) (release func())

// webgpuGateRow loads fixture path at options o, runs it, and returns the row: the build's ledger after the build and after a short
// generation, the plan for the same load, and the ledger after Close. ok is false when WebGPU's own build declines.
func webgpuGateRow(t *testing.T, row, path string, o decoder.Options) (memgate.Reading, bool) {
	t.Helper()
	m, err := decoder.Load(path, o)
	if err != nil {
		return memgate.Reading{}, false
	}
	closed := false
	defer func() {
		if !closed {
			m.Close()
		}
	}()
	rd, ok := m.ResidentForwardForTest().(*residentDecoder)
	if !ok {
		return memgate.Reading{}, false
	}
	l := rd.c.Ledger()
	read := memgate.Reading{Row: row, KVSlots: rd.KVSlots()}
	read.Plan = m.Plan("webgpu", math.MaxInt64/4, decoder.PlanRequest{Ctx: rd.ctxCap, CtxPinned: true, KVF16: o.KVPrecision == "f16", KVI8: o.KVPrecision == "i8"})
	// A rotation table a layer and a few constants: what the build uploads beside the weights, from the geometry alone.
	_, layers, _, _, headDim, _, _ := m.Dims()
	facts := map[string]int64{"vectors": memgate.DenseVectorBound(m.Config()), "tables": int64(layers) * int64(headDim*2+64), "expertsFull": read.Plan.ExpertBytesFull}
	read.Open = append(read.Open, memgate.Snap("built", l, facts))
	ch, g := m.Generate(context.Background(), []int{1, 2, 3, 4, 5}, 4, decoder.SamplingParams{})
	for range ch {
	}
	if err := g.Err(); err != nil {
		t.Errorf("%s: generation: %v", row, err)
	}
	release := func() {}
	if webgpuGatePlantForTest != nil {
		release = webgpuGatePlantForTest(rd)
	}
	read.Open = append(read.Open, memgate.Snap("generated", l, facts))
	release()
	// Scratch has no plan line: it is the headroom WebGPU's slot rule leaves on the device for the per-call buffers (webgpuSlotHeadroom).
	read.Named = map[mem.LineID]int64{mem.LineScratch: webgpuSlotHeadroom}
	closed = true
	m.Close()
	read.Closed = l.Usage()
	return read, true
}

// TestMemGate_webgpu is MM7's gate on WebGPU (docs/tasks/task-subsystems-2026-10.md): every census fixture at every option cell whose
// build goes resident, and each at int4 with two KV slots. It runs every row in one test, since a filtered run would see known gaps go quiet.
func TestMemGate_webgpu(t *testing.T) {
	requireGPU(t, nil)
	t.Setenv("GOINFER_SSM_RESIDENT", "")
	d, err := sb2census.ReadLayerD("..", "webgpu")
	if err != nil {
		t.Fatal(err)
	}
	g := memgate.Gate{Allowance: webgpuRounding, Gaps: webgpuMemGaps, GapsPin: webgpuMemGapsPin}
	built, declined := 0, 0
	run := func(row, path string, o decoder.Options) {
		o.Backend, o.ResidentContext = "webgpu", 256
		read, ok := webgpuGateRow(t, row, path, o)
		if !ok {
			declined++
			return
		}
		built++
		g.Check(t, read)
	}
	for _, f := range d.Fixtures {
		path, err := sb2census.Path("..", f)
		if err != nil {
			t.Fatal(err)
		}
		for _, c := range sb2census.Cells {
			if !d.Resident[f+" "+c.Name] {
				continue
			}
			o := decoder.Options{Quant: c.Quant, ActQuantGroup: c.ActGroup, KVPrecision: c.KVPrec, KVQuant: c.KVQuant}
			if c.SSM != "" {
				o.Knobs = &decoder.Knobs{"GOINFER_SSM_RESIDENT": c.SSM}
			}
			run(f+" "+c.Name, path, o)
		}
		if d.Resident[f+" int4"] {
			run(f+" int4/slots2", path, decoder.Options{Quant: "int4", ResidentKVSlots: 2})
		}
	}
	t.Logf("%d builds held, %d options declined by WebGPU's own build", built, declined)
	if built < 100 {
		t.Errorf("only %d builds held: a gate that read little covers little", built)
	}
	g.Finish(t)
}

// TestMemGate_webgpuPlantsThroughTheTable: a buffer really allocated over plan after the generation stays red through the table, on a line
// it forgives part of and on lines it forgives nothing of. A lease left open after Close is held by memgate's own test and the CPU's plant:
// here it would fail this package's ledger check at exit.
func TestMemGate_webgpuPlantsThroughTheTable(t *testing.T) {
	requireGPU(t, nil)
	o := decoder.Options{Backend: "webgpu", Quant: "int4", ResidentContext: 256}
	const fx, row = "../testdata/llama-tiny", "testdata/llama-tiny int4"
	g := memgate.Gate{Allowance: webgpuRounding, Gaps: webgpuMemGaps, GapsPin: webgpuMemGapsPin}
	clean, ok := webgpuGateRow(t, row, fx, o)
	if !ok {
		t.Fatal("llama-tiny did not build resident")
	}
	if u := g.Unknown(clean); len(u) != 0 {
		t.Fatalf("the unplanted row is red through the table: %v", u)
	}
	for _, line := range []mem.LineID{mem.LineDense, mem.LineKV, mem.LineState, mem.LineExperts} {
		t.Run(string(line), func(t *testing.T) {
			ran := false
			webgpuGatePlantForTest = func(rd *residentDecoder) func() {
				extra, err := kernels.CreateBufferOn(rd.c, string(line), &wgpu.BufferDescriptor{Label: "gate-plant", Size: 64 << 20, Usage: wgpu.BufferUsageStorage})
				if err != nil {
					t.Fatal(err)
				}
				ran = true
				return func() { kernels.ReleaseBuffer(extra) }
			}
			defer func() { webgpuGatePlantForTest = nil }()
			read, ok := webgpuGateRow(t, row, fx, o)
			if !ok {
				t.Fatal("llama-tiny did not build resident")
			}
			if !ran {
				t.Fatal("the plant did not run")
			}
			red := false
			for _, f := range g.Unknown(read) {
				red = red || (f.Line == line && f.Kind == memgate.OverPlan)
			}
			if !red {
				t.Errorf("64 MiB on %q is not red through the table: %v", line, g.Unknown(read))
			}
		})
	}
}
