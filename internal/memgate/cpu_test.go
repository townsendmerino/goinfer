//go:build goinfer_testhooks

package memgate

import (
	"flag"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/mem"
	"github.com/townsendmerino/goinfer/internal/sb2census"
)

// cpuGateCtx is the positions a CPU row writes: past every fixture's conv kernel, so each window is full.
const cpuGateCtx = 64

// cpuGateRowsPath names every row that must load and run, and cpuGateRowsUpdate has TestMemGate_cpu rewrite it.
const cpuGateRowsPath = "testdata/cpu_rows.txt"

var cpuGateRowsUpdate = flag.Bool("memgate.cpurows.update", false, "have TestMemGate_cpu rewrite testdata/cpu_rows.txt")

// cpuRounding is the CPU's allocator as a row sees it: the Go heap's size classes are not in what the walk leases.
var cpuRounding = PerLease("none: the walk leases each array's own bytes", 0)

// cpuDenseVectorFixtures are the fixtures whose dense line reads over plan on the bare vectors alone.
var cpuDenseVectorFixtures = []vectorFixture{
	{"internal/vision/testdata/qwen25vl-vision-tiny", 4, 38},
	{"selftestdata/gemma3-vl-tiny", 4, 46},
	{"selftestdata/llama-tiny", 2, 66},
	{"selftestdata/qwen35-tiny", 2, 90},
	{"selftestdata/qwen35vl-tiny", 3, 66},
	{"testdata/bailing_hybrid-tiny", 3, 76},
	{"testdata/cohere-tiny", 1, 65},
	{"testdata/cohere2-tiny", 1, 65},
	{"testdata/deepseek-tiny", 3, 82},
	{"testdata/gemma1-tiny", 2, 51},
	{"testdata/gemma1-tiny.gguf", 2, 45},
	{"testdata/gemma2-hd-tiny", 4, 68},
	{"testdata/gemma2-hd-tiny.gguf", 4, 60},
	{"testdata/gemma2-tiny", 4, 68},
	{"testdata/gemma3-vl-tiny", 4, 46},
	{"testdata/gguf-split/glm-tiny-00001-of-00004.gguf", 3, 35},
	{"testdata/glm-ocr-tiny", 4, 45},
	{"testdata/glm-tiny", 3, 52},
	{"testdata/glm-tiny-bias", 4, 63},
	{"testdata/glm-tiny.gguf", 3, 35},
	{"testdata/gptoss_tiny.gguf", 5, 50},
	{"testdata/granite-dense-tiny", 2, 66},
	{"testdata/internlm2-tiny", 2, 43},
	{"testdata/internlm3-tiny", 2, 51},
	{"testdata/laguna-m1-tiny", 3, 62},
	{"testdata/laguna-xs2-tiny", 3, 63},
	{"testdata/laguna-xs21-tiny", 3, 63},
	{"testdata/lfm2-tiny", 2, 77},
	{"testdata/llama-attnfa-tiny", 2, 0},
	{"testdata/llama-tiny", 2, 66},
	{"testdata/llama4-tiny", 2, 59},
	{"testdata/ministral3-tiny", 2, 66},
	{"testdata/mistral-tiny-window", 2, 44},
	{"testdata/mixtral-tiny", 2, 69},
	{"testdata/nemotron-tiny", 2, 83},
	{"testdata/nemotron3nano-tiny", 3, 71},
	{"testdata/olmo3-tiny", 4, 68},
	{"testdata/olmo_hybrid-tiny", 3, 63},
	{"testdata/phi3-tiny", 2, 43},
	{"testdata/qwen25vl-tiny", 4, 38},
	{"testdata/qwen35-tiny", 2, 90},
	{"testdata/qwen35vl-tiny", 3, 66},
	{"testdata/qwen3_5-tiny", 2, 88},
	{"testdata/qwen3_5-tiny-normw", 2, 88},
	{"testdata/qwen3_5_moe-tiny", 2, 93},
	{"testdata/qwen3asr-tiny", 3, 42},
	{"testdata/qwen3moe-tiny", 2, 77},
	{"testdata/qwen3moe-tiny-k3", 2, 77},
	{"testdata/qwen3next-tiny", 2, 126},
	{"testdata/qwen3vlmoe-tiny", 3, 54},
	{"testdata/smollm3-tiny", 2, 66},
	{"testdata/spark2-5-tiny", 2, 59},
	{"testdata/voxtral-tiny", 2, 38},
}

// cpuMemGaps is the CPU's table of known reds. It may only shrink: the fix that removes a row lowers cpuMemGapsPin.
var cpuMemGaps = append(denseVectorGaps(cpuDenseVectorFixtures), []Gap{
	gemma4MoEDenseGap("testdata/gemma4-moe-kv-tiny"), gemma4MoEDenseGap("testdata/gemma4-moe-kv-tiny32"),
	{Row: "testdata/gptoss_tiny.gguf *", Line: mem.LineExperts, What: "gpt-oss's expert biases are held with its experts, and the plan's expert figure sums weight matrices",
		Item: "MM8 CPU, gpt-oss experts", Bytes: func(map[string]int64) int64 { return 8192 }},
}...)

// cpuExpertFields are the fixtures whose experts line reads over plan on the struct scalars of their experts' weight matrices, and how many
// matrices that is.
var cpuExpertFields = []struct {
	row      string
	matrices int
}{
	{"selftestdata/qwen35-tiny", 48},
	{"testdata/bailing_hybrid-tiny", 72},
	{"testdata/deepseek-tiny", 72},
	{"testdata/gemma4-moe-kv-tiny", 32},
	{"testdata/gemma4-moe-kv-tiny32", 32},
	{"testdata/gguf-split/glm-tiny-00001-of-00004.gguf", 48},
	{"testdata/glm-tiny", 48},
	{"testdata/glm-tiny-bias", 48},
	{"testdata/glm-tiny.gguf", 48},
	{"testdata/gptoss_tiny.gguf", 48},
	{"testdata/laguna-m1-tiny", 48},
	{"testdata/laguna-xs2-tiny", 72},
	{"testdata/laguna-xs21-tiny", 72},
	{"testdata/llama4-tiny", 24},
	{"testdata/mixtral-tiny", 96},
	{"testdata/nemotron3nano-tiny", 24},
	{"testdata/qwen35-tiny", 48},
	{"testdata/qwen3_5_moe-tiny", 48},
	{"testdata/qwen3moe-tiny", 96},
	{"testdata/qwen3moe-tiny-k3", 96},
	{"testdata/qwen3next-tiny", 72},
	{"testdata/qwen3vlmoe-tiny", 96},
}

func init() {
	cpuMemGaps = append(cpuMemGaps, cpuKVGaps...)
	for _, e := range cpuExpertFields {
		cpuMemGaps = append(cpuMemGaps, Gap{Row: e.row + " *", Line: mem.LineExperts, What: "an expert's weight matrices hold struct scalars beside their arrays, which the plan's sum of arrays leaves out",
			Item: "MM8, weight matrix struct fields", Bytes: func(f map[string]int64) int64 { return int64(e.matrices) * f["matrixFields"] }})
	}
}

const cpuMemGapsPin = 108

// cpuKVGaps are the CPU plan's KV figures that a row reads wrong, each in MM8's row. The plan prices ctx positions of K and V on every layer.
var cpuKVGaps = func() []Gap {
	var out []Gap
	over := func(fixtures []string, what, item string) {
		for _, f := range fixtures {
			out = append(out, Gap{Row: f + " *", Line: mem.LineKV, OverPriced: true, What: what, Item: item, Bytes: KVPricedOver})
		}
	}
	over([]string{"testdata/cohere2-tiny", "testdata/gemma2-hd-tiny", "testdata/gemma2-hd-tiny.gguf", "testdata/gemma2-tiny", "selftestdata/gemma3-vl-tiny", "testdata/gemma3-vl-tiny",
		"testdata/gptoss_tiny.gguf", "testdata/laguna-xs2-tiny", "testdata/laguna-xs21-tiny", "testdata/mistral-tiny-window", "testdata/olmo3-tiny", "testdata/spark2-5-tiny"},
		"a sliding layer's ring holds its window, and the plan prices the whole context (at int8, int8's scale error nets inside this row's figure: Geometry names it apart)", "MM8 CPU, KV of a sliding window")
	over([]string{"testdata/bailing_hybrid-tiny", "testdata/deepseek-tiny"}, "an MLA layer holds one latent row, and the plan prices K and V", "MM8, MLA KV counted twice")
	over([]string{"testdata/llama-attnfa-tiny"}, "int8 K/V holds one scale a head, and the plan prices 1/8 of a byte an element, which over-prices a wide head", "MM8, int8 KV's scale")
	for _, f := range []string{"testdata/cohere-tiny", "testdata/granite-dense-tiny", "testdata/internlm2-tiny", "testdata/internlm3-tiny", "selftestdata/llama-tiny", "testdata/llama-tiny",
		"testdata/ministral3-tiny", "testdata/phi3-tiny", "testdata/qwen25vl-tiny", "internal/vision/testdata/qwen25vl-vision-tiny", "testdata/qwen3asr-tiny", "testdata/smollm3-tiny", "testdata/voxtral-tiny"} {
		out = append(out, Gap{Row: f + " int4/kv-i8", Line: mem.LineKV, What: "int8 K/V holds one scale a head, and the plan prices 1/8 of a byte an element, which under-prices a narrow head",
			Item: "MM8, int8 KV's scale", Bytes: KVPricedUnder})
	}
	for _, f := range []string{"testdata/nemotron-tiny", "testdata/nemotron3nano-tiny"} {
		out = append(out, Gap{Row: f + " *", Line: mem.LineState, What: "a Mamba-2 conv window row is a view into the projection's output, whose tail it keeps alive",
			Item: "MM8 CPU, Mamba window views", Bytes: func(fs map[string]int64) int64 { return fs["mambaTail"] }})
	}
	return out
}()

// gemma4MoEDenseGap is Gemma 4's MoE block on the dense line, past the bare vectors every family holds: its router's and experts' small
// tables. Not diagnosed, so it names this fixture's own bytes.
func gemma4MoEDenseGap(fixture string) Gap {
	return Gap{Row: fixture + " *", Line: mem.LineDense, What: "Gemma 4's MoE block holds router and expert tables beside the vectors, priced nowhere: not diagnosed",
		Item: "MM8, Gemma 4 MoE dense", Bytes: func(map[string]int64) int64 { return 10542 }}
}

func denseVectorGaps(fixtures []vectorFixture) []Gap {
	var out []Gap
	for _, v := range fixtures {
		out = append(out, Gap{Row: v.row + " *", Line: mem.LineDense, What: "the plan's dense figure sums weight matrices, and the norms and biases are held beside them",
			Item: "MM8, dense vectors", Bytes: func(f map[string]int64) int64 {
				return int64(v.perLayer)*f["vectorUnit"] + f["vectorBase"] + int64(v.matrixFields)*f["matrixFields"]
			}})
	}
	return out
}

// vectorFixture is a fixture whose dense line reads over plan on what the plan's sum of weight matrices leaves out: how many hidden-width vectors
// a layer it holds, and how many matrices' worth of struct scalars (decoder.WeightMatScalarBytesForTest: a weight matrix's own fields).
type vectorFixture struct {
	row          string
	perLayer     int
	matrixFields int
}

// cpuGateRow loads fixture path on the CPU, writes cpuGateCtx positions, and returns the row. The CPU has no allocator to account, so its
// ledger is filled by walking what the model and the cache hold (Model.CPUHoldsForTest). Closed reads that ledger after the model closed and
// those leases are released, so it holds a plant's lease and not the model's own freeing, which no ledger sees here. plant, when set, runs on the ledger before it is read. ok is false when the load is refused, which has nothing to hold.
func cpuGateRow(t *testing.T, row, path string, o decoder.Options, plant func(*mem.Ledger)) (Reading, bool) {
	t.Helper()
	o.Backend, o.DirectLoad = "cpu", true
	m, err := decoder.Load(path, o)
	if err != nil {
		return Reading{}, false
	}
	defer m.Close()
	var l mem.Ledger
	c := m.NewCache(cpuGateCtx)
	cfg := m.Config()
	for i := range cpuGateCtx {
		if _, err := m.ForwardForTest(i%cfg.VocabSize, c); err != nil {
			t.Errorf("%s: forward %d: %v", row, i, err)
			return Reading{}, false
		}
	}
	release := m.CPUHoldsForTest(&l, c)
	if plant != nil {
		plant(&l)
	}
	kv := m.KVDeliveredFor("cpu") // what the cache really is: Options.KVPrecision is the GPU's, and a family can refuse what was asked
	p := m.Plan("cpu", math.MaxInt64/4, decoder.PlanRequest{Ctx: cpuGateCtx, CtxPinned: true, KVF16: kv == mem.KVF16, KVI8: kv == mem.KVI8})
	read := Reading{Row: row, KVSlots: 1, Plan: p, Exact: []mem.LineID{mem.LineKV, mem.LineState}, Named: map[mem.LineID]int64{mem.LineScratch: decoder.CPUScratchBudgetForTest}}
	facts := GeometryOf(m, cpuGateCtx, 1, kv, true).Facts()
	facts["matrixFields"] = decoder.WeightMatScalarBytesForTest()
	read.Open = append(read.Open, Snap("generated", &l, facts))
	m.Close()
	release()
	read.Closed = l.Usage()
	return read, true
}

// TestMemGate_cpu is MM7's gate on the CPU (docs/tasks/task-subsystems-2026-10.md): every census fixture at every option cell that loads.
// It runs every row in one test, since a filtered run would see known gaps go quiet.
func TestMemGate_cpu(t *testing.T) {
	d, err := sb2census.ReadLayerD("../..", "cuda") // the fixture list; which cells go resident on CUDA is not read
	if err != nil {
		t.Fatal(err)
	}
	g := Gate{Allowance: cpuRounding, Gaps: cpuMemGaps, GapsPin: cpuMemGapsPin}
	var built []string
	refused := 0
	for _, f := range d.Fixtures {
		path, err := sb2census.Path("../..", f)
		if err != nil {
			t.Fatal(err)
		}
		for _, c := range sb2census.Cells {
			if c.SSM != "" || c.KVPrec != "" {
				continue // a GPU's setting: on the CPU the row would be int4's again
			}
			read, ok := cpuGateRow(t, f+" "+c.Name, path, decoder.Options{Quant: c.Quant, ActQuantGroup: c.ActGroup, KVPrecision: c.KVPrec, KVQuant: c.KVQuant}, nil)
			if !ok {
				refused++
				continue
			}
			built = append(built, f+" "+c.Name)
			g.Check(t, read)
		}
		// The census's cells ask the GPU's KVPrecision for f16 and the CPU's KVQuant for int8: this is the CPU's own f16.
		if read, ok := cpuGateRow(t, f+" int4/kvquant-f16", path, decoder.Options{Quant: "int4", KVQuant: "f16"}, nil); ok {
			built = append(built, f+" int4/kvquant-f16")
			g.Check(t, read)
		} else {
			refused++
		}
	}
	t.Logf("%d rows held, %d option cells refused at load", len(built), refused)
	HoldRows(t, cpuGateRowsPath, built, *cpuGateRowsUpdate)
	g.Finish(t)
}

// TestMemGate_cpuPlantsThroughTheTable: bytes leased over plan on the row's ledger stay red through the table, on a line it forgives
// part of and on lines it forgives nothing of; a lease never released is red as open after Close; and an unknown line is red.
func TestMemGate_cpuPlantsThroughTheTable(t *testing.T) {
	const fx, row = "../../testdata/llama-tiny", "testdata/llama-tiny int4"
	g := Gate{Allowance: cpuRounding, Gaps: cpuMemGaps, GapsPin: cpuMemGapsPin}
	o := decoder.Options{Quant: "int4"}
	clean, ok := cpuGateRow(t, row, fx, o, nil)
	if !ok {
		t.Fatal("llama-tiny did not load")
	}
	if u := g.Unknown(clean); len(u) != 0 {
		t.Fatalf("the unplanted row is red through the table: %v", u)
	}
	for _, line := range []mem.LineID{mem.LineDense, mem.LineKV, mem.LineState, mem.LineExperts} {
		read, ok := cpuGateRow(t, row, fx, o, func(l *mem.Ledger) {
			ls := l.Reserve(mem.HostAnon, line, 1<<20, 1<<20)
			t.Cleanup(ls.Release)
		})
		if !ok {
			t.Fatal("llama-tiny did not load")
		}
		red := false
		for _, f := range g.Unknown(read) {
			red = red || (f.Line == line && f.Kind == OverPlan)
		}
		if !red {
			t.Errorf("a mebibyte on %q is not red through the table: %v", line, g.Unknown(read))
		}
	}
	read, ok := cpuGateRow(t, row, fx, o, func(l *mem.Ledger) { l.Reserve(mem.HostAnon, mem.LineID("nowhere"), 8, 8) })
	if !ok {
		t.Fatal("llama-tiny did not load")
	}
	if s := fmt.Sprint(g.Unknown(read)); !strings.Contains(s, "unknown line") || !strings.Contains(s, "open after Close") {
		t.Errorf("a lease on no line, never released, reads: %s", s)
	}
}

// TestGeometry_namesEachCause: on a sliding int8 row the window's over-price and the scale's under-price are each their own figure, and what the
// layers hold is the formula less and plus them, so a row's netted figure does not stand for either.
func TestGeometry_namesEachCause(t *testing.T) {
	m, err := decoder.Load("../../testdata/olmo3-tiny", decoder.Options{Backend: "cpu", Quant: "int4", DirectLoad: true})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	g := GeometryOf(m, cpuGateCtx, 1, mem.KVI8, true)
	if g.KVWindowOver <= 0 || g.KVScaleUnder == 0 {
		t.Fatalf("olmo3-tiny at int8: window %d, scale %d: want both", g.KVWindowOver, g.KVScaleUnder)
	}
	if got, want := g.KVHeld, g.KVFormula-g.KVWindowOver-g.KVLatentOver+g.KVScaleUnder; got != want {
		t.Errorf("held %d, the formula less and plus its causes %d", got, want)
	}
}
