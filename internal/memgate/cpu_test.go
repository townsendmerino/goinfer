//go:build goinfer_testhooks

package memgate

import (
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

// cpuRounding is the CPU's allocator as a row sees it: the Go heap's size classes are not in what the walk leases.
var cpuRounding = PerLease("none: the walk leases each array's own bytes", 0)

// cpuDenseVectorFixtures are the fixtures whose dense line reads over plan on the bare vectors alone.
var cpuDenseVectorFixtures = []string{
	"internal/vision/testdata/qwen25vl-vision-tiny", "selftestdata/gemma3-vl-tiny", "selftestdata/llama-tiny",
	"selftestdata/qwen35-tiny", "selftestdata/qwen35vl-tiny", "testdata/bailing_hybrid-tiny",
	"testdata/cohere-tiny", "testdata/cohere2-tiny", "testdata/deepseek-tiny",
	"testdata/gemma1-tiny", "testdata/gemma1-tiny.gguf", "testdata/gemma2-hd-tiny",
	"testdata/gemma2-hd-tiny.gguf", "testdata/gemma2-tiny", "testdata/gemma3-vl-tiny",
	"testdata/gguf-split/glm-tiny-00001-of-00004.gguf",
	"testdata/glm-ocr-tiny", "testdata/glm-tiny", "testdata/glm-tiny-bias",
	"testdata/glm-tiny.gguf", "testdata/gptoss_tiny.gguf", "testdata/granite-dense-tiny",
	"testdata/internlm2-tiny", "testdata/internlm3-tiny", "testdata/laguna-m1-tiny",
	"testdata/laguna-xs2-tiny", "testdata/laguna-xs21-tiny", "testdata/lfm2-tiny",
	"testdata/llama-attnfa-tiny", "testdata/llama-tiny", "testdata/llama4-tiny",
	"testdata/ministral3-tiny", "testdata/mistral-tiny-window", "testdata/mixtral-tiny",
	"testdata/nemotron-tiny", "testdata/nemotron3nano-tiny", "testdata/olmo3-tiny",
	"testdata/olmo_hybrid-tiny", "testdata/phi3-tiny", "testdata/qwen25vl-tiny",
	"testdata/qwen35-tiny", "testdata/qwen35vl-tiny", "testdata/qwen3_5-tiny",
	"testdata/qwen3_5-tiny-normw", "testdata/qwen3_5_moe-tiny", "testdata/qwen3asr-tiny",
	"testdata/qwen3moe-tiny", "testdata/qwen3moe-tiny-k3", "testdata/qwen3next-tiny",
	"testdata/qwen3vlmoe-tiny", "testdata/smollm3-tiny", "testdata/spark2-5-tiny",
	"testdata/voxtral-tiny",
}

// cpuMemGaps is the CPU's table of known reds. It may only shrink: the fix that removes a row lowers cpuMemGapsPin.
var cpuMemGaps = append(denseVectorGaps(cpuDenseVectorFixtures), []Gap{
	gemma4MoEDenseGap("testdata/gemma4-moe-kv-tiny"), gemma4MoEDenseGap("testdata/gemma4-moe-kv-tiny32"),
	{Row: "testdata/gemma4-moe-kv-tiny *", Line: mem.LineKV, What: "NewCache reserves every layer at the model's kvDim, and a wider layer grows past that by append, holding more than it stores",
		Item: "MM8 CPU, KV reserve by layer", Bytes: cpuKVGrowth},
	{Row: "testdata/gemma4-moe-kv-tiny32 *", Line: mem.LineKV, What: "NewCache reserves every layer at the model's kvDim, and a wider layer grows past that by append, holding more than it stores",
		Item: "MM8 CPU, KV reserve by layer", Bytes: cpuKVGrowth},
	{Row: "testdata/gptoss_tiny.gguf *", Line: mem.LineExperts, What: "gpt-oss's expert biases are held with its experts, and the plan's expert figure sums weight matrices",
		Item: "MM8 CPU, gpt-oss experts", Bytes: func(map[string]int64) int64 { return 8192 }},
}...)

func init() { cpuMemGaps = append(cpuMemGaps, cpuKVGaps...) }

const cpuMemGapsPin = 88

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
		"a sliding layer's ring holds its window, and the plan prices the whole context", "MM8 CPU, KV of a sliding window")
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
		Item: "MM8, Gemma 4 MoE dense", Bytes: func(map[string]int64) int64 { return 8640 }}
}

// cpuKVGrowth bounds an append-grown layer: Go's growth from the reserved capacity holds at most about four times what it stores.
func cpuKVGrowth(f map[string]int64) int64 { return 3 * f["kvFormula"] }

func denseVectorGaps(fixtures []string) []Gap {
	var out []Gap
	for _, f := range fixtures {
		out = append(out, Gap{Row: f + " *", Line: mem.LineDense, What: "the plan's dense figure sums weight matrices, and the norms and biases are held beside them",
			Item: "MM8, dense vectors", Bytes: func(f map[string]int64) int64 { return f["vectors"] }})
	}
	return out
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
	read := Reading{Row: row, KVSlots: 1, Plan: p, Exact: []mem.LineID{mem.LineKV, mem.LineState}}
	read.Open = append(read.Open, Snap("generated", &l, GeometryOf(m, cpuGateCtx, 1, kv, true).Facts()))
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
	built, refused := 0, 0
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
			built++
			g.Check(t, read)
		}
		// The census's cells ask the GPU's KVPrecision for f16 and the CPU's KVQuant for int8: this is the CPU's own f16.
		if read, ok := cpuGateRow(t, f+" int4/kvquant-f16", path, decoder.Options{Quant: "int4", KVQuant: "f16"}, nil); ok {
			built++
			g.Check(t, read)
		}
	}
	t.Logf("%d rows held, %d option cells refused at load", built, refused)
	if built < 350 {
		t.Errorf("only %d rows held: a gate that read little covers little", built)
	}
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
