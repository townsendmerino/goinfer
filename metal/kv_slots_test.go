//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"slices"
	"sync/atomic"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/cost"
	"github.com/townsendmerino/goinfer/internal/mem"
)

// TestMetalKVSlots_interleavedMatchesAlone is MC1's gate (docs/tasks/task-concurrency-2026-09.md): with two resident KV
// slots, two conversations that share only a lead, interleaved turn by turn through the production Generate path,
// must each emit exactly the token ids they emit when served alone, and reuse exactly as much of their own history
// on every turn. A control — the same interleaving on one slot — must thrash, so this test can see the failure it
// guards against.
//
// GOINFER_METAL_KVSLOTS_MODEL=<checkpoint> runs the same scenario on a real checkpoint (int4, 48 tokens per turn)
// instead of the synthetic tiny model; its prompts are ids from a fixed range, not text.
func TestMetalKVSlots_interleavedMatchesAlone(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	dir, quant, maxTok, base := os.Getenv("GOINFER_METAL_KVSLOTS_MODEL"), "int4", 48, 1000
	if dir == "" {
		w := genTinyWeights(rand.New(rand.NewSource(21)))
		dir = t.TempDir()
		writeDense(t, dir, w)
		quant, maxTok, base = "int8int8", 12, 0
	}

	lead := []int{base + 1, base + 2, base + 3} // the shared "chat template" lead
	first := map[string][]int{
		"A": append(slices.Clone(lead), base+10, base+11, base+12, base+13),
		"B": append(slices.Clone(lead), base+20, base+21, base+22, base+23),
	}
	type turn = kvSlotTurn
	run := func(m *decoder.Model, order []string) map[string][]turn {
		prompts := map[string][]int{"A": slices.Clone(first["A"]), "B": slices.Clone(first["B"])}
		out := map[string][]turn{}
		for _, c := range order {
			ch, gen := m.Generate(context.Background(), prompts[c], maxTok, decoder.SamplingParams{})
			var ids []int
			for id := range ch {
				ids = append(ids, id)
			}
			if err := gen.Err(); err != nil {
				t.Fatalf("%s turn %d: %v", c, len(out[c])+1, err)
			}
			out[c] = append(out[c], turn{ids, gen.PrefillReused})
			// the next user turn extends the conversation with its reply and two new ids
			prompts[c] = append(append(slices.Clone(prompts[c]), ids...), base+30+len(out[c]), base+40+len(out[c]))
		}
		return out
	}
	load := func(slots int) *decoder.Model {
		m, err := decoder.Load(dir, decoder.Options{Backend: "metal", Quant: quant, ResidentKVSlots: slots})
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		if got := m.ResidentKVSlots(); got != slots {
			m.Close()
			t.Fatalf("ResidentKVSlots() = %d with %d requested — the resident did not allocate them", got, slots)
		}
		return m
	}
	interleaved := []string{"A", "B", "A", "B", "A", "B"}

	mTwo := load(2)
	inter := run(mTwo, interleaved)
	mTwo.Close()
	alone := map[string][]turn{}
	for _, c := range []string{"A", "B"} { // each conversation alone on a fresh one-slot model
		m := load(1)
		alone[c] = run(m, []string{c, c, c})[c]
		m.Close()
	}
	mOne := load(1)
	thrash := run(mOne, interleaved) // the control
	mOne.Close()

	for _, c := range []string{"A", "B"} {
		for i := range alone[c] {
			a, b := alone[c][i], inter[c][i]
			if !slices.Equal(a.ids, b.ids) {
				t.Errorf("%s turn %d: interleaved on 2 slots emitted %v, alone %v — not bit-identical", c, i+1, b.ids, a.ids)
			}
			if a.reused != b.reused {
				t.Errorf("%s turn %d: interleaved on 2 slots reused %d, alone %d", c, i+1, b.reused, a.reused)
			}
			if i > 0 && b.reused <= len(lead) {
				t.Errorf("%s turn %d: reused only %d (the lead is %d) — its own history was not kept", c, i+1, b.reused, len(lead))
			}
			if i > 0 && thrash[c][i].reused > len(lead) {
				t.Errorf("control: %s turn %d reused %d on ONE slot interleaved — the control no longer thrashes, so "+
					"this test cannot see a slot regression", c, i+1, thrash[c][i].reused)
			}
		}
		t.Logf("%s: reused per turn alone %v, interleaved on 2 slots %v, interleaved on 1 slot %v", c,
			reusedOf(alone[c]), reusedOf(inter[c]), reusedOf(thrash[c]))
	}
}

// kvSlotTurn is one generation of TestMetalKVSlots_interleavedMatchesAlone: its token ids and PrefillReused.
type kvSlotTurn struct {
	ids    []int
	reused int
}

func reusedOf(ts []kvSlotTurn) []int {
	out := make([]int, len(ts))
	for i, t := range ts {
		out[i] = t.reused
	}
	return out
}

// TestKVSlotsWithin pins MC1's fit-guard clamp (docs/tasks/task-concurrency-2026-09.md MC1 gate 4) without allocating a
// real clamp's worth of KV on a 16 GB machine: the largest count up to the request that fits the budget, never below 1.
func TestKVSlotsWithin(t *testing.T) {
	const mb = int64(1 << 20)
	for _, tc := range []struct {
		name               string
		want               int
		budget, base, slot int64
		got                int
	}{
		{"all fit", 4, 10000 * mb, 2000 * mb, 117 * mb, 4},
		{"clamped to what fits", 8, 2500 * mb, 2000 * mb, 117 * mb, 5}, // 2000 + 4*117 = 2468 <= 2500; +5*117 > 2500
		{"only the first fits", 4, 2050 * mb, 2000 * mb, 117 * mb, 1},
		{"even the first over budget: still 1, the decline is elsewhere", 4, 1000 * mb, 2000 * mb, 117 * mb, 1},
		{"request of 0 means 1", 0, 10000 * mb, 2000 * mb, 117 * mb, 1},
	} {
		if got := kvSlotsWithin(tc.want, tc.budget, tc.base, tc.slot); got != tc.got {
			t.Errorf("%s: kvSlotsWithin(%d, ...) = %d, want %d", tc.name, tc.want, got, tc.got)
		}
	}
}

// TestMetalKVSlots_pricedBeforeTheBuild pins where buildResident prices MC1's slots. metalKVSlots' budget reads the
// live available memory, and the base it compares against already includes the resident's weights, so it must be
// asked BEFORE those are allocated. Here the live probe is stubbed to fall by exactly what this device allocates
// (MTLDevice.currentAllocatedSize, one device per process), from a starting figure where precisely `want` slots fit:
// priced before the build, all of them are granted; priced after, the resident's own buffers eat the margin.
func TestMetalKVSlots_pricedBeforeTheBuild(t *testing.T) {
	if os.Getenv("GOINFER_NO_RESIDENT_MEM_GUARD") != "" {
		t.Skip("GOINFER_NO_RESIDENT_MEM_GUARD is set — the guard is disabled, nothing to price")
	}
	dev, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	const want = 4
	dir := t.TempDir()
	writeDense(t, dir, genTinyWeights(rand.New(rand.NewSource(21))))
	m, err := decoder.Load(dir, decoder.Options{Quant: "int8int8", ResidentKVSlots: want})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	base, perSlot := residentNeedBytes(m), residentKVBytes(m)
	if base <= 0 || perSlot <= 0 {
		t.Skip("tiny fixture prices nothing")
	}
	live0 := base + int64(want-1)*perSlot + perSlot/2 // want slots fit, want+1 would not
	alloc0 := int64(dev.CurrentAllocatedSize())
	orig := metalLiveAvailable
	metalLiveAvailable = func() int64 { return live0 - (int64(dev.CurrentAllocatedSize()) - alloc0) }
	t.Cleanup(func() { metalLiveAvailable = orig })
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("build resident: %v", err)
	}
	defer r.Close()
	if grew := int64(dev.CurrentAllocatedSize()) - alloc0; grew < perSlot {
		t.Skipf("the resident's allocations (%d bytes) are not visible through this device handle, or smaller than one "+
			"slot (%d bytes): the stub cannot show the double count", grew, perSlot)
	}
	if got := len(r.kvSlotBufs); got != want {
		t.Errorf("resident allocated %d KV slots, want %d: the slot count was priced after the build's own buffers "+
			"had already come out of the live figure", got, want)
	}
}

var sb11Update = flag.Bool("sb11.update", false, "rewrite testdata/sb11_decode_llama_tiny.<goarch>.json from this device (review the diff)")

// sb11Token is one counted token's per-operation calls, launches, submits, syncs and allocations (time is not compared).
type sb11Token map[string][5]int64

func sb11Counts(r cost.Report) sb11Token {
	out := sb11Token{}
	for name, c := range r.Decode.Ops {
		out[name] = [5]int64{c.Calls, c.Launches, c.Submits, c.Syncs, c.Allocs}
	}
	return out
}

// sb11Counted runs work on a's resident counted into a fresh counter, as one token of the decode phase.
func sb11Counted(t *testing.T, a *metalResident, work func() error) cost.Report {
	t.Helper()
	c := cost.New()
	if err := a.r.costToken(c, cost.Decode, work); err != nil {
		t.Fatal(err)
	}
	return c.Report()
}

// sb11Decode runs one decode token at pos through the production path (the encode-ahead executor), counted.
func sb11Decode(t *testing.T, m *decoder.Model, a *metalResident, id, pos int) (cost.Report, []float32) {
	t.Helper()
	var lg []float32
	r := sb11Counted(t, a, func() error {
		var e error
		lg, e = a.Forward(m.EmbedResidentForTest(id), pos)
		return e
	})
	return r, append([]float32(nil), lg...)
}

// sb11Attributed checks that every dispatch the device counted was attributed (no dispatch_other), and that some were.
func sb11Attributed(t *testing.T, what string, r cost.Report) {
	t.Helper()
	if v, other := sb11Counts(r)["dispatch_other"]; other {
		t.Errorf("%s: %d dispatches the device counted were not attributed", what, v[1])
	}
	if r.Decode.Totals.Launches == 0 {
		t.Errorf("%s: no dispatch counted", what)
	}
}

// TestCost_metalDecodeExact is G-SB11-16 and G-SB11-17 (docs/tasks/task-subsystems-2026-10.md, SB11 "Step 4: Metal"):
// llama-tiny resident on Metal at int4.
//   - Every counted path attributes all the dispatches the device counts (no dispatch_other): the first token, whose buffer is
//     encoded in line, tokens 1 to 3, a token without logits and the synchronous forward.
//   - Tokens 1 to 3 each count the same per-pipeline dispatches, submits, syncs and allocations, equal to the committed golden;
//     a steady token allocates nothing; its wall time reaches Timings, and its totals' calls equal its launches.
//   - Logits match the uncounted token's bit for bit.
//   - The plants: an extra command buffer moves submit and sync by one each, so the golden turns red; a dispatch encoded past
//     the attribution is one dispatch_other and leaves the attribution equal to the golden.
func TestCost_metalDecodeExact(t *testing.T) {
	m, err := decoder.Load(filepath.Join("..", "testdata", "llama-tiny"), decoder.Options{Backend: "metal", Quant: "int4"})
	if err != nil {
		t.Fatalf("load llama-tiny: %v", err)
	}
	defer m.Close()
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Fatalf("llama-tiny is not Metal-resident: %s", m.ResidentDecline())
	}
	rep, _ := sb11Decode(t, m, a, 1, 0)
	sb11Attributed(t, "the first token", rep)
	var first sb11Token
	for pos := 1; pos <= 3; pos++ {
		rep, _ := sb11Decode(t, m, a, 2+pos, pos)
		d := rep.Decode
		if tm := rep.Timings(); d.WallNs <= 0 || tm.DecodeMs == nil {
			t.Errorf("token %d: wall time %d ns, decode_ms %v: want it counted", pos, d.WallNs, tm.DecodeMs)
		}
		if d.Totals.Calls != d.Totals.Launches {
			t.Errorf("token %d: %d calls against %d launches; a device count is in Calls", pos, d.Totals.Calls, d.Totals.Launches)
		}
		sb11Attributed(t, fmt.Sprintf("token %d", pos), rep)
		got := sb11Counts(rep)
		if got["alloc"][4] != 0 {
			t.Errorf("token %d allocated %d leases", pos, got["alloc"][4])
		}
		if pos == 1 {
			first = got
			continue
		}
		if !reflect.DeepEqual(got, first) {
			t.Errorf("token %d counts differ from token 1's:\n %v\n %v", pos, got, first)
		}
	}
	var dispatches int64
	for _, v := range first {
		dispatches += v[1]
	}
	t.Logf("a decode token: %d pipelines, %d dispatches, %d submits, %d syncs", len(first)-3, dispatches, first["submit"][2], first["sync"][3])

	// Counting changes nothing: the same token at the same position, uncounted and counted.
	off, err := a.Forward(m.EmbedResidentForTest(9), 4)
	if err != nil {
		t.Fatal(err)
	}
	off = append([]float32(nil), off...)
	_, on := sb11Decode(t, m, a, 9, 4)
	for i := range on {
		if math.Float32bits(on[i]) != math.Float32bits(off[i]) {
			t.Fatalf("logit %d differs with the counter on", i)
		}
	}

	// The plants (G-SB11-17).
	defer func() { costPlantForTest = 0 }()
	costPlantForTest = 1
	rep, _ = sb11Decode(t, m, a, 10, 5)
	costPlantForTest = 0
	planted := sb11Counts(rep)
	if reflect.DeepEqual(planted, first) {
		t.Errorf("the extra command buffer left the counts equal to a steady token's: %v", planted)
	}
	if planted["submit"][2] != first["submit"][2]+1 || planted["sync"][3] != first["sync"][3]+1 {
		t.Errorf("the extra command buffer gives %d submits and %d syncs, want %d and %d", planted["submit"][2], planted["sync"][3],
			first["submit"][2]+1, first["sync"][3]+1)
	}
	// One counted token after uncounted decode: the executor re-encodes the buffer it encoded ahead uncounted, so the token
	// counts as a steady one, attributed in full.
	if _, err := a.Forward(m.EmbedResidentForTest(11), 6); err != nil {
		t.Fatal(err)
	}
	if rep, _ := sb11Decode(t, m, a, 12, 7); !reflect.DeepEqual(sb11Counts(rep), first) {
		t.Errorf("one counted token after uncounted decode counts %v, want a steady token's %v", sb11Counts(rep), first)
	}

	// The plant: the executor keeps the buffer it encoded ahead uncounted, and the counted token commits it. The device counts
	// its dispatches and nothing attributes them, so all of them are dispatch_other.
	if _, err := a.Forward(m.EmbedResidentForTest(13), 8); err != nil {
		t.Fatal(err)
	}
	costPlantForTest = 2
	rep, _ = sb11Decode(t, m, a, 14, 9)
	costPlantForTest = 0
	planted = sb11Counts(rep)
	if got := planted["dispatch_other"]; got != [5]int64{dispatches, dispatches, 0, 0, 0} {
		t.Errorf("the buffer encoded uncounted gives dispatch_other %v, want all %d of its dispatches", got, dispatches)
	}
	for name, v := range planted {
		switch name {
		case "sync", "submit", "alloc", "dispatch_other":
		default:
			t.Errorf("the buffer encoded uncounted attributed %v to %s", v, name)
		}
	}
	if rep, _ := sb11Decode(t, m, a, 15, 10); !reflect.DeepEqual(sb11Counts(rep), first) {
		t.Errorf("the token after it counts %v, want a steady token's %v", sb11Counts(rep), first)
	}

	// A lease taken and released inside a token counts as one allocation of its requested bytes.
	rep = sb11Counted(t, a, func() error {
		a.r.d.ReleaseBuf(a.r.d.NewBufferBytesOn(mem.LineTest, 4096))
		return nil
	})
	if got := rep.Decode.Ops["alloc"]; got.Allocs != 1 || got.AllocBytes != 4096 {
		t.Errorf("a lease taken and released in the token counts %d allocs of %d bytes, want 1 of 4096", got.Allocs, got.AllocBytes)
	}

	// The other submitting paths.
	sb11Attributed(t, "a token without logits", sb11Counted(t, a, func() error {
		return a.ForwardNoLogits(m.EmbedResidentForTest(16), 11)
	}))
	sb11Attributed(t, "the synchronous forward", sb11Counted(t, a, func() error {
		a.r.ForwardEmb(m.EmbedResidentForTest(17), 12)
		return a.r.takeExecErr()
	}))

	golden := filepath.Join("testdata", "sb11_decode_llama_tiny."+runtime.GOARCH+".json")
	b, _ := json.MarshalIndent(first, "", " ")
	if *sb11Update {
		if err := os.WriteFile(golden, append(b, '\n'), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	raw, err := os.ReadFile(golden)
	if errors.Is(err, os.ErrNotExist) {
		t.Fatalf("no %s golden; this device's counts are %s (record them with -sb11.update and review them)", golden, b)
	}
	if err != nil {
		t.Fatal(err)
	}
	var want sb11Token
	if err := json.Unmarshal(raw, &want); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first, want) {
		t.Errorf("a decode token's counts differ from the golden:\n got  %v\n want %v", first, want)
	}
}

// sb11OffOps are the counters the off-cost bound charges per dispatch: three atomic adds and the hook's load.
var (
	sb11OffA, sb11OffB, sb11OffC atomic.Int64
	sb11OffHook                  atomic.Pointer[func(Pipeline)]
)

// sb11OffBoundNs is the slowest of five repeats of 10^7 rounds of three atomic adds and an atomic pointer load with its nil
// check, in ns a round.
func sb11OffBoundNs() float64 {
	const n = 10_000_000
	var worst float64
	for range 5 {
		t0 := time.Now()
		for range n {
			sb11OffA.Add(1)
			sb11OffB.Add(1)
			sb11OffC.Add(1)
			if h := sb11OffHook.Load(); h != nil {
				(*h)(Pipeline{})
			}
		}
		worst = max(worst, float64(time.Since(t0).Nanoseconds())/n)
	}
	return worst
}

func sb11Median(xs []float64) float64 {
	s := slices.Clone(xs)
	slices.Sort(s)
	if len(s)%2 == 1 {
		return s[len(s)/2]
	}
	return (s[len(s)/2-1] + s[len(s)/2]) / 2
}

// TestCost_metalOffCost is G-SB11-18 (docs/tasks/task-subsystems-2026-10.md, SB11 "G-SB11-18's method"): the always-on
// counting, charged as three atomic adds and a hook nil check per dispatch, times the device's dispatches per decode token, as
// a share of the median uncounted decode token, must be under 0.5%. It also reports the on cost (counted ÷ uncounted tokens,
// the first token of each block dropped), which is not graded. A timed run: hold the timing lock exclusively, on AC.
//
//	GOINFER_SB11_OFFCOST_MODEL=~/models/<0.5B>.int4.metal.v17.giw go test -count=1 -tags goinfer_testhooks -run '^TestCost_metalOffCost$' -v ./metal/
func TestCost_metalOffCost(t *testing.T) {
	path := os.Getenv("GOINFER_SB11_OFFCOST_MODEL")
	if path == "" {
		t.Skip("set GOINFER_SB11_OFFCOST_MODEL to a Metal int4 .giw (a timed run, under the timing lock, on AC)")
	}
	la, _ := exec.Command("sysctl", "-n", "vm.loadavg").Output()
	fmt.Fprintf(os.Stderr, "[G-SB11-18] start %s, load %s", time.Now().Format(time.RFC3339), la)
	m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4"})
	if err != nil {
		t.Fatalf("load %s: %v", path, err)
	}
	defer m.Close()
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Fatalf("not Metal-resident: %s", m.ResidentDecline())
	}
	const warm, timed, block, rounds = 16, 64, 9, 4
	embs := make([][]float32, warm+timed+4*rounds*block)
	for i := range embs {
		embs[i] = slices.Clone(m.EmbedResidentForTest(1000 + i*37%5000))
	}
	pos := 0
	step := func(counted bool) float64 {
		e, p := embs[pos], pos
		pos++
		t0 := time.Now()
		if counted {
			sb11Counted(t, a, func() error { _, err := a.Forward(e, p); return err })
		} else if _, err := a.Forward(e, p); err != nil {
			t.Fatal(err)
		}
		return float64(time.Since(t0).Nanoseconds())
	}
	for range warm {
		step(false)
	}
	d0, s0, w0 := a.r.d.Activity()
	off := make([]float64, timed)
	for i := range off {
		off[i] = step(false)
	}
	d1, s1, w1 := a.r.d.Activity()
	perTok := float64(d1-d0) / timed
	tokNs := sb11Median(off)
	boundNs := sb11OffBoundNs()
	share := boundNs * perTok / tokNs
	fmt.Fprintf(os.Stderr, "[G-SB11-18] decode token: median %.1f µs over %d (min %.1f, max %.1f); per token %.2f dispatches, %.2f submits, %.2f waits\n",
		tokNs/1e3, timed, slices.Min(off)/1e3, slices.Max(off)/1e3, perTok, float64(s1-s0)/timed, float64(w1-w0)/timed)
	fmt.Fprintf(os.Stderr, "[G-SB11-18] bound %.2f ns a dispatch x %.2f = %.2f µs a token: share %.4f%% (bar 0.5%%)\n",
		boundNs, perTok, boundNs*perTok/1e3, 100*share)

	// Context: the on cost, A B B A blocks, the first token of each block dropped (the switch, not a counted token).
	var on, unc []float64
	for range rounds {
		for _, counted := range []bool{false, true, true, false} {
			for j := range block {
				ns := step(counted)
				if j == 0 {
					continue
				}
				if counted {
					on = append(on, ns)
				} else {
					unc = append(unc, ns)
				}
			}
		}
	}
	fmt.Fprintf(os.Stderr, "[G-SB11-18] on cost (not graded): counted median %.1f µs over %d, uncounted %.1f µs over %d: %.4fx\n",
		sb11Median(on)/1e3, len(on), sb11Median(unc)/1e3, len(unc), sb11Median(on)/sb11Median(unc))
	if share >= 0.005 {
		t.Errorf("G-SB11-18: the off cost is %.4f%% of a decode token, at or over the 0.5%% bar", 100*share)
	}
}
