//go:build darwin && goinfer_testhooks

package metal

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/cost"
)

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
	costPlantForTest = 2
	rep, _ = sb11Decode(t, m, a, 11, 6)
	costPlantForTest = 0
	planted = sb11Counts(rep)
	if got := planted["dispatch_other"]; got != [5]int64{1, 1, 0, 0, 0} {
		t.Errorf("the dispatch encoded past the attribution gives dispatch_other %v, want one dispatch", got)
	}
	delete(planted, "dispatch_other")
	if !reflect.DeepEqual(planted, first) {
		t.Errorf("the unattributed dispatch moved the attribution:\n got  %v\n want %v", planted, first)
	}

	// The other submitting paths.
	sb11Attributed(t, "a token without logits", sb11Counted(t, a, func() error {
		return a.ForwardNoLogits(m.EmbedResidentForTest(13), 7)
	}))
	sb11Attributed(t, "the synchronous forward", sb11Counted(t, a, func() error {
		a.r.ForwardEmb(m.EmbedResidentForTest(14), 8)
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
