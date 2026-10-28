package cost

import (
	"encoding/json"
	"reflect"
	"strings"
	"sync"
	"testing"
)

// TestReport_roundTrips (G-SB11-1): a report survives JSON unchanged; an unmeasured phase is absent from it; an unmeasured field
// is listed, not zero; Timings gives X1's names.
func TestReport_roundTrips(t *testing.T) {
	c := New()
	c.Tokens(Decode, 1)
	c.Add(Decode, "attention", Counts{Calls: 2, BytesRead: 4096, BytesWritten: 64, TimeNs: 1500, Unmeasured: []string{FieldAllocs, FieldLaunches}})
	c.Add(Decode, "norm", Counts{Calls: 3, BytesRead: 96, BytesWritten: 48, TimeNs: 500_000, Unmeasured: []string{FieldSyncs, FieldAllocs}})
	r := c.Report()
	b, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(b), `"prefill"`) {
		t.Fatalf("an unmeasured phase is in the JSON: %s", b)
	}
	var back Report
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(r, back) {
		t.Fatalf("round trip changed the report:\n%+v\n%+v", r, back)
	}
	if got := r.Decode.Totals; got.Calls != 5 || got.BytesRead != 4192 || got.TimeNs != 501_500 ||
		!reflect.DeepEqual(got.Unmeasured, []string{FieldAllocs, FieldLaunches, FieldSyncs}) {
		t.Fatalf("decode totals %+v", got)
	}
	tm := r.Timings()
	if tm.PrefillMs != nil || tm.DecodeMs == nil || *tm.DecodeMs != 0.5015 {
		t.Fatalf("timings %+v", tm)
	}
	tb, _ := json.Marshal(tm)
	if string(tb) != `{"decode_ms":0.5015}` {
		t.Fatalf("timings JSON %s, want X1's decode_ms alone", tb)
	}
	c.Add(Prefill, "norm", Counts{Calls: 1, Unmeasured: []string{FieldTime}})
	if tm := c.Report().Timings(); tm.PrefillMs != nil {
		t.Fatalf("prefill time was not measured, yet prefill_ms is %v", *tm.PrefillMs)
	}
}

// TestCounter_offAndConcurrent (G-SB11-1): a nil Counter is off and reports nothing; adds from many goroutines sum exactly.
func TestCounter_offAndConcurrent(t *testing.T) {
	var off *Counter
	off.Add(Decode, "x", Counts{Calls: 1})
	off.Tokens(Decode, 1)
	if r := off.Report(); r.Prefill != nil || r.Decode != nil {
		t.Fatalf("a nil counter reported %+v", r)
	}
	c := New()
	var wg sync.WaitGroup
	for range 16 {
		wg.Go(func() {
			for range 1000 {
				c.Add(Prefill, "matmul", Counts{Calls: 1, BytesRead: 3})
			}
		})
	}
	wg.Wait()
	if got := c.Report().Prefill.Ops["matmul"]; got.Calls != 16000 || got.BytesRead != 48000 {
		t.Fatalf("%+v", got)
	}
}
