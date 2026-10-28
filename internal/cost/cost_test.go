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

// TestCounts_unmeasuredIsNotSummed (G-SB11-1): an operation whose bytes are unmeasured makes its phase's bytes unmeasured, read
// as 0 rather than as the other operations' partial sum, and the same holds when one operation is counted both ways.
func TestCounts_unmeasuredIsNotSummed(t *testing.T) {
	c := New()
	c.Add(Decode, "proj_o", Counts{Calls: 1, BytesRead: 100, BytesWritten: 10, TimeNs: 7})
	c.Add(Decode, "ffn", Counts{Calls: 1, BytesRead: 50, TimeNs: 3, Unmeasured: []string{FieldBytesRead, FieldBytesWritten}})
	c.Add(Decode, "ffn", Counts{Calls: 1, BytesRead: 60, TimeNs: 4})
	r := c.Report().Decode
	if got := r.Ops["proj_o"]; got.BytesRead != 100 || got.BytesWritten != 10 || got.Unmeasured != nil {
		t.Errorf("a measured operation changed: %+v", got)
	}
	want := Counts{Calls: 2, TimeNs: 7, Unmeasured: []string{FieldBytesRead, FieldBytesWritten}}
	if got := r.Ops["ffn"]; !reflect.DeepEqual(got, want) {
		t.Errorf("ffn %+v, want %+v (partly counted bytes are unmeasured)", got, want)
	}
	want = Counts{Calls: 3, TimeNs: 14, Unmeasured: []string{FieldBytesRead, FieldBytesWritten}}
	if !reflect.DeepEqual(r.Totals, want) {
		t.Errorf("totals %+v, want %+v", r.Totals, want)
	}
}

// TestCounts_fieldNamesAreJSONNames (G-SB11-1): every name Unmeasured may list is a Counts JSON name, and every counter has one.
func TestCounts_fieldNamesAreJSONNames(t *testing.T) {
	names := []string{FieldBytesRead, FieldBytesWritten, FieldLaunches, FieldSyncs, FieldAllocs, FieldAllocBytes, FieldTime}
	tags := map[string]bool{}
	ty := reflect.TypeFor[Counts]()
	for i := range ty.NumField() {
		tag, _, _ := strings.Cut(ty.Field(i).Tag.Get("json"), ",")
		tags[tag] = true
	}
	var c Counts
	for _, f := range names {
		if !tags[f] || c.field(f) == nil {
			t.Errorf("%q: JSON name %v, a counter %v", f, tags[f], c.field(f) != nil)
		}
	}
	if len(names) != len(tags)-2 { // calls is always measured; unmeasured is the list itself
		t.Errorf("%d field names for %d counters", len(names), len(tags)-2)
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
