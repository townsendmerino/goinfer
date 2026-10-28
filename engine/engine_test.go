package engine

import (
	"context"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

type fake struct{ Model }

func TestHandleOf_nilStaysNil(t *testing.T) {
	if h := HandleOf(nil); h != nil {
		t.Fatalf("HandleOf(nil) = %v, want a nil Model", h)
	}
	if sum := Call(nil).LoadProfile().Summary(); sum != "" {
		t.Errorf("no model's load profile summary = %q, want none", sum)
	}
	f := &fake{}
	if Call(f) != Model(f) {
		t.Error("Call of a non-nil handle is not that handle")
	}
	var typed *fake
	if m, ok := Call(typed).(*decoder.Model); !ok || m != nil {
		t.Errorf("Call of a typed-nil handle = %#v, want a nil *decoder.Model, as IsNil counts it as no model", Call(typed))
	}
}

func TestIsNil_typedNil(t *testing.T) {
	var typed *fake
	for _, h := range []Model{nil, HandleOf(nil), typed} {
		if !IsNil(h) {
			t.Errorf("IsNil(%#v) = false", h)
		}
	}
	if IsNil(&fake{}) {
		t.Error("IsNil of a non-nil handle is true")
	}
	if AsDecoder(&fake{}) != nil {
		t.Error("AsDecoder of a fake is not nil")
	}
}

func TestReport_BudgetFor(t *testing.T) {
	for _, c := range []struct {
		name      string
		r         *Report
		requested int
		want      int
	}{
		{"clamped below request", &Report{Budget: 96, BudgetClamped: true}, 512, 96},
		{"clamped to zero", &Report{Budget: 0, BudgetClamped: true}, 512, 0},
		{"unclamped zero", &Report{Budget: 0}, 512, 512},
		{"nil report", nil, 512, 512},
	} {
		if got := c.r.BudgetFor(c.requested); got != c.want {
			t.Errorf("%s: BudgetFor = %d, want %d", c.name, got, c.want)
		}
	}
	if ReportNow(nil) != nil || ReportOf(nil) != nil {
		t.Error("no report function, or no generation, gives a non-nil report")
	}
}

func TestLogprobsOf_fieldForField(t *testing.T) {
	if LogprobsOf(nil) != nil {
		t.Fatal("LogprobsOf(nil) is not nil")
	}
	in := []decoder.SampleInfo{{ID: 3, Logprob: -0.5, Top: []decoder.TokenLogprob{{ID: 3, Logprob: -0.5}, {ID: 9, Logprob: -2}}}, {ID: 4, Logprob: -1}}
	got := LogprobsOf(in)
	if len(got) != 2 || got[0].ID != 3 || got[0].Logprob != -0.5 || len(got[0].Top) != 2 || got[0].Top[1] != (TokenLogprob{ID: 9, Logprob: -2}) {
		t.Fatalf("LogprobsOf = %+v", got)
	}
	if got[1].Top != nil {
		t.Error("a nil Top became non-nil")
	}
	if e := LogprobsOf([]decoder.SampleInfo{}); e == nil || len(e) != 0 {
		t.Error("empty logprobs did not stay empty")
	}
}

func TestScripted_streamsItsIDsWithinTheBudget(t *testing.T) {
	var seen []Request
	s := Scripted{IDs: []int{5, 6, 7}, OnStart: func(r Request) { seen = append(seen, r) }}
	for _, c := range []struct{ max, want int }{{2, 2}, {3, 3}, {9, 3}, {0, 0}, {-1, 0}} {
		tokens, report, done := s.Start(context.Background(), Request{MaxTokens: c.max, Scope: KVScope{Keep: 4}})
		n := 0
		for range tokens {
			n++
		}
		done()
		if n != c.want || ReportNow(report) != nil {
			t.Errorf("max %d: %d tokens (want %d), report %v (want nil)", c.max, n, c.want, ReportNow(report))
		}
	}
	if len(seen) != 5 || seen[0].Scope.Keep != 4 {
		t.Errorf("OnStart saw %+v, want five requests carrying their scope", seen)
	}
}

type fakeDrafter struct{ BlockDrafter }

func TestDecoder_typedNilDrafterIsNone(t *testing.T) {
	var typed *fakeDrafter
	if (Decoder{Drafter: typed}).drafter() != nil {
		t.Error("a typed-nil drafter counts as attached")
	}
	if (Decoder{}).drafter() != nil {
		t.Error("no drafter counts as attached")
	}
	if (Decoder{Drafter: &fakeDrafter{}}).drafter() == nil {
		t.Error("an attached drafter counts as none")
	}
}
