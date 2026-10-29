package enginetest

import (
	"context"
	"testing"

	"github.com/townsendmerino/goinfer/engine"
)

func TestScripted_streamsItsIDsWithinTheBudget(t *testing.T) {
	var seen []engine.Request
	s := Scripted{IDs: []int{5, 6, 7}, OnStart: func(r engine.Request) { seen = append(seen, r) }}
	for _, c := range []struct{ max, want int }{{2, 2}, {3, 3}, {9, 3}, {0, 0}, {-1, 0}} {
		req := engine.Request{MaxTokens: c.max}
		req.Scope.Keep = 4
		st := s.Start(context.Background(), req)
		n := 0
		for range st.Tokens() {
			n++
		}
		st.Close()
		if n != c.want || st.Report() != nil {
			t.Errorf("max %d: %d tokens (want %d), report %v (want nil)", c.max, n, c.want, st.Report())
		}
	}
	if len(seen) != 5 || seen[0].Scope.Keep != 4 {
		t.Errorf("OnStart saw %+v, want five requests carrying their scope", seen)
	}
}
