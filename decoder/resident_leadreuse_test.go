package decoder

import "testing"

// fastPrefillResident is a slotResident that also reports a fast-prefill floor, the way CUDA's resident does.
type fastPrefillResident struct {
	*slotResident
	floor int
}

func (f *fastPrefillResident) FastPrefillFloor() int { return f.floor }

func longPrompt(lead []int, n int) []int {
	p := append([]int(nil), lead...)
	for i := len(p); i < n; i++ {
		p = append(p, 1000+i)
	}
	return p
}

// A short reuse under a resident with non-exact prefill kernels is declined for a prompt that will run them: the rows a short earlier request left in the slot were
// computed by the exact kernels, the rest of the prompt runs fast, and the mixed KV changes the reply (MC4 ROOT CAUSE). Driven through residentAcquire, the function every
// reuse caller goes through, against a slot that really holds a short earlier conversation.
func TestResidentAcquire_declinesShortLeadUnderFastPrefill(t *testing.T) {
	const floor = 512
	lead := []int{7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}                           // a 13-token shared lead, as the chat header was
	earlier := append(append([]int(nil), lead...), 5001, 5002, 5003, 5004, 5005, 5006, 5007) // a short earlier conversation: under the floor, so its rows are exact; it shares only the lead
	for _, tc := range []struct {
		name  string
		res   func() any
		next  []int
		want  int
		wantN string
	}{
		{"fast kernels, long prompt, short lead: declined", func() any { return &fastPrefillResident{&slotResident{n: 2}, floor} }, longPrompt(lead, 941), 0, "reuse declined"},
		{"fast kernels, prompt under the floor: kept (same kernel class)", func() any { return &fastPrefillResident{&slotResident{n: 2}, floor} }, longPrompt(lead, 300), len(lead), "kept"},
		{"fast kernels off (floor 0): kept", func() any { return &fastPrefillResident{&slotResident{n: 2}, 0} }, longPrompt(lead, 941), len(lead), "kept"},
		{"resident without the capability: kept", func() any { return &slotResident{n: 2} }, longPrompt(lead, 941), len(lead), "kept"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := &Model{}
			switch r := tc.res().(type) {
			case *fastPrefillResident:
				m.resident = r
			case *slotResident:
				m.resident = r
			}
			m.residentAcquire(earlier, nil, nil)
			m.residentForgetIDs()
			m.residentCommitIDs(earlier, []int{1, 2, 3, 4}, nil, nil)
			if got := m.residentAcquire(tc.next, nil, nil); got != tc.want {
				t.Errorf("reuse = %d, want %d (%s)", got, tc.want, tc.wantN)
			}
		})
	}
}

// A real continuation is never declined: at or past minLeadReuseFast the rows are the conversation's own history, and re-prefilling them would be the cost this design avoids.
func TestResidentAcquire_keepsALongReuseUnderFastPrefill(t *testing.T) {
	rs := &fastPrefillResident{&slotResident{n: 2}, 512}
	m := &Model{resident: rs}
	hist := longPrompt([]int{7, 8, 9}, 600) // a long first turn (itself above the floor)
	m.residentAcquire(hist, nil, nil)
	m.residentForgetIDs()
	reply := []int{1, 2, 3, 4, 5}
	m.residentCommitIDs(hist, reply, nil, nil)
	next := append(append(append([]int(nil), hist...), reply...), longPrompt(nil, 700)[:200]...)
	want := len(hist) + len(reply)
	if got := m.residentAcquire(next, nil, nil); got != want {
		t.Errorf("a %d-token continuation reused %d, want all %d: a long reuse must survive", want, got, want)
	}
}

// The boundary: minLeadReuseFast-1 is declined, minLeadReuseFast is kept.
func TestDeclineShortLeadReuse_boundary(t *testing.T) {
	m := &Model{resident: &fastPrefillResident{&slotResident{n: 1}, 512}}
	long := longPrompt(nil, 941)
	for reuse, want := range map[int]int{0: 0, 1: 0, minLeadReuseFast - 1: 0, minLeadReuseFast: minLeadReuseFast, minLeadReuseFast + 1: minLeadReuseFast + 1, 900: 900} {
		if got := m.declineShortLeadReuse(long, reuse); got != want {
			t.Errorf("reuse %d -> %d, want %d", reuse, got, want)
		}
	}
	if got := m.declineShortLeadReuse(longPrompt(nil, 511), 5); got != 5 {
		t.Errorf("a prompt one under the floor must keep its reuse, got %d", got)
	}
	if got := m.declineShortLeadReuse(longPrompt(nil, 512), 5); got != 0 {
		t.Errorf("a prompt AT the floor runs the fast kernels (passPromptLen >= floor), so its short reuse is declined, got %d", got)
	}
}

// A conversation that takes another's idle slot in place keeps only as many of the shared rows as leave its own tail a pass
// of the class its cold prefill runs: a 14-row tail after a 1,000-row shared preamble would run the exact kernels where the
// cold prompt ran the fast ones, and the reply would depend on which conversation ran before it. Backed off to the block
// edge at 960, the tail is 54 rows. A continuation whose reused rows include its own decoded reply, and a prompt under
// the floor, keep their reuse.
func TestResidentAcquire_capsInPlaceReuseToTheColdClass(t *testing.T) {
	pre := longPrompt(nil, 1000)
	tail := func(seed, n int) []int {
		p := append([]int(nil), pre...)
		for i := range n {
			p = append(p, 50000+seed*100+i)
		}
		return p
	}
	reply := []int{1, 2, 3, 4}
	for _, slots := range []int{1, 2} {
		m := &Model{resident: &fastPrefillResident{&slotResident{n: slots}, 512}}
		a := tail(1, 5)
		m.residentAcquire(a, nil, nil)
		m.residentForgetIDs()
		m.residentCommitIDs(a, reply, nil, nil)
		if got := m.residentAcquire(tail(2, 14), nil, nil); got != 960 {
			t.Errorf("%d slots: a 14-row tail after a shared 1,000-row preamble reused %d, want 960", slots, got)
		}
	}

	m := &Model{resident: &fastPrefillResident{&slotResident{n: 1}, 512}}
	a := tail(1, 5)
	m.residentAcquire(a, nil, nil)
	m.residentForgetIDs()
	m.residentCommitIDs(a, reply, nil, nil)
	next := append(append(append([]int(nil), a...), reply...), 7, 8, 9)
	if got, want := m.residentAcquire(next, nil, nil), len(a)+len(reply); got != want {
		t.Errorf("a continuation over its own reply reused %d, want all %d", got, want)
	}

	m = &Model{resident: &fastPrefillResident{&slotResident{n: 1}, 512}}
	short := longPrompt(nil, 300)
	m.residentAcquire(short, nil, nil)
	m.residentForgetIDs()
	m.residentCommitIDs(short, reply, nil, nil)
	under := append(append([]int(nil), short[:290]...), 9001, 9002, 9003)
	if got := m.residentAcquire(under, nil, nil); got != 290 {
		t.Errorf("a prompt under the floor reused %d, want 290 (its cold prefill is exact too)", got)
	}
}
