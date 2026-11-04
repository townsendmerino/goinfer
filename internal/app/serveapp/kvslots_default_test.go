package serveapp

import (
	"flag"
	"io"
	"testing"
)

// TestKVSessions_defaultReachesTheBackend is E-P09's serve half: -kv-sessions not given travels to the backend as a
// default it may lower (decoder.Options.ResidentKVSlotsDefault; Metal keeps 2 slots), and a given -kv-sessions, even
// the default's own value 4, travels as the operator's count. Parsed through registerFlags and markGivenFlags, the way
// main does it.
func TestKVSessions_defaultReachesTheBackend(t *testing.T) {
	for _, tc := range []struct {
		args        []string
		wantSlots   int
		wantDefault bool
	}{
		{nil, 4, true},
		{[]string{"-kv-sessions", "4"}, 4, false},
		{[]string{"-kv-sessions=3"}, 3, false},
		{[]string{"-kv-sessions", "0"}, 0, false},
	} {
		fs := flag.NewFlagSet("serve", flag.ContinueOnError)
		fs.SetOutput(io.Discard)
		sf := registerFlags(fs)
		if err := fs.Parse(tc.args); err != nil {
			t.Fatalf("%v: parse: %v", tc.args, err)
		}
		markGivenFlags(fs, &sf.cfg)
		o := modelSpec{}.options(sf.cfg)
		if o.ResidentKVSlots != tc.wantSlots || o.ResidentKVSlotsDefault != tc.wantDefault {
			t.Errorf("%v: ResidentKVSlots %d, ResidentKVSlotsDefault %v; want %d, %v", tc.args, o.ResidentKVSlots, o.ResidentKVSlotsDefault, tc.wantSlots, tc.wantDefault)
		}
	}
}
