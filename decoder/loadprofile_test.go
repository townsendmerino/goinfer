package decoder

import (
	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"strings"
	"testing"
	"time"
)

// The profile has to be safe to use unconditionally: paths that do not instrument (safetensors,
// .giw) hand back nil, and the banner calls Summary() on whatever it gets.
func TestLoadProfile_nilIsUsable(t *testing.T) {
	var p *darch.LoadProfile
	p.Record("x", time.Second) // must not panic
	p.SetBytes(1 << 30)
	if got := p.Phases(); got != nil {
		t.Errorf("nil profile returned %v phases", got)
	}
	if got := p.Total(); got != 0 {
		t.Errorf("nil profile total = %v", got)
	}
	if got := p.Summary(); got != "" {
		t.Errorf("nil profile summary = %q, want empty so an uninstrumented load prints nothing", got)
	}
	if _, _, share := p.Dominant(); share != 0 {
		t.Errorf("nil profile share = %v", share)
	}
	var m *Model
	if m.LoadProfile() != nil {
		t.Error("nil model returned a profile")
	}
}

// An EMPTY profile must also print nothing. A load path that recorded no phase should be silent,
// not report "load 0s — 0% " — a zero that looks like a measurement is worse than no line.
func TestLoadProfile_emptyPrintsNothing(t *testing.T) {
	if got := (&darch.LoadProfile{}).Summary(); got != "" {
		t.Errorf("empty profile summary = %q, want empty", got)
	}
}

func TestLoadProfile_dominantAndSummary(t *testing.T) {
	p := &darch.LoadProfile{}
	p.Record("map", 50*time.Millisecond)
	p.Record("build", 4950*time.Millisecond)
	p.SetBytes(2 * (1 << 30))

	if got, want := p.Total(), 5*time.Second; got != want {
		t.Errorf("Total = %v, want %v", got, want)
	}
	name, d, share := p.Dominant()
	if name != "build" || d != 4950*time.Millisecond {
		t.Errorf("Dominant = %q %v, want build 4.95s", name, d)
	}
	if share < 0.98 || share > 1.0 {
		t.Errorf("share = %.3f, want ~0.99", share)
	}
	sum := p.Summary()
	for _, want := range []string{"load 5s", "map 50ms", "build 4.95s", "99% build", "2.00 GB source"} {
		if !strings.Contains(sum, want) {
			t.Errorf("Summary() = %q, missing %q", sum, want)
		}
	}
}

// Total is the SUM of phases, not a wall-clock span. The distinction matters: an unaccounted gap
// between phases must show up as the phases failing to add to the caller's own timing, rather than
// being silently folded into whichever phase happened to be adjacent.
func TestLoadProfile_totalIsTheSumNotASpan(t *testing.T) {
	p := &darch.LoadProfile{}
	p.Record("a", time.Second)
	time.Sleep(20 * time.Millisecond) // an unattributed gap
	p.Record("b", time.Second)
	if got := p.Total(); got != 2*time.Second {
		t.Errorf("Total = %v, want exactly 2s — the gap must not be absorbed into a phase", got)
	}
}
