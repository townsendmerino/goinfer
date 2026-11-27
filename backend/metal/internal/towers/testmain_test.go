//go:build darwin

package towers

import (
	"fmt"
	"os"
	"slices"
	"testing"

	gpu "github.com/townsendmerino/goinfer/backend/metal/internal/gpudev"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
)

type leakSwitch struct {
	name string
	read func() any
}

func sw[T comparable](name string, p *T) leakSwitch {
	return leakSwitch{name, func() any { return *p }}
}

func towerSwitches() []leakSwitch {
	return []leakSwitch{
		sw("eg2Block", &eg2Block),
		sw("gridIdleRelease", &gridIdleRelease),
		sw("gridReleaseCeiling", &gridReleaseCeiling),
		sw("gvKeepF32ForAB", &gvKeepF32ForAB),
		sw("siglipForceInt8", &siglipForceInt8),
	}
}

func TestMain(m *testing.M) {
	ss := towerSwitches()
	before := make([]any, len(ss))
	for i, s := range ss {
		before[i] = s.read()
	}
	gpu.UnlinedLine = mem.LineTest
	code := m.Run()
	var leaks []string
	for i, s := range ss {
		if now := s.read(); now != before[i] {
			leaks = append(leaks, fmt.Sprintf("%s: %v -> %v", s.name, before[i], now))
		}
	}
	slices.Sort(leaks)
	if len(leaks) > 0 {
		fmt.Fprintf(os.Stderr, "FAIL: %d package-level switch(es) were left changed by a test; each test restores the value it found:\n", len(leaks))
		for _, l := range leaks {
			fmt.Fprintln(os.Stderr, "  "+l)
		}
		code = 1
	}
	os.Exit(code)
}
