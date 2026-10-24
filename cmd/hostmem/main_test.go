package main

import (
	"testing"

	"github.com/townsendmerino/goinfer/internal/hostmem"
)

// TestFigure: -reclaimable picks Reclaimable and the default Available, each in bytes or with -mb in MiB rounded down.
func TestFigure(t *testing.T) {
	// Each figure is one byte short of the next MiB, so -mb must round down (to 3072 and 1024); to nearest would read one higher.
	r := hostmem.Reading{Available: 3<<30 + 1<<20 - 1, Reclaimable: 1<<30 + 1<<20 - 1}
	for _, c := range []struct {
		recl, mb bool
		want     int64
	}{{false, false, 3<<30 + 1<<20 - 1}, {false, true, 3072}, {true, false, 1<<30 + 1<<20 - 1}, {true, true, 1024}} {
		if got := figure(r, c.recl, c.mb); got != c.want {
			t.Errorf("figure(reclaimable %v, mb %v) = %d, want %d", c.recl, c.mb, got, c.want)
		}
	}
}
