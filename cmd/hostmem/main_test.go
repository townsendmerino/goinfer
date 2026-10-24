package main

import (
	"testing"

	"github.com/townsendmerino/goinfer/internal/hostmem"
)

// TestFigure: -reclaimable picks Reclaimable and the default Available, each in bytes or with -mb in MiB rounded down.
func TestFigure(t *testing.T) {
	r := hostmem.Reading{Available: 3<<30 + 5, Reclaimable: 1<<30 + 7}
	for _, c := range []struct {
		recl, mb bool
		want     int64
	}{{false, false, 3<<30 + 5}, {false, true, 3072}, {true, false, 1<<30 + 7}, {true, true, 1024}} {
		if got := figure(r, c.recl, c.mb); got != c.want {
			t.Errorf("figure(reclaimable %v, mb %v) = %d, want %d", c.recl, c.mb, got, c.want)
		}
	}
}
