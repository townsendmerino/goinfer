package probe

import (
	"runtime"
	"testing"
)

// frozen is the readers as they were before they moved here (the commit is named in docs/tasks/task-subsystems-2026-10.md,
// MM0's gates), copied into the per-platform frozen_*_test.go files with their bodies unchanged and their names prefixed. A nil
// reader is one the platform did not have.
type frozen struct {
	physical  func() int64
	available func() int64
	swapUsed  func() (int64, bool)
	selfSwap  func() (int64, bool)
}

// TestHost_matchesFrozenReaders (MM0 gate G-MM0-1b): the probe, through Host and through the package functions, reads what the
// pre-move readers read on this box. Physical is compared exactly; a live figure is compared against two frozen readings taken
// around it, five tries at most; a reading's known-ness must agree exactly.
func TestHost_matchesFrozenReaders(t *testing.T) {
	f := frozenReaders()
	if f.physical == nil {
		t.Skipf("no frozen readers for %s", runtime.GOOS)
	}
	if got, want := Physical(), f.physical(); got != want || Host.Physical() != want {
		t.Errorf("Physical = %d (Host %d), frozen %d", got, Host.Physical(), want)
	}
	live := func(name string, ref func() (int64, bool), reads ...func() (int64, bool)) {
		for i, read := range reads {
			var lo, got, hi int64
			var lok, gok, hok bool
			for range 5 {
				lo, lok = ref()
				got, gok = read()
				hi, hok = ref()
				if lok == gok && gok == hok && (got == lo || got == hi || (got-lo)*(got-hi) <= 0) {
					break
				}
			}
			if lok != gok || gok != hok {
				t.Errorf("%s reader %d: known %v, frozen %v/%v", name, i, gok, lok, hok)
			} else if !(got == lo || got == hi || (got-lo)*(got-hi) <= 0) {
				t.Errorf("%s reader %d = %d, outside the frozen %d..%d on five tries", name, i, got, lo, hi)
			}
		}
	}
	if f.available != nil {
		ref := func() (int64, bool) { return f.available(), true }
		live("Available", ref, func() (int64, bool) { return Available(), true }, func() (int64, bool) { return Host.Available(), true })
	}
	if f.swapUsed != nil {
		live("SwapUsed", f.swapUsed, SwapUsed, Host.SwapUsed)
		if v, ok := f.swapUsed(); ok && v == 0 {
			t.Log("this box has no swap in use: SwapUsed compared 0 with 0")
		}
	}
	if f.selfSwap != nil {
		r, why := SelfSwapReader()
		if r == nil {
			t.Fatalf("SelfSwapReader = nil (%s) where the frozen reader read", why)
		}
		live("SelfSwap", f.selfSwap, r, Host.SelfSwap)
	} else if r, _ := SelfSwapReader(); r != nil {
		t.Error("SelfSwapReader reads on a platform whose frozen readers had none")
	}
}
