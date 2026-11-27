package giwkv

import "testing"

// checkGlobalLen is the predicate both storage arms share. Unit-tested directly because
// reaching it through a snapshot needs a populated KV and byte surgery on the payload; what
// matters is that 0 stays legal (KV-shared layers) and any other mismatch is refused.
func TestCheckGlobalLen(t *testing.T) {
	if e := checkGlobalLen(0, "keys", 0, 128); e != nil {
		t.Errorf("0 rejected: %v — a KV-shared layer stores nothing of its own", e)
	}
	if e := checkGlobalLen(0, "keys", 128, 128); e != nil {
		t.Errorf("an exact length rejected: %v", e)
	}
	for _, got := range []int{1, 127, 129, 1 << 20} {
		if e := checkGlobalLen(3, "vals", got, 128); e == nil {
			t.Errorf("len %d accepted against an expected 128 — the forward derives its key "+
				"count from `keys` and indexes `vals` at the same positions, so a short one is "+
				"an out-of-range read in the generation goroutine (M-04)", got)
		}
	}
}

func TestCheckGlobalPair_rejectsAsymmetricKV(t *testing.T) {
	if err := checkGlobalPair(0, "keys", "vals", 0, 0, 16); err != nil {
		t.Fatalf("both zero should pass: %v", err)
	}
	if err := checkGlobalPair(0, "keys", "vals", 16, 16, 16); err != nil {
		t.Fatalf("both matching should pass: %v", err)
	}
	if err := checkGlobalPair(0, "keys", "vals", 16, 0, 16); err == nil {
		t.Fatal("expected mismatch error when keys present and vals empty")
	}
}
