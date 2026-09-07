//go:build darwin

package decoder

import "testing"

// R13-follow-on (docs/measurements/cold-user-2026-09-07-macbook-arm64.md's live re-run): real
// vm_stat output shape, 16 KB pages (Apple Silicon — the M1 Pro that found this bug, not the
// traditional 4 KB), so a parser tested only against a 4 KB assumption would pass here and still
// be wrong on the machine that matters.
const sampleVMStat = `Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                5945.
Pages active:                             123456.
Pages inactive:                            98765.
Pages speculative:                          1234.
Pages throttled:                               0.
Pages wired down:                         271736.
Pages purgeable:                           61118.
"Translation faults":                  123456789.
Pages copy-on-write:                     1234567.
Pages zero filled:                      12345678.
Pages reactivated:                          1234.
Pages purged:                              12345.
File-backed pages:                        123456.
Anonymous pages:                          234567.
Pages stored in compressor:               345678.
Pages occupied by compressor:             123456.
Decompressions:                            12345.
Compressions:                              23456.
Pageins:                                  123456.
Pageouts:                                    1234.
Swapins:                                       0.
Swapouts:                                 621588.
`

func TestParseVMStatAvailable_realShape(t *testing.T) {
	got := parseVMStatAvailable(sampleVMStat)
	want := int64(5945+98765+1234+61118) * 16384
	if got != want {
		t.Errorf("parseVMStatAvailable = %d, want %d", got, want)
	}
	if got <= 0 {
		t.Errorf("available = %d, want a positive figure for this fixture", got)
	}
}

func TestParseVMStatAvailable_missingHeaderIsUnknown(t *testing.T) {
	if got := parseVMStatAvailable("garbage, no page size header\n"); got != 0 {
		t.Errorf("missing page-size header: got %d, want 0 (unknown)", got)
	}
}

func TestParseVMStatAvailable_missingFieldIsUnknown(t *testing.T) {
	// Mutation-relevant: if vm_stat's output ever drops one of the fields this depends on, the
	// parse must say "unknown", never silently sum the fields it did find — a partial sum would
	// UNDER-report usage in exactly the wrong direction (looks like more room than there is).
	const missingPurgeable = `Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                5945.
Pages inactive:                            98765.
Pages speculative:                          1234.
`
	if got := parseVMStatAvailable(missingPurgeable); got != 0 {
		t.Errorf("missing a required field: got %d, want 0 (unknown, not a partial sum)", got)
	}
}

func TestParseVMStatAvailable_zeroPageSizeIsUnknown(t *testing.T) {
	if got := parseVMStatAvailable("Mach Virtual Memory Statistics: (page size of 0 bytes)\nPages free: 100.\n"); got != 0 {
		t.Errorf("zero page size: got %d, want 0", got)
	}
}
