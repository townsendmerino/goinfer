//go:build darwin || linux

package gpudev

import "testing"

// TestCoalesce_dispatchCount is the one part of the device-copy work that can be
// verified WITHOUT a GPU, and it covers the thing that actually costs the
// motivating consumer its bandwidth.
//
// The gap between one contiguous 20 MiB copy (~347 GB/s, 78% of a 2070 SUPER's
// peak) and the same bytes as 36 separate copies (~174 GB/s, 39%) is dispatch
// count, not bandwidth: gocudrv submits each op to a thread-locked executor and
// waits on a channel round trip, ~9 µs regardless of size. So "how many copies
// does this batch actually issue" is the number worth asserting, and it is a
// counting question — no device required. The bandwidth that results from it is
// a timing question and lives in the CUDA-only benchmarks.
func TestCoalesce_dispatchCount(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no GPU device: %v", err)
	}
	defer d.ReleaseObjects()

	const n = 4096
	src := d.NewBufferBytes(n)
	dst := d.NewBufferBytes(n)

	// Eight 512-byte pairs laid end to end in both buffers — the shape a caller
	// gets from walking per-layer sub-ranges of one combined allocation.
	var runs []DeviceCopy
	for i := range 8 {
		off := i * 512
		runs = append(runs, DeviceCopy{Dst: dst.At(off), Src: src.At(off), Bytes: 512})
	}

	cases := []struct {
		name   string
		in     []DeviceCopy
		want   int
		wantSz []int
	}{
		{"fully adjacent collapses to one", runs, 1, []int{4096}},
		{"a gap in src splits the run", []DeviceCopy{
			{Dst: dst.At(0), Src: src.At(0), Bytes: 512},
			{Dst: dst.At(512), Src: src.At(1024), Bytes: 512}, // src jumps
		}, 2, []int{512, 512}},
		{"a gap in dst splits the run", []DeviceCopy{
			{Dst: dst.At(0), Src: src.At(0), Bytes: 512},
			{Dst: dst.At(1024), Src: src.At(512), Bytes: 512}, // dst jumps
		}, 2, []int{512, 512}},
		{"different buffers never merge", []DeviceCopy{
			{Dst: dst.At(0), Src: src.At(0), Bytes: 512},
			{Dst: src.At(512), Src: dst.At(512), Bytes: 512},
		}, 2, []int{512, 512}},
		{"zero-byte pairs drop out", []DeviceCopy{
			{Dst: dst.At(0), Src: src.At(0), Bytes: 512},
			{Dst: dst.At(512), Src: src.At(512), Bytes: 0},
			{Dst: dst.At(512), Src: src.At(512), Bytes: 512},
		}, 1, []int{1024}},
		{"single pair is returned untouched", runs[:1], 1, []int{512}},
		{"empty stays empty", nil, 0, nil},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := coalesce(c.in)
			if len(got) != c.want {
				t.Fatalf("coalesce(%d pairs) issued %d copies, want %d", len(c.in), len(got), c.want)
			}
			for i, sz := range c.wantSz {
				if got[i].Bytes != sz {
					t.Errorf("copy %d is %d bytes, want %d", i, got[i].Bytes, sz)
				}
			}
		})
	}
}

// TestCoalesce_movesTheSameBytes is the correctness half: coalescing must be
// invisible in the result, not merely fewer calls. It runs the same eight
// adjacent pairs through CopyDeviceBatch (which coalesces) and compares against
// a byte pattern written by hand, so a merge that got an offset or a length
// wrong shows up as wrong data rather than as a passing count.
func TestCoalesce_movesTheSameBytes(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no GPU device: %v", err)
	}
	defer d.ReleaseObjects()

	const n = 4096
	want := make([]byte, n)
	for i := range want {
		want[i] = byte(i*7 + 3) // not a constant, so a misplaced range is visible
	}
	src := d.NewBufferBytes(n)
	dst := d.NewBufferBytes(n)
	if err := Upload(src, want); err != nil {
		t.Fatalf("upload: %v", err)
	}

	var runs []DeviceCopy
	for i := range 8 {
		off := i * 512
		runs = append(runs, DeviceCopy{Dst: dst.At(off), Src: src.At(off), Bytes: 512})
	}
	if err := CopyDeviceBatch(runs); err != nil {
		t.Fatalf("CopyDeviceBatch: %v", err)
	}

	got := make([]byte, n)
	if err := Download(dst, got); err != nil {
		t.Fatalf("download: %v", err)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("byte %d: got %d, want %d (first of possibly many)", i, got[i], want[i])
		}
	}
}

// TestCopyDevice_rejectsBadPairs pins the validation, which is the part a caller
// meets when something is wrong. The messages name the offsets and sizes on
// purpose: a batch of 36 that fails on one pair is unusable feedback otherwise.
func TestCopyDevice_rejectsBadPairs(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no GPU device: %v", err)
	}
	defer d.ReleaseObjects()

	small := d.NewBufferBytes(256)
	big := d.NewBufferBytes(1024)

	if err := CopyDevice(small, big, 1024); err == nil {
		t.Error("copy overrunning dst: want an error, got nil")
	}
	if err := CopyDevice(big, small, 1024); err == nil {
		t.Error("copy overrunning src: want an error, got nil")
	}
	if err := CopyDevice(big.At(512), small, -1); err == nil {
		t.Error("negative byte count: want an error, got nil")
	}
	// Zero bytes is a no-op, not an error — a caller looping over optional
	// per-layer buffers should not have to special-case an empty one.
	if err := CopyDevice(big, small, 0); err != nil {
		t.Errorf("zero-byte copy: want nil, got %v", err)
	}
	if err := CopyDeviceBatch(nil); err != nil {
		t.Errorf("empty batch: want nil, got %v", err)
	}
}
