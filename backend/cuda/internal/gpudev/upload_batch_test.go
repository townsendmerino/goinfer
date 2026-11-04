//go:build darwin || linux

package gpudev

import "testing"

// TestUploadBatch_movesTheSameBytes is the correctness gate: a batch must land
// exactly what N separate Uploads would, including bind offsets, out-of-order
// destinations, and interleaved sizes. The batch's whole reason for existing is
// that it synchronizes once instead of N times, so the risk it introduces is a
// copy that was issued but not awaited — which shows up here as wrong bytes.
func TestUploadBatch_movesTheSameBytes(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no GPU device: %v", err)
	}
	defer d.ReleaseObjects()

	const n = 8192
	want := make([]byte, n)
	for i := range want {
		want[i] = byte(i*31 + 7) // not constant, so a misplaced range is visible
	}
	dst := d.NewBufferBytes(n)

	// Deliberately NOT in address order, and deliberately uneven — this mirrors the
	// motivating caller, where destination offsets come from LRU victim selection and
	// so arrive scattered rather than ascending.
	var batch []HostCopy
	for _, r := range [][2]int{{4096, 2048}, {0, 1024}, {6144, 2048}, {1024, 3072}} {
		off, size := r[0], r[1]
		batch = append(batch, HostCopy{Dst: dst.At(off), Src: want[off : off+size]})
	}
	if err := UploadBatch(batch); err != nil {
		t.Fatalf("UploadBatch: %v", err)
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

// TestUploadBatch_matchesLoopedUpload pins the property the caller actually swaps
// on: replacing N Uploads with one UploadBatch must be invisible in the result.
// Two buffers, same pairs, one filled each way, compared byte for byte.
func TestUploadBatch_matchesLoopedUpload(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no GPU device: %v", err)
	}
	defer d.ReleaseObjects()

	const n = 4096
	src := make([]byte, n)
	for i := range src {
		src[i] = byte(i*13 + 5)
	}
	viaLoop := d.NewBufferBytes(n)
	viaBatch := d.NewBufferBytes(n)

	ranges := [][2]int{{2048, 1024}, {0, 512}, {3072, 1024}, {512, 1536}}
	var batch []HostCopy
	for _, r := range ranges {
		off, size := r[0], r[1]
		if err := Upload(viaLoop.At(off), src[off:off+size]); err != nil {
			t.Fatalf("Upload at %d: %v", off, err)
		}
		batch = append(batch, HostCopy{Dst: viaBatch.At(off), Src: src[off : off+size]})
	}
	if err := UploadBatch(batch); err != nil {
		t.Fatalf("UploadBatch: %v", err)
	}

	a, b := make([]byte, n), make([]byte, n)
	if err := Download(viaLoop, a); err != nil {
		t.Fatalf("download loop: %v", err)
	}
	if err := Download(viaBatch, b); err != nil {
		t.Fatalf("download batch: %v", err)
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("byte %d differs: looped Upload %d, UploadBatch %d", i, a[i], b[i])
		}
	}
}

// TestUploadBatch_rejectsBadPairs pins the validation. The messages name the size
// and offset on purpose: the motivating caller submits ~240 pairs per token, and
// "one of them overran" is not usable feedback.
func TestUploadBatch_rejectsBadPairs(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no GPU device: %v", err)
	}
	defer d.ReleaseObjects()

	small := d.NewBufferBytes(256)
	src := make([]byte, 512)

	if err := UploadBatch([]HostCopy{{Dst: small, Src: src}}); err == nil {
		t.Error("upload overrunning dst: want an error, got nil")
	}
	// Overrun via the bind offset, not the length — the offset is the part a caller
	// computes and therefore the part they get wrong.
	if err := UploadBatch([]HostCopy{{Dst: small.At(200), Src: src[:100]}}); err == nil {
		t.Error("upload overrunning via bind offset: want an error, got nil")
	}
	// A bad pair must abort the whole batch, not apply the good ones first.
	if err := UploadBatch([]HostCopy{
		{Dst: small.At(0), Src: src[:64]},
		{Dst: small.At(240), Src: src[:64]}, // overruns
	}); err == nil {
		t.Error("batch with one bad pair: want an error, got nil")
	}
	// Empty and zero-length are no-ops, not errors — a caller looping over optional
	// per-layer buffers should not have to special-case them.
	if err := UploadBatch(nil); err != nil {
		t.Errorf("empty batch: want nil, got %v", err)
	}
	if err := UploadBatch([]HostCopy{{Dst: small, Src: nil}}); err != nil {
		t.Errorf("zero-length pair: want nil, got %v", err)
	}
}
