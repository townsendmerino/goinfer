//go:build linux

package gpudev

import (
	"bytes"
	"testing"
)

// TestCUDA_uploadAsyncAt_offsetsAndEvents pins the three properties the stream-ordered primitives
// promise: an offset copy lands exactly its bytes and nothing else; ZeroAsync clears exactly its
// range; and a copy on one queue is visible to another queue that Waits on the event recorded
// after it, and to the host after Event.Sync — with no full-context synchronize anywhere.
func TestCUDA_uploadAsyncAt_offsetsAndEvents(t *testing.T) {
	d, q, _ := setup(t, "vadd")
	const N = 4096
	base := make([]byte, N)
	for i := range base {
		base[i] = byte(i * 7)
	}
	dst := d.NewBufferBytes(N)
	if err := Upload(dst, base); err != nil {
		t.Fatalf("Upload: %v", err)
	}
	src, err := NewHostBuffer[uint8](d, 2*N)
	if err != nil {
		t.Fatalf("NewHostBuffer: %v", err)
	}
	s := src.Slice()
	for i := range s {
		s[i] = byte(0xA0 + i%13)
	}

	// 1. offset copy: dst[1024:1536) <- src[2048:2560)
	if err := q.UploadAsyncAt(dst.At(1024), src, 2048, 512); err != nil {
		t.Fatalf("UploadAsyncAt: %v", err)
	}
	// 2. zero: dst[256:384)
	if err := q.ZeroAsync(dst.At(256), 128); err != nil {
		t.Fatalf("ZeroAsync: %v", err)
	}
	if err := q.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	want := append([]byte(nil), base...)
	copy(want[1024:1536], s[2048:2560])
	for i := 256; i < 384; i++ {
		want[i] = 0
	}
	got := make([]byte, N)
	if err := Download(dst, got); err != nil {
		t.Fatalf("Download: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("offset copy/zero: device bytes differ from expectation (first diff at %d)", firstDiff(got, want))
	}

	// 3. cross-queue: copy on q2, Record; q Waits; a host Sync on the event sees the bytes.
	q2 := d.NewCommandQueue()
	ev, err := d.NewEvent()
	if err != nil {
		t.Fatalf("NewEvent: %v", err)
	}
	if err := q2.UploadAsyncAt(dst, src, 0, N); err != nil {
		t.Fatalf("UploadAsyncAt(q2): %v", err)
	}
	if err := q2.Record(ev); err != nil {
		t.Fatalf("Record: %v", err)
	}
	if err := q.Wait(ev); err != nil {
		t.Fatalf("Wait: %v", err)
	}
	if err := ev.Sync(); err != nil {
		t.Fatalf("Event.Sync: %v", err)
	}
	if err := Download(dst, got); err != nil {
		t.Fatalf("Download: %v", err)
	}
	if !bytes.Equal(got, s[:N]) {
		t.Fatalf("cross-queue copy not visible after Event.Sync (first diff at %d)", firstDiff(got, s[:N]))
	}

	// Bounds are refused, not clamped.
	if err := q.UploadAsyncAt(dst.At(N-100), src, 0, 512); err == nil {
		t.Fatal("UploadAsyncAt overrunning dst was accepted")
	}
	if err := q.UploadAsyncAt(dst, src, 2*N-10, 512); err == nil {
		t.Fatal("UploadAsyncAt overrunning src was accepted")
	}
	if err := q.ZeroAsync(dst.At(N-1), 2); err == nil {
		t.Fatal("ZeroAsync overrunning dst was accepted")
	}
}

func firstDiff(a, b []byte) int {
	for i := range a {
		if i >= len(b) || a[i] != b[i] {
			return i
		}
	}
	return -1
}
