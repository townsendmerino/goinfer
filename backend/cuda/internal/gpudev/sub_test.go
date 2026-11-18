//go:build linux

package gpudev

import "testing"

// TestBufferSub: a sub-allocation is a buffer of its own for At, Len, Upload and Download, bounded by its own end, sharing its
// allocation's memory, and ReleaseBuf of it frees nothing.
func TestBufferSub(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	defer d.ReleaseObjects()
	whole := d.NewBufferBytes(4096)
	if err := Upload(whole, make([]byte, 4096)); err != nil {
		t.Fatal(err)
	}
	a, b := whole.Sub(0, 256, 64), whole.Sub(256, 512, 128) // 64 and 128 floats
	if b.Len() != 128 {
		t.Errorf("Len = %d, want 128", b.Len())
	}
	src := make([]float32, 128)
	for i := range src {
		src[i] = float32(i + 1)
	}
	if err := Upload(b, src); err != nil {
		t.Fatal(err)
	}
	if err := Upload(b.At(4*100), src[:28]); err != nil { // At binds inside the sub-allocation: floats 100..127
		t.Fatal(err)
	}
	if err := Upload(b.At(4*100), src[:29]); err == nil { // one float past its end, though the allocation has room
		t.Error("an upload past a sub-allocation's end was accepted")
	}
	if err := Upload(a, make([]float32, 65)); err == nil {
		t.Error("an upload into the next sub-allocation was accepted")
	}
	all := make([]float32, 1024)
	if err := Download(whole, all); err != nil {
		t.Fatal(err)
	}
	for i := range 128 {
		want := float32(i + 1)
		if i >= 100 {
			want = float32(i - 99)
		}
		if got := all[64+i]; got != want {
			t.Fatalf("float %d of the sub-allocation = %v, want %v", i, got, want)
		}
	}
	if all[0] != 0 || all[64+128] != 0 {
		t.Errorf("bytes outside the sub-allocation changed: %v, %v", all[0], all[64+128])
	}
	n, _ := d.LedgerLen()
	d.ReleaseBuf(b)
	if m, _ := d.LedgerLen(); m != n {
		t.Errorf("ReleaseBuf of a sub-allocation freed its allocation (ledger %d -> %d)", n, m)
	}
	got := make([]float32, 128)
	if err := Download(b, got); err != nil || got[0] != 1 {
		t.Errorf("the sub-allocation after ReleaseBuf: %v, first float %v", err, got[0])
	}
}
