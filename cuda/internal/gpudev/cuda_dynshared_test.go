//go:build linux

package gpudev

import "testing"

// dynSharedPTX's touch_dyn writes off XOR 0x1234ABCD at byte offset off of its dynamic shared memory, reads it back through a
// volatile access (so the JIT cannot forward the value or drop the store) and stores it to out[0]. touch_static_dyn does the
// same beside 1 KiB of static shared memory, which it also touches. sm_50 so it JITs on every part with an opt-in past 48 KiB.
const dynSharedPTX = `
.version 7.0
.target sm_50
.address_size 64
.extern .shared .align 4 .b8 dyn[];
.shared .align 4 .b8 stat[1024];
.visible .entry touch_dyn(.param .u64 out, .param .u32 off)
{
	.reg .b32 %r<6>;
	.reg .b64 %rd<3>;
	ld.param.u64 %rd1, [out];
	ld.param.u32 %r1, [off];
	mov.u32 %r2, dyn;
	add.u32 %r3, %r2, %r1;
	xor.b32 %r4, %r1, 305441741;
	st.volatile.shared.u32 [%r3], %r4;
	ld.volatile.shared.u32 %r5, [%r3];
	cvta.to.global.u64 %rd2, %rd1;
	st.global.u32 [%rd2], %r5;
	ret;
}
.visible .entry touch_static_dyn(.param .u64 out, .param .u32 off)
{
	.reg .b32 %r<8>;
	.reg .b64 %rd<3>;
	ld.param.u64 %rd1, [out];
	ld.param.u32 %r1, [off];
	mov.u32 %r2, dyn;
	add.u32 %r3, %r2, %r1;
	xor.b32 %r4, %r1, 305441741;
	st.volatile.shared.u32 [%r3], %r4;
	mov.u32 %r6, stat;
	st.volatile.shared.u32 [%r6], %r1;
	ld.volatile.shared.u32 %r5, [%r3];
	ld.volatile.shared.u32 %r7, [%r6];
	add.u32 %r5, %r5, %r7;
	cvta.to.global.u64 %rd2, %rd1;
	st.global.u32 [%rd2], %r5;
	ret;
}
`

// TestCUDA_maxDynamicSharedBytes: a launch asking for more than the 48 KiB default fails until the kernel opts in with
// SetMaxDynamicSharedBytes, then runs and reads back a value it can only have written at the far end of that shared memory
// (out is zeroed before each launch, and the value depends on the offset); a request past the kernel's maximum is refused. A
// kernel with 1 KiB of static shared memory can opt into exactly the device limit less that 1 KiB (MaxDynamicSharedBytes).
func TestCUDA_maxDynamicSharedBytes(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	t.Cleanup(d.ReleaseObjects)
	opt := d.MaxSharedBytesPerBlockOptIn()
	if opt <= 48*1024 {
		t.Skipf("this device's opt-in maximum is %d bytes: no room past the 48 KiB default", opt)
	}
	lib, err := d.CompileLibrary([]byte(dynSharedPTX))
	if err != nil {
		t.Fatalf("CompileLibrary: %v", err)
	}
	q := d.NewCommandQueue()
	out := NewBufferOf(d, []uint32{0})
	run := func(p Pipeline, shared int) (uint32, error) {
		if err := Upload(out, []uint32{0}); err != nil {
			t.Fatal(err)
		}
		if err := q.Launch(p, GridOne(1, shared), Arg(out), ArgValue(uint32(shared-4))); err != nil {
			return 0, err
		}
		if err := q.Sync(); err != nil {
			return 0, err
		}
		got := make([]uint32, 1)
		if err := Download(out, got); err != nil {
			t.Fatal(err)
		}
		return got[0], nil
	}
	want := func(shared int, extra uint32) uint32 { return (uint32(shared-4) ^ 0x1234ABCD) + extra }

	p, err := d.NewComputePipeline(lib, "touch_dyn")
	if err != nil {
		t.Fatal(err)
	}
	if got := p.MaxDynamicSharedBytes(d); got != opt {
		t.Errorf("touch_dyn (no static shared memory): MaxDynamicSharedBytes %d, want the device limit %d", got, opt)
	}
	if v, err := run(p, 48*1024); err != nil || v != want(48*1024, 0) {
		t.Fatalf("a 48 KiB launch without the opt-in: %#x, %v; want %#x", v, err, want(48*1024, 0))
	}
	if _, err := run(p, opt); err == nil {
		t.Fatalf("a %d-byte launch ran without the opt-in: the test cannot see the attribute", opt)
	}
	if err := p.SetMaxDynamicSharedBytes(opt); err != nil {
		t.Fatalf("SetMaxDynamicSharedBytes(%d): %v", opt, err)
	}
	if v, err := run(p, opt); err != nil || v != want(opt, 0) {
		t.Fatalf("a %d-byte launch after the opt-in: %#x, %v; want %#x (written at byte %d)", opt, v, err, want(opt, 0), opt-4)
	}
	if err := p.SetMaxDynamicSharedBytes(opt + 1024); err == nil {
		t.Errorf("SetMaxDynamicSharedBytes(%d) past the opt-in maximum %d was accepted", opt+1024, opt)
	}

	ps, err := d.NewComputePipeline(lib, "touch_static_dyn")
	if err != nil {
		t.Fatal(err)
	}
	dynMax := ps.MaxDynamicSharedBytes(d)
	if dynMax != opt-1024 {
		t.Fatalf("touch_static_dyn (1 KiB static): MaxDynamicSharedBytes %d, want %d", dynMax, opt-1024)
	}
	if err := ps.SetMaxDynamicSharedBytes(dynMax); err != nil {
		t.Fatalf("SetMaxDynamicSharedBytes(%d) beside 1 KiB static: %v", dynMax, err)
	}
	if v, err := run(ps, dynMax); err != nil || v != want(dynMax, uint32(dynMax-4)) {
		t.Fatalf("a %d-byte launch beside 1 KiB static: %#x, %v; want %#x", dynMax, v, err, want(dynMax, uint32(dynMax-4)))
	}
	t.Logf("opt-in maximum %d bytes (static plus dynamic); a launch at it runs only after the opt-in; beside 1 KiB static, %d dynamic", opt, dynMax)
}
