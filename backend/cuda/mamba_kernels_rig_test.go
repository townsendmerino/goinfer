//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"os"
	"regexp"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
)

// The part of cuda/kernels' attention-digest rig that mamba_kernels_test.go drives its kernels with. The digest tests
// moved to cuda/kernels; this test stays here because it loads a model through decoder for its reference.

type adAnchor struct {
	Device       string `json:"device"`
	Compute      string `json:"compute_capability"`
	DriverAPI    int    `json:"driver_api"`
	NvidiaModule string `json:"nvidia_module"`
}

// adMod names one PTX module a kernel ships in.
type adMod struct {
	name string
	ptx  []byte
}

// adRig holds the device, the stream and what a case allocates; release frees the case's buffers.
type adRig struct {
	t    *testing.T
	cx   *gc.Context
	st   *gc.Stream
	bg   context.Context
	mods map[string]*gc.Module
	fns  map[string]*gc.Function
	free []func()
}

func newAdRig(t *testing.T) (*adRig, adAnchor) {
	t.Helper()
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	dev, err := gc.GetDevice(0)
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	cx, err := dev.Primary()
	if err != nil {
		t.Skipf("primary ctx: %v", err)
	}
	t.Cleanup(func() { cx.Close() })
	a := adAnchor{}
	a.Device, _ = dev.Name()
	if major, minor, e := dev.ComputeCapability(); e == nil {
		a.Compute = itoa(major) + "." + itoa(minor)
	}
	a.DriverAPI, _ = gc.DriverVersion()
	if b, e := os.ReadFile("/proc/driver/nvidia/version"); e == nil {
		if m := regexp.MustCompile(`\s([0-9]+\.[0-9]+(?:\.[0-9]+)?)\s+Release`).FindSubmatch(b); m != nil {
			a.NvidiaModule = string(m[1])
		}
	}
	return &adRig{t: t, cx: cx, st: mustStream(t, cx), bg: context.Background(), mods: map[string]*gc.Module{}, fns: map[string]*gc.Function{}}, a
}

func (r *adRig) fn(m adMod, kernel string) *gc.Function {
	r.t.Helper()
	key := m.name + "/" + kernel
	if f, ok := r.fns[key]; ok {
		return f
	}
	mod, ok := r.mods[m.name]
	if !ok {
		var err error
		if mod, err = r.cx.LoadModule(m.ptx); err != nil {
			r.t.Fatalf("LoadModule(%s): %v", m.name, err)
		}
		r.mods[m.name] = mod
	}
	f, err := mod.Function(kernel)
	if err != nil {
		r.t.Fatalf("%s: Function(%s): %v", m.name, kernel, err)
	}
	r.fns[key] = f
	return f
}

func (r *adRig) launch(m adMod, kernel string, cfg gc.LaunchConfig, args ...gc.KernelArg) {
	r.t.Helper()
	if err := r.fn(m, kernel).LaunchOn(r.bg, r.st, cfg, args...); err != nil {
		r.t.Fatalf("%s/%s: launch: %v", m.name, kernel, err)
	}
	if err := r.st.Synchronize(r.bg); err != nil {
		r.t.Fatalf("%s/%s: sync: %v", m.name, kernel, err)
	}
}

func (r *adRig) release() {
	for _, f := range r.free {
		f()
	}
	r.free = r.free[:0]
}

// f32 uploads vals.
func (r *adRig) f32(vals []float32) *gc.Buffer[float32] {
	r.t.Helper()
	b := mustAlloc[float32](r.t, r.cx, max(len(vals), 1))
	r.free = append(r.free, func() { b.Close() })
	if len(vals) > 0 {
		if err := gc.CopyHtoD(r.bg, b, vals); err != nil {
			r.t.Fatalf("H2D: %v", err)
		}
	}
	return b
}

// zeros is an all-zero float buffer: a kernel that leaves a slot unwritten leaves a deterministic one.
func (r *adRig) zeros(n int) *gc.Buffer[float32] { return r.f32(make([]float32, n)) }

func (r *adRig) read(b *gc.Buffer[float32], n int) []float32 {
	r.t.Helper()
	out := make([]float32, n)
	if err := gc.CopyDtoH(r.bg, out, b); err != nil {
		r.t.Fatalf("D2H: %v", err)
	}
	return out
}

func adCfg(gx, gy, gz, bx int, shared int) gc.LaunchConfig {
	return gc.LaunchConfig{GridX: uint32(gx), GridY: uint32(gy), GridZ: uint32(gz), BlockX: uint32(bx), BlockY: 1, BlockZ: 1, SharedMemBytes: uint32(shared)}
}

func adI(v int) gc.KernelArg { return gc.ArgValue(int32(v)) }

// itoa formats a non-negative int for test labels.
func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for v > 0 {
		i--
		b[i] = byte('0' + v%10)
		v /= 10
	}
	return string(b[i:])
}
