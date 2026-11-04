//go:build cuda

package kernels

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"flag"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
)

// G1-K, OP1 on CUDA (docs/tasks/task-forward-architecture-2026-10.md §4.6a): every text-path attention kernel is driven on fixed inputs and
// its output bytes are hashed, so a change to how a score is formed shows as a changed digest. The digests are committed in
// testdata/attn_digests.json, recorded at base (the commit before step 1) and compared at every later commit.
//
//	go test -count=1 -tags cuda -run '^TestAttnKernelDigests' -v ./cuda/kernels/
//	go test -count=1 -tags cuda -run '^TestAttnKernelDigests$' -update-attn-digests ./cuda/kernels/   # re-record, at base only
//
// The digests belong to one device and one driver: the driver JIT-compiles the PTX, so another driver can move a bit without any source
// change. A run on another anchor FAILS as void and does not compare; re-anchoring is a deliberate act (CLAUDE.md, "CUDA rows are anchored
// to a specific NVIDIA driver version").

var updateAttnDigests = flag.Bool("update-attn-digests", false, "rewrite testdata/attn_digests.json from this run (at base only)")

const attnDigestFile = "../testdata/attn_digests.json"

type adAnchor struct {
	Device       string `json:"device"`
	Compute      string `json:"compute_capability"`
	DriverAPI    int    `json:"driver_api"`
	NvidiaModule string `json:"nvidia_module"`
}

type adFile struct {
	Anchor  adAnchor          `json:"anchor"`
	Digests map[string]string `json:"digests"`
}

// adMod names one PTX module a kernel ships in.
type adMod struct {
	name string
	ptx  []byte
}

// adCase is one kernel driven at one shape. kernel is the entry name (the coverage check keys on it); id is unique and names the shape.
type adCase struct {
	kernel string
	id     string
	run    func(r *adRig) [][]float32
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
	adUpload(r, b, vals)
	return b
}

// adUpload copies vals into b and returns only once they are on the device. A pageable cuMemcpyHtoD returns once the bytes
// are staged, and the rig launches on a non-blocking stream that is not ordered after it, so without the context synchronize a
// kernel can read a buffer before its copy lands: a stale block table then sends a KV kernel out of bounds. It is every upload
// the rig makes; TestAttnDigestRig_uploadsSynchronize holds it to that.
func adUpload[T gc.Supported](r *adRig, b *gc.Buffer[T], vals []T) {
	r.t.Helper()
	if len(vals) == 0 {
		return
	}
	if err := gc.CopyHtoD(r.bg, b, vals); err != nil {
		r.t.Fatalf("H2D: %v", err)
	}
	if err := r.cx.Synchronize(r.bg); err != nil {
		r.t.Fatalf("H2D sync: %v", err)
	}
}

// zeros is an all-zero float buffer: a kernel that leaves a slot unwritten leaves a deterministic one.
func (r *adRig) zeros(n int) *gc.Buffer[float32] { return r.f32(make([]float32, n)) }

func (r *adRig) i32(vals []int32) *gc.Buffer[int32] {
	r.t.Helper()
	b := mustAlloc[int32](r.t, r.cx, max(len(vals), 1))
	r.free = append(r.free, func() { b.Close() })
	adUpload(r, b, vals)
	return b
}

// kv uploads a K or V cache as the module's KV_T: f32 words, or the same values rounded to f16 for the _kv16 twin.
func (r *adRig) kv(vals []float32, f16 bool) gc.KernelArg {
	r.t.Helper()
	if !f16 {
		return gc.Arg(r.f32(vals))
	}
	h := make([]uint16, len(vals))
	f32SliceToF16RNE(h, vals)
	b := mustAlloc[uint16](r.t, r.cx, max(len(h), 1))
	r.free = append(r.free, func() { b.Close() })
	adUpload(r, b, h)
	return gc.Arg(b)
}

// kvt is the identity block table for a cache of positions rows, the table argument every KV kernel takes beside its cache.
func (r *adRig) kvt(positions int) gc.KernelArg { return gc.Arg(r.i32(kvIdentityIDs(positions))) }

func (r *adRig) read(b *gc.Buffer[float32], n int) []float32 {
	r.t.Helper()
	out := make([]float32, n)
	if err := gc.CopyDtoH(r.bg, out, b); err != nil {
		r.t.Fatalf("D2H: %v", err)
	}
	return out
}

// adRand is a fixed linear-congruential stream in [-amp, amp]: the same bytes on every box and every run.
func adRand(n int, seed uint32, amp float32) []float32 {
	out := make([]float32, n)
	s := seed
	for i := range out {
		s = s*1664525 + 1013904223
		out[i] = amp * float32(int32(s>>8)%2000-1000) / 1000
	}
	return out
}

func adScale(hd int) float32 { return float32(1.0 / math.Sqrt(float64(hd))) }

// adDigest is the SHA-256 of the outputs' bits, each buffer length-prefixed.
func adDigest(outs [][]float32) string {
	h := sha256.New()
	var b4 [4]byte
	for _, o := range outs {
		binary.LittleEndian.PutUint32(b4[:], uint32(len(o)))
		h.Write(b4[:])
		for _, v := range o {
			binary.LittleEndian.PutUint32(b4[:], math.Float32bits(v))
			h.Write(b4[:])
		}
	}
	return hex.EncodeToString(h.Sum(nil))
}

// adSourceKernels returns the attention kernels the .cu sources declare: every __global__ whose name says attention, split-KV, flash-decode
// or MLA attention. The vision and audio towers are not text-path kernels and are listed by name, with the reason, so that a new one is
// not exempted by accident.
func adSourceKernels(t *testing.T) []string {
	t.Helper()
	files, err := os.ReadDir("..")
	if err != nil {
		t.Fatal(err)
	}
	decl := regexp.MustCompile(`__global__\s+(?:void\s+)?(?:__launch_bounds__\([^)]*\)\s+)?(?:void\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\(`)
	name := regexp.MustCompile(`^(attention|attn_|splitkv_|fa_|mla_attn)`)
	seen := map[string]bool{}
	for _, f := range files {
		if !strings.HasSuffix(f.Name(), ".cu") {
			continue
		}
		b, err := os.ReadFile(filepath.Join("..", f.Name()))
		if err != nil {
			t.Fatal(err)
		}
		for _, m := range decl.FindAllSubmatch(b, -1) {
			if k := string(m[1]); name.MatchString(k) && !strings.HasPrefix(k, "attn_vit_") {
				seen[k] = true
			}
		}
	}
	out := make([]string, 0, len(seen))
	for k := range seen {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
