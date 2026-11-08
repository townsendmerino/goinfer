//go:build linux

package gpudev

import (
	"bufio"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"unsafe"

	"github.com/eitamring/gocudrv/cudaresult"
	"github.com/eitamring/gocudrv/cudasys"
)

// cuMemHostRegister flags and the pointer attribute the probe reads (cuda.h).
const (
	cuMemHostRegisterReadOnly     = 0x08
	cuPointerAttributeMemoryType  = 2
	cuMemoryTypeHost              = 1
	cuDevAttrReadOnlyHostRegister = 113
)

// TestMEM10_fileMappingRegistration_probe is MEM10's tier-0 probe (docs/measurements/mem10-2026-11/README.md): a stack of the int4
// expert layout's shape (packed words and f16 scales; gpudev may not import the packer), written to a file, is mapped and registered in
// place with cuMemHostRegister, once per mapping and flag. It logs, per case, whether the driver accepts the registration and what that costs in memory (RssAnon, RssShmem,
// RssFile deltas) and in the file (its mtime: a write-intent pin dirties a shared mapping's pages). Where a registration succeeds, every
// expert's slot uploaded from the mapping must equal the bytes written, byte for byte.
func TestMEM10_fileMappingRegistration_probe(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no GPU device: %v", err)
	}
	defer d.ReleaseObjects()
	if v, err := d.dev.Attribute(cuDevAttrReadOnlyHostRegister); err == nil {
		t.Logf("device attribute READ_ONLY_HOST_REGISTER_SUPPORTED = %d", v)
	}

	// A stack of nE experts, each rows x k, in the packed int4 layout's sizes: K/8 weight words and K/32 f16 scales a row.
	const nE, rows, k = 128, 256, 2048
	wpr, gpr := k/8, k/32
	expW, expS := rows*wpr, rows*gpr // one expert's packed words and scales
	refW := make([]uint32, nE*expW)
	refS := make([]uint16, nE*expS)
	rng := lcg(20261108)
	for i := range refW {
		refW[i] = uint32(rng.word())
	}
	for i := range refS {
		refS[i] = f32tof16(float32(1+int(uint32(rng.word())%7)) / 64)
	}
	wBytes := unsafe.Slice((*byte)(unsafe.Pointer(&refW[0])), len(refW)*4)
	sBytes := unsafe.Slice((*byte)(unsafe.Pointer(&refS[0])), len(refS)*2)

	page := os.Getpagesize()
	for _, c := range []struct {
		name        string
		off         int // where the weight region starts in the file
		flags       uint32
		prot, share int
		rw          bool // open the file read-write
	}{
		{"read-only shared mapping, flags 0", 0, 0, syscall.PROT_READ, syscall.MAP_SHARED, false},
		{"read-only shared mapping, READ_ONLY", 0, cuMemHostRegisterReadOnly, syscall.PROT_READ, syscall.MAP_SHARED, false},
		{"read-only shared mapping at +64 B, READ_ONLY", page + 64, cuMemHostRegisterReadOnly, syscall.PROT_READ, syscall.MAP_SHARED, false},
		{"read-write shared mapping, flags 0", 0, 0, syscall.PROT_READ | syscall.PROT_WRITE, syscall.MAP_SHARED, true},
		{"read-write private mapping, flags 0", 0, 0, syscall.PROT_READ | syscall.PROT_WRITE, syscall.MAP_PRIVATE, false},
	} {
		t.Run(c.name, func(t *testing.T) {
			dir := t.TempDir() // /tmp is tmpfs on nobara, where file pages count as shared memory: point GOINFER_MEM10_PROBE_DIR at the disk the .giw lives on
			if d := os.Getenv("GOINFER_MEM10_PROBE_DIR"); d != "" {
				if dir, err = os.MkdirTemp(d, "mem10-probe-"); err != nil {
					t.Fatal(err)
				}
				defer os.RemoveAll(dir)
			}
			path := filepath.Join(dir, "stack.bin")
			sOff := (c.off + len(wBytes) + page - 1) / page * page
			buf := make([]byte, sOff+len(sBytes))
			copy(buf[c.off:], wBytes)
			copy(buf[sOff:], sBytes)
			if err := os.WriteFile(path, buf, 0o644); err != nil {
				t.Fatal(err)
			}
			mode := os.O_RDONLY
			if c.rw {
				mode = os.O_RDWR
			}
			f, err := os.OpenFile(path, mode, 0)
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			var st0 syscall.Stat_t
			if err := syscall.Fstat(int(f.Fd()), &st0); err != nil {
				t.Fatal(err)
			}
			m, err := syscall.Mmap(int(f.Fd()), 0, len(buf), c.prot, c.share)
			if err != nil {
				t.Fatal(err)
			}
			defer syscall.Munmap(m)
			buf = nil
			runtime.GC()

			before := procStatusMiB(t)
			runtime.LockOSThread()
			defer runtime.UnlockOSThread()
			if err := d.BindCurrentThread(); err != nil {
				t.Fatal(err)
			}
			drv := d.cx.Driver()
			regs := [][]byte{m[c.off : c.off+len(wBytes)], m[sOff : sOff+len(sBytes)]}
			var regErr error
			for i, r := range regs {
				if regErr = cudaresult.MemHostRegister(drv, &r[0], uint64(len(r)), c.flags); regErr != nil {
					regs = regs[:i]
					break
				}
			}
			defer func() {
				for _, r := range regs {
					_ = cudaresult.MemHostUnregister(drv, &r[0])
				}
				var st1 syscall.Stat_t
				if err := syscall.Fstat(int(f.Fd()), &st1); err == nil {
					t.Logf("file mtime changed: %v", st1.Mtim != st0.Mtim)
				}
			}()
			if regErr != nil {
				t.Logf("cuMemHostRegister refused: %v", regErr)
				return
			}
			after := procStatusMiB(t)
			var mt uint32
			if err := cudaresult.PointerGetAttribute(drv, unsafe.Pointer(&mt), cuPointerAttributeMemoryType, cudasys.CUdeviceptr(uintptr(unsafe.Pointer(&regs[0][0])))); err != nil || mt != cuMemoryTypeHost {
				t.Errorf("the registered mapping does not read as registered host memory: type %d, err %v", mt, err)
			}
			t.Logf("registered %d + %d B; RssAnon %+d MiB, RssShmem %+d MiB, RssFile %+d MiB, VmLck %+d MiB", len(wBytes), len(sBytes),
				after["RssAnon"]-before["RssAnon"], after["RssShmem"]-before["RssShmem"], after["RssFile"]-before["RssFile"], after["VmLck"]-before["VmLck"])
			if a := after["RssAnon"] - before["RssAnon"]; a > 8 {
				t.Logf("registering the mapping added %d MiB of anonymous memory: the pin copied the stack's pages", a)
			}

			// Upload every expert's slot from the mapping, as the slot loads do, and compare with the bytes written.
			slotW, slotS := NewBufferLenOf[uint32](d, expW), NewBufferLenOf[uint16](d, expS)
			gotW, gotS := make([]uint32, expW), make([]uint16, expS)
			bad := 0
			for e := range nE {
				if err := slotW.upload(regs[0][e*expW*4 : (e+1)*expW*4]); err != nil {
					t.Fatal(err)
				}
				if err := slotS.upload(regs[1][e*expS*2 : (e+1)*expS*2]); err != nil {
					t.Fatal(err)
				}
				if err := Download(slotW, gotW); err != nil {
					t.Fatal(err)
				}
				if err := Download(slotS, gotS); err != nil {
					t.Fatal(err)
				}
				for i := range gotW {
					if gotW[i] != refW[e*expW+i] {
						bad++
						break
					}
				}
				for i := range gotS {
					if gotS[i] != refS[e*expS+i] {
						bad++
						break
					}
				}
			}
			if bad != 0 {
				t.Errorf("%d of %d expert slots uploaded from the mapping differ from the bytes written", bad, nE)
			}
			t.Logf("%d expert slots uploaded from the mapping, %d differ", nE, bad)
		})
	}
}

// procStatusMiB reads this process's memory counters from /proc/self/status, in MiB.
func procStatusMiB(t *testing.T) map[string]int {
	t.Helper()
	f, err := os.Open("/proc/self/status")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	out := map[string]int{}
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		name, rest, ok := strings.Cut(sc.Text(), ":")
		if !ok {
			continue
		}
		fs := strings.Fields(rest)
		if len(fs) == 2 && fs[1] == "kB" {
			if v, err := strconv.Atoi(fs[0]); err == nil {
				out[name] = v / 1024
			}
		}
	}
	return out
}
