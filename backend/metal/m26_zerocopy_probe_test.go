//go:build darwin

package metal

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"syscall"
	"testing"
	"time"
	"unsafe"
)

// TestM26ZeroCopyProbe is lever 2's probe (docs/tasks/task-m26-mac-2026-10.md): can the routed experts be read by the GPU
// straight from the mapped .giw through no-copy buffers, instead of being pread into the expert pool's slots? No model:
// a file of random bytes laid out as M26's expert tensors are (every expert's gate|up nibbles stacked, then every
// expert's down nibbles; 1,982,464 and 991,232 bytes, so half the down tensors start mid-page), mapped as decoder.Load
// maps a .giw (MAP_SHARED, VM_INHERIT_NONE), written uncached so a first read is a real page-in.
//
// Two arms, alternated round by round, each round one layer's routed set (8 experts, 16 tensors), drawn at random:
//   - pread, today's path: 16 concurrent preads into an 8-slot pool, then one command buffer of 16 dispatches that read
//     every byte of the slots;
//   - zero-copy: one command buffer of the same 16 dispatches, each binding the expert's own window of the mapping (one
//     no-copy buffer per tensor, made up front, page-rounded, with the expert's offset inside it).
//
// Cold rounds draw experts neither arm has touched; hot rounds draw from the whole file after it has been read once.
// Wired and anonymous pages are read (vm_stat, from a forked process, which VM_INHERIT_NONE keeps from copying the
// mapping) before the buffers exist, after the cold rounds, after 2 s idle, and after the hot rounds plus idle, all
// with every buffer still alive. Scales are left out (C-P01 preads them with the nibbles; 11% more bytes). Exploratory,
// by day: a microbenchmark gives direction, not size.
//
//	GOINFER_ZC_PROBE=1 go test -count=1 -run '^TestM26ZeroCopyProbe$' -v ./metal/
func TestM26ZeroCopyProbe(t *testing.T) {
	if os.Getenv("GOINFER_ZC_PROBE") != "1" {
		t.Skip("set GOINFER_ZC_PROBE=1 (writes a ~2 GiB scratch file and maps it)")
	}
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	defer d.ReleaseObjects()
	const (
		page    = 16384
		guBytes = 2 * 704 * 2816 / 2 // M26 gate|up nibbles per expert
		dBytes  = 2816 * 704 / 2     // M26 down nibbles per expert
		nE      = 704                // 5.5 layers' worth of experts
		topK    = 8
		rounds  = 40
	)
	dBase := int64(nE * guBytes)
	fileBytes := (dBase + nE*dBytes + (16<<20 - 1)) &^ (16<<20 - 1) // writeUncachedRandom writes 16 MiB chunks
	dir := os.Getenv("GOINFER_ZC_PROBE_DIR")
	if dir == "" {
		dir = t.TempDir()
	}
	path := filepath.Join(dir, "zcprobe.bin")
	writeUncachedRandom(t, path, fileBytes)
	defer os.Remove(path)
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	data, err := syscall.Mmap(int(f.Fd()), 0, int(fileBytes), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		t.Fatal(err)
	}
	defer syscall.Munmap(data)
	if _, _, e := syscall.Syscall(syscall.SYS_MINHERIT, uintptr(unsafe.Pointer(&data[0])), uintptr(len(data)), 2); e != 0 {
		t.Fatalf("minherit: %v", e)
	}

	lib, err := d.CompileLibrary(zcProbeSrc, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "read_all")
	if err != nil {
		t.Fatal(err)
	}
	const nThreads = 16384
	out := d.NewBufferLen(nThreads)
	uGU, uD := NewBufferU32(d, guBytes/16), NewBufferU32(d, dBytes/16)
	q := d.NewCommandQueue()
	mb := func(pages int64) float64 { return float64(pages) * page / (1 << 20) }

	base := vmStat(t)
	t0 := time.Now()
	type win struct {
		b   Buffer
		off int
	}
	window := func(off, n int64) win {
		lo, hi := off&^(page-1), (off+n+page-1)&^(page-1)
		return win{d.NewBufferNoCopy(unsafe.Pointer(&data[lo]), int(hi-lo)), int(off - lo)}
	}
	guWin, dWin := make([]win, nE), make([]win, nE)
	for e := range nE {
		guWin[e] = window(int64(e*guBytes), guBytes)
		dWin[e] = window(dBase+int64(e*dBytes), dBytes)
	}
	t.Logf("made %d no-copy buffers in %.1f ms (M26 needs 30 x 128 x 2 = 7,680 for the nibbles, x2 with scales)", 2*nE, float64(time.Since(t0).Microseconds())/1e3)
	afterMake := vmStat(t)

	poolGU, poolD := d.NewBufferLen(topK*guBytes/4), d.NewBufferLen(topK*dBytes/4)
	fd := int(f.Fd())
	encode := func(bind func(j int) (Buffer, Buffer)) (wall, gpuMs float64) {
		st := time.Now()
		e := q.Begin()
		for j := range topK {
			gu, dn := bind(j)
			e.Dispatch(pipe, nThreads, 256, gu, out, uGU)
			e.Dispatch(pipe, nThreads, 256, dn, out, uD)
		}
		e.End()
		return float64(time.Since(st).Microseconds()) / 1e3, (e.GPUEnd() - e.GPUStart()) * 1e3
	}
	preadArm := func(ids []int) (stage, cb, gpuMs float64) {
		st := time.Now()
		var wg sync.WaitGroup
		errs := make([]error, 2*topK)
		for j, id := range ids {
			wg.Add(2)
			go func() {
				defer wg.Done()
				errs[2*j] = preadIntoPoolSlot(fd, poolGU, j, guBytes/4, int64(id*guBytes))
			}()
			go func() {
				defer wg.Done()
				errs[2*j+1] = preadIntoPoolSlot(fd, poolD, j, dBytes/4, dBase+int64(id*dBytes))
			}()
		}
		wg.Wait()
		for _, err := range errs {
			if err != nil {
				t.Fatalf("pread: %v", err)
			}
		}
		stage = float64(time.Since(st).Microseconds()) / 1e3
		cb, gpuMs = encode(func(j int) (Buffer, Buffer) { return poolGU.At(j * guBytes), poolD.At(j * dBytes) })
		return stage, cb, gpuMs
	}
	zcArm := func(ids []int) (cb, gpuMs float64) {
		return encode(func(j int) (Buffer, Buffer) {
			g, dn := guWin[ids[j]], dWin[ids[j]]
			return g.b.At(g.off), dn.b.At(dn.off)
		})
	}

	rng := rand.New(rand.NewSource(7))
	fresh := rng.Perm(nE) // cold rounds take experts from here, never reused
	type stats struct{ stage, cb, gpu, total []float64 }
	run := func(label string, pick func() []int) (p, z stats) {
		for r := range rounds {
			for k := range 2 {
				ids := pick()
				if (r+k)%2 == 0 {
					s, c, g := preadArm(ids)
					p.stage, p.cb, p.gpu, p.total = append(p.stage, s), append(p.cb, c), append(p.gpu, g), append(p.total, s+c)
				} else {
					c, g := zcArm(ids)
					z.cb, z.gpu, z.total = append(z.cb, c), append(z.gpu, g), append(z.total, c)
				}
			}
		}
		med := func(v []float64) float64 {
			if len(v) == 0 {
				return 0
			}
			s := append([]float64(nil), v...)
			sort.Float64s(s)
			return s[len(s)/2]
		}
		t.Logf("RESULT %s (%d rounds x %d experts, medians, ms): pread stage %.2f + cb %.2f (gpu %.2f) = %.2f; zero-copy cb %.2f (gpu %.2f); zero-copy / pread %.3f",
			label, rounds, topK, med(p.stage), med(p.cb), med(p.gpu), med(p.total), med(z.cb), med(z.gpu), med(z.total)/med(p.total))
		return p, z
	}
	run("cold", func() []int {
		ids := fresh[:topK]
		fresh = fresh[topK:]
		return ids
	})
	afterCold := vmStat(t)
	time.Sleep(2 * time.Second)
	coldIdle := vmStat(t)
	for off := 0; off < len(data); off += page { // read the file once, so every page is in the page cache
		_ = data[off]
	}
	run("hot", func() []int { return rng.Perm(nE)[:topK] })
	time.Sleep(2 * time.Second)
	hotIdle := vmStat(t)

	// Sustained: what a decode does, the GPU never idle. 5 s of back-to-back zero-copy rounds over the whole file (a
	// 2 GiB working set, as M26's 16 GB is to RAM), wired pages sampled each second. If they climb toward the working
	// set and only fall when the GPU idles, the wire does not release under load and lever 2 is killed as built. Two
	// variants: the persistent buffers above, and buffers made for each round and released after it completes.
	sustained := func(label string, fresh bool) {
		w0 := vmStat(t)["wired"]
		var samples []string
		var peak int64
		var roundMs []float64
		st, next := time.Now(), time.Second
		for time.Since(st) < 5*time.Second {
			ids := rng.Perm(nE)[:topK]
			r0 := time.Now()
			if fresh {
				var made []Buffer
				encode(func(j int) (Buffer, Buffer) {
					g := window(int64(ids[j]*guBytes), guBytes)
					dn := window(dBase+int64(ids[j]*dBytes), dBytes)
					made = append(made, g.b, dn.b)
					return g.b.At(g.off), dn.b.At(dn.off)
				})
				for _, b := range made {
					d.ReleaseBuf(b)
				}
			} else {
				zcArm(ids)
			}
			roundMs = append(roundMs, float64(time.Since(r0).Microseconds())/1e3)
			if time.Since(st) >= next {
				w := vmStat(t)["wired"] - w0
				peak = max(peak, w)
				samples = append(samples, fmt.Sprintf("%.0f", mb(w)))
				next += time.Second
			}
		}
		sort.Float64s(roundMs)
		time.Sleep(2 * time.Second)
		t.Logf("RESULT sustained %s: %d rounds in 5 s, median %.2f ms a round; wired MB at each second %v (peak %+.0f MB), after 2 s idle %+.0f MB",
			label, len(roundMs), roundMs[len(roundMs)/2], samples, mb(peak), mb(vmStat(t)["wired"]-w0))
	}
	sustained("persistent buffers", false)
	sustained("fresh buffers, released each round", true)
	d.ReleaseAll()
	time.Sleep(500 * time.Millisecond)
	released := vmStat(t)
	touched := int64(rounds*topK) * (guBytes + dBytes) // the zero-copy arm's cold rounds' bytes
	for _, s := range []struct {
		label string
		m     map[string]int64
	}{{"buffers made", afterMake}, {"after cold rounds", afterCold}, {"cold + 2 s idle", coldIdle}, {"hot rounds + 2 s idle", hotIdle}, {"buffers released", released}} {
		t.Logf("RESULT pages %-22s wired %+8.1f MB, anonymous %+8.1f MB, file-backed %+8.1f MB (vs before the buffers)",
			s.label, mb(s.m["wired"]-base["wired"]), mb(s.m["anon"]-base["anon"]), mb(s.m["file"]-base["file"]))
	}
	fmt.Fprintf(os.Stderr, "[zc-probe] the zero-copy arm's cold rounds bound %.1f MB of experts\n", float64(touched)/(1<<20))
}

const zcProbeSrc = `
#include <metal_stdlib>
using namespace metal;
// Reads every 16-byte word of the bound window once (a grid-stride loop), as an expert GEMV reads its nibbles.
kernel void read_all(device const uint4* w [[buffer(0)]], device uint* out [[buffer(1)]], constant uint& n4 [[buffer(2)]],
    uint gid [[thread_position_in_grid]], uint gs [[threads_per_grid]]) {
    uint4 acc = 0;
    for (uint i = gid; i < n4; i += gs) acc ^= w[i];
    out[gid] = acc.x ^ acc.y ^ acc.z ^ acc.w;
}
`
