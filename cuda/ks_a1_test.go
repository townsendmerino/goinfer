//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	_ "embed"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"
	"testing"
	"time"

	gc "github.com/eitamring/gocudrv/cuda"
)

// ksA1ProtoPTX is ks_a1_proto.cu: KS-A1's prototype native Q5_K / Q6_K GEMVs. Test-only.
//
//go:embed testdata/ks_a1_proto.ptx
var ksA1ProtoPTX []byte

// ksA1Shape is one GEMV shape of A1-G0 (docs/tasks/task-kernel-survey-2026-10.md, "KS-A1 on CUDA"). count is how many
// tensors of the shape a Phi-3 mini token reads; 0 marks a shape that is read but not graded.
type ksA1Shape struct {
	name  string
	kind  string // "q5k", "q6k" or "q4k"
	N, K  int
	count int
}

var ksA1Shapes = []ksA1Shape{
	{"phi3 qkv", "q5k", 9216, 3072, 32},
	{"phi3 q|k|v each (as dispatched)", "q5k", 3072, 3072, 0},
	{"phi3 ffn_down", "q6k", 3072, 8192, 16},
	{"phi3 head", "q6k", 32064, 3072, 1},
	{"1.5B ffn_down", "q6k", 1536, 8960, 0},
	{"1.5B attn_v", "q6k", 256, 1536, 0},
	{"phi3 gate|up (shipped q4k)", "q4k", 16384, 3072, 0},
}

// ksA1Half returns random f16 bits in [2^-9, 2^-5) and their value.
func ksA1Half(rng *rand.Rand) (uint16, float64) {
	e := 6 + rng.Intn(4)
	m := rng.Intn(1024)
	return uint16(e<<10 | m), math.Ldexp(1+float64(m)/1024, e-15)
}

func ksA1HalfVal(b uint16) float64 {
	e, m := int(b>>10)&31, float64(b&1023)
	v := math.Ldexp(1+m/1024, e-15)
	if e == 0 {
		v = math.Ldexp(m/1024, -14) // subnormal
	}
	if b&0x8000 != 0 {
		v = -v
	}
	return v
}

func ksA1ScaleMin(j int, q []byte) (int, int) {
	if j < 4 {
		return int(q[j] & 63), int(q[j+4] & 63)
	}
	return int((q[j+4] & 0x0F) | ((q[j-4] >> 6) << 4)), int((q[j+4] >> 4) | ((q[j] >> 6) << 4))
}

// ksA1Q5KRow dequantizes one row of Q5_K blocks by ggml's dequantize_row_q5_K, in float64.
func ksA1Q5KRow(raw []byte, out []float64) {
	for sb := 0; sb < len(raw)/176; sb++ {
		b := raw[sb*176 : sb*176+176]
		d, dmin := ksA1HalfVal(binary.LittleEndian.Uint16(b)), ksA1HalfVal(binary.LittleEndian.Uint16(b[2:]))
		sc, qh, qs := b[4:16], b[16:48], b[48:176]
		for j := range 4 {
			s1, m1 := ksA1ScaleMin(2*j, sc)
			s2, m2 := ksA1ScaleMin(2*j+1, sc)
			for l := range 32 {
				lo, hi := float64(qs[32*j+l]&15), float64(qs[32*j+l]>>4)
				if qh[l]&(1<<(2*j)) != 0 {
					lo += 16
				}
				if qh[l]&(2<<(2*j)) != 0 {
					hi += 16
				}
				out[256*sb+64*j+l] = d*float64(s1)*lo - dmin*float64(m1)
				out[256*sb+64*j+32+l] = d*float64(s2)*hi - dmin*float64(m2)
			}
		}
	}
}

// ksA1Q6KRow dequantizes one row of Q6_K blocks (ql[128], qh[64], scales[16], d) by ggml's dequantize_row_q6_K.
func ksA1Q6KRow(raw []byte, out []float64) {
	for sb := 0; sb < len(raw)/210; sb++ {
		b := raw[sb*210 : sb*210+210]
		d := ksA1HalfVal(binary.LittleEndian.Uint16(b[208:]))
		for n := range 2 {
			ql, qh, sc := b[64*n:], b[128+32*n:], b[192+8*n:]
			y := out[256*sb+128*n:]
			for l := range 32 {
				is := l / 16
				q1 := int(ql[l]&15|((qh[l]>>0)&3)<<4) - 32
				q2 := int(ql[l+32]&15|((qh[l]>>2)&3)<<4) - 32
				q3 := int(ql[l]>>4|((qh[l]>>4)&3)<<4) - 32
				q4 := int(ql[l+32]>>4|((qh[l]>>6)&3)<<4) - 32
				y[l] = d * float64(int8(sc[is])) * float64(q1)
				y[l+32] = d * float64(int8(sc[is+2])) * float64(q2)
				y[l+64] = d * float64(int8(sc[is+4])) * float64(q3)
				y[l+96] = d * float64(int8(sc[is+6])) * float64(q4)
			}
		}
	}
}

// ksA1Q4KRow dequantizes one row of Q4_K blocks.
func ksA1Q4KRow(raw []byte, out []float64) {
	for sb := 0; sb < len(raw)/144; sb++ {
		b := raw[sb*144 : sb*144+144]
		d, dmin := ksA1HalfVal(binary.LittleEndian.Uint16(b)), ksA1HalfVal(binary.LittleEndian.Uint16(b[2:]))
		sc, qs := b[4:16], b[16:144]
		for j := range 4 {
			s1, m1 := ksA1ScaleMin(2*j, sc)
			s2, m2 := ksA1ScaleMin(2*j+1, sc)
			for l := range 32 {
				out[256*sb+64*j+l] = d*float64(s1)*float64(qs[32*j+l]&15) - dmin*float64(m1)
				out[256*sb+64*j+32+l] = d*float64(s2)*float64(qs[32*j+l]>>4) - dmin*float64(m2)
			}
		}
	}
}

// ksA1Blocks returns N rows of random blocks of the kind, with sane f16 headers (d, dmin) and every other byte random.
func ksA1Blocks(rng *rand.Rand, kind string, N, K int) []byte {
	bs := map[string]int{"q5k": 176, "q6k": 210, "q4k": 144}[kind]
	raw := make([]byte, N*K/256*bs)
	rng.Read(raw)
	for o := 0; o < len(raw); o += bs {
		if kind == "q6k" {
			h, _ := ksA1Half(rng)
			binary.LittleEndian.PutUint16(raw[o+208:], h)
			continue
		}
		h, _ := ksA1Half(rng)
		hm, _ := ksA1Half(rng)
		binary.LittleEndian.PutUint16(raw[o:], h)
		binary.LittleEndian.PutUint16(raw[o+2:], hm)
	}
	return raw
}

// ksA1Q6KPlanes splits Q6_K blocks into the segment-wide planes the prototype reads: ql, qh, scales, d.
func ksA1Q6KPlanes(raw []byte) (ql, qh []uint8, sc []int8, d []uint16) {
	nb := len(raw) / 210
	ql, qh, sc, d = make([]uint8, nb*128), make([]uint8, nb*64), make([]int8, nb*16), make([]uint16, nb)
	for i := range nb {
		b := raw[210*i:]
		copy(ql[128*i:], b[:128])
		copy(qh[64*i:], b[128:192])
		for k := range 16 {
			sc[16*i+k] = int8(b[192+k])
		}
		d[i] = binary.LittleEndian.Uint16(b[208:])
	}
	return
}

func ksA1Words(raw []byte) []uint32 {
	w := make([]uint32, len(raw)/4)
	for i := range w {
		w[i] = binary.LittleEndian.Uint32(raw[4*i:])
	}
	return w
}

// ksA1Arm is one kernel at one shape: its weight bytes, its copies' launch args (rotated per launch) and its grid.
type ksA1Arm struct {
	name  string
	fn    *gc.Function
	bytes int64
	args  [][]gc.KernelArg
	cfg   gc.LaunchConfig
	out   *gc.Buffer[float32]
}

// TestKSA1_gemvTiming is A1-G0: every prototype checked against a float64 host reference (G-Q1's bar), then timed against
// the shipped gemv_w8a8_g32 at the same shape and a streaming-read ceiling, kernels interleaved within each batch.
// Run under the timing lock: python3 scripts/timing_lock.py run --label ks-a1-g0 -- go test -tags 'cuda goinfer_testhooks' -run TestKSA1_gemvTiming -v ./cuda/
func TestKSA1_gemvTiming(t *testing.T) {
	if os.Getenv("GOINFER_KSA1_TIMING") == "" {
		t.Skip("A1-G0 is a timed reading: set GOINFER_KSA1_TIMING=1 and run under the timing lock")
	}
	dev, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	t.Cleanup(dev.ReleaseObjects) // registered first, so the buffers' cleanups free them before the context goes
	ctx := dev.Context()
	bg := context.Background()
	// CLAUDE.md's void rule as A1-G0 registers it: the 1-minute load above 8.0 (half nobara's 16 cores), or free VRAM under the
	// run's need (about 0.6 GB, its largest shape's arms) plus 2 GB.
	const needVRAM = 0.6 + 2.0 // GB
	la := kso1cLoad1()
	free, total, err := ctx.MemInfo()
	if err != nil {
		t.Fatalf("cuMemGetInfo: %v", err)
	}
	fmt.Fprintf(os.Stderr, "A1-G0 start %s load %.2f; VRAM free %.0f MiB of %.0f\n", time.Now().Format(time.RFC3339), la,
		float64(free)/(1<<20), float64(total)/(1<<20))
	if la > 8.0 || float64(free)/1e9 < needVRAM {
		t.Fatalf("A1-G0 VOID: load %.2f (line 8.0), %.2f GB of VRAM free (floor %.1f GB); re-run", la, float64(free)/1e9, needVRAM)
	}
	proto, err := ctx.LoadModule(ksA1ProtoPTX)
	if err != nil {
		t.Fatalf("LoadModule(ks_a1_proto): %v", err)
	}
	shipped, err := ctx.LoadModule(actGroupPTX)
	if err != nil {
		t.Fatalf("LoadModule(actgroup): %v", err)
	}
	fnOf := func(m *gc.Module, name string) *gc.Function {
		f, err := m.Function(name)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		return f
	}
	stream := mustStream(t, ctx)
	const minRot = 64 << 20
	rng := rand.New(rand.NewSource(20261024))
	null := gc.ArgDevicePtr(0)

	sums := map[string]map[string]float64{} // kind → kernel → Σ count·ms over graded shapes
	var report []string
	bestQ5KGBs := 0.0 // the best Q5_K kernel's GB/s over gemv_w8a8_g32's at the graded qkv shape (KS-A1's bar)

	for si, sh := range ksA1Shapes {
		N, K, nSB := sh.N, sh.K, sh.K/256
		// Activations: the Phi-3 regime (an outlier, zero groups), per-32 codes, scales then sums.
		av := outlierVec(rng, K)
		aC, aSc := quantG32Go(av)
		nG := K / 32
		aBuf := make([]float32, 2*nG)
		copy(aBuf, aSc)
		for g := range nG {
			var s int32
			for _, c := range aC[32*g : 32*g+32] {
				s += int32(c)
			}
			aBuf[nG+g] = aSc[g] * float32(s)
		}
		da, das := upload(t, ctx, packActI8(aC)), upload(t, ctx, aBuf)

		// The native weights and the float64 reference.
		raw := ksA1Blocks(rng, sh.kind, N, K)
		bs := map[string]int{"q5k": 176, "q6k": 210, "q4k": 144}[sh.kind]
		rowBytes := nSB * bs
		want, bar := make([]float64, N), make([]float64, N)
		wrow := make([]float64, K)
		for n := range N {
			r := raw[n*rowBytes : (n+1)*rowBytes]
			switch sh.kind {
			case "q5k":
				ksA1Q5KRow(r, wrow)
			case "q6k":
				ksA1Q6KRow(r, wrow)
			default:
				ksA1Q4KRow(r, wrow)
			}
			for k := range K {
				x := wrow[k] * float64(aC[k]) * float64(aSc[k/32])
				want[n] += x
				bar[n] += math.Abs(x)
			}
		}
		nativeBytes := int64(len(raw))
		copiesOf := func(b int64) int { return int(max(1, (minRot+b-1)/b)) }

		var arms []*ksA1Arm
		// The shipped int8 kernel: random int8 rows and row scales (its own bytes; its output is not graded here).
		{
			w8 := make([]int32, N*K/4)
			for i := range w8 {
				w8[i] = int32(rng.Uint32())
			}
			ws := make([]float32, N)
			for i := range ws {
				ws[i] = 0.01
			}
			b := int64(N)*int64(K) + 4*int64(N)
			arm := &ksA1Arm{name: "int8 gemv_w8a8_g32", fn: fnOf(shipped, "gemv_w8a8_g32"), bytes: b,
				cfg: gc.LaunchConfig{GridX: uint32((N + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
				out: mustAlloc[float32](t, ctx, N)}
			for range copiesOf(b) {
				dw, dws := upload(t, ctx, w8), upload(t, ctx, ws)
				arm.args = append(arm.args, []gc.KernelArg{gc.Arg(dw), gc.Arg(da), gc.Arg(dws), gc.Arg(das), null,
					gc.ArgValue(int32(N)), gc.ArgValue(int32(K / 4)), gc.Arg(arm.out), gc.ArgValue(int32(0))})
			}
			arms = append(arms, arm)
		}
		var natives []*ksA1Arm
		switch sh.kind {
		case "q5k", "q4k":
			words := ksA1Words(raw)
			var dws []*gc.Buffer[uint32]
			for range copiesOf(nativeBytes) {
				dws = append(dws, upload(t, ctx, words))
			}
			type v struct {
				name string
				R    int
			}
			vs := []v{{"gemv_q5k_a_r1", 1}, {"gemv_q5k_a_r2", 2}, {"gemv_q5k_a_r4", 4}, {"gemv_q5k_b_r1", 1}, {"gemv_q5k_b_r2", 2}, {"gemv_q5k_b_r4", 4}}
			mod := proto
			if sh.kind == "q4k" {
				vs, mod = []v{{"gemv_q4k_g32", 1}}, shipped
			}
			for _, x := range vs {
				arm := &ksA1Arm{name: x.name, fn: fnOf(mod, x.name), bytes: nativeBytes,
					cfg: gc.LaunchConfig{GridX: uint32((N + 8*x.R - 1) / (8 * x.R)), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
					out: mustAlloc[float32](t, ctx, N)}
				for _, dw := range dws {
					arm.args = append(arm.args, []gc.KernelArg{gc.Arg(dw), gc.Arg(da), gc.Arg(das), null,
						gc.ArgValue(int32(N)), gc.ArgValue(int32(nSB)), gc.Arg(arm.out), gc.ArgValue(int32(0))})
				}
				natives = append(natives, arm)
			}
		case "q6k":
			ql, qh, sc, d := ksA1Q6KPlanes(raw)
			type planes struct {
				ql, qh *gc.Buffer[uint8]
				sc     *gc.Buffer[int8]
				d      *gc.Buffer[uint16]
			}
			var ps []planes
			for range copiesOf(nativeBytes) {
				ps = append(ps, planes{upload(t, ctx, ql), upload(t, ctx, qh), upload(t, ctx, sc), upload(t, ctx, d)})
			}
			for _, R := range []int{1, 2, 4} {
				name := fmt.Sprintf("gemv_q6k_r%d", R)
				arm := &ksA1Arm{name: name, fn: fnOf(proto, name), bytes: nativeBytes,
					cfg: gc.LaunchConfig{GridX: uint32((N + 8*R - 1) / (8 * R)), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
					out: mustAlloc[float32](t, ctx, N)}
				for _, p := range ps {
					arm.args = append(arm.args, []gc.KernelArg{gc.Arg(p.ql), gc.Arg(p.qh), gc.Arg(p.sc), gc.Arg(p.d), gc.Arg(da), gc.Arg(das), null,
						gc.ArgValue(int32(N)), gc.ArgValue(int32(nSB)), gc.Arg(arm.out), gc.ArgValue(int32(0))})
				}
				natives = append(natives, arm)
			}
		}
		// The streaming ceiling over the int8 arm's bytes (at least minRot, rounded to 16-byte words).
		ceilWords := max(int64(minRot), arms[0].bytes) / 16
		{
			buf := mustAlloc[uint32](t, ctx, int(ceilWords*4))
			sink := mustAlloc[uint32](t, ctx, 2048)
			arm := &ksA1Arm{name: "stream_read ceiling", fn: fnOf(proto, "stream_read"), bytes: ceilWords * 16,
				cfg:  gc.LaunchConfig{GridX: 2048, GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
				args: [][]gc.KernelArg{{gc.Arg(buf), gc.ArgValue(ceilWords), gc.Arg(sink)}}}
			arms = append(arms, arm)
		}

		// Correctness first: every native arm within G-Q1's bar on every row.
		for _, arm := range natives {
			if err := arm.fn.LaunchOn(bg, stream, arm.cfg, arm.args[0]...); err != nil {
				t.Fatalf("%s %s launch: %v", sh.name, arm.name, err)
			}
			if err := stream.Synchronize(bg); err != nil {
				t.Fatalf("%s %s sync: %v", sh.name, arm.name, err)
			}
			got := download(t, arm.out, N)
			worst := 0.0
			for n := range N {
				e := math.Abs(float64(got[n])-want[n]) / (bar[n] + 1e-30)
				if math.IsNaN(e) || e > worst {
					worst = e
					if math.IsNaN(e) {
						worst = math.Inf(1)
					}
				}
			}
			if worst > 1e-5 {
				t.Fatalf("%s %s: worst |err|/Σ|w·a| %.3g > 1e-5 — not timed; A1-G0 is void for %s", sh.name, arm.name, worst, sh.kind)
			}
			fmt.Fprintf(os.Stderr, "A1-G0 check %-28s %-20s worst |err|/Σ|w·a| %.3g (bar 1e-5)\n", sh.name, arm.name, worst)
		}
		arms = append(arms, natives...)

		// Timing: 3 warm launches each, then 7 batches; within a batch every arm runs 50 launches, rotating its copies.
		const iters, batches = 50, 7
		times := make([][]float64, len(arms))
		run := func(arm *ksA1Arm, n int) {
			for i := range n {
				if err := arm.fn.LaunchOn(bg, stream, arm.cfg, arm.args[i%len(arm.args)]...); err != nil {
					t.Fatalf("%s %s launch: %v", sh.name, arm.name, err)
				}
			}
		}
		for _, arm := range arms {
			run(arm, 3)
		}
		if err := stream.Synchronize(bg); err != nil {
			t.Fatal(err)
		}
		for range batches {
			for ai, arm := range arms {
				start, _ := ctx.NewEvent()
				done, _ := ctx.NewEvent()
				if err := start.Record(stream); err != nil {
					t.Fatal(err)
				}
				run(arm, iters)
				if err := done.Record(stream); err != nil {
					t.Fatal(err)
				}
				if err := stream.Synchronize(bg); err != nil {
					t.Fatal(err)
				}
				el, err := start.Elapsed(done)
				if err != nil {
					t.Fatal(err)
				}
				times[ai] = append(times[ai], float64(el.Nanoseconds())/1e6/iters)
				start.Close()
				done.Close()
			}
		}
		med := func(xs []float64) float64 {
			s := append([]float64(nil), xs...)
			sort.Float64s(s)
			return s[len(s)/2]
		}
		int8ms := med(times[0])
		fmt.Fprintf(os.Stderr, "A1-G0 [%d/%d] %s %d×%d (%s, ×%d a token)\n", si+1, len(ksA1Shapes), sh.name, N, K, sh.kind, sh.count)
		for ai, arm := range arms {
			ms := med(times[ai])
			gbps := float64(arm.bytes) / (ms * 1e-3) / 1e9
			line := fmt.Sprintf("  %-22s %8.4f ms  %6.1f GB/s  time/int8 %.3f  GB/s/int8 %.3f  (%.2f MB, %d copies)",
				arm.name, ms, gbps, ms/int8ms, gbps/(float64(arms[0].bytes)/(int8ms*1e-3)/1e9), float64(arm.bytes)/1e6, len(arm.args))
			fmt.Fprintln(os.Stderr, line)
			report = append(report, sh.name+" "+line)
			if sh.name == "phi3 qkv" && strings.HasPrefix(arm.name, "gemv_q5k") {
				bestQ5KGBs = max(bestQ5KGBs, gbps/(float64(arms[0].bytes)/(int8ms*1e-3)/1e9))
			}
			if sh.count > 0 && arm.name != "stream_read ceiling" {
				k := sh.kind
				if arm.name == arms[0].name {
					k = sh.kind + " int8"
				}
				if sums[k] == nil {
					sums[k] = map[string]float64{}
				}
				sums[k][arm.name] += float64(sh.count) * ms
			}
		}
	}
	fmt.Fprintln(os.Stderr, "A1-G0 summed over Phi-3's graded tensors (ms a token):")
	for _, kind := range []string{"q5k", "q6k"} {
		base := sums[kind+" int8"]["int8 gemv_w8a8_g32"]
		names := make([]string, 0, len(sums[kind]))
		for n := range sums[kind] {
			names = append(names, n)
		}
		sort.Strings(names)
		best := math.Inf(1)
		for _, n := range names {
			fmt.Fprintf(os.Stderr, "  %s %-16s %.4f ms  /int8 %.4f (int8 %.4f)\n", kind, n, sums[kind][n], sums[kind][n]/base, base)
			best = min(best, sums[kind][n]/base)
		}
		if !(best <= 0.95) {
			t.Errorf("A1-G0: the best %s kernel's summed time is %.4f of the int8 kernel's, over the build rule's 0.95", kind, best)
		}
	}
	if !(bestQ5KGBs >= 0.95) {
		t.Errorf("A1-G0: KS-A1 NO-GO by the survey's bar: the best Q5_K kernel reaches %.4f of gemv_w8a8_g32's GB/s, under 0.95", bestQ5KGBs)
	}
	fmt.Fprintf(os.Stderr, "A1-G0 KS-A1: the best Q5_K kernel at 9216x3072 reaches %.4f of gemv_w8a8_g32's GB/s (bar 0.95)\n", bestQ5KGBs)
	t.Log(strings.Join(report, "\n"))
}
