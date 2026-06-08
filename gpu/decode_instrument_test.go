//go:build gpu

package gpu

import (
	"testing"
	"time"

	"github.com/cogentcore/webgpu/wgpu"
)

// TestDecode_instrument decomposes the ~44.7 ms/token that the §1 single-pass
// DecodeRunner leaves on the table (34.7 GB/s = ~10% of roofline). Two
// hypotheses about this code have already missed by ~4× (the §0.5 1.48× probe,
// then §1's "93% pass overhead" → actually ~24 µs/pass), so instead of guessing
// where the time goes a third time we measure it:
//
//	(a) one resident gate GEMV (13.76 MB) standalone → its real GB/s. The fork:
//	    bandwidth-correct kernel ⇒ the cost is dispatch count → §2/§3 pays;
//	    slow kernel ⇒ fusion won't save it.
//	(b) a 1-workgroup dependent no-op dispatch → the per-dispatch launch+barrier
//	    floor, isolated from kernel work.
//	(c) reconstruct the token: fixed + 197 gemv-dispatches + 338 glue-dispatches,
//	    and check it sums to the measured 44.7 ms → matmul-time vs glue-time.
//
// Method: record K identical dispatches into ONE compute pass, ONE submit, ONE
// blocking Poll; the slope (t(Khi)-t(Klo))/(Khi-Klo) is the per-dispatch GPU
// cost with the fixed submit/poll/encode overhead differenced out; the intercept
// is that fixed overhead. Logs; run -v.
func TestDecode_instrument(t *testing.T) {
	if testing.Short() {
		t.Skip("instrument")
	}
	ctx, err := New()
	if err != nil {
		t.Skipf("no GPU adapter: %v", err)
	}
	defer ctx.Close()
	if err := ctx.ensureGEMV(); err != nil {
		t.Fatal(err)
	}

	// timeK records K dispatches of (pl,bg) into one pass, submits once, polls
	// once (blocking → GPU done), and returns the wall time. Min over reps trims
	// scheduler noise; the blocking Poll makes wall ≈ encode + GPU-exec + sync.
	timeK := func(pl *wgpu.ComputePipeline, bg *wgpu.BindGroup, gx, gy uint32, K, reps int) time.Duration {
		best := time.Hour
		for r := 0; r < reps; r++ {
			enc, _ := ctx.device.CreateCommandEncoder(nil)
			pass := enc.BeginComputePass(nil)
			pass.SetPipeline(pl)
			pass.SetBindGroup(0, bg, nil)
			for i := 0; i < K; i++ {
				pass.DispatchWorkgroups(gx, gy, 1)
			}
			pass.End()
			pass.Release()
			cmd, _ := enc.Finish(nil)
			t0 := time.Now()
			ctx.queue.Submit(cmd)
			ctx.device.Poll(true, nil)
			d := time.Since(t0)
			cmd.Release()
			enc.Release()
			if d < best {
				best = d
			}
		}
		return best
	}
	// perDispatch returns (slope µs/dispatch, intercept µs) from two K points.
	perDispatch := func(pl *wgpu.ComputePipeline, bg *wgpu.BindGroup, gx, gy uint32, klo, khi int) (float64, float64) {
		tlo := timeK(pl, bg, gx, gy, klo, 20)
		thi := timeK(pl, bg, gx, gy, khi, 20)
		us := func(d time.Duration) float64 { return float64(d.Microseconds()) }
		slope := (us(thi) - us(tlo)) / float64(khi-klo)
		intercept := us(tlo) - slope*float64(klo)
		return slope, intercept
	}

	// ---- (a) resident gate GEMV [inter=8960, hidden=1536], ~13.76 MB int8 ----
	const inter, hidden = 8960, 1536
	bq, s := quantW(inter, hidden, 42)
	gate, err := ctx.UploadW8A8(bq, s, inter, hidden)
	if err != nil {
		t.Fatal(err)
	}
	defer gate.Release()
	gr, err := ctx.NewGEMVRunner(gate)
	if err != nil {
		t.Fatal(err)
	}
	defer gr.Release()
	// prime the activation buffers so the dispatch reads real data
	gr.Run(make([]int8, hidden), 1.0)
	gx, gy := gemvGrid(gate.rows)
	gSlope, gFixed := perDispatch(ctx.gemvPipeline, gr.bg, gx, gy, 1, 200)
	gemvBytes := float64(gate.rows*gate.kp + gate.rows*4) // int8 weights + f32 scales
	gemvGBs := gemvBytes / (gSlope * 1e3)                 // bytes / (µs→s ×1e-6) /1e9
	t.Logf("(a) gate GEMV [%d×%d]  %.1f µs/dispatch  →  %.1f GB/s  (%.0f%% of 350)  [fixed submit/poll %.0f µs]",
		inter, hidden, gSlope, gemvGBs, gemvGBs/350*100, gFixed)

	// ---- (b) per-dispatch launch+barrier floor: 1-workgroup dependent no-op ----
	noopWGSL := `
@group(0) @binding(0) var<storage, read_write> d: array<f32>;
@compute @workgroup_size(64)
fn main() { d[0] = d[0] + 1.0; }`
	nsh, err := ctx.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{Label: "noop", WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: noopWGSL}})
	if err != nil {
		t.Fatal(err)
	}
	defer nsh.Release()
	npl, err := ctx.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{Label: "noop", Compute: wgpu.ProgrammableStageDescriptor{Module: nsh, EntryPoint: "main"}})
	if err != nil {
		t.Fatal(err)
	}
	defer npl.Release()
	nbuf, _ := ctx.device.CreateBuffer(&wgpu.BufferDescriptor{Size: 64, Usage: wgpu.BufferUsageStorage})
	defer nbuf.Release()
	nbg, _ := ctx.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: npl.GetBindGroupLayout(0), Entries: []wgpu.BindGroupEntry{{Binding: 0, Buffer: nbuf, Size: nbuf.GetSize()}}})
	defer nbg.Release()
	// every dispatch reads+writes d[0] → WAW/RAW hazard → backend must barrier
	// between them: this is the launch+barrier floor a dependent chain pays.
	nSlope, nFixed := perDispatch(npl, nbg, 1, 1, 1, 1000)
	t.Logf("(b) no-op 1-wg dependent (same bg)  %.2f µs/dispatch (launch+barrier floor)  [fixed %.0f µs]", nSlope, nFixed)

	// ---- (b2) distinct-buffer RAW chain: the real pass's structure ----
	// The real token is a deep dependency chain where dispatch i reads what i-1
	// wrote, each with its OWN bind group over its OWN buffers — unlike (b) which
	// reuses one bind group. If this is far above (b), the cost is serial
	// dependent-dispatch latency (the GPU can't overlap and pays full launch→
	// signal latency per step), and the lever is cutting dispatch COUNT (§2/§3).
	chainWGSL := `
@group(0) @binding(0) var<storage, read>       src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) { dst[lid.x] = src[lid.x] + 1.0; }`
	csh, err := ctx.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{Label: "chain", WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: chainWGSL}})
	if err != nil {
		t.Fatal(err)
	}
	defer csh.Release()
	cpl, err := ctx.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{Label: "chain", Compute: wgpu.ProgrammableStageDescriptor{Module: csh, EntryPoint: "main"}})
	if err != nil {
		t.Fatal(err)
	}
	defer cpl.Release()
	const chainN = 600 // ≥ the ~535 real dispatches
	cbufs := make([]*wgpu.Buffer, chainN+1)
	for i := range cbufs {
		cbufs[i], _ = ctx.device.CreateBuffer(&wgpu.BufferDescriptor{Size: 256, Usage: wgpu.BufferUsageStorage})
		defer cbufs[i].Release()
	}
	cbgs := make([]*wgpu.BindGroup, chainN)
	for i := range cbgs {
		cbgs[i], _ = ctx.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: cpl.GetBindGroupLayout(0), Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: cbufs[i], Size: cbufs[i].GetSize()},
			{Binding: 1, Buffer: cbufs[i+1], Size: cbufs[i+1].GetSize()},
		}})
		defer cbgs[i].Release()
	}
	timeChain := func(K, reps int) time.Duration {
		best := time.Hour
		for r := 0; r < reps; r++ {
			enc, _ := ctx.device.CreateCommandEncoder(nil)
			pass := enc.BeginComputePass(nil)
			pass.SetPipeline(cpl)
			for i := 0; i < K; i++ {
				pass.SetBindGroup(0, cbgs[i], nil) // distinct bg → distinct buffers → real RAW chain
				pass.DispatchWorkgroups(1, 1, 1)
			}
			pass.End()
			pass.Release()
			cmd, _ := enc.Finish(nil)
			t0 := time.Now()
			ctx.queue.Submit(cmd)
			ctx.device.Poll(true, nil)
			d := time.Since(t0)
			cmd.Release()
			enc.Release()
			if d < best {
				best = d
			}
		}
		return best
	}
	clo, chi := timeChain(50, 20), timeChain(550, 20)
	cSlope := float64((chi - clo).Microseconds()) / float64(550-50)
	t.Logf("(b2) distinct-buffer RAW chain  %.1f µs/dispatch  (50→%.0fµs, 550→%.0fµs)  ← the real pass's per-dispatch cost",
		cSlope, float64(clo.Microseconds()), float64(chi.Microseconds()))

	// ---- (c) reconstruct the 44.7 ms token from the measured pieces ----
	// Per-token dispatch census (DecodeRunner, 28 layers):
	//   gemv: q,k,v,o,gate,up,down = 7/layer ×28 + 1 lm_head        = 197
	//   glue: 2 rmsnorm + 3 quant + rope + ropeStore + vStore + attn
	//         + swiglu + 2 residual = 12/layer ×28 + final norm+quant = 338
	const nGemv, nGlue = 197, 338
	// gemv dispatches stream weights of many sizes; scale the measured 13.76 MB
	// gemv cost by total weight bytes so size mix is accounted for, not dispatch
	// count alone. glue dispatches touch only a few KB → ≈ the no-op floor.
	totalWeightBytes := 1.55e9
	matmulMs := totalWeightBytes / (gemvGBs * 1e9) * 1e3
	glueMs := float64(nGlue) * nSlope / 1e3
	gemvFloorMs := float64(nGemv) * nSlope / 1e3 // the launch+barrier part of the 197 gemvs
	predicted := matmulMs + glueMs + gFixed/1e3
	t.Logf("(c) reconstruct token (measured 44.7 ms):")
	t.Logf("    matmul stream  %5.1f ms  (%d gemv, %.0f GB/s over 1.55 GB)", matmulMs, nGemv, gemvGBs)
	t.Logf("    glue dispatch  %5.1f ms  (%d glue × %.1f µs floor)", glueMs, nGlue, nSlope)
	t.Logf("    gemv dispatch  %5.1f ms  (%d gemv × %.1f µs floor, already inside matmul-stream)", gemvFloorMs, nGemv, nSlope)
	t.Logf("    fixed sync     %5.1f ms", gFixed/1e3)
	t.Logf("    → predicted    %5.1f ms   (matmul + glue + fixed)", predicted)
	t.Logf("    → all-dispatch floor %5.1f ms  ((%d+%d) × %.1f µs)", float64(nGemv+nGlue)*nSlope/1e3, nGemv, nGlue, nSlope)
}
