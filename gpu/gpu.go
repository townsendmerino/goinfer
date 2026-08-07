//go:build gpu

package gpu

import (
	"fmt"

	"github.com/cogentcore/webgpu/wgpu"
)

// matmulShaderWGSL computes dst[m,n] = Σ_k a[m,k]·b[n,k], i.e. dst =
// a·bᵀ — the encoder's matmulBT contract: a is [M,K] row-major, b is
// [N,K] row-major (the PyTorch [out,in] weight layout, so no transpose
// is needed), dst is [M,N] row-major.
//
// One invocation per output element, 16×16 workgroup. This is the
// NAIVE kernel — no shared-memory tiling, every invocation streams a
// full a-row and b-row from global memory. It is correct and proves
// the whole upload/dispatch/readback pipeline; a tiled kernel that
// stages K-strips into workgroup memory is the throughput follow-up.
const matmulShaderWGSL = `
struct Dims { m: u32, k: u32, n: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       a:    array<f32>;
@group(0) @binding(1) var<storage, read>       b:    array<f32>;
@group(0) @binding(2) var<storage, read_write> dst:  array<f32>;
@group(0) @binding(3) var<uniform>             dims: Dims;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;
    if (row >= dims.m || col >= dims.n) {
        return;
    }
    let K = dims.k;
    let aBase = row * K;
    let bBase = col * K;
    var acc: f32 = 0.0;
    for (var i: u32 = 0u; i < K; i = i + 1u) {
        acc = acc + a[aBase + i] * b[bBase + i];
    }
    dst[row * dims.n + col] = acc;
}
`

// Context holds the persistent WebGPU objects (device, queue, compiled
// pipeline). Create one with New and reuse it across MatmulBT calls —
// device/pipeline creation is expensive, per-call buffer allocation is
// not. Not safe for concurrent use by multiple goroutines; wrap in your
// own mutex or use one Context per worker if you need that.
type Context struct {
	instance *wgpu.Instance
	adapter  *wgpu.Adapter
	device   *wgpu.Device
	queue    *wgpu.Queue
	shader   *wgpu.ShaderModule
	pipeline *wgpu.ComputePipeline
	layout   *wgpu.BindGroupLayout

	// releases holds one closure per lazily-created device object (shader module + compute
	// pipeline), registered AT CREATION by mkPipeline/track and drained LIFO by Close (audit C-26).
	//
	// It replaces the hand-maintained per-field release list Close used to carry, which had drifted
	// to 14 of ~40 pipelines — every ensure* added since simply leaked, and ensureVision's shader
	// modules were dropped on the floor entirely (never stored, so unreleasable at any later point).
	// A hand list cannot stay correct: it is edited in a different file from the code that allocates.
	// Registering at the allocation site makes the default behaviour correct for pipelines that do
	// not exist yet.
	releases []func()
	// closed makes Close IDEMPOTENT. `defer m.Close()` alongside an explicit m.Close() is the
	// ordinary Go shape, and decoder.Model.Close calls m.be.Close() unconditionally — so a second
	// Close used to double-release the wgpu handles, a use-after-free inside the native layer. The
	// cpu and cuda backends were already idempotent (cuda guards on r.reqCh == nil); only WebGPU
	// crashed, and only on a machine with a real GPU.
	closed bool

	// W8A8 (int8×int8) pipeline, compiled lazily by ensureQuant (quant.go).
	quantShader   *wgpu.ShaderModule
	quantPipeline *wgpu.ComputePipeline
	quantLayout   *wgpu.BindGroupLayout

	// On-GPU activation int8 quantize pipeline, lazy via ensureQuantize (device.go).
	quantizeShader   *wgpu.ShaderModule
	quantizePipeline *wgpu.ComputePipeline
	quantizeLayout   *wgpu.BindGroupLayout

	// Coalesced W8A8 GEMV (decode) pipeline, lazy via ensureGEMV (gemv.go).
	gemvShader   *wgpu.ShaderModule
	gemvPipeline *wgpu.ComputePipeline
	gemvLayout   *wgpu.BindGroupLayout
	// W8A8 GEMV with a per-output bias folded into the epilogue (ensureGEMVBias):
	// dst[n] = scale·acc + bias[n], for the q/k/v projections of bias models (Qwen2).
	gemvBiasShader   *wgpu.ShaderModule
	gemvBiasPipeline *wgpu.ComputePipeline
	gemvBiasLayout   *wgpu.BindGroupLayout

	// Tiled W8A8 GEMM (prefill) pipeline, lazy via ensureTiled (gemm.go).
	tiledShader   *wgpu.ShaderModule
	tiledPipeline *wgpu.ComputePipeline
	tiledLayout   *wgpu.BindGroupLayout

	// Thin-M (multi-row GEMV) W8A8 GEMM for the Stage-B verify (gemm_rows.go): one
	// workgroup per output column, each weight word read once and reused across all
	// M rows. Beats the 16×16 tiled GEMM at the small M (≈K+1) of a verify block.
	gemmRowShader   *wgpu.ShaderModule
	gemmRowPipeline *wgpu.ComputePipeline
	gemmRowLayout   *wgpu.BindGroupLayout

	// Elementwise/norm pipelines for the fused MLP, lazy via ensureLayer (layer.go).
	rmsnormShader    *wgpu.ShaderModule
	rmsnormPipeline  *wgpu.ComputePipeline
	rmsnormLayout    *wgpu.BindGroupLayout
	swigluShader     *wgpu.ShaderModule
	swigluPipeline   *wgpu.ComputePipeline
	swigluLayout     *wgpu.BindGroupLayout
	residualShader   *wgpu.ShaderModule
	residualPipeline *wgpu.ComputePipeline
	residualLayout   *wgpu.BindGroupLayout

	// Resident vision-encoder pipelines, lazy via ensureVision (vision.go).
	lnRowsPipeline   *wgpu.ComputePipeline
	lnRowsLayout     *wgpu.BindGroupLayout
	geluPipeline     *wgpu.ComputePipeline
	geluLayout       *wgpu.BindGroupLayout
	softmaxPipeline  *wgpu.ComputePipeline
	softmaxLayout    *wgpu.BindGroupLayout
	addRowsPipeline  *wgpu.ComputePipeline
	addRowsLayout    *wgpu.BindGroupLayout
	copyHeadPipeline *wgpu.ComputePipeline
	copyHeadLayout   *wgpu.BindGroupLayout

	// RoPE + single-query attention pipelines, lazy via ensureAttn (attention.go).
	ropeShader   *wgpu.ShaderModule
	ropePipeline *wgpu.ComputePipeline
	ropeLayout   *wgpu.BindGroupLayout
	// Fused q-rope + k-rope-store + v-store (decode fusion, f32 KV): one dispatch for the
	// three post-projection KV ops, cutting two dispatches/layer off the decode chain.
	qkvFinShader   *wgpu.ShaderModule
	qkvFinPipeline *wgpu.ComputePipeline
	qkvFinLayout   *wgpu.BindGroupLayout
	attnShader     *wgpu.ShaderModule
	attnPipeline   *wgpu.ComputePipeline
	attnLayout     *wgpu.BindGroupLayout

	// KV-cache writers (ensureAttn): rope-and-store K into KCache, store V into
	// VCache — both at pos*kvDim via a per-token uniform base, so the decode token
	// is a single compute pass with no CopyBufferToBuffer to break it.
	ropeStoreShader   *wgpu.ShaderModule
	ropeStorePipeline *wgpu.ComputePipeline
	ropeStoreLayout   *wgpu.BindGroupLayout
	kvStoreShader     *wgpu.ShaderModule
	kvStorePipeline   *wgpu.ComputePipeline
	kvStoreLayout     *wgpu.BindGroupLayout

	// f16-KV variants of the three above (ensureAttn): the cache is array<u32>
	// (2 f16/word, packed/read via core pack2x16float/unpack2x16float — no
	// shader-f16 feature). Opt-in precision knob; the f32 path above is untouched
	// and stays bit-exact. See task-gpu-f16-kv.md.
	attnF16Shader        *wgpu.ShaderModule
	attnF16Pipeline      *wgpu.ComputePipeline
	attnF16Layout        *wgpu.BindGroupLayout
	ropeStoreF16Shader   *wgpu.ShaderModule
	ropeStoreF16Pipeline *wgpu.ComputePipeline
	ropeStoreF16Layout   *wgpu.BindGroupLayout
	kvStoreF16Shader     *wgpu.ShaderModule
	kvStoreF16Pipeline   *wgpu.ComputePipeline
	kvStoreF16Layout     *wgpu.BindGroupLayout

	// int8-KV variants (ensureAttn): the cache is array<u32> (4 int8/word) + a
	// per-(position,KV-head) f32 scale side buffer. WRITE kernels (ropeStoreI8,
	// kvStoreI8) reduce per-head absmax → scale → quantize (one thread per KV
	// head); the READ kernel (attnI8) unpacks int8 and multiplies by qd·scale in
	// f32. 4× vs f32 / 2× vs f16. Opt-in; f32 + f16 paths untouched. task-gpu-kv-i8.md.
	attnI8Shader        *wgpu.ShaderModule
	attnI8Pipeline      *wgpu.ComputePipeline
	attnI8Layout        *wgpu.BindGroupLayout
	ropeStoreI8Shader   *wgpu.ShaderModule
	ropeStoreI8Pipeline *wgpu.ComputePipeline
	ropeStoreI8Layout   *wgpu.BindGroupLayout
	kvStoreI8Shader     *wgpu.ShaderModule
	kvStoreI8Pipeline   *wgpu.ComputePipeline
	kvStoreI8Layout     *wgpu.BindGroupLayout

	// §2 fused glue kernels (ensureFuse, decodefuse.go): fold quantize into its
	// producer to shorten the serialized decode dependency chain.
	rmsQuantShader      *wgpu.ShaderModule
	rmsQuantPipeline    *wgpu.ComputePipeline
	rmsQuantLayout      *wgpu.BindGroupLayout
	swigluQuantShader   *wgpu.ShaderModule
	swigluQuantPipeline *wgpu.ComputePipeline
	swigluQuantLayout   *wgpu.BindGroupLayout

	// Per-head QK-norm (Lever C, qknorm.go): in-place RMSNorm of each q/k head over
	// headDim before RoPE (Qwen3 / GLM / Mellum). One workgroup per head.
	qkNormShader   *wgpu.ShaderModule
	qkNormPipeline *wgpu.ComputePipeline
	qkNormLayout   *wgpu.BindGroupLayout

	// Resident Mamba-2 SSM decode (mamba.go): the hybrid mixer as a bounded per-token
	// recurrence — mambaConv (causal-conv ring), mambaSSM (selective state update),
	// mambaGatedNorm (gated grouped RMSNorm). Granite-4.0-H / Nemotron-H.
	mambaSSMShader     *wgpu.ShaderModule
	mambaSSMPipeline   *wgpu.ComputePipeline
	mambaSSMLayout     *wgpu.BindGroupLayout
	mambaConvShader    *wgpu.ShaderModule
	mambaConvPipeline  *wgpu.ComputePipeline
	mambaConvLayout    *wgpu.BindGroupLayout
	mambaGNormShader   *wgpu.ShaderModule
	mambaGNormPipeline *wgpu.ComputePipeline
	mambaGNormLayout   *wgpu.BindGroupLayout
	// f16 Mamba in/out_proj GEMV (mamba_f16.go): f16 weight × f32 activation — the
	// mixed-precision quality fix (granite int8 loss localized to the SSM projections).
	mambaF16Shader   *wgpu.ShaderModule
	mambaF16Pipeline *wgpu.ComputePipeline
	mambaF16Layout   *wgpu.BindGroupLayout
	// W8A16 GEMVs (gemv_w8a16.go): int8 weight × f32 activation — the granite-resident
	// activation-precision fix (stops the int8 re-quant cascade across the deep stack).
	gemvW8A16Shader        *wgpu.ShaderModule
	gemvW8A16Pipeline      *wgpu.ComputePipeline
	gemvW8A16Layout        *wgpu.BindGroupLayout
	moeExpertW8A16Shader   *wgpu.ShaderModule
	moeExpertW8A16Pipeline *wgpu.ComputePipeline
	moeExpertW8A16Layout   *wgpu.BindGroupLayout
	// Nemotron-H squared-ReLU FFN (relu2.go): fused relu²(up)→int8 for the non-gated MLP block.
	relu2Shader   *wgpu.ShaderModule
	relu2Pipeline *wgpu.ComputePipeline
	relu2Layout   *wgpu.BindGroupLayout

	// MoE residency (Lever C3, moe.go): router top-k selection (moeRoute) feeding the
	// indexed sparse-expert GEMV — Mixtral / Qwen2-MoE / GLM / DeepSeek on the resident path.
	moeRouteShader    *wgpu.ShaderModule
	moeRoutePipeline  *wgpu.ComputePipeline
	moeRouteLayout    *wgpu.BindGroupLayout
	moeExpertShader   *wgpu.ShaderModule
	moeExpertPipeline *wgpu.ComputePipeline
	moeExpertLayout   *wgpu.BindGroupLayout
	// Indexed stacked-expert GEMV at int4 (W4A8) — the int4 twin of moeExpert*.
	moeExpertW4Shader   *wgpu.ShaderModule
	moeExpertW4Pipeline *wgpu.ComputePipeline
	moeExpertW4Layout   *wgpu.BindGroupLayout
	// Gated shared-expert combine (Lever C3d, qwen2_moe): xd[n] += sigmoid(gl[0])·src[n].
	sharedGateShader   *wgpu.ShaderModule
	sharedGatePipeline *wgpu.ComputePipeline
	sharedGateLayout   *wgpu.BindGroupLayout

	// W4A8 decode GEMV (ensureGEMVW4, gemv_w4a8.go): int4 group-wise weights.
	gemvW4Shader   *wgpu.ShaderModule
	gemvW4Pipeline *wgpu.ComputePipeline
	gemvW4Layout   *wgpu.BindGroupLayout

	// MLA absorb-path attention (Lever C4, mla.go): rank-space single-query attention
	// over the compressed latent cache (DeepSeek/Kimi). wsum[h] = Σ_j softmax_j·cn_j.
	mlaAttnShader   *wgpu.ShaderModule
	mlaAttnPipeline *wgpu.ComputePipeline
	mlaAttnLayout   *wgpu.BindGroupLayout
	// MLA latent store (Lever C4b): kvA-norm the rank latent + decoupled-RoPE the key,
	// written into the latent cache at the token's position (the append).
	mlaStoreShader   *wgpu.ShaderModule
	mlaStorePipeline *wgpu.ComputePipeline
	mlaStoreLayout   *wgpu.BindGroupLayout
	// MLA per-head f32 matvec (Lever C4b): the W_UK absorb (q_nope→rank) and W_UV lift
	// (rank→v) — block-diagonal per head, so a dedicated batched matvec, not a GEMV.
	mlaHeadMVShader   *wgpu.ShaderModule
	mlaHeadMVPipeline *wgpu.ComputePipeline
	mlaHeadMVLayout   *wgpu.BindGroupLayout
	// MLA query RoPE (Lever C4c): gather + decoupled-RoPE the qk_rope query slice into
	// the combined qAbs buffer, after the absorbed qNopeAbs.
	mlaQRopeShader   *wgpu.ShaderModule
	mlaQRopePipeline *wgpu.ComputePipeline
	mlaQRopeLayout   *wgpu.BindGroupLayout
}

// New initializes a GPU context: instance → adapter (high-performance
// preference) → device → compiled matmul pipeline. Returns an error if
// no adapter/device is available (e.g. a headless box with no GPU), so
// callers can fall back to the CPU path or skip GPU tests cleanly.
func New() (*Context, error) {
	// Quiet wgpu-native's benign warnings ("No windowing system present. Using surfaceless
	// platform", "No config found!") — pure noise on a headless/server box. Errors still log.
	wgpu.SetLogLevel(wgpu.LogLevelError)
	inst := wgpu.CreateInstance(nil)
	adapter, err := inst.RequestAdapter(&wgpu.RequestAdapterOptions{
		PowerPreference: wgpu.PowerPreferenceHighPerformance,
	})
	if err != nil {
		inst.Release()
		return nil, fmt.Errorf("gpu: request adapter: %w", err)
	}
	// Raise the storage-buffer binding limit: the DEFAULT device caps it at 128 MB,
	// smaller than a real model's LM head / embedding at int8 (e.g. 152k vocab ×
	// hidden ≈ 233 MB). Start from the valid default limit set and bump only the two
	// size limits to the adapter's max (requiring the adapter's full limit set
	// verbatim fails — some advertised limits, e.g. maxBufferSize, aren't valid as
	// required limits). maxBufferSize must be ≥ the binding size.
	lim := wgpu.DefaultLimits()
	al := adapter.GetLimits().Limits
	lim.MaxStorageBufferBindingSize = al.MaxStorageBufferBindingSize
	// Raise MaxBufferSize to the binding max (2 GB on this card) so large single
	// weights fit — a 7B's LM head is ~272 MB int4 / ~545 MB int8, past the 256 MB
	// WebGPU default. Set unconditionally: DefaultLimits() leaves MaxBufferSize at
	// the u64-max "unset" sentinel, so a `<` guard never fires and the device
	// silently keeps the 256 MB default (it must be a concrete value to take).
	lim.MaxBufferSize = al.MaxStorageBufferBindingSize
	// The default cap (65535) is below a vocab-sized GEMV (one workgroup per output
	// column → 152k for the LM head); raise it to the adapter's max.
	if al.MaxComputeWorkgroupsPerDimension > lim.MaxComputeWorkgroupsPerDimension {
		lim.MaxComputeWorkgroupsPerDimension = al.MaxComputeWorkgroupsPerDimension
	}
	device, err := adapter.RequestDevice(&wgpu.DeviceDescriptor{
		RequiredLimits: &wgpu.RequiredLimits{Limits: lim},
	})
	if err != nil {
		adapter.Release()
		inst.Release()
		return nil, fmt.Errorf("gpu: request device: %w", err)
	}
	shader, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "matmulBT",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: matmulShaderWGSL},
	})
	if err != nil {
		device.Release()
		adapter.Release()
		inst.Release()
		return nil, fmt.Errorf("gpu: compile shader: %w", err)
	}
	pipeline, err := device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:   "matmulBT",
		Compute: wgpu.ProgrammableStageDescriptor{Module: shader, EntryPoint: "main"},
		// Layout nil ⇒ auto layout inferred from the shader bindings.
	})
	if err != nil {
		shader.Release()
		device.Release()
		adapter.Release()
		inst.Release()
		return nil, fmt.Errorf("gpu: create pipeline: %w", err)
	}
	return &Context{
		instance: inst,
		adapter:  adapter,
		device:   device,
		queue:    device.GetQueue(),
		shader:   shader,
		pipeline: pipeline,
		layout:   pipeline.GetBindGroupLayout(0),
	}, nil
}

// Backend reports the underlying graphics backend ("Metal", "Vulkan",
// "D3D12", …) — useful for test logging to confirm which GPU API is in
// play.
func (c *Context) Backend() string {
	return c.adapter.GetInfo().BackendType.String()
}

// track registers release closures to be run by Close, LIFO. Call it at the ALLOCATION site —
// that is the whole point (see Context.releases): a teardown list maintained anywhere else drifts
// out of date the first time someone adds a pipeline without reading Close.
func (c *Context) track(fs ...func()) { c.releases = append(c.releases, fs...) }

// bgl captures a pipeline's group-0 bind-group layout AND registers its Release, so the ~20
// c.*Layout fields built outside mkPipeline don't leak a native BGL (each holding a device ref)
// per Context (audit R-17/C-26). GetBindGroupLayout returns a NEW object per call, so it must be
// captured once — never call it again for the same field.
func (c *Context) bgl(pl *wgpu.ComputePipeline) *wgpu.BindGroupLayout {
	lay := pl.GetBindGroupLayout(0)
	c.track(lay.Release)
	return lay
}

// mkPipeline compiles one WGSL shader into a compute pipeline, registers BOTH objects for release,
// and returns them plus the auto bind-group layout. It is the single tracked constructor the
// ensure* builders share; it was four byte-identical `mk` closures (attention.go, decodefuse.go,
// layer.go, vision.go), one of which — vision's — discarded its *wgpu.ShaderModule so it could
// never be released at all (audit C-26a).
//
// On pipeline-creation failure the shader is released immediately and NOTHING is registered, so a
// failed ensure* leaves the Context exactly as it found it.
func (c *Context) mkPipeline(label, code string) (*wgpu.ShaderModule, *wgpu.ComputePipeline, *wgpu.BindGroupLayout, error) {
	sh, err := c.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: label, WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: code},
	})
	if err != nil {
		return nil, nil, nil, fmt.Errorf("gpu: compile %s: %w", label, err)
	}
	pl, err := c.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: label, Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return nil, nil, nil, fmt.Errorf("gpu: pipeline %s: %w", label, err)
	}
	// Track the bind-group layout for release too: GetBindGroupLayout returns a new native BGL
	// object (each holding a device ref) that must be Released, and every caller stores it in a
	// c.*Layout field for the Context's lifetime — untracked, ~40 leaked per Context (audit R-17/C-26).
	lay := pl.GetBindGroupLayout(0)
	c.track(sh.Release, pl.Release, lay.Release)
	return sh, pl, lay, nil
}

// Close releases all GPU resources. Safe to call once; the Context must
// not be used afterward.
func (c *Context) Close() error {
	if c.closed {
		return nil // idempotent: `defer m.Close()` + an explicit Close must not double-release (C-26b)
	}
	c.closed = true
	// Drain every lazily-created pipeline/shader, newest first. Registered at the allocation site
	// (mkPipeline / track), so this stays complete as new ensure* builders are added — unlike the
	// hand-maintained field list this replaces, which covered 14 of ~40.
	for i := len(c.releases) - 1; i >= 0; i-- {
		c.releases[i]()
	}
	c.releases = nil
	// The base objects created by New, released last (pipelines depend on the device). Each is
	// nil-checked: New releases what it built and returns nil on a mid-construction failure, and
	// the nil-out below means a Context is only ever partially populated OR already drained.
	// Nil each handle after release so a use-after-free is a nil dereference at the Go boundary —
	// a stack trace pointing at the bug — rather than undefined behaviour inside the native layer.
	if c.layout != nil { // the primary pipeline's group-0 BGL (New's struct literal), never released (audit R-17)
		c.layout.Release()
	}
	if c.pipeline != nil {
		c.pipeline.Release()
	}
	if c.shader != nil {
		c.shader.Release()
	}
	if c.queue != nil {
		c.queue.Release()
	}
	if c.device != nil {
		c.device.Release()
	}
	if c.adapter != nil {
		c.adapter.Release()
	}
	if c.instance != nil {
		c.instance.Release()
	}
	c.layout, c.pipeline, c.shader, c.queue, c.device, c.adapter, c.instance = nil, nil, nil, nil, nil, nil, nil
	return nil
}

// ResidentMatrix is a weight matrix [rows, cols] uploaded to a GPU storage
// buffer once and reused across many MatmulBTResident calls — the fix for the
// per-call weight re-upload. The decoder uploads each constant weight matrix
// once at first use and keeps the handle for every subsequent token. Release
// frees the GPU buffer.
type ResidentMatrix struct {
	buf  *wgpu.Buffer
	rows int // N (out features)
	cols int // K (in features)
}

// Release frees the resident GPU buffer. Safe to call once.
func (rm *ResidentMatrix) Close() error {
	if rm.buf != nil {
		rm.buf.Release()
		rm.buf = nil
	}
	return nil
}

// UploadMatrix copies a [rows, cols] f32 matrix to a resident GPU storage
// buffer. b must hold ≥ rows*cols f32s. The returned ResidentMatrix is reused
// by MatmulBTResident; the caller owns it and must Release it.
func (c *Context) UploadMatrix(b []float32, rows, cols int) (*ResidentMatrix, error) {
	if rows <= 0 || cols <= 0 {
		return nil, fmt.Errorf("gpu: UploadMatrix non-positive dim rows=%d cols=%d", rows, cols)
	}
	if len(b) < rows*cols {
		return nil, fmt.Errorf("gpu: UploadMatrix input too small: len(b)=%d need %d", len(b), rows*cols)
	}
	buf, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "resident-b", Contents: wgpu.ToBytes(b[:rows*cols]), Usage: wgpu.BufferUsageStorage,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: create resident buffer: %w", err)
	}
	return &ResidentMatrix{buf: buf, rows: rows, cols: cols}, nil
}

// MatmulBTResident computes dst = a · rm.bᵀ ([M, rm.rows]), uploading only the
// (small) activation a and reading back the result — the weight rm stays
// resident. This removes the dominant transfer cost for repeated matmuls
// against a constant weight (the decoder's per-token projections + LM head).
func (c *Context) MatmulBTResident(a []float32, rm *ResidentMatrix, M int) ([]float32, error) {
	K, N := rm.cols, rm.rows
	if M <= 0 {
		return nil, fmt.Errorf("gpu: MatmulBTResident non-positive M=%d", M)
	}
	if len(a) < M*K {
		return nil, fmt.Errorf("gpu: MatmulBTResident input too small: len(a)=%d need %d", len(a), M*K)
	}
	aBuf, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "a", Contents: wgpu.ToBytes(a[:M*K]), Usage: wgpu.BufferUsageStorage,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: create a buffer: %w", err)
	}
	defer aBuf.Release()
	return c.run(aBuf, rm.buf, M, K, N)
}

// MatmulBT computes dst = a·bᵀ on the GPU and returns the [M,N] result.
// a must hold ≥ M*K f32s, b ≥ N*K. This allocates fresh GPU buffers,
// uploads a and b, dispatches the kernel, and reads the result back —
// so it pays full host↔device transfer on every call. Use UploadMatrix +
// MatmulBTResident when b is constant across calls (the decoder's weights).
func (c *Context) MatmulBT(a, b []float32, M, K, N int) ([]float32, error) {
	if M <= 0 || K <= 0 || N <= 0 {
		return nil, fmt.Errorf("gpu: matmulBT non-positive dim M=%d K=%d N=%d", M, K, N)
	}
	if len(a) < M*K || len(b) < N*K {
		return nil, fmt.Errorf("gpu: matmulBT input too small: len(a)=%d need %d, len(b)=%d need %d",
			len(a), M*K, len(b), N*K)
	}
	aBuf, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "a", Contents: wgpu.ToBytes(a[:M*K]), Usage: wgpu.BufferUsageStorage,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: create a buffer: %w", err)
	}
	defer aBuf.Release()

	bBuf, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "b", Contents: wgpu.ToBytes(b[:N*K]), Usage: wgpu.BufferUsageStorage,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: create b buffer: %w", err)
	}
	defer bBuf.Release()
	return c.run(aBuf, bBuf, M, K, N)
}

// run dispatches the matmul pipeline against already-created a and b buffers:
// allocate dst+dims+staging, bind, dispatch the M×N grid, copy dst→staging,
// submit, and block on the readback. Shared by MatmulBT (fresh b) and
// MatmulBTResident (resident b).
func (c *Context) run(aBuf, bBuf *wgpu.Buffer, M, K, N int) ([]float32, error) {
	dstSize := uint64(M * N * 4)

	dstBuf, err := c.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "dst", Size: dstSize,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: create dst buffer: %w", err)
	}
	defer dstBuf.Release()

	// Dims uniform: 3 u32 + 1 pad word (uniform buffers need 16-byte size).
	dims := []uint32{uint32(M), uint32(K), uint32(N), 0}
	dimsBuf, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "dims", Contents: wgpu.ToBytes(dims), Usage: wgpu.BufferUsageUniform,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: create dims buffer: %w", err)
	}
	defer dimsBuf.Release()

	stage, err := c.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "stage", Size: dstSize,
		Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: create staging buffer: %w", err)
	}
	defer stage.Release()

	bindGroup, err := c.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: c.layout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: aBuf, Size: aBuf.GetSize()},
			{Binding: 1, Buffer: bBuf, Size: bBuf.GetSize()},
			{Binding: 2, Buffer: dstBuf, Size: dstBuf.GetSize()},
			{Binding: 3, Buffer: dimsBuf, Size: dimsBuf.GetSize()},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: create bind group: %w", err)
	}
	defer bindGroup.Release()

	enc, err := c.device.CreateCommandEncoder(nil)
	if err != nil {
		return nil, fmt.Errorf("gpu: create command encoder: %w", err)
	}
	defer enc.Release()

	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(c.pipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	// global_invocation_id.x ranges over rows (M), .y over cols (N);
	// 16×16 threads per workgroup, so ceil-divide the counts.
	pass.DispatchWorkgroups((uint32(M)+15)/16, (uint32(N)+15)/16, 1)
	if err := pass.End(); err != nil {
		pass.Release()
		return nil, fmt.Errorf("gpu: end compute pass: %w", err)
	}
	pass.Release()

	if err := enc.CopyBufferToBuffer(dstBuf, 0, stage, 0, dstSize); err != nil {
		return nil, fmt.Errorf("gpu: copy dst→stage: %w", err)
	}
	cmd, err := enc.Finish(nil)
	if err != nil {
		return nil, fmt.Errorf("gpu: finish encoder: %w", err)
	}
	defer cmd.Release()
	c.queue.Submit(cmd)

	// Map the staging buffer and block until the GPU work + map complete.
	mapStatus := wgpu.BufferMapAsyncStatusUnknown
	if err := stage.MapAsync(wgpu.MapModeRead, 0, dstSize, func(s wgpu.BufferMapAsyncStatus) {
		mapStatus = s
	}); err != nil {
		return nil, fmt.Errorf("gpu: map async: %w", err)
	}
	c.device.Poll(true, nil) // wait=true: flush queue + fire map callback
	if mapStatus != wgpu.BufferMapAsyncStatusSuccess {
		return nil, fmt.Errorf("gpu: staging map failed: %v", mapStatus)
	}

	raw := stage.GetMappedRange(0, uint(dstSize))
	out := make([]float32, M*N)
	copy(out, wgpu.FromBytes[float32](raw))
	if err := stage.Unmap(); err != nil {
		return nil, fmt.Errorf("gpu: unmap staging: %w", err)
	}
	return out, nil
}
