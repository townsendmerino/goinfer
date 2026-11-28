//go:build gpu

package kernels

import (
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"

	"github.com/townsendmerino/goinfer/backend/webgpu/internal/wgsl"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"

	"github.com/oliverbestmann/webgpu/wgpu"
)

// attnPipes holds the Context's RoPE, KV-store, single-query, wide-head, and batched causal attention pipelines.
type attnPipes struct {
	ropeShader              *wgpu.ShaderModule
	ropePipeline            *wgpu.ComputePipeline
	ropeLayout              *wgpu.BindGroupLayout
	ropeBatchedShader       *wgpu.ShaderModule
	ropeBatchedPipeline     *wgpu.ComputePipeline
	ropeBatchedLayout       *wgpu.BindGroupLayout
	ropeMRoPERows           tiledPipes
	pairwise                *pairwisePipes
	attnBatchedShader       *wgpu.ShaderModule
	attnBatchedPipeline     *wgpu.ComputePipeline
	attnBatchedLayout       *wgpu.BindGroupLayout
	attnKeysBatchedShader   *wgpu.ShaderModule
	attnKeysBatchedPipeline *wgpu.ComputePipeline
	attnKeysBatchedLayout   *wgpu.BindGroupLayout
	attnBatchedF16Shader    *wgpu.ShaderModule
	attnBatchedF16Pipeline  *wgpu.ComputePipeline
	attnBatchedF16Layout    *wgpu.BindGroupLayout
	qkvFinShader            *wgpu.ShaderModule
	qkvFinPipeline          *wgpu.ComputePipeline
	qkvFinLayout            *wgpu.BindGroupLayout
	attnWideShader          *wgpu.ShaderModule
	attnWidePipeline        *wgpu.ComputePipeline
	attnWideLayout          *wgpu.BindGroupLayout
	attnF16WideShader       *wgpu.ShaderModule
	attnF16WidePipeline     *wgpu.ComputePipeline
	attnF16WideLayout       *wgpu.BindGroupLayout
	attnI8WideShader        *wgpu.ShaderModule
	attnI8WidePipeline      *wgpu.ComputePipeline
	attnI8WideLayout        *wgpu.BindGroupLayout
	attnShader              *wgpu.ShaderModule
	attnPipeline            *wgpu.ComputePipeline
	attnLayout              *wgpu.BindGroupLayout
	attnKeysShader          *wgpu.ShaderModule
	attnKeysPipeline        *wgpu.ComputePipeline
	attnKeysLayout          *wgpu.BindGroupLayout
	attnKeysWide            tiledPipes
	attnKeysRows            [2]tiledPipes
	ropeStoreShader         *wgpu.ShaderModule
	ropeStorePipeline       *wgpu.ComputePipeline
	ropeStoreLayout         *wgpu.BindGroupLayout
	kvStoreShader           *wgpu.ShaderModule
	kvStorePipeline         *wgpu.ComputePipeline
	kvStoreLayout           *wgpu.BindGroupLayout
	attnF16Shader           *wgpu.ShaderModule
	attnF16Pipeline         *wgpu.ComputePipeline
	attnF16Layout           *wgpu.BindGroupLayout
	ropeStoreF16Shader      *wgpu.ShaderModule
	ropeStoreF16Pipeline    *wgpu.ComputePipeline
	ropeStoreF16Layout      *wgpu.BindGroupLayout
	kvStoreF16Shader        *wgpu.ShaderModule
	kvStoreF16Pipeline      *wgpu.ComputePipeline
	kvStoreF16Layout        *wgpu.BindGroupLayout
	attnI8Shader            *wgpu.ShaderModule
	attnI8Pipeline          *wgpu.ComputePipeline
	attnI8Layout            *wgpu.BindGroupLayout
	ropeStoreI8Shader       *wgpu.ShaderModule
	ropeStoreI8Pipeline     *wgpu.ComputePipeline
	ropeStoreI8Layout       *wgpu.BindGroupLayout
	kvStoreI8Shader         *wgpu.ShaderModule
	kvStoreI8Pipeline       *wgpu.ComputePipeline
	kvStoreI8Layout         *wgpu.BindGroupLayout
}

// Stage 3 attention primitives: RoPE and single-query attention on the GPU, so a decode token's full forward records into
// one command buffer (no CPU interleave). Both match the CPU (decoder.applyRoPE / attendQuery) to f32 tolerance (the CPU
// accumulates in f64, the GPU in f32): cosine ~1.0, not bit-exact.
//
// Every `theta = f32(pos) * invFreq[d]` below feeds WGSL's sin/cos an angle that grows with position, and how a GPU
// range-reduces a large argument is implementation-defined (vendor/driver-specific), unlike the CPU's f64. At a long
// context's far end this is a real, measured error: TestRoPE_parityAtLongContextCeiling records it at pos=65535 rather than
// asserting a bar. Whether it matters for model quality is a product decision this file does not make, and the measurement
// does not carry over to Metal. Figures: docs/code-notes/gpu.md#Stage 3 attention primitives: long-context RoPE.

// attnWG is the single-query attention kernels' NARROW workgroup width, attnWGWide the wide one. The kernels put one lane on
// each head dim, so the workgroup width IS the largest head_dim they can dot: above it the tail dims go un-dotted and the
// o-projection consumes half-zero context, which is plausible-looking WRONG output with no error.
//
// The wide kernel (attnWideTemplateWGSL) gives each lane a STRIDE of dims, so head_dim is no longer a reason to decline. The
// narrow kernels still serve head_dim <= attnWG: striding costs a per-lane array and two extra loops that ordinary models
// should not pay.
//
// attnWG MUST stay equal to the @workgroup_size(128) and `red: array<f32, 128>` in the three narrow single-query kernels
// below. History: docs/code-notes/gpu.md#attnWG.
const (
	AttnWG         = 128 // shipped narrow kernels: one lane per dim
	attnWGWide     = 256 // wide kernel workgroup — WebGPU's guaranteed max invocations
	attnMaxPerLane = 8   // dims each wide lane strides over
	// attnMaxHeadDim is the widest head_dim the wide kernel serves.
	attnMaxHeadDim = attnWGWide * attnMaxPerLane
)

// attnKeysTile is the key-tile width compiled into attnKeysShaderWGSL's sc[] array. Kept
// beside the shader because the two MUST agree: a larger TILE in the WGSL without a larger
// sc[] writes out of bounds.
const attnKeysTile = 2048

// AttnKeysDisabled force-disables the key-split kernel (GOINFER_ATTN_KEYS=0), so the old dim-split kernel can be A/B'd in
// the same binary. Read once: the plan is recorded per runner, and a mid-run flip would leave a half-converted plan.
var AttnKeysDisabled = os.Getenv("GOINFER_ATTN_KEYS") == "0"

// attnKeysEligible reports whether the key-split attention kernel can serve this geometry: f32 KV only (the f16/int8 caches
// have their own packed kernels), hd within the narrow kernel, and hd and kvDim multiples of 4 so the vec4 K/q loads are in
// bounds and aligned.
func attnKeysEligible(hd, kvDim int, kvF16, kvI8 bool) bool {
	if kvF16 || kvI8 || hd > AttnWG || hd <= 0 {
		return false
	}
	return hd%4 == 0 && kvDim%4 == 0
}

// attnKernel picks the non-int8-KV attention pipeline+layout for a geometry. Every caller that dispatches attention against an
// f32 or f16 KV cache must go through it: two copies of this choice once made PrefillLastW8A8 and the sequential
// DecodeRunner dispatch DIFFERENT kernels for one geometry, and the two kernels are not bit-identical. It does not cover
// kvI8, which binds a different (9-argument) bind group shape; callers that can see kvI8 handle it separately, as
// decoderunner.go does. History: docs/code-notes/gpu.md#attnKernel.
func (c *Context) attnKernel(hd, kvDim int, kvF16 bool) (*wgpu.ComputePipeline, *wgpu.BindGroupLayout) {
	wide := hd > AttnWG
	switch {
	case kvF16 && wide:
		return c.attnF16WidePipeline, c.attnF16WideLayout
	case kvF16:
		return c.attnF16Pipeline, c.attnF16Layout
	case wide && !AttnKeysDisabled && attnKeysWideEligible(hd, kvDim, kvF16):
		// Key-split attention for a head of up to 256: the per-dimension kernel below pays a cross-lane reduction for
		// every key, which is most of a long prompt's time on a wide head.
		return c.attnKeysWide.Pl, c.attnKeysWide.Layout
	case wide:
		return c.attnWidePipeline, c.attnWideLayout
	case !AttnKeysDisabled && attnKeysEligible(hd, kvDim, kvF16, false):
		// Key-split attention (one reduction per tile, not per key). Last case on purpose: the f16/wide paths above have their own
		// kernels, so this only claims the plain f32 narrow geometry.
		return c.attnKeysPipeline, c.attnKeysLayout
	default:
		return c.attnPipeline, c.attnLayout
	}
}

// --- f16-KV variants (opt-in precision knob; see task-gpu-f16-kv.md) ---
// The KV cache is array<u32>, two f16 per word, packed/read with the core WGSL
// builtins pack2x16float / unpack2x16float — NO shader-f16 device feature, so the
// CI software adapter still compiles them. The query and the ctx output stay f32;
// only the resident cache is f16. Logical element e of a cache lives at word e>>1,
// half e&1; base (pos*kvDim) is even (kvDim even), so a token's words align and
// each store thread owns one whole word (writes both halves) — no atomics, no race.

// --- int8-KV variants (opt-in; see task-gpu-kv-i8.md) ---
// Cache is array<u32>, 4 int8/word; a side buffer holds one f32 scale per
// (position, KV-head) at [pos*nKV + head]. WRITE kernels run one thread per KV
// head: rotate (K) / read (V) the head's headDim elements, reduce absmax → scale
// (maxabs/127, matching the CPU QuantizeRowInt8), quantize, pack 4/word. The
// query + ctx stay f32; only the resident cache is int8. base (pos*kvDim) and
// headDim are multiples of 4, so a head's elements pack into whole words.

func (c *Context) ensureAttn() error {
	// Guard on the LAST pipeline built, not the first: these are created in order, so a non-nil last field means every earlier
	// one succeeded too. Guarding on ropePipeline (the first) let a mid-build failure leave the guard satisfied with later
	// pipelines nil, and the next call dispatched a nil pipeline. On a retry after a partial build the earlier fields are
	// rebuilt (the old ones stay tracked for release at Close).
	if c.kvStoreI8Pipeline != nil {
		return nil
	}
	// Shared tracked constructor (gpu.go): registers shader+pipeline for release.
	mk := c.mkPipeline
	var err error
	if c.ropeShader, c.ropePipeline, c.ropeLayout, err = mk("rope", wgsl.RopeShaderWGSL); err != nil {
		return err
	}
	if c.attnShader, c.attnPipeline, c.attnLayout, err = mk("attn", wgsl.AttnShaderWGSL); err != nil {
		return err
	}
	// Built here (not lazily at first use) so a shader-compile failure surfaces at ensureAttn
	// alongside every other attention pipeline, rather than mid-decode. Cheap: one compile.
	if c.attnKeysShader, c.attnKeysPipeline, c.attnKeysLayout, err = mk("attn-keys", wgsl.AttnKeysShaderWGSL); err != nil {
		return err
	}
	if c.qkvFinShader, c.qkvFinPipeline, c.qkvFinLayout, err = mk("qkvFinalize", wgsl.QkvFinalizeShaderWGSL); err != nil {
		return err
	}
	if c.ropeStoreShader, c.ropeStorePipeline, c.ropeStoreLayout, err = mk("rope-store", wgsl.RopeStoreShaderWGSL); err != nil {
		return err
	}
	if c.kvStoreShader, c.kvStorePipeline, c.kvStoreLayout, err = mk("kv-store", wgsl.KvStoreShaderWGSL); err != nil {
		return err
	}
	if c.attnF16Shader, c.attnF16Pipeline, c.attnF16Layout, err = mk("attn-f16", wgsl.AttnF16ShaderWGSL); err != nil {
		return err
	}
	if c.ropeStoreF16Shader, c.ropeStoreF16Pipeline, c.ropeStoreF16Layout, err = mk("rope-store-f16", wgsl.RopeStoreF16ShaderWGSL); err != nil {
		return err
	}
	if c.kvStoreF16Shader, c.kvStoreF16Pipeline, c.kvStoreF16Layout, err = mk("kv-store-f16", wgsl.KvStoreF16ShaderWGSL); err != nil {
		return err
	}
	attnI8Code, attnI8Label := wgsl.AttnI8ShaderWGSL, "attn-i8"
	if c.hasDP4A {
		attnI8Code, attnI8Label = wgsl.AttnI8DP4AShaderWGSL, "attn-i8-dp4a"
	}
	if c.attnI8Shader, c.attnI8Pipeline, c.attnI8Layout, err = mk(attnI8Label, attnI8Code); err != nil {
		return err
	}
	if c.ropeStoreI8Shader, c.ropeStoreI8Pipeline, c.ropeStoreI8Layout, err = mk("rope-store-i8", wgsl.RopeStoreI8ShaderWGSL); err != nil {
		return err
	}
	if c.kvStoreI8Shader, c.kvStoreI8Pipeline, c.kvStoreI8Layout, err = mk("kv-store-i8", wgsl.KvStoreI8ShaderWGSL); err != nil {
		return err
	}
	return nil
}

// RoPE applies rotary embeddings to a host q/k vector [heads*headDim] at position
// pos, returning the rotated vector. invFreq is the layer's [half] inverse freqs.
// (Host in/out — a standalone op for parity; the graph uses the device variant.)
func (c *Context) RoPE(vec []float32, heads, headDim, pos int, invFreq []float32, scale float32) ([]float32, error) {
	if err := c.ensureAttn(); err != nil {
		return nil, err
	}
	half := len(invFreq)
	vBuf, err := c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{Label: "rope-vec", Contents: wgpu.ToBytes(vec), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
	if err != nil {
		return nil, err
	}
	defer releaseBuf(vBuf)
	fBuf, err := c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{Label: "rope-invfreq", Contents: wgpu.ToBytes(invFreq), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return nil, err
	}
	defer releaseBuf(fBuf)
	pBuf, err := c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{Label: "rope-p", Contents: wgpu.ToBytes([]uint32{uint32(heads), uint32(headDim), uint32(half), uint32(pos), F32bits(scale), 0, 0, 0}), Usage: wgpu.BufferUsageUniform})
	if err != nil {
		return nil, err
	}
	defer releaseBuf(pBuf)
	bg, err := c.device.TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: c.ropeLayout, Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: vBuf, Size: vBuf.GetSize()},
		{Binding: 1, Buffer: fBuf, Size: fBuf.GetSize()},
		{Binding: 2, Buffer: pBuf, Size: pBuf.GetSize()},
	}})
	if err != nil {
		return nil, err
	}
	defer bg.Release()
	if err := c.submitUnary(c.ropePipeline, bg, heads*half); err != nil {
		return nil, err
	}
	return c.readbackRaw(vBuf, len(vec))
}

// Attention runs single-query attention on host inputs (q RoPE'd [nH*hd], keys/
// vals [nKeys*nKV*hd]) and returns ctx [nH*hd]. Standalone op for parity.
func (c *Context) Attention(q, keys, vals []float32, nH, nKV, hd, nKeys, start int, scale float32) ([]float32, error) {
	if err := c.ensureAttn(); err != nil {
		return nil, err
	}
	return c.attentionOn(c.attnPipeline, c.attnLayout, q, keys, vals, nH, nKV, hd, nKeys, start, scale)
}

// attentionOn is Attention's body, parameterised by which attention pipeline to run. Split out
// so the key-split kernel is exercised through EXACTLY the same host path as the dim-split one —
// a parity test that supplied its own dispatch would be testing its own calling convention rather
// than the kernel the runner uses.
func (c *Context) attentionOn(pl *wgpu.ComputePipeline, ly *wgpu.BindGroupLayout, q, keys, vals []float32, nH, nKV, hd, nKeys, start int, scale float32) ([]float32, error) {
	group := nH / nKV
	qBuf, err := c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{Label: "attn-q", Contents: wgpu.ToBytes(q), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return nil, fmt.Errorf("gpu: attentionOn: attn-q: %w", err)
	}
	defer releaseBuf(qBuf)
	kBuf, err := c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{Label: "attn-k", Contents: wgpu.ToBytes(keys), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return nil, fmt.Errorf("gpu: attentionOn: attn-k: %w", err)
	}
	defer releaseBuf(kBuf)
	vBuf, err := c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{Label: "attn-v", Contents: wgpu.ToBytes(vals), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return nil, fmt.Errorf("gpu: attentionOn: attn-v: %w", err)
	}
	defer releaseBuf(vBuf)
	cBuf, err := c.tryCreateBuffer(mem.LineScratch, &wgpu.BufferDescriptor{Label: "attn-ctx", Size: uint64(nH * hd * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
	if err != nil {
		return nil, err
	}
	defer releaseBuf(cBuf)
	pBuf, err := c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{Label: "attn-p", Contents: wgpu.ToBytes([]uint32{uint32(nH), uint32(nKV), uint32(hd), uint32(nKeys), uint32(start), uint32(group), F32bits(scale), 0}), Usage: wgpu.BufferUsageUniform})
	if err != nil {
		return nil, fmt.Errorf("gpu: attentionOn: attn-p: %w", err)
	}
	defer releaseBuf(pBuf)
	// FeatAttnSink: always bound (WGSL bind groups can't bind a null storage buffer); this
	// test helper never carries a real sink, so a one-element dummy + hasSink=0, matching attnShaderWGSL's convention.
	sinksBuf, err := c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{Label: "attn-sinks", Contents: wgpu.ToBytes([]float32{0}), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return nil, fmt.Errorf("gpu: attentionOn: attn-sinks: %w", err)
	}
	defer releaseBuf(sinksBuf)
	hsBuf, err := c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{Label: "attn-hs", Contents: wgpu.ToBytes([]uint32{0, 0, 0, 0}), Usage: wgpu.BufferUsageUniform})
	if err != nil {
		return nil, fmt.Errorf("gpu: attentionOn: attn-hs: %w", err)
	}
	defer releaseBuf(hsBuf)
	bg, err := c.device.TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: ly, Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: qBuf, Size: qBuf.GetSize()},
		{Binding: 1, Buffer: kBuf, Size: kBuf.GetSize()},
		{Binding: 2, Buffer: vBuf, Size: vBuf.GetSize()},
		{Binding: 3, Buffer: cBuf, Size: cBuf.GetSize()},
		{Binding: 4, Buffer: sinksBuf, Size: sinksBuf.GetSize()},
		{Binding: 5, Buffer: pBuf, Size: pBuf.GetSize()},
		{Binding: 6, Buffer: hsBuf, Size: hsBuf.GetSize()},
	}})
	if err != nil {
		return nil, err
	}
	defer bg.Release()
	enc, err := c.device.TryCreateCommandEncoder(nil)
	if err != nil {
		return nil, fmt.Errorf("gpu: attentionOn: command encoder: %w", err)
	}
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pl)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(uint32(nH), 1, 1) // one workgroup per query head
	if err := pass.TryEnd(); err != nil {
		pass.Release()
		return nil, err
	}
	pass.Release()
	cmd, err := enc.TryFinish(nil)
	if err != nil {
		return nil, fmt.Errorf("gpu: attentionOn: finish: %w", err)
	}
	defer cmd.Release()
	c.queue.Submit(cmd)
	return c.readbackRaw(cBuf, nH*hd)
}

func F32bits(f float32) uint32 { return math.Float32bits(f) }

// attnWideVariant is the per-precision half of the template above: the K/V bindings and the two
// fetch expressions, in terms of `ki` (the logical element index) and `sci` (the per-(pos,head)
// scale index the int8 cache needs).
type attnWideVariant struct {
	name         string
	bindings     string
	helpers      string
	kRead, vRead string
}

var attnWideVariants = []attnWideVariant{
	{
		name: "attn-wide",
		bindings: `@group(0) @binding(1) var<storage, read>       keys:  array<f32>;
@group(0) @binding(2) var<storage, read>       vals:  array<f32>;
@group(0) @binding(3) var<storage, read_write> ctx:   array<f32>;
@group(0) @binding(4) var<storage, read>       sinks: array<f32>;
@group(0) @binding(5) var<uniform>             p:     P;
@group(0) @binding(6) var<uniform>             hs:    HS;`,
		kRead: "keys[ki]",
		vRead: "vals[ki]",
	},
	{
		name: "attn-f16-wide",
		bindings: `@group(0) @binding(1) var<storage, read>       keys:  array<u32>;
@group(0) @binding(2) var<storage, read>       vals:  array<u32>;
@group(0) @binding(3) var<storage, read_write> ctx:   array<f32>;
@group(0) @binding(4) var<storage, read>       sinks: array<f32>;
@group(0) @binding(5) var<uniform>             p:     P;
@group(0) @binding(6) var<uniform>             hs:    HS;`,
		helpers: `fn f16at(w: u32, e: u32) -> f32 {
    let pair = unpack2x16float(w);
    return select(pair.x, pair.y, (e & 1u) == 1u);
}`,
		kRead: "f16at(keys[ki >> 1u], ki)",
		vRead: "f16at(vals[ki >> 1u], ki)",
	},
	{
		name: "attn-i8-wide",
		bindings: `@group(0) @binding(1) var<storage, read>       keys:   array<u32>;
@group(0) @binding(2) var<storage, read>       vals:   array<u32>;
@group(0) @binding(3) var<storage, read>       kScale: array<f32>;
@group(0) @binding(4) var<storage, read>       vScale: array<f32>;
@group(0) @binding(5) var<storage, read_write> ctx:    array<f32>;
@group(0) @binding(6) var<storage, read>       sinks:  array<f32>;
@group(0) @binding(7) var<uniform>             p:      P;
@group(0) @binding(8) var<uniform>             hs:     HS;`,
		helpers: `fn unpacki8(w: u32, e: u32) -> f32 {
    let b = (e & 3u) * 8u;
    return f32(i32(w << (24u - b)) >> 24u);
}`,
		kRead: "unpacki8(keys[ki >> 2u], ki) * kScale[sci]",
		vRead: "unpacki8(vals[ki >> 2u], ki) * vScale[sci]",
	},
}

// buildWideAttnWGSL instantiates the template for one variant. It leaves no placeholder behind:
// an unsubstituted __TOKEN__ would be a WGSL compile error rather than a silently wrong kernel,
// but the check is here anyway so the failure names the token instead of a parser column.
func buildWideAttnWGSL(v attnWideVariant) (string, error) {
	src := strings.NewReplacer(
		"__BINDINGS__", v.bindings,
		"__HELPERS__", v.helpers,
		"__KREAD__", v.kRead,
		"__VREAD__", v.vRead,
		"__MAXPER__", strconv.Itoa(attnMaxPerLane),
		"__HALFWG__", strconv.Itoa(attnWGWide/2),
		"__WG__", strconv.Itoa(attnWGWide),
	).Replace(wgsl.AttnWideTemplateWGSL)
	if i := strings.Index(src, "__"); i >= 0 {
		return "", fmt.Errorf("gpu: %s wide attention shader has an unsubstituted placeholder near %q",
			v.name, src[i:min(i+24, len(src))])
	}
	return src, nil
}

// ensureAttnWide compiles the 256-lane variants. Separate from ensureAttn and called only when a plan has head_dim > attnWG,
// so every other family pays no extra compiles and does not depend on a device limit it does not need.
func (c *Context) ensureAttnWide() error {
	// Guard on the LAST pipeline, not the first: this compiles THREE variants (f32, f16 KV, int8 KV), and returning early when
	// the first existed would let a failure on the second or third leave those nil while the next call reported success. The
	// three are all-or-nothing (any error returns before assigning the rest), so the last being non-nil implies all three are.
	if c.attnI8WidePipeline != nil {
		return nil
	}
	// maxComputeInvocationsPerWorkgroup is 256 in the WebGPU default limits, but "default" is a
	// floor for conformant implementations, not a promise from this adapter. Check it: a decline
	// here falls back to the staged path, whereas compiling past it is a device error mid-build.
	if lim := c.device.GetLimits().MaxComputeInvocationsPerWorkgroup; lim < attnWGWide {
		return fmt.Errorf("gpu: device maxComputeInvocationsPerWorkgroup=%d < %d — cannot run the "+
			"wide attention kernel needed for head_dim > %d", lim, attnWGWide, AttnWG)
	}
	if c.attnKeysWide.Pl == nil {
		src, err := attnKeysWideSource()
		if err != nil {
			return err
		}
		sh, pl, ly, err := c.mkPipeline("attn-keys-wide", src)
		if err != nil {
			return err
		}
		c.attnKeysWide = tiledPipes{sh: sh, Pl: pl, Layout: ly}
	}
	dst := [][3]any{
		{&c.attnWideShader, &c.attnWidePipeline, &c.attnWideLayout},
		{&c.attnF16WideShader, &c.attnF16WidePipeline, &c.attnF16WideLayout},
		{&c.attnI8WideShader, &c.attnI8WidePipeline, &c.attnI8WideLayout},
	}
	for i, v := range attnWideVariants {
		src, err := buildWideAttnWGSL(v)
		if err != nil {
			return err
		}
		sh, pl, ly, err := c.mkPipeline(v.name, src)
		if err != nil {
			return err
		}
		*dst[i][0].(**wgpu.ShaderModule) = sh
		*dst[i][1].(**wgpu.ComputePipeline) = pl
		*dst[i][2].(**wgpu.BindGroupLayout) = ly
	}
	return nil
}
