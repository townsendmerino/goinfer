//go:build gpu

package kernels

import (
	"fmt"

	"github.com/townsendmerino/goinfer/backend/webgpu/internal/wgsl"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"

	"github.com/oliverbestmann/webgpu/wgpu"
)

// ensureMoEExpertGptOssDownW4 compiles moeExpertGptOssDownGEMVW4WGSL.
func (c *Context) ensureMoEExpertGptOssDownW4() error {
	if c.moeExpertGptOssDownW4Pipeline != nil {
		return nil
	}
	sh, err := c.device.TryCreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "moeExpertGptOssDownGEMVW4", WGSLSource: &wgpu.ShaderSourceWGSL{Code: wgsl.MoeExpertGptOssDownGEMVW4WGSL},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile moeExpertGptOssDownGEMVW4: %w", err)
	}
	pl, err := c.device.TryCreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "moeExpertGptOssDownGEMVW4", Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return fmt.Errorf("gpu: pipeline moeExpertGptOssDownGEMVW4: %w", err)
	}
	c.track(sh.Release, pl.Release)
	c.moeExpertGptOssDownW4Shader, c.moeExpertGptOssDownW4Pipeline, c.moeExpertGptOssDownW4Layout = sh, pl, c.bgl(pl)
	return nil
}

func (c *Context) ensureMoEExpertW4() error {
	if c.moeExpertW4Pipeline != nil {
		return nil
	}
	sh, err := c.device.TryCreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "moeExpertGEMVW4", WGSLSource: &wgpu.ShaderSourceWGSL{Code: wgsl.MoeExpertGEMVW4WGSL},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile moeExpertGEMVW4: %w", err)
	}
	pl, err := c.device.TryCreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "moeExpertGEMVW4", Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return fmt.Errorf("gpu: pipeline moeExpertGEMVW4: %w", err)
	}
	c.track(sh.Release, pl.Release) // registered at creation, so Close releases them
	c.moeExpertW4Shader, c.moeExpertW4Pipeline, c.moeExpertW4Layout = sh, pl, c.bgl(pl)
	return nil
}

// UploadStackedExpertsInt4 packs nE experts' int4 group-wise weights (each [N,K]
// as nibbles 0..15 = value+8, plus per-group f32 scales [N*nGroups], group=32) into
// one resident buffer (expert e's row n at packed-row e*N+n), the layout the indexed
// W4A8 expert GEMV reads. The returned stacked buffer carries w4=true so moeExpert
// dispatches the int4 kernel. K must be a multiple of 32 (the W4A8 group).
func (c *Context) UploadStackedExpertsInt4(nib [][]uint8, scales [][]float32, nE, N, K int) (*ResidentStackedW8A8, error) {
	if nE <= 0 || N <= 0 || K <= 0 || len(nib) < nE || len(scales) < nE {
		return nil, fmt.Errorf("gpu: UploadStackedExpertsInt4 bad dims nE=%d N=%d K=%d", nE, N, K)
	}
	kp := PadK32(K)
	ng := kp / W4a8GroupSize // groups per row
	wpr := kp / 8            // u32 words per row (nibbles, 8/u32)
	packed := make([]uint32, nE*N*wpr)
	allScales := make([]float32, nE*N*ng)
	for e := range nE {
		if len(nib[e]) < N*K || len(scales[e]) < N*ng {
			return nil, fmt.Errorf("gpu: UploadStackedExpertsInt4 expert %d too small (nib %d<%d, sc %d<%d)", e, len(nib[e]), N*K, len(scales[e]), N*ng)
		}
		copy(packed[e*N*wpr:(e+1)*N*wpr], packNibbles(nib[e], N, K))
		copy(allScales[e*N*ng:(e+1)*N*ng], scales[e][:N*ng])
	}
	bq, err := c.tryCreateBufferInit(mem.LineExperts, &wgpu.BufferInitDescriptor{Label: "moe-experts-w4", Contents: wgpu.ToBytes(packed), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return nil, fmt.Errorf("gpu: stacked w4 experts buffer: %w", err)
	}
	sc, err := c.tryCreateBufferInit(mem.LineExperts, &wgpu.BufferInitDescriptor{Label: "moe-expert-w4-scales", Contents: wgpu.ToBytes(PackF16Pairs(allScales)), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		releaseBuf(bq)
		return nil, fmt.Errorf("gpu: stacked w4 scales buffer: %w", err)
	}
	accountAlloc(int64(bq.GetSize()) + int64(sc.GetSize()))
	return traced(&ResidentStackedW8A8{ctx: c, bq: bq, bScales: sc, nE: nE, rows: N, cols: K, kp: kp, w4: true}, int64(bq.GetSize())+int64(sc.GetSize())), nil
}

// UploadStackedExpertsInt4Packed is the fast path of UploadStackedExpertsInt4: each
// expert's q4 is ALREADY the GPU packed layout (decoder int4 == packNibbles when K%32==0,
// TestInt4LayoutMatch), so it just concatenates the per-expert byte slices (one memcpy each,
// ~10× a nibble unpack+repack) and CreateBufferInits — no per-element shuffle. q4[e] is
// expert e's decoder int4 bytes (≥ N*K/2); scales[e] its per-group f32 (≥ N*K/32).
func (c *Context) UploadStackedExpertsInt4Packed(q4 [][]byte, scales [][]float32, nE, N, K int) (*ResidentStackedW8A8, error) {
	if nE <= 0 || N <= 0 || K <= 0 || len(q4) < nE || len(scales) < nE {
		return nil, fmt.Errorf("gpu: UploadStackedExpertsInt4Packed bad dims nE=%d N=%d K=%d", nE, N, K)
	}
	if K%W4a8GroupSize != 0 {
		return nil, fmt.Errorf("gpu: UploadStackedExpertsInt4Packed needs K%%%d==0, got K=%d", W4a8GroupSize, K)
	}
	kp := K
	ng := kp / W4a8GroupSize
	bpe := N * kp / 2 // bytes per expert
	packed := make([]byte, nE*bpe)
	allScales := make([]float32, nE*N*ng)
	for e := range nE {
		if len(q4[e]) < bpe || len(scales[e]) < N*ng {
			return nil, fmt.Errorf("gpu: UploadStackedExpertsInt4Packed expert %d too small (q4 %d<%d, sc %d<%d)", e, len(q4[e]), bpe, len(scales[e]), N*ng)
		}
		copy(packed[e*bpe:(e+1)*bpe], q4[e][:bpe])
		copy(allScales[e*N*ng:(e+1)*N*ng], scales[e][:N*ng])
	}
	bq, err := c.tryCreateBufferInit(mem.LineExperts, &wgpu.BufferInitDescriptor{Label: "moe-experts-w4", Contents: packed, Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return nil, fmt.Errorf("gpu: stacked w4 experts buffer: %w", err)
	}
	sc, err := c.tryCreateBufferInit(mem.LineExperts, &wgpu.BufferInitDescriptor{Label: "moe-expert-w4-scales", Contents: wgpu.ToBytes(PackF16Pairs(allScales)), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		releaseBuf(bq)
		return nil, fmt.Errorf("gpu: stacked w4 scales buffer: %w", err)
	}
	accountAlloc(int64(bq.GetSize()) + int64(sc.GetSize()))
	return traced(&ResidentStackedW8A8{ctx: c, bq: bq, bScales: sc, nE: nE, rows: N, cols: K, kp: kp, w4: true}, int64(bq.GetSize())+int64(sc.GetSize())), nil
}
