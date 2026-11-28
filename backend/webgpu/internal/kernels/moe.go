//go:build gpu

package kernels

import (
	"fmt"

	"github.com/townsendmerino/goinfer/backend/webgpu/internal/wgsl"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"

	"github.com/oliverbestmann/webgpu/wgpu"
)

// MoE residency: sparse expert routing + dispatch on the GPU so the MoE families (Mixtral / Qwen2-MoE / GLM / DeepSeek) run
// on the resident DecodeRunner instead of the staged path. This file holds the router top-k SELECTION kernel (moeRoute), the
// indexed sparse-expert GEMV, the gated shared-expert combine and the gpt-oss variants.
//
// The selection mirrors the CPU routeExperts (decoder/mlp.go): score the router logits (softmax for Mixtral/Qwen2-MoE, or
// per-expert sigmoid for DeepSeek/GLM), optionally add a per-expert selection bias, take the top-k by selection score, set
// each chosen expert's WEIGHT to its un-biased score, optionally renormalize the k weights to sum 1 (Mixtral
// norm_topk_prob) and scale them (DeepSeek routed_scaling_factor). The group-limited variant (DeepSeek nGroup>1) mirrors
// decoder.groupLimit. nE is tiny (8–512) so one single-lane workgroup is plenty: selection is not the cost, the expert GEMVs
// are.
// moePipes holds the Context's MoE router, sparse-expert GEMV, Gemma 4 MoE, gpt-oss MoE, and shared-gate pipelines.
type moePipes struct {
	moeRouteShader                *wgpu.ShaderModule
	moeRoutePipeline              *wgpu.ComputePipeline
	moeRouteLayout                *wgpu.BindGroupLayout
	g4RouteScalePipeline          *wgpu.ComputePipeline
	g4RouteScaleLayout            *wgpu.BindGroupLayout
	g4TailCopyPipeline            *wgpu.ComputePipeline
	g4TailCopyLayout              *wgpu.BindGroupLayout
	moeExpertShader               *wgpu.ShaderModule
	moeExpertPipeline             *wgpu.ComputePipeline
	moeExpertLayout               *wgpu.BindGroupLayout
	moeExpertW4Shader             *wgpu.ShaderModule
	moeExpertW4Pipeline           *wgpu.ComputePipeline
	moeExpertW4Layout             *wgpu.BindGroupLayout
	routeGptOssShader             *wgpu.ShaderModule
	routeGptOssPipeline           *wgpu.ComputePipeline
	routeGptOssLayout             *wgpu.BindGroupLayout
	gptossGluQuantShader          *wgpu.ShaderModule
	gptossGluQuantPipeline        *wgpu.ComputePipeline
	gptossGluQuantLayout          *wgpu.BindGroupLayout
	moeExpertGptOssDownShader     *wgpu.ShaderModule
	moeExpertGptOssDownPipeline   *wgpu.ComputePipeline
	moeExpertGptOssDownLayout     *wgpu.BindGroupLayout
	moeExpertGptOssDownW4Shader   *wgpu.ShaderModule
	moeExpertGptOssDownW4Pipeline *wgpu.ComputePipeline
	moeExpertGptOssDownW4Layout   *wgpu.BindGroupLayout
	sharedGateShader              *wgpu.ShaderModule
	sharedGatePipeline            *wgpu.ComputePipeline
	sharedGateLayout              *wgpu.BindGroupLayout
}

func (c *Context) ensureSharedGate() error {
	if c.sharedGatePipeline != nil {
		return nil
	}
	sh, err := c.device.TryCreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "sharedGate", WGSLSource: &wgpu.ShaderSourceWGSL{Code: wgsl.SharedGateWGSL},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile sharedGate: %w", err)
	}
	pl, err := c.device.TryCreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "sharedGate", Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return fmt.Errorf("gpu: pipeline sharedGate: %w", err)
	}
	c.track(sh.Release, pl.Release) // registered at creation, so Close releases them
	c.sharedGateShader, c.sharedGatePipeline, c.sharedGateLayout = sh, pl, c.bgl(pl)
	return nil
}

// ResidentStackedW8A8 holds nE experts' W8A8 weight for ONE projection (gate, up, or
// down) packed back-to-back in the gemvW8A8 layout, indexable by expert: row (e*N+n).
type ResidentStackedW8A8 struct {
	ctx         *Context // see ResidentW4A8.ctx
	bq, bScales *wgpu.Buffer
	nE, rows    int
	cols, kp    int
	// w4 ⇒ this stack is int4 (W4A8): bq holds packed nibbles, bScales holds f16
	// group scales (kp = padK32), and moeExpert dispatches the int4 kernel.
	w4 bool
}

// Close frees the stacked buffers.
func (s *ResidentStackedW8A8) Close() error {
	s.ctx.releaseOwned(func() {
		traceFree(s)
		if s.bq != nil {
			accountFree(int64(s.bq.GetSize()))
			releaseBuf(s.bq)
			s.bq = nil // a second Close must not release again
		}
		if s.bScales != nil {
			accountFree(int64(s.bScales.GetSize()))
			releaseBuf(s.bScales)
			s.bScales = nil
		}
	})
	return nil
}

// UploadStackedExperts packs nE experts' int8 weights [each N,K] + per-row scales [each N]
// into one resident buffer (expert e's row n at packed-row e*N+n), the layout the indexed
// expert GEMV reads. q8[e] / scales[e] are expert e's quantized projection.
func (c *Context) UploadStackedExperts(q8 [][]int8, scales [][]float32, nE, N, K int) (*ResidentStackedW8A8, error) {
	if nE <= 0 || N <= 0 || K <= 0 || len(q8) < nE || len(scales) < nE {
		return nil, fmt.Errorf("gpu: UploadStackedExperts bad dims nE=%d N=%d K=%d", nE, N, K)
	}
	kp := padK(K)
	words := kp / 4
	packed := make([]uint32, nE*N*words)
	allScales := make([]float32, nE*N)
	for e := range nE {
		if len(q8[e]) < N*K || len(scales[e]) < N {
			return nil, fmt.Errorf("gpu: UploadStackedExperts expert %d too small", e)
		}
		copy(packed[e*N*words:(e+1)*N*words], packInt8(q8[e], N, K))
		copy(allScales[e*N:(e+1)*N], scales[e][:N])
	}
	bq, err := c.tryCreateBufferInit(mem.LineExperts, &wgpu.BufferInitDescriptor{Label: "moe-experts", Contents: wgpu.ToBytes(packed), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return nil, fmt.Errorf("gpu: stacked experts buffer: %w", err)
	}
	sc, err := c.tryCreateBufferInit(mem.LineExperts, &wgpu.BufferInitDescriptor{Label: "moe-expert-scales", Contents: wgpu.ToBytes(allScales), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		releaseBuf(bq)
		return nil, fmt.Errorf("gpu: stacked expert scales buffer: %w", err)
	}
	accountAlloc(int64(bq.GetSize()) + int64(sc.GetSize()))
	return traced(&ResidentStackedW8A8{ctx: c, bq: bq, bScales: sc, nE: nE, rows: N, cols: K, kp: kp}, int64(bq.GetSize())+int64(sc.GetSize())), nil
}

func (c *Context) ensureMoEExpert() error {
	if c.moeExpertPipeline != nil {
		return nil
	}
	sh, err := c.device.TryCreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "moeExpertGEMV", WGSLSource: &wgpu.ShaderSourceWGSL{Code: wgsl.MoeExpertGEMVWGSL},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile moeExpertGEMV: %w", err)
	}
	pl, err := c.device.TryCreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "moeExpertGEMV", Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return fmt.Errorf("gpu: pipeline moeExpertGEMV: %w", err)
	}
	c.track(sh.Release, pl.Release) // registered at creation, so Close releases them
	c.moeExpertShader, c.moeExpertPipeline, c.moeExpertLayout = sh, pl, c.bgl(pl)
	return nil
}

func (c *Context) ensureMoERoute() error {
	if c.moeRoutePipeline != nil {
		return nil
	}
	sh, err := c.device.TryCreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "moeRoute", WGSLSource: &wgpu.ShaderSourceWGSL{Code: wgsl.MoeRouteWGSL},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile moeRoute: %w", err)
	}
	pl, err := c.device.TryCreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "moeRoute", Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return fmt.Errorf("gpu: pipeline moeRoute: %w", err)
	}
	c.track(sh.Release, pl.Release) // registered at creation, so Close releases them
	c.moeRouteShader, c.moeRoutePipeline, c.moeRouteLayout = sh, pl, c.bgl(pl)
	return nil
}

// gpt-oss's three MoE kernels — own router, own
// clamped-gated activation, own biased down-combine — each ensure-once, mirroring
// ensureMoERoute/ensureMoEExpert exactly.
func (c *Context) ensureRouteGptOss() error {
	if c.routeGptOssPipeline != nil {
		return nil
	}
	sh, err := c.device.TryCreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "routeGptOss", WGSLSource: &wgpu.ShaderSourceWGSL{Code: wgsl.RouteGptOssWGSL},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile routeGptOss: %w", err)
	}
	pl, err := c.device.TryCreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "routeGptOss", Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return fmt.Errorf("gpu: pipeline routeGptOss: %w", err)
	}
	c.track(sh.Release, pl.Release)
	c.routeGptOssShader, c.routeGptOssPipeline, c.routeGptOssLayout = sh, pl, c.bgl(pl)
	return nil
}

func (c *Context) ensureGptOssGluQuant() error {
	if c.gptossGluQuantPipeline != nil {
		return nil
	}
	code, err := c.quantSrc("gptossGluQuant", wgsl.GptossGluQuantWGSL)
	if err != nil {
		return err
	}
	sh, err := c.device.TryCreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "gptossGluQuant", WGSLSource: &wgpu.ShaderSourceWGSL{Code: code},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile gptossGluQuant: %w", err)
	}
	pl, err := c.device.TryCreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "gptossGluQuant", Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return fmt.Errorf("gpu: pipeline gptossGluQuant: %w", err)
	}
	c.track(sh.Release, pl.Release)
	c.gptossGluQuantShader, c.gptossGluQuantPipeline, c.gptossGluQuantLayout = sh, pl, c.bgl(pl)
	return nil
}

func (c *Context) ensureMoEExpertGptOssDown() error {
	if c.moeExpertGptOssDownPipeline != nil {
		return nil
	}
	sh, err := c.device.TryCreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "moeExpertGptOssDownGEMV", WGSLSource: &wgpu.ShaderSourceWGSL{Code: wgsl.MoeExpertGptOssDownGEMVWGSL},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile moeExpertGptOssDownGEMV: %w", err)
	}
	pl, err := c.device.TryCreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "moeExpertGptOssDownGEMV", Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return fmt.Errorf("gpu: pipeline moeExpertGptOssDownGEMV: %w", err)
	}
	c.track(sh.Release, pl.Release)
	c.moeExpertGptOssDownShader, c.moeExpertGptOssDownPipeline, c.moeExpertGptOssDownLayout = sh, pl, c.bgl(pl)
	return nil
}

// gptOssDownPipelineFor picks gpt-oss's down-projection combine kernel for a stacked expert set. It is the one place that
// choice is made, shared by the resident builder and GptOssDownForTest, so the test exercises the builder's own selection.
func (c *Context) gptOssDownPipelineFor(s *ResidentStackedW8A8) (*wgpu.ComputePipeline, *wgpu.BindGroupLayout) {
	if s.w4 { // int4 stack: nibbles + f16 group scales, which the int8 kernel misreads
		return c.moeExpertGptOssDownW4Pipeline, c.moeExpertGptOssDownW4Layout
	}
	return c.moeExpertGptOssDownPipeline, c.moeExpertGptOssDownLayout
}
