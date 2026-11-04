//go:build gpu

package kernels

import (
	"github.com/oliverbestmann/webgpu/wgpu"
)

// Accessors for package gpu, the decoder glue beside this package: each reaches one unexported field or method of a
// public type, so gpu can use it without the name becoming public. A field's accessor returns a pointer to it.

func ContextAdapter(x *Context) **wgpu.Adapter { return &x.adapter }

func ContextDeltaGNormLayout(x *Context) **wgpu.BindGroupLayout { return &x.deltaGNormLayout }

func ContextDeltaGNormPipeline(x *Context) **wgpu.ComputePipeline { return &x.deltaGNormPipeline }

func ContextDeltaGatesLayout(x *Context) **wgpu.BindGroupLayout { return &x.deltaGatesLayout }

func ContextDeltaGatesPipeline(x *Context) **wgpu.ComputePipeline { return &x.deltaGatesPipeline }

func ContextDeltaNormLayout(x *Context) **wgpu.BindGroupLayout { return &x.deltaNormLayout }

func ContextDeltaNormPipeline(x *Context) **wgpu.ComputePipeline { return &x.deltaNormPipeline }

func ContextDeltaRuleLayout(x *Context) **wgpu.BindGroupLayout { return &x.deltaRuleLayout }

func ContextDeltaRulePipeline(x *Context) **wgpu.ComputePipeline { return &x.deltaRulePipeline }

func ContextDevice(x *Context) **wgpu.Device { return &x.device }

func ContextEnsureAttn(x *Context) error { return x.ensureAttn() }

func ContextEnsureAttnFunc(x *Context) func() error { return x.ensureAttn }

func ContextEnsureDeltaGNorm(x *Context) error { return x.ensureDeltaGNorm() }

func ContextEnsureDeltaGNormFunc(x *Context) func() error { return x.ensureDeltaGNorm }

func ContextEnsureDeltaGates(x *Context) error { return x.ensureDeltaGates() }

func ContextEnsureDeltaGatesFunc(x *Context) func() error { return x.ensureDeltaGates }

func ContextEnsureDeltaNorm(x *Context) error { return x.ensureDeltaNorm() }

func ContextEnsureDeltaNormFunc(x *Context) func() error { return x.ensureDeltaNorm }

func ContextEnsureDeltaRule(x *Context) error { return x.ensureDeltaRule() }

func ContextEnsureDeltaRuleFunc(x *Context) func() error { return x.ensureDeltaRule }

func ContextEnsureFuse(x *Context) error { return x.ensureFuse() }

func ContextEnsureFuseFunc(x *Context) func() error { return x.ensureFuse }

func ContextEnsureFusedRows(x *Context) error { return x.ensureFusedRows() }

func ContextEnsureFusedRowsFunc(x *Context) func() error { return x.ensureFusedRows }

func ContextEnsureGEMVBias(x *Context) error { return x.ensureGEMVBias() }

func ContextEnsureGEMVBiasFunc(x *Context) func() error { return x.ensureGEMVBias }

func ContextEnsureGptOssGluQuant(x *Context) error { return x.ensureGptOssGluQuant() }

func ContextEnsureGptOssGluQuantFunc(x *Context) func() error { return x.ensureGptOssGluQuant }

func ContextEnsureLayer(x *Context) error { return x.ensureLayer() }

func ContextEnsureLayerFunc(x *Context) func() error { return x.ensureLayer }

func ContextEnsureMRoPERows(x *Context) error { return x.ensureMRoPERows() }

func ContextEnsureMRoPERowsFunc(x *Context) func() error { return x.ensureMRoPERows }

func ContextEnsureMambaConv(x *Context) error { return x.ensureMambaConv() }

func ContextEnsureMambaConvFunc(x *Context) func() error { return x.ensureMambaConv }

func ContextEnsureMambaGNorm(x *Context) error { return x.ensureMambaGNorm() }

func ContextEnsureMambaGNormFunc(x *Context) func() error { return x.ensureMambaGNorm }

func ContextEnsureMambaSSM(x *Context) error { return x.ensureMambaSSM() }

func ContextEnsureMambaSSMFunc(x *Context) func() error { return x.ensureMambaSSM }

func ContextEnsurePrefillBatched(x *Context) error { return x.ensurePrefillBatched() }

func ContextEnsurePrefillBatchedFunc(x *Context) func() error { return x.ensurePrefillBatched }

func ContextEnsureQuantize(x *Context) error { return x.ensureQuantize() }

func ContextEnsureQuantizeFunc(x *Context) func() error { return x.ensureQuantize }

func ContextEnsureRelu2(x *Context) error { return x.ensureRelu2() }

func ContextEnsureRelu2Func(x *Context) func() error { return x.ensureRelu2 }

func ContextEnsureTiled(x *Context) error { return x.ensureTiled() }

func ContextEnsureTiledFunc(x *Context) func() error { return x.ensureTiled }

func ContextFusedRowsSource(x *Context, kind FusedRowsKind) (string, string, error) {
	return x.fusedRowsSource(kind)
}

func ContextGemvBiasLayout(x *Context) **wgpu.BindGroupLayout { return &x.gemvBiasLayout }

func ContextGemvBiasPipeline(x *Context) **wgpu.ComputePipeline { return &x.gemvBiasPipeline }

func ContextHasDP4A(x *Context) *bool { return &x.hasDP4A }

func ContextInfo(x *Context) *wgpu.AdapterInfo { return &x.info }

func ContextMambaConvLayout(x *Context) **wgpu.BindGroupLayout { return &x.mambaConvLayout }

func ContextMambaConvPipeline(x *Context) **wgpu.ComputePipeline { return &x.mambaConvPipeline }

func ContextMambaGNormLayout(x *Context) **wgpu.BindGroupLayout { return &x.mambaGNormLayout }

func ContextMambaGNormPipeline(x *Context) **wgpu.ComputePipeline { return &x.mambaGNormPipeline }

func ContextMambaSSMLayout(x *Context) **wgpu.BindGroupLayout { return &x.mambaSSMLayout }

func ContextMambaSSMPipeline(x *Context) **wgpu.ComputePipeline { return &x.mambaSSMPipeline }

func ContextMkPipeline(x *Context, label string, code string) (*wgpu.ShaderModule, *wgpu.ComputePipeline, *wgpu.BindGroupLayout, error) {
	return x.mkPipeline(label, code)
}

func ContextNewDecodeRunner(x *Context, m RunModel, hidden int, nH int, nKV int, hd int, inter int, start int, eps float32, scale float32, addOne bool) (*DecodeRunner, error) {
	return x.newDecodeRunner(m, hidden, nH, nKV, hd, inter, start, eps, scale, addOne)
}

func ContextPrefillLastRows(x *Context, xs [][]float32, m ModelW, pp prefillProjs, mrope [][3]int, hidden int, nH int, nKV int, hd int, inter int, positions []int, start int, eps float32, scale float32, addOne bool) ([]float32, error) {
	return x.prefillLastRows(xs, m, pp, mrope, hidden, nH, nKV, hd, inter, positions, start, eps, scale, addOne)
}

func ContextPrefillProf(x *Context) **PrefillProf { return &x.prefillProf }

func ContextQuantNarrow(x *Context) *bool { return &x.quantNarrow }

func ContextQueue(x *Context) **wgpu.Queue { return &x.queue }

func ContextReadbackRaw(x *Context, buf *wgpu.Buffer, n int) ([]float32, error) {
	return x.readbackRaw(buf, n)
}

func ContextReleaseOwned(x *Context, f func()) { x.releaseOwned(f) }

func ContextResidualLayout(x *Context) **wgpu.BindGroupLayout { return &x.residualLayout }

func ContextResidualPipeline(x *Context) **wgpu.ComputePipeline { return &x.residualPipeline }

func ContextRmsQuantLayout(x *Context) **wgpu.BindGroupLayout { return &x.rmsQuantLayout }

func ContextRmsQuantPipeline(x *Context) **wgpu.ComputePipeline { return &x.rmsQuantPipeline }

func ContextRopeLayout(x *Context) **wgpu.BindGroupLayout { return &x.ropeLayout }

func ContextRopeMRoPERows(x *Context) *tiledPipes { return &x.ropeMRoPERows }

func ContextRopePipeline(x *Context) **wgpu.ComputePipeline { return &x.ropePipeline }

func ContextTiledLayout(x *Context) **wgpu.BindGroupLayout { return &x.tiledLayout }

func ContextTiledPipeline(x *Context) **wgpu.ComputePipeline { return &x.tiledPipeline }

func ContextW4MultiPin(x *Context) *int { return &x.w4MultiPin }

func DecodeRunnerBaseSteps(x *DecodeRunner) *[]RunStep { return &x.baseSteps }

func DecodeRunnerC(x *DecodeRunner) **Context { return &x.c }

func DecodeRunnerDropPending(x *DecodeRunner) { x.dropPending() }

func DecodeRunnerLastLogits(x *DecodeRunner) **wgpu.Buffer { return &x.lastLogits }

func DecodeRunnerLoraLayers(x *DecodeRunner) *[]loraRunLayer { return &x.loraLayers }

func DecodeRunnerPending(x *DecodeRunner) **wgpu.CommandBuffer { return &x.pending }

func DecodeRunnerRecord(x *DecodeRunner, pass *wgpu.ComputePassEncoder) { x.record(pass) }

func DecodeRunnerStag(x *DecodeRunner) **wgpu.Buffer { return &x.stag }

func DecodeRunnerSteps(x *DecodeRunner) *[]RunStep { return &x.steps }

func DecodeRunnerVocab(x *DecodeRunner) *int { return &x.vocab }

func DecodeWeightNRows(x DecodeWeight) int { return x.nRows() }

func DeviceBufferBuf(x *DeviceBuffer) **wgpu.Buffer { return &x.buf }

func DeviceBufferN(x *DeviceBuffer) *int { return &x.n }

func ResidentStackedW8A8Rows(x *ResidentStackedW8A8) *int { return &x.rows }

func ResidentW8A8BScales(x *ResidentW8A8) **wgpu.Buffer { return &x.bScales }

func ResidentW8A8Bq(x *ResidentW8A8) **wgpu.Buffer { return &x.bq }

func ResidentW8A8Kp(x *ResidentW8A8) *int { return &x.kp }

func ResidentW8A8Rows(x *ResidentW8A8) *int { return &x.rows }

func DeviceBufferLit(buf *wgpu.Buffer, n int) *DeviceBuffer { return &DeviceBuffer{buf: buf, n: n} }
