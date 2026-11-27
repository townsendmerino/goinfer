package decoder

import (
	"context"
	"encoding/json"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/decoder/quant"
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
	"github.com/townsendmerino/goinfer/internal/runtime/wbuild"
)

type (
	rowReader       = wbuild.RowReader
	streamTestHooks = wbuild.StreamTestHooks
)

func withStreamTestHooks(ctx context.Context, h streamTestHooks) context.Context {
	return wbuild.WithStreamTestHooks(ctx, h)
}

func wmBytes(w *linalg.WeightMat) int64 { return wbuild.WMBytes(w) }

func matmulQuant(base quantMode, name string) quantMode { return wbuild.MatmulQuant(base, name) }

func forRowRanges(rows, stride, minChunk, workers int, newReader func() rowReader, work func(lo, hi int, read rowReader) error) error {
	return wbuild.ForRowRanges(rows, stride, minChunk, workers, newReader, work)
}

func parseQuantConfig(raw json.RawMessage) (*quant.Config, error) { return quant.ParseConfig(raw) }

func loadFusedExperts(st *embed.SafetensorsFile, gateUpName, downName string, nExpert, inter, hidden int, quant quantMode, needCanonical, skipRow4 bool, sink *giwWriter) ([]archparts.ExpertWeights, error) {
	var sw *wbuild.StreamWriter
	if sink != nil {
		sw = &sink.StreamWriter
	}
	h := wbuildHooks()
	return wbuild.LoadFusedExperts(st, gateUpName, downName, nExpert, inter, hidden, quant, needCanonical, skipRow4, sw, &h)
}

func fusedLayout(name string, shape []int, nExpert, a, b int) (bool, error) {
	h := wbuildHooks()
	return wbuild.FusedLayout(name, shape, nExpert, a, b, &h)
}

func forExperts(n int, sink *giwWriter, one func(e int) error) error {
	var sw *wbuild.StreamWriter
	if sink != nil {
		sw = &sink.StreamWriter
	}
	return wbuild.ForExperts(n, sw, one)
}

func streamLayers(w *darch.Weights, arch *darch.Architecture, sink *giwWriter, load func(i int) error) error {
	return wbuild.StreamLayers(w, arch, &sink.StreamWriter, load)
}

func parallelLayers(n int, abort <-chan struct{}, fn func(i int) error) error {
	return wbuild.ParallelLayers(n, abort, fn)
}

func wantsRow4Fallback(backendName string) bool { return wbuild.WantsRow4Fallback(backendName) }

func isBatchedProjTensor(name string) bool { return wbuild.IsBatchedProjTensor(name) }

func repackForDecode(sink *giwWriter) bool {
	if sink == nil {
		return true
	}
	return wbuild.RepackForDecode(&sink.StreamWriter)
}

func quantizeWM(w linalg.WeightMat, mode quantMode) linalg.WeightMat {
	h := wbuildHooks()
	return h.QuantizeWM(w, mode)
}

func quantizeEmbedWM(w linalg.WeightMat, mode quantMode, needCanonical bool) linalg.WeightMat {
	h := wbuildHooks()
	return h.QuantizeEmbedWM(w, mode, needCanonical)
}

func quantizeBatchedProjWM(w linalg.WeightMat, mode quantMode, needCanonical, skipRow4 bool) linalg.WeightMat {
	h := wbuildHooks()
	return h.QuantizeBatchedProjWM(w, mode, needCanonical, skipRow4)
}

func quantizeLayerWM(w linalg.WeightMat, mode quantMode, needCanonical, skipRow4 bool) linalg.WeightMat {
	h := wbuildHooks()
	return h.QuantizeLayerWM(w, mode, needCanonical, skipRow4)
}

func streamQuantized(rows, cols int, mode quantMode, rowInto func(r int, dst []float32) error) (linalg.WeightMat, error) {
	h := wbuildHooks()
	return h.StreamQuantized(rows, cols, mode, rowInto)
}

func streamQuantizedEmbed(rows, cols int, mode quantMode, needCanonical, skipRow4 bool, stride, workers int, newReader func() rowReader) (linalg.WeightMat, error) {
	h := wbuildHooks()
	return h.StreamQuantizedEmbed(rows, cols, mode, needCanonical, skipRow4, stride, workers, newReader)
}

func fakeQuantInt4(scheme string, w []float32, rows, cols, group int) []float32 {
	return wbuild.FakeQuantInt4(scheme, w, rows, cols, group)
}

func reorderVHeads(src []float32, lead, outer, inner, hd, trail int) []float32 {
	return wbuild.ReorderVHeads(src, lead, outer, inner, hd, trail)
}

func untileVHeads(src []float32, lead, numKHeads, numVPerK, hd, trail int) []float32 {
	return wbuild.UntileVHeads(src, lead, numKHeads, numVPerK, hd, trail)
}

func ggufQKPermuted(archName string) bool { return wbuild.GGUFQKPermuted(archName) }

func buildWeightsFromGGUF(cfg *Config, arch *darch.Architecture, g *embed.GGUFFile, quant quantMode, embedInt4, needCanonical, skipRow4 bool, sink *giwWriter, id string, abort <-chan struct{}) (*darch.Weights, error) {
	var sw *wbuild.StreamWriter
	if sink != nil {
		sw = &sink.StreamWriter
	}
	return wbuild.BuildWeightsFromGGUF(cfg, arch, g, quant, embedInt4, needCanonical, skipRow4, sw, id, abort, wbuildHooks())
}

func tensorName(layer int, suffix string) string { return wbuild.TensorName(layer, suffix) }

func ggufInvPermute(w []float32, out, in, nHead int) []float32 {
	return wbuild.GGUFInvPermute(w, out, in, nHead)
}
