package memprice

import (
	"slices"
	"unsafe"

	"github.com/townsendmerino/aikit/linalg"

	darch "github.com/townsendmerino/goinfer/decoder/arch"
)

// WMBytes returns one weight matrix's in-memory footprint from its backing slices.
func WMBytes(w *linalg.WeightMat) int64 {
	var n int64
	if q8, sc, _, ok := w.Int8(); ok {
		n += int64(len(q8)) + 4*int64(len(sc))
	}
	if q4, sc, _, ok := w.Int4F16(); ok {
		n += int64(len(q4)) + 2*int64(len(sc))
	}
	if p4, sc, ok := w.Int4Row4F16(); ok {
		n += int64(len(p4)) + 2*int64(len(sc))
	}
	if f, ok := w.F32(); ok {
		n += 4 * int64(len(f))
	}
	for _, get := range []func() ([]byte, bool){w.Q4K, w.Q5K, w.Q6K} {
		if raw, ok := get(); ok {
			n += int64(len(raw))
		}
	}
	if u, _, ok := w.Half16(); ok {
		n += 2 * int64(len(u))
	}
	n += int64(w.SplitHalfBytes())
	return n
}

// F32SliceBytes returns 4 * sum(len(v)) across plain float32 slices.
func F32SliceBytes(s ...[]float32) int64 {
	var n int64
	for _, v := range s {
		n += 4 * int64(len(v))
	}
	return n
}

// MmapAliasedBytes counts the quantized payload and scale bytes of w for which in reports true.
// When in is nil, it returns 0.
func MmapAliasedBytes(w *linalg.WeightMat, in func([]byte) bool) int64 {
	if in == nil {
		return 0
	}
	var n int64
	if q8, sc, _, ok := w.Int8(); ok && len(q8) > 0 {
		if in(unsafe.Slice((*byte)(unsafe.Pointer(&q8[0])), len(q8))) {
			n += int64(len(q8)) + 4*int64(len(sc))
		}
	}
	if q4, sc, _, ok := w.Int4F16(); ok && len(q4) > 0 {
		if in(q4) {
			n += int64(len(q4)) + 2*int64(len(sc))
		}
	}
	if p4, sc, ok := w.Int4Row4F16(); ok && len(p4) > 0 {
		if in(p4) {
			n += int64(len(p4)) + 2*int64(len(sc))
		}
	}
	for _, get := range []func() ([]byte, bool){w.Q4K, w.Q5K, w.Q6K} {
		if raw, ok := get(); ok && len(raw) > 0 && in(raw) {
			n += int64(len(raw))
		}
	}
	if u, _, ok := w.Half16(); ok && len(u) > 0 {
		if in(unsafe.Slice((*byte)(unsafe.Pointer(&u[0])), 2*len(u))) {
			n += 2 * int64(len(u))
		}
	}
	return n
}

// DenseVectorBytes sums the norms, biases, sinks, and small f32 tables beside w's matrices, counting aliases once.
func DenseVectorBytes(w *darch.Weights) int64 {
	if w == nil {
		return 0
	}
	n := F32SliceBytes(w.FinalNorm, w.FinalNormBias, w.PerLayerProjNorm)
	if w.Arch != nil && w.Arch.Inkling2Head != nil {
		n += F32SliceBytes(w.Arch.Inkling2Head.EmbedNorm, w.Arch.Inkling2Head.OutputResQuery)
	}
	for i := range w.Layers {
		if w.Arch != nil && w.Arch.WeightLayerOf(i) != i {
			continue // a later visit of an Inkling 2 loop layer: its tensors are the first visit's
		}
		l := &w.Layers[i]
		if x := l.Inkling; x != nil {
			n += F32SliceBytes(x.QClip, x.RelClip, x.RelBias, x.KConv, x.VConv, x.AttnConv, x.FFConv, x.KInpNorm, x.VInpNorm,
				x.AttnResQuery, x.FFNResQuery, x.MsgQuery)
			for c := range x.Ctrl {
				n += F32SliceBytes(x.Ctrl[c].Down, x.Ctrl[c].Gain, x.Ctrl[c].Forget)
			}
		}
		n += F32SliceBytes(l.QBias, l.KBias, l.VBias, l.OBias, l.QNorm, l.KNorm, l.PreAttnNorm, l.PreAttnNormBias, l.PostAttnNorm,
			l.UpBias, l.DownBias, l.PreMLPNorm, l.PreMLPNormBias, l.PostMLPNorm, l.RouterBias, l.AttnSinks, l.PostPLENorm,
			l.SharedExpert.GateBias, l.SharedExpert.UpBias, l.SharedExpert.DownBias)
		if q := l.QAttn; q != nil {
			n += F32SliceBytes(q.QNorm, q.KNorm)
		}
		if mla := l.MLA; mla != nil {
			n += F32SliceBytes(mla.QALayernorm, mla.KVALayernorm)
		}
		if mb := l.Mamba; mb != nil {
			n += F32SliceBytes(mb.ConvB, mb.ALog, mb.D, mb.DtBias, mb.NormW)
		}
		if mo := l.Gemma4MoE; mo != nil {
			n += F32SliceBytes(mo.PostFFNNorm1, mo.PreFFNNorm2, mo.PostFFNNorm2, mo.RouterScale, mo.PerExpertScale)
		}
	}
	return n
}

// ExpertBiasBytes sums per-expert f32 biases (gpt-oss).
func ExpertBiasBytes(w *darch.Weights) int64 {
	if w == nil {
		return 0
	}
	var n int64
	for i := range w.Layers {
		for j := range w.Layers[i].Experts {
			e := &w.Layers[i].Experts[j]
			n += F32SliceBytes(e.GateBias, e.UpBias, e.DownBias)
		}
	}
	return n
}

// MambaProjSavings returns the bytes a resident backend saves by uploading f32 Mamba-2 projections as per-row int8.
func MambaProjSavings(backend string, w *darch.Weights) int64 {
	if backend == "cpu" || w == nil || w.Cfg.HiddenDim <= 0 {
		return 0
	}
	saved := func(floats, rows int) int64 { return int64(floats)*3 - int64(rows)*4 }
	hidden := w.Cfg.HiddenDim
	var n int64
	for i := range w.Layers {
		if mb := w.Layers[i].Mamba; mb != nil && mb.InQ.Rows() == 0 {
			n += saved(len(mb.InProj), len(mb.InProj)/hidden) + saved(len(mb.OutProj), hidden)
		}
	}
	return n
}

// HostSideBytes returns the dense-sum portion backend keeps on the host rather than uploading to device memory.
func HostSideBytes(backend string, w *darch.Weights) int64 {
	if (backend != "cuda" && backend != "webgpu") || w == nil {
		return 0
	}
	n := WMBytes(&w.PosEmbed) + WMBytes(&w.PerLayerTokenEmbed) + WMBytes(&w.PerLayerModelProj)
	if w.LMHead.Rows() != 0 {
		n += WMBytes(&w.Embed)
	}
	return n
}

// WeightBytesSplit enumerates w's dense and routed-expert bytes, and the mmap-aliased subsets reported by in.
func WeightBytesSplit(w *darch.Weights, in func([]byte) bool) (dense, mmapDense int64, expertBytesAt, mmapExpertBytesAt func(slots int) int64) {
	noop := func(int) int64 { return 0 }
	if w == nil {
		return 0, 0, noop, noop
	}
	count := func(mat *linalg.WeightMat) {
		dense += WMBytes(mat)
		mmapDense += MmapAliasedBytes(mat, in)
	}
	count(&w.Embed)
	count(&w.LMHead)
	count(&w.PosEmbed)
	count(&w.PerLayerTokenEmbed)
	count(&w.PerLayerModelProj)
	dense += DenseVectorBytes(w)
	var fullExpertBytes, mmapExpertBytes []int64
	var expertCounts []int
	for i := range w.Layers {
		if w.Arch != nil && w.Arch.WeightLayerOf(i) != i {
			continue // a later visit of an Inkling 2 loop layer: its tensors are the first visit's
		}
		l := &w.Layers[i]
		for _, mat := range []*linalg.WeightMat{
			&l.QProj, &l.KProj, &l.VProj, &l.OProj, &l.GProj,
			&l.GateProj, &l.UpProj, &l.DownProj,
			&l.Router, &l.SharedGate, &l.PLEGate, &l.PLEProj,
		} {
			count(mat)
		}
		var expertBytes, mmapEBytes int64
		for j := range l.Experts {
			e := &l.Experts[j]
			expertBytes += WMBytes(&e.Gate) + WMBytes(&e.Up) + WMBytes(&e.Down)
			mmapEBytes += MmapAliasedBytes(&e.Gate, in) + MmapAliasedBytes(&e.Up, in) + MmapAliasedBytes(&e.Down, in)
		}
		fullExpertBytes = append(fullExpertBytes, expertBytes)
		mmapExpertBytes = append(mmapExpertBytes, mmapEBytes)
		expertCounts = append(expertCounts, len(l.Experts))
		count(&l.SharedExpert.Gate)
		count(&l.SharedExpert.Up)
		count(&l.SharedExpert.Down)

		if d := l.Delta; d != nil {
			count(&d.InProjQKV)
			count(&d.InProjZ)
			count(&d.OutProj)
			dense += F32SliceBytes(d.InProjB, d.InProjA, d.ConvW, d.DtBias, d.NegExpA, d.NormW)
		}
		if q := l.QAttn; q != nil {
			count(&q.QProj)
			count(&q.KProj)
			count(&q.VProj)
			count(&q.OProj)
		}
		if mla := l.MLA; mla != nil {
			dense += F32SliceBytes(mla.QAProj, mla.QBProj, mla.QProj, mla.KVAProj, mla.KVBProj, mla.OProj, mla.GProj) +
				WMBytes(&mla.QAQ) + WMBytes(&mla.QBQ) + WMBytes(&mla.QQ) + WMBytes(&mla.KVAQ) + WMBytes(&mla.OQ)
		}
		if k := l.KDA; k != nil {
			for _, wm := range []*linalg.WeightMat{&k.QProj, &k.KProj, &k.VProj, &k.FProj, &k.GProj, &k.OProj} {
				count(wm)
			}
			dense += F32SliceBytes(k.QConvW, k.KConvW, k.VConvW, k.BProj, k.DtBias, k.ALog, k.ONormW)
		}
		if mb := l.Mamba; mb != nil {
			dense += F32SliceBytes(mb.InProj, mb.ConvW, mb.OutProj) + WMBytes(&mb.InQ) + WMBytes(&mb.OutQ)
		}
		if x := l.Inkling; x != nil {
			for _, m := range []*linalg.WeightMat{&x.SinkProj, &x.Rel, &x.KConvProj, &x.VConvProj} {
				count(m)
			}
		}
		if sc := l.ShortConv; sc != nil {
			count(&sc.InProj)
			count(&sc.OutProj)
			dense += F32SliceBytes(sc.ConvW)
		}
		if mo := l.Gemma4MoE; mo != nil {
			count(&mo.RouterProj)
			var fusedBytes, mmapFusedBytes int64
			for e := range mo.ExpertsGateUp {
				fusedBytes += WMBytes(&mo.ExpertsGateUp[e]) + WMBytes(&mo.ExpertsDown[e])
				mmapFusedBytes += MmapAliasedBytes(&mo.ExpertsGateUp[e], in) + MmapAliasedBytes(&mo.ExpertsDown[e], in)
			}
			fullExpertBytes = append(fullExpertBytes, fusedBytes)
			mmapExpertBytes = append(mmapExpertBytes, mmapFusedBytes)
			expertCounts = append(expertCounts, len(mo.ExpertsGateUp))
		}
	}
	capAt := func(full []int64) func(slots int) int64 {
		return func(slots int) int64 {
			var total int64
			for k, fullK := range full {
				nExperts := expertCounts[k]
				if nExperts == 0 || slots <= 0 || slots >= nExperts {
					total += fullK
					continue
				}
				total += fullK / int64(nExperts) * int64(slots)
			}
			return total
		}
	}
	return dense, mmapDense, capAt(fullExpertBytes), capAt(mmapExpertBytes)
}

// WeightBytesPaged returns w's total resident weight footprint with routed experts capped at slots per layer (0 = unpaged).
func WeightBytesPaged(w *darch.Weights, slots int) int64 {
	dense, _, expertBytesAt, _ := WeightBytesSplit(w, nil)
	return dense + expertBytesAt(slots) + ExpertBiasBytes(w)
}

// HostCopyBytes returns the non-mmap-aliased host copy of w that a unified-memory backend holds alongside its device buffers.
func HostCopyBytes(w *darch.Weights, slots int, in func([]byte) bool) int64 {
	dense, mmapDense, expertBytesAt, mmapExpertBytesAt := WeightBytesSplit(w, in)
	if slots > 0 {
		return dense - mmapDense + ExpertBiasBytes(w)
	}
	return dense + expertBytesAt(0) + ExpertBiasBytes(w) - mmapDense - mmapExpertBytesAt(0)
}

// KQuantLane returns the weight matrices backend's q4k lane uploads as native Q5_K/Q6_K blocks in place of int8 copies.
func KQuantLane(backend string, w *darch.Weights, quant string, actQuantGroup int, nativeOff bool) []*linalg.WeightMat {
	if nativeOff || w == nil || len(w.KQuant) == 0 || quant != "q4k" || (backend != "metal" && backend != "cuda") {
		return nil
	}
	if backend == "cuda" && actQuantGroup != 32 {
		return nil
	}
	var ws []*linalg.WeightMat
	for l := range w.Layers {
		lw := &w.Layers[l]
		ps := []*linalg.WeightMat{&lw.QProj, &lw.KProj, &lw.VProj, &lw.OProj, &lw.GateProj, &lw.UpProj, &lw.DownProj}
		if backend == "cuda" && w.Arch != nil {
			switch {
			case w.Arch.Gemma4KVSrcAt(l) != l:
				ps = []*linalg.WeightMat{&lw.QProj, &lw.OProj, &lw.GateProj, &lw.UpProj, &lw.DownProj}
			case VFromK(w, l):
				ps = []*linalg.WeightMat{&lw.QProj, &lw.KProj, &lw.OProj, &lw.GateProj, &lw.UpProj, &lw.DownProj}
			}
		}
		for _, wm := range ps {
			if _, ok := w.KQuantOf(wm); ok {
				ws = append(ws, wm)
			}
		}
	}
	if _, ok := w.KQuantOf(&w.LMHead); ok && backend == "cuda" && w.LMHead.Rows() > 0 {
		ws = append(ws, &w.LMHead)
	}
	return ws
}

// KQuantSavings returns the device bytes backend's q4k lane saves by uploading native Q5_K/Q6_K blocks instead of int8 copies.
func KQuantSavings(backend string, w *darch.Weights, quant string, actQuantGroup int, nativeOff bool) int64 {
	var n int64
	for _, wm := range KQuantLane(backend, w, quant, actQuantGroup, nativeOff) {
		s, _ := w.KQuantOf(wm)
		n += s.Int8Bytes - s.NativeBytes
	}
	return n
}

// KQuantNativeKind returns "q5k" or "q6k" when backend's q4k lane uploads wm as native file blocks, else "".
func KQuantNativeKind(backend string, w *darch.Weights, quant string, actQuantGroup int, nativeOff bool, wm *linalg.WeightMat) string {
	if slices.Contains(KQuantLane(backend, w, quant, actQuantGroup, nativeOff), wm) {
		s, _ := w.KQuantOf(wm)
		return s.Kind
	}
	return ""
}
