package vision

import (
	"fmt"
	"math"

	"github.com/townsendmerino/aikit/linalg"
)

// qwen_resident.go is the device-resident seam for the Qwen2.5-VL vision tower —
// the counterpart of resident.go's ResidentEncoder for SigLIP, and the follow-on
// qwen_encoder.go's header flagged ("the resident-GPU seam is a follow-on").
//
// Same inversion as everywhere else in aikit: this package defines the interface and
// the weight export, a GPU module implements it and registers a factory from its
// init(), and `vision` never imports the GPU package — so the default build stays
// pure-Go CPU.

// QwenResidentEncoder is a device-resident Qwen2.5-VL ViT forward.
//
// The seam is at ForwardViT — the transformer tower — and NOT at the patch merger,
// deliberately. The tower is essentially all of the FLOPs (32 blocks over thousands
// of patches at dynamic resolution); the merger is three small ops over
// n_patches/merge² groups. Keeping the merger on the CPU makes the resident surface
// small and leaves one obvious place to move next if it ever shows up in a profile.
type QwenResidentEncoder interface {
	// ForwardViT runs the ViT blocks and returns the pre-merge hidden state
	// [n_patches * hidden] in ORIGINAL patch order (de-windowed), matching
	// QwenVisionEncoder.ForwardViT's contract exactly.
	ForwardViT(pixelValues []float32, gridTHW [][3]int) ([]float32, error)
	Close()
}

var qwenResidentFactory func(*QwenVisionEncoder) (QwenResidentEncoder, error)

// RegisterQwenResident plugs in a device-resident Qwen ViT backend. Called from a GPU
// module's init(); the core never imports it.
func RegisterQwenResident(f func(*QwenVisionEncoder) (QwenResidentEncoder, error)) {
	qwenResidentFactory = f
}

// EnableResident attaches a device-resident ViT. Returns an error (leaving the CPU
// path intact) when no backend is registered or the upload fails.
func (e *QwenVisionEncoder) EnableResident() error {
	if qwenResidentFactory == nil {
		return fmt.Errorf("vision: no resident Qwen ViT backend registered (import a gpu/qwen* module)")
	}
	r, err := qwenResidentFactory(e)
	if err != nil {
		return err
	}
	e.resident = r
	return nil
}

// ResidentEnabled reports whether the ViT runs on a device backend.
func (e *QwenVisionEncoder) ResidentEnabled() bool { return e.resident != nil }

// Close releases a resident backend if one is attached and the checkpoint mappings the weights alias (aliasQMat); the tower's
// forwards return an error afterward. Idempotent.
func (e *QwenVisionEncoder) Close() {
	if e.resident != nil {
		e.resident.Close()
		e.resident = nil
	}
	e.blocks, e.merger0w, e.merger2w = nil, linalg.WeightMat{}, linalg.WeightMat{}
	e.maps.close()
}

// --- GPU weight export ---
//
// The resident backend reads the loaded tower through these exported types, so
// `vision` stays pure-Go and the GPU module never touches unexported fields.

// QwenGPUMat is one matmul weight. A Qwen tower may be loaded fp32 (quant=false, the
// parity configuration) or int8 (quant=true), so BOTH forms are carried and the
// consumer picks: Q/Scales when Quantized, F32 otherwise.
type QwenGPUMat struct {
	Quantized  bool
	Q          []int8
	Scales     []float32
	F32        []float32
	Rows, Cols int
}

// QwenGPUBlock is one ViT block's weights.
type QwenGPUBlock struct {
	Norm1w, Norm2w []float32
	QKVw           QwenGPUMat
	QKVb           []float32
	Projw          QwenGPUMat
	Projb          []float32
	Gatew, Upw     QwenGPUMat
	Gateb, Upb     []float32
	Downw          QwenGPUMat
	Downb          []float32
}

// QwenGPUWeights is the whole vision tower, flattened for a device backend.
type QwenGPUWeights struct {
	Hidden, Inter, NumHeads, HeadDim int
	NumLayers                        int
	PatchDim                         int
	SpatialMergeSize, WindowSize     int
	PatchSize, TemporalPatchSize     int
	FullattBlockIndexes              []int
	OutHiddenSize                    int
	PatchW                           []float32 // [hidden, patch_dim], f32
	Blocks                           []QwenGPUBlock
	MergerLNw                        []float32
	Merger0w, Merger2w               QwenGPUMat
	Merger0b, Merger2b               []float32
	RotInvFreq                       []float32
}

// qwenGPUMat converts a WeightMat, carrying whichever representation it holds.
func qwenGPUMat(m linalg.WeightMat) QwenGPUMat {
	out := QwenGPUMat{Rows: m.Rows(), Cols: m.Cols()}
	if q, scales, _, ok := m.Int8(); ok {
		out.Quantized, out.Q, out.Scales = true, q, scales
		return out
	}
	if f, ok := m.F32(); ok {
		out.F32 = f
		return out
	}
	out.F32 = widenHalf16(m)
	return out
}

// GPUWeights exports the tower for a resident backend. Unlike SigLIP's, this does not
// require int8: the Qwen parity configuration loads fp32, and a backend that only
// supports one form should check QwenGPUMat.Quantized and decline clearly.
func (e *QwenVisionEncoder) GPUWeights() QwenGPUWeights {
	if err := e.ensureBlocks(); err != nil {
		panic(err) // GPUWeights has no error return; a released tower whose checkpoint is gone cannot export
	}
	c := e.Cfg
	w := QwenGPUWeights{
		Hidden: c.HiddenSize, Inter: c.IntermediateSize,
		NumHeads: c.NumHeads, HeadDim: c.HiddenSize / c.NumHeads,
		NumLayers:        len(e.blocks),
		PatchDim:         c.InChans * c.TemporalPatchSize * c.PatchSize * c.PatchSize,
		SpatialMergeSize: c.SpatialMergeSize, WindowSize: c.WindowSize,
		PatchSize: c.PatchSize, TemporalPatchSize: c.TemporalPatchSize,
		FullattBlockIndexes: c.FullattBlockIndexes,
		OutHiddenSize:       c.OutHiddenSize,
		PatchW:              e.patchW,
		Blocks:              make([]QwenGPUBlock, len(e.blocks)),
		MergerLNw:           e.mergerLNw,
		Merger0w:            qwenGPUMat(e.merger0w), Merger0b: e.merger0b,
		Merger2w: qwenGPUMat(e.merger2w), Merger2b: e.merger2b,
		RotInvFreq: e.rotInvFreq,
	}
	for i := range e.blocks {
		b := &e.blocks[i]
		w.Blocks[i] = QwenGPUBlock{
			Norm1w: b.norm1w, Norm2w: b.norm2w,
			QKVw: qwenGPUMat(b.qkvw), QKVb: b.qkvb,
			Projw: qwenGPUMat(b.projw), Projb: b.projb,
			Gatew: qwenGPUMat(b.gatew), Upw: qwenGPUMat(b.upw),
			Gateb: b.gateb, Upb: b.upb,
			Downw: qwenGPUMat(b.downw), Downb: b.downb,
		}
	}
	return w
}

// WindowPlan is everything the device forward needs that depends on gridTHW rather
// than on the weights: the window permutation, the per-patch attention-segment bounds
// for both the windowed and full-attention blocks, and the per-patch rotary cos/sin —
// all computed on the host, where the index arithmetic already lives.
type WindowPlan struct {
	NPatches int
	WinIdx   []int // merge-unit group g reads from source group WinIdx[g]
	// Per-PATCH segment bounds, in WINDOW order, for the two block kinds.
	WinStart, WinEnd   []int32
	FullStart, FullEnd []int32
	MaxWinSeg          int
	MaxFullSeg         int
	Cos, Sin           []float32 // [nPatches * headDim], window order
}

// BuildWindowPlan computes the per-call layout for a gridTHW. Exported so a resident
// backend reuses the CPU path's exact windowing and rotary maths rather than
// reimplementing them — those indices are where a ViT silently goes wrong.
func (e *QwenVisionEncoder) BuildWindowPlan(gridTHW [][3]int) (*WindowPlan, error) {
	c := e.Cfg
	merge := c.SpatialMergeSize
	mergeUnit := merge * merge
	nPatches := 0
	for _, g := range gridTHW {
		if g[0] <= 0 || g[1] <= 0 || g[2] <= 0 {
			return nil, fmt.Errorf("vision: non-positive grid dim in %v", g)
		}
		if g[1]%merge != 0 || g[2]%merge != 0 {
			return nil, fmt.Errorf("vision: grid %v h/w not divisible by spatial_merge_size %d", g, merge)
		}
		nPatches += g[0] * g[1] * g[2]
	}
	if nPatches == 0 {
		return nil, fmt.Errorf("vision: empty gridTHW (no images/patches)")
	}
	if nPatches%mergeUnit != 0 {
		return nil, fmt.Errorf("vision: n_patches %d not a multiple of merge² %d", nPatches, mergeUnit)
	}

	freqs := e.rotaryFreqs(gridTHW)
	winIdx, cuWin := e.windowIndex(gridTHW)
	cuFull := cuSeqlensFull(gridTHW)
	groups := nPatches / mergeUnit
	rdim := len(freqs) / nPatches
	headDim := c.HiddenSize / c.NumHeads

	// freqs → window order, then cos/sin over the full head_dim (emb = cat(f,f)).
	fWin := make([]float32, nPatches*rdim)
	for g := range groups {
		src := winIdx[g]
		for u := range mergeUnit {
			df, sf := (g*mergeUnit+u)*rdim, (src*mergeUnit+u)*rdim
			copy(fWin[df:df+rdim], freqs[sf:sf+rdim])
		}
	}
	p := &WindowPlan{
		NPatches: nPatches, WinIdx: winIdx,
		Cos: make([]float32, nPatches*headDim),
		Sin: make([]float32, nPatches*headDim),
	}
	for i := range nPatches {
		fr := fWin[i*rdim : i*rdim+rdim]
		for d := range headDim {
			f := float64(fr[d%rdim])
			p.Cos[i*headDim+d] = float32(math.Cos(f))
			p.Sin[i*headDim+d] = float32(math.Sin(f))
		}
	}
	p.WinStart, p.WinEnd, p.MaxWinSeg = expandSegments(cuWin, nPatches)
	p.FullStart, p.FullEnd, p.MaxFullSeg = expandSegments(cuFull, nPatches)
	return p, nil
}

// expandSegments turns a cu_seqlens prefix array into per-patch [start,end) bounds,
// which is what a device kernel wants (no in-kernel search).
func expandSegments(cu []int, n int) (start, end []int32, maxSeg int) {
	start = make([]int32, n)
	end = make([]int32, n)
	for s := 1; s < len(cu); s++ {
		a, b := cu[s-1], cu[s]
		if b-a > maxSeg {
			maxSeg = b - a
		}
		for i := a; i < b && i < n; i++ {
			start[i], end[i] = int32(a), int32(b)
		}
	}
	return start, end, maxSeg
}

// IsFullAtt reports whether block li uses full (per-image) attention rather than
// windowed. Exported for resident backends; mirrors the CPU path's isFullAtt.
func (e *QwenVisionEncoder) IsFullAtt(li int) bool { return e.isFullAtt(li) }

// MergeHidden runs the patch merger on a pre-merge hidden state — the CPU tail of the
// resident path (see QwenResidentEncoder's doc for why the merger stays here).
func (e *QwenVisionEncoder) MergeHidden(hidden []float32, gridTHW [][3]int) []float32 {
	return e.merge(hidden, gridTHW)
}
