//go:build gpu

//lint:file-ignore SA1019 Release is Close's deprecated alias, called here as before; it is deprecated for new code only.

package webgpu

import (
	"fmt"
	"slices"

	"github.com/oliverbestmann/webgpu/wgpu"
	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/backend/webgpu/internal/kernels"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
)

// webgpuBuild is the state of one webgpuBackend.BuildResident: the model, the device context and the resident decoder being
// assembled. Its methods are the build's phases and upload helpers; each registers what it allocates with keep, so a failed
// build releases everything through rd.release.
type webgpuBuild struct {
	m  *decoder.Model
	c  *Context
	rd *residentDecoder

	// ropeInvCache holds the invFreq tables uploaded so far; a layer whose table is equal reuses the buffer.
	ropeInvCache []ropeEntry
}

type ropeEntry struct {
	invf []float32
	buf  *wgpu.Buffer
}

// keep registers a release for rd.Close to run.
func (bs *webgpuBuild) keep(f func()) { bs.rd.keep = append(bs.rd.keep, f) }

// up32 uploads v as a dense f32 buffer.
func (bs *webgpuBuild) up32(v []float32) (*wgpu.Buffer, error) {
	d, err := bs.c.UploadF32On(mem.LineDense, v)
	if err != nil {
		return nil, err
	}
	bs.keep(d.Release)
	return (*kernels.DeviceBufferBuf(d)), nil
}

// proj uploads one projection at its native precision (uploadProj).
func (bs *webgpuBuild) proj(pw *linalg.WeightMat) (kernels.DecodeWeight, error) {
	dw, err := uploadProj(bs.c, pw)
	if err != nil {
		return nil, err
	}
	switch t := dw.(type) {
	case *ResidentW4A8:
		bs.keep(t.Release)
	case *ResidentW8A8:
		bs.keep(t.Release)
	}
	return dw, nil
}

// projF32 quantizes a raw f32 [N,K] weight to int8 row-wise and uploads it (the MLA
// projections are stored f32 in the decoder; the resident GEMVs are W8A8).
func (bs *webgpuBuild) projF32(w []float32, N, K int) (kernels.DecodeWeight, error) {
	q8, scales := linalg.QuantizeRowsInt8(w, N, K)
	rm, err := bs.c.UploadW8A8(q8, scales, N, K)
	if err != nil {
		return nil, err
	}
	bs.keep(rm.Release)
	return rm, nil
}

// projInt8 uploads an already-int8 weight matrix (per-row scales) as projF32 uploads its quantized f32.
func (bs *webgpuBuild) projInt8(w *linalg.WeightMat, N, K int) (kernels.DecodeWeight, error) {
	q8, scales, _, ok := w.Int8()
	if !ok || w.Rows() != N || w.Cols() != K {
		return nil, fmt.Errorf("gpu: an int8 projection of %d×%d (int8 %v), want %d×%d", w.Rows(), w.Cols(), ok, N, K)
	}
	rm, err := bs.c.UploadW8A8(q8, scales, N, K)
	if err != nil {
		return nil, err
	}
	bs.keep(rm.Release)
	return rm, nil
}

// Granite multiplier folding: ResidMul scales every residual-add weight (o-proj / mamba out-proj / MoE expert-down /
// shared-down); 1/LogitScale folds into the lm_head; EmbMul applies to the embedding in decoder.embedResident. mult==1 ⇒
// the plain proj path, byte-identical for non-granite models.
func scaleCopy(s []float32, m float32) []float32 {
	out := make([]float32, len(s))
	for i, v := range s {
		out[i] = v * m
	}
	return out
}

// projMul is proj with mult folded into the row scales; mult == 1 is proj itself.
func (bs *webgpuBuild) projMul(pw *linalg.WeightMat, mult float32) (kernels.DecodeWeight, error) {
	if mult == 1 {
		return bs.proj(pw)
	}
	N, K := pw.Rows(), pw.Cols()
	var q8 []int8
	var scales []float32
	switch pw.Kind() {
	case "int8":
		q8, scales, _, _ = pw.Int8()
	case "f32":
		f32, ok := pw.F32()
		if !ok {
			return nil, fmt.Errorf("gpu: granite mult-fold f32 has no data")
		}
		q8, scales = linalg.QuantizeRowsInt8(f32, N, K)
	default:
		return nil, fmt.Errorf("gpu: granite mult-fold unsupported kind %q", pw.Kind())
	}
	rm, err := bs.c.UploadW8A8(q8, scaleCopy(scales, mult), N, K)
	if err != nil {
		return nil, err
	}
	bs.keep(rm.Release)
	return rm, nil
}

// projF32Mul is projF32 with mult folded into the row scales.
func (bs *webgpuBuild) projF32Mul(w []float32, N, K int, mult float32) (kernels.DecodeWeight, error) {
	q8, scales := linalg.QuantizeRowsInt8(w, N, K)
	if mult != 1 {
		scales = scaleCopy(scales, mult)
	}
	rm, err := bs.c.UploadW8A8(q8, scales, N, K)
	if err != nil {
		return nil, err
	}
	bs.keep(rm.Release)
	return rm, nil
}

// stateBuf allocates a build-once, zeroed, in-place-updatable Mamba state buffer
// (Storage|CopyDst so Reset can re-zero it per generation).
func (bs *webgpuBuild) stateBuf(n int) (*wgpu.Buffer, error) {
	b2, err := kernels.CreateBufferOn(bs.c, mem.LineState, &wgpu.BufferDescriptor{Label: "mamba-state", Size: uint64(n * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst})
	if err != nil {
		return nil, err
	}
	(*kernels.ContextQueue(bs.c)).TryWriteBuffer(b2, 0, wgpu.ToBytes(make([]float32, n)))
	bs.keep(kernels.BufferReleaser(b2))
	return b2, nil
}

// buildStacked packs one projection (gate/up/down) across all nE experts into a resident stacked buffer (int8 or int4; a
// mixed set is an error) that the indexed expert GEMV reads. mult folds Granite's ResidMul into the scales.
func (bs *webgpuBuild) buildStacked(nE int, get func(e int) *linalg.WeightMat, mult float32) (*ResidentStackedW8A8, error) {
	w0 := get(0)
	N, K := w0.Rows(), w0.Cols()
	switch w0.Kind() {
	case "int8":
		q8 := make([][]int8, nE)
		sc := make([][]float32, nE)
		for e := range nE {
			w := get(e)
			if w.Kind() != "int8" {
				return nil, fmt.Errorf("gpu: MoE residency expert %d kind %q (mixed; want int8)", e, w.Kind())
			}
			q, s, _, _ := w.Int8()
			if mult != 1 { // Granite ResidMul fold into the expert-down residual add
				s = scaleCopy(s, mult)
			}
			q8[e], sc[e] = q, s
		}
		return bs.c.UploadStackedExperts(q8, sc, nE, N, K)
	case "int4":
		// int4 stacked experts (W4A8) fit VRAM where int8 would spill. Fast path (K%32==0): the decoder's 2-nibble/byte int4 is
		// byte-identical to the GPU packed layout, so its bytes go straight to the Packed upload, with no per-element unpack and
		// re-pack.
		if K%kernels.W4a8GroupSize == 0 {
			q4s := make([][]byte, nE)
			scs := make([][]float32, nE)
			for e := range nE {
				w := get(e)
				if w.Kind() != "int4" {
					return nil, fmt.Errorf("gpu: MoE residency expert %d kind %q (mixed; want int4)", e, w.Kind())
				}
				q4, q4sc, group, _ := decoder.Int4F32(w)
				if group != kernels.W4a8GroupSize {
					return nil, fmt.Errorf("gpu: MoE residency int4 group %d != %d", group, kernels.W4a8GroupSize)
				}
				s := q4sc
				if mult != 1 { // Granite ResidMul fold (no-op for Mellum)
					s = scaleCopy(s, mult)
				}
				q4s[e], scs[e] = q4, s
			}
			return bs.c.UploadStackedExpertsInt4Packed(q4s, scs, nE, N, K)
		}
		// Fallback (K not a multiple of 32): unpack 2-nibble/byte → one nibble per element.
		nib := make([][]uint8, nE)
		sc := make([][]float32, nE)
		for e := range nE {
			w := get(e)
			if w.Kind() != "int4" {
				return nil, fmt.Errorf("gpu: MoE residency expert %d kind %q (mixed; want int4)", e, w.Kind())
			}
			q4, q4s, group, _ := decoder.Int4F32(w)
			if group != kernels.W4a8GroupSize {
				return nil, fmt.Errorf("gpu: MoE residency int4 group %d != %d", group, kernels.W4a8GroupSize)
			}
			un := make([]uint8, N*K)
			for r := range N {
				row := q4[r*((K+1)/2):]
				d := un[r*K : r*K+K]
				for k := range K {
					b := row[k>>1]
					if k&1 == 0 {
						d[k] = b & 0x0F
					} else {
						d[k] = b >> 4
					}
				}
			}
			s := q4s
			if mult != 1 { // Granite ResidMul fold (no-op for Mellum)
				s = scaleCopy(s, mult)
			}
			nib[e], sc[e] = un, s
		}
		return bs.c.UploadStackedExpertsInt4(nib, sc, nE, N, K)
	default:
		return nil, fmt.Errorf("gpu: MoE residency expert kind %q (int8/int4 only)", w0.Kind())
	}
}

// ropeInvBuf returns the invFreq table buffer layer binds, nil when the layer has none. Most models share one table (the cache is
// keyed on table equality); a family with local/global RoPE has two. Layer 0 resolves a NoPE or NoPositionEncoding layer (SmolLM3,
// Olmo Hybrid) to the rotaryDim/2 zero table rather than an empty slice.
func (bs *webgpuBuild) ropeInvBuf(layer int) (*wgpu.Buffer, error) {
	invf := bs.m.RopeInvFreqLayerResident(layer)
	if len(invf) == 0 {
		return nil, nil
	}
	if nonzero := bs.m.RopeInvFreq(); noPEOffForTest && len(nonzero) > 0 && !slices.ContainsFunc(invf, func(v float32) bool { return v != 0 }) {
		invf = nonzero
	}
	for _, ent := range bs.ropeInvCache {
		if slices.Equal(ent.invf, invf) {
			return ent.buf, nil
		}
	}
	b, err := bs.up32(invf)
	if err != nil {
		return nil, err
	}
	bs.ropeInvCache = append(bs.ropeInvCache, ropeEntry{invf: invf, buf: b})
	return b, nil
}
