package quant

import (
	"fmt"
	"strings"
	"unsafe"

	"github.com/townsendmerino/aikit/embed"
)

// TensorU32 reads a DType "U32" tensor from st as []uint32, optionally validating
// its shape against want.
func TensorU32(st *embed.SafetensorsFile, name string, want ...int) ([]uint32, error) {
	t, err := st.Tensor(name)
	if err != nil {
		return nil, err
	}
	if len(want) > 0 {
		if len(t.Shape) != len(want) {
			return nil, fmt.Errorf("safetensors: tensor %q shape %v != want %v", name, t.Shape, want)
		}
		for i := range want {
			if t.Shape[i] != want[i] {
				return nil, fmt.Errorf("safetensors: tensor %q shape %v != want %v", name, t.Shape, want)
			}
		}
	}
	if t.DType != "U32" {
		return nil, fmt.Errorf("safetensors: tensor %q: expected U32, got %s", name, t.DType)
	}
	elems := t.Elements()
	if elems < 0 {
		return nil, fmt.Errorf("safetensors: tensor %q shape %v overflows", name, t.Shape)
	}
	t.DType = "I32"
	i32s, err := t.Int32s()
	if err != nil {
		return nil, err
	}
	if len(i32s) != elems {
		return nil, fmt.Errorf("safetensors: tensor %q has %d U32 words, shape %v implies %d", name, len(i32s), t.Shape, elems)
	}
	if len(i32s) == 0 {
		return nil, nil
	}
	return unsafe.Slice((*uint32)(unsafe.Pointer(&i32s[0])), len(i32s)), nil
}

// MLXDequantAffineRow unpacks one row of in affine-quantized MLX elements (bits in {4, 8})
// from words (len(words) == in * bits / 32) using per-group scales and biases
// (len == in / groupSize) into dst[:in].
func MLXDequantAffineRow(words []uint32, scales, biases []float32, in, bits, groupSize int, dst []float32) error {
	if groupSize <= 0 || in <= 0 || in%groupSize != 0 {
		return fmt.Errorf("decoder(mlx): invalid row geometry in=%d group_size=%d", in, groupSize)
	}
	nGroups := in / groupSize
	if len(scales) != nGroups || len(biases) != nGroups {
		return fmt.Errorf("decoder(mlx): %d scales / %d biases for %d groups", len(scales), len(biases), nGroups)
	}
	if len(dst) < in {
		return fmt.Errorf("decoder(mlx): dst len %d < %d", len(dst), in)
	}
	switch bits {
	case 4:
		if in%8 != 0 || len(words) != in/8 {
			return fmt.Errorf("decoder(mlx): %d U32 words for 4-bit row of width %d", len(words), in)
		}
		for u, w := range words {
			base := u * 8
			for k := range 8 {
				c := base + k
				g := c / groupSize
				q := float32((w >> (4 * k)) & 0x0F)
				dst[c] = q*scales[g] + biases[g]
			}
		}
	case 8:
		if in%4 != 0 || len(words) != in/4 {
			return fmt.Errorf("decoder(mlx): %d U32 words for 8-bit row of width %d", len(words), in)
		}
		for u, w := range words {
			base := u * 4
			for k := range 4 {
				c := base + k
				g := c / groupSize
				q := float32((w >> (8 * k)) & 0xFF)
				dst[c] = q*scales[g] + biases[g]
			}
		}
	default:
		return fmt.Errorf("decoder(mlx): unsupported affine bit-width %d (want 4 or 8)", bits)
	}
	return nil
}

// MLXDequantNVFP4Row unpacks one row of in MLX "nvfp4" elements (group_size=16,
// 8 E2M1 nibbles per uint32 word, F8_E4M3 scales) into dst[:in].
func MLXDequantNVFP4Row(words []uint32, scaleBytes []byte, in int, dst []float32) error {
	if in <= 0 || in%16 != 0 {
		return fmt.Errorf("decoder(mlx-nvfp4): row width %d is not a positive multiple of 16", in)
	}
	if len(words) != in/8 || len(scaleBytes) != in/16 {
		return fmt.Errorf("decoder(mlx-nvfp4): %d U32 words / %d scales for row width %d", len(words), len(scaleBytes), in)
	}
	for u, w := range words {
		base := u * 8
		s := FP8E4M3Table[scaleBytes[base/16]]
		for k := range 8 {
			q := (w >> (4 * k)) & 0x0F
			dst[base+k] = NVFP4E2M1Table[q] * s
		}
	}
	return nil
}

// MLXDequantMXFP4Row unpacks one row of in MLX "mxfp4" elements (group_size=32,
// 8 E2M1 nibbles per uint32 word, U8 E8M0 scales) into dst[:in].
func MLXDequantMXFP4Row(words []uint32, scaleBytes []byte, in int, dst []float32) error {
	if in <= 0 || in%32 != 0 {
		return fmt.Errorf("decoder(mlx-mxfp4): row width %d is not a positive multiple of 32", in)
	}
	if len(words) != in/8 || len(scaleBytes) != in/32 {
		return fmt.Errorf("decoder(mlx-mxfp4): %d U32 words / %d scales for row width %d", len(words), len(scaleBytes), in)
	}
	for u, w := range words {
		base := u * 8
		s := embed.MXFP4Scale(scaleBytes[base/32]) * 2
		for k := range 8 {
			q := (w >> (4 * k)) & 0x0F
			dst[base+k] = NVFP4E2M1Table[q] * s
		}
	}
	return nil
}

// MLXRowReader2D validates a 2-D MLX quantized tensor (<prefix>.weight U32 [out, packedCols]
// + <prefix>.scales + optional <prefix>.biases) and returns a closure that dequantizes
// row r (0 <= r < out) into dst[:in] without materializing the full [out, in] matrix.
func MLXRowReader2D(st *embed.SafetensorsFile, name string, out, in int) (func(r int, dst []float32) error, error) {
	wT, err := st.Tensor(name)
	if err != nil {
		return nil, fmt.Errorf("mlx %q: %w", name, err)
	}
	if wT.DType != "U32" {
		return nil, fmt.Errorf("mlx %q: dtype %s, want U32", name, wT.DType)
	}
	if out <= 0 || in <= 0 || len(wT.Shape) != 2 || wT.Shape[0] != out || wT.Shape[1] <= 0 {
		return nil, fmt.Errorf("mlx %q: shape %v incompatible with logical [%d,%d]", name, wT.Shape, out, in)
	}
	packedCols := wT.Shape[1]
	if (32*packedCols)%in != 0 {
		return nil, fmt.Errorf("mlx %q: packed cols %d * 32 is not divisible by in=%d", name, packedCols, in)
	}
	bits := (32 * packedCols) / in
	if bits != 4 && bits != 8 {
		return nil, fmt.Errorf("mlx %q: inferred bits=%d from shape %v and in=%d (want 4 or 8)", name, bits, wT.Shape, in)
	}
	words, err := TensorU32(st, name, out, packedCols)
	if err != nil {
		return nil, err
	}
	prefix := strings.TrimSuffix(name, ".weight")
	sName := prefix + ".scales"
	sT, err := st.Tensor(sName)
	if err != nil {
		return nil, fmt.Errorf("mlx %q: %w", sName, err)
	}
	if len(sT.Shape) != 2 || sT.Shape[0] != out || sT.Shape[1] <= 0 || in%sT.Shape[1] != 0 {
		return nil, fmt.Errorf("mlx %q: shape %v incompatible with logical [%d,%d]", sName, sT.Shape, out, in)
	}
	scaleCols := sT.Shape[1]
	groupSize := in / scaleCols

	bName := prefix + ".biases"
	if _, berr := st.Tensor(bName); berr == nil {
		scales, err := st.TensorF32(sName, out, scaleCols)
		if err != nil {
			return nil, fmt.Errorf("mlx %q: %w", sName, err)
		}
		biases, err := st.TensorF32(bName, out, scaleCols)
		if err != nil {
			return nil, fmt.Errorf("mlx %q: %w", bName, err)
		}
		return func(r int, dst []float32) error {
			wRow := words[r*packedCols : (r+1)*packedCols]
			sRow := scales[r*scaleCols : (r+1)*scaleCols]
			bRow := biases[r*scaleCols : (r+1)*scaleCols]
			return MLXDequantAffineRow(wRow, sRow, bRow, in, bits, groupSize, dst)
		}, nil
	}

	if bits != 4 {
		return nil, fmt.Errorf("mlx %q: missing %q for %d-bit tensor", name, bName, bits)
	}
	scaleBytes, err := sT.Uint8s()
	if err != nil {
		return nil, fmt.Errorf("mlx %q: %w", sName, err)
	}
	switch {
	case groupSize == 16 && (sT.DType == "F8_E4M3" || sT.DType == "U8"):
		return func(r int, dst []float32) error {
			wRow := words[r*packedCols : (r+1)*packedCols]
			sRow := scaleBytes[r*scaleCols : (r+1)*scaleCols]
			return MLXDequantNVFP4Row(wRow, sRow, in, dst)
		}, nil
	case groupSize == 32 && sT.DType == "U8":
		return func(r int, dst []float32) error {
			wRow := words[r*packedCols : (r+1)*packedCols]
			sRow := scaleBytes[r*scaleCols : (r+1)*scaleCols]
			return MLXDequantMXFP4Row(wRow, sRow, in, dst)
		}, nil
	default:
		return nil, fmt.Errorf("mlx %q: unsupported bias-free mode (group_size=%d, scales dtype=%s)", name, groupSize, sT.DType)
	}
}

// MLXReconstruct2D reads one 2-D MLX projection or embedding from st and returns
// a row-major [out, in] []float32. Unquantized tensors (F32/BF16/F16) fall back
// to st.TensorF32(name, out, in).
func MLXReconstruct2D(st *embed.SafetensorsFile, name string, out, in int) ([]float32, error) {
	wT, err := st.Tensor(name)
	if err != nil {
		return nil, fmt.Errorf("mlx %q: %w", name, err)
	}
	if wT.DType == "F32" || wT.DType == "BF16" || wT.DType == "F16" {
		return st.TensorF32(name, out, in)
	}
	rowInto, err := MLXRowReader2D(st, name, out, in)
	if err != nil {
		return nil, err
	}
	dst := make([]float32, out*in)
	for r := range out {
		if err := rowInto(r, dst[r*in:(r+1)*in]); err != nil {
			return nil, fmt.Errorf("mlx %q row %d: %w", name, r, err)
		}
	}
	return dst, nil
}

// MLXExpertSlice3D returns a per-expert reader for a 3-D stacked MLX switch_mlp
// projection (<projPrefix>.weight of shape [nExpert, out, packedCols] when U32
// quantized, or [nExpert, out, in] when F32/BF16/F16). Each call readExpert(e)
// reconstructs only expert e's [out, in] matrix.
func MLXExpertSlice3D(st *embed.SafetensorsFile, projPrefix string, nExpert, out, in int) (func(e int) ([]float32, error), error) {
	wName := projPrefix + ".weight"
	wT, err := st.Tensor(wName)
	if err != nil {
		return nil, fmt.Errorf("mlx switch_mlp %q: %w", wName, err)
	}
	if wT.DType == "F32" || wT.DType == "BF16" || wT.DType == "F16" {
		if len(wT.Shape) != 3 || wT.Shape[0] != nExpert || wT.Shape[1] != out || wT.Shape[2] != in {
			return nil, fmt.Errorf("mlx switch_mlp %q: shape %v, want [%d %d %d]", wName, wT.Shape, nExpert, out, in)
		}
		stride := out * in
		return func(e int) ([]float32, error) {
			f32, err := wT.SubF32(e*stride, stride)
			if err != nil {
				return nil, err
			}
			if wT.DType == "F32" {
				return append([]float32(nil), f32...), nil
			}
			return f32, nil
		}, nil
	}
	if wT.DType != "U32" {
		return nil, fmt.Errorf("mlx switch_mlp %q: dtype %s, want U32 or F32/BF16/F16", wName, wT.DType)
	}
	if len(wT.Shape) != 3 || wT.Shape[0] != nExpert || wT.Shape[1] != out || wT.Shape[2] <= 0 {
		return nil, fmt.Errorf("mlx switch_mlp %q: shape %v incompatible with [%d %d %d]", wName, wT.Shape, nExpert, out, in)
	}
	packedCols := wT.Shape[2]
	if (32*packedCols)%in != 0 {
		return nil, fmt.Errorf("mlx switch_mlp %q: packed cols %d * 32 not divisible by in=%d", wName, packedCols, in)
	}
	bits := (32 * packedCols) / in
	if bits != 4 && bits != 8 {
		return nil, fmt.Errorf("mlx switch_mlp %q: inferred bits=%d (want 4 or 8)", wName, bits)
	}
	words, err := TensorU32(st, wName, nExpert, out, packedCols)
	if err != nil {
		return nil, err
	}
	sName := projPrefix + ".scales"
	sT, err := st.Tensor(sName)
	if err != nil {
		return nil, fmt.Errorf("mlx switch_mlp %q: %w", sName, err)
	}
	if len(sT.Shape) != 3 || sT.Shape[0] != nExpert || sT.Shape[1] != out || sT.Shape[2] <= 0 || in%sT.Shape[2] != 0 {
		return nil, fmt.Errorf("mlx switch_mlp %q: shape %v incompatible with [%d %d %d]", sName, sT.Shape, nExpert, out, in)
	}
	scaleCols := sT.Shape[2]
	groupSize := in / scaleCols
	bName := projPrefix + ".biases"
	if _, berr := st.Tensor(bName); berr == nil {
		scales, err := st.TensorF32(sName, nExpert, out, scaleCols)
		if err != nil {
			return nil, fmt.Errorf("mlx switch_mlp %q: %w", sName, err)
		}
		biases, err := st.TensorF32(bName, nExpert, out, scaleCols)
		if err != nil {
			return nil, fmt.Errorf("mlx switch_mlp %q: %w", bName, err)
		}
		return func(e int) ([]float32, error) {
			dst := make([]float32, out*in)
			wBase := e * out * packedCols
			sBase := e * out * scaleCols
			for r := range out {
				wRow := words[wBase+r*packedCols : wBase+(r+1)*packedCols]
				sRow := scales[sBase+r*scaleCols : sBase+(r+1)*scaleCols]
				bRow := biases[sBase+r*scaleCols : sBase+(r+1)*scaleCols]
				if err := MLXDequantAffineRow(wRow, sRow, bRow, in, bits, groupSize, dst[r*in:(r+1)*in]); err != nil {
					return nil, err
				}
			}
			return dst, nil
		}, nil
	}
	scaleBytes, err := sT.Uint8s()
	if err != nil {
		return nil, fmt.Errorf("mlx switch_mlp %q: %w", sName, err)
	}
	return func(e int) ([]float32, error) {
		dst := make([]float32, out*in)
		wBase := e * out * packedCols
		sBase := e * out * scaleCols
		for r := range out {
			wRow := words[wBase+r*packedCols : wBase+(r+1)*packedCols]
			sRow := scaleBytes[sBase+r*scaleCols : sBase+(r+1)*scaleCols]
			var rerr error
			if groupSize == 16 {
				rerr = MLXDequantNVFP4Row(wRow, sRow, in, dst[r*in:(r+1)*in])
			} else {
				rerr = MLXDequantMXFP4Row(wRow, sRow, in, dst[r*in:(r+1)*in])
			}
			if rerr != nil {
				return nil, rerr
			}
		}
		return dst, nil
	}, nil
}

// MLXReconstructor implements Reconstructor and RowStreamReconstructor for MLX quantized tensors.
type MLXReconstructor struct{}

func (MLXReconstructor) Method() string { return "mlx" }

func (MLXReconstructor) ReconstructMat2D(st *embed.SafetensorsFile, name string, out, in int) ([]float32, error) {
	return MLXReconstruct2D(st, name, out, in)
}

func (MLXReconstructor) RowReader2D(st *embed.SafetensorsFile, name string, out, in int) (func(r int, dst []float32) error, error) {
	return MLXRowReader2D(st, name, out, in)
}
