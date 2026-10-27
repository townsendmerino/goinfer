//go:build cuda

package cuda

import "github.com/townsendmerino/goinfer/cuda/internal/wpack"

// f32ToF16RNE and f32SliceToF16RNE are the KV cache's f16 conversion, round to nearest even (package wpack).
func f32ToF16RNE(f float32) uint16                 { return wpack.F32ToF16RNE(f) }
func f32SliceToF16RNE(dst []uint16, src []float32) { wpack.F32SliceToF16RNE(dst, src) }
