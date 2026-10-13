package decoder

import "github.com/townsendmerino/goinfer/internal/f16"

// F16Bits converts an f32 to the IEEE binary16 bit pattern the Metal W4A8 kernels read their group scales in. It is
// f16.Bits, the one definition of that conversion; see there for the rounding contract.
func F16Bits(f float32) uint16 { return f16.Bits(f) }
