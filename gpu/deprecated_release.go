//go:build gpu

package gpu

import "io"

// Compile-time proof that the backend implements io.Closer; the GPU resource types' own checks and their deprecated
// Release aliases are in release_aliases.go.
var _ io.Closer = (*webgpuBackend)(nil)
