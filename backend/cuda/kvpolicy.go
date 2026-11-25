//go:build cuda

package cuda

import "sync/atomic"

// kvF16Auto is whether "auto" means f16 on CUDA for a covered family. Registered in register.go;
// CUDA tests pin it false in init so exact-cache contract tests keep their f32 cache.
var kvF16Auto atomic.Bool
