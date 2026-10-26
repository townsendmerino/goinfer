// Package kernels holds the CUDA kernel-level tests: each loads a PTX module the cuda package ships, launches its
// kernels on small inputs and checks them against a CPU reference. It has no code of its own and imports nothing
// from goinfer, so a decoder edit does not invalidate its test cache (internal/layout's
// TestBackendTests_decoderImportsOnlyFall counts the tests left behind).
//
// A test belongs here when it needs only the PTX, the device and the helpers in this directory. One that needs the
// resident, a loaded model or a production Go helper of package cuda stays there.
//
// The PTX is read from ../testdata, the same files cuda/kernels.go embeds.
//
//	go test -tags cuda ./cuda/kernels/
package kernels
