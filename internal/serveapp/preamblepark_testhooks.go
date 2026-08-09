//go:build goinfer_testhooks

package serveapp

// preamblePark is the settable form under -tags goinfer_testhooks (CI runs with this tag). Tests set
// it to block a request inside the liveness window; default is a no-op. See preamblepark.go for why
// the seam lives here and not as a branch in production code.
var preamblePark = func() {}
