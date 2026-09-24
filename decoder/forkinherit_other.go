//go:build !darwin

package decoder

// excludeFromFork is a no-op off darwin: the fork-copies-a-wired-private-mapping hazard it closes
// (forkinherit_darwin.go) is XNU's, and no non-darwin backend wires the .giw mapping.
func excludeFromFork(b []byte) error { return nil }
