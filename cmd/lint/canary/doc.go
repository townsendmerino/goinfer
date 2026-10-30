// Package canary is cmd/lint's proof that staticcheck is analysing. Built with the goinfer_staticcheck_canary tag it
// holds a struct with an unused field in a file that imports a standard-library package, which a working staticcheck
// reports as "field deadField is unused (U1000)". The import is the point: a canary that imports nothing reads no export
// data, so it passes on a staticcheck that cannot read the toolchain's. Without the tag the package is empty.
package canary
