//go:build race

package decoder

// raceDetector reports whether the test binary runs under the race detector, whose shadow memory follows every
// allocation and is not returned to the OS with it.
const raceDetector = true
