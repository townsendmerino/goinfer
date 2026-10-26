// Package layout holds the repository's package-layout checks as tests. It has no code of its own.
//
// TestBackendTests_decoderImportsOnlyFall is a ratchet on how many test files of the cuda, metal and gpu modules
// depend on decoder, counted two ways: direct (the file's own imports name decoder) and reach (the file's package's
// test build reaches decoder through any chain of goinfer imports, so a decoder edit invalidates it). Either count may
// only fall, and a fall is recorded by lowering the committed figure. Imports are read by parsing, with build tags
// ignored, so it runs in milliseconds. Its -v output is the census.
//
//	go test -v -run TestBackendTests_decoderImportsOnlyFall ./internal/layout/
//
// The plan this serves is docs/tasks/task-subsystems-2026-10.md, SB1 and SB13.
package layout
