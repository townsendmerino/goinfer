// Package gpudev is aikit's cgo-free native-GPU device substrate: Metal on darwin
// (metal.go) and CUDA on linux (cuda.go), presenting the same Device / Buffer /
// Queue / Pipeline / Encoder vocabulary on both. On every other platform this
// package is empty and registers no ann.Backend, so ann.FlatI8 scores on the CPU.
//
// THIS FILE CARRIES THE PACKAGE COMMENT AND HAS NO BUILD TAG, deliberately. It used
// to live in doc_other.go, which is tagged `!darwin && !linux` — so on the two
// platforms anyone actually builds for, the package doc was whichever of cuda.go or
// metal.go happened to be compiled, and both open with a file-level explainer rather
// than "Package gpu ...". staticcheck's ST1000 says so on both, and nobody saw it
// because golangci-lint had never been run against this module (2026-08-12).
package gpudev
