// Package giwcodec implements the .giw binary weight-bundle format: writing, streaming,
// reading, CRC verification, shape validation, island bf16/f32 helpers, and target-specific
// int4 row4 emission.
//
// It sits below decoder (rule giwcodec-leaf in internal/layout/imports.json), depending only
// on decoder/arch, decoder/config, internal/runtime/archparts, internal/runtime/families,
// and internal/runtime/program. decoder re-exports the public .giw entry points and types.
package giwcodec
