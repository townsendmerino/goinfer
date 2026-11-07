package decoder

import (
	"io/fs"

	"github.com/townsendmerino/goinfer/decoder/config"
)

// Config is a checkpoint's config.json, parsed. The type, its validators and its loader live in decoder/config, which
// imports no goinfer package; the name stays here as the published API (docs/tasks/task-decoder-config-2026-11.md §5).
type Config = config.Config

// RecommendedSampling is a checkpoint's own recommended sampling settings (config.RecommendedSampling).
type RecommendedSampling = config.RecommendedSampling

// loadConfig reads and parses config.json from fsys, and the recommended sampling beside it (config.Load).
func loadConfig(fsys fs.FS, name string) (*Config, error) { return config.Load(fsys, name) }

// resolveEOSIDs returns the ids that end generation for the checkpoint in dir (config.ResolveEOSIDs).
func resolveEOSIDs(dir string, cfg *Config) []int { return config.ResolveEOSIDs(dir, cfg) }

// lenientConfigJSON makes a model config file's Python non-finite tokens parse (config.LenientJSON).
func lenientConfigJSON(b []byte) []byte { return config.LenientJSON(b) }
