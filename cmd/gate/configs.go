package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// The committed matrix configs. One per migrated script — this is where "six scripts are one
// program" becomes literal: each former script is a value, not a file.

func heavyConfig() *gateConfig {
	models := env("GOINFER_GATE_MODELS", filepath.Join(home(), "models"))
	run := env("GOINFER_HEAVY_RUN", "")
	timeout := env("GOINFER_HEAVY_TIMEOUT", "120m")
	pkgs := strings.Fields(env("GOINFER_HEAVY_PKGS", "./decoder/ ./internal/serveapp/"))

	cells := make([]cell, 0, len(pkgs))
	for _, p := range pkgs {
		cells = append(cells, cell{
			Name: p,
			Pkgs: []string{p},
			// TWO GATING LAYERS, both handled here. `//go:build realckpt` files do not COMPILE
			// without the tag, so plain `go test ./...` (and CI) never even builds them — the
			// strongest form of "run by nothing". The tag makes them compile; the env below makes
			// the requireHeavyModel ones actually run instead of skipping.
			Tags:    []string{"goinfer_testhooks", "realckpt"},
			Run:     run,
			Timeout: timeout,
			// Sequential. Each heavy test loads a multi-GB checkpoint; parallel packages contend
			// for RAM and fail as spurious numerics or an OOM kill rather than a clean result.
			Serial: true,
			Env: map[string]string{
				"GOINFER_HEAVY_TESTS": "1",
				"GOINFER_MODELS_DIR":  models,
			},
		})
	}

	return &gateConfig{
		Name:         "heavy",
		Desc:         "the pure-Go HEAVY tier: real-checkpoint tests `go test ./...` never executes",
		Cells:        cells,
		Decision:     "tally",
		TopLevelOnly: true,
		// A skip is not a pass, and a run of zero tests is not a green run: with
		// GOINFER_HEAVY_TESTS=1 set a heavy test STILL skips if its specific checkpoint is absent,
		// so green here means "ran and passed" and nothing else.
		ZeroPolicy:       "no-pass",
		RCIsFailure:      true,
		PkgFailIsFailure: true,
		Precondition: func() (string, bool) {
			if st, err := os.Stat(models); err != nil || !st.IsDir() {
				return fmt.Sprintf("models dir %s missing — nothing to run against; refusing to report a verdict", models), false
			}
			return "", true
		},
	}
}

// censusConfig is skip_census.py: PASS/SKIP/FAIL over the whole tree with every SKIP bucketed by
// why. passthrough replaces the default cell entirely (`gate census -- -tags cuda ./cuda/`).
func censusConfig(passthrough []string) *gateConfig {
	c := cell{
		Name: "./...",
		// CGO_ENABLED=0 because the census describes the PURE-GO suite: a census whose numbers
		// depend on whether the box happened to have a C toolchain is not a census.
		Env: map[string]string{"CGO_ENABLED": env("CGO_ENABLED", "0")},
	}
	if len(passthrough) > 0 {
		// Verbatim passthrough: the caller is spelling out the whole matrix cell. It should carry
		// goinfer_testhooks itself if it wants the relocated hooks' tests to run rather than skip.
		c.Name = strings.Join(passthrough, " ")
		c.Extra = passthrough
	} else {
		c.Tags = []string{"goinfer_testhooks"}
		c.Pkgs = []string{"./..."}
	}
	return &gateConfig{
		Name:     "census",
		Desc:     "the release-ritual test census — PASS/SKIP/FAIL with SKIPs bucketed by reason",
		Cells:    []cell{c},
		Decision: "census",
		// Subtests count. skip_census.py keyed on (Package, Test) straight out of the JSON, which
		// includes them; heavy_gate did not. See gateConfig.TopLevelOnly.
		TopLevelOnly:     false,
		ZeroPolicy:       "no-tests",
		RCIsFailure:      false,
		PkgFailIsFailure: false,
	}
}

func env(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func home() string {
	h, err := os.UserHomeDir()
	if err != nil {
		return "."
	}
	return h
}
