package gatecore

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// HeavyConfig builds the matrix configuration for `gate heavy` (real-checkpoint pure-Go tests).
func HeavyConfig() *GateConfig {
	models := env("GOINFER_GATE_MODELS", filepath.Join(Home(), "models"))
	run := env("GOINFER_HEAVY_RUN", "")
	timeout := env("GOINFER_HEAVY_TIMEOUT", "120m")
	pkgs := strings.Fields(env("GOINFER_HEAVY_PKGS", "./decoder/ ./internal/app/serveapp/"))

	cells := make([]Cell, 0, len(pkgs))
	for _, p := range pkgs {
		cells = append(cells, Cell{
			Name:    p,
			Pkgs:    []string{p},
			Tags:    []string{"goinfer_testhooks", "realckpt"},
			Run:     run,
			Timeout: timeout,
			Serial:  true,
			Env: map[string]string{
				"GOINFER_HEAVY_TESTS": "1",
				"GOINFER_MODELS_DIR":  models,
				"GOINFER_MODELS":      models,
			},
		})
	}

	return &GateConfig{
		Name:             "heavy",
		Desc:             "the pure-Go HEAVY tier: real-checkpoint tests `go test ./...` never executes",
		Cells:            cells,
		Decision:         "tally",
		TopLevelOnly:     true,
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

// CensusConfig builds the matrix configuration for `gate census`.
func CensusConfig(passthrough []string) *GateConfig {
	c := Cell{
		Name: "./...",
		Env:  map[string]string{"CGO_ENABLED": env("CGO_ENABLED", "0")},
	}
	if len(passthrough) > 0 {
		c.Name = strings.Join(passthrough, " ")
		c.Extra = passthrough
	} else {
		c.Tags = []string{"goinfer_testhooks"}
		c.Pkgs = []string{"./..."}
	}
	return &GateConfig{
		Name:             "census",
		Desc:             "the release-ritual test census — PASS/SKIP/FAIL with SKIPs bucketed by reason",
		Cells:            []Cell{c},
		Decision:         "census",
		TopLevelOnly:     false,
		ZeroPolicy:       "no-tests",
		RCIsFailure:      false,
		PkgFailIsFailure: false,
	}
}

// Env returns os.Getenv(k) when non-empty, else def.
func Env(k, def string) string { return env(k, def) }

func env(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

// Home returns os.UserHomeDir, falling back to "." when unavailable.
func Home() string {
	h, err := os.UserHomeDir()
	if err != nil {
		return "."
	}
	return h
}
