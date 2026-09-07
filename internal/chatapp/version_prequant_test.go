//go:build prequant

package chatapp

import (
	"strings"
	"testing"
)

// R6 (docs/measurements/cold-user-2026-09-06-nobara-pc.md): the release's embedded-tier chat
// binaries are baked at a FIXED quant chosen at build time (cmd/prequant's own default), and
// never read the --quant flag at all — but --help's shared --quant text says "Default int4"
// regardless of build. This is the "help text default equals runtime default" gate, taken on
// the --version line rather than rewriting the shared flag text: it must state whatever was
// actually baked in, not the --model flag's unrelated default.
//
// Requires -tags prequant AND internal/chatapp/model.giw staged (build-embed.sh's own build
// input, gitignored, not committed) — that is why this file is build-tag gated rather than
// part of the always-built version_test.go: a plain `go test ./...` must not need a 600+ MB
// local asset to pass.
func TestEmbedBuild_versionReportsWhatWasActuallyBaked(t *testing.T) {
	if !hasEmbeddedModel {
		t.Fatal("hasEmbeddedModel is false under -tags prequant — the build tag wiring broke")
	}

	t.Run("unset ldflags name themselves as unset, not a wrong default", func(t *testing.T) {
		oldTier, oldQuant := embeddedTier, embeddedQuant
		embeddedTier, embeddedQuant = "", ""
		defer func() { embeddedTier, embeddedQuant = oldTier, oldQuant }()

		report := versionReport("goinfer-chat-test")
		if !strings.Contains(report, "embedded:") || !strings.Contains(report, "tier=(unset") || !strings.Contains(report, "quant=(unset") {
			t.Errorf("unset embed build-time vars must say so plainly, got:\n%s", report)
		}
		// The mutation this guards against: silently falling back to the unrelated --quant
		// FLAG's default ("int4") when the embed build actually ignores that flag entirely —
		// the exact discrepancy the cold-user run found (help said "Default int4", the
		// binary ran int8int8).
		if strings.Contains(report, "quant=int4 ") || strings.Contains(report, "quant=int4\n") {
			t.Errorf("unset embeddedQuant must not read as the unrelated --quant flag default:\n%s", report)
		}
	})

	t.Run("injected tier/quant are reported verbatim", func(t *testing.T) {
		oldTier, oldQuant := embeddedTier, embeddedQuant
		embeddedTier, embeddedQuant = "1.5b", "int8int8"
		defer func() { embeddedTier, embeddedQuant = oldTier, oldQuant }()

		report := versionReport("goinfer-chat-test")
		if !strings.Contains(report, "tier=1.5b") || !strings.Contains(report, "quant=int8int8") {
			t.Errorf("injected embeddedTier/embeddedQuant not reflected in --version output:\n%s", report)
		}
	})
}
