//go:build prequant

package chatapp

import (
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// R6: the release's embedded-tier chat binaries are baked at a FIXED quant chosen at build time (cmd/prequant's own
// default) and never read the --quant flag, while --help's shared --quant text says "Default int4" regardless of build.
// This is the "help text default equals runtime default" gate, taken on the --version line instead of by rewriting the
// shared flag text: it must state whatever was actually baked in, not the --model flag's unrelated default.
//
// Requires -tags prequant AND internal/chatapp/model.giw staged (build-embed.sh's own build input, gitignored, not
// committed): that is why this file is build-tag gated rather than part of the always-built version_test.go, since a
// plain `go test ./...` must not need a 600+ MB local asset. Origin:
// docs/code-notes/internal-chatapp.md#TestEmbedBuild_versionReportsWhatWasActuallyBaked.
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
		// Guards against silently falling back to the unrelated --quant FLAG's default ("int4") when the embed build ignores
		// that flag entirely: the --version line must state what was baked in.
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

// TestLoadEmbedded_rejectsLoRAInsteadOfSilentlyIgnoringIt pins that loadEmbedded rejects --lora: decoder.NewModel (what
// loadEmbedded's prequant path calls, unlike loadFromPath's decoder.Load) takes no *decoder.Options, so nothing could
// merge opts.LoRA into a .giw bundle's already-serialized weights, and ignoring it would be silent. It drives
// loadEmbedded directly with a staged model.giw present (see the file-level reason that asset is required under -tags
// prequant), and relies on the LoRA check firing BEFORE giw.Read, so the stub's bytes are never parsed as a bundle.
// Origin (N-78): docs/code-notes/internal-chatapp.md#TestLoadEmbedded_rejectsLoRAInsteadOfSilentlyIgnoringIt.
func TestLoadEmbedded_rejectsLoRAInsteadOfSilentlyIgnoringIt(t *testing.T) {
	_, err := loadEmbedded(false, decoder.Options{LoRA: "/some/adapter/dir"})
	if err == nil {
		t.Fatal("loadEmbedded(opts.LoRA set) returned no error — a prequant build has no way to apply it and used to silently ignore it")
	}
	if !strings.Contains(err.Error(), "lora") && !strings.Contains(err.Error(), "LoRA") {
		t.Errorf("error should name LoRA as the reason, got: %v", err)
	}
}
