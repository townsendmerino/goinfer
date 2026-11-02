package decoder_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/modelload"
)

// TestLoad_refusedQ4KIsNotRetriedWithStreaming goes through modelload.Load, which reads the refusal's DenseStreamable: a
// q4k .gguf the fit guard refuses keeps that refusal, because weight paging does not read a q4k .giw's kind. The same file
// at int8, the do-nothing arm, still takes the retry, so the q4k result is the quant's doing. -direct-load reaches the
// refusal on the .gguf itself; the default sidecar route meets its own check on the bundle first.
func TestLoad_refusedQ4KIsNotRetriedWithStreaming(t *testing.T) {
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	const retry = "auto weight-streaming retry"
	load := func(t *testing.T, quant string) error {
		src := decoder.Q4KTinyGGUFForTest(t, t.TempDir())
		res, err := modelload.Load(context.Background(), modelload.Request{Spec: src, DirectLoad: true, ExplicitQuant: quant,
			Opts: decoder.Options{Backend: "cpu", Quant: quant}})
		if err == nil {
			res.Model.Close()
		}
		return err
	}
	decoder.HostRAMAvailableForTest(t, 1)

	err := load(t, "q4k")
	var fde *decoder.FitDeclineError
	if !errors.As(err, &fde) {
		t.Fatalf("q4k: %v, want the fit guard's refusal", err)
	}
	if strings.Contains(err.Error(), retry) {
		t.Errorf("q4k took the retry: %v", err)
	}
	if strings.Contains(err.Error(), "Re-run goinfer-serve with -stream-weights") {
		t.Errorf("q4k's refusal recommends what routeFor refuses:\n%v", err)
	}

	if err := load(t, "int8"); err == nil || !strings.Contains(err.Error(), retry) {
		t.Errorf("int8, the same file: %v, want the retry taken", err)
	}
}
