package decoder_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/modelload"
)

// TestLoad_retriesWithStreamingOnlyAtAQuantThatPages goes through modelload.Load, which reads the refusal's
// DenseStreamable: a .gguf the fit guard refuses at a quant weight paging cannot read keeps that refusal, and at one it can
// read takes the retry, so the result is the quant's doing (mem.QuantPages; the pager's side is
// TestStreamedGIW_pagesOnlyTheQuantsTheFitGuardOffers). -direct-load reaches the refusal on the .gguf itself; the default
// sidecar route meets its own check on the bundle first.
func TestLoad_retriesWithStreamingOnlyAtAQuantThatPages(t *testing.T) {
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	const retry = "auto weight-streaming retry"
	decoder.HostRAMAvailableForTest(t, 1)
	for _, c := range []struct {
		quant string
		pages bool
	}{{"q4k", false}, {"f32", false}, {"int8", true}, {"int8int8", true}, {"int4", true}, {"int4mix", true}} {
		t.Run(c.quant, func(t *testing.T) {
			src := decoder.Q4KTinyGGUFForTest(t, t.TempDir())
			res, err := modelload.Load(context.Background(), modelload.Request{Spec: src, DirectLoad: true, ExplicitQuant: c.quant,
				Opts: decoder.Options{Backend: "cpu", Quant: c.quant}})
			if err == nil {
				res.Model.Close()
				t.Fatal("loaded under a refusal")
			}
			if got := strings.Contains(err.Error(), retry); got != c.pages {
				t.Errorf("retry taken = %v, want %v: %v", got, c.pages, err)
			}
			var fde *decoder.FitDeclineError
			if !c.pages && !errors.As(err, &fde) {
				t.Errorf("%v, want the fit guard's refusal", err)
			}
			if !c.pages && strings.Contains(err.Error(), "Re-run goinfer-serve with -stream-weights") {
				t.Errorf("the refusal recommends what cannot help:\n%v", err)
			}
		})
	}
}
