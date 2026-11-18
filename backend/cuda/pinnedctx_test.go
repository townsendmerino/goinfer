//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/runtime/modelload"
)

// TestPinnedCtx_wontFitStopsTheLoad: a -ctx the operator pinned whose KV does not fit the card stops the load through
// modelload.Load, serve's path, with the build's refusal (it names the context and -ctx); it does not continue on the CPU. The
// KV fit's free-VRAM reading (cudaFreeVRAM) is held at nothing, so the weights still fit and only the KV is refused. An
// unpinned load under the same reading keeps its historical decline and runs on the CPU, and with the real reading an unpinned
// load goes resident at the context its plan chooses.
func TestPinnedCtx_wontFitStopsTheLoad(t *testing.T) {
	const path = "../../testdata/gemma1-tiny.gguf"
	requireDeviceAndFixtureFile(t, path)
	load := func(ctx int) (*decoder.Model, error) {
		t.Helper()
		opts := decoder.Options{Backend: "cuda", Quant: "int4", ResidentContext: ctx}
		res, err := modelload.Load(context.Background(), modelload.Request{Spec: path, Opts: opts, DirectLoad: true, ExplicitQuant: "int4"})
		if err != nil {
			return nil, err
		}
		t.Cleanup(func() { res.Model.Close() })
		return res.Model, nil
	}

	m, err := load(0)
	if err != nil {
		t.Fatal(err)
	}
	if !m.ResidentActive() || m.ResidentContextCap() <= 0 {
		t.Fatalf("unpinned, real free VRAM: not resident (%s), or no context; the cases below need a model CUDA builds", m.DecodePath())
	}

	saved := cudaFreeVRAM
	cudaFreeVRAM = func(*cudaResident) (uint64, error) { return 0, nil }
	t.Cleanup(func() { cudaFreeVRAM = saved })

	m, err = load(0)
	if err != nil {
		t.Fatalf("unpinned, no free VRAM for the KV: the load failed (%v); want the historical decline to the CPU", err)
	}
	if m.ResidentActive() {
		t.Fatalf("unpinned, no free VRAM for the KV: resident; want the decline")
	}

	m, err = load(64)
	if err == nil {
		t.Fatalf("-ctx 64 pinned, no free VRAM for the KV: loaded on %s; want the load stopped with the build's refusal", m.DecodePath())
	}
	if !errors.Is(err, decoder.ErrPinnedContextWontFit) {
		t.Errorf("the refusal %v does not match decoder.ErrPinnedContextWontFit", err)
	}
	for _, want := range []string{"64 positions", "-ctx"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the refusal %q does not name %q", err, want)
		}
	}
}
