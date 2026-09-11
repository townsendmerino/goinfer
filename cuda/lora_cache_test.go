//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestLoRAAdapterCacheCUDA gates audit P-10's adapter device cache through the real resident:
// SetAdapter reuses the last-bound adapter's buffers instead of re-uploading them, and must never
// hand one adapter's buffers to another. Every step drives SetAdapter with a FRESH
// []ResidentAdapterLayer, exactly as generateInto does per request, so a cache keyed on the slice
// it was handed would miss every time and one keyed on nothing would serve the wrong adapter.
func TestLoRAAdapterCacheCUDA(t *testing.T) {
	const ckpt = "../testdata/llama-tiny"
	requireDeviceAndFixture(t, ckpt)
	dirA, dirB := buildLlamaTinyLoRAFixtureCUDA(t), buildLlamaTinyLoRAFixtureCUDASeeded(t, 1000)

	mRes, err := decoder.Load(ckpt, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load cuda: %v", err)
	}
	defer mRes.Close()
	rf := mRes.ResidentForwardForTest()
	cr, ok := rf.(*cudaResident)
	if !ok {
		t.Fatalf("resident is %T, not *cudaResident", rf)
	}
	mCPU, err := decoder.Load(ckpt, decoder.Options{Backend: "cpu"})
	if err != nil {
		t.Fatalf("load cpu: %v", err)
	}
	defer mCPU.Close()
	for _, m := range []*decoder.Model{mRes, mCPU} {
		if err := m.LoadAdapter("a", dirA); err != nil {
			t.Fatalf("LoadAdapter a: %v", err)
		}
		if err := m.LoadAdapter("b", dirB); err != nil {
			t.Fatalf("LoadAdapter b: %v", err)
		}
	}
	_, _, _, _, _, _, vocab := mCPU.Dims()
	prompt := make([]int, 8)
	for i := range prompt {
		prompt[i] = (i*47 + 5) % vocab
	}
	want := func(name string) []float32 {
		w, err := mCPU.PrefillLogitsWithAdapterForTest(context.Background(), prompt, name)
		if err != nil {
			t.Fatalf("cpu reference %s: %v", name, err)
		}
		return w
	}
	wantA, wantB := want("a"), want("b")
	run := func() []float32 {
		t.Helper()
		rf.Reset()
		var last []float32
		for i, tok := range prompt {
			lr, err := rf.Forward(mRes.EmbedResidentForTest(tok), i)
			if err != nil {
				t.Fatalf("resident forward[%d]: %v", i, err)
			}
			last = lr
		}
		return append([]float32(nil), last...)
	}
	bind := func(name string) {
		t.Helper()
		var layers []decoder.ResidentAdapterLayer
		if name != "" {
			layers = mRes.ResidentAdapterLayersForTest(name) // a fresh slice every call, like generateInto
		}
		if err := cr.SetAdapter(layers); err != nil {
			t.Fatalf("SetAdapter(%q): %v", name, err)
		}
	}
	stats := func(step string, wantUp, wantHit uint64) {
		t.Helper()
		if u, h := cr.LoraCacheStatsForTest(); u != wantUp || h != wantHit {
			t.Errorf("%s: uploads=%d hits=%d, want uploads=%d hits=%d", step, u, h, wantUp, wantHit)
		}
	}
	same := func(step string, x, y []float32) {
		t.Helper()
		if _, maxAbs := cosF32(x, y); maxAbs > 1e-5 {
			t.Errorf("%s: maxAbs %.3g — not the same computation", step, maxAbs)
		}
	}
	near := func(step string, ref, got []float32) {
		t.Helper()
		if cos, _ := cosF32(ref, got); cos < 0.95 {
			t.Errorf("%s: cosine %.6f vs its CPU reference < 0.95", step, cos)
		}
	}

	base := run()
	bind("a")
	stats("bind a", 1, 0)
	gotA := run()
	near("adapter a", wantA, gotA)

	bind("")
	same("clear after a -> base (clearing must UNBIND even though the buffers stay cached)", base, run())

	bind("a")
	stats("rebind a through a fresh slice", 1, 1)
	same("rebind a -> the cached buffers compute exactly what the first upload did", gotA, run())

	bind("b")
	stats("bind b (a different adapter)", 2, 1)
	gotB := run()
	near("adapter b — the step a cache keyed on nothing gets wrong, serving a's buffers", wantB, gotB)
	if cos, _ := cosF32(gotA, gotB); cos > 0.999999 {
		t.Error("adapters a and b produced indistinguishable logits — the harness cannot tell them apart")
	}

	bind("a")
	stats("bind a after b (single-entry cache: b evicted a)", 3, 1)
	same("a after b", gotA, run())

	t.Setenv("GOINFER_NO_LORA_CACHE", "1")
	bind("")
	bind("a")
	stats("GOINFER_NO_LORA_CACHE=1: every bind uploads", 4, 1)
	bind("")
	same("clear with the cache disabled", base, run())
}
