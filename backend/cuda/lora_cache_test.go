//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestLoRAAdapterCacheCUDA gates audit P-10's adapter device cache through the real resident:
// SetAdapter reuses the last-bound adapter's buffers instead of re-uploading them, and must never
// hand one adapter's buffers to another. Every step drives SetAdapter with a FRESH
// []ResidentAdapterLayer, exactly as generateInto does per request, so a cache keyed on the slice
// it was handed would miss every time and one keyed on nothing would serve the wrong adapter.
func TestLoRAAdapterCacheCUDA(t *testing.T) {
	const ckpt = "../../testdata/llama-tiny"
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

	decoder.SetKnobEnvForTest(t, mRes, "GOINFER_NO_LORA_CACHE", "1")
	bind("")
	bind("a")
	stats("GOINFER_NO_LORA_CACHE=1: every bind uploads", 4, 1)
	bind("")
	same("clear with the cache disabled", base, run())
}

// TestLoRAUploadOOM_returnsEveryBuffer — a device allocation that fails partway through an adapter upload (at every buffer in turn)
// leaves the device's allocation ledger where it was before the bind, no adapter bound and no half-built cache, and the next bind
// of the same adapter works.
func TestLoRAUploadOOM_returnsEveryBuffer(t *testing.T) {
	const ckpt = "../../testdata/llama-tiny"
	requireDeviceAndFixture(t, ckpt)
	dir := buildLlamaTinyLoRAFixtureCUDA(t)
	m, err := decoder.Load(ckpt, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load cuda: %v", err)
	}
	defer m.Close()
	if err := m.LoadAdapter("a", dir); err != nil {
		t.Fatalf("LoadAdapter: %v", err)
	}
	cr, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatalf("resident is %T, not *cudaResident", m.ResidentForwardForTest())
	}
	ledger := func() int { a, _ := cr.dev.LedgerLen(); return a }
	before := ledger()

	// A successful bind sizes the upload and proves the failure points below are inside it.
	if err := cr.SetAdapter(m.ResidentAdapterLayersForTest("a")); err != nil {
		t.Fatalf("SetAdapter: %v", err)
	}
	total := ledger() - before
	if total < 4 {
		t.Fatalf("a bind allocates %d buffers; want a few, so a failure can land after some succeeded", total)
	}
	if err := cr.SetAdapter(nil); err != nil {
		t.Fatal(err)
	}
	cr.doOn("", func() error { cr.releaseLoraLayers(cr.lora.cache); cr.lora.cache, cr.lora.key = nil, nil; return nil })
	if got := ledger(); got != before {
		t.Fatalf("releasing the cached adapter left %d allocations, want %d", got, before)
	}

	defer func() { loraAllocFailAfterForTest = 0 }()
	for n := 1; n <= total; n++ {
		loraAllocFailAfterForTest = n
		if err := cr.SetAdapter(m.ResidentAdapterLayersForTest("a")); err == nil {
			t.Fatalf("failing allocation %d of %d: SetAdapter returned no error", n, total)
		}
		loraAllocFailAfterForTest = 0
		if got := ledger(); got != before {
			t.Errorf("failing allocation %d of %d: %d device allocations remain, want %d (leaked %d)", n, total, got, before, got-before)
		}
		if cr.loraLayers != nil || cr.lora.cache != nil {
			t.Errorf("failing allocation %d of %d: an adapter or cache is still bound", n, total)
		}
	}
	if err := cr.SetAdapter(m.ResidentAdapterLayersForTest("a")); err != nil {
		t.Fatalf("a bind after the failures: %v", err)
	}
	if got := ledger() - before; got != total {
		t.Errorf("the bind after the failures allocated %d buffers, want %d", got, total)
	}
}

// loraOOMRig loads llama-tiny on the card with the adapters "a" and "b" registered, and returns the resident, the model and a
// ledger-count reader.
func loraOOMRig(t *testing.T) (*decoder.Model, *cudaResident, func() int) {
	t.Helper()
	const ckpt = "../../testdata/llama-tiny"
	requireDeviceAndFixture(t, ckpt)
	m, err := decoder.Load(ckpt, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load cuda: %v", err)
	}
	t.Cleanup(func() { _ = m.Close() })
	for name, dir := range map[string]string{"a": buildLlamaTinyLoRAFixtureCUDA(t), "b": buildLlamaTinyLoRAFixtureCUDASeeded(t, 1000)} {
		if err := m.LoadAdapter(name, dir); err != nil {
			t.Fatalf("LoadAdapter %s: %v", name, err)
		}
	}
	cr, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatalf("resident is %T, not *cudaResident", m.ResidentForwardForTest())
	}
	return m, cr, func() int { a, _ := cr.dev.LedgerLen(); return a }
}

// TestLoRAUploadOOM_afterAnAdapterIsBound: switching from a bound adapter A to B, a B upload that fails partway leaves nothing bound
// (not A, whose buffers the switch returned first) and the ledger at the level before any adapter was bound, at every failure point.
func TestLoRAUploadOOM_afterAnAdapterIsBound(t *testing.T) {
	m, cr, ledger := loraOOMRig(t)
	before := ledger()
	if err := cr.SetAdapter(m.ResidentAdapterLayersForTest("a")); err != nil {
		t.Fatal(err)
	}
	total := ledger() - before
	if total < 4 {
		t.Fatalf("a bind allocates %d buffers", total)
	}
	defer func() { loraAllocFailAfterForTest = 0 }()
	for n := 1; n <= total; n++ {
		if err := cr.SetAdapter(m.ResidentAdapterLayersForTest("a")); err != nil { // A is bound again and cached
			t.Fatalf("rebind a: %v", err)
		}
		if cr.loraLayers == nil {
			t.Fatal("a is not bound before the switch")
		}
		loraAllocFailAfterForTest = n
		if err := cr.SetAdapter(m.ResidentAdapterLayersForTest("b")); err == nil {
			t.Fatalf("switch to b failing at allocation %d of %d: no error", n, total)
		}
		loraAllocFailAfterForTest = 0
		if cr.loraLayers != nil || cr.lora.cache != nil {
			t.Errorf("switch to b failing at %d of %d: an adapter or cache is still bound (loraLayers nil: %v)", n, total, cr.loraLayers == nil)
		}
		if got := ledger(); got != before {
			t.Errorf("switch to b failing at %d of %d: %d device allocations remain, want %d", n, total, got, before)
		}
	}
}

// TestLoRAUploadOOM_anyPanicReturnsTheBuffers: a panic that is not a device allocation failure, partway through an upload, still returns
// every buffer the upload had allocated; the job reports it as an error.
func TestLoRAUploadOOM_anyPanicReturnsTheBuffers(t *testing.T) {
	m, cr, ledger := loraOOMRig(t)
	before := ledger()
	defer func() { loraPanicAfterForTest = 0 }()
	for _, n := range []int{2, 5, 20} {
		loraPanicAfterForTest = n
		err := cr.SetAdapter(m.ResidentAdapterLayersForTest("a"))
		loraPanicAfterForTest = 0
		if err == nil || !strings.Contains(err.Error(), "test seam panic") {
			t.Fatalf("panic at allocation %d: err = %v, want the job's error naming the panic", n, err)
		}
		if got := ledger(); got != before {
			t.Errorf("panic at allocation %d: %d device allocations remain, want %d (leaked %d)", n, got, before, got-before)
		}
		if cr.loraLayers != nil || cr.lora.cache != nil {
			t.Errorf("panic at allocation %d: an adapter or cache is still bound", n)
		}
	}
}
