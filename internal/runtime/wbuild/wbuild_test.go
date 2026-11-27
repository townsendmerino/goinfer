package wbuild

import (
	"errors"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

func TestMatmulQuantAndEmbeddingPolicy(t *testing.T) {
	if got := MatmulQuant(QuantInt4Mix, "blk.0.ffn_gate.weight"); got != QuantInt4 {
		t.Fatalf("int4mix ffn_gate = %v, want QuantInt4", got)
	}
	if got := MatmulQuant(QuantInt4Mix, "blk.0.ffn_gate_inp.weight"); got != QuantInt8 {
		t.Fatalf("int4mix router = %v, want QuantInt8", got)
	}
	if got := MatmulQuant(QuantInt4Mix, "blk.0.attn_q.weight"); got != QuantInt8 {
		t.Fatalf("int4mix attn_q = %v, want QuantInt8", got)
	}
	if got := MatmulQuant(QuantQ4K, "blk.0.ffn_gate_inp.weight"); got != QuantInt8I8 {
		t.Fatalf("q4k router = %v, want QuantInt8I8", got)
	}
	if got := QuantInt4.Embedding(); got != QuantInt8I8 {
		t.Fatalf("QuantInt4.Embedding() = %v, want QuantInt8I8", got)
	}
	if got := QuantInt4.EmbeddingWith(true); got != QuantInt4 {
		t.Fatalf("QuantInt4.EmbeddingWith(true) = %v, want QuantInt4", got)
	}
	if got := QuantInt4.GatherTable(); got != QuantInt4 {
		t.Fatalf("QuantInt4.GatherTable() = %v, want QuantInt4", got)
	}
	if !QuantBF16.Half16() || !QuantF16.Half16() || QuantInt4.Half16() {
		t.Fatalf("Half16 mismatch")
	}
}

func TestWMBytesAndQuantize(t *testing.T) {
	f32 := make([]float32, 64*64)
	for i := range f32 {
		f32[i] = float32(i%17-8) * 0.1
	}
	wm := linalg.WrapF32(f32, 64, 64)
	if got := WMBytes(&wm); got != 4*64*64 {
		t.Fatalf("f32 WMBytes = %d, want %d", got, 4*64*64)
	}
	var h Hooks
	q8 := h.QuantizeWM(wm, QuantInt8I8)
	if WMBytes(&q8) <= 0 || WMBytes(&q8) >= WMBytes(&wm) {
		t.Fatalf("int8 WMBytes = %d, want (0, %d)", WMBytes(&q8), WMBytes(&wm))
	}
	q4 := h.QuantizeWM(wm, QuantInt4)
	if !q4.IsInt4() || WMBytes(&q4) <= 0 {
		t.Fatalf("int4 quantize failed: IsInt4=%v bytes=%d", q4.IsInt4(), WMBytes(&q4))
	}
}

func TestFakeQuantInt4Schemes(t *testing.T) {
	w := make([]float32, 4*32)
	for i := range w {
		w[i] = float32(i%13-6) * 0.25
	}
	for _, scheme := range []string{"sym", "symmse", "affine"} {
		out := FakeQuantInt4(scheme, w, 4, 32, Int4GroupSize)
		if len(out) != len(w) {
			t.Fatalf("%s: len=%d, want %d", scheme, len(out), len(w))
		}
	}
}

func TestParallelLayersAbortAndRealError(t *testing.T) {
	closed := make(chan struct{})
	close(closed)
	if err := ParallelLayers(4, closed, func(int) error { return nil }); !errors.Is(err, ErrLoadAborted) {
		t.Fatalf("aborted ParallelLayers = %v, want ErrLoadAborted", err)
	}
	sentinel := errors.New("boom")
	if err := ParallelLayers(4, nil, func(i int) error {
		if i == 2 {
			return sentinel
		}
		return nil
	}); !errors.Is(err, sentinel) {
		t.Fatalf("failing ParallelLayers = %v, want boom", err)
	}
}

func TestReorderAndUntileVHeadsRoundTrip(t *testing.T) {
	src := make([]float32, 2*3*2*4)
	for i := range src {
		src[i] = float32(i + 1)
	}
	tiled := ReorderVHeads(src, 1, 2, 3, 2, 4)
	back := UntileVHeads(tiled, 1, 2, 3, 2, 4)
	for i := range src {
		if back[i] != src[i] {
			t.Fatalf("round-trip mismatch at %d: got %v, want %v", i, back[i], src[i])
		}
	}
}

func TestIsBatchedProjTensor(t *testing.T) {
	if !IsBatchedProjTensor("blk.0.attn_q.weight") || !IsBatchedProjTensor("blk.3.ffn_gate.weight") {
		t.Fatalf("expected standard projections to match")
	}
	if IsBatchedProjTensor("blk.0.ffn_gate_exps.weight") || IsBatchedProjTensor("blk.0.ffn_gate_inp.weight") {
		t.Fatalf("experts/router must not match batched projections")
	}
}
