package cpuengine

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/decoder/config"
	"github.com/townsendmerino/goinfer/internal/runtime/families"
	"github.com/townsendmerino/goinfer/internal/runtime/wbuild"
)

type testCPUBackend struct{}

func (testCPUBackend) MatmulBT(a, b, dst []float32, M, K, N int) {
	linalg.MatmulBT(a, b, dst, M, K, N)
}

func loadTestEngine(t *testing.T, dir string, q wbuild.QuantMode) *Engine {
	t.Helper()
	var (
		w   *darch.Weights
		err error
	)
	if strings.HasSuffix(dir, ".gguf") {
		g, oerr := embed.OpenGGUFMmap(dir)
		if oerr != nil {
			t.Fatalf("open gguf %s: %v", dir, oerr)
		}
		t.Cleanup(func() { g.Close() })
		t.Skip("gguf fixture handled via decoder")
	} else {
		cfg, cerr := config.Load(os.DirFS(dir), "config.json")
		if cerr != nil {
			t.Fatalf("load config %s: %v", dir, cerr)
		}
		arch, schema, rerr := families.Resolve(cfg)
		if rerr != nil {
			t.Fatalf("resolve %s: %v", dir, rerr)
		}
		st, serr := embed.OpenSafetensorsMmap(filepath.Join(dir, "model.safetensors"))
		if serr != nil {
			t.Fatalf("open safetensors %s: %v", dir, serr)
		}
		t.Cleanup(func() { st.Close() })
		w, err = wbuild.BuildWeightsFromSafetensorsTo(cfg, arch, schema, st, q, false, true, false, nil, nil, "", nil, wbuild.Hooks{})
		if err != nil {
			t.Fatalf("build weights %s: %v", dir, err)
		}
	}
	prog := BuildProgram(w.Arch, w.HasExperts, nil)
	return &Engine{
		W:    w,
		Prog: prog,
		Be:   testCPUBackend{},
	}
}

func newTestCache(e *Engine, capHint int) *Cache {
	return NewCacheForArch(e.W.Arch, capHint, false, false, nil)
}

func TestEngine_ForwardAndBatchedIdentity(t *testing.T) {
	dir := filepath.Join("..", "..", "..", "testdata", "llama-tiny")
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
		t.Skipf("fixture missing: %s", dir)
	}
	e := loadTestEngine(t, dir, wbuild.QuantNone)
	ids := []int{1, 5, 9, 3, 7}

	// Sequential forward
	cSeq := newTestCache(e, len(ids)+4)
	var seqLogits []float32
	for _, id := range ids {
		out, err := e.Forward(id, cSeq)
		if err != nil {
			t.Fatalf("Forward: %v", err)
		}
		seqLogits = slices.Clone(out)
	}

	// Batched prefill
	cBat := newTestCache(e, len(ids)+4)
	batLogits, err := e.PrefillLogits(context.Background(), ids, cBat)
	if err != nil {
		t.Fatalf("PrefillLogits: %v", err)
	}
	if len(batLogits) != len(seqLogits) {
		t.Fatalf("logit len mismatch: %d vs %d", len(batLogits), len(seqLogits))
	}
	var maxDiff float64
	for i := range seqLogits {
		d := math.Abs(float64(seqLogits[i] - batLogits[i]))
		if d > maxDiff {
			maxDiff = d
		}
	}
	if maxDiff > 1e-4 {
		t.Fatalf("seq vs batched prefill max diff = %g, want <= 1e-4", maxDiff)
	}

	// DecodeMultiStep matches single-token Forward bit-for-bit
	nextIDs := []int{11, 13}
	c1, c2 := newTestCache(e, 16), newTestCache(e, 16)
	r1, r2 := newTestCache(e, 16), newTestCache(e, 16)
	_, _ = e.PrefillLogits(context.Background(), ids[:3], c1)
	_, _ = e.PrefillLogits(context.Background(), ids[:3], r1)
	_, _ = e.PrefillLogits(context.Background(), ids[:4], c2)
	_, _ = e.PrefillLogits(context.Background(), ids[:4], r2)
	got, err := e.DecodeMultiStep(nextIDs, []*Cache{c1, c2})
	if err != nil {
		t.Fatalf("DecodeMultiStep: %v", err)
	}
	for bi, rc := range []*Cache{r1, r2} {
		want, err := e.Forward(nextIDs[bi], rc)
		if err != nil {
			t.Fatalf("ref Forward %d: %v", bi, err)
		}
		for i := range want {
			if math.Float32bits(got[bi][i]) != math.Float32bits(want[i]) {
				t.Fatalf("seq %d logit[%d]: got %v, want %v (not bit-identical)", bi, i, got[bi][i], want[i])
			}
		}
	}
}

func TestCache_TruncateAndWindow(t *testing.T) {
	c := NewCache(2, 2, 4, 4, 8, nil)
	kv := make([]float32, 8)
	for pos := range 6 {
		for i := range kv {
			kv[i] = float32(pos*10 + i)
		}
		c.Append(0, kv, kv)
		c.Append(1, kv, kv)
	}
	if c.Pos() != 6 {
		t.Fatalf("Pos = %d, want 6", c.Pos())
	}
	if exact := c.TruncateTo(3); !exact {
		t.Fatalf("global TruncateTo(3) = inexact, want exact")
	}
	if c.Pos() != 3 {
		t.Fatalf("Pos after truncate = %d, want 3", c.Pos())
	}
	if got := c.WindowStart(5, false); got != 2 {
		t.Fatalf("WindowStart(5, false) = %d, want 2", got)
	}
	if got := c.WindowStart(5, true); got != 0 {
		t.Fatalf("WindowStart(5, true) = %d, want 0", got)
	}
}
