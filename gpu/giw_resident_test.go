//go:build gpu

package gpu_test

import (
	"context"
	"os"
	"slices"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/gpu"
)

// TestGIWInt4_resident gates the int4 `.giw` → GPU-resident seam (docs/task-mellum2-fast-load.md
// acceptance #2): a prequant int4 bundle must deserialize into the int4 linalg.WeightMat that the
// resident builders (dense uploadProj + the stacked-MoE buildStacked "int4" path) consume — i.e.
// loading the bundle goes RESIDENT at int4 with NO requant, and decodes token-IDENTICALLY to a
// direct int4 load of the source model (the bundle is just those same int4 weights serialized).
// This is the one previously-untested step in the fast-load path.
//
// Build the bundle first (any model with an embedded tokenizer):
//
//	go run ./cmd/prequant -quant int4 -o /tmp/m.int4.giw ~/models/qwen05/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
//	GOINFER_GIW_INT4=/tmp/m.int4.giw \
//	GOINFER_GIW_SRC=~/models/qwen05/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf \
//	  go test -tags gpu ./gpu/ -run TestGIWInt4_resident -v -timeout 30m
func TestGIWInt4_resident(t *testing.T) {
	if _, err := gpu.New(); err != nil {
		t.Skipf("no WebGPU adapter: %v", err)
	}
	giwPath := os.Getenv("GOINFER_GIW_INT4")
	srcPath := os.Getenv("GOINFER_GIW_SRC")
	if giwPath == "" || srcPath == "" {
		t.Skip("set GOINFER_GIW_INT4 (prebuilt int4 .giw) + GOINFER_GIW_SRC (its source model)")
	}
	for _, p := range []string{giwPath, srcPath} {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("missing %s: %v", p, err)
		}
	}

	ctx := context.Background()
	greedy := decoder.SamplingParams{Temperature: 0}
	prompt := []int{785, 6722, 315, 9625, 374} // "The capital of France is"
	const n = 24

	// Load → decode → CLOSE each model in turn (never both resident at once: two int4
	// copies of a multi-GB model would exceed an 8 GB card). Returns the greedy tokens,
	// the load wall time, and whether it went resident.
	run := func(path string, opts decoder.Options) (toks []int, load time.Duration, resident bool) {
		t0 := time.Now()
		m, err := decoder.Load(path, opts)
		if err != nil {
			t.Fatalf("load %s: %v", path, err)
		}
		defer m.Close()
		load = time.Since(t0)
		resident = m.ResidentActive()
		if !resident {
			return nil, load, false
		}
		ch, _ := m.Generate(ctx, prompt, n, greedy)
		for id := range ch {
			toks = append(toks, id)
		}
		return toks, load, true
	}

	// The .giw bundle: already int4, loaded by mmap (no --quant) — the fast path.
	giwToks, giwLoad, giwRes := run(giwPath, decoder.Options{Backend: "webgpu"})
	if !giwRes {
		t.Fatal(".giw int4 bundle did not go GPU-resident — the int4 deserialize→resident seam is broken")
	}
	// The source model loaded + quantized to int4 directly (the slow path the bundle replaces).
	srcToks, srcLoad, srcRes := run(srcPath, decoder.Options{Backend: "webgpu", Quant: "int4"})
	if !srcRes {
		t.Skip("source did not go resident at int4 — can't cross-check")
	}

	t.Logf(".giw int4 resident: load %v, %d tok | direct int4: load %v, %d tok | load speedup %.1f×",
		giwLoad.Round(time.Millisecond), len(giwToks), srcLoad.Round(time.Millisecond), len(srcToks),
		srcLoad.Seconds()/giwLoad.Seconds())
	if !slices.Equal(giwToks, srcToks) {
		t.Fatalf(".giw-int4 decode != direct-int4 decode (the bundle is not the same int4 weights)\n giw %v\n src %v", giwToks, srcToks)
	}
}
