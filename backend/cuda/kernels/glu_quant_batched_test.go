//go:build cuda

package kernels

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
	"math/rand"
	"testing"

	gpu "github.com/townsendmerino/goinfer/backend/cuda/internal/gpudev"
)

// gluQuantBatchedDigest is the SHA-256 of the packed int8 rows and the row scales glu_quant_batched leaves for the fixed inputs below, recorded from main's
// out-of-place kernel (docs/measurements/kv3-s1-2026-11/README.md). The in-place kernel that replaced it must leave the same bytes.
const gluQuantBatchedDigest = "793db9f987f8355e94cc24b9f9b4785d8f9031b67a4e8bcf73f5b2a2e950b3df"

// TestGluQuantBatched_inPlaceIsBitIdentical runs the kernel over both activations, a width that is not a multiple of the block, and an up operand read at an
// offset, and holds its quantized rows and scales to the digest of main's out-of-place kernel.
func TestGluQuantBatched_inPlaceIsBitIdentical(t *testing.T) {
	dev := requireBareCUDADevice(t)
	mod, err := dev.CompileLibrary(prefillBatchedPTX)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := dev.NewComputePipeline(mod, "glu_quant_batched")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	q := dev.NewCommandQueue()
	h := sha256.New()
	for _, c := range []struct{ act, I, M, uOff int }{{1, 4096, 7, 0}, {0, 4096, 7, 0}, {1, 1500, 5, 0}, {0, 1500, 5, 0}, {0, 768, 6, 512}, {1, 768, 6, 512}} {
		rng := rand.New(rand.NewSource(int64(c.act*1_000_000 + c.I*10 + c.M)))
		g := make([]float32, c.M*c.I)
		u := make([]float32, c.M*c.I+c.uOff)
		for i := range g {
			g[i] = float32(rng.NormFloat64()) * 3
			if i%97 == 0 {
				g[i] *= 12 // saturating tails of both activations
			}
		}
		for i := range u {
			u[i] = float32(rng.NormFloat64())
		}
		gb, ub := gpu.NewBufferLenOf[float32](dev, len(g)), gpu.NewBufferLenOf[float32](dev, len(u))
		qb, sb := gpu.NewBufferLenOf[int32](dev, c.M*c.I/4), gpu.NewBufferLenOf[float32](dev, c.M)
		if err := gpu.Upload(gb, g); err != nil {
			t.Fatal(err)
		}
		if err := gpu.Upload(ub, u); err != nil {
			t.Fatal(err)
		}
		launchGluQuantBatched(t, q, pipe, c.M, c.I, c.act, c.uOff, gb, ub, qb, sb)
		if err := q.Sync(); err != nil {
			t.Fatal(err)
		}
		packed, scale := make([]int32, c.M*c.I/4), make([]float32, c.M)
		if err := gpu.Download(qb, packed); err != nil {
			t.Fatal(err)
		}
		if err := gpu.Download(sb, scale); err != nil {
			t.Fatal(err)
		}
		var w [4]byte
		for _, v := range packed {
			binary.LittleEndian.PutUint32(w[:], uint32(v))
			h.Write(w[:])
		}
		for _, v := range scale {
			if !(v > 0) || math.IsInf(float64(v), 0) {
				t.Fatalf("VOID: act %d I %d: a row scale is %v, the inputs do not exercise the kernel", c.act, c.I, v)
			}
			binary.LittleEndian.PutUint32(w[:], math.Float32bits(v))
			h.Write(w[:])
		}
	}
	got := hex.EncodeToString(h.Sum(nil))
	if got != gluQuantBatchedDigest {
		t.Errorf("digest %s, want %s", got, gluQuantBatchedDigest)
	}
}

// launchGluQuantBatched is the kernel's launch as the prefill pass makes it: the product is left in gb.
func launchGluQuantBatched(t *testing.T, q gpu.Queue, pipe gpu.Pipeline, M, I, act, uOff int, gb, ub, qb, sb gpu.Buffer) {
	t.Helper()
	if err := q.Launch(pipe, LaunchConfig{GridX: uint32(M), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4},
		Arg(gb), Arg(ub), gpu.ArgValue(int32(uOff)), gpu.ArgValue(int32(I)), gpu.ArgValue(int32(act)), Arg(qb), Arg(sb), gpu.ArgValue(int32(M))); err != nil {
		t.Fatalf("launch: %v", err)
	}
}
