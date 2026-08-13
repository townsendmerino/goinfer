//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"os"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestA13_LoadModuleOnResidentExecutor fills the missing cell of the 1×2 and is the CONTROL that
// decides whether the multi-model unload null means anything.
//
// The context factor is already eliminated by reading: CreateSystemDefaultDevice calls dev.Primary(),
// gocudrv never binds cuCtxCreate, so the resident's context and a test's context are THE SAME
// primary context. Two factors remain:
//
//	A — module route : ctx.LoadModule(prebuilt PTX)   vs  CompileLibrary (NVRTC at runtime)
//	B — launch site  : test goroutine                 vs  the resident's pinned executor
//
//	known: (LoadModule, test)          POISONS
//	known: (CompileLibrary, resident)  does not, with the stimulus applied through rf.do
//
// This is (LoadModule, resident): a prebuilt PTX module loaded, launched, and stimulated entirely on
// the resident's executor thread.
//
//	poisons -> resident-executor launches ARE poisonable, so the multi-model null is real evidence,
//	           and the variable is the MODULE ROUTE rather than the thread
//	clean   -> nothing has ever poisoned this route; the multi-model null means nothing yet and the
//	           harness question is still open
func TestA13_LoadModuleOnResidentExecutor(t *testing.T) {
	if os.Getenv("GOINFER_A13_CELL") == "" {
		t.Skip("set GOINFER_A13_CELL=1 — A13 probe, deliberately not part of the tier")
	}
	const path = "../testdata/mistral-tiny-window"
	requireDeviceAndFixture(t, path)
	mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer mc.Close()
	rf, ok := mc.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Skip("not resident on cuda")
	}

	const nH, nKV, hd, nKeys = 12, 2, 64, 129
	const scale = float32(0.05)
	kvDim := nKV * hd

	// Everything below runs ON THE RESIDENT'S PINNED EXECUTOR — module load, allocation, launch and
	// the stimulus. That is the whole point of the cell: same thread, same context, prebuilt module.
	var beforeNZ, afterNZ int
	if e := rf.do(func() error {
		cx := rf.dev.Context()
		mod, e := cx.LoadModule(gluePTX)
		if e != nil {
			return e
		}
		fn, e := mod.Function("attention")
		if e != nil {
			return e
		}
		bg := context.Background()
		stream, e := cx.NewStream()
		if e != nil {
			return e
		}
		run := func() (int, error) {
			q := make([]float32, nH*hd)
			k := make([]float32, nKeys*kvDim)
			v := make([]float32, nKeys*kvDim)
			for i := range q {
				q[i] = float32(i%13) * 0.01
			}
			for i := range k {
				k[i] = float32(i%7) * 0.02
				v[i] = float32(i%11) * 0.03
			}
			dq, e1 := gc.Alloc[float32](cx, len(q))
			dk, e2 := gc.Alloc[float32](cx, len(k))
			dv, e3 := gc.Alloc[float32](cx, len(v))
			dc, e4 := gc.Alloc[float32](cx, nH*hd)
			if e1 != nil || e2 != nil || e3 != nil || e4 != nil {
				return 0, e1
			}
			defer func() { dq.Close(); dk.Close(); dv.Close(); dc.Close() }()
			if e := gc.CopyHtoD(bg, dq, q); e != nil {
				return 0, e
			}
			if e := gc.CopyHtoD(bg, dk, k); e != nil {
				return 0, e
			}
			if e := gc.CopyHtoD(bg, dv, v); e != nil {
				return 0, e
			}
			if e := fn.LaunchOn(bg, stream, gc.LaunchConfig{GridX: uint32(nH), GridY: 1, GridZ: 1,
				BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((nKeys + 128) * 4)},
				gc.Arg(dq), gc.Arg(dk), gc.Arg(dv), gc.ArgValue(int32(nH)), gc.ArgValue(int32(nKV)),
				gc.ArgValue(int32(hd)), gc.ArgValue(int32(nKeys)), gc.ArgValue(scale),
				gc.ArgValue(int32(0)), gc.Arg(dc)); e != nil {
				return 0, e
			}
			if e := stream.Synchronize(bg); e != nil {
				return 0, e
			}
			out := make([]float32, nH*hd)
			if e := gc.CopyDtoH(bg, out, dc); e != nil {
				return 0, e
			}
			return countNonZero(out), nil
		}

		if beforeNZ, e = run(); e != nil {
			return e
		}

		// THE STIMULUS, on this same executor: hold ~half the free VRAM and release it.
		free, _, _ := cx.MemInfo()
		var held []Buffer
		const chunk = 64 << 20
		for got := 0; got+chunk <= int(free)/2; got += chunk {
			held = append(held, rf.dev.MustBuf(chunk, chunk/4, "a13-cell"))
		}
		t.Logf("stimulus: held %d x 64 MiB = %.1f MiB on the resident executor, releasing",
			len(held), float64(len(held)*chunk)/(1<<20))
		for _, b := range held {
			rf.dev.ReleaseBuf(b)
		}

		afterNZ, e = run()
		return e
	}); e != nil {
		t.Fatalf("cell: %v", e)
	}

	t.Logf("(LoadModule, resident executor): non-zero before=%d after=%d (of %d)", beforeNZ, afterNZ, nH*hd)
	if beforeNZ == 0 {
		t.Fatal("the BASELINE launch produced zeros — this cell says nothing")
	}
	if afterNZ == 0 {
		t.Logf("POISONS: resident-executor launches ARE poisonable, so the multi-model null is real " +
			"evidence and the variable is the MODULE ROUTE, not the thread")
		t.Fail()
		return
	}
	t.Logf("CLEAN: nothing has poisoned this route — the multi-model null is NOT yet evidence, and " +
		"the harness question stays open")
}
