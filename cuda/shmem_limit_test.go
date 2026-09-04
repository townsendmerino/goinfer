//go:build cuda && goinfer_testhooks

package cuda

import (
	"errors"
	"go/ast"
	"go/parser"
	"go/token"
	"strings"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
)

// M-16: the single-block attention kernels size their scratch (nWin+128)*4 with NO ceiling, so
// past 12,160 attended keys the launch exceeds the 48 KB default and is refused by the driver.
// Decode fails at that position; batched prefill errors at layer 0 and falls back to the ~9x
// slower sequential path with nothing logged. The trigger is -ctx 16384+ on any geometry whose
// perf table says splitkvNever (nH >= 24: Qwen2.5-7B, Llama-3-8B, phi3-mini) — the -ctx 32768
// rows in benchmarks.md were on 0.5B/1.5B, whose split-KV engages at 3072/1024.
//
// The audit rated this medium confidence because it hinged on whether anything raises the
// kernel into the opt-in range. Nothing does, and this pins BOTH halves against the device.
func TestAttnShmemLimit_matchesDevice(t *testing.T) {
	// The arithmetic half runs anywhere.
	if got := attnShmemBytes(12160); got > singleBlockAttnShmemLimit {
		t.Errorf("12160 keys needs %d B, over the %d B limit — the constant and the sizing "+
			"formula disagree about where the boundary is", got, singleBlockAttnShmemLimit)
	}
	if got := attnShmemBytes(12161); got <= singleBlockAttnShmemLimit {
		t.Errorf("12161 keys needs only %d B — the boundary moved", got)
	}
	if !splitKVRequired(12161) {
		t.Error("splitKVRequired is false one key past the limit: the single-block launch would " +
			"be attempted and refused by the driver (M-16)")
	}
	if splitKVRequired(12160) {
		t.Error("splitKVRequired is true AT the limit, which would force split-KV one key early")
	}

	// The device half. MEASURED, not assumed: the constant must not exceed what this GPU allows,
	// or the guard passes launches the driver will refuse.
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	dev, err := gc.GetDevice(0)
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	const attrMaxShmemPerBlock = 8 // CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
	lim, err := dev.Attribute(gc.DeviceAttribute(attrMaxShmemPerBlock))
	if err != nil {
		t.Skipf("attribute query: %v", err)
	}
	t.Logf("device MAX_SHARED_MEMORY_PER_BLOCK = %d B (%.0f KB); guard is %d B => nWin <= %d keys",
		lim, float64(lim)/1024, singleBlockAttnShmemLimit, singleBlockAttnShmemLimit/4-128)
	if singleBlockAttnShmemLimit > lim {
		t.Errorf("the guard allows %d B but this device permits only %d B: launches inside the "+
			"guard would still be refused (M-16)", singleBlockAttnShmemLimit, lim)
	}
}

// V-05 (docs/review-2026-09-04.md): M-16 fixed decode (above) but left batched prefill (also the
// spec-decode verify path, since prefillCore serves both) and the drafter block attention with the
// SAME unguarded (nWin+128)*4 launch and no split-KV fallback. -ctx 16384+ on a splitkvNever
// geometry (nH >= 24: Qwen2.5-7B, Llama-3-8B, phi3-mini) hit the driver refusal at prefill/verify
// time instead of decode time, and the caller (decoder/model.go's PrefillLast handling) silently
// fell through to the ~9x-slower sequential path with nothing distinguishing "declined" from
// "crashed". These need no device: they drive checkPrefillShmem/the drafter's inline check
// directly on a struct-only fixture, the same way TestPrefillPath_matchesPrefillCore does.

// TestCheckPrefillShmem_declinesPastTheLimit pins the boundary exactly at declineFixture's default
// geometry (hd=4, so attnShmemBytes(startPos+M) is what's under test, no window clamp).
func TestCheckPrefillShmem_declinesPastTheLimit(t *testing.T) {
	r := declineFixture(2, "int4")
	if err := r.checkPrefillShmem(0, 12160); err != nil {
		t.Errorf("12160 attended keys is AT the limit and must not decline: %v", err)
	}
	err := r.checkPrefillShmem(0, 12161)
	if err == nil {
		t.Fatal("12161 attended keys is past the limit — checkPrefillShmem must decline")
	}
	if !errors.Is(err, errPrefillDeclined) {
		t.Errorf("decline must wrap errPrefillDeclined so callers can fall back cleanly: %v", err)
	}
	if !strings.Contains(err.Error(), "layer 0") {
		t.Errorf("decline should locate the declining layer: %v", err)
	}
}

// TestCheckPrefillShmem_slidingWindowClampsPerLayer: a global layer at the limit declines even
// when a sliding-window layer in the SAME model, at the same startPos+M, does not — because the
// window clamps its OWN maxNWin below the boundary. Mirrors the launch site's own per-layer
// maxNWin computation exactly.
func TestCheckPrefillShmem_slidingWindowClampsPerLayer(t *testing.T) {
	r := declineFixture(2, "int4")
	r.layers[1].window = 4096 // well under the 12160-key boundary
	if err := r.checkPrefillShmem(0, 12161); err == nil {
		t.Fatal("layer 0 (no window) is past the limit at 12161 keys — must decline")
	}
	// Isolate layer 1: alone, its window clamp keeps it under the limit at any startPos+M.
	r2 := declineFixture(1, "int4")
	r2.layers[0].window = 4096
	if err := r2.checkPrefillShmem(0, 100000); err != nil {
		t.Errorf("a single sliding-window(4096) layer must not decline regardless of prompt "+
			"length — its own window clamps maxNWin: %v", err)
	}
}

// TestCheckDrafterShmem_declinesPastTheLimit drives checkDrafterShmem directly — extracted from
// DraftBlock specifically so this needs no live device to exercise the guard itself.
func TestCheckDrafterShmem_declinesPastTheLimit(t *testing.T) {
	if err := checkDrafterShmem(0, 12160); err != nil {
		t.Errorf("12160 attended keys is AT the limit and must not decline: %v", err)
	}
	err := checkDrafterShmem(0, 12161)
	if err == nil {
		t.Fatal("12161 attended keys is past the limit — checkDrafterShmem must decline")
	}
	if !strings.Contains(err.Error(), "drafter") {
		t.Errorf("decline should identify itself as the drafter's, not prefill's: %v", err)
	}
	// ctxLen + M split the same way DraftBlock computes nKeys — a long-running drafter context
	// hitting the boundary mid-block must decline exactly as a single large M would.
	if err := checkDrafterShmem(12000, 160); err != nil {
		t.Errorf("ctxLen=12000 + M=160 = 12160 is AT the limit and must not decline: %v", err)
	}
	if err := checkDrafterShmem(12000, 161); err == nil {
		t.Error("ctxLen=12000 + M=161 = 12161 is past the limit — must decline")
	}
}

// TestPrefillCoreAndDraftBlockCallTheShmemGuards is the wiring half the tests above cannot cover:
// they drive checkPrefillShmem/checkDrafterShmem directly, which proves the guards are correct but
// says nothing about whether the real call sites still invoke them. Caught in practice while
// mutation-testing this fix: removing prefillCore's call to checkPrefillShmem still builds clean
// (an unused METHOD is not a Go compile error the way an unused import or local var is) and every
// unit test above still passes, because none of them go through prefillCore/DraftBlock at all.
// Asserted structurally, the same way TestStreamTokens_decodesAsAContinuation and
// TestWebUI_rootRouteIsUnauthenticated pin their own wiring.
func TestPrefillCoreAndDraftBlockCallTheShmemGuards(t *testing.T) {
	check := func(t *testing.T, file, fnName, wantCall string) {
		t.Helper()
		fset := token.NewFileSet()
		af, err := parser.ParseFile(fset, file, nil, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", file, err)
		}
		var fn *ast.FuncDecl
		ast.Inspect(af, func(n ast.Node) bool {
			if d, ok := n.(*ast.FuncDecl); ok && d.Name.Name == fnName {
				fn = d
			}
			return true
		})
		if fn == nil {
			t.Fatalf("%s not found in %s — this guard is watching nothing", fnName, file)
		}
		found := false
		ast.Inspect(fn, func(n ast.Node) bool {
			call, ok := n.(*ast.CallExpr)
			if !ok {
				return true
			}
			switch f := call.Fun.(type) {
			case *ast.Ident:
				if f.Name == wantCall {
					found = true
				}
			case *ast.SelectorExpr:
				if f.Sel.Name == wantCall {
					found = true
				}
			}
			return true
		})
		if !found {
			t.Errorf("%s does not call %s — the shared-memory decline would never fire (V-05)",
				fnName, wantCall)
		}
	}
	t.Run("prefillCore/checkPrefillShmem", func(t *testing.T) {
		check(t, "prefill.go", "prefillCore", "checkPrefillShmem")
	})
	t.Run("DraftBlock/checkDrafterShmem", func(t *testing.T) {
		check(t, "drafter.go", "DraftBlock", "checkDrafterShmem")
	})
}
