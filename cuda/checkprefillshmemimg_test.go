//go:build cuda && goinfer_testhooks

package cuda

import (
	"errors"
	"go/ast"
	"go/parser"
	"go/token"
	"strings"
	"testing"
)

// TestImgBlockMaxNWin_formula pins imgBlockMaxNWin against every case its own derivation
// distinguishes (cuda/attn_img_prefill.cu's header) — the pure-function half of the plan's top
// risk, needing no device.
func TestImgBlockMaxNWin_formula(t *testing.T) {
	for _, tc := range []struct {
		name                  string
		causalMaxNWin, window int
		imgStart, imgEnd      int
		want                  int
	}{
		{"no block: unchanged", 100, 16, 0, 0, 100},
		{"unwindowed: block never exceeds startPos+M (v1 invariant)", 266, 0, 6, 262, 266},
		{
			// causalMaxNWin here is min(startPos+M, window) as checkPrefillShmemImg's own caller
			// computes it — window=50 < startPos+M, so causalMaxNWin=window=50. imgStart=10 <=
			// window-1=49: some in-block row still has winStart=0, so blockMax=imgEnd=300 — a
			// SHORT window near a LONG block's start still needs shmem sized to the WHOLE block.
			"windowed, block starts at/before window-1: some row still has winStart=0", 50, 50, 10, 300, 300,
		},
		{
			// causalMaxNWin = min(startPos+M, window) = window = 8 (the realistic value a caller
			// with window < startPos+M would pass). imgStart=20 (>7=window-1), imgEnd=32 →
			// blockMax = imgLen+window-1 = 12+8-1 = 19.
			"windowed, block starts PAST window-1: decreasing width, max at pos=imgStart",
			8, 8, 20, 32, 19,
		},
		{
			"windowed, causal already exceeds the block's own widened span",
			// causalMaxNWin (e.g. from a huge startPos+M or an already-wide window) can exceed
			// blockMax — max() must still pick it.
			5000, 4096, 20, 32, 5000,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := imgBlockMaxNWin(tc.causalMaxNWin, tc.window, tc.imgStart, tc.imgEnd)
			if got != tc.want {
				t.Errorf("imgBlockMaxNWin(%d, %d, %d, %d) = %d, want %d",
					tc.causalMaxNWin, tc.window, tc.imgStart, tc.imgEnd, got, tc.want)
			}
		})
	}
}

// TestImgBlockMaxNWin_differsFromNaiveCoupledFormula is the regression case named in the plan:
// prove the decoupled formula actually DIFFERS from attn_batched's own coupled one
// (winStart = nKeys-window, using the WIDENED nKeys) at the scenario that matters — a naive
// port that reused the coupled formula would silently under-report the window here, which is
// exactly the shared-memory-under-allocation hazard this whole design exists to avoid.
func TestImgBlockMaxNWin_differsFromNaiveCoupledFormula(t *testing.T) {
	const window, imgStart, imgEnd = 8, 20, 32 // imgLen=12
	causalMaxNWin := imgEnd                    // stand-in startPos+M large enough to not itself dominate
	correct := imgBlockMaxNWin(causalMaxNWin, window, imgStart, imgEnd)
	// The naive coupled formula: winStart = nKeys-window using the WIDENED nKeys=imgEnd, so its
	// implied window width is nKeys-winStart = imgEnd-(imgEnd-window) = window — clamped to the
	// plain window size, oblivious to the block at all.
	naiveCoupledWidth := window
	if correct <= naiveCoupledWidth {
		t.Fatalf("decoupled formula gave %d, naive coupled gives %d — the fix must produce a LARGER "+
			"(more inclusive) window here, or a naive copy-paste would pass this test undetected",
			correct, naiveCoupledWidth)
	}
	t.Logf("decoupled formula: %d attended keys; naive coupled formula would give only %d — the "+
		"decoupling is load-bearing at this scenario, not cosmetic", correct, naiveCoupledWidth)
}

// TestCheckPrefillShmemImg_declinesPastTheLimit mirrors TestCheckPrefillShmem_declinesPastTheLimit's
// exact boundary (12160 AT the limit, 12161 past it) but through the image-aware path, at a shape
// where the widened block (not startPos+M) is what crosses the boundary.
func TestCheckPrefillShmemImg_declinesPastTheLimit(t *testing.T) {
	r := declineFixture(1, "int4")
	// window=0 (unwindowed, the fixture's default): blockMax=imgEnd, causalMaxNWin=startPos+M.
	// Pick M small and imgEnd large so the BLOCK is what crosses the boundary, not the causal term.
	const startPos, M = 0, 20
	if err := r.checkPrefillShmemImg(startPos, M, 0, 12160); err != nil {
		t.Errorf("imgEnd=12160 is AT the limit and must not decline: %v", err)
	}
	err := r.checkPrefillShmemImg(startPos, M, 0, 12161)
	if err == nil {
		t.Fatal("imgEnd=12161 is past the limit — checkPrefillShmemImg must decline")
	}
	if !errors.Is(err, errPrefillDeclined) {
		t.Errorf("decline must wrap errPrefillDeclined so callers can fall back cleanly: %v", err)
	}
	if !strings.Contains(err.Error(), "layer 0") || !strings.Contains(err.Error(), "image-block") {
		t.Errorf("decline should identify itself as the image-block check's and locate the layer: %v", err)
	}
}

// TestCheckPrefillShmemImg_windowedBlockPastWindow exercises the branch
// TestImgBlockMaxNWin_formula's "windowed, block starts PAST window-1" case pins at the formula
// level, now through the fixture-based decline check — a windowed layer whose block-widened span
// crosses the limit even though its plain windowed span alone would not.
func TestCheckPrefillShmemImg_windowedBlockPastWindow(t *testing.T) {
	r := declineFixture(1, "int4")
	r.layers[0].window = 100 // small window: plain causal/windowed span never approaches the limit
	// imgStart > window-1 (99): blockMax = imgLen+window-1. Choose imgLen so blockMax lands exactly
	// at, then one past, the 12160-key boundary.
	const imgStart = 200
	const imgLenAt = 12160 - 100 + 1 // blockMax = imgLenAt+100-1 = 12160 exactly
	const imgLenOver = imgLenAt + 1  // blockMax = 12161
	if err := r.checkPrefillShmemImg(0, imgStart+imgLenAt+50, imgStart, imgStart+imgLenAt); err != nil {
		t.Errorf("block-widened span AT the limit must not decline: %v", err)
	}
	if err := r.checkPrefillShmemImg(0, imgStart+imgLenOver+50, imgStart, imgStart+imgLenOver); err == nil {
		t.Fatal("block-widened span past the limit must decline — the window's own small span alone would not have caught this")
	}
}

// TestPrefillCoreCallsCheckPrefillShmemImg is TestPrefillCoreAndDraftBlockCallTheShmemGuards'
// twin for the new check — proves prefillCore's SOURCE actually calls checkPrefillShmemImg, not
// just that the function itself is correct in isolation. V-05's own lesson: "an unused METHOD is
// not a Go compile error the way an unused import or local var is."
func TestPrefillCoreCallsCheckPrefillShmemImg(t *testing.T) {
	fset := token.NewFileSet()
	af, err := parser.ParseFile(fset, "prefill.go", nil, 0)
	if err != nil {
		t.Fatalf("parse prefill.go: %v", err)
	}
	var fn *ast.FuncDecl
	ast.Inspect(af, func(n ast.Node) bool {
		if d, ok := n.(*ast.FuncDecl); ok && d.Name.Name == "prefillCore" {
			fn = d
		}
		return true
	})
	if fn == nil {
		t.Fatal("prefillCore not found in prefill.go — this guard is watching nothing")
	}
	found := false
	ast.Inspect(fn, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		switch f := call.Fun.(type) {
		case *ast.Ident:
			if f.Name == "checkPrefillShmemImg" {
				found = true
			}
		case *ast.SelectorExpr:
			if f.Sel.Name == "checkPrefillShmemImg" {
				found = true
			}
		}
		return true
	})
	if !found {
		t.Error("prefillCore does not call checkPrefillShmemImg — the image-block shared-memory decline would never fire")
	}
}
