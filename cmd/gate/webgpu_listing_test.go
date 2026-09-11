package main

import (
	"regexp"
	"strings"
	"testing"
)

// EVERY GATE-SHAPED, goinfer_testhooks-TAGGED WEBGPU TEST IS LISTED, ONE WAY OR THE OTHER.
//
// audit-2026-09-10 G-10: webgpu-parity's -run was "ResidentParity", so nine gate-shaped gpu/
// tests, among them the G6 staged-int4 matmul gate and ForwardN parity, matched no cell and ran
// in no CI runner, with nothing to say so. This is TestMetalGateIsListedOrExplicitlyNotRequired's
// WebGPU twin.
func TestWebGPUGateIsListedOrExplicitlyNotRequired(t *testing.T) {
	root, err := repoRoot()
	if err != nil {
		t.Fatalf("cannot locate the repo root: %v", err)
	}
	found := webgpuGateTests(root)
	if len(found) == 0 {
		t.Fatalf("no gate-shaped test found in any //go:build goinfer_testhooks file under %v — the "+
			"scan is broken, and a broken scan both empties this check and narrows the cell's -run", webgpuDirs)
	}
	runRe, err := regexp.Compile(webgpuParityRun)
	if err != nil {
		t.Fatalf("webgpuParityRun does not compile as a regexp: %v", err)
	}
	for _, test := range found {
		reason, excluded := webgpuNotRequired[test]
		matched := runRe.MatchString(test)
		switch {
		case matched && excluded:
			t.Errorf("%s is BOTH matched by webgpuParityRun and in webgpuNotRequired", test)
		case matched:
		case excluded && strings.TrimSpace(reason) == "":
			t.Errorf("%s is in webgpuNotRequired with an EMPTY reason", test)
		case excluded:
		default:
			t.Errorf("webgpu gate-shaped test %s matches no cell's -run pattern (webgpuParityRun) and "+
				"is not in webgpuNotRequired: it runs nowhere (G-10). Add it to webgpuParityRun or to "+
				"webgpuNotRequired with a reason.", test)
		}
	}
	inTree := map[string]bool{}
	for _, test := range found {
		inTree[test] = true
	}
	for test := range webgpuNotRequired {
		if !inTree[test] {
			t.Errorf("webgpuNotRequired names %q, which the scan does not find — a stale exemption", test)
		}
	}
	t.Logf("%d gate-shaped WebGPU tests; webgpuParityRun = %q", len(found), webgpuParityRun)
}
