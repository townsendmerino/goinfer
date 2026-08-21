package main

import (
	"regexp"
	"strings"
)

// Skip bucketing — ported verbatim in BEHAVIOUR from scripts/skip_census.py, whose rule order is
// itself load-bearing (first match wins: fixture before device before heavy). The buckets come
// from docs/parity-coverage-policy.md, "A gate must be able to run, and able to fail":
//
//	missing-fixture  a committed golden over a gitignored checkpoint (regen: scripts/pin_*.py)
//	missing-golden   no *_golden.json recorded yet
//	no-gpu-device    needs a real GPU/Metal/CUDA device (CI has none)
//	heavy-model      gated behind GOINFER_HEAVY_TESTS / a real ~/models checkpoint
//	integration-env  needs a runtime env var (GOINFER_SERVE_MODEL / _EMBED_MODEL)
//	other            unclassified — inspect these; they should be rare
//
// The point of bucketing at all: a green `go test ./...` is not "all green" if most of the suite
// skipped, and a run that skipped 200 asset-gated tests must never be mistakable for one that
// exercised them.
var skipRules = []struct {
	bucket string
	rx     *regexp.Regexp
}{
	{"heavy-model", regexp.MustCompile(`(?i)heavy|GOINFER_HEAVY|~/models|large model|big box|real (model|checkpoint)`)},
	{"integration-env", regexp.MustCompile(`(?i)\bset GOINFER_|GOINFER_SERVE_MODEL|GOINFER_EMBED_MODEL|GOINFER_VISION|integration`)},
	{"missing-golden", regexp.MustCompile(`(?i)no golden|_golden\.json|run scripts/pin`)},
	{"missing-fixture", regexp.MustCompile(`(?i)no tiny|checkpoint|fixture|\.safetensors|not present|not found|absent|no such|no (gguf|model|tokenizer|qwen|mellum|gemma|llama|mistral|phi|glm|deepseek|kimi|granite|nemotron|cohere)`)},
	{"no-gpu-device", regexp.MustCompile(`(?i)\bgpu\b|\bdevice\b|webgpu|no adapter|headless|no metal|no cuda`)},
}

// bucketOrder is the REPORT order, deliberately different from the MATCH order above: the report
// leads with the buckets a release is allowed to be blocked by.
var bucketOrder = []string{"missing-fixture", "missing-golden", "no-gpu-device", "heavy-model", "integration-env", "other"}

func classifySkip(reason string) string {
	for _, r := range skipRules {
		if r.rx.MatchString(reason) {
			return r.bucket
		}
	}
	return "other"
}

// skipReason extracts the reason a test skipped from its output events: the LAST non-empty line
// that is not a test-framework marker. That is where t.Skipf's message lands (`foo_test.go:42:
// no tiny fixture`), and taking the last one rather than the first is what makes it survive a
// test that logged before it skipped.
func skipReason(lines []string) string {
	reason := ""
	for _, ln := range lines {
		ls := strings.TrimSpace(ln)
		if ls == "" ||
			strings.Contains(ls, "--- SKIP") ||
			strings.Contains(ls, "=== RUN") ||
			strings.Contains(ls, "=== PAUSE") ||
			strings.Contains(ls, "=== CONT") {
			continue
		}
		reason = ls
	}
	return reason
}

// crashMarkers are the strings that distinguish a package-level failure that is a NATIVE CRASH
// from one that is an ordinary build error. Recorded because the Metal suite has a known flaky
// single-process `fault 0x10` tail (purego-objc / no-ARC) that is not a test failure — every test
// passes in isolation or in shards — and a reader who cannot tell the two apart will either chase
// a phantom or dismiss a real one.
var crashMarkers = []string{"SIGSEGV", "fault 0x", "signal arrived", "panic:"}

func looksLikeCrash(blob string) bool {
	for _, m := range crashMarkers {
		if strings.Contains(blob, m) {
			return true
		}
	}
	return false
}
