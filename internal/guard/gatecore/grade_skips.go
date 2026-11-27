package gatecore

import (
	"regexp"
	"strings"
)

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

var skipTokens = []struct{ token, bucket string }{
	{"SKIP-HEAVY:", "heavy-model"},
	{"SKIP-DEVICE:", "no-gpu-device"},
	{"SKIP-ASSET:", "missing-fixture"},
	{"SKIP-MEASURE:", "integration-env"},
}

// BucketOrder is the report order for census and quick skip summaries.
var BucketOrder = []string{"missing-fixture", "missing-golden", "no-gpu-device", "heavy-model", "integration-env", "other"}

// ClassifySkip assigns a skip message to its bucket, checking leading SKIP-* tokens before wording regexes.
func ClassifySkip(reason string) string {
	for _, k := range skipTokens {
		if strings.Contains(reason, k.token) {
			return k.bucket
		}
	}
	for _, r := range skipRules {
		if r.rx.MatchString(reason) {
			return r.bucket
		}
	}
	return "other"
}

// SkipReason returns the last non-empty, non-framework output line of a skipped test.
func SkipReason(lines []string) string {
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

var crashMarkers = []string{"SIGSEGV", "fault 0x", "signal arrived", "panic:"}

// LooksLikeCrash reports whether package output contains a native crash or panic marker.
func LooksLikeCrash(blob string) bool {
	for _, m := range crashMarkers {
		if strings.Contains(blob, m) {
			return true
		}
	}
	return false
}
