package serveapp

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// resolvedFields are the decoder.Options fields loadflags.Resolve sets.
var resolvedFields = []string{"Backend", "BackendAuto", "Quant", "LoRA", "KVPrecision", "KVQuant", "ResidentContext",
	"StreamWeights", "WeightCacheBytes", "MoECacheExperts", "MoECacheSlots", "MoEPager", "AcceptSlowMoE", "EmbedInt4",
	"DisableFit", "ExactPrefill", "ResidentWindowedKV", "Knobs", "ResidentKVSlots", "ResidentKVSlotsDefault",
	"ResidentPrefillChunk", "KVShareOff", "KVHostBytes", "CPUBatchDecode"}

// oneMergeExceptions are assignments that derive a field from something other than configuration.
var oneMergeExceptions = map[string]string{
	"o.LoRA = a": "specHead: a decision head's unmerged adapter becomes the model's LoRA",
}

var (
	resolvedField = regexp.MustCompile(`\b([A-Za-z_]\w*)\.(` + strings.Join(resolvedFields, "|") + `)\b`)
	plainAssign   = regexp.MustCompile(`[^=!<>:+\-*/%&|^]=[^=]`)
)

// notOptions are receivers whose fields share names with decoder.Options': a loadflags.Flags (load) and the
// loadflags.Overrides serve builds (ov).
var notOptions = map[string]bool{"load": true, "ov": true}

// optionsAssignments lists the lines of src whose left-hand side assigns a resolved field.
func optionsAssignments(src string) []string {
	var out []string
	for _, line := range strings.Split(src, "\n") {
		t := strings.TrimSpace(line)
		loc := plainAssign.FindStringIndex(t)
		if strings.HasPrefix(t, "//") || loc == nil {
			continue
		}
		for _, m := range resolvedField.FindAllStringSubmatch(t[:loc[0]+1], -1) {
			if notOptions[m[1]] {
				continue
			}
			if _, ok := oneMergeExceptions[t]; !ok {
				out = append(out, t)
			}
			break
		}
	}
	return out
}

// TestOneMerge is G-SB10-14 (docs/tasks/task-subsystems-2026-10.md § "Step 3b's gates"): serve's non-test code
// assigns no decoder.Options field loadflags.Resolve sets, so a per-model setting is merged in one place.
func TestOneMerge(t *testing.T) {
	planted := "\to.KVHostBytes = cfg.kvHostBytes\n\tc.load.Quant, c.load.QuantSet = q, true\n\tov.Quant = &q\n\tf(o.Quant, x == y)\n"
	if got := optionsAssignments(planted); len(got) != 1 {
		t.Fatalf("planted assignment: the scan found %q, want the one o.KVHostBytes line", got)
	}
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range files {
		if strings.HasSuffix(f, "_test.go") {
			continue
		}
		b, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		for _, line := range optionsAssignments(string(b)) {
			t.Errorf("%s: %q sets a field loadflags.Resolve owns; pass it through loadflags.Overrides", f, line)
		}
	}
}
