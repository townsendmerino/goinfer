package decoder

import (
	"os/exec"
	"reflect"
	"strings"
	"testing"
)

// R10 (docs/measurements/cold-user-2026-09-06-nobara-pc.md): a README+pkg.go.dev-only reader
// could not find Options'/SamplingParams' field names — every FUNCTION signature came back
// cleanly from pkg.go.dev, but the struct bodies came back truncated on two separate fetches.
// Checked here against the actual source (not the live pkg.go.dev page, which this repo does
// not control and cannot gate on): `go doc` — the same tool pkg.go.dev itself runs to render a
// package — already lists every field of both structs in full. This is the regression gate for
// that: reflect the REAL field set (so a field added later cannot silently stop being checked)
// and require go doc's own text to name each one.
func TestGoDoc_listsEveryFieldOfOptionsAndSamplingParams(t *testing.T) {
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("no go toolchain")
	}
	for _, typ := range []reflect.Type{
		reflect.TypeOf(Options{}),
		reflect.TypeOf(SamplingParams{}),
	} {
		name := typ.Name()
		t.Run(name, func(t *testing.T) {
			out, err := exec.Command("go", "doc", ".", name).CombinedOutput()
			if err != nil {
				t.Fatalf("go doc . %s: %v\n%s", name, err, out)
			}
			doc := string(out)
			if n := typ.NumField(); n == 0 {
				t.Fatalf("%s has no fields — this test would pass having checked nothing", name)
			}
			for i := 0; i < typ.NumField(); i++ {
				f := typ.Field(i).Name
				if !strings.Contains(doc, f) {
					t.Errorf("go doc . %s does not mention field %s:\n%s", name, f, doc)
				}
			}
		})
	}
}
