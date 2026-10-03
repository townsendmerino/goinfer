package main

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/constrain"
	"github.com/townsendmerino/goinfer/multimodal"
)

const schemaFile = "../../testdata/glm_ocr/invoice.schema.json"

// The committed schema file (what `goinfer-chat --schema` and the serve demo read) is this struct's schema. They are two
// spellings of one source of truth, so a field added to Invoice and not to the file fails here instead of in a demo.
func TestInvoiceSchemaFileIsTheStructsSchema(t *testing.T) {
	raw, err := os.ReadFile(schemaFile)
	if err != nil {
		t.Fatal(err)
	}
	want, err := constrain.SchemaFromStruct(Invoice{})
	if err != nil {
		t.Fatal(err)
	}
	var a, b any
	if err := json.Unmarshal(raw, &a); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(want, &b); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("%s is not constrain.SchemaFromStruct(Invoice{}):\nfile %s\nstruct %s", schemaFile, raw, want)
	}
	// And the prompt a request built from the file shows is the prompt the struct shows (the same keys in the same order),
	// which is what makes the serve route (schema in) and this example (struct in) the same request to the model.
	fromFile, err := constrain.TemplateFromSchema(raw)
	if err != nil {
		t.Fatal(err)
	}
	fromStruct, err := constrain.TemplateFromStruct(Invoice{})
	if err != nil {
		t.Fatal(err)
	}
	if fromFile != fromStruct {
		t.Fatalf("template from the schema file differs from the struct's:\n%s\n%s", fromFile, fromStruct)
	}
	if !strings.Contains(multimodal.GlmOcrExtractionPrompt(fromStruct), `"line_items": [`) {
		t.Errorf("the template does not show the line item list: %s", fromStruct)
	}
	if _, err := constrain.GrammarFromStruct(Invoice{}); err != nil {
		t.Fatal(err)
	}
}

// TestInvoiceExample runs the real binary on the real GLM-OCR checkpoint and the committed rendered invoice
// (testdata/glm_ocr/invoice.png: procedurally rendered, NOT a real scan). Heavy: the CPU tower alone is about a minute and
// the whole run a few minutes, so it needs GOINFER_HEAVY_TESTS=1 as well as the checkpoint, and skips cleanly without either.
//
// The assertion is deliberately loose, because the exact text is int4-sensitive (O3: int4 drops a header line f32 reads):
// the output must unmarshal into Invoice (the shape guarantee) and the invoice's total must read 1140.55, the figure the
// page prints in bold at the foot of the totals. The strict per-field numbers are in the O5 accuracy report, not here.
func TestInvoiceExample(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1 to opt in (loads a 2.7 GB model, runs the CPU tower)")
	}
	ckpt := filepath.Join(os.Getenv("HOME"), "models", "glm-ocr")
	if _, err := os.Stat(filepath.Join(ckpt, "model.safetensors")); err != nil {
		t.Skipf("no GLM-OCR checkpoint at %s", ckpt)
	}
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("no go toolchain")
	}
	bin := filepath.Join(t.TempDir(), "invoice")
	if runtime.GOOS == "windows" {
		bin += ".exe"
	}
	if out, err := exec.Command("go", "build", "-o", bin, ".").CombinedOutput(); err != nil {
		t.Fatalf("build: %v\n%s", err, out)
	}
	out, err := exec.Command(bin, ckpt, "../../testdata/glm_ocr/invoice.png").Output()
	if err != nil {
		t.Fatalf("run: %v\n%s", err, out)
	}
	var inv Invoice
	if err := json.Unmarshal(out, &inv); err != nil {
		t.Fatalf("output does not unmarshal into Invoice: %v\n%s", err, out)
	}
	if inv.Total != 1140.55 {
		t.Errorf("total %v, the page prints 1140.55\n%s", inv.Total, out)
	}
	t.Logf("%s", out)
}
