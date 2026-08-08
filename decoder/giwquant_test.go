package decoder

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/giw"
)

// TestCheckGiwQuantMatch is the T1-7 gate: a --quant that cannot take effect on an already-baked
// .giw must become a startup error naming the constraint, the requested value, the baked value, and
// the file — not be silently dropped. It bakes a real int4 .giw, loads it (so GiwPath is set), and
// exercises every case, including the two that must stay quiet (a matching quant and a bare
// default) and the one that proves a non-.giw model is untouched.
func TestCheckGiwQuantMatch(t *testing.T) {
	path := prequantGGUF(t)

	// Load at int4 and bake a .giw to disk.
	m, err := Load(path, Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load int4: %v", err)
	}
	blob, serr := SerializeWeights(m.w, "t17")
	m.Close()
	if serr != nil {
		t.Fatalf("serialize: %v", serr)
	}
	giwPath := filepath.Join(t.TempDir(), "q05.int4.giw")
	if err := os.WriteFile(giwPath, giw.Write(blob, nil), 0o644); err != nil {
		t.Fatalf("write .giw: %v", err)
	}

	g, err := Load(giwPath, Options{})
	if err != nil {
		t.Fatalf("load .giw: %v", err)
	}
	defer g.Close()
	if g.GiwPath() == "" {
		t.Fatal("GiwPath is empty — the check keys on it")
	}
	if got := g.Quant(); got != "int4" {
		t.Fatalf("baked bundle Quant() = %q, want int4", got)
	}

	// Match → nil. This is the normal cross-format case (`--quant int4` over a mix of GGUF and a
	// .giw baked at int4); it must proceed WITHOUT a warning.
	if err := g.CheckGiwQuantMatch("int4"); err != nil {
		t.Errorf("matching quant must pass silently: %v", err)
	}
	// Empty request (the user relied on the default) → nil. The default must never conflict.
	if err := g.CheckGiwQuantMatch(""); err != nil {
		t.Errorf("empty (default) request must never conflict: %v", err)
	}
	// Mismatch → error naming the requested value, the baked value, the file, and the flag.
	err = g.CheckGiwQuantMatch("int8int8")
	if err == nil {
		t.Fatal("an explicit int8int8 request on an int4 .giw must error, not be dropped")
	}
	msg := err.Error()
	for _, want := range []string{"int8int8", "int4", giwPath, "--quant"} {
		if !strings.Contains(msg, want) {
			t.Errorf("error %q is missing %q", msg, want)
		}
	}
	t.Logf("T1-7 startup error: %s", msg)

	// A non-.giw model (GiwPath empty) is never subject to the check — --quant is honored at load,
	// so a "mismatch" there is meaningless.
	direct, err := Load(path, Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("direct load: %v", err)
	}
	defer direct.Close()
	if direct.GiwPath() != "" {
		t.Fatalf("a direct GGUF load should have an empty GiwPath, got %q", direct.GiwPath())
	}
	if err := direct.CheckGiwQuantMatch("int8int8"); err != nil {
		t.Errorf("a non-.giw model must be a no-op for the check, got: %v", err)
	}
}
