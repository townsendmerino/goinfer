//go:build darwin && goinfer_testhooks

package metal

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestG44_deltaNetInt8Declines is G44's tier 0 gate (docs/queue-correctness.md): qwen35-tiny at int8int8 does not go
// resident on Metal, where int8 projections would be re-quantized to int4; at int4 it builds as before. (int4mix declines by
// the same predicate, recurrentInt8Load, but is GGUF-only, so the tiny safetensors fixture cannot load it.) With the
// decline removed (dnetInt8DeclineOffForTest, the planted defect) the int8int8 build goes through, which turns it red.
func TestG44_deltaNetInt8Declines(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	dir := filepath.Join("..", "testdata", "qwen35-tiny")
	build := func(quant string) error {
		m, err := decoder.Load(dir, decoder.Options{Quant: quant})
		if err != nil {
			t.Fatalf("load at %s: %v", quant, err)
		}
		defer m.Close()
		r, err := buildResident(m)
		if err == nil {
			_ = r.Close()
		}
		return err
	}
	declines := func() bool {
		err := build("int8int8")
		return err != nil && strings.Contains(err.Error(), "int4 only")
	}
	if !declines() {
		t.Errorf("int8int8: buildResident did not give the int4-only decline")
	}
	if err := build("int4"); err != nil {
		t.Errorf("int4: buildResident = %v, want a build", err)
	}
	m, err := decoder.Load(dir, decoder.Options{Backend: "metal", Quant: "int8int8"})
	if err != nil {
		t.Fatal(err)
	}
	if m.ResidentForwardForTest() != nil {
		t.Errorf("a Metal load of qwen35-tiny at int8int8 went resident (%s)", m.DecodePath())
	}
	t.Logf("int8int8 through the backend: %s", m.DecodePath())
	_ = m.Close()

	dnetInt8DeclineOffForTest = true
	defer func() { dnetInt8DeclineOffForTest = false }()
	if declines() {
		t.Error("with the decline removed the int8int8 build still declines: the gate cannot see the defect")
	}
}
