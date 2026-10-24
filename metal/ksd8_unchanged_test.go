//go:build darwin && goinfer_testhooks

package metal

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestKSD8_otherFamiliesUnchanged is D8-G6's byte check: llama-tiny, qwen35-tiny and (with GOINFER_HEAVY_TESTS=1) the 1.5B's
// Metal .giw at int4, one batched pass over 100 seeded rows (fewer where the context is shorter), the last logits. GOINFER_KSD8_DUMP=dir writes them (run on the
// base build); GOINFER_KSD8_BASE=dir compares against what the base build wrote, byte for byte.
func TestKSD8_otherFamiliesUnchanged(t *testing.T) {
	dump, base := os.Getenv("GOINFER_KSD8_DUMP"), os.Getenv("GOINFER_KSD8_BASE")
	if dump == "" && base == "" {
		t.Skip("set GOINFER_KSD8_DUMP (base build) or GOINFER_KSD8_BASE (this build)")
	}
	paths := []string{"../testdata/llama-tiny", "../testdata/qwen35-tiny"}
	if os.Getenv("GOINFER_HEAVY_TESTS") != "" {
		paths = append(paths, os.ExpandEnv(fmt.Sprintf("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.int4.metal.v%d.giw", decoder.GIWFormatVersion)))
	}
	for _, p := range paths {
		m, err := decoder.Load(p, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 256})
		if err != nil {
			t.Fatalf("load %s: %v", p, err)
		}
		a, ok := m.ResidentForwardForTest().(*metalResident)
		if !ok || !a.r.prefillOK {
			m.Close()
			t.Fatalf("%s: the batched pass is not admitted", p)
		}
		rng := rand.New(rand.NewSource(73))
		ids := make([]int, min(100, a.r.ctxCap-1)) // qwen35-tiny's context is 64
		for i := range ids {
			ids[i] = rng.Intn(min(a.r.V, 1000))
		}
		a.r.resetRecurrent()
		lg, err := a.PrefillLast(context.Background(), getEmbs(a.r, ids), 0)
		m.Close()
		if err != nil {
			t.Fatalf("%s: prefill: %v", p, err)
		}
		var buf bytes.Buffer
		binary.Write(&buf, binary.LittleEndian, lg)
		name := filepath.Base(p) + ".logits"
		if dump != "" {
			if err := os.WriteFile(filepath.Join(dump, name), buf.Bytes(), 0o644); err != nil {
				t.Fatal(err)
			}
			continue
		}
		want, err := os.ReadFile(filepath.Join(base, name))
		if err != nil {
			t.Fatalf("%s: base logits: %v", p, err)
		}
		diff := 0
		for i := 0; i+4 <= len(want) && i+4 <= buf.Len(); i += 4 {
			if !bytes.Equal(want[i:i+4], buf.Bytes()[i:i+4]) {
				diff++
			}
		}
		fmt.Fprintf(os.Stderr, "[D8-G6] %s: %d of %d logits differ from the base build\n", p, diff, len(lg))
		if diff != 0 || len(want) != buf.Len() {
			t.Errorf("%s: %d logits differ from the base build", p, diff)
		}
	}
}
