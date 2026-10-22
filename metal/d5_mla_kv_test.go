//go:build darwin && goinfer_testhooks

package metal

import (
	"errors"
	"fmt"
	"hash/fnv"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/loadflags"
)

// d5Opts is a Metal load of --kv v as serve sets it (KVPrecision and the CPU cache's KVQuant).
func d5Opts(v string) decoder.Options {
	kv := v
	if v == "auto" {
		kv = ""
	}
	return decoder.Options{Backend: "metal", Quant: "int4", KVPrecision: kv, KVQuant: loadflags.CPUKV("", kv), ResidentContext: 64}
}

// d5LogitsHash is an FNV-64a over the bits of the Metal resident's logits for 32 teacher-forced positions.
func d5LogitsHash(t *testing.T, m *decoder.Model) uint64 {
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Fatalf("not Metal-resident: %s", m.ResidentDecline())
	}
	_, _, _, _, _, _, vocab := m.Dims()
	h := fnv.New64a()
	a.Reset()
	for i := range 32 {
		l, err := a.Forward(m.EmbedResidentForTest((i*37+11)%vocab), i)
		if err != nil {
			t.Fatal(err)
		}
		for _, x := range l {
			b := math.Float32bits(x)
			h.Write([]byte{byte(b), byte(b >> 8), byte(b >> 16), byte(b >> 24)})
		}
	}
	return h.Sum64()
}

// TestD5_mlaOnMetal is D5-G2 and D5-G3 (docs/tasks/task-kv-precision-everywhere-2026-10.md, "D5 gates"): an MLA model on Metal
// refuses an explicit --kv f16 or i8 on the metal path naming the f32 latent, and its auto and f32 logits are byte-equal to the
// build before D5 (the hashes below).
func TestD5_mlaOnMetal(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	dir := p3FixtureDir("deepseek-tiny")
	want := map[string]uint64{"auto": 0x7e63e08991a5ac55, "f32": 0x7e63e08991a5ac55} // before D5
	for _, v := range []string{"auto", "f32"} {
		m, err := decoder.Load(dir, d5Opts(v))
		if err != nil {
			t.Fatalf("--kv %s: %v", v, err)
		}
		got := d5LogitsHash(t, m)
		m.Close()
		fmt.Fprintf(os.Stderr, "[D5-G3] deepseek-tiny --kv %s: logits hash %#x\n", v, got)
		if got != want[v] {
			t.Errorf("--kv %s: logits hash %#x, before D5 %#x", v, got, want[v])
		}
	}
	for _, v := range []string{"f16", "i8"} {
		m, err := decoder.Load(dir, d5Opts(v))
		var r *decoder.KVRefusal
		if !errors.As(err, &r) {
			if m != nil {
				m.Close()
			}
			t.Errorf("--kv %s: Load returned %v, want a KVRefusal", v, err)
			continue
		}
		fmt.Fprintf(os.Stderr, "[D5-G2] deepseek-tiny --kv %s: %v\n", v, err)
		if r.Path != "metal" || !strings.Contains(r.Reason, "latent") {
			t.Errorf("--kv %s: refused on %q with %q, want the metal path naming the latent", v, r.Path, r.Reason)
		}
	}
}
