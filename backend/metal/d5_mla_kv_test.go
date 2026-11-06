//go:build darwin && goinfer_testhooks

package metal

import (
	"bytes"
	"errors"
	"fmt"
	"hash/fnv"
	"math"
	"os"
	"slices"
	"strings"
	"testing"

	gpu "github.com/townsendmerino/goinfer/backend/metal/internal/gpudev"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/app/loadflags"
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

// TestKVPages_mlaOneSlot: a one-slot MLA build holds its pool as every other build does. Slot 0's table names its latent rows past the
// zero null block, and the logits are the first slot's of a two-slot build.
func TestKVPages_mlaOneSlot(t *testing.T) {
	dir := p3FixtureDir("deepseek-tiny")
	if _, err := os.Stat(dir); err != nil {
		t.Skipf("no fixture: %v", err)
	}
	load := func(slots int) (*decoder.Model, *resident) {
		o := d5Opts("auto")
		o.ResidentKVSlots = slots
		m, err := decoder.Load(dir, o)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { m.Close() })
		a, ok := m.ResidentForwardForTest().(*metalResident)
		if !ok {
			t.Fatalf("not Metal-resident: %s", m.ResidentDecline())
		}
		if a.r.mla == nil || a.r.kvSlotCount() != slots {
			t.Fatalf("MLA %v, %d KV slots; want MLA and %d", a.r.mla != nil, a.r.kvSlotCount(), slots)
		}
		return m, a.r
	}
	m, r := load(1)
	got := d5LogitsHash(t, m)
	m2, _ := load(2)
	if want := d5LogitsHash(t, m2); got != want {
		t.Errorf("one slot's logits hash %#x, the first of two slots' %#x", got, want)
	}
	if err := r.useKVSlot(0); err != nil {
		t.Fatal(err)
	}
	read := func(b Buffer, n int) []byte {
		h := make([]byte, n)
		if err := gpu.Download(b, h); err != nil {
			t.Fatal(err)
		}
		return h
	}
	seen := false
	for l := range r.layers {
		null := r.kvNullBytes[l]
		if r.layers[l].mla == nil || null == 0 {
			continue
		}
		seen = true
		const rows = 8
		n := rows * r.mla.latDim * 4
		if !bytes.Equal(read(r.kc[l].At(0), null), make([]byte, null)) {
			t.Errorf("layer %d: the null block is not zero", l)
		}
		written := read(r.kc[l].At(int(r.kvTab.U32s()[0])*null), n) // the first block slot 0's table names
		if bytes.Equal(written, make([]byte, n)) {
			t.Errorf("layer %d: slot 0's first block holds %d zero bytes where decode wrote latent rows", l, n)
		}
		if !bytes.Equal(written, read(r.kc[l].At(null), n)) {
			t.Errorf("layer %d: slot 0's first block is not the rows past the null block", l)
		}
	}
	if !seen {
		t.Fatal("no MLA layer holds a pool")
	}
}

// TestKVPages_mlaThroughTheTable: an MLA build stores and reads its latent rows through the table, so blocks handed out reversed and
// interleaved across slots decode as in-order blocks bit for bit, on two slots in turn past a block edge. The plant swaps slot 0's first
// two entries under a filled cache and must move the next token's logits.
func TestKVPages_mlaThroughTheTable(t *testing.T) {
	dir := p3FixtureDir("deepseek-tiny")
	if _, err := os.Stat(dir); err != nil {
		t.Skipf("no fixture: %v", err)
	}
	const depth = 80
	run := func(scramble, plant bool) [][]float32 {
		kvScrambleForTest = scramble
		defer func() { kvScrambleForTest = false }()
		o := d5Opts("auto")
		o.ResidentKVSlots, o.ResidentContext = 2, 256
		m, err := decoder.Load(dir, o)
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		a, ok := m.ResidentForwardForTest().(*metalResident)
		if !ok {
			t.Fatalf("not Metal-resident: %s", m.ResidentDecline())
		}
		r := a.r
		if r.mla == nil || r.kvSlotCount() != 2 {
			t.Fatalf("MLA %v, %d KV slots; want MLA and 2", r.mla != nil, r.kvSlotCount())
		}
		var out [][]float32
		for s := range 2 {
			if err := r.useKVSlot(s); err != nil {
				t.Fatal(err)
			}
			for pos := range depth {
				out = append(out, slices.Clone(r.ForwardEmb(mc3Emb(r, (pos*7+s*13+1)%r.V), pos)))
			}
		}
		if err := r.useKVSlot(0); err != nil {
			t.Fatal(err)
		}
		if plant {
			r.stopExec()
			tab := r.kvTab.U32s()
			tab[0], tab[1] = tab[1], tab[0]
		}
		return append(out, slices.Clone(r.ForwardEmb(mc3Emb(r, 3), depth)))
	}
	differ := func(a, b [][]float32) int {
		n := 0
		for i := range a {
			for j := range a[i] {
				if math.Float32bits(a[i][j]) != math.Float32bits(b[i][j]) {
					n++
				}
			}
		}
		return n
	}
	inOrder := run(false, false)
	if n := differ(inOrder, run(true, false)); n != 0 {
		t.Errorf("%d logits differ between scrambled and in-order blocks", n)
	}
	if n := differ(inOrder, run(false, true)); n == 0 {
		t.Error("two table entries swapped under a filled cache moved no logit: MLA does not read the table")
	}
}
