package decoder

import (
	"path/filepath"
	"strings"
	"testing"
)

// TestCUDAKVI8_resolves is S2's policy (docs/measurements/cuda-kv-i8-2026-11/scope.md): `--kv i8` on the CUDA resident is delivered for the
// dense families on one KV slot, and every other cell is refused by name. The CUDA policy the decoder registers is the one the cuda package
// registers too, so this reads the same answer a CUDA load does.
func TestCUDAKVI8_resolves(t *testing.T) {
	i8 := Options{Backend: "cuda", KVPrecision: kvPrecI8}
	i := 0
	for _, c := range []struct {
		dir     string
		o       func(*Options)
		deliver bool
		reason  string // a refusal's reason contains this
	}{
		{dir: "llama-tiny", deliver: true},
		{dir: "phi3-tiny", deliver: true},
		{dir: "llama-tiny", o: func(o *Options) { o.ResidentKVSlots, o.ResidentKVSlotsDefault = 4, true }, deliver: true},
		{dir: "llama-tiny", o: func(o *Options) { o.ResidentKVSlots = 2 }, reason: "one KV slot"},
		{dir: "llama-tiny", o: func(o *Options) { o.Drafter = true }, reason: "drafter"},
		{dir: "mistral-tiny-window", reason: "sliding-window"},
		{dir: "qwen3moe-tiny", reason: "qwen3_moe"},
		{dir: "gemma2-tiny", reason: "gemma2"},
		{dir: "cohere-tiny", reason: "cohere"},
		{dir: "qwen25vl-tiny", reason: "qwen2_5_vl"},
		{dir: "deepseek-tiny", reason: "deepseek"},
		{dir: "qwen35-tiny", reason: "qwen3_5"},
		{dir: "gemma4-moe-kv-tiny", reason: "gemma4"},
	} {
		i++
		o := i8
		if c.o != nil {
			c.o(&o)
		}
		arch := kvMatrixArch(t, c.dir)
		got, why := ResolveKV("cuda", arch, o)
		r := kvRefusal("cuda", arch, o)
		switch {
		case c.deliver && (got != kvPrecI8 || r != nil):
			t.Errorf("case %d (%s): delivered %q (%s), refusal %v; want i8", i, c.dir, got, why, r)
		case !c.deliver && r == nil:
			t.Errorf("case %d (%s): delivered %q, want a refusal naming %q", i, c.dir, got, c.reason)
		case !c.deliver && !strings.Contains(r.Reason, c.reason):
			t.Errorf("case %d (%s): refused with %q, want it to name %q", i, c.dir, r.Reason, c.reason)
		}
	}
	// auto and f16 are not moved by the int8 policy.
	arch := kvMatrixArch(t, "llama-tiny")
	if got, _ := ResolveKV("cuda", arch, Options{Backend: "cuda"}); got != kvPrecF16 {
		t.Errorf("llama-tiny under auto: %q, want f16 (unchanged)", got)
	}
}

// TestCUDAKVI8_oneSlot: a default slot count is lowered to one KV slot on an int8 CUDA cache; an f16 one keeps the request.
func TestCUDAKVI8_oneSlot(t *testing.T) {
	for _, c := range []struct {
		kv   string
		want int
	}{{kvPrecI8, 1}, {kvPrecF16, 4}} {
		m, err := Load(filepath.Join("..", "testdata", "llama-tiny"), Options{KVPrecision: c.kv, ResidentKVSlots: 4, ResidentKVSlotsDefault: true})
		if err != nil {
			t.Fatal(err)
		}
		if got := m.ResidentKVSlotsRequestFor("cuda"); got != c.want {
			t.Errorf("--kv %s, a default of 4 slots: CUDA is asked for %d, want %d", c.kv, got, c.want)
		}
		m.Close()
	}
}

// TestCUDAKVI8_prices is G3's arithmetic for the CUDA int8 cache, written out for llama-tiny (4 layers, 2 KV heads of 16): each position of
// each layer holds K and V at 1 B an element plus one 4 B scale per KV head for each, so a 64-row block of one layer is
// 2 * 64 * (32 + 2*4) = 5,120 B. At 128 positions (2 blocks) the slot is 4 layers * 2 blocks * 5,120 = 40,960 B, the null blocks one more
// block of every layer (20,480 B) and the tables 4 B per block per group (16 B): 61,456 B.
func TestCUDAKVI8_prices(t *testing.T) {
	m, err := Load(filepath.Join("..", "testdata", "llama-tiny"), Options{KVPrecision: kvPrecI8})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if got := m.ResidentKVBytes("cuda", 128, false, true); got != 61456 {
		t.Errorf("ResidentKVBytes(cuda, 128, i8) = %d B, want 61,456", got)
	}
	if got := m.CUDAKVSlotBytes(128, false, true); got != 40976 {
		t.Errorf("CUDAKVSlotBytes(128, i8) = %d B, want 40,976 (the slot's blocks and its tables)", got)
	}
	f16 := m.ResidentKVBytes("cuda", 128, true, false)
	if i8 := m.ResidentKVBytes("cuda", 128, false, true); i8 >= f16 {
		t.Errorf("i8 prices %d B, not under f16's %d B", i8, f16)
	}
	// A family the int8 cache does not cover prices what it would build: nothing, since the load is refused; asked anyway, f32.
	q, err := Load(filepath.Join("..", "testdata", "mistral-tiny-window"), Options{KVPrecision: kvPrecI8})
	if err != nil {
		t.Fatal(err)
	}
	defer q.Close()
	if i8, f32 := q.ResidentKVBytes("cuda", 128, false, true), q.ResidentKVBytes("cuda", 128, false, false); i8 != f32 {
		t.Errorf("mistral-tiny-window: i8 asked prices %d B, f32 %d B; a refused cell prices f32", i8, f32)
	}
}
