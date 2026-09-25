//go:build darwin

package metal

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// The KV term Metal's resident guard — and, since docs/tasks/task-memory-accounting-2026-09.md, Plan("metal") —
// prices is decoder.Model.ResidentKVBytes("metal", …), which claims to be exact to what buildResident allocates.
// This checks that claim against the real buffers: every layer's K and V payload (NewBufferBytes: Len is bytes)
// plus the int8 per-head scales (NewBufferLen: Len is float32s). Fixtures cover a dense model, per-layer geometry
// with a sliding-window pattern (Metal allocates the full ctx on local layers too — the reason Metal cannot share
// kvBytesForCtx's sliding-window cap), a Gated-DeltaNet hybrid (no KV on linear layers), and an int8 KV cache;
// the ctx is not a multiple of Metal's 8-position padding, so the padding is exercised too.
func TestResidentKVBytes_matchesMetalAllocation(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	cases := []struct{ fixture, kv string }{
		{"llama-tiny", ""},
		{"llama-tiny", "i8"},
		{"gemma4-dense-twogeom-tiny", ""},
		{"gemma3-vl-tiny", ""},
		{"qwen35-tiny", ""},
	}
	const ctx = 100
	for _, c := range cases {
		t.Run(c.fixture+"/kv="+c.kv, func(t *testing.T) {
			dir := filepath.Join("..", "testdata", c.fixture)
			if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
				t.Skipf("no fixture: %v", err)
			}
			m, err := decoder.Load(dir, decoder.Options{Quant: "int4", KVPrecision: c.kv, ResidentContext: ctx})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer m.Close()
			want := residentKVBytes(m)
			r, err := buildResident(m)
			if err != nil {
				t.Skipf("not resident on Metal: %v", err)
			}
			defer r.Close()
			var got int64
			for l := range r.kc {
				got += int64(r.kc[l].Len() + r.vc[l].Len())
			}
			for l := range r.ks {
				got += 4 * int64(r.ks[l].Len()+r.vs[l].Len())
			}
			t.Logf("ctxCap %d: allocated %d B, ResidentKVBytes %d B", r.ctxCap, got, want)
			if got == 0 {
				t.Fatal("no KV buffers found — the comparison would be vacuous")
			}
			if got != want {
				t.Errorf("ResidentKVBytes(\"metal\") = %d B, Metal allocated %d B — the guard and Plan price a KV cache Metal does not build", want, got)
			}
		})
	}
}

// ResidentKVPrecision reports what the Metal resident runner allocates: f16 with no request and with an explicit
// f32 request (the f32 KV kernels are compiled out), i8 when int8 KV was asked for. Serve's banner prints it.
func TestResidentKVPrecision_metalReportsWhatRuns(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	dir := filepath.Join("..", "testdata", "llama-tiny")
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
		t.Skipf("no fixture: %v", err)
	}
	for req, want := range map[string]string{"": "f16", "f32": "f16", "i8": "i8"} {
		m, err := decoder.Load(dir, decoder.Options{Backend: "metal", Quant: "int4", KVPrecision: req})
		if err != nil {
			t.Fatalf("load (kv=%q): %v", req, err)
		}
		if !m.ResidentActive() {
			m.Close()
			t.Skipf("not resident on Metal: %s", m.ResidentDecline())
		}
		if got := m.ResidentKVPrecision(); got != want {
			t.Errorf("kv request %q: ResidentKVPrecision() = %q, want %q (what Metal allocates)", req, got, want)
		}
		m.Close()
	}
}
