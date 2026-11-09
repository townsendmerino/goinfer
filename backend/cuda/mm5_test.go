//go:build cuda && goinfer_testhooks

package cuda

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/diag/sb2census"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

// TestMM5_cudaPoolsAreTheDeclaredRows is G-MP2 on CUDA (docs/tasks/task-forward-architecture-2026-10.md §4.9b): every census fixture
// layer D admits at int4 and CUDA builds resident at -kv f16 and -kv i8, at two requested slots and 256 positions, holds each layer's pools at the rows its
// operations declare (decoder.Model.LayerBuffers): K and V, or MLA's one latent, of the declared width over the pool's rows, int8's
// scales at one per row and declared head, and nothing on a layer that declares no row. The build's whole KV (every pool, the scales and
// the tables) is what the plan prices for its slots, to the byte: a term priced twice or not at all is red either way.
func TestMM5_cudaPoolsAreTheDeclaredRows(t *testing.T) {
	requireCUDADevice(t)
	d, err := sb2census.ReadLayerD("../..", "cuda")
	if err != nil {
		t.Fatal(err)
	}
	pools, builds, int8s, f16s := 0, 0, 0, 0
	for _, f := range d.Fixtures {
		path, err := sb2census.Path("../..", f)
		if err != nil {
			t.Fatal(err)
		}
		if !d.Resident[f+" int4"] {
			continue
		}
		for _, kv := range []string{"f16", "i8"} {
			var mu sync.Mutex
			seen := map[*cudaResident]int64{}
			kvAllocSeenForTest = func(r *cudaResident, b int) { mu.Lock(); seen[r] += int64(b); mu.Unlock() }
			m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4", KVPrecision: kv, ResidentContext: 256, ResidentKVSlots: 2})
			kvAllocSeenForTest = nil
			if err != nil { // a precision CUDA does not deliver for the family is refused at load, by name
				t.Logf("%s kv=%s: %v", f, kv, err)
				continue
			}
			r, ok := m.ResidentForwardForTest().(*cudaResident)
			if !ok { // layer D admits it; CUDA's build declines it (its kernels' own limits), so no pool exists to check
				t.Logf("%s kv=%s: not CUDA-resident: %s", f, kv, m.ResidentDecline())
				m.Close()
				continue
			}
			builds++
			if r.kvI8 {
				int8s++
			}
			if r.kvF16 {
				f16s++
			}
			pools += mm5CheckCUDAPools(t, f+" kv="+kv, m, r, seen[r])
			m.Close()
		}
	}
	// The windowed cache, at production's slack. A tiny fixture's position limit holds its context under its window plus the slack, so
	// each runs from a copy whose config allows 2,048 positions (its weights linked), at 1,024. With no headroom no rollback grant raises
	// the slack past what the plan prices.
	restore := stubKVGrantFree(0)
	defer restore()
	windowed := 0
	for _, fx := range []string{"mistral-tiny-window", "gemma2-tiny", "cohere2-tiny"} {
		for _, kv := range []string{"f16", "f32"} {
			var mu sync.Mutex
			seen := map[*cudaResident]int64{}
			kvAllocSeenForTest = func(r *cudaResident, b int) { mu.Lock(); seen[r] += int64(b); mu.Unlock() }
			m, err := decoder.Load(mm5LongerFixture(t, fx, 2048), decoder.Options{Backend: "cuda", Quant: "int4", KVPrecision: kv, ResidentContext: 1024,
				ResidentKVSlots: 2, ResidentWindowedKV: true})
			kvAllocSeenForTest = nil
			if err != nil {
				t.Fatalf("%s kv=%s windowed: %v", fx, kv, err)
			}
			r, ok := m.ResidentForwardForTest().(*cudaResident)
			if !ok || !r.kvWin || r.kvSlack != kvWindowSlack {
				m.Close()
				t.Fatalf("%s kv=%s: the windowed cache did not engage at production's slack (resident %v)", fx, kv, ok)
			}
			windowed++
			pools += mm5CheckCUDAPools(t, fx+" kv="+kv+" windowed", m, r, seen[r])
			m.Close()
		}
	}
	t.Logf("%d pools on %d builds (%d int8, %d f16, %d windowed) held to their declared rows and to the plan", pools, builds+windowed, int8s, f16s, windowed)
	if pools < 150 || int8s < 4 || f16s < 25 {
		t.Fatalf("only %d pools, %d int8 and %d f16 builds checked: the test reads too little", pools, int8s, f16s)
	}
}

// mm5LongerFixture is a copy of testdata/fx whose config allows positions positions: its other files linked, its config rewritten.
func mm5LongerFixture(t *testing.T, fx string, positions int) string {
	t.Helper()
	src, dst := filepath.Join("../../testdata", fx), t.TempDir()
	ents, err := os.ReadDir(src)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range ents {
		abs, _ := filepath.Abs(filepath.Join(src, e.Name()))
		if e.Name() != "config.json" {
			if err := os.Symlink(abs, filepath.Join(dst, e.Name())); err != nil {
				t.Fatal(err)
			}
			continue
		}
		b, err := os.ReadFile(abs)
		if err != nil {
			t.Fatal(err)
		}
		var cfg map[string]any
		if err := json.Unmarshal(b, &cfg); err != nil {
			t.Fatal(err)
		}
		if _, ok := cfg["max_position_embeddings"]; !ok {
			t.Fatalf("%s: no max_position_embeddings to raise", fx)
		}
		cfg["max_position_embeddings"] = positions
		if b, err = json.Marshal(cfg); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dst, "config.json"), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dst
}

// mm5CheckCUDAPools holds r's pools to m's declared rows and r's whole KV (alloc: the pool and scale bytes the build allocated, read from
// the allocations) to the plan, and returns the pools it checked.
func mm5CheckCUDAPools(t *testing.T, f string, m *decoder.Model, r *cudaResident, alloc int64) int {
	t.Helper()
	pools := 0
	for l := range r.layers {
		Ly := &r.layers[l]
		var rows []program.Buf
		for _, b := range m.LayerBuffers(l) {
			if b.Unit == program.PerPosition {
				rows = append(rows, b)
			}
		}
		if Ly.kvShared {
			if len(rows) > 0 {
				t.Errorf("%s layer %d reads layer %d's KV but declares %d rows", f, l, Ly.kvSrc, len(rows))
			}
			if r.kc[l] != r.kc[Ly.kvSrc] || r.vc[l] != r.vc[Ly.kvSrc] {
				t.Errorf("%s layer %d reads layer %d's KV but holds pools of its own", f, l, Ly.kvSrc)
			}
			continue
		}
		if len(rows) == 0 {
			if r.kc[l] != (Buffer{}) || r.vc[l] != (Buffer{}) {
				t.Errorf("%s layer %d declares no KV row but holds a pool", f, l)
			}
			continue
		}
		n := r.kvPools[r.kvGroupOf(l)].blocks * kvBlock // the pool's rows: every slot's blocks and the null block
		bufs := map[string]Buffer{"k": r.kc[l], "v": r.vc[l], "latent": r.kc[l]}
		scales := map[string]Buffer{"k": r.ksc[l], "v": r.vsc[l]}
		for _, b := range rows {
			if got, want := bufs[b.Name].Len(), n*b.Elems; got != want {
				t.Errorf("%s layer %d: its %s pool holds %d elements; the declared %d-element row over %d rows is %d", f, l, b.Name, got, b.Elems, n, want)
			}
			if Ly.kvI8 && b.Prec == program.PrecKV {
				if got, want := scales[b.Name].Len(), n*b.Heads; got != want {
					t.Errorf("%s layer %d: its %s scales hold %d; one a row and declared head is %d", f, l, b.Name, got, want)
				}
			}
			pools++
		}
	}
	// The plan for r's slots: each slot's blocks at its guaranteed share with tables as long as one conversation's cap, and the null
	// blocks once (TestResidentKVBytes_matchesCUDAAllocation states the layout).
	f16, i8 := r.kvF16, r.kvI8
	share, slots := r.kvShareCtx, int64(r.kvSlotsN)
	tabs := func(ctx int) int64 { return int64(kvGroups * 4 * kvBlocksFor(ctx)) }
	slot := m.CUDAKVSlotBytes(share, f16, i8)
	null := m.ResidentKVBytes("cuda", share, f16, i8) - slot
	got := alloc + int64(r.kvTabDev.Len())*4
	if want := slots*(slot-tabs(share)+tabs(r.ctxCap)) + null; got != want {
		t.Errorf("%s: the build holds %d bytes of KV over %d slots at a %d-position share; the plan prices %d", f, got, slots, share, want)
	}
	return pools
}

// TestMM5_cudaBuildRefusesADeclarationItsKernelsDoNotAddress: CUDA's build checks each layer's declared per-position rows against what
// its kernels address before it sizes the pools, and refuses a mismatch. On llama-tiny's layer 0: K one element short, V with one head
// too many, K at f32, K with a window on a full-attention layer, a row no pool has, K twice and V missing; on deepseek-tiny's MLA layer 0,
// the latent at the KV precision (CUDA's latent has no int8 form) and one element short; on qwen35-tiny, a DeltaNet layer, which holds no
// pool, declaring K and V. The declarations as built pass.
func TestMM5_cudaBuildRefusesADeclarationItsKernelsDoNotAddress(t *testing.T) {
	requireCUDADevice(t)
	build := func(m *decoder.Model) error {
		rf, ok, err := (&cudaBackend{}).BuildResident(m)
		if c, isC := rf.(interface{ Close() error }); isC && rf != nil {
			c.Close()
		}
		if err == nil && !ok {
			t.Fatal("the build neither built nor refused")
		}
		return err
	}
	load := func(fx string) (*decoder.Model, []program.Buf, []program.Buf) {
		m, err := decoder.Load(fx, decoder.Options{Backend: "cuda", Quant: "int4", ResidentContext: 256})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { m.Close() })
		var rows, rest []program.Buf
		for _, b := range m.LayerBuffers(0) {
			if b.Unit == program.PerPosition {
				rows = append(rows, b)
			} else {
				rest = append(rest, b)
			}
		}
		return m, rows, rest
	}
	plant := func(m *decoder.Model, name string, all []program.Buf) {
		m.SetLayerBuffersForTest(0, all)
		if err := build(m); err == nil {
			t.Errorf("%s: the build accepted it", name)
		} else if !strings.Contains(err.Error(), "declares") {
			t.Errorf("%s: refused for another reason: %v", name, err)
		}
	}
	m, rows, rest := load("../../testdata/llama-tiny")
	if len(rows) != 2 || rows[0].Name != "k" || rows[1].Name != "v" {
		t.Fatalf("llama-tiny's layer 0 declares %+v per position; the test needs its K and V", rows)
	}
	edit := func(rows, rest []program.Buf, f func([]program.Buf) []program.Buf) []program.Buf {
		return append(f(slices.Clone(rows)), rest...)
	}
	for _, c := range []struct {
		name string
		f    func([]program.Buf) []program.Buf
	}{
		{"K one element short", func(b []program.Buf) []program.Buf { b[0].Elems--; return b }},
		{"V with one head too many", func(b []program.Buf) []program.Buf { b[1].Heads++; return b }},
		{"K at f32", func(b []program.Buf) []program.Buf { b[0].Prec = program.PrecF32; return b }},
		{"K windowed on a full layer", func(b []program.Buf) []program.Buf { b[0].Window = 16; return b }},
		{"a row no pool has", func(b []program.Buf) []program.Buf {
			return append(b, program.Buf{Name: "x", Unit: program.PerPosition, Prec: program.PrecKV, Elems: b[0].Elems})
		}},
		{"K twice", func(b []program.Buf) []program.Buf { return append(b, b[0]) }},
		{"V missing", func(b []program.Buf) []program.Buf { return b[:1] }},
	} {
		plant(m, "llama-tiny: "+c.name, edit(rows, rest, c.f))
	}
	m.SetLayerBuffersForTest(0, append(slices.Clone(rows), rest...))
	if err := build(m); err != nil {
		t.Fatalf("llama-tiny as declared: %v", err)
	}
	ds, lat, drest := load("../../testdata/deepseek-tiny")
	if len(lat) != 1 || lat[0].Name != "latent" {
		t.Fatalf("deepseek-tiny's layer 0 declares %+v per position; the test needs its latent", lat)
	}
	plant(ds, "deepseek-tiny: the latent at the KV precision", edit(lat, drest, func(b []program.Buf) []program.Buf { b[0].Prec = program.PrecKV; return b }))
	plant(ds, "deepseek-tiny: the latent one element short", edit(lat, drest, func(b []program.Buf) []program.Buf { b[0].Elems--; return b }))
	ds.SetLayerBuffersForTest(0, append(slices.Clone(lat), drest...))
	if err := build(ds); err != nil {
		t.Fatalf("deepseek-tiny as declared: %v", err)
	}
	q, qrows, qrest := load("../../decoder/selftestdata/qwen35-tiny")
	if len(qrows) != 0 {
		t.Fatalf("qwen35-tiny's layer 0 declares %+v per position; the test needs a DeltaNet layer", qrows)
	}
	plant(q, "qwen35-tiny: a DeltaNet layer declaring K and V", append(slices.Clone(rows), qrest...))
	q.SetLayerBuffersForTest(0, qrest)
	if err := build(q); err != nil {
		t.Fatalf("qwen35-tiny as declared: %v", err)
	}
}
