//go:build cuda && goinfer_testhooks

package cuda

import (
	"encoding/json"
	"fmt"
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

// TestResidentKVBytes_matchesCUDAAllocation pins the CUDA half of docs/completed/task-memory-accounting-2026-09.md
// item 1: decoder.Model.ResidentKVBytes("cuda", …) — what Plan("cuda") and `fit` price — equals the K/V
// CUDA actually allocates, on real residents built from real fixtures, one per layout family:
//
//   - dense (llama-tiny, phi3-tiny, qwen2.5-0.5b), sliding window (mistral-tiny-window, gemma3-vl-tiny,
//     gemma3-1b: CUDA gives local layers the FULL context — the window is a mask, not a ring buffer),
//     per-layer geometry (gemma4-dense-twogeom, gemma4-moe), a DeltaNet hybrid (qwen35-tiny: no cache on
//     linear layers), and MLA (deepseek-tiny: ONE latent buffer per layer, not K and V);
//   - at every KV precision a caller can request: int8 asked prices the int8 pools where the CUDA int8 kernels cover the family (the dense
//     ones), whatever the load itself holds, and f32 where they do not.
//
// Without the "cuda" branch Plan's per-position formula prices MLA at twice the allocation and a
// requested f16 / i8 at a fraction of it (docs/measurements/memory-accounting-cuda-2026-09-25.md):
// deepseek-tiny and the f16 / i8 rows fail. Three links are checked, so a drift anywhere shows:
// the buffers' bytes == kvBytesForCap (the resident's own fit figure) == ResidentKVBytes("cuda"), and each
// buffer's driver allocation stays within one allocQuantumBytes (2 MiB) of its bytes.
//
// Every fixture is built with 2 resident KV slots (MC1, docs/tasks/parked/task-concurrency-2026-09.md) over the paged pools
// (kvpool.go): the pools and tables hold exactly 2 × kvBytesForCap plus the null blocks, and ResidentKVBytes("cuda") is
// a one-slot load's figure, kvBytesForCap plus the null blocks. The DeltaNet hybrid must build one slot (recurrent state
// is not part of a slot); every other layout, 2.
func TestResidentKVBytes_matchesCUDAAllocation(t *testing.T) {
	requireCUDADevice(t) // a host with no CUDA driver or device skips once, here; on a GPU host every priced fixture must build
	home, _ := os.UserHomeDir()
	fixtures := []string{
		"../../testdata/llama-tiny", "../../testdata/phi3-tiny", "../../testdata/mistral-tiny-window",
		"../../testdata/gemma3-vl-tiny", "../../testdata/gemma4-dense-twogeom-tiny", "../../testdata/gemma4-moe-tiny", "../../testdata/gemma4-emodel-tiny",
		"../../testdata/qwen35-tiny", "../../testdata/deepseek-tiny",
		filepath.Join(home, "models", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"),
		filepath.Join(home, "models", "gemma3-1b-q4_k_m.gguf"),
	}
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")
	ran := 0
	for _, p := range fixtures {
		t.Run(filepath.Base(p), func(t *testing.T) {
			if _, err := os.Stat(p); err != nil {
				t.Skipf("no fixture at %s", p)
			}
			// Phi-3 builds a resident only at int8int8 with per-32 activation scales (kv16Quant, kv16ActGroup), as the served bench loads it.
			m, err := decoder.Load(p, decoder.Options{Backend: "cuda", Quant: kv16Quant(p), ActQuantGroup: kv16ActGroup(p), ResidentKVSlots: 2})
			if err != nil {
				t.Skipf("load: %v", err)
			}
			defer m.Close()
			r, ok := m.ResidentForwardForTest().(*cudaResident)
			if !ok || r == nil {
				if kv8PricedFixtures[filepath.Base(p)] { // a fixture this test claims to price at int8 must run, not skip
					t.Fatalf("not CUDA-resident: %s", m.ResidentDecline())
				}
				t.Skipf("not CUDA-resident: %s", m.ResidentDecline())
			}
			ran++
			if r.KVSlots() != 2 {
				t.Fatalf("built %d KV slots with 2 requested", r.KVSlots())
			}
			// The pools hold every slot's blocks and the null blocks, and the tables 4 B per block per group per slot (kvpool.go).
			// A slot's guaranteed share is the plan's per-slot context (kvShareCtx); the tables are as long as one conversation's cap.
			forCap := kvBytesForCap(r.kvShareCtx, r.layers)
			tabs := func(ctx int) int64 { return int64(kvGroups * 4 * kvBlocksFor(ctx)) }
			null := kvPoolNullBytes(r.layers)
			bytes := int64(r.kvTabDev.Len()) * 4
			for l := range r.layers {
				if r.layers[l].kvShared {
					// An E-model's KV-shared layer holds its source's pools, not bytes of its own: counted once, at the owner. Assert the alias instead.
					if r.kc[l] != r.kc[r.layers[l].kvSrc] || r.vc[l] != r.vc[r.layers[l].kvSrc] {
						t.Errorf("layer %d: shared layer does not alias layer %d's pools", l, r.layers[l].kvSrc)
					}
					continue
				}
				for _, b := range []Buffer{r.kc[l], r.vc[l]} {
					n := int64(b.Len()) * 4 // Len is in elements: f32 (r.af), or f16 (r.akv) on a layer whose cache is half
					if r.layers[l].kvHalf {
						n = int64(b.Len()) * 2
					}
					bytes += n
					if alloc := (n + allocQuantumBytes - 1) / allocQuantumBytes * allocQuantumBytes; alloc-n >= allocQuantumBytes {
						t.Errorf("layer %d: a buffer of %d B rounds to %d B — more than one allocation quantum", l, n, alloc)
					}
				}
			}
			if want := int64(2)*(forCap-tabs(r.kvShareCtx)+tabs(r.ctxCap)) + null; bytes != want {
				t.Fatalf("the pools and tables hold %d B, %d slots of kvBytesForCap (%d B at ctx %d) and the null blocks (%d B) say %d B — the resident's own fit figure no longer matches its allocation",
					bytes, 2, forCap, r.ctxCap, null, want)
			}
			// What the pricing says against what was allocated: ResidentKVBytes is a one-slot load, kvBytesForCap and the null blocks. The resident's own
			// precision prices exactly its allocation. Asking for the other one prices the other; f16 on a model the f16 kernels do not cover is still f32 on
			// the device; int8 likewise, its pools where the kernels cover the family (TestKV8_deliveredBytes holds that figure against an int8 build).
			asF32 := slices.Clone(r.layers)
			for i := range asF32 {
				asF32[i].kvHalf = false
			}
			asF16 := slices.Clone(r.layers)
			for i := range asF16 {
				asF16[i].kvHalf = !asF16[i].noKV() && !asF16[i].kvShared
			}
			asI8 := slices.Clone(asF32)
			for i := range asI8 {
				asI8[i].kvI8 = !asI8[i].noKV() && !asI8[i].kvShared && !asI8[i].isMLA()
			}
			one := forCap + null
			f32Bytes := kvBytesForCap(r.kvShareCtx, asF32) + kvPoolNullBytes(asF32)
			i8Bytes := kvBytesForCap(r.kvShareCtx, asI8) + kvPoolNullBytes(asI8)
			f16Bytes := kvBytesForCap(r.kvShareCtx, asF16) + kvPoolNullBytes(asF16)
			// The one figure each request implies for this fixture. Written out here, not read from the policy, so a policy that mis-covers a family
			// shows: every family below has f16 kernels; the int8 kernels cover the dense ones without a sliding window (llama-tiny, phi3-tiny, the
			// 0.5B), and every other fixture asked for int8 is built, and priced, at f32.
			wantI8 := f32Bytes
			if kv8PricedFixtures[filepath.Base(p)] {
				wantI8 = i8Bytes
			}
			if r.kvF16 && one != f16Bytes || !r.kvF16 && one != f32Bytes {
				t.Errorf("the resident (kvF16=%v) allocates %d B a one-slot load; its precision's figure is f32 %d / f16 %d B", r.kvF16, one, f32Bytes, f16Bytes)
			}
			for _, c := range []struct {
				name     string
				f16, i8b bool
				want     int64
			}{{"f32", false, false, f32Bytes}, {"f16 requested", true, false, f16Bytes}, {"i8 requested", false, true, wantI8}} {
				got := m.ResidentKVBytes("cuda", r.kvShareCtx, c.f16, c.i8b)
				if got != c.want {
					t.Errorf("%s: ResidentKVBytes(\"cuda\", %d) = %d B, want %d B (f32 %d, f16 %d, i8 %d; resident kvF16=%v, allocates %d B a slot)",
						c.name, r.kvShareCtx, got, c.want, f32Bytes, f16Bytes, i8Bytes, r.kvF16, forCap)
				}
			}
		})
	}
	if ran == 0 {
		t.Skip("no fixture built a CUDA resident here")
	}
}

// kv8PricedFixtures are the fixtures of TestResidentKVBytes_matchesCUDAAllocation whose family the CUDA int8 kernels cover (dense, no sliding
// window): asked for int8, they price the int8 pools; every other fixture prices f32.
var kv8PricedFixtures = map[string]bool{"llama-tiny": true, "phi3-tiny": true, "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf": true}

// TestKVPool_detailNamesThePool is the CUDA half of KV4's §5.16 pool banner (docs/tasks/task-kv-architecture-2026-10.md): the pool's
// detail (decoder.KVPoolDetail) names the groups its layers use, the context positions kvPoolLine names, what a window group keeps per
// conversation, its 64-row blocks and its precision, and its bytes are what the plan charges for the pools and tables (kvChargedBytes at the guaranteed share,
// which TestKSH6C_marginPriced holds to the device buffers). A windowed pool also holds what kvGrantRollback added from free VRAM after
// the fit: on a stubbed card with no headroom it is exactly the plan's bytes, and with known headroom the plan's plus the grant's. The
// cases are one full group, one window group, and both (cohere2's sliding layers beside its full one).

func TestKVPool_detailNamesThePool(t *testing.T) {
	requireDeviceAndFixture(t, "../../testdata/llama-tiny")
	load := func(fixture, kv string, window bool) (*decoder.Model, *cudaResident) {
		m, err := decoder.Load(filepath.Join("../../testdata", fixture), decoder.Options{Backend: "cuda", Quant: "int4", ResidentKVSlots: 2, KVPrecision: kv, ResidentWindowedKV: window})
		if err != nil {
			t.Fatalf("%s at %s: load: %v", fixture, kv, err)
		}
		r, ok := m.ResidentForwardForTest().(*cudaResident)
		if !ok {
			m.Close()
			t.Fatalf("%s at %s: not CUDA-resident: %s", fixture, kv, m.ResidentDecline())
		}
		return m, r
	}
	for _, c := range []struct {
		fixture, kv string
		window      bool
		groups      []string
	}{
		{"llama-tiny", "f32", false, []string{"full"}},
		{"llama-tiny", "f16", false, []string{"full"}},
		{"mistral-tiny-window", "f32", true, []string{"sliding window"}},
		{"cohere2-tiny", "f32", true, []string{"full", "sliding window"}},
	} {
		// Known headroom, far past the grant's cap: the case does not read this card's free memory.
		restore := stubKVGrantFree(4 << 30)
		defer restore()
		m, r := load(c.fixture, c.kv, c.window)
		restore()
		d := m.ResidentKVPool()
		groups, windowRows := c.groups, 0
		if c.window {
			windowRows = r.kvSlotBlocks(kvGroupWin) * kvBlock
		}
		t.Logf("%s at %s: %+v (slack %d, planned %d)", c.fixture, c.kv, d, r.kvSlack, kvWindowSlack)
		want := r.kvSlotsN * r.kvShareBlocks * kvBlock
		if d.Positions != want || d.Guaranteed != r.kvShareCtx || d.BlockRows != kvBlock || d.WindowRows != windowRows || d.Precision != c.kv || !slices.Equal(d.Groups, groups) {
			t.Errorf("%s at %s: pool %+v, want %d positions, share %d, %d window rows, groups %v", c.fixture, c.kv, d, want, r.kvShareCtx, windowRows, groups)
		}
		if d.Positions < d.Guaranteed {
			t.Errorf("%s at %s: a pool of %d positions is smaller than one conversation's share of %d", c.fixture, c.kv, d.Positions, d.Guaranteed)
		}
		if line := fmt.Sprintf("KV pool: %d positions", d.Positions); !strings.Contains(r.kvPoolLine(), line) {
			t.Errorf("%s at %s: the build's line %q does not say %q", c.fixture, c.kv, r.kvPoolLine(), line)
		}
		bytes := kvChargedBytes(r.kvShareCtx, r.kvSlotsN, r.layers)
		if c.window {
			// The grant's bytes, beside the plan's: the slack the fit priced, and the one the grant raised it to.
			bytes += r.windowedPoolCharged(r.kvSlack, r.kvSlotsN) - r.windowedPoolCharged(kvWindowSlack, r.kvSlotsN)
			if r.kvSlack <= kvWindowSlack {
				t.Errorf("%s: slack %d: no grant ran, so this case reads nothing of it", c.fixture, r.kvSlack)
			}
		}
		if d.Bytes != bytes {
			t.Errorf("%s at %s: pool bytes %d, the plan and any grant charge %d", c.fixture, c.kv, d.Bytes, bytes)
		}
		m.Close()
		if !c.window {
			continue
		}
		// No headroom: no grant, so the pools are exactly what the fit priced.
		restore = stubKVGrantFree(0)
		defer restore()
		m, r = load(c.fixture, c.kv, c.window)
		restore()
		if b, p := m.ResidentKVPool().Bytes, kvChargedBytes(r.kvShareCtx, r.kvSlotsN, r.layers); r.kvSlack != kvWindowSlack || b != p {
			t.Errorf("%s with no headroom: slack %d (planned %d), pool bytes %d, the plan charges %d", c.fixture, r.kvSlack, kvWindowSlack, b, p)
		} else {
			t.Logf("%s with no headroom: slack %d, pool bytes %d, equal to the plan's", c.fixture, r.kvSlack, b)
		}
		m.Close()
	}
}

// stubKVGrantFree makes kvGrantRollback read free bytes of VRAM until restore runs.
func stubKVGrantFree(free uint64) (restore func()) {
	old := kvGrantFreeVRAM
	kvGrantFreeVRAM = func(*cudaResident) (uint64, error) { return free, nil }
	return func() { kvGrantFreeVRAM = old }
}

// TestKVPool_mixedGroupsAndPrecision reads KVPool's naming on a resident built by hand, with no device: the precision of pools at f16 and
// f32 together, the groups the owning layers use (a layer with no KV and a KV-shared one name none), and what a window group keeps.
func TestKVPool_mixedGroupsAndPrecision(t *testing.T) {
	r := &cudaResident{ctxCap: 4096, kvShareCtx: 4096, kvShareBlocks: 64, kvWin: true, kvWindow: 16, kvSlack: 512,
		layers: []cudaLayer{
			{kvDim: 8, kvHalf: true},                 // full, f16
			{kvDim: 8, kvWin: true},                  // window, f32
			{kvDim: 8, kvShared: true, kvWin: false}, // KV-shared: owns nothing
			{mixer: decoder.MixerNone},               // no KV
		}}
	r.kc, r.vc = make([]Buffer, len(r.layers)), make([]Buffer, len(r.layers))
	d := r.KVPool()
	if d.Precision != "f16 and f32" || !slices.Equal(d.Groups, []string{"full", "sliding window"}) || d.WindowRows != kvWindowBlocks(4096, 16+512)*kvBlock {
		t.Errorf("mixed: %+v", d)
	}
	r.layers[1].kvHalf = true
	if d := r.KVPool(); d.Precision != "f16" {
		t.Errorf("all f16: %+v", d)
	}
	r.layers[0].kvWin = true
	if d := r.KVPool(); !slices.Equal(d.Groups, []string{"sliding window"}) {
		t.Errorf("window only: %+v", d)
	}
	r.layers[0].kvHalf, r.layers[1].kvHalf, r.layers[0].kvWin, r.layers[1].kvWin = false, false, false, false
	if d := r.KVPool(); d.Precision != "f32" || !slices.Equal(d.Groups, []string{"full"}) || d.WindowRows != 0 {
		t.Errorf("all f32, full: %+v", d)
	}
}

// TestMM5_cudaPoolsAreTheDeclaredRows is G-MP2 on CUDA (docs/completed/task-forward-architecture-2026-10.md §4.9b): every census fixture
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
			if !ok && strings.Contains(m.ResidentDecline(), "the program declares") {
				t.Errorf("%s kv=%s: the build refuses the declaration as built: %s", f, kv, m.ResidentDecline())
			}
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
