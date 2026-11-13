package decoder

import (
	"fmt"
	"testing"
)

// hostKVFixture is a census fixture the guard prices and the CPU loads: its config as the guard reads it, and the loaded model.
type hostKVFixture struct {
	name, path string
	cfg        *Config
	m          *Model
}

// hostKVFixtures is every census fixture that loads on the CPU. A fixture the decoder cannot load is one the guard never prices
// either (its config does not resolve), so it is listed and passed over; none loading is VOID.
func hostKVFixtures(t *testing.T) []hostKVFixture {
	t.Helper()
	var out []hostKVFixture
	for _, fam := range censusFamilies(t) {
		cfg, err := censusConfig(fam.path, fam.dir)
		if err != nil {
			t.Logf("%s: no config (%v), passed over", fam.name, err)
			continue
		}
		m, err := censusLoad(fam.path, "f32")
		if err != nil {
			t.Logf("%s: does not load (%v), passed over", fam.name, err)
			continue
		}
		out = append(out, hostKVFixture{fam.name, fam.path, cfg, m})
	}
	if len(out) == 0 {
		t.Fatal("VOID: no census fixture loads")
	}
	return out
}

// hostKVContexts is the contexts a host KV price is checked at for a, each side of a sliding window included.
func hostKVContexts(a *Architecture) []int {
	ctxs := []int{1, 4096, 32768}
	if w := a.SlidingWindow; w > 1 {
		ctxs = append(ctxs, w-1, w, w+1)
	}
	return ctxs
}

// TestHostFitKV_isTheCPUCachesPrice: for every census fixture, the load-time host guard (fitCheckFor) prices a CPU load's KV at what
// the CPU cache holds (ResidentKVBytes("cpu"), which the CPU memory gate holds to NewCache), at every precision a caller can ask
// and each side of a sliding window, and the re-pricing a smaller fitting context uses is the same function.
func TestHostFitKV_isTheCPUCachesPrice(t *testing.T) {
	var checked, moved int
	for _, fx := range hostKVFixtures(t) {
		for _, ctx := range hostKVContexts(fx.m.w.arch) {
			for _, kv := range []string{"", "f16", "i8"} {
				o := Options{Backend: "cpu", ResidentContext: ctx, KVQuant: kv}
				qm, _ := parseQuant("f32")
				f := fitCheckFor(fx.path, "f32", qm, o)
				want := ctx
				if fx.cfg.MaxPositions > 0 {
					want = min(ctx, fx.cfg.MaxPositions)
				}
				if f.effCtx != want {
					t.Fatalf("%s ctx=%d kv=%q: the guard prices %d positions, want %d", fx.name, ctx, kv, f.effCtx, want)
				}
				price := fx.m.ResidentKVBytes("cpu", f.effCtx, kv == "f16", kv == "i8")
				where := fmt.Sprintf("%s ctx=%d kv=%q", fx.name, f.effCtx, kv)
				if f.kvBytes != price {
					t.Errorf("%s: the host guard prices %d bytes of KV, the CPU cache holds %d", where, f.kvBytes, price)
					moved++
				}
				if half := f.effCtx / 2; half > 0 {
					if got, held := f.kvAt(half), fx.m.ResidentKVBytes("cpu", half, kv == "f16", kv == "i8"); got != held {
						t.Errorf("%s: re-priced at %d positions, the guard says %d, the CPU cache holds %d", where, half, got, held)
					}
				}
				checked++
			}
		}
	}
	t.Logf("%d prices checked, %d differ", checked, moved)
}

// TestHostFitKV_isMetalsPrice: for every census fixture, a load that will be Metal-resident is priced by the host guards (fitCheckFor
// and the .giw guard's decideGIWFitAt) at what Metal's pools allocate for one slot (ResidentKVBytes("metal"), which Metal's
// allocation tests hold to its buffers): the precision ResolveKV delivers there and the full padded context on every attention
// layer, sliding ones included. A load Metal's gate declines falls back to the CPU and is priced as the CPU cache.
func TestHostFitKV_isMetalsPrice(t *testing.T) {
	withFakeMetalCompiled(t)
	qm, _ := parseQuant("f32")
	var metal, cpu, moved int
	for _, fx := range hostKVFixtures(t) {
		for _, pin := range []int{0, 1000, 32768} {
			for _, kv := range []string{"", "f16", "f32", "i8"} {
				o := Options{Backend: "metal", ResidentContext: pin, KVPrecision: kv}
				f := fitCheckFor(fx.path, "f32", qm, o)
				p, ok := kvPricingFor(fx.cfg, o)
				where := fmt.Sprintf("%s pin=%d kv=%q", fx.name, pin, kv)
				if !ok {
					continue
				}
				var price int64
				if p.metalSizing {
					m, err := censusLoadKV(fx.path, kv)
					if err != nil {
						t.Fatalf("%s: %v", where, err)
					}
					price = m.ResidentKVBytes("metal", p.ctx, false, false)
					metal++
				} else {
					price = fx.m.ResidentKVBytes("cpu", p.ctx, false, false)
					cpu++
				}
				if f.kvBytes != price {
					t.Errorf("%s ctx=%d: the host guard prices %d bytes of KV, the backend allocates %d", where, p.ctx, f.kvBytes, price)
					moved++
				}
				if half := p.ctx / 2; half > 0 && p.metalSizing {
					m, _ := censusLoadKV(fx.path, kv)
					if got, held := f.kvAt(half), m.ResidentKVBytes("metal", half, false, false); got != held {
						t.Errorf("%s: re-priced at %d positions, the guard says %d, Metal allocates %d", where, half, got, held)
					}
				}
			}
		}
	}
	if metal == 0 {
		t.Fatal("VOID: no fixture priced as Metal-resident")
	}
	t.Logf("%d Metal-resident and %d CPU-fallback prices checked, %d differ", metal, cpu, moved)
}

// censusLoadKV is censusLoad's f32 model loaded with KVPrecision kv, which ResidentKVBytes("metal") reads for the precision Metal
// delivers; "" is censusLoad's own.
func censusLoadKV(path, kv string) (*Model, error) {
	if kv == "" {
		return censusLoad(path, "f32")
	}
	key := path + "\x00f32\x00kv=" + kv
	if v, ok := censusLoaded.Load(key); ok {
		if m, ok := v.(*Model); ok {
			return m, nil
		}
		return nil, v.(error)
	}
	m, err := Load(path, Options{Backend: "cpu", Quant: "f32", KVPrecision: kv})
	if err != nil {
		censusLoaded.Store(key, err)
		return nil, err
	}
	censusLoaded.Store(key, m)
	return m, nil
}
