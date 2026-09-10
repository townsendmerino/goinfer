package decoder

import (
	"reflect"
	"testing"
	"unsafe"
)

// The recurrent-state census (audit 2026-09-10 C-03 / G-05): helpers that ASK THE STRUCT which
// recurrent kinds a KVCache holds, and generic fill / check-zero over them, so no test here carries
// a list of kinds or families to forget. hasRecurrentState's own comment predicted the failure —
// "the fourth kind will be added here, once, or it will be missed at four sites again" — and the
// fourth kind (KDA) was missed. With these, a fifth is covered the day its field is added.

// ownForwardModelType maps an ownForwards table name to the model_type representativeConfig
// takes. Only names that differ appear; an unknown family makes realCacheFor fail loudly.
func ownForwardModelType(name string) string {
	switch name {
	case "deepseek_v2/v3":
		return "deepseek_v3"
	case "gpt-oss":
		return "gpt_oss"
	}
	return name
}

// realCacheFor builds family f's REAL cache via NewCache from its representative config — the
// allocation production uses — rather than setting a cache field by hand.
func realCacheFor(t *testing.T, f ownForwardFamily) (*Model, *KVCache) {
	t.Helper()
	cfg := representativeConfig(ownForwardModelType(f.Name))
	if cfg == nil {
		t.Fatalf("representativeConfig has no case for own-forward family %q — add one, or this "+
			"family's cache is never inspected by the recurrent-state census", f.Name)
	}
	arch, _, err := resolveArchitecture(cfg)
	if err != nil {
		t.Fatalf("resolveArchitecture(%s): %v", f.Name, err)
	}
	if !f.is(arch) {
		t.Fatalf("representativeConfig(%q) does not resolve to an arch the table routes to %s", ownForwardModelType(f.Name), f.Name)
	}
	m := &Model{w: &Weights{arch: arch}}
	return m, m.NewCache(8)
}

// recurrentKinds returns KVCache's allocated recurrent-state kinds: every []*struct field except
// `rings`. The rule is chosen to FAIL SAFE. A new recurrent kind is included automatically; a new
// positional []*struct field is (wrongly) treated as recurrent and fails the census until someone
// decides what it is. `rings` is excluded because it is positional storage that TruncateTo DOES
// rewind (ring.truncate), reporting inexact only for a wrapped ring.
func recurrentKinds(c *KVCache) (names []string, fields []reflect.Value) {
	v := reflect.ValueOf(c).Elem()
	ty := v.Type()
	for i := 0; i < ty.NumField(); i++ {
		ft := ty.Field(i)
		if ft.Type.Kind() != reflect.Slice || ft.Type.Elem().Kind() != reflect.Pointer ||
			ft.Type.Elem().Elem().Kind() != reflect.Struct || ft.Name == "rings" {
			continue
		}
		if f := v.Field(i); !f.IsNil() {
			names = append(names, ft.Name)
			fields = append(fields, f)
		}
	}
	return names, fields
}

// writable returns a settable view of an unexported value. Reflection alone cannot set unexported
// fields; this is the one place the census needs to, and it is confined to tests.
func writable(v reflect.Value) reflect.Value {
	return reflect.NewAt(v.Type(), unsafe.Pointer(v.UnsafeAddr())).Elem()
}

// fillRecurrent writes nonzero values into every float field of every state in every recurrent
// kind — the state a real conversation would leave behind — allocating one state when a kind's
// slots are all nil, so a reset arm is never tested against an empty slate.
func fillRecurrent(t *testing.T, fields []reflect.Value) {
	t.Helper()
	for _, f := range fields {
		any := false
		for i := 0; i < f.Len(); i++ {
			if !f.Index(i).IsNil() {
				any = true
			}
		}
		if !any && f.Len() > 0 {
			writable(f.Index(0)).Set(reflect.New(f.Type().Elem().Elem()))
		}
		for i := 0; i < f.Len(); i++ {
			if f.Index(i).IsNil() {
				continue
			}
			st := f.Index(i).Elem()
			for j := 0; j < st.NumField(); j++ {
				w := writable(st.Field(j))
				switch w.Interface().(type) {
				case []float32:
					buf := make([]float32, max(w.Len(), 4))
					for k := range buf {
						buf[k] = 1.5
					}
					w.Set(reflect.ValueOf(buf))
				case [][]float32:
					w.Set(reflect.ValueOf([][]float32{{1, 2}, {3, 4}}))
				}
			}
		}
	}
}

// recurrentDirt lists every float field still holding state: a [][]float32 window that is not
// empty, or a []float32 matrix with a nonzero entry. Empty result = fully reset.
func recurrentDirt(names []string, fields []reflect.Value) []string {
	var dirt []string
	for n, f := range fields {
		for i := 0; i < f.Len(); i++ {
			if f.Index(i).IsNil() {
				continue
			}
			st := f.Index(i).Elem()
			for j := 0; j < st.NumField(); j++ {
				switch x := writable(st.Field(j)).Interface().(type) {
				case [][]float32:
					if len(x) > 0 {
						dirt = append(dirt, names[n]+"."+st.Type().Field(j).Name)
					}
				case []float32:
					for _, e := range x {
						if e != 0 {
							dirt = append(dirt, names[n]+"."+st.Type().Field(j).Name)
							break
						}
					}
				}
			}
		}
	}
	return dirt
}

// TestKVCache_recurrentKindsResetAndRewindHonestly pins the two cache-level halves of C-03 for
// EVERY recurrent kind, generically: a partial rewind must report inexact (the state has no
// per-position history), and TruncateTo(0) must zero every kind (the next conversation must not
// start on the last one's state).
func TestKVCache_recurrentKindsResetAndRewindHonestly(t *testing.T) {
	for _, f := range ownForwards {
		t.Run(f.Name, func(t *testing.T) {
			_, c := realCacheFor(t, f)
			names, fields := recurrentKinds(c)
			if len(names) == 0 {
				return // no recurrent kinds: nothing to rewind dishonestly
			}
			fillRecurrent(t, fields)
			for range 3 {
				c.Advance()
			}
			if c.TruncateTo(2) {
				t.Errorf("TruncateTo(2) on a 3-position cache holding %v reported exact — the state already "+
					"consumed position 2 and cannot be rewound, so a caller will warm-reuse a leaked state", names)
			}
			c.TruncateTo(0)
			if d := recurrentDirt(names, fields); len(d) > 0 {
				t.Errorf("after TruncateTo(0) these still hold the previous sequence's state: %v", d)
			}
		})
	}
}

// TestSession_bailingHybridNeverReusesKDAState drives C-03's own served failure sequence through
// the REAL caller — Session — not the cache helpers above: (b) an identical resend must not
// warm-reuse a prefix whose KDA state already consumed the reused tail, and (a) Reset, which is how
// sessionLRU evicts the coldest conversation at the default --kv-sessions, must not hand the next
// conversation this one's state.
func TestSession_bailingHybridNeverReusesKDAState(t *testing.T) {
	var fam ownForwardFamily
	for _, f := range ownForwards {
		if f.Name == "bailing_hybrid" {
			fam = f
		}
	}
	if fam.Name == "" {
		t.Fatal("bailing_hybrid is no longer in ownForwards — this gate would test nothing")
	}
	m, _ := realCacheFor(t, fam)
	s := m.NewSession(8)
	names, fields := recurrentKinds(s.cache)
	if len(names) == 0 {
		t.Fatal("the bailing_hybrid session cache holds no recurrent kinds — NewCache stopped allocating c.kda")
	}

	fillRecurrent(t, fields)
	for range 3 {
		s.cache.Advance()
	}
	s.tokens = []int{1, 2, 3}
	if got := s.rewindForReuse([]int{1, 2, 3}); got != 0 {
		t.Errorf("identical resend warm-reused %d positions over KDA state that already consumed them "+
			"(want 0: cold prefill)", got)
	}
	if d := recurrentDirt(names, fields); len(d) > 0 {
		t.Errorf("after the refused rewind the KDA state still holds the old sequence: %v", d)
	}

	fillRecurrent(t, fields)
	s.Reset()
	if d := recurrentDirt(names, fields); len(d) > 0 {
		t.Errorf("Session.Reset (sessionLRU eviction) left the previous conversation's state in %v", d)
	}
}
