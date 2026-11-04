//go:build cuda

package cuda

import (
	"fmt"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

type recorder struct{ msgs []string }

func (r *recorder) Helper()                   {}
func (r *recorder) Fatalf(f string, a ...any) { r.msgs = append(r.msgs, fmt.Sprintf(f, a...)) }

var mellumFlags = []string{"-backend", "cuda", "-ctx", "32768", "-moe-cache-experts"}

// serveOptions carries serve's defaults for the fields that moved the plan in B37, and the flags given.
func TestServeOptions_carryServesDefaults(t *testing.T) {
	o := serveOptions(t, mellumFlags...)
	if !o.ResidentWindowedKV {
		t.Error("ResidentWindowedKV is off: serve's --windowed-kv defaults to on")
	}
	if o.ResidentContext != 32768 || !o.MoECacheExperts || o.Backend != "cuda" {
		t.Errorf("flags not applied: ctx %d, moe cache %v, backend %q", o.ResidentContext, o.MoECacheExperts, o.Backend)
	}
	if o.ResidentKVSlots != serveKVSessions || !o.ResidentKVSlotsDefault {
		t.Errorf("KV slots %d (default flag %v), want serve's request of %d as a default", o.ResidentKVSlots, o.ResidentKVSlotsDefault, serveKVSessions)
	}
	if !o.EmbedInt4 || o.Quant != "int4" {
		t.Errorf("EmbedInt4 %v, Quant %q: serve defaults to both on/int4", o.EmbedInt4, o.Quant)
	}
	// and --windowed-kv=false must reach it, or the helper would pin the default regardless of the flags
	if serveOptions(t, append([]string{"-windowed-kv=false"}, mellumFlags...)...).ResidentWindowedKV {
		t.Error("-windowed-kv=false did not reach Options.ResidentWindowedKV")
	}
}

// The harness before B37: hand-written options, windowed KV off, one KV slot requested. The guard must name both, and a harness on serveOptions must pass.
// This is the plant: it reproduces the defect, so the guard's VOID case is shown to fire on the very options that planned 26 slots.
func TestServeOptions_guardVoidsTheB37HarnessOptions(t *testing.T) {
	old := decoder.Options{Backend: "cuda", MoECacheExperts: true, ResidentContext: 32768, Quant: "int4", EmbedInt4: true}
	d := serveShapeDiff(t, old, mellumFlags...)
	joined := strings.Join(d, "; ")
	for _, want := range []string{"ResidentWindowedKV: got false, serve true", "ResidentKVSlots: got 0, serve 4"} {
		if !strings.Contains(joined, want) {
			t.Errorf("the diff misses %q: %v", want, d)
		}
	}
	var rec recorder
	requireServeShaped(t, &rec, old, mellumFlags...)
	if len(rec.msgs) != 1 || !strings.HasPrefix(rec.msgs[0], "VOID:") {
		t.Errorf("the guard did not VOID the old options: %v", rec.msgs)
	}
	rec = recorder{}
	requireServeShaped(t, &rec, serveOptions(t, mellumFlags...), mellumFlags...)
	if len(rec.msgs) != 0 {
		t.Errorf("the guard VOIDed serve's own options: %v", rec.msgs)
	}
	// Each memory-moving field is watched: perturbing any one of them on serve's options is a diff.
	base := serveOptions(t, mellumFlags...)
	for name, mut := range map[string]func(*decoder.Options){
		"ResidentWindowedKV": func(o *decoder.Options) { o.ResidentWindowedKV = !o.ResidentWindowedKV },
		"ResidentKVSlots":    func(o *decoder.Options) { o.ResidentKVSlots++ },
		"ResidentContext":    func(o *decoder.Options) { o.ResidentContext++ },
		"MoECacheExperts":    func(o *decoder.Options) { o.MoECacheExperts = !o.MoECacheExperts },
		"EmbedInt4":          func(o *decoder.Options) { o.EmbedInt4 = !o.EmbedInt4 },
		"KVPrecision":        func(o *decoder.Options) { o.KVPrecision = "f16" },
	} {
		o := base
		mut(&o)
		if len(serveShapeDiff(t, o, mellumFlags...)) == 0 {
			t.Errorf("a change to %s is not seen by the guard", name)
		}
	}
}
