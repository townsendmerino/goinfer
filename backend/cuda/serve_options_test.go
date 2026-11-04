//go:build cuda

package cuda

import (
	"flag"
	"fmt"
	"io"
	"reflect"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/loadflags"
)

// The real-model harnesses load through serveOptions: the decoder.Options `goinfer serve` builds for the same flags (internal/loadflags, the code serve
// runs, plus serve's own -kv-sessions request), so a flag serve gains reaches the harness and a load takes the memory serve's takes. A harness that wrote its
// own Options{Backend, Quant, ResidentContext, MoECacheExperts} left ResidentWindowedKV off (serve's default is on) and planned 26 expert slots a layer where the
// server planned 53 (docs/queue-engineering.md B37).

// serveKVSessions is serve's -kv-sessions default (internal/serveapp/main.go): the number of KV slots it asks a resident backend for.
const serveKVSessions = 4

// serveOptions returns the decoder.Options serve would load a model with for these flags (e.g. "-backend", "cuda", "-ctx", "32768", "-moe-cache-experts").
func serveOptions(tb testing.TB, flags ...string) decoder.Options {
	tb.Helper()
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	f := loadflags.Register(fs, loadflags.Serve)
	if err := fs.Parse(flags); err != nil {
		tb.Fatalf("serveOptions: %v", err)
	}
	if err := f.Validate(); err != nil {
		tb.Fatalf("serveOptions: %v", err)
	}
	o := f.Options()
	o.ResidentKVSlots = serveKVSessions
	o.ResidentKVSlotsDefault = true // -kv-sessions was not given
	return o
}

// memoryOptionFields are the decoder.Options fields that move how much device or host memory a load takes, or which path it takes.
var memoryOptionFields = []string{
	"Backend", "Quant", "KVPrecision", "ResidentContext", "MoECacheExperts", "MoECacheSlots", "EmbedInt4",
	"ResidentWindowedKV", "ResidentKVSlots", "ResidentKVSlotsDefault", "StreamWeights", "ExactPrefill",
}

// serveShapeDiff lists the memory-moving fields on which got differs from serveOptions(flags...), as "Field: got X, serve Y".
func serveShapeDiff(tb testing.TB, got decoder.Options, flags ...string) []string {
	tb.Helper()
	want := reflect.ValueOf(serveOptions(tb, flags...))
	have := reflect.ValueOf(got)
	var out []string
	for _, name := range memoryOptionFields {
		g, w := have.FieldByName(name), want.FieldByName(name)
		if !g.IsValid() {
			tb.Fatalf("serveShapeDiff: decoder.Options has no field %q", name)
		}
		if !reflect.DeepEqual(g.Interface(), w.Interface()) {
			out = append(out, fmt.Sprintf("%s: got %v, serve %v", name, g.Interface(), w.Interface()))
		}
	}
	return out
}

// reporter is the part of testing.TB the guard needs, so a test can hand it a recorder.
type reporter interface {
	Helper()
	Fatalf(format string, args ...any)
}

// requireServeShaped VOIDs a timed harness whose Options are not the ones serve builds for the flags it claims: the reading would describe a load nobody serves.
func requireServeShaped(tb testing.TB, r reporter, got decoder.Options, flags ...string) {
	tb.Helper()
	r.Helper()
	if d := serveShapeDiff(tb, got, flags...); len(d) > 0 {
		r.Fatalf("VOID: this harness's decoder.Options differ from serve's for %v on the memory-moving fields: %v", flags, d)
	}
}

// loadLikeServe loads path with serveOptions(flags...): the way a timed real-model harness loads, guard included.
func loadLikeServe(tb testing.TB, path string, flags ...string) (*decoder.Model, error) {
	tb.Helper()
	o := serveOptions(tb, flags...)
	requireServeShaped(tb, tb, o, flags...)
	return decoder.Load(path, o)
}
