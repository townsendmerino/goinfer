// Package loadflags registers the model-loading flags goinfer-chat and goinfer-serve share, and turns
// their values into a decoder.Options. The two binaries have loaded through one path
// (internal/modelload) since 2026-09-24, but each still registered its own flags, and the sets
// drifted: chat had no --ctx, --stream-weights, --moe-cache-experts or --moe-cache-slots, and a
// cold-user run reached for --moe-cache-experts in chat and found nothing. A loading flag registered
// here exists in both binaries by construction. What stays app-specific is what does not describe how
// to load a model: serve's routes, queues, sessions and per-model zoo; chat's sampling and REPL.
package loadflags

import (
	"flag"
	"fmt"
	"os"
	"runtime"
	"strconv"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/cliutil"
	"github.com/townsendmerino/goinfer/internal/config"
)

// App is the binary registering the flags. A few help texts differ: serve's --model takes per-model
// overrides of some of these flags, and serve can refuse to start where chat just falls back.
type App int

const (
	Chat App = iota
	Serve
)

// Flags holds the shared loading flags' values once the flag set has been parsed.
type Flags struct {
	Backend         string              // --backend; "auto" (the default) is replaced by the backend it resolves to in Validate
	Auto            *decoder.AutoChoice // what -backend auto chose and why; nil when the user named a backend
	Quant           string              // --quant
	QuantSet        bool                // --quant was given on the command line, not left at its default
	LoRA            string              // --lora
	KV              string              // --kv
	Ctx             int                 // --ctx
	StreamWeights   bool                // --stream-weights
	WeightCacheGB   float64             // --weight-cache
	MoECacheExperts bool                // --moe-cache-experts
	MoECacheSlots   int                 // --moe-cache-slots
	MoEPager        string              // --moe-pager
	AcceptSlow      bool                // --accept-slow
	EmbedInt4       bool                // --embed-int4
	EmbedInt4Set    bool                // --embed-int4 was given on the command line, not left at its default
	DirectLoad      bool                // --direct-load
	Fit             bool                // --fit
	ExactPrefill    bool                // --exact-prefill
	CPUExactPrefill bool                // --cpu-exact-prefill
	WindowedKV      bool                // --windowed-kv
}

// Register adds the shared loading flags to fs and returns the Flags their values are parsed into. Each flag's
// name, default and usage text come from internal/config's schema, on this OS and with this process's environment.
func Register(fs *flag.FlagSet, app App) *Flags {
	return register(fs, app, runtime.GOOS, os.Getenv("GOINFER_GGUF_DIRECT") != "")
}

// register is Register on goos, with ggufDirect for GOINFER_GGUF_DIRECT being set.
func register(fs *flag.FlagSet, app App, goos string, ggufDirect bool) *Flags {
	def := func(name string) string { return config.Lookup(name).DefaultFor(goos) }
	boolDef := func(name string) bool { return must(strconv.ParseBool(def(name))) }
	intDef := func(name string) int { return must(strconv.Atoi(def(name))) }
	floatDef := func(name string) float64 { return must(strconv.ParseFloat(def(name), 64)) }
	bit := config.Chat
	if app == Serve {
		bit = config.Serve
	}
	use := func(name string) string { return config.Lookup(name).Usage(bit) }
	f := &Flags{Quant: def("quant"), Fit: boolDef("fit"), EmbedInt4: boolDef("embed-int4")}
	fs.StringVar(&f.Backend, "backend", def("backend"), use("backend"))
	fs.Var(quantFlag{f}, "quant", use("quant"))
	fs.StringVar(&f.LoRA, "lora", def("lora"), use("lora"))
	fs.StringVar(&f.KV, "kv", def("kv"), use("kv"))
	fs.IntVar(&f.Ctx, "ctx", intDef("ctx"), use("ctx"))
	fs.BoolVar(&f.StreamWeights, "stream-weights", boolDef("stream-weights"), use("stream-weights"))
	fs.Float64Var(&f.WeightCacheGB, "weight-cache", floatDef("weight-cache"), use("weight-cache"))
	fs.BoolVar(&f.MoECacheExperts, "moe-cache-experts", boolDef("moe-cache-experts"), use("moe-cache-experts"))
	fs.IntVar(&f.MoECacheSlots, "moe-cache-slots", intDef("moe-cache-slots"), use("moe-cache-slots"))
	fs.StringVar(&f.MoEPager, "moe-pager", def("moe-pager"), use("moe-pager"))
	fs.BoolVar(&f.AcceptSlow, "accept-slow", boolDef("accept-slow"), use("accept-slow"))
	fs.Var(embedInt4Flag{f}, "embed-int4", use("embed-int4"))
	fs.BoolVar(&f.DirectLoad, "direct-load", boolDef("direct-load") || ggufDirect, use("direct-load"))
	fs.Var((*cliutil.OnOff)(&f.Fit), "fit", use("fit")) // lenient: --fit=off works (M-14)
	fs.BoolVar(&f.ExactPrefill, "exact-prefill", boolDef("exact-prefill"), use("exact-prefill"))
	fs.BoolVar(&f.CPUExactPrefill, "cpu-exact-prefill", boolDef("cpu-exact-prefill"), use("cpu-exact-prefill"))
	fs.BoolVar(&f.WindowedKV, "windowed-kv", boolDef("windowed-kv"), use("windowed-kv"))
	return f
}

// must is a schema default parsed; one that does not parse panics, and TestRegister_golden registers every flag.
func must[T any](v T, err error) T {
	if err != nil {
		panic("loadflags: schema default: " + err.Error())
	}
	return v
}

// Options is the decoder.Options the flags describe on this machine, with no per-model overrides. It resolves a
// Backend of "auto" first, probing the devices once.
func (f *Flags) Options() decoder.Options { return f.OptionsWith(Overrides{}) }

// OptionsWith is Options with ov layered on: serve's per-model keys and its own load settings.
func (f *Flags) OptionsWith(ov Overrides) decoder.Options {
	f.resolveAuto() // Validate has normally done it
	return Resolve(*f, Machine{}, ov)
}

// Machine is what Resolve knows of the host.
type Machine struct {
	Auto decoder.AutoChoice // what -backend auto picks here; read only when the flags' Backend is "auto"
}

// Overrides are what serve layers on the shared flags for one model: the keys of its --model name=path,key=…
// (nil = not given) and, in Serve, serve's own load settings (nil for chat). A decision head's quant is serve's
// rule; serve passes the quant it picks as Quant.
type Overrides struct {
	Quant, LoRA, KV, KVQuant *string
	Stream, EmbedInt4        *bool
	WeightCacheGB            *float64
	Ctx                      *int
	Serve                    *ServeLoad
}

// ServeLoad is serve's load settings, the same for every model it serves.
type ServeLoad struct {
	KVQuant       string // --kv-quant; "" follows --kv
	KVSessions    int    // --kv-sessions, also the KV slots a GPU-resident backend is asked for
	KVSessionsSet bool   // --kv-sessions was given; without it Metal keeps its own slot default
	PrefillChunk  int    // --prefill-chunk; 0 = off
	KVShareOff    bool   // no request takes KV rows another conversation computed
	KVHostBytes   int64  // the prefix-snapshot budget; 0 = off
	CPUBatch      int    // --cpu-batch: decoder.CPUBatchAuto, CPUBatchOn or CPUBatchOff
}

// Resolve is the decoder.Options f describes on m, with ov layered on. It reads nothing but its arguments: the
// defaults, the environment and the OS reach f when its flags are registered, and m stands in for the device probe.
func Resolve(f Flags, m Machine, ov Overrides) decoder.Options {
	if f.Backend == "auto" {
		f.Backend, f.Auto = m.Auto.Backend, &m.Auto
	}
	kv := or(ov.KV, f.KV)
	cpuKV := ""
	if ov.Serve != nil {
		cpuKV = ov.Serve.KVQuant
	}
	o := decoder.Options{
		Backend:            f.Backend,
		BackendAuto:        f.Auto != nil,
		Quant:              or(ov.Quant, f.Quant),
		LoRA:               or(ov.LoRA, f.LoRA),
		KVPrecision:        kv,
		KVQuant:            CPUKV(or(ov.KVQuant, cpuKV), kv),
		ResidentContext:    or(ov.Ctx, f.Ctx),
		StreamWeights:      or(ov.Stream, f.StreamWeights),
		WeightCacheBytes:   int64(or(ov.WeightCacheGB, f.WeightCacheGB) * 1e9),
		MoECacheExperts:    f.MoECacheExperts,
		MoECacheSlots:      f.MoECacheSlots,
		MoEPager:           f.MoEPager,
		AcceptSlowMoE:      f.AcceptSlow,
		EmbedInt4:          or(ov.EmbedInt4, f.embedInt4()),
		DisableFit:         !f.Fit,
		ExactPrefill:       f.ExactPrefill,
		ResidentWindowedKV: f.WindowedKV,
		Knobs:              f.Knobs(),
	}
	if sv := ov.Serve; sv != nil {
		o.ResidentKVSlots = sv.KVSessions
		o.ResidentKVSlotsDefault = !sv.KVSessionsSet
		o.ResidentPrefillChunk = sv.PrefillChunk
		o.KVShareOff = sv.KVShareOff
		o.KVHostBytes = sv.KVHostBytes
		o.CPUBatchDecode = sv.CPUBatch
	}
	return o
}

// or is *p, or def when p is nil.
func or[T any](p *T, def T) T {
	if p != nil {
		return *p
	}
	return def
}

// Validate checks what decoder.Options.Validate does not: the flag values that only make sense as a
// choice from a list, or as a size. Call it after parsing, before Options().Validate().
func (f *Flags) Validate() error {
	// Resolve "auto" first: every check below and every caller after this reads Backend by name (the Metal
	// --embed-int4 default, serve's tower and -require-backend logic), and "auto" matches none of them. Each app
	// calls Validate right after parsing its flags, before anything reads Backend; Options resolves too, for a
	// caller that skipped Validate.
	f.resolveAuto()
	if f.MoEPager != "mmap" && f.MoEPager != "pool" {
		return fmt.Errorf("-moe-pager must be \"mmap\" or \"pool\" (got %q)", f.MoEPager)
	}
	if f.Ctx < 0 {
		return fmt.Errorf("-ctx %d: must be >= 0 (0 = the backend default)", f.Ctx)
	}
	if f.MoECacheSlots < 0 {
		return fmt.Errorf("-moe-cache-slots %d: must be >= 0 (0 = the built-in default)", f.MoECacheSlots)
	}
	if f.WeightCacheGB < 0 {
		return fmt.Errorf("-weight-cache %g: must be >= 0 (0 = auto)", f.WeightCacheGB)
	}
	return nil
}

// embedInt4 is --embed-int4 as the load should see it. The default is on, except on Metal: the Metal resident runner
// does not accept an int4 embedding/LM-head table yet ("weight kind \"int4\" is not int8"), so with the default on, a
// plain `--backend metal` load declined the GPU and decoded on the CPU (found 2026-09-28, docs/quantization.md "Known
// issue"; reproduced 2026-09-30 on the 0.5B, while CUDA and WebGPU stayed resident with the int4 table). An explicit
// --embed-int4 is still honoured on Metal, and then runs on the CPU as asked.
func (f *Flags) embedInt4() bool {
	if f.Backend == "metal" && !f.EmbedInt4Set {
		return false
	}
	return f.EmbedInt4
}

// autoBackend is decoder.AutoBackend, a variable so a test can stand in a GPU this test binary does not link.
var autoBackend = decoder.AutoBackend

// resolveAuto replaces a Backend of "auto" with the backend auto picks and keeps the choice in Auto.
// Once Backend names a backend it does nothing, so calling it twice probes the devices once.
func (f *Flags) resolveAuto() {
	if f.Backend != "auto" {
		return
	}
	c := autoBackend()
	f.Backend, f.Auto = c.Backend, &c
}

// BackendLine is the one line an app prints when -backend auto chose its backend: what it chose, why, and how to
// choose otherwise. It is "" when the user named a backend.
func (f *Flags) BackendLine() string {
	switch {
	case f.Auto == nil:
		return ""
	case f.Backend == "cpu":
		return fmt.Sprintf("backend: cpu (%s)", f.Auto.Reason)
	}
	return fmt.Sprintf("backend: %s (%s; -backend cpu to use the CPU)", f.Backend, f.Auto.Reason)
}

// ExplicitQuant is --quant when it was given, else "". A .giw carries its own quant, and only an
// explicit --quant that disagrees is refused (decoder.Model.CheckGiwQuantMatch): the default must
// never conflict with an already-baked bundle.
func (f *Flags) ExplicitQuant() string {
	if f.QuantSet {
		return f.Quant
	}
	return ""
}

// Knobs carries the prompt-ingestion flags to the model's decoder.Options (phase 5,
// docs/tasks/task-env-config-2026-09.md). --exact-prefill itself travels as Options.ExactPrefill,
// which every backend consults per model, so only the CPU-specific flag needs a knob.
//
// The CPU knob is set EXPLICITLY either way rather than left unset. The decoder treats unset as on,
// but an inherited GOINFER_CPU_FAST_ATTENTION in the caller's environment would otherwise outrank the
// flags — the binary's own flags must win over whatever the shell happened to export, and
// Options.Knobs does win. --exact-prefill and --cpu-exact-prefill both disable it.
func (f *Flags) Knobs() *decoder.Knobs {
	fast := "1"
	if f.ExactPrefill || f.CPUExactPrefill {
		fast = "0"
	}
	return &decoder.Knobs{"GOINFER_CPU_FAST_ATTENTION": fast}
}

// CPUKV is the CPU KV cache precision: serve's deprecated --kv-quant when given, else --kv mapped onto the CPU cache: f16 and
// i8 select those caches, and auto (the default) and f32 the f32 reference. auto moves to f16 only after the gates read
// (docs/tasks/task-cpu-kv-f16-2026-10.md, decision 1). (Both Options fields remain; only the command line unified.)
func CPUKV(kvQuant, kv string) string {
	if kvQuant != "" {
		return kvQuant
	}
	switch kv {
	case "i8", "f16":
		return kv
	}
	return "f32"
}

// quantFlag is --quant: a string flag that also records that it was given, so ExplicitQuant can tell
// an explicit --quant from its default.
type quantFlag struct{ f *Flags }

func (q quantFlag) String() string {
	if q.f == nil {
		return ""
	}
	return q.f.Quant
}

func (q quantFlag) Set(v string) error {
	q.f.Quant, q.f.QuantSet = v, true
	return nil
}

// embedInt4Flag is --embed-int4: a bool flag that also records that it was given, so the Metal default (embedInt4) can
// tell an explicit --embed-int4 from its default.
type embedInt4Flag struct{ f *Flags }

func (e embedInt4Flag) String() string {
	if e.f == nil {
		return "true"
	}
	return strconv.FormatBool(e.f.EmbedInt4)
}

func (e embedInt4Flag) Set(v string) error {
	b, err := strconv.ParseBool(v)
	if err != nil {
		return err
	}
	e.f.EmbedInt4, e.f.EmbedInt4Set = b, true
	return nil
}

func (e embedInt4Flag) IsBoolFlag() bool { return true }
