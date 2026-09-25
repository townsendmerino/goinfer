package loadflags

import (
	"flag"
	"runtime"
	"sort"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

func parse(t *testing.T, app App, args ...string) (*flag.FlagSet, *Flags) {
	t.Helper()
	fs := flag.NewFlagSet("probe", flag.ContinueOnError)
	f := Register(fs, app)
	if err := fs.Parse(args); err != nil {
		t.Fatalf("parse %q: %v", args, err)
	}
	return fs, f
}

// TestRegister_chatAndServeGetTheSameFlags is the drift this package exists to end: chat lacked
// --ctx, --stream-weights and --moe-cache-experts/--moe-cache-slots while serve had them, and a
// cold-user run reached for --moe-cache-experts in chat. Same names, same defaults; the help may
// differ only by serve's per-model-override notes and its serve-only fallback wording.
func TestRegister_chatAndServeGetTheSameFlags(t *testing.T) {
	chat, _ := parse(t, Chat)
	serve, _ := parse(t, Serve)
	names := func(fs *flag.FlagSet) (out []string) {
		fs.VisitAll(func(f *flag.Flag) { out = append(out, f.Name+"="+f.DefValue) })
		sort.Strings(out)
		return out
	}
	if c, s := strings.Join(names(chat), " "), strings.Join(names(serve), " "); c != s {
		t.Fatalf("chat and serve register different loading flags or defaults:\n chat:  %s\n serve: %s", c, s)
	}
	for _, want := range []string{"ctx", "stream-weights", "moe-cache-experts", "moe-cache-slots"} {
		if chat.Lookup(want) == nil {
			t.Errorf("chat has no --%s", want)
		}
	}
	for _, name := range []string{"quant", "lora", "kv", "ctx", "stream-weights", "weight-cache", "embed-int4"} {
		if u := chat.Lookup(name).Usage; strings.Contains(u, "per-model override") {
			t.Errorf("chat's --%s help names serve's per-model override:\n%s", name, u)
		}
		if u := serve.Lookup(name).Usage; !strings.Contains(u, "per-model override") {
			t.Errorf("serve's --%s help does not name its per-model override:\n%s", name, u)
		}
	}
}

// TestOptions_everyFlagReachesOptions: a flag that is registered but never read is exactly how a
// binary accepts an option and ignores it. One row per registered flag, set to a non-default value;
// the table must cover every flag Register adds, so a new flag without a row fails here.
func TestOptions_everyFlagReachesOptions(t *testing.T) {
	rows := map[string]struct {
		arg  string
		took func(o decoder.Options, f *Flags) bool
	}{
		"backend":           {"--backend=cuda", func(o decoder.Options, _ *Flags) bool { return o.Backend == "cuda" }},
		"quant":             {"--quant=int8", func(o decoder.Options, _ *Flags) bool { return o.Quant == "int8" }},
		"lora":              {"--lora=/adapter", func(o decoder.Options, _ *Flags) bool { return o.LoRA == "/adapter" }},
		"kv":                {"--kv=i8", func(o decoder.Options, _ *Flags) bool { return o.KVPrecision == "i8" && o.KVQuant == "i8" }},
		"ctx":               {"--ctx=12345", func(o decoder.Options, _ *Flags) bool { return o.ResidentContext == 12345 }},
		"stream-weights":    {"--stream-weights", func(o decoder.Options, _ *Flags) bool { return o.StreamWeights }},
		"weight-cache":      {"--weight-cache=2.5", func(o decoder.Options, _ *Flags) bool { return o.WeightCacheBytes == 2_500_000_000 }},
		"moe-cache-experts": {"--moe-cache-experts", func(o decoder.Options, _ *Flags) bool { return o.MoECacheExperts }},
		"moe-cache-slots":   {"--moe-cache-slots=7", func(o decoder.Options, _ *Flags) bool { return o.MoECacheSlots == 7 }},
		"moe-pager":         {"--moe-pager=" + otherPager(), func(o decoder.Options, _ *Flags) bool { return o.MoEPager == otherPager() }},
		"accept-slow":       {"--accept-slow", func(o decoder.Options, _ *Flags) bool { return o.AcceptSlowMoE }},
		"embed-int4":        {"--embed-int4", func(o decoder.Options, _ *Flags) bool { return o.EmbedInt4 }},
		"fit":               {"--fit=off", func(o decoder.Options, _ *Flags) bool { return o.DisableFit }},
		"exact-prefill": {"--exact-prefill", func(o decoder.Options, _ *Flags) bool {
			return o.ExactPrefill && (*o.Knobs)["GOINFER_CPU_FAST_ATTENTION"] == "0"
		}},
		"cpu-exact-prefill": {"--cpu-exact-prefill", func(o decoder.Options, _ *Flags) bool { return (*o.Knobs)["GOINFER_CPU_FAST_ATTENTION"] == "0" }},
		// Not a decoder option: modelload.Request.DirectLoad reads it.
		"direct-load": {"--direct-load", func(_ decoder.Options, f *Flags) bool { return f.DirectLoad }},
	}
	fs, _ := parse(t, Chat)
	fs.VisitAll(func(fl *flag.Flag) {
		if _, ok := rows[fl.Name]; !ok {
			t.Errorf("--%s is registered but has no row here: add one, and make sure Options (or its caller) reads it", fl.Name)
		}
	})
	for name, r := range rows {
		_, f := parse(t, Chat, r.arg)
		if !r.took(f.Options(), f) {
			t.Errorf("--%s: %s did not reach the options (got %+v)", name, r.arg, f.Options())
		}
		// And the default does NOT look like the non-default value, or the row proves nothing.
		_, def := parse(t, Chat)
		if r.took(def.Options(), def) {
			t.Errorf("--%s: the default already satisfies the check for %s — the row cannot tell them apart", name, r.arg)
		}
	}
}

func otherPager() string {
	if decoder.MoEPagerDefault(runtime.GOOS) == "pool" {
		return "mmap"
	}
	return "pool"
}

// TestExplicitQuant: a .giw carries its own quant, and only an EXPLICIT --quant that disagrees is
// refused — the default must never conflict, so the two have to be told apart.
func TestExplicitQuant(t *testing.T) {
	if _, f := parse(t, Chat); f.ExplicitQuant() != "" || f.Quant != "int4" {
		t.Errorf("default: ExplicitQuant()=%q Quant=%q, want \"\" and int4", f.ExplicitQuant(), f.Quant)
	}
	if _, f := parse(t, Chat, "--quant=int4"); f.ExplicitQuant() != "int4" {
		t.Errorf("--quant=int4 given explicitly: ExplicitQuant()=%q, want int4", f.ExplicitQuant())
	}
}

func TestValidate(t *testing.T) {
	if _, f := parse(t, Chat); f.Validate() != nil {
		t.Fatalf("defaults do not validate: %v", f.Validate())
	}
	for _, bad := range [][]string{{"--moe-pager=swap"}, {"--ctx=-1"}, {"--moe-cache-slots=-2"}, {"--weight-cache=-1"}} {
		if _, f := parse(t, Chat, bad...); f.Validate() == nil {
			t.Errorf("%q validated", bad)
		}
	}
}

// TestCPUExactPrefillDisclosesTheDefaultsTrade: the CPU's f32 prompt attention is ON by default and is
// a documented divergence, so its opt-out's help has to disclose the trade — "disclosed in --help, not
// something a user has to already know to type". Since --cpu-fast-attention was removed,
// --cpu-exact-prefill's help is the ONLY place --help says any of this, so it has to carry all of it.
func TestCPUExactPrefillDisclosesTheDefaultsTrade(t *testing.T) {
	// Read the default off the REAL registration, not a zero value: a zero-value check passes
	// unchanged if someone flips the flag's default, which is exactly the regression it exists for.
	fs, f := parse(t, Chat)
	if f.CPUExactPrefill {
		t.Error("cpu-exact-prefill must default to false — it is the opt-OUT; the fast path is the default")
	}
	if got := fs.Lookup("cpu-exact-prefill").Usage; got != CPUExactPrefillHelp {
		t.Errorf("--cpu-exact-prefill is registered with other help text:\n%s", got)
	}
	for _, must := range []string{
		"BIT-EXACT",         // what the flag buys
		"DEFAULT",           // that the other path is the default, not an opt-in
		"NOT bit-identical", // the default's divergence is named, not implied
		"0.9976",            // and quantified
		"2.28x",             // and so is the win, so the trade is legible
		"512",               // the floor below which the default runs the exact kernel anyway
		"MoE",               // MoE takes the same path — so this is its only exact route too
		"speculative",       // the guarantee that holds regardless
		"--exact-prefill",   // and the all-backends version
	} {
		if !strings.Contains(CPUExactPrefillHelp, must) {
			t.Errorf("--cpu-exact-prefill help does not mention %q — the default's trade must be disclosed in --help:\n%s", must, CPUExactPrefillHelp)
		}
	}
	// --exact-prefill's help quotes each backend's floor; Metal's is 64 (metal/backend.go metalFastPrefillFloor),
	// which the help used to give as 512 (and the removed --metal-fast-prefill's as 256).
	if !strings.Contains(ExactPrefillHelp, "above 64 prompt tokens") {
		t.Errorf("--exact-prefill help does not give Metal's 64-token floor:\n%s", ExactPrefillHelp)
	}
	for _, gone := range []string{"--cpu-fast-attention", "--metal-fast-prefill"} {
		if strings.Contains(ExactPrefillHelp, gone) || strings.Contains(CPUExactPrefillHelp, gone) {
			t.Errorf("help text still names the removed flag %s", gone)
		}
	}
}

// TestMoEPagerDefault gates S5's registered default: pool on darwin (where MADV_DONTNEED is a no-op,
// so mmap mode cannot enforce its budget), mmap everywhere else — and --moe-pager's registered
// default is that function's answer for this platform.
func TestMoEPagerDefault(t *testing.T) {
	for goos, want := range map[string]string{"darwin": "pool", "linux": "mmap", "windows": "mmap", "freebsd": "mmap"} {
		if got := decoder.MoEPagerDefault(goos); got != want {
			t.Errorf("MoEPagerDefault(%q) = %q, want %q", goos, got, want)
		}
	}
	if fs, _ := parse(t, Serve); fs.Lookup("moe-pager").DefValue != decoder.MoEPagerDefault(runtime.GOOS) {
		t.Errorf("--moe-pager default %q, want MoEPagerDefault(%s) = %q", fs.Lookup("moe-pager").DefValue, runtime.GOOS, decoder.MoEPagerDefault(runtime.GOOS))
	}
}

// TestFitFlag_registeredLenient: --fit is cliutil.OnOff (its spellings are tested there); this checks
// the registration — --fit=off parses, and a bare --fit after it turns it back on.
func TestFitFlag_registeredLenient(t *testing.T) {
	if _, fl := parse(t, Chat, "--fit=off"); fl.Fit {
		t.Error("--fit=off did not turn fit off")
	}
	if _, fl := parse(t, Chat, "--fit=off", "--fit"); !fl.Fit {
		t.Error("a bare --fit after --fit=off did not turn it back on")
	}
}
