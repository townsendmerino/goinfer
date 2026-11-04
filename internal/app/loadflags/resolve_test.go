package loadflags

import (
	"encoding/json"
	"flag"
	"os"
	"reflect"
	"runtime"
	"slices"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

var updateResolveGolden = flag.Bool("update-resolve-golden", false, "rewrite testdata/resolve.golden.json")

type resolveCase struct {
	name string
	args []string
	auto decoder.AutoChoice // what -backend auto picks on this case's machine
}

func resolveCases() []resolveCase {
	cpu := decoder.AutoChoice{Backend: "cpu", Reason: "auto: this binary has no GPU backend"}
	cs := []resolveCase{
		{"defaults", nil, cpu},
		{"auto cuda", nil, decoder.AutoChoice{Backend: "cuda", Reason: "cuda found"}},
		{"auto metal", nil, decoder.AutoChoice{Backend: "metal", Reason: "metal found"}},
		{"auto cpu skipped", nil, decoder.AutoChoice{Backend: "cpu", Reason: "auto: cuda is built in, but no CUDA device answered", Skipped: true}},
		{"auto metal embed-int4", []string{"-embed-int4"}, decoder.AutoChoice{Backend: "metal", Reason: "metal found"}},
		{"metal", []string{"-backend", "metal"}, cpu},
		{"metal embed-int4", []string{"-backend", "metal", "-embed-int4"}, cpu},
	}
	for _, a := range [][]string{{"-backend", "cpu"}, {"-quant", "int8"}, {"-lora", "/x"}, {"-kv", "f16"}, {"-kv", "i8"},
		{"-ctx", "8192"}, {"-stream-weights"}, {"-weight-cache", "2.5"}, {"-moe-cache-experts"}, {"-moe-cache-slots", "8"},
		{"-moe-pager", "pool"}, {"-moe-pager", "mmap"}, {"-accept-slow"}, {"-embed-int4=false"}, {"-direct-load"}, {"-fit=off"}, {"-exact-prefill"},
		{"-cpu-exact-prefill"}, {"-windowed-kv=false"}, {"-moe-step=false"}, {"-recurrent-step"}, {"-gemma4-step"}} {
		cs = append(cs, resolveCase{strings.Join(a, " "), a, cpu})
	}
	return cs
}

// optionsJSON is o without LoadAbort (a channel, which the cases never set). When the case leaves --moe-pager unset its
// per-OS default is written as such, so the golden holds on every OS; a case that sets it compares the value.
func optionsJSON(t *testing.T, c resolveCase, o decoder.Options) string {
	t.Helper()
	m := map[string]any{}
	v := reflect.ValueOf(o)
	for i := range v.NumField() {
		if f := v.Type().Field(i); f.IsExported() && f.Name != "LoadAbort" {
			m[f.Name] = v.Field(i).Interface()
		}
	}
	if !slices.Contains(c.args, "-moe-pager") && o.MoEPager == decoder.MoEPagerDefault(runtime.GOOS) {
		m["MoEPager"] = "(this OS's default)"
	}
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

// parsed is the Flags Register gives app for args.
func parsed(t *testing.T, app App, args []string) *Flags {
	t.Helper()
	fs := flag.NewFlagSet("resolve", flag.ContinueOnError)
	f := Register(fs, app)
	if err := fs.Parse(args); err != nil {
		t.Fatal(err)
	}
	return f
}

// TestOptions_golden is G-SB10-9 (docs/tasks/task-subsystems-2026-10.md § "Step 3a's gates"): Options over the case
// table, auto resolved to each case's machine, is the golden recorded before Resolve existed.
func TestOptions_golden(t *testing.T) {
	t.Setenv("GOINFER_GGUF_DIRECT", "")
	saved := autoBackend
	t.Cleanup(func() { autoBackend = saved })
	got := map[string]string{}
	for _, app := range []struct {
		name string
		app  App
	}{{"chat", Chat}, {"serve", Serve}} {
		for _, c := range resolveCases() {
			autoBackend = func() decoder.AutoChoice { return c.auto }
			got[app.name+": "+c.name] = optionsJSON(t, c, parsed(t, app.app, c.args).Options())
		}
	}
	const path = "testdata/resolve.golden.json"
	if *updateResolveGolden {
		b, err := json.MarshalIndent(got, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, append(b, '\n'), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var want map[string]string
	if err := json.Unmarshal(raw, &want); err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Errorf("%d cases, golden has %d", len(got), len(want))
	}
	for k, w := range want {
		if got[k] != w {
			t.Errorf("%s:\n got %s\nwant %s", k, got[k], w)
		}
	}
}

// TestResolve_pure is G-SB10-10: Resolve over the case table, with the device probe made to fail the test, gives the
// golden from each case's Machine alone; and register takes the OS and GOINFER_GGUF_DIRECT from its parameters, not
// from the host.
func TestResolve_pure(t *testing.T) {
	t.Setenv("GOINFER_GGUF_DIRECT", "")
	saved := autoBackend
	t.Cleanup(func() { autoBackend = saved })
	autoBackend = func() decoder.AutoChoice { t.Fatal("Resolve probed the devices"); return decoder.AutoChoice{} }
	raw, err := os.ReadFile("testdata/resolve.golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var want map[string]string
	if err := json.Unmarshal(raw, &want); err != nil {
		t.Fatal(err)
	}
	for _, app := range []struct {
		name string
		app  App
	}{{"chat", Chat}, {"serve", Serve}} {
		for _, c := range resolveCases() {
			k := app.name + ": " + c.name
			if got := optionsJSON(t, c, Resolve(*parsed(t, app.app, c.args), Machine{Auto: c.auto}, Overrides{})); got != want[k] {
				t.Errorf("%s:\n got %s\nwant %s", k, got, want[k])
			}
		}
	}

	// Both OSes, each with the variable the other way round from the process's, so neither passes by matching the host.
	t.Setenv("GOINFER_GGUF_DIRECT", "1")
	for _, c := range []struct {
		goos       string
		ggufDirect bool
		pager      string
	}{{"darwin", true, "pool"}, {"linux", false, "mmap"}} {
		f := register(flag.NewFlagSet(c.goos, flag.ContinueOnError), Chat, c.goos, c.ggufDirect)
		if f.MoEPager != c.pager || f.DirectLoad != c.ggufDirect {
			t.Errorf("register on %s with GOINFER_GGUF_DIRECT set=%v: moe-pager %q, direct-load %v; want %s, %v",
				c.goos, c.ggufDirect, f.MoEPager, f.DirectLoad, c.pager, c.ggufDirect)
		}
	}
}

// TestResolve_stepFlagsFollowServeSessions: in serve, a step flag left unset follows the session count and is marked as a default, a given
// one is kept, and chat holds one session, so each is off until given. A recurrent family keeps one slot unless --kv-sessions asks for
// more, so --recurrent-step follows a given count; a Gemma 4 dense model keeps Metal's default slots, so --gemma4-step follows the default
// too. Every row parses real flags, so a given flag reaches Resolve the way it does from any caller.
func TestResolve_stepFlagsFollowServeSessions(t *testing.T) {
	for _, c := range []struct {
		name                           string
		app                            App
		args                           []string
		serve                          *ServeLoad
		wantRecurrent, wantGemma       bool
		recurrentDefault, gemmaDefault bool
	}{
		{"chat, unset", Chat, nil, nil, false, false, false, false},
		{"chat, given", Chat, []string{"-recurrent-step", "-gemma4-step"}, nil, true, true, false, false},
		{"serve, the default count", Serve, nil, &ServeLoad{KVSessions: 4}, false, true, true, true},
		{"serve, 4 sessions", Serve, nil, &ServeLoad{KVSessions: 4, KVSessionsSet: true}, true, true, true, true},
		{"serve, 2 sessions", Serve, nil, &ServeLoad{KVSessions: 2, KVSessionsSet: true}, true, true, true, true},
		{"serve, 1 session", Serve, nil, &ServeLoad{KVSessions: 1, KVSessionsSet: true}, false, false, true, true},
		{"serve, given off", Serve, []string{"-recurrent-step=false", "-gemma4-step=false"}, &ServeLoad{KVSessions: 4, KVSessionsSet: true}, false, false, false, false},
		{"serve, given on at 1 session", Serve, []string{"-recurrent-step", "-gemma4-step"}, &ServeLoad{KVSessions: 1, KVSessionsSet: true}, true, true, false, false},
		{"serve, one given on at the default count", Serve, []string{"-recurrent-step"}, &ServeLoad{KVSessions: 4}, true, true, false, true},
		{"serve, one given off", Serve, []string{"-gemma4-step=false"}, &ServeLoad{KVSessions: 2, KVSessionsSet: true}, true, false, true, false},
	} {
		t.Run(c.name, func(t *testing.T) {
			o := parsed(t, c.app, c.args).OptionsWith(Overrides{Serve: c.serve})
			if o.ResidentRecurrentStep != c.wantRecurrent || o.ResidentGemma4Step != c.wantGemma {
				t.Errorf("recurrent %v, gemma4 %v; want %v, %v", o.ResidentRecurrentStep, o.ResidentGemma4Step, c.wantRecurrent, c.wantGemma)
			}
			if o.ResidentRecurrentStepDefault != c.recurrentDefault || o.ResidentGemma4StepDefault != c.gemmaDefault {
				t.Errorf("marked as defaults: recurrent %v, gemma4 %v; want %v, %v", o.ResidentRecurrentStepDefault, o.ResidentGemma4StepDefault, c.recurrentDefault, c.gemmaDefault)
			}
		})
	}
}
