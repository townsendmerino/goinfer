package loadflags

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"runtime"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

var updateRegisterGolden = flag.Bool("update-register-golden", false, "rewrite testdata/register.golden.json")

type registeredFlag struct {
	Name, Usage, DefValue, Type string
}

// TestRegister_golden is G-SB10-5 (docs/tasks/task-subsystems-2026-10.md § "Step 2's gates"): every flag Register
// adds, for each app, keeps its usage text, default and value type byte for byte. --moe-pager's default is the
// one that varies by OS, so the golden holds the linux value and the test checks it against MoEPagerDefault.
func TestRegister_golden(t *testing.T) {
	t.Setenv("GOINFER_GGUF_DIRECT", "")
	got := map[string][]registeredFlag{}
	for name, app := range map[string]App{"chat": Chat, "serve": Serve} {
		fs := flag.NewFlagSet(name, flag.ContinueOnError)
		Register(fs, app)
		fs.VisitAll(func(f *flag.Flag) {
			r := registeredFlag{f.Name, f.Usage, f.DefValue, fmt.Sprintf("%T", f.Value)}
			if f.Name == "moe-pager" {
				if want := decoder.MoEPagerDefault(runtime.GOOS); r.DefValue != want {
					t.Errorf("--moe-pager default %q on %s, want %q", r.DefValue, runtime.GOOS, want)
				}
				r.DefValue = decoder.MoEPagerDefault("linux")
			}
			got[name] = append(got[name], r)
		})
	}
	const path = "testdata/register.golden.json"
	if *updateRegisterGolden {
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
	var want map[string][]registeredFlag
	if err := json.Unmarshal(raw, &want); err != nil {
		t.Fatal(err)
	}
	for _, app := range []string{"chat", "serve"} {
		if len(got[app]) != len(want[app]) {
			t.Errorf("%s: %d flags registered, golden has %d", app, len(got[app]), len(want[app]))
			continue
		}
		for i := range want[app] {
			if got[app][i] != want[app][i] {
				t.Errorf("%s --%s changed:\n got %+v\nwant %+v", app, want[app][i].Name, got[app][i], want[app][i])
			}
		}
	}
}
