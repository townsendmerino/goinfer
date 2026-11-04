package serveapp

import (
	"flag"
	"runtime"
	"testing"

	schema "github.com/townsendmerino/goinfer/internal/config"
)

// TestConfigDefaults is G-SB10-6 (docs/tasks/task-subsystems-2026-10.md § "Step 2's gates"): every flag serve
// registers has a schema setting whose default is the flag's registered default, with GOINFER_GGUF_DIRECT unset.
// moe-pager's is checked on this OS.
func TestConfigDefaults(t *testing.T) {
	t.Setenv("GOINFER_GGUF_DIRECT", "")
	byFlag := map[string]schema.Setting{}
	for _, s := range schema.Settings {
		if s.Flag != "" {
			byFlag[s.Flag] = s
		}
	}
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	registerFlags(fs)
	fs.VisitAll(func(f *flag.Flag) {
		s, ok := byFlag[f.Name]
		if !ok {
			t.Errorf("--%s: no setting in internal/config", f.Name)
			return
		}
		if got := s.DefaultFor(runtime.GOOS); got != f.DefValue {
			t.Errorf("--%s: schema default %q, registered default %q", f.Name, got, f.DefValue)
		}
	})
}
