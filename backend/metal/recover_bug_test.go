//go:build darwin

package metal

import (
	"runtime"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// A nil dereference inside the build is a bug, not a device that declined: it panics out of BuildResident and buildResident instead
// of becoming a decline that sends the model to the CPU path. The build's string panics (its documented declines) are not
// runtime.Errors and still decline (mem.TestRepanicBug_raisesOnlyRuntimeErrors).
func TestBuildResident_aBugPanicsInsteadOfDeclining(t *testing.T) {
	for _, c := range []struct {
		name  string
		build func()
	}{
		{"metalBackend.BuildResident", func() { _, _, _ = (&metalBackend{}).BuildResident(nil) }},
		{"buildResident", func() { _, _ = buildResident((*decoder.Model)(nil)) }},
	} {
		t.Run(c.name, func(t *testing.T) {
			defer func() {
				p := recover()
				if p == nil {
					t.Fatalf("%s on a nil model returned: a programming bug was absorbed into a decline", c.name)
				}
				if _, ok := p.(runtime.Error); !ok {
					t.Fatalf("%s panicked with %T %v, want the runtime.Error itself", c.name, p, p)
				}
			}()
			c.build()
		})
	}
}
