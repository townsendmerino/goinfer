package decoder

import (
	"encoding/json"
	"testing"

	"github.com/townsendmerino/goinfer/internal/mem/probe"
)

// TestFitQueryAvailable_isFitQueryAtThatReading (MM6 gate G-MM6-1): FitQueryAvailable with a reading answers as FitQuery
// does when the probe reads the same, for a safetensors directory and a .gguf, at readings that fit, cap and refuse.
func TestFitQueryAvailable_isFitQueryAtThatReading(t *testing.T) {
	fake := &probe.Fake{PhysicalBytes: 64 << 30}
	useProbe(t, fake)
	for _, path := range []string{longContextFixture(t, 1<<17), "testdata/gptoss_tiny.gguf"} {
		for _, avail := range []int64{0, 1, 64 << 20, 1 << 30, 64 << 30} {
			fake.Set(func(f *probe.Fake) { f.AvailableBytes = avail })
			resetAvailProbeCache()
			want, err1 := FitQuery(path, Options{Quant: "int4"})
			got, err2 := FitQueryAvailable(path, Options{Quant: "int4"}, avail)
			w, _ := json.Marshal(want)
			g, _ := json.Marshal(got)
			if (err1 == nil) != (err2 == nil) || string(w) != string(g) {
				t.Errorf("%s at %d: FitQueryAvailable %s (%v), FitQuery %s (%v)", path, avail, g, err2, w, err1)
			}
		}
	}
}
