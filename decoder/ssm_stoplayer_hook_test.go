//go:build goinfer_testhooks

package decoder

import (
	"reflect"
	"testing"
)

// SetSSMStopLayerForTest must actually truncate the granite forward — the gpu layer sweeps depend on
// it for their CPU reference, which os.Setenv could not reach (the env var is read once at init).
func TestSetSSMStopLayerForTest_truncatesTheForward(t *testing.T) {
	m, err := Load("../testdata/granite-tiny", Options{Backend: "cpu"})
	if err != nil {
		t.Skipf("no granite-tiny fixture: %v", err)
	}
	defer m.Close()
	fwd := func(stop int) []float32 {
		restore := SetSSMStopLayerForTest(stop)
		defer restore()
		lg, err := m.ForwardForTest(1, m.NewCache(4))
		if err != nil {
			t.Fatal(err)
		}
		return append([]float32(nil), lg...)
	}
	full := fwd(-1)
	first := fwd(0)
	if reflect.DeepEqual(full, first) {
		t.Fatal("stopping after layer 0 gave the full model's logits: the hook does not reach the forward")
	}
	if again := fwd(-1); !reflect.DeepEqual(full, again) {
		t.Fatal("restore() did not put the full forward back")
	}
}
