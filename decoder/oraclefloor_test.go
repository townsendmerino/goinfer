//go:build realckpt

package decoder

import "testing"

// The bars are per-precision because int4 is a coarser grid than int8, and for a while int4 was
// judged by the int8 number simply because that was the only number there was.
func TestOracleCosFloorPerPrecision(t *testing.T) {
	for _, c := range []struct {
		quant string
		want  float64
	}{
		{"int8int8", 0.99},
		{"int8", 0.99},
		{"int4", 0.98},
	} {
		if got := oracleCosFloor(t, c.quant); got != c.want {
			t.Errorf("oracleCosFloor(%q) = %v, want %v", c.quant, got, c.want)
		}
	}
	if oracleCosFloor(t, "int4") >= oracleCosFloor(t, "int8") {
		t.Error("the int4 bar must be looser than the int8 one; if they are equal the split is pointless")
	}
}

// An unregistered precision must FAIL rather than default. A default is how int4 silently
// inherited the int8 bar, and it would let the next precision do the same without a decision.
func TestOracleCosFloorRefusesUnknownPrecision(t *testing.T) {
	fake := &testing.T{}
	done := make(chan bool)
	go func() {
		defer func() { done <- true }() // t.Fatalf runs Goexit; the defer still fires
		oracleCosFloor(fake, "fp6-someday")
	}()
	<-done
	if !fake.Failed() {
		t.Error("an unregistered quant must fail the test, not silently return a bar")
	}
}
