package decoder

import (
	"strings"
	"testing"
)

// LFM2 with int8 KV (Options.KVQuant == "i8") on the CPU: its own forward loop keeps an f32 cache, so the precision cannot
// be delivered and Load refuses it, rather than enabling an int8 store the family's attention would size its scores
// buffer past.
func TestLFM2_kvQuantI8_isRefusedOnTheCPU(t *testing.T) {
	m, err := Load("../testdata/lfm2-tiny", Options{Backend: "cpu", KVQuant: "i8"})
	if err == nil {
		m.Close()
		t.Fatal("Load accepted --kv i8 for LFM2 on the CPU, which keeps an f32 cache")
	}
	if !strings.Contains(err.Error(), "--kv i8 cannot be delivered") {
		t.Fatalf("refusal %q does not say the precision cannot be delivered", err)
	}
}
