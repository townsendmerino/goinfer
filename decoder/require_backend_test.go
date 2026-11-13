package decoder

import (
	"errors"
	"testing"
)

// TestLoad_requireBackendRefusesADecline: with RequireBackend, a load whose named backend builds no resident runner (here
// "cuda", which this test binary does not link, so the load falls back to the CPU) is refused with ErrBackendDeclined and
// the decline's reason; without it the same load runs on the CPU; and a CPU load is never refused by it.
func TestLoad_requireBackendRefusesADecline(t *testing.T) {
	backendMu.RLock()
	_, linked := backendRegistry["cuda"]
	backendMu.RUnlock()
	if linked {
		t.Skip("a cuda backend is registered in this test binary")
	}
	path := tinyFixture(t)
	m, err := Load(path, Options{Backend: "cuda", RequireBackend: true})
	if err == nil {
		m.Close()
		t.Fatal("Load with RequireBackend succeeded on a backend that built no resident runner")
	}
	if !errors.Is(err, ErrBackendDeclined) {
		t.Errorf("Load's error %q is not ErrBackendDeclined", err)
	}
	if m, err = Load(path, Options{Backend: "cuda"}); err != nil {
		t.Fatalf("without RequireBackend the load should fall back to the CPU: %v", err)
	} else if m.ResidentActive() {
		m.Close()
		t.Fatal("VOID: the fallback load built a resident runner")
	} else {
		m.Close()
	}
	if m, err = Load(path, Options{Backend: "cpu", RequireBackend: true}); err != nil {
		t.Errorf("a CPU load with RequireBackend was refused: %v", err)
	} else {
		m.Close()
	}
}
