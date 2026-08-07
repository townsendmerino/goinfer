package decoder

import (
	"sync"
	"testing"
)

// TestRegisterAdapter_retiresNotCloses_C29 gates C-29: re-registering an adapter name must RETIRE
// the displaced runtime (keep it alive for any Session still holding it via cache.lora), not
// munmap it — and the map access must be synchronized. The first half asserts the retire
// bookkeeping; the second is a -race probe of concurrent register/read.
func TestRegisterAdapter_retiresNotCloses_C29(t *testing.T) {
	m := &Model{}
	rt1 := &loraRuntime{name: "a"}
	rt2 := &loraRuntime{name: "a"}

	m.registerAdapter("a", rt1)
	if m.adapter("a") != rt1 {
		t.Fatal("adapter a not registered")
	}
	m.registerAdapter("a", rt2) // re-register same name
	if got := m.adapter("a"); got != rt2 {
		t.Errorf("re-register did not replace: got %p want %p", got, rt2)
	}
	if len(m.adapters.retired) != 1 || m.adapters.retired[0] != rt1 {
		t.Errorf("displaced rt1 not retired (retired=%v) — a live session holding it would read freed memory (C-29)", m.adapters.retired)
	}

	// Concurrent register (re-register) + reads must be race-free (run under -race).
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(2)
		go func() { defer wg.Done(); m.registerAdapter("a", &loraRuntime{name: "a"}) }()
		go func() { defer wg.Done(); _ = m.adapter("a"); _ = m.HasAdapter("a") }()
	}
	wg.Wait()
}
