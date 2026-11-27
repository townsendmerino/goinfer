package stores

import (
	"fmt"
	"strings"
	"testing"
)

func strSize(model string, msgs []string) int {
	n := len(model)
	for _, m := range msgs {
		n += len(m)
	}
	return n
}

func checkResponseStoreBytes(t *testing.T, s *ResponseStore[string]) {
	t.Helper()
	sum := 0
	for _, e := range s.m {
		sum += e.size
	}
	if s.bytes != sum || len(s.order) != len(s.m) {
		t.Fatalf("store holds %d bytes by its total but %d by its entries (%d ids in order, %d entries)", s.bytes, sum, len(s.order), len(s.m))
	}
}

func TestResponseStore_byteBudgetEvictsOldestFirst(t *testing.T) {
	one := strSize("", []string{strings.Repeat("x", 1000)})
	s := NewResponseStoreWithMaxBytes(256, 3*one, strSize)
	for i := range 4 {
		s.Put(fmt.Sprintf("r%d", i), "", []string{strings.Repeat("x", 1000)})
	}
	if s.Get("r0") != nil {
		t.Errorf("the oldest entry survived a put past the %d-byte budget (store total %d)", s.maxBytes, s.bytes)
	}
	for _, id := range []string{"r1", "r2", "r3"} {
		if s.Get(id) == nil {
			t.Errorf("%s was evicted, want only the oldest gone", id)
		}
	}
	checkResponseStoreBytes(t, s)
	if s.bytes > s.maxBytes {
		t.Errorf("store holds %d bytes, over its %d-byte budget", s.bytes, s.maxBytes)
	}

	s.Put("r3", "", []string{strings.Repeat("x", 10)})
	checkResponseStoreBytes(t, s)
	s.Put("r4", "", []string{strings.Repeat("x", 900)})
	if s.Get("r1") == nil || s.Get("r2") == nil {
		t.Errorf("a put that fits after an overwrite shrank r3 evicted an entry")
	}
	checkResponseStoreBytes(t, s)
}

func TestResponseStore_oversizedEntryIsRefused(t *testing.T) {
	s := NewResponseStoreWithMaxBytes(256, 4096, strSize)
	s.Put("a", "", []string{strings.Repeat("x", 100)})
	s.Put("b", "", []string{strings.Repeat("x", 100)})
	s.Put("big", "", []string{strings.Repeat("x", 5000)})
	if s.Get("big") != nil {
		t.Errorf("an entry over the %d-byte budget was stored", s.maxBytes)
	}
	if s.Get("a") == nil || s.Get("b") == nil {
		t.Errorf("refusing an oversized entry evicted others")
	}
	checkResponseStoreBytes(t, s)

	s.Put("a", "", []string{strings.Repeat("x", 5000)})
	if s.Get("a") != nil {
		t.Errorf("an oversized overwrite left the id's older conversation in place")
	}
	if s.Get("b") == nil {
		t.Errorf("an oversized overwrite evicted another entry")
	}
	checkResponseStoreBytes(t, s)
}

func TestNewResponseStore_budgetIsTheProductionConstant(t *testing.T) {
	if got := NewResponseStore(256, strSize).MaxBytes(); got != MaxResponseStoreBytes || MaxResponseStoreBytes != 64<<20 {
		t.Errorf("store budget %d (constant %d), want 64 MiB", got, MaxResponseStoreBytes)
	}
}
