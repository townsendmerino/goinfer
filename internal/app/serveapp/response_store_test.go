package serveapp

import (
	"fmt"
	"strings"
	"testing"
)

// storeEntry is a one-message entry whose text is n bytes.
func storeEntry(n int) *responseEntry {
	return &responseEntry{messages: []chatMessage{{Content: rawStr(strings.Repeat("x", n))}}}
}

// checkStoreBytes fails unless the store's byte total is the sum of its entries' sizes and order lists exactly its ids.
func checkStoreBytes(t *testing.T, s *responseStore) {
	t.Helper()
	sum := 0
	for _, e := range s.m {
		sum += e.size
	}
	if s.bytes != sum || len(s.order) != len(s.m) {
		t.Fatalf("store holds %d bytes by its total but %d by its entries (%d ids in order, %d entries)", s.bytes, sum, len(s.order), len(s.m))
	}
}

// TestResponseStore_byteBudgetEvictsOldestFirst: a put that would take the store past its byte budget evicts the
// oldest entries until the new one fits, and an overwrite of an id replaces its bytes rather than adding to them.
func TestResponseStore_byteBudgetEvictsOldestFirst(t *testing.T) {
	one := responseEntryBytes(storeEntry(1000))
	s := newResponseStore(256)
	s.maxBytes = 3 * one
	for i := range 4 {
		s.put(fmt.Sprintf("r%d", i), storeEntry(1000))
	}
	if s.get("r0") != nil {
		t.Errorf("the oldest entry survived a put past the %d-byte budget (store total %d)", s.maxBytes, s.bytes)
	}
	for _, id := range []string{"r1", "r2", "r3"} {
		if s.get(id) == nil {
			t.Errorf("%s was evicted, want only the oldest gone", id)
		}
	}
	checkStoreBytes(t, s)
	if s.bytes > s.maxBytes {
		t.Errorf("store holds %d bytes, over its %d-byte budget", s.bytes, s.maxBytes)
	}

	s.put("r3", storeEntry(10)) // an overwrite frees the old bytes
	checkStoreBytes(t, s)
	s.put("r4", storeEntry(900))
	if s.get("r1") == nil || s.get("r2") == nil {
		t.Errorf("a put that fits after an overwrite shrank r3 evicted an entry")
	}
	checkStoreBytes(t, s)
}

// TestResponseStore_oversizedEntryIsRefused: an entry larger than the whole budget is not stored and evicts nothing.
func TestResponseStore_oversizedEntryIsRefused(t *testing.T) {
	s := newResponseStore(256)
	s.maxBytes = 4096
	s.put("a", storeEntry(100))
	s.put("b", storeEntry(100))
	s.put("big", storeEntry(5000))
	if s.get("big") != nil {
		t.Errorf("an entry over the %d-byte budget was stored", s.maxBytes)
	}
	if s.get("a") == nil || s.get("b") == nil {
		t.Errorf("refusing an oversized entry evicted others")
	}
	checkStoreBytes(t, s)

	s.put("a", storeEntry(5000)) // an oversized overwrite drops the old conversation instead of serving it under the id
	if s.get("a") != nil {
		t.Errorf("an oversized overwrite left the id's older conversation in place")
	}
	if s.get("b") == nil {
		t.Errorf("an oversized overwrite evicted another entry")
	}
	checkStoreBytes(t, s)
}

// TestNewResponseStore_budgetIsTheProductionConstant pins the budget a server's store gets.
func TestNewResponseStore_budgetIsTheProductionConstant(t *testing.T) {
	if got := newResponseStore(256).maxBytes; got != maxResponseStoreBytes || maxResponseStoreBytes != 64<<20 {
		t.Errorf("store budget %d (constant %d), want 64 MiB", got, maxResponseStoreBytes)
	}
}
