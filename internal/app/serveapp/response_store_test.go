package serveapp

import (
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/app/serveapp/stores"
)

// TestNewResponseStore_budgetIsTheProductionConstant pins the budget a server's store gets.
func TestNewResponseStore_budgetIsTheProductionConstant(t *testing.T) {
	if got := newResponseStore(256).MaxBytes(); got != stores.MaxResponseStoreBytes || stores.MaxResponseStoreBytes != 64<<20 {
		t.Errorf("store budget %d (constant %d), want 64 MiB", got, stores.MaxResponseStoreBytes)
	}
}

// TestResponseEntryBytes_countsCarriedIDs: an assistant message's carried-id blob counts toward the entry's size, at 8 bytes per
// id, so a long reply's ids cannot grow the store past its byte budget unseen.
func TestResponseEntryBytes_countsCarriedIDs(t *testing.T) {
	bare := []chatMessage{{Content: rawStr(strings.Repeat("x", 10))}}
	withIDs := []chatMessage{{
		Content: rawStr(strings.Repeat("x", 10)),
		carried: &idsCarrier{Blob: &idsBlob{Text: "abc", IDs: make([]int, 1000)}},
	}}
	if got, want := responseMessagesBytes("", withIDs)-responseMessagesBytes("", bare), 3+8*1000; got != want {
		t.Errorf("a 1000-id blob adds %d bytes to the entry's size, want %d", got, want)
	}
}
