package banner

import (
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

func TestModelAndServerLines(t *testing.T) {
	lines := ModelLines(Facts{
		DecodePath:   "metal-resident (int4)",
		PrefillPath:  "metal-batched",
		HasTemplate:  true,
		ToolCallForm: true,
		Resident:     true,
		KVSlots:      2,
		CtxWindow:    16384,
		MaxPositions: 32768,
		KV:           []decoder.KVPrecisionReport{{Cache: "resident", Requested: "auto", Delivered: "f16", Reason: "auto's pick"}},
	}, Config{KVSessions: 4})

	joined := strings.Join(lines, "\n")
	for _, want := range []string{
		"decode path: metal-resident (int4)",
		"KV f16 (requested auto",
		"KV plan: 2 conversations x 16384 positions",
		"tools (constrainable call form)",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("banner missing %q:\n%s", want, joined)
		}
	}

	srv := strings.Join(ServerLines(ServerConfig{HasModels: true, Web: true}), "\n")
	if !strings.Contains(srv, "/ (web UI)") || strings.Contains(srv, "after a model is loaded") {
		t.Errorf("ServerLines = %q", srv)
	}
}

func TestConcurrencyAndKVPoolLines(t *testing.T) {
	c := ConcurrencyLine(Facts{Concurrent: 4, CPUBatched: true}, Config{MaxConcurrent: 4, KVSessions: 4})
	if !strings.Contains(c, "decode tokens batched on the CPU") {
		t.Errorf("ConcurrencyLine = %q", c)
	}
	pool := KVPoolBannerLine(Facts{
		CtxWindow:    32768,
		KVShareState: "on",
		KVPoolDetail: decoder.KVPoolDetail{Positions: 58000, Guaranteed: 14500, BlockRows: 64, Groups: []string{"full"}, Precision: "f16", Bytes: 2 << 30},
	})
	if !strings.Contains(pool, "58,000 tokens in blocks of 64") || !strings.Contains(pool, "up to 32,768") {
		t.Errorf("KVPoolBannerLine = %q", pool)
	}
}
