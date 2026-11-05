package serveapp

import (
	"strings"
	"testing"
	"time"
)

// KV8a's gate 4(b) reads the save's own time from every request's log line, and gate 5 reads where a restore came from. Each field
// appears only when its event happened, as a key=value token (KV0-H's analyzer splits the line on spaces and keys on "=").
func TestReqTrace_checkpointFields(t *testing.T) {
	tr := &reqTrace{start: time.Now(), model: "m"}
	if l := tr.line("/v1/chat/completions", 200, time.Now()); strings.Contains(l, "ckpt_") {
		t.Errorf("a request with no checkpoint event: %q", l)
	}
	tr.ckptSaved, tr.ckptSave = true, 1234*time.Microsecond
	tr.ckptRestored, tr.ckptRestore = 6139, 812*time.Microsecond
	l := tr.line("/v1/chat/completions", 200, time.Now())
	kv := map[string]string{}
	for _, tok := range strings.Fields(l) {
		if k, v, ok := strings.Cut(tok, "="); ok {
			kv[k] = v
		}
	}
	if kv["ckpt_save_ms"] != "1.234" || kv["ckpt_restore"] != "6139" || kv["ckpt_restore_ms"] != "0.812" {
		t.Errorf("save and restore fields: %q", l)
	}
	tr = &reqTrace{start: time.Now(), model: "m", ckptDeclined: "suffix under the final-pass minimum"}
	if l := tr.line("/v1/chat/completions", 200, time.Now()); !strings.Contains(l, ` ckpt_declined="suffix under the final-pass minimum"`) {
		t.Errorf("a declined hit: %q", l)
	}
}
