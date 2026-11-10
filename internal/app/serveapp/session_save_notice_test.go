package serveapp

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/app/loadflags"
)

// A session checkpoint that cannot be written is said, at shutdown (checkpointSessions) and at an unload's drain
// (startDrain): the sessions will not restore, and the operator learns it now rather than at the next start. The run
// goes on either way. The session directory here sits under a regular file, so the save's MkdirAll fails.
func TestSessionCheckpoint_failureIsSaid(t *testing.T) {
	blocker := filepath.Join(t.TempDir(), "a-file")
	if err := os.WriteFile(blocker, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg := config{models: modelFlag{{name: "tiny", path: tinyChatCheckpoint(t)}}, load: loadflags.Flags{Backend: "cpu", Quant: "int8int8"},
		kvSessions: 1, sessionDir: filepath.Join(blocker, "sessions")}
	srv, err := newServer(cfg)
	if err != nil {
		t.Fatalf("newServer: %v", err)
	}
	var out bytes.Buffer
	prev := sessionNoticeOut
	sessionNoticeOut = &out
	t.Cleanup(func() { sessionNoticeOut = prev })

	srv.checkpointSessions(cfg)
	if !strings.Contains(out.String(), "shutdown: model \"tiny\": session checkpoint") {
		t.Errorf("a failed shutdown checkpoint is not said; got %q", out.String())
	}

	out.Reset()
	lms := srv.modelList()
	if len(lms) != 1 {
		t.Fatalf("%d models loaded, want 1", len(lms))
	}
	<-srv.startDrain(lms[0], nil, false)
	if !strings.Contains(out.String(), "unload: model \"tiny\": session checkpoint") {
		t.Errorf("a failed unload checkpoint is not said; got %q", out.String())
	}
}
