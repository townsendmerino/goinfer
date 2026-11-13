package jobs

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// dirPerm and filePerm are the journal's owner-only modes; TestJobJournal_permissions pins them.
const (
	dirPerm  = 0o700
	filePerm = 0o600
)

// Journal is the optional durability layer: one JSONL line per job state transition, appended to <dir>/jobs.jsonl. Each
// line is a FULL snapshot of the job at that point, not a diff, so reconstruction on restart is "keep the last line per id".
// dir is created at dirPerm and the file opened at filePerm, owner-only like the session KV blobs (sessionDirPerm and
// sessionFilePerm in package serveapp): a job record replays the prompt (docs/tasks/task-work-queue-2026-09.md).
type Journal struct {
	mu sync.Mutex
	f  *os.File
}

// NewJournal creates dir if needed, replays any existing journal, opens it append-only, and
// completes the record for anything the replay marked interrupted (appending that transition now
// rather than leaving the journal's own last line for that id silently say "running" forever).
// Returns the journal and the reconstructed job map (by id) for Store to adopt.
func NewJournal(dir string) (*Journal, map[string]*Job, error) {
	if err := os.MkdirAll(dir, dirPerm); err != nil {
		return nil, nil, fmt.Errorf("job journal: mkdir %s: %w", dir, err)
	}
	path := filepath.Join(dir, "jobs.jsonl")
	restored, err := LoadJournal(path)
	if err != nil {
		return nil, nil, err
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, filePerm)
	if err != nil {
		return nil, nil, fmt.Errorf("job journal: open %s: %w", path, err)
	}
	jj := &Journal{f: f}
	for _, j := range restored {
		if j.State == Interrupted {
			if err := jj.Record(*j); err != nil {
				fmt.Fprintf(os.Stderr, "job journal: recording interrupted %s: %v\n", j.ID, err)
			}
		}
	}
	return jj, restored, nil
}

// Close closes the journal's file. serve keeps its journal for the process's life and never calls it; tests do,
// because Windows cannot delete a file that is still open (TempDir's cleanup fails on one).
func (jj *Journal) Close() error {
	jj.mu.Lock()
	defer jj.mu.Unlock()
	return jj.f.Close()
}

// Record appends one line (a full job snapshot) and fsyncs when State is terminal, the point where losing this line matters
// (mirroring sessions.go's "fsync on terminal states"; this is the record an operator reads after a halt).
func (jj *Journal) Record(j Job) error {
	line, err := json.Marshal(j)
	if err != nil {
		return fmt.Errorf("marshal: %w", err)
	}
	line = append(line, '\n')
	jj.mu.Lock()
	defer jj.mu.Unlock()
	if _, err := jj.f.Write(line); err != nil {
		return fmt.Errorf("write: %w", err)
	}
	switch j.State {
	case Done, Failed, Cancelled, Interrupted:
		if err := jj.f.Sync(); err != nil {
			return fmt.Errorf("fsync: %w", err)
		}
	}
	return nil
}

// LoadJournal replays every line in path (absent ⇒ empty map, not an error — the first run ever),
// keeping the LAST line per id, then rewrites any job still "running" to "interrupted": a
// transition into running with no later terminal line means the process ended (crashed, was
// killed, lost power) mid-generation. Never silently "failed" — that would claim the generation
// ran and produced an error, which nobody observed; "interrupted" is what actually happened.
func LoadJournal(path string) (map[string]*Job, error) {
	f, err := os.Open(path)
	if os.IsNotExist(err) {
		return map[string]*Job{}, nil
	}
	if err != nil {
		return nil, fmt.Errorf("job journal: open %s: %w", path, err)
	}
	defer f.Close()

	out := map[string]*Job{}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 4<<20) // a prompt's ids can be long; don't truncate silently
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		var j Job
		if err := json.Unmarshal(line, &j); err != nil {
			return nil, fmt.Errorf("job journal: corrupt line in %s: %w", path, err)
		}
		out[j.ID] = &j // j is fresh per loop iteration (declared inside the body) — safe to take its address
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("job journal: read %s: %w", path, err)
	}
	for _, j := range out {
		if j.State == Running {
			j.State = Interrupted
			now := time.Now()
			j.Finished = &now
		}
	}
	return out, nil
}
