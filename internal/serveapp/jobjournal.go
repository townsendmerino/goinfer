package serveapp

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// jobJournal is the optional durability layer: one JSONL line per job state transition, appended to <dir>/jobs.jsonl. Each
// line is a FULL snapshot of the job at that point, not a diff, so reconstruction on restart is "keep the last line per id".
// dir is created at sessionDirPerm (0o700) and the file opened at sessionFilePerm (0o600), for the reason sessions.go's
// KV-snapshot permissions have: a job record replays the prompt, so it gets the same owner-only treatment
// (docs/tasks/task-work-queue-2026-09.md).
type jobJournal struct {
	mu sync.Mutex
	f  *os.File
}

// newJobJournal creates dir if needed, replays any existing journal, opens it append-only, and
// completes the record for anything the replay marked interrupted (appending that transition now
// rather than leaving the journal's own last line for that id silently say "running" forever).
// Returns the journal and the reconstructed job map (by id) for jobStore to adopt.
func newJobJournal(dir string) (*jobJournal, map[string]*job, error) {
	if err := os.MkdirAll(dir, sessionDirPerm); err != nil {
		return nil, nil, fmt.Errorf("job journal: mkdir %s: %w", dir, err)
	}
	path := filepath.Join(dir, "jobs.jsonl")
	restored, err := loadJournal(path)
	if err != nil {
		return nil, nil, err
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, sessionFilePerm)
	if err != nil {
		return nil, nil, fmt.Errorf("job journal: open %s: %w", path, err)
	}
	jj := &jobJournal{f: f}
	for _, j := range restored {
		if j.State == jobInterrupted {
			if err := jj.record(*j); err != nil {
				fmt.Fprintf(os.Stderr, "job journal: recording interrupted %s: %v\n", j.ID, err)
			}
		}
	}
	return jj, restored, nil
}

// close closes the journal's file. serve keeps its journal for the process's life and never calls it; tests do,
// because Windows cannot delete a file that is still open (TempDir's cleanup fails on one).
func (jj *jobJournal) close() error {
	jj.mu.Lock()
	defer jj.mu.Unlock()
	return jj.f.Close()
}

// record appends one line (a full job snapshot) and fsyncs when State is terminal, the point where losing this line matters
// (mirroring sessions.go's "fsync on terminal states"; this is the record an operator reads after a halt).
func (jj *jobJournal) record(j job) error {
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
	case jobDone, jobFailed, jobCancelled, jobInterrupted:
		if err := jj.f.Sync(); err != nil {
			return fmt.Errorf("fsync: %w", err)
		}
	}
	return nil
}

// loadJournal replays every line in path (absent ⇒ empty map, not an error — the first run ever),
// keeping the LAST line per id, then rewrites any job still "running" to "interrupted": a
// transition into running with no later terminal line means the process ended (crashed, was
// killed, lost power) mid-generation. Never silently "failed" — that would claim the generation
// ran and produced an error, which nobody observed; "interrupted" is what actually happened.
func loadJournal(path string) (map[string]*job, error) {
	f, err := os.Open(path)
	if os.IsNotExist(err) {
		return map[string]*job{}, nil
	}
	if err != nil {
		return nil, fmt.Errorf("job journal: open %s: %w", path, err)
	}
	defer f.Close()

	out := map[string]*job{}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 4<<20) // a prompt's ids can be long; don't truncate silently
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		var j job
		if err := json.Unmarshal(line, &j); err != nil {
			return nil, fmt.Errorf("job journal: corrupt line in %s: %w", path, err)
		}
		out[j.ID] = &j // j is fresh per loop iteration (declared inside the body) — safe to take its address
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("job journal: read %s: %w", path, err)
	}
	for _, j := range out {
		if j.State == jobRunning {
			j.State = jobInterrupted
			now := time.Now()
			j.Finished = &now
		}
	}
	return out, nil
}
