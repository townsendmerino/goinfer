// Package batchio is the file format of goinfer's batch runs: the OpenAI-style JSONL that serve's
// POST /v1/batches reads and writes (J4) and that `goinfer-chat --batch` reads and writes (J5), kept in ONE place so a file
// really is interchangeable between the two — the input parser, the byte layout of an output and an error line, and the
// resume scan are the same code on both sides, not two copies that agree today.
//
//	input line   {"custom_id": "a1", "method": "POST", "url": "/v1/chat/completions", "body": {chat request…}}
//	output line  {"custom_id": "a1", "error": null, "id": "batch_req_…", "response": {"body": {…}, "status_code": 200}}
//	error line   {"custom_id": "a2", "error": {"code": "…", "message": "…"}, "id": "batch_req_…", "response": null}
//
// The split is the real API's: a line that produced a response goes to the output file, a line that failed goes to the error
// file, never both and never neither. Resuming (ScanOutput, OpenAppend) is what the CLI adds: it lets a run that died at
// line 14,000 of 20,000 start at 14,001.
package batchio

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
)

// Line is one line of an input file. Method and URL are carried but not acted on (serve's endpoint comes from the batch
// object, the CLI's is always chat completions); Body is the request, decoded by whoever runs the line.
type Line struct {
	CustomID string          `json:"custom_id"`
	Method   string          `json:"method"`
	URL      string          `json:"url"`
	Body     json.RawMessage `json:"body"`
}

// maxLineBytes bounds one input line (a long prompt is a long line).
const maxLineBytes = 16 << 20

// ParseInput splits a JSONL input file into its lines, skipping blank ones. The first malformed line aborts the whole file,
// with a message naming its line number — the messages are the ones serve answers a bad upload with.
func ParseInput(data []byte) ([]Line, error) {
	var lines []Line
	sc := bufio.NewScanner(bytes.NewReader(data))
	sc.Buffer(make([]byte, 0, 64*1024), maxLineBytes)
	lineNo := 0
	for sc.Scan() {
		lineNo++
		raw := strings.TrimSpace(sc.Text())
		if raw == "" {
			continue
		}
		var l Line
		if err := json.Unmarshal([]byte(raw), &l); err != nil {
			return nil, fmt.Errorf("input file line %d: invalid JSON: %s", lineNo, err.Error())
		}
		if l.CustomID == "" {
			return nil, fmt.Errorf("input file line %d: custom_id is required", lineNo)
		}
		lines = append(lines, l)
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("reading input file: %w", err)
	}
	if len(lines) == 0 {
		return nil, errors.New("input file has no request lines")
	}
	return lines, nil
}

// CheckUnique refuses a file in which two lines share a custom_id. OpenAI's and Anthropic's batch APIs require uniqueness, and
// results are matched to requests by it: a resumable run keys its output on it (a second line with the same id would be skipped
// as "already done"), and over HTTP two results under one id cannot be told apart.
func CheckUnique(lines []Line) error {
	ids := make([]string, len(lines))
	for i, l := range lines {
		ids[i] = l.CustomID
	}
	return CheckUniqueIDs(ids)
}

// CheckUniqueIDs is CheckUnique over bare ids, for a request list that never was a file (Anthropic's inline requests). Entries
// are numbered from 1, in the order given.
func CheckUniqueIDs(ids []string) error {
	first := make(map[string]int, len(ids))
	var dup []string
	for i, id := range ids {
		if j, ok := first[id]; ok {
			dup = append(dup, fmt.Sprintf("%q (entries %d and %d)", id, j+1, i+1))
			continue
		}
		first[id] = i
	}
	if len(dup) > 0 {
		const show = 5
		more := ""
		if len(dup) > show {
			more = fmt.Sprintf(" … and %d more", len(dup)-show)
			dup = dup[:show]
		}
		return fmt.Errorf("custom_id must be unique: %s%s", strings.Join(dup, ", "), more)
	}
	return nil
}

// OKLine is the output-file line for a request that produced body. id is the line's own request id.
func OKLine(id, customID string, status int, body any) []byte {
	return marshalLine(map[string]any{
		"id": id, "custom_id": customID,
		"response": map[string]any{"status_code": status, "body": body},
		"error":    nil,
	})
}

// ErrLine is the error-file line for a request that failed.
func ErrLine(id, customID, code, message string) []byte {
	return marshalLine(map[string]any{
		"id": id, "custom_id": customID,
		"response": nil,
		"error":    map[string]any{"code": code, "message": message},
	})
}

func marshalLine(m map[string]any) []byte {
	b, _ := json.Marshal(m)
	return append(b, '\n')
}

// Progress is what an existing output file says about a run.
type Progress struct {
	Done      map[string]bool // custom_ids that hold a completed response
	Lines     int             // complete lines read
	GoodBytes int64           // the length of the file up to and including its last complete line
	Torn      bool            // the file ended in a partial line (a write the process did not finish)
}

// ScanOutput reads an output file and reports which custom_ids are done. A file that does not exist is an empty run.
//
// A torn final line — bytes after the last newline, which is what a process killed mid-write leaves — is tolerated: the
// caller truncates the file to GoodBytes and the line is simply run again. A COMPLETE line that is not valid output JSON is
// not tolerated, because appending after it would bury the corruption under a file that looks fine; the error names the line.
func ScanOutput(path string) (Progress, error) {
	p := Progress{Done: map[string]bool{}}
	f, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return p, nil
	}
	if err != nil {
		return p, err
	}
	defer f.Close()
	r := bufio.NewReaderSize(f, 1<<20)
	for {
		raw, err := r.ReadBytes('\n')
		if err != nil && !errors.Is(err, io.EOF) {
			return p, err
		}
		if len(raw) == 0 && errors.Is(err, io.EOF) {
			return p, nil
		}
		if raw[len(raw)-1] != '\n' { // EOF inside a line
			p.Torn = true
			return p, nil
		}
		p.Lines++
		var rec struct {
			CustomID string          `json:"custom_id"`
			Response json.RawMessage `json:"response"`
		}
		if jerr := json.Unmarshal(bytes.TrimSpace(raw), &rec); jerr != nil || rec.CustomID == "" {
			return p, fmt.Errorf("%s line %d is not a batch output line (%s); refusing to resume past it — fix or remove the file",
				path, p.Lines, describeBad(jerr))
		}
		if len(rec.Response) > 0 && string(rec.Response) != "null" {
			p.Done[rec.CustomID] = true
		}
		p.GoodBytes += int64(len(raw))
	}
}

func describeBad(err error) string {
	if err != nil {
		return "invalid JSON: " + err.Error()
	}
	return "no custom_id"
}

// Writer appends output lines durably: each Append is written and fsynced before it returns, so a line that was reported
// done survives the process, and only the line being written at the moment of a crash can be torn.
type Writer struct{ f *os.File }

// OpenAppend opens path for appending after truncating it to keep bytes (Progress.GoodBytes; 0 starts a fresh file).
func OpenAppend(path string, keep int64) (*Writer, error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, err
	}
	if err := f.Truncate(keep); err != nil {
		f.Close()
		return nil, err
	}
	if _, err := f.Seek(keep, io.SeekStart); err != nil {
		f.Close()
		return nil, err
	}
	return &Writer{f: f}, nil
}

// Append writes one complete line (it must end in a newline, as OKLine and ErrLine return it) and syncs it to disk.
func (w *Writer) Append(line []byte) error {
	if _, err := w.f.Write(line); err != nil {
		return err
	}
	return w.f.Sync()
}

// Close closes the file.
func (w *Writer) Close() error { return w.f.Close() }

// ErrorPath is where a run's failed lines go, beside the output file: out.jsonl → out.errors.jsonl.
func ErrorPath(out string) string {
	if base, ok := strings.CutSuffix(out, ".jsonl"); ok {
		return base + ".errors.jsonl"
	}
	return out + ".errors"
}

// Orphans returns the done custom_ids that no input line carries, sorted — a sign the output file belongs to a different
// input, which the caller reports instead of quietly mixing two runs.
func Orphans(done map[string]bool, lines []Line) []string {
	in := make(map[string]bool, len(lines))
	for _, l := range lines {
		in[l.CustomID] = true
	}
	var out []string
	for id := range done {
		if !in[id] {
			out = append(out, id)
		}
	}
	sort.Strings(out)
	return out
}
