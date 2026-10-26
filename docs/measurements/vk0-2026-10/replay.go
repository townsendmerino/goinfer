//go:build ignore

// replay.go replays decoder/parity_manifest_test.go:freshDepsHash over every commit of a window, reading every
// input from git objects rather than the working tree, and writes one JSON record per step to stdout.
//
//	go run docs/measurements/vk0-2026-10/replay.go -end <sha> -days 30 > steps.jsonl
//	go run docs/measurements/vk0-2026-10/replay.go -end <sha> -check   # the proof: replay == committed deps_hash at <sha>
//
// Each step is hashed twice: with today's function (Go files as their token stream, from 5ff89161) and with the function
// of the day (raw bytes before 5ff89161), chosen by whether the step's own parity_manifest_test.go defines goCodeTokens.
// Both read the step's own manifest (sets, uses, own) and the step's root go.mod aikit require, as freshDepsHash does.
package main

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"go/scanner"
	"go/token"
	"io"
	"os"
	"os/exec"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

// parityManifest and familyParity mirror the fields freshDepsHash reads, plus the ones a refresh can change.
type parityManifest struct {
	SharedSets map[string][]string     `json:"shared_sets"`
	Families   map[string]familyParity `json:"families"`
}

type familyParity struct {
	Uses        []string        `json:"uses"`
	Own         []string        `json:"own"`
	DepsHash    string          `json:"deps_hash"`
	Status      string          `json:"status"`
	ValidatedAt json.RawMessage `json:"validated_at"`
	Date        json.RawMessage `json:"date"`
	Metrics     json.RawMessage `json:"metrics"`
}

// familyDepFiles is decoder/parity_manifest_test.go:familyDepFiles, verbatim.
func familyDepFiles(m *parityManifest, fam familyParity) []string {
	seen := map[string]bool{}
	for _, set := range fam.Uses {
		for _, f := range m.SharedSets[set] {
			seen[f] = true
		}
	}
	for _, f := range fam.Own {
		seen[f] = true
	}
	out := make([]string, 0, len(seen))
	for f := range seen {
		out = append(out, f)
	}
	sort.Strings(out)
	return out
}

// aikitVersionRE is decoder/parity_manifest_test.go:aikitVersionRE, verbatim.
var aikitVersionRE = regexp.MustCompile(`(?m)^\s*github\.com/townsendmerino/aikit (v[0-9][^\s]*)`)

// goCodeTokens is decoder/parity_manifest_test.go:goCodeTokens, verbatim.
func goCodeTokens(name string, src []byte) ([]byte, error) {
	fset := token.NewFileSet()
	file := fset.AddFile(name, -1, len(src))
	var s scanner.Scanner
	var scanErr error
	s.Init(file, src, func(pos token.Position, msg string) {
		if scanErr == nil {
			scanErr = fmt.Errorf("%s: %s", pos, msg)
		}
	}, scanner.ScanComments)
	var out bytes.Buffer
	for {
		_, tok, lit := s.Scan()
		if tok == token.EOF {
			break
		}
		if tok == token.COMMENT && !strings.HasPrefix(lit, "//go:") && !strings.HasPrefix(lit, "// +build") {
			continue
		}
		out.WriteString(tok.String())
		out.WriteByte(' ')
		out.WriteString(lit)
		out.WriteByte('\n')
	}
	if scanErr != nil {
		return nil, scanErr
	}
	return out.Bytes(), nil
}

// repo reads git objects through one long-lived `git cat-file --batch`.
type repo struct {
	in   io.WriteCloser
	out  *bufio.Reader
	blob map[string][]byte // object id -> content
	tok  map[string][]byte // object id -> goCodeTokens(content)
}

func newRepo() (*repo, error) {
	cmd := exec.Command("git", "cat-file", "--batch")
	in, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	out, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return &repo{in: in, out: bufio.NewReaderSize(out, 1<<20), blob: map[string][]byte{}, tok: map[string][]byte{}}, nil
}

// object returns the content of rev (a "<commit>:<path>" or an object id); ok is false when it does not exist.
func (r *repo) object(rev string) (id string, b []byte, ok bool, err error) {
	if _, err = fmt.Fprintln(r.in, rev); err != nil {
		return
	}
	hdr, err := r.out.ReadString('\n')
	if err != nil {
		return
	}
	f := strings.Fields(hdr)
	if len(f) == 2 && f[1] == "missing" {
		return "", nil, false, nil
	}
	if len(f) != 3 {
		return "", nil, false, fmt.Errorf("cat-file %s: %q", rev, hdr)
	}
	n, err := strconv.Atoi(f[2])
	if err != nil {
		return
	}
	b = make([]byte, n+1)
	if _, err = io.ReadFull(r.out, b); err != nil {
		return
	}
	return f[0], b[:n], true, nil
}

func git(args ...string) string {
	out, err := exec.Command("git", args...).Output()
	if err != nil {
		fmt.Fprintf(os.Stderr, "git %v: %v\n", args, err)
		os.Exit(1)
	}
	return string(out)
}

// snapshot is one commit's inputs and both hashes per family.
type snapshot struct {
	commit    string
	manifest  *parityManifest
	aikit     string
	tokenMode bool              // this commit's own freshDepsHash reads Go tokens
	pathID    map[string]string // dependency path -> blob id ("" when missing)
	pathTok   map[string]string // dependency path -> sha256 of what today's function hashes for it
	today     map[string]string // family -> replayed hash, today's function
	ofDay     map[string]string // family -> replayed hash, the function of the day
	errs      map[string]string // family -> why it could not be hashed
}

func (r *repo) snapshot(commit string) (*snapshot, error) {
	_, mb, ok, err := r.object(commit + ":testdata/parity_manifest.json")
	if err != nil || !ok {
		return nil, fmt.Errorf("%s: no manifest (%v)", commit, err)
	}
	s := &snapshot{commit: commit, manifest: &parityManifest{}, pathID: map[string]string{}, pathTok: map[string]string{},
		today: map[string]string{}, ofDay: map[string]string{}, errs: map[string]string{}}
	if err := json.Unmarshal(mb, s.manifest); err != nil {
		return nil, fmt.Errorf("%s: manifest: %v", commit, err)
	}
	_, gm, _, err := r.object(commit + ":go.mod")
	if err != nil {
		return nil, err
	}
	if m := aikitVersionRE.FindStringSubmatch(string(gm)); m != nil {
		s.aikit = m[1]
	}
	_, tf, _, err := r.object(commit + ":decoder/parity_manifest_test.go")
	if err != nil {
		return nil, err
	}
	s.tokenMode = bytes.Contains(tf, []byte("func goCodeTokens("))

	for name, fam := range s.manifest.Families {
		hToday, hDay := sha256.New(), sha256.New()
		var famErr string
		for _, p := range familyDepFiles(s.manifest, fam) {
			id, ok := s.pathID[p]
			if !ok {
				var b []byte
				var found bool
				id, b, found, err = r.object(commit + ":" + p)
				if err != nil {
					return nil, err
				}
				if !found {
					id = ""
				} else if _, cached := r.blob[id]; !cached {
					r.blob[id] = b
				}
				s.pathID[p] = id
			}
			if id == "" {
				famErr = "missing " + p
				break
			}
			raw := r.blob[id]
			tok := raw
			if strings.HasSuffix(p, ".go") {
				if t, ok := r.tok[id]; ok {
					tok = t
				} else if t, err := goCodeTokens(p, raw); err != nil {
					famErr = err.Error()
					break
				} else {
					r.tok[id] = t
					tok = t
				}
			}
			if _, ok := s.pathTok[p]; !ok {
				d := sha256.Sum256(tok)
				s.pathTok[p] = hex.EncodeToString(d[:8])
			}
			for _, w := range []struct {
				h io.Writer
				b []byte
			}{{hToday, tok}, {hDay, raw}} {
				w.h.Write([]byte(p))
				w.h.Write([]byte{0})
				w.h.Write(w.b)
				w.h.Write([]byte{0})
			}
		}
		if famErr == "" && s.aikit == "" {
			famErr = "root go.mod does not require aikit"
		}
		if famErr != "" {
			s.errs[name] = famErr
			continue
		}
		hToday.Write([]byte("aikit_version=" + s.aikit))
		hDay.Write([]byte("aikit_version=" + s.aikit))
		s.today[name] = "sha256:" + hex.EncodeToString(hToday.Sum(nil))
		if s.tokenMode {
			s.ofDay[name] = s.today[name]
		} else {
			s.ofDay[name] = "sha256:" + hex.EncodeToString(hDay.Sum(nil))
		}
	}
	return s, nil
}

// step is one commit's change against its parents. A family counts as re-staled by a commit when its hash there differs
// from its hash at every parent: a merge that only brings one side's validated state forward re-stales nothing, and a
// merge whose combination neither side held re-stales the families it reaches.
type step struct {
	Commit  string `json:"commit"`
	Parents int    `json:"parents"`
	Date    string `json:"date"`
	Subject string `json:"subject"`
	// StaleToday / StaleOfDay: families re-staled at this commit, by today's function and by the function of the day.
	StaleToday []string `json:"stale_today"`
	StaleOfDay []string `json:"stale_of_day"`
	// Triggers, per input of today's hash that differs from the first parent, over the families in StaleToday:
	// "set:<name>:<path>", "own:<path>", "aikit:<old>-><new>", "deps:<family>" (its file list changed in the manifest).
	Triggers []string `json:"triggers"`
	// TriggerFams maps each trigger to how many of StaleToday it reaches.
	TriggerFams map[string]int `json:"trigger_families"`
	// ByteOnly: dependency paths whose bytes changed against the first parent but whose tokens did not.
	ByteOnly []string `json:"byte_only"`
	// Goldens: forward golden files changed against the first parent, each "<A|M|D|R> <path>".
	Goldens []string `json:"goldens"`
	// CommittedStaleOfDay: validated families whose committed deps_hash is not the function of the day's replay, the
	// families TestParityManifest_fresh fails on at this commit.
	CommittedStaleOfDay []string `json:"committed_stale_of_day"`
	// Refreshed: families whose committed deps_hash differs from every parent's; MetricsMoved: of those, the ones whose
	// metrics, validated_at, date or status differ from the first parent's too.
	Refreshed    []string `json:"refreshed"`
	MetricsMoved []string `json:"metrics_moved"`
	Errors       []string `json:"errors,omitempty"`
}

// goldenRE matches a forward golden: a .json or .json.gz under testdata/ or decoder/testdata/ whose name says golden,
// outside the chat-template goldens and the decision records.
var goldenRE = regexp.MustCompile(`^(decoder/)?testdata/([^/]+/)*[^/]*golden[^/]*\.json(\.gz)?$`)

func isForwardGolden(p string) bool {
	return goldenRE.MatchString(p) && !strings.HasPrefix(p, "testdata/chat_") && !strings.HasPrefix(p, "testdata/decisions/")
}

func setsOf(m *parityManifest, fam familyParity, p string) []string {
	var out []string
	for _, u := range fam.Uses {
		for _, f := range m.SharedSets[u] {
			if f == p {
				out = append(out, "set:"+u+":"+p)
			}
		}
	}
	for _, f := range fam.Own {
		if f == p {
			out = append(out, "own:"+p)
		}
	}
	return out
}

// newAtAll reports whether cur differs from get(p) for every parent p that has the family.
func newAtAll(parents []*snapshot, get func(*snapshot) (string, bool), cur string) bool {
	seen := false
	for _, p := range parents {
		h, ok := get(p)
		if !ok {
			continue
		}
		seen = true
		if h == cur {
			return false
		}
	}
	return seen
}

func diff(parents []*snapshot, cur *snapshot, commit, date, subject string, changed []string) step {
	st := step{Commit: commit, Parents: len(parents), Date: date, Subject: subject, TriggerFams: map[string]int{}}
	prev := parents[0]
	trig := map[string]bool{}
	for name, h := range cur.today {
		if !newAtAll(parents, func(p *snapshot) (string, bool) { v, ok := p.today[name]; return v, ok }, h) {
			continue
		}
		st.StaleToday = append(st.StaleToday, name)
		fam := cur.manifest.Families[name]
		pf := prev.manifest.Families[name]
		var these []string
		if strings.Join(familyDepFiles(prev.manifest, pf), "\x00") != strings.Join(familyDepFiles(cur.manifest, fam), "\x00") {
			these = append(these, "deps:"+name)
		}
		for _, p := range familyDepFiles(cur.manifest, fam) {
			if pt, ok := prev.pathTok[p]; ok && pt != cur.pathTok[p] {
				these = append(these, setsOf(cur.manifest, fam, p)...)
			}
		}
		if prev.aikit != cur.aikit {
			these = append(these, "aikit:"+prev.aikit+"->"+cur.aikit)
		}
		for _, t := range these {
			trig[t] = true
			st.TriggerFams[t]++
		}
	}
	for name, h := range cur.ofDay {
		if newAtAll(parents, func(p *snapshot) (string, bool) { v, ok := p.ofDay[name]; return v, ok }, h) {
			st.StaleOfDay = append(st.StaleOfDay, name)
		}
	}
	for t := range trig {
		st.Triggers = append(st.Triggers, t)
	}
	for p, id := range cur.pathID {
		if pid, ok := prev.pathID[p]; ok && pid != id && prev.pathTok[p] == cur.pathTok[p] {
			st.ByteOnly = append(st.ByteOnly, p)
		}
	}
	for _, c := range changed {
		// c is a `git diff-tree --name-status` entry: "<status>\t<path>" ("R100\t<old>\t<new>" for a rename).
		f := strings.Split(c, "\t")
		if len(f) >= 2 && isForwardGolden(f[len(f)-1]) {
			st.Goldens = append(st.Goldens, f[0][:1]+" "+f[len(f)-1])
		}
	}
	for name, fam := range cur.manifest.Families {
		if fam.Status == "validated" && cur.ofDay[name] != fam.DepsHash {
			st.CommittedStaleOfDay = append(st.CommittedStaleOfDay, name)
		}
		if !newAtAll(parents, func(p *snapshot) (string, bool) {
			f, ok := p.manifest.Families[name]
			return f.DepsHash, ok
		}, fam.DepsHash) {
			continue
		}
		st.Refreshed = append(st.Refreshed, name)
		pf := prev.manifest.Families[name]
		if !bytes.Equal(pf.Metrics, fam.Metrics) || !bytes.Equal(pf.ValidatedAt, fam.ValidatedAt) ||
			!bytes.Equal(pf.Date, fam.Date) || pf.Status != fam.Status {
			st.MetricsMoved = append(st.MetricsMoved, name)
		}
	}
	for name, e := range cur.errs {
		st.Errors = append(st.Errors, name+": "+e)
	}
	for _, s := range [][]string{st.StaleToday, st.StaleOfDay, st.Triggers, st.ByteOnly, st.Goldens, st.Refreshed, st.MetricsMoved, st.Errors, st.CommittedStaleOfDay} {
		sort.Strings(s)
	}
	return st
}

func main() {
	end := flag.String("end", "origin/main", "last commit of the window")
	days := flag.Int("days", 30, "window length, back from the end commit's committer date")
	check := flag.Bool("check", false, "only prove the replay: every family's replayed hash at -end equals its committed deps_hash")
	flag.Parse()

	r, err := newRepo()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	endSHA := strings.TrimSpace(git("rev-parse", *end))

	if *check {
		s, err := r.snapshot(endSHA)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		names := make([]string, 0, len(s.manifest.Families))
		for n := range s.manifest.Families {
			names = append(names, n)
		}
		sort.Strings(names)
		bad := 0
		for _, n := range names {
			want := s.manifest.Families[n].DepsHash
			got := s.today[n]
			mark := "ok"
			if got != want {
				mark = "MISMATCH"
				bad++
			}
			fmt.Printf("%-24s %-12s %s %s\n", n, s.manifest.Families[n].Status, mark, got)
		}
		fmt.Printf("commit %s aikit %s: %d families, %d match, %d mismatch, %d unhashable\n",
			endSHA, s.aikit, len(names), len(names)-bad, bad, len(s.errs))
		if bad > 0 || len(s.errs) > 0 {
			os.Exit(1)
		}
		return
	}

	endUnix, _ := strconv.ParseInt(strings.TrimSpace(git("log", "-1", "--format=%ct", endSHA)), 10, 64)
	since := time.Unix(endUnix, 0).Add(-time.Duration(*days) * 24 * time.Hour)
	log := strings.TrimSpace(git("log", "--topo-order", "--reverse", "--format=%H%x09%P%x09%cI%x09%s",
		"--since="+since.Format(time.RFC3339), endSHA))
	lines := strings.Split(log, "\n")
	snaps := map[string]*snapshot{}
	get := func(c string) *snapshot {
		if s, ok := snaps[c]; ok {
			return s
		}
		s, err := r.snapshot(c)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		snaps[c] = s
		return s
	}
	fmt.Fprintf(os.Stderr, "window %s .. %s: %d commits\n", since.Format(time.RFC3339), endSHA[:8], len(lines))
	enc := json.NewEncoder(os.Stdout)
	t0 := time.Now()
	for i, ln := range lines {
		f := strings.SplitN(ln, "\t", 4)
		var parents []*snapshot
		for _, p := range strings.Fields(f[1]) {
			parents = append(parents, get(p))
		}
		cur := get(f[0])
		changed := strings.Split(strings.TrimSpace(git("diff-tree", "--no-commit-id", "-r", "--name-status", f[0]+"^1", f[0])), "\n")
		if err := enc.Encode(diff(parents, cur, f[0], f[2], f[3], changed)); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		if (i+1)%500 == 0 {
			fmt.Fprintf(os.Stderr, "  %d/%d commits, %s\n", i+1, len(lines), time.Since(t0).Round(time.Second))
		}
	}
	fmt.Fprintf(os.Stderr, "done: %d commits in %s\n", len(lines), time.Since(t0).Round(time.Second))
}
