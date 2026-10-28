package serveapp

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// The scheduling census (docs/tasks/task-subsystems-2026-10.md § "SB6 — the scheduling census"): today's scheduling
// decisions replayed over fixed traces and compared with goldens under testdata/sched/, so a move of the policy out of
// this package can show it changed nothing. -sched.update rewrites the goldens.
var updateSched = flag.Bool("sched.update", false, "rewrite the scheduling census goldens under testdata/sched/")

// schedTrace is one admission trace: a model's turn count and -max-queue, then events in virtual time (milliseconds).
// session, class and deadline are recorded beside the outcome and passed to nothing: admission has no field for them.
type schedTrace struct {
	Name        string       `json:"name"`
	About       string       `json:"about"`
	Concurrency int          `json:"concurrency"`
	MaxQueue    int          `json:"max_queue"`
	Events      []schedEvent `json:"events"`
}

type schedEvent struct {
	T        int    `json:"t"`
	Op       string `json:"op"` // arrive | finish | cancel | setcap | halt | resume
	ID       string `json:"id,omitempty"`
	Session  string `json:"session,omitempty"`
	Class    string `json:"class,omitempty"`
	Deadline int    `json:"deadline,omitempty"`
	Prompt   []int  `json:"prompt,omitempty"`
	N        int    `json:"n,omitempty"`
	Reason   string `json:"reason,omitempty"`
}

type schedWaiter struct {
	ID    string `json:"id"`
	Place int    `json:"place"`
}

type schedRefusal struct {
	ID     string `json:"id"`
	State  string `json:"state"`
	Status int    `json:"status"`
	Reason string `json:"reason"`
}

// schedStep is the state after one event has settled.
type schedStep struct {
	T        int            `json:"t"`
	Event    string         `json:"event"`
	Granted  []string       `json:"granted,omitempty"`
	Refused  []schedRefusal `json:"refused,omitempty"`
	Running  []string       `json:"running"`
	Queue    []schedWaiter  `json:"queue"`
	Waiting  int            `json:"waiting"`
	Load     int            `json:"load"`
	Occupied bool           `json:"occupied"` // position's running report: a turn is held
	QueueLen int            `json:"queue_slots_held"`
}

type schedOutcome struct {
	ID           string `json:"id"`
	Session      string `json:"session,omitempty"`
	Class        string `json:"class,omitempty"`
	Deadline     int    `json:"deadline,omitempty"`
	PromptLen    int    `json:"prompt_len,omitempty"`
	Arrived      int    `json:"arrived"`
	Outcome      string `json:"outcome"` // granted | queue_full | cancelled | halted
	GrantedAt    *int   `json:"granted_at,omitempty"`
	Wait         *int   `json:"wait,omitempty"`
	RefusedAt    *int   `json:"refused_at,omitempty"`
	Finished     *int   `json:"finished,omitempty"`
	GrantOrder   int    `json:"grant_order,omitempty"`            // 1 = first granted
	SharedPrev   *int   `json:"shared_with_prev_grant,omitempty"` // common prefix with the previously granted prompt
	PastDeadline *bool  `json:"granted_past_deadline,omitempty"`
}

type schedRecord struct {
	Trace    string         `json:"trace"`
	About    string         `json:"about"`
	Steps    []schedStep    `json:"steps"`
	Requests []schedOutcome `json:"requests"`
}

// schedReq is one request's goroutine as the simulator sees it.
type schedReq struct {
	ev     schedEvent
	cancel context.CancelFunc
	ctx    context.Context
	done   chan struct{}
	ok     bool
	why    refusal
	out    schedOutcome
}

// schedSim drives a bare loadedModel's real admission (tryEnter, exit, position, setCap) one event at a time.
type schedSim struct {
	t      *testing.T
	lm     *loadedModel
	haltMu sync.Mutex
	halt   *haltInfo
	reqs   map[string]*schedReq
	order  []string
	run    []string // granted and not finished, in grant order
	grants int
	prev   []int // the prompt of the last request granted
}

// newSchedSim sizes the model's admission and queue with the real setConcurrency. Above one turn it needs a model that is
// safe to run concurrently on the CPU, which m is; at one turn the model is left nil, as setConcurrency then needs none.
func newSchedSim(t *testing.T, tr schedTrace, m *decoder.Model) *schedSim {
	lm := &loadedModel{name: "census"}
	if tr.Concurrency > 1 {
		lm.model = m
	}
	lm.setConcurrency(config{maxConcurrent: tr.Concurrency, maxQueue: tr.MaxQueue})
	if lm.concurrent != tr.Concurrency {
		t.Fatalf("census: setConcurrency chose %d turns, the trace asks for %d", lm.concurrent, tr.Concurrency)
	}
	return &schedSim{t: t, lm: lm, reqs: map[string]*schedReq{}}
}

func (s *schedSim) haltState() *haltInfo {
	s.haltMu.Lock()
	defer s.haltMu.Unlock()
	return s.halt
}

// settle waits until every live request has returned from tryEnter or stands in the queue. Admission's state changes
// under its own lock before a goroutine wakes, so once this holds, nothing moves until the next event.
//
// must is a request that has to have returned: a waiter whose context was just cancelled still stands in the queue
// until its goroutine takes the lock to leave.
func (s *schedSim) settle(must string) {
	deadline := time.Now().Add(5 * time.Second)
	for {
		still := false
		if r := s.reqs[must]; r != nil {
			select {
			case <-r.done:
			default:
				still = true
			}
		}
		for _, id := range s.order {
			r := s.reqs[id]
			select {
			case <-r.done:
				continue
			default:
			}
			if _, _, _, ok := s.lm.turns.position(id); !ok {
				still = true
				break
			}
		}
		if !still {
			return
		}
		if time.Now().After(deadline) {
			s.t.Fatal("census: admission did not settle within 5s")
		}
		time.Sleep(50 * time.Microsecond)
	}
}

func (s *schedSim) step(e schedEvent) schedStep {
	before := map[string]bool{}
	for _, id := range s.order {
		select {
		case <-s.reqs[id].done:
			before[id] = true
		default:
		}
	}
	label, must := e.Op, ""
	switch e.Op {
	case "arrive":
		label += " " + e.ID
		ctx, cancel := context.WithCancel(context.Background())
		r := &schedReq{ev: e, ctx: ctx, cancel: cancel, done: make(chan struct{})}
		r.out = schedOutcome{ID: e.ID, Session: e.Session, Class: e.Class, Deadline: e.Deadline, PromptLen: len(e.Prompt), Arrived: e.T}
		s.reqs[e.ID] = r
		s.order = append(s.order, e.ID)
		go func() {
			defer close(r.done)
			ok, haltReason := s.lm.tryEnter(ctx, admissionRecord{promptIDs: e.Prompt, id: e.ID}, s.haltState)
			r.ok = ok
			if !ok {
				r.why = notAdmitted(ctx, haltReason, s.lm)
			}
		}()
	case "finish":
		label += " " + e.ID
		r := s.reqs[e.ID]
		if r == nil || !r.ok || r.out.Finished != nil {
			s.t.Fatalf("census: finish %s, which is not running", e.ID)
		}
		v := e.T
		r.out.Finished = &v
		s.run = slices.DeleteFunc(s.run, func(id string) bool { return id == e.ID })
		s.lm.exit()
	case "cancel":
		label += " " + e.ID
		r := s.reqs[e.ID]
		if _, _, _, queued := s.lm.turns.position(e.ID); queued {
			must = e.ID
		}
		r.cancel()
	case "setcap":
		label += fmt.Sprintf(" %d", e.N)
		s.lm.turns.setCap(e.N)
	case "halt":
		s.haltMu.Lock()
		s.halt = &haltInfo{reason: e.Reason, trigger: "admin"}
		s.haltMu.Unlock()
	case "resume":
		s.haltMu.Lock()
		s.halt = nil
		s.haltMu.Unlock()
	default:
		s.t.Fatalf("census: unknown op %q", e.Op)
	}
	s.settle(must)

	st := schedStep{T: e.T, Event: label}
	for _, id := range s.order { // arrival order, which is queue order: a halt's cascade refuses in it
		r := s.reqs[id]
		if before[id] {
			continue
		}
		select {
		case <-r.done:
		default:
			continue
		}
		v := e.T
		if r.ok {
			st.Granted = append(st.Granted, id)
			s.grants++
			r.out.Outcome, r.out.GrantedAt, r.out.GrantOrder = "granted", &v, s.grants
			w := v - r.out.Arrived
			r.out.Wait = &w
			if r.out.Deadline > 0 {
				past := v > r.out.Deadline
				r.out.PastDeadline = &past
			}
			if len(r.ev.Prompt) > 0 {
				if s.prev != nil {
					n := commonPrefix(s.prev, r.ev.Prompt)
					r.out.SharedPrev = &n
				}
				s.prev = r.ev.Prompt
			}
			s.run = append(s.run, id)
			continue
		}
		switch {
		case r.why.state == jobFailed:
			r.out.Outcome = "queue_full"
		case r.ctx.Err() != nil && r.why.reason == "cancelled before a turn was granted":
			r.out.Outcome = "cancelled"
		default:
			r.out.Outcome = "halted"
		}
		r.out.RefusedAt = &v
		st.Refused = append(st.Refused, schedRefusal{ID: id, State: string(r.why.state), Status: r.why.status, Reason: r.why.reason})
	}
	if len(st.Granted) > 1 {
		s.t.Fatalf("census: %d grants in one event (%s); their order is not observable here", len(st.Granted), label)
	}
	st.Running = append([]string{}, s.run...)
	st.Queue = []schedWaiter{}
	for _, id := range s.order {
		if place, _, _, ok := s.lm.turns.position(id); ok {
			st.Queue = append(st.Queue, schedWaiter{ID: id, Place: place})
		}
	}
	sort.Slice(st.Queue, func(i, j int) bool { return st.Queue[i].Place < st.Queue[j].Place })
	_, st.Waiting, st.Occupied, _ = s.lm.turns.position("\x00census")
	st.Load = s.lm.turns.load()
	if s.lm.queue != nil {
		st.QueueLen = len(s.lm.queue)
	}
	return st
}

func runSchedTrace(t *testing.T, tr schedTrace, m *decoder.Model) schedRecord {
	s := newSchedSim(t, tr, m)
	rec := schedRecord{Trace: tr.Name, About: tr.About}
	last := -1
	for _, e := range tr.Events {
		if e.T < last {
			t.Fatalf("census: trace %s goes back in time at t=%d", tr.Name, e.T)
		}
		last = e.T
		rec.Steps = append(rec.Steps, s.step(e))
	}
	for _, id := range s.order {
		r := s.reqs[id]
		if r.out.Outcome == "" {
			t.Fatalf("census: trace %s ends with %s still queued", tr.Name, id)
		}
		if r.ok && r.out.Finished == nil {
			t.Fatalf("census: trace %s ends with %s still running", tr.Name, id)
		}
		r.cancel()
		rec.Requests = append(rec.Requests, r.out)
	}
	return rec
}

// TestSchedCensus_admission replays each trace under testdata/sched/admission_traces/ through the real admission and compares
// the record with testdata/sched/admission/<trace>.json.
func TestSchedCensus_admission(t *testing.T) {
	traces, err := filepath.Glob(filepath.Join("testdata", "sched", "admission_traces", "*.json"))
	if err != nil || len(traces) == 0 {
		t.Fatalf("no traces: %v", err)
	}
	m, err := decoder.Load(filepath.Join("..", "..", "testdata", "llama-attnfa-tiny"), decoder.Options{Backend: "cpu"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	for _, p := range traces {
		name := strings.TrimSuffix(filepath.Base(p), ".json")
		t.Run(name, func(t *testing.T) {
			var tr schedTrace
			readSchedJSON(t, p, &tr)
			checkSchedGolden(t, filepath.Join("testdata", "sched", "admission", name+".json"), runSchedTrace(t, tr, m))
		})
	}
}

func readSchedJSON(t *testing.T, p string, v any) {
	t.Helper()
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		t.Fatalf("%s: %v", p, err)
	}
}

// checkSchedGolden compares got, as indented JSON, with the golden at p byte for byte, or writes it under -sched.update.
func checkSchedGolden(t *testing.T, p string, got any) {
	t.Helper()
	b, err := json.MarshalIndent(got, "", " ")
	if err != nil {
		t.Fatal(err)
	}
	b = append(b, '\n')
	if *updateSched {
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, b, 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	want, err := os.ReadFile(p)
	if err != nil {
		t.Fatalf("no golden (record with -sched.update): %v", err)
	}
	if !bytes.Equal(want, b) {
		t.Errorf("%s differs from today's scheduling record:\n%s", p, firstDiff(string(want), string(b)))
	}
}

// firstDiff names the first line where want and got part, with a few lines of context.
func firstDiff(want, got string) string {
	w, g := strings.Split(want, "\n"), strings.Split(got, "\n")
	for i := 0; i < max(len(w), len(g)); i++ {
		var wl, gl string
		if i < len(w) {
			wl = w[i]
		}
		if i < len(g) {
			gl = g[i]
		}
		if wl != gl {
			lo := max(0, i-3)
			return fmt.Sprintf("line %d:\n context: %s\n want:    %s\n got:     %s", i+1, strings.Join(g[lo:i], "\n          "), wl, gl)
		}
	}
	return "(identical lines; length differs)"
}

// placementTrace is one session-LRU trace: the LRU's size, sharing and tiering, and conversations' requests in order.
type placementTrace struct {
	Name       string           `json:"name"`
	About      string           `json:"about"`
	Size       int              `json:"size"`
	ShareOff   bool             `json:"share_off,omitempty"`
	DemoteMS   int              `json:"demote_after_ms,omitempty"` // tiering on when > 0
	DemotedMax int              `json:"demoted_max,omitempty"`
	Reply      int              `json:"reply"` // tokens each generation commits
	Events     []placementEvent `json:"events"`
}

// placementEvent: acquire hands request ID a session for a prompt built from its conversation (new: preamble tokens of
// the shared preamble, then n tokens of its own; continue: the last prompt, its reply, then n new tokens; edit: the
// last prompt less drop tokens, then n new; rollback: turn's prompt (1-based), then n new; repeat: the last prompt).
// generate runs the request's generation; checkin returns its session; turn is all three; tick runs the idle demotion
// at virtual time t.
type placementEvent struct {
	T        int    `json:"t,omitempty"`
	Op       string `json:"op"`
	ID       string `json:"id,omitempty"`
	Conv     string `json:"conv,omitempty"`
	Build    string `json:"build,omitempty"`
	Preamble int    `json:"preamble,omitempty"`
	N        int    `json:"n,omitempty"`
	Drop     int    `json:"drop,omitempty"`
	Turn     int    `json:"turn,omitempty"`
}

type placementSlot struct {
	Session string `json:"session"`
	Holds   string `json:"holds"` // the conversation whose tokens it holds, and how many
	Busy    bool   `json:"busy,omitempty"`
	First   int    `json:"first,omitempty"`
}

type placementStep struct {
	T         int             `json:"t,omitempty"`
	Event     string          `json:"event"`
	PromptLen int             `json:"prompt_len,omitempty"`
	Session   string          `json:"session,omitempty"`
	Action    string          `json:"action,omitempty"` // reuse | new | evict | in_place | fault_back | transient
	HadTokens *int            `json:"had_tokens,omitempty"`
	Common    *int            `json:"common_prefix,omitempty"` // its tokens' common prefix with the prompt, before acquire
	Holds     *int            `json:"holds_after_acquire,omitempty"`
	Reused    *int            `json:"prefill_reused,omitempty"`
	Shared    *int            `json:"prefill_shared,omitempty"`
	Demoted   *int            `json:"demoted,omitempty"`
	Order     []placementSlot `json:"order"`
	Cold      []string        `json:"cold,omitempty"`
}

type placementRecord struct {
	Trace string          `json:"trace"`
	About string          `json:"about"`
	Steps []placementStep `json:"steps"`
}

type placementConv struct {
	idx     int
	prompts [][]int // each turn's prompt
	history []int   // the last prompt and its reply, as a client resends them
}

// placementSim drives a real sessionLRU. Each conversation's reply is forced to one token of its own by LogitBias, so
// the record depends on no logit and is the same on every machine.
type placementSim struct {
	t      *testing.T
	tr     placementTrace
	l      *sessionLRU
	now    time.Time
	convs  map[string]*placementConv
	nconv  int
	labels map[*decoder.Session]string
	nlabel int
	held   map[string]*decoder.Session
	prompt map[string][]int
	conv   map[string]string
}

func placementTok(conv, k int) int { return 3 + (conv*97+k*31+7)%250 }

func placementPreamble(n int) []int {
	p := make([]int, n)
	for i := range p {
		p[i] = 3 + (i*37+11)%250
	}
	return p
}

func (s *placementSim) label(x *decoder.Session) string {
	if v, ok := s.labels[x]; ok {
		return v
	}
	s.nlabel++
	v := fmt.Sprintf("S%d", s.nlabel)
	s.labels[x] = v
	return v
}

// holds names the conversation whose history shares the most with toks, and how many tokens toks has.
func (s *placementSim) holds(toks []int) string {
	if len(toks) == 0 {
		return "empty"
	}
	best, name := 0, "?"
	names := make([]string, 0, len(s.convs))
	for n := range s.convs {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		c := s.convs[n]
		for _, p := range append(slices.Clone(c.prompts), c.history) {
			if m := commonPrefix(p, toks); m > best {
				best, name = m, n
			}
		}
	}
	return fmt.Sprintf("%s:%d", name, len(toks))
}

func (s *placementSim) order() []placementSlot {
	out := []placementSlot{}
	for _, x := range s.l.order {
		out = append(out, placementSlot{Session: s.label(x), Holds: s.holds(x.Tokens()), Busy: s.l.busy[x], First: s.l.first[x]})
	}
	return out
}

func (s *placementSim) cold() []string {
	var out []string
	for _, c := range s.l.cold {
		out = append(out, s.holds(c.tokens))
	}
	return out
}

func (s *placementSim) build(e placementEvent) []int {
	c := s.convs[e.Conv]
	if c == nil {
		c = &placementConv{idx: s.nconv}
		s.nconv++
		s.convs[e.Conv] = c
	}
	var p []int
	switch e.Build {
	case "new":
		p = placementPreamble(e.Preamble)
	case "continue":
		p = slices.Clone(c.history)
	case "edit":
		last := c.prompts[len(c.prompts)-1]
		p = slices.Clone(last[:len(last)-e.Drop])
	case "rollback":
		p = slices.Clone(c.prompts[e.Turn-1])
	case "repeat":
		return slices.Clone(c.prompts[len(c.prompts)-1])
	default:
		s.t.Fatalf("census: unknown build %q", e.Build)
	}
	for k := range e.N {
		p = append(p, placementTok(c.idx, len(p)+k))
	}
	return p
}

func (s *placementSim) acquire(e placementEvent) placementStep {
	prompt := s.build(e)
	st := placementStep{T: e.T, Event: "acquire " + e.ID + " (" + e.Conv + " " + e.Build + ")", PromptLen: len(prompt)}
	before := map[*decoder.Session][]int{}
	for _, x := range s.l.order {
		before[x] = slices.Clone(x.Tokens())
	}
	coldBefore := map[string][]int{}
	for _, c := range s.l.cold {
		coldBefore[c.path] = c.tokens
	}
	x := s.l.acquire(prompt)
	st.Session = s.label(x)
	had, inLRU := before[x]
	switch {
	case !slices.Contains(s.l.order, x):
		st.Action = "transient"
	case !inLRU && s.faulted(coldBefore, x):
		st.Action = "fault_back"
	case !inLRU:
		st.Action = "new"
	case len(x.Tokens()) == len(had):
		st.Action = "reuse"
	case len(x.Tokens()) == 0:
		st.Action = "evict"
	default:
		st.Action = "in_place"
	}
	if inLRU {
		n, m := len(had), commonPrefix(had, prompt)
		st.HadTokens, st.Common = &n, &m
	}
	h := len(x.Tokens())
	st.Holds = &h
	s.held[e.ID], s.prompt[e.ID], s.conv[e.ID] = x, prompt, e.Conv
	c := s.convs[e.Conv]
	c.prompts = append(c.prompts, prompt)
	st.Order, st.Cold = s.order(), s.cold()
	return st
}

// faulted reports whether x holds the tokens of a cold session listed in before that has left the cold tier: a
// fault-back. A fresh session seeded by copy holds a resident donor's rows instead.
func (s *placementSim) faulted(before map[string][]int, x *decoder.Session) bool {
	now := map[string]bool{}
	for _, c := range s.l.cold {
		now[c.path] = true
	}
	for p, toks := range before {
		if !now[p] && len(toks) > 0 && slices.Equal(toks, x.Tokens()) {
			return true
		}
	}
	return false
}

func (s *placementSim) generate(e placementEvent) placementStep {
	x, prompt, c := s.held[e.ID], s.prompt[e.ID], s.convs[s.conv[e.ID]]
	tok := placementTok(c.idx, 1000)
	ch, g := x.Generate(context.Background(), prompt, s.tr.Reply, decoder.SamplingParams{LogitBias: map[int]float32{tok: 1e6}})
	var reply []int
	for id := range ch {
		reply = append(reply, id)
	}
	if g.Err() != nil {
		s.t.Fatalf("census: generate %s: %v", e.ID, g.Err())
	}
	if len(reply) != s.tr.Reply || slices.ContainsFunc(reply, func(id int) bool { return id != tok }) {
		s.t.Fatalf("census: %s's reply %v is not %d of token %d", e.ID, reply, s.tr.Reply, tok)
	}
	c.history = append(slices.Clone(prompt), reply...)
	r, sh := g.PrefillReused, g.PrefillShared
	return placementStep{T: e.T, Event: "generate " + e.ID, Session: s.label(x), Reused: &r, Shared: &sh, Order: s.order(), Cold: s.cold()}
}

func (s *placementSim) checkin(e placementEvent) placementStep {
	s.l.checkin(s.held[e.ID])
	delete(s.held, e.ID)
	return placementStep{T: e.T, Event: "checkin " + e.ID, Order: s.order(), Cold: s.cold()}
}

func runPlacementTrace(t *testing.T, m *decoder.Model, tr placementTrace) placementRecord {
	s := &placementSim{t: t, tr: tr, convs: map[string]*placementConv{}, labels: map[*decoder.Session]string{},
		held: map[string]*decoder.Session{}, prompt: map[string][]int{}, conv: map[string]string{},
		now: time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)}
	s.l = newSessionLRU(m, tr.Size, 0, "census")
	start := s.now
	s.l.now = func() time.Time { return s.now }
	if tr.DemoteMS > 0 {
		s.l.enableTiering(t.TempDir(), time.Duration(tr.DemoteMS)*time.Millisecond, tr.DemotedMax)
		defer s.l.removeColdFiles()
	}
	rec := placementRecord{Trace: tr.Name, About: tr.About}
	for _, e := range tr.Events {
		if at := start.Add(time.Duration(e.T) * time.Millisecond); at.Before(s.now) {
			t.Fatalf("census: trace %s goes back in time at t=%d", tr.Name, e.T)
		} else {
			s.now = at
		}
		switch e.Op {
		case "acquire":
			rec.Steps = append(rec.Steps, s.acquire(e))
		case "generate":
			rec.Steps = append(rec.Steps, s.generate(e))
		case "checkin":
			rec.Steps = append(rec.Steps, s.checkin(e))
		case "turn":
			rec.Steps = append(rec.Steps, s.acquire(e), s.generate(e), s.checkin(e))
		case "tick":
			n := s.l.demoteIdle()
			rec.Steps = append(rec.Steps, placementStep{T: e.T, Event: "tick", Demoted: &n, Order: s.order(), Cold: s.cold()})
		default:
			t.Fatalf("census: unknown op %q", e.Op)
		}
	}
	return rec
}

// TestSchedCensus_placement replays each trace under testdata/sched/placement_traces/ through a real sessionLRU over
// testdata/llama-attnfa-tiny and compares the record with testdata/sched/placement/<trace>.json.
func TestSchedCensus_placement(t *testing.T) {
	dir := filepath.Join("..", "..", "testdata", "llama-attnfa-tiny")
	traces, err := filepath.Glob(filepath.Join("testdata", "sched", "placement_traces", "*.json"))
	if err != nil || len(traces) == 0 {
		t.Fatalf("no traces: %v", err)
	}
	models := map[bool]*decoder.Model{}
	for _, p := range traces {
		name := strings.TrimSuffix(filepath.Base(p), ".json")
		t.Run(name, func(t *testing.T) {
			var tr placementTrace
			readSchedJSON(t, p, &tr)
			m := models[tr.ShareOff]
			if m == nil {
				if m, err = decoder.Load(dir, decoder.Options{Backend: "cpu", KVShareOff: tr.ShareOff}); err != nil {
					t.Fatal(err)
				}
				models[tr.ShareOff] = m
			}
			checkSchedGolden(t, filepath.Join("testdata", "sched", "placement", name+".json"), runPlacementTrace(t, m, tr))
		})
	}
	for _, m := range models {
		m.Close()
	}
}
