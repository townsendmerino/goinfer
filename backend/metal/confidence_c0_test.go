//go:build darwin && goinfer_testhooks

package metal

// C0 of docs/tasks/task-constrained-confidence.md: does a per-field number on constrained output mean anything?
// This is the measurement harness only; the gates, the aggregations and the decision rule are pre-registered in the
// task doc's C0 section, and the analysis (docs/measurements/confidence-c0-2026-09-27/analyze.py) reads what this
// writes. It collects, for every generated token of a schema-constrained answer:
//   - p: the token's probability under the model's distribution restricted to the grammar-legal tokens (T = 1);
//   - forced: whether the grammar forced it byte for byte (every byte had at most one legal non-whitespace
//     alternative) — a forced token's probability is tokenization preference, not the model's view of the value;
//   - field: which schema field's value it overlaps (scaffolding otherwise);
//   - readout_ns: the in-situ cost of the readout a C1 hook would add (log-sum-exp over the legal logits).
// For enum and boolean fields it also records the option-level distribution at the deciding position (the first
// token of the value that is consistent with exactly one option), summing the mass of every legal token by the option
// it spells, so a BPE split of a literal does not smear the answer's probability across tokens.
// A second, mask-only pass per item gives the per-token decode time the readout is priced against.
//
//	GOINFER_C0_MODEL=<.gguf|.giw> [GOINFER_C0_TOKENIZER=<.gguf>] GOINFER_C0_OUT=<file.jsonl> \
//	  go test -tags goinfer_testhooks -run TestConfidenceC0 -v -timeout 60m ./metal/

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/constrain"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/confidence"
	"github.com/townsendmerino/goinfer/tokenizer"
)

const c0Schema = `{"type":"object","properties":{` +
	`"category":{"enum":["billing","technical","account","shipping","other"]},` +
	`"urgent":{"type":"boolean"},` +
	`"order_count":{"type":"integer"},` +
	`"refund_amount":{"type":"number"},` +
	`"customer_name":{"type":"string"},` +
	`"summary":{"type":"string"}},` +
	`"required":["category","urgent","order_count","refund_amount","customer_name","summary"],` +
	`"additionalProperties":false}`

const c0System = `You triage customer support tickets. Read the ticket and return a JSON object with these fields:
- category: the root cause of the problem. "billing" = charges, invoices, refunds of charges, payment methods. "technical" = the product, app or website malfunctioning. "account" = login credentials, profile, subscription settings, account closure. "shipping" = delivery, tracking, returns, wrong or damaged items, wrong address. "other" = anything else, including questions about pricing or partnerships.
- urgent: true only if the customer (a) states a deadline within the next 48 hours, (b) reports a physical safety hazard, or (c) says they are completely unable to work or use the service right now. Otherwise false.
- order_count: the number of distinct order numbers (written like #12345) in the ticket.
- refund_amount: the total number of dollars the customer explicitly asks to have refunded; 0 if they do not ask for a refund.
- customer_name: the name the customer signs with; "unknown" if the ticket is not signed.
- summary: one short sentence.`

var c0Options = map[string][]string{
	"category": {"billing", "technical", "account", "shipping", "other"},
	"urgent":   {"true", "false"},
}

type c0Ticket struct {
	ID   int             `json:"id"`
	Text string          `json:"text"`
	Gold json.RawMessage `json:"gold"`
}

type c0Token struct {
	ID        int     `json:"id"`
	Text      string  `json:"text"`
	P         float64 `json:"p"`
	Legal     int     `json:"legal"`
	Forced    bool    `json:"forced"`
	Field     string  `json:"field"`
	ReadoutNs int64   `json:"readout_ns"`
}

type c0Decision struct {
	Field     string             `json:"field"`
	Position  int                `json:"position"`
	Dist      map[string]float64 `json:"dist"`      // option → mass among decided tokens, renormalized
	Undecided float64            `json:"undecided"` // mass of legal tokens consistent with >1 option
}

type c0Record struct {
	Model     string          `json:"model"`
	Backend   string          `json:"backend"`
	TicketID  int             `json:"ticket_id"`
	Output    string          `json:"output"`
	Complete  bool            `json:"complete"`
	Gold      json.RawMessage `json:"gold"`
	Tokens    []c0Token       `json:"tokens"`
	Decisions []c0Decision    `json:"decisions"`
	TokenNs   []int64         `json:"token_ns"`     // mask-only pass: interval between consecutive processor calls
	SameIDs   bool            `json:"same_ids_a_b"` // the mask-only pass emitted the same ids
}

func TestConfidenceC0(t *testing.T) {
	modelPath := os.Getenv("GOINFER_C0_MODEL")
	outPath := os.Getenv("GOINFER_C0_OUT")
	if modelPath == "" || outPath == "" {
		t.Skip("C0 measurement: set GOINFER_C0_MODEL and GOINFER_C0_OUT (see the file comment)")
	}
	tokPath := os.Getenv("GOINFER_C0_TOKENIZER")
	if tokPath == "" {
		tokPath = modelPath
	}
	backend := os.Getenv("GOINFER_C0_BACKEND")
	if backend == "" {
		backend = "metal"
	}
	dataPath := os.Getenv("GOINFER_C0_DATA")
	if dataPath == "" {
		dataPath = filepath.Join("..", "docs", "measurements", "confidence-c0-2026-09-27", "tickets.jsonl")
	}
	const maxTok = 200
	if dl, ok := t.Deadline(); ok {
		t.Logf("deadline %s", dl.Format(time.TimeOnly))
	}

	raw, err := os.ReadFile(dataPath)
	if err != nil {
		t.Fatal(err)
	}
	var tickets []c0Ticket
	for _, line := range bytes.Split(bytes.TrimSpace(raw), []byte("\n")) {
		var tk c0Ticket
		if err := json.Unmarshal(line, &tk); err != nil {
			t.Fatalf("ticket: %v", err)
		}
		tickets = append(tickets, tk)
	}

	tk, err := tokenizer.LoadGGUF(tokPath)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	m, err := decoder.Load(modelPath, decoder.Options{Backend: backend, Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	fmt.Fprintf(os.Stderr, "C0: %s on %s (decode path %s), %d tickets\n", filepath.Base(modelPath), backend, m.DecodePath(), len(tickets))

	tmpl, err := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has})
	if err != nil {
		t.Fatalf("chat template: %v", err)
	}
	var stop []int
	sp := tk.Special()
	for _, id := range []int{sp.EOS, sp.EndOfTurn} {
		if id >= 0 {
			stop = append(stop, id)
		}
	}
	for _, s := range tmpl.Stops().Strings {
		if id, ok := tk.TokenID(s); ok && !slices.Contains(stop, id) {
			stop = append(stop, id)
		}
	}
	tokBytes := constrain.TokenBytes(m.Config().VocabSize, tk.TokenText)
	newGrammar := func() constrain.Grammar {
		g, err := constrain.JSONSchema([]byte(c0Schema))
		if err != nil {
			t.Fatalf("schema: %v", err)
		}
		return g
	}

	out, err := os.Create(outPath)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()
	enc := json.NewEncoder(out)

	start := time.Now()
	for ti, tc := range tickets {
		ids, err := tk.EncodeSegments(tmpl.RenderSegments(c0System, []chat.Turn{{Role: "user", Content: "Ticket:\n" + tc.Text}}), false)
		if err != nil {
			t.Fatalf("encode: %v", err)
		}
		// Pass B (the data): mask, then the readout under test, timed; the logits copy is the harness's, not the
		// readout's, and is taken after the clock stops.
		mkB := constrain.NewMasker(newGrammar(), tokBytes, stop).StopWhenComplete()
		var recs []c0Pos
		procB := func(gen []int, logits []float32) {
			mkB.Process(gen, logits)
			t0 := time.Now()
			lse, n := confidence.LogSumExpFinite(logits)
			dt := time.Since(t0).Nanoseconds()
			recs = append(recs, c0Pos{logits: slices.Clone(logits), lse: lse, legal: n, readoutNs: dt})
		}
		genB := collect(t, m, ids, maxTok, procB, stop)
		// Pass A (the price): the mask alone, timestamped at each call.
		mkA := constrain.NewMasker(newGrammar(), tokBytes, stop).StopWhenComplete()
		var stamps []time.Time
		procA := func(gen []int, logits []float32) {
			mkA.Process(gen, logits)
			stamps = append(stamps, time.Now())
		}
		genA := collect(t, m, ids, maxTok, procA, stop)

		rec := analyzeC0(t, tc, genB, recs, tokBytes, newGrammar)
		rec.Model, rec.Backend, rec.Gold = filepath.Base(modelPath), backend, tc.Gold
		rec.SameIDs = slices.Equal(genA, genB)
		for i := 1; i < len(stamps); i++ {
			rec.TokenNs = append(rec.TokenNs, stamps[i].Sub(stamps[i-1]).Nanoseconds())
		}
		if err := enc.Encode(rec); err != nil {
			t.Fatal(err)
		}
		el := time.Since(start)
		eta := time.Duration(float64(el) / float64(ti+1) * float64(len(tickets)-ti-1))
		fmt.Fprintf(os.Stderr, "%s C0 %d/%d ticket %d: %d tokens, complete=%v, same A/B=%v; elapsed %s, eta %s\n",
			time.Now().Format(time.TimeOnly), ti+1, len(tickets), tc.ID, len(rec.Tokens), rec.Complete, rec.SameIDs,
			el.Round(time.Second), eta.Round(time.Second))
	}
}

// c0Pos is one processor call of the data pass: the masked logits (the harness's copy), the readout's
// normalizer and legal count, and the readout's in-situ time.
type c0Pos struct {
	logits    []float32
	lse       float64
	legal     int
	readoutNs int64
}

// collect runs one greedy constrained generation and returns its ids.
func collect(t *testing.T, m *decoder.Model, prompt []int, maxTok int, proc func([]int, []float32), stop []int) []int {
	t.Helper()
	ch, g := m.Generate(context.Background(), prompt, maxTok, decoder.SamplingParams{LogitProcessor: proc, StopIDs: stop})
	var ids []int
	for id := range ch {
		ids = append(ids, id)
	}
	if err := g.Err(); err != nil {
		t.Fatalf("generate: %v", err)
	}
	return ids
}

// c0Span is one top-level field of the output object: where its colon ends and its value lies.
type c0Span struct {
	key                  string
	colonEnd, start, end int
}

// c0Fields scans a flat JSON object of scalar values and returns each field's value span. It accepts exactly what
// the c0Schema grammar produces; anything else stops the scan (an incomplete output keeps the fields before it).
func c0Fields(s string) []c0Span {
	var out []c0Span
	i := 0
	ws := func() {
		for i < len(s) && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r') {
			i++
		}
	}
	str := func() (int, bool) { // s[i] == '"'; returns the index just past the closing quote
		j := i + 1
		for j < len(s) {
			switch s[j] {
			case '\\':
				j += 2
				continue
			case '"':
				return j + 1, true
			}
			j++
		}
		return j, false
	}
	ws()
	if i >= len(s) || s[i] != '{' {
		return nil
	}
	i++
	for {
		ws()
		if i >= len(s) || s[i] != '"' {
			return out
		}
		kend, ok := str()
		if !ok {
			return out
		}
		key := s[i+1 : kend-1]
		i = kend
		ws()
		if i >= len(s) || s[i] != ':' {
			return out
		}
		i++
		colonEnd := i
		ws()
		vs := i
		if i < len(s) && s[i] == '"' {
			ve, ok := str()
			if !ok {
				out = append(out, c0Span{key, colonEnd, vs, len(s)})
				return out
			}
			i = ve
		} else {
			for i < len(s) && s[i] != ',' && s[i] != '}' && s[i] != ' ' && s[i] != '\n' && s[i] != '\t' && s[i] != '\r' {
				i++
			}
		}
		out = append(out, c0Span{key, colonEnd, vs, i})
		ws()
		if i >= len(s) || s[i] != ',' {
			return out
		}
		i++
	}
}

// c0Forced reports whether the grammar forces bs byte for byte from state g: at every byte, at most one legal
// non-whitespace byte exists. Whitespace alternatives are the grammar's optional structural whitespace, which says
// nothing about the value. g is advanced over bs.
func c0Forced(g constrain.Grammar, bs []byte) bool {
	forced := true
	var probe [1]byte
	for _, b := range bs {
		if forced {
			n := 0
			for x := 0; x < 256 && n < 2; x++ {
				if x == ' ' || x == '\t' || x == '\n' || x == '\r' {
					continue
				}
				probe[0] = byte(x)
				if g.TryBytes(probe[:]) {
					n++
				}
			}
			if n > 1 {
				forced = false
			}
		}
		g.Commit([]byte{b})
	}
	return forced
}

// c0Consistent lists the options of an enum/boolean field that the text r (the value's bytes so far, leading
// whitespace trimmed, possibly running past the value into a delimiter) is consistent with.
func c0Consistent(field, r string) []string {
	var out []string
	for _, o := range c0Options[field] {
		enc := o
		if field == "category" {
			enc = `"` + o + `"`
		}
		switch {
		case len(r) <= len(enc) && enc[:len(r)] == r:
			out = append(out, o)
		case len(r) > len(enc) && r[:len(enc)] == enc && bytes.ContainsRune([]byte(",} \t\n\r"), rune(r[len(enc)])):
			out = append(out, o)
		}
	}
	return out
}

func analyzeC0(t *testing.T, tc c0Ticket, ids []int, recs []c0Pos, tokBytes [][]byte, newGrammar func() constrain.Grammar) c0Record {
	t.Helper()
	rec := c0Record{TicketID: tc.ID}
	var buf []byte
	starts := make([]int, len(ids))
	for i, id := range ids {
		starts[i] = len(buf)
		if id >= 0 && id < len(tokBytes) {
			buf = append(buf, tokBytes[id]...)
		}
	}
	rec.Output = string(buf)
	rec.Complete = len(recs) > len(ids) // a processor call past the last emitted token is the EOS step
	if len(recs) < len(ids) {
		t.Fatalf("ticket %d: %d processor calls for %d tokens", tc.ID, len(recs), len(ids))
	}
	fields := c0Fields(rec.Output)
	fieldAt := func(s, e int) *c0Span {
		for k := range fields {
			if s < fields[k].end && e > fields[k].start {
				return &fields[k]
			}
		}
		return nil
	}
	g := newGrammar()
	decided := map[string]bool{}
	for i, id := range ids {
		b := tokBytes[id]
		p := math.Exp(float64(recs[i].logits[id]) - recs[i].lse)
		tok := c0Token{ID: id, Text: string(b), P: p, Legal: recs[i].legal, ReadoutNs: recs[i].readoutNs}
		tok.Forced = c0Forced(g, b)
		f := fieldAt(starts[i], starts[i]+len(b))
		if f != nil {
			tok.Field = f.key
		}
		rec.Tokens = append(rec.Tokens, tok)
		// The deciding position of an enum/boolean field: the first of its tokens consistent with one option only.
		if f == nil || c0Options[f.key] == nil || decided[f.key] {
			continue
		}
		trim := func(s string) string { return string(bytes.TrimLeft([]byte(s), " \t\n\r")) }
		if got := c0Consistent(f.key, trim(rec.Output[f.colonEnd:starts[i]+len(b)])); len(got) != 1 {
			continue
		}
		decided[f.key] = true
		d := c0Decision{Field: f.key, Position: i, Dist: map[string]float64{}}
		prefix := rec.Output[f.colonEnd:starts[i]]
		var total float64
		for alt, lg := range recs[i].logits {
			if math.IsInf(float64(lg), -1) || alt >= len(tokBytes) {
				continue
			}
			pa := math.Exp(float64(lg) - recs[i].lse)
			switch opts := c0Consistent(f.key, trim(prefix+string(tokBytes[alt]))); len(opts) {
			case 0:
			case 1:
				d.Dist[opts[0]] += pa
				total += pa
			default:
				d.Undecided += pa
			}
		}
		if total > 0 {
			keys := make([]string, 0, len(d.Dist))
			for k := range d.Dist {
				keys = append(keys, k)
			}
			sort.Strings(keys)
			for _, k := range keys {
				d.Dist[k] /= total
			}
		}
		rec.Decisions = append(rec.Decisions, d)
	}
	return rec
}

// TestC0Helpers pins the harness's pure pieces on hand-worked cases, so a scanner or classifier bug cannot pass
// silently into the measurement.
func TestC0Helpers(t *testing.T) {
	s := `{"category": "billing", "urgent":false,"order_count":2,"refund_amount":34.5,"customer_name":"Olu \"O\" A","summary":"x"}`
	fs := c0Fields(s)
	want := map[string]string{"category": `"billing"`, "urgent": "false", "order_count": "2", "refund_amount": "34.5",
		"customer_name": `"Olu \"O\" A"`, "summary": `"x"`}
	if len(fs) != len(want) {
		t.Fatalf("c0Fields found %d fields, want %d: %+v", len(fs), len(want), fs)
	}
	for _, f := range fs {
		if got := s[f.start:f.end]; got != want[f.key] {
			t.Errorf("field %s value %q, want %q", f.key, got, want[f.key])
		}
		if s[f.colonEnd-1] != ':' {
			t.Errorf("field %s colonEnd %d does not follow a colon", f.key, f.colonEnd)
		}
	}

	g, err := constrain.JSONSchema([]byte(c0Schema))
	if err != nil {
		t.Fatal(err)
	}
	// A key is a choice (the grammar accepts the properties in any order); it is scaffolding, never a field value.
	// The enum's first letter is a choice and the rest of "billing" is forced; likewise the boolean.
	for _, step := range []struct {
		bs     string
		forced bool
	}{{`{"category":`, false}, {` "`, true}, {`b`, false}, {`illing"`, true}, {`,"urgent":`, false}, {`f`, false}, {`alse`, true}} {
		if got := c0Forced(g, []byte(step.bs)); got != step.forced {
			t.Errorf("c0Forced(%q) = %v, want %v", step.bs, got, step.forced)
		}
	}

	for _, c := range []struct {
		field, r string
		want     []string
	}{
		{"category", `"`, []string{"billing", "technical", "account", "shipping", "other"}},
		{"category", `"bil`, []string{"billing"}},
		{"category", `"billing",`, []string{"billing"}},
		{"category", `"billingx`, nil},
		{"urgent", `t`, []string{"true"}},
		{"urgent", `false,"`, []string{"false"}},
		{"urgent", ``, []string{"true", "false"}},
	} {
		if got := c0Consistent(c.field, c.r); !slices.Equal(got, c.want) {
			t.Errorf("c0Consistent(%s, %q) = %v, want %v", c.field, c.r, got, c.want)
		}
	}
}
