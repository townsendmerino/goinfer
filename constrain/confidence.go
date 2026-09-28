package constrain

import (
	"encoding/json"
	"errors"
	"math"
	"strconv"
	"strings"
)

// Per-field confidence on constrained output (C1 of docs/tasks/task-constrained-confidence.md).
//
// WHAT THE NUMBER IS. A field's confidence is the model's probability over what the grammar allowed at the position(s)
// that decided the field's value — narrower than the probability that the value is right, and not a calibrated
// confidence unless a fitted temperature was supplied (FieldConfidence.Calibrated). C0 measured that it discriminates
// (a low-confidence value is wrong more often: AUROC 0.85 / 0.73 / 0.68 for enum / boolean / integer on the 1.5B,
// docs/measurements/confidence-c0-2026-09-27.md); it did not measure calibration.
//
// WHICH FIELDS. Only the kinds C0's gates cleared are reported: enum, boolean and integer. Number and string fields
// are omitted (C0 parked them for want of errors to judge them by), and so is any field with no free token — a
// value the grammar forced says nothing about the model.
//
// HOW. CaptureConfidence makes Process record, at each position outside a free string, the normalizer of the
// masked logits and the legal tokens' logits (every one when there are at most confCompleteMax, else the top
// confTopK). After generation FieldConfidence replays the tokens through a fresh copy of the grammar, byte by
// byte: which schema value each byte belongs to, and whether the grammar forced it (at most one legal
// non-whitespace next byte). Then, per field:
//   - enum / boolean: at the deciding token — the first whose value bytes are consistent with one option only — the
//     mass of every legal token, summed by the option it spells. A BPE split of a literal ("bill" + "ing") does not
//     smear the answer across positions. Confidence is the chosen option's share.
//   - integer: the minimum probability over the value's free tokens (a number is as weak as its weakest digit).
// Free strings are never read (nearly the whole vocabulary is legal inside one, which is where the readout is
// expensive, and no string field is reported).

// Readout limits: at a position with at most confCompleteMax legal tokens every one is kept (an enum/boolean deciding
// position's legal set is small); above it only the confTopK highest, which always holds a greedy pick.
const (
	confCompleteMax = 256
	confTopK        = 64
)

// ConfidenceOptions configures CaptureConfidence.
type ConfidenceOptions struct {
	// EnumTemperature and BooleanTemperature, when > 0, divide the logits at an enum / boolean field's deciding
	// position before its distribution is taken, and mark those fields Calibrated — a temperature fitted on labelled
	// data (the choice / noul kinds of a decisions calibration.json). Zero leaves them raw, and uncalibrated.
	EnumTemperature, BooleanTemperature float64
}

// FieldConfidence is one reported field of a constrained answer.
type FieldConfidence struct {
	Path       string          `json:"path"`  // e.g. "category", "items[2].status"; "" for a top-level scalar
	Kind       string          `json:"kind"`  // "enum", "boolean" or "integer"
	Value      json.RawMessage `json:"value"` // the value as generated (JSON)
	Confidence float64         `json:"confidence"`
	// Distribution (enum / boolean) is over the field's options at its deciding position, renormalized over the mass
	// that spells exactly one option. Undecided is the legal mass that did not (a token shared by several options'
	// prefixes), before that renormalization.
	Distribution map[string]float64 `json:"distribution,omitempty"`
	Undecided    float64            `json:"undecided,omitempty"`
	FreeTokens   int                `json:"free_tokens"`
	Calibrated   bool               `json:"calibrated"`
}

type confPos struct {
	read     bool
	lse      float64
	legal    int
	ids      []int32
	logits   []float32
	complete bool // ids/logits hold every legal token
}

type confCapture struct {
	opts ConfidenceOptions
	pos  []confPos
}

// CaptureConfidence makes the Masker record what FieldConfidence needs while Process masks. It is off unless called;
// off, Process pays one nil check. It needs a JSON Schema grammar (JSONSchema, GrammarFromStruct) for
// FieldConfidence to know each field's kind; the recording itself works on any grammar. Returns the Masker for
// chaining, like StopWhenComplete.
func (m *Masker) CaptureConfidence(opts ConfidenceOptions) *Masker {
	m.conf = &confCapture{opts: opts}
	return m
}

// record is Process's hook: position pos's masked logits, unless the grammar is inside a free string.
func (c *confCapture) record(pos int, logits []float32, plain bool) {
	var rec confPos
	if !plain {
		rec = readout(logits)
	}
	for len(c.pos) < pos {
		c.pos = append(c.pos, confPos{})
	}
	if pos < len(c.pos) {
		c.pos[pos] = rec
	} else {
		c.pos = append(c.pos, rec)
	}
}

// readout is the per-position readout: the log-normalizer over the legal (finite) logits, their count, and their
// entries — all of them when few, else the confTopK largest.
func readout(logits []float32) confPos {
	maxv, legal := math.Inf(-1), 0
	for _, v := range logits {
		if f := float64(v); !math.IsInf(f, 0) && !math.IsNaN(f) {
			legal++
			if f > maxv {
				maxv = f
			}
		}
	}
	rec := confPos{read: true, legal: legal, lse: math.Inf(-1), complete: legal <= confCompleteMax}
	if legal == 0 {
		return rec
	}
	var sum float64
	keep := min(legal, confTopK)
	if rec.complete {
		keep = legal
	}
	rec.ids, rec.logits = make([]int32, 0, keep), make([]float32, 0, keep)
	for id, v := range logits {
		f := float64(v)
		if math.IsInf(f, 0) || math.IsNaN(f) {
			continue
		}
		sum += math.Exp(f - maxv)
		switch {
		case rec.complete || len(rec.ids) < keep:
			rec.ids, rec.logits = append(rec.ids, int32(id)), append(rec.logits, v)
		default: // top-K: replace the smallest kept entry when this one is larger
			mi := 0
			for j := range rec.logits {
				if rec.logits[j] < rec.logits[mi] {
					mi = j
				}
			}
			if v > rec.logits[mi] {
				rec.ids[mi], rec.logits[mi] = int32(id), v
			}
		}
	}
	rec.lse = maxv + math.Log(sum)
	return rec
}

func (p *confPos) logitOf(id int) (float32, bool) {
	for j, x := range p.ids {
		if int(x) == id {
			return p.logits[j], true
		}
	}
	return 0, false
}

// confField accumulates one schema value during the replay.
type confField struct {
	path          string
	n             *node
	start, end    int // value bytes in the output
	boundary      int // just after the ':' / ',' / '[' that precedes the value's leading whitespace
	tokens        []int
	free          []bool
	firstTokStart []int // output offset where each of its tokens starts
}

// FieldConfidence reports the enum, boolean and integer fields of a generation this Masker constrained, from the ids
// it generated (the full list, including a final token Process may not have seen). It needs CaptureConfidence to
// have been on for the whole generation and a JSON Schema grammar. Fields with no free token, and number and
// string fields, are omitted (see the package comment above).
func (m *Masker) FieldConfidence(generated []int) ([]FieldConfidence, error) {
	if m.conf == nil {
		return nil, errors.New("constrain: FieldConfidence needs CaptureConfidence to have been on")
	}
	sg, ok := m.g.(*schemaGrammar)
	if !ok {
		return nil, errors.New("constrain: FieldConfidence needs a JSON Schema grammar (JSONSchema or GrammarFromStruct)")
	}
	g := sg.Clone().(*schemaGrammar)
	g.Reset()
	var out []byte
	var order []string
	fields := map[string]*confField{}
	lastStruct := 0 // offset just after the last ':' / ',' / '[' seen outside every scalar
	var probe [1]byte
	for i, id := range generated {
		if m.eosAt(id) {
			break
		}
		b := m.tokenBytes(id)
		start := len(out)
		forced := true
		var owner *confField
		for _, c := range b {
			if forced { // byte forcedness, before committing c
				n := 0
				for x := 0; x < 256 && n < 2; x++ {
					if isWS(byte(x)) {
						continue
					}
					probe[0] = byte(x)
					if g.TryBytes(probe[:]) {
						n++
					}
				}
				forced = n <= 1
			}
			before := len(g.stack)
			var beforePath string
			var beforeNode *node
			if top := scalarTop(g); top != nil && top.state == fsStr {
				beforePath, beforeNode = stackPath(g), top.n
			}
			g.Commit([]byte{c})
			out = append(out, c)
			pos := len(out) - 1
			var path string
			var nd *node
			switch top := scalarTop(g); {
			case top != nil && top.state != fsValue:
				path, nd = stackPath(g), top.n
			case beforeNode != nil && len(g.stack) < before: // this byte closed a string
				path, nd = beforePath, beforeNode
			}
			if nd == nil {
				if c == ':' || c == ',' || c == '[' {
					lastStruct = pos + 1
				}
				continue
			}
			f := fields[path]
			if f == nil {
				f = &confField{path: path, n: nd, start: pos, boundary: lastStruct}
				fields[path] = f
				order = append(order, path)
			}
			f.end = pos + 1
			if owner == nil {
				owner = f
			}
		}
		if owner != nil {
			owner.tokens = append(owner.tokens, i)
			owner.free = append(owner.free, !forced)
			owner.firstTokStart = append(owner.firstTokStart, start)
		}
	}

	var res []FieldConfidence
	for _, path := range order {
		f := fields[path]
		kind := fieldKind(f.n)
		if kind == "" {
			continue
		}
		nfree := 0
		for _, fr := range f.free {
			if fr {
				nfree++
			}
		}
		if nfree == 0 {
			continue
		}
		fc := FieldConfidence{Path: path, Kind: kind, Value: json.RawMessage(append([]byte(nil), out[f.start:f.end]...)), FreeTokens: nfree}
		switch kind {
		case "integer":
			minp, any := 1.0, false
			for j, ti := range f.tokens {
				if !f.free[j] || ti >= len(m.conf.pos) || !m.conf.pos[ti].read {
					continue
				}
				if lg, ok := m.conf.pos[ti].logitOf(generated[ti]); ok {
					minp, any = min(minp, math.Exp(float64(lg)-m.conf.pos[ti].lse)), true
				}
			}
			if !any {
				continue
			}
			fc.Confidence = minp
		default: // enum, boolean
			temp := m.conf.opts.EnumTemperature
			if kind == "boolean" {
				temp = m.conf.opts.BooleanTemperature
			}
			if !m.decide(&fc, f, out, generated, temp) {
				continue
			}
		}
		res = append(res, fc)
	}
	return res, nil
}

// decide fills an enum/boolean field's distribution from its deciding token. False if the replay found none (a
// value the capture never read, or one decided by a forced token).
func (m *Masker) decide(fc *FieldConfidence, f *confField, out []byte, generated []int, temp float64) bool {
	opts := f.n.enum
	valueText := func(tokStart int, tok []byte) (string, bool) {
		// The value's bytes if this token were emitted at tokStart: the output between the preceding structural byte
		// and the token, then the token, with the value's leading whitespace dropped.
		var s []byte
		if tokStart >= f.boundary {
			s = append(append(s, out[f.boundary:tokStart]...), tok...)
		} else {
			pre := out[tokStart:f.boundary]
			if !strings.HasPrefix(string(tok), string(pre)) {
				return "", false
			}
			s = append(s, tok[len(pre):]...)
		}
		return strings.TrimLeft(string(s), " \t\n\r"), true
	}
	consistent := func(r string) []int {
		var hit []int
		for k, e := range opts {
			enc := string(e)
			switch {
			case len(r) <= len(enc) && enc[:len(r)] == r:
				hit = append(hit, k)
			case len(r) > len(enc) && r[:len(enc)] == enc && strings.ContainsRune(",]} \t\n\r", rune(r[len(enc)])):
				hit = append(hit, k)
			}
		}
		return hit
	}
	for j, ti := range f.tokens {
		if !f.free[j] || ti >= len(m.conf.pos) || !m.conf.pos[ti].read {
			continue
		}
		r, ok := valueText(f.firstTokStart[j], m.tokenBytes(generated[ti]))
		if !ok || len(consistent(r)) != 1 {
			continue
		}
		p := &m.conf.pos[ti]
		lse, t := p.lse, 1.0
		if temp > 0 && p.complete {
			t = temp
			mx := math.Inf(-1)
			for _, v := range p.logits {
				mx = math.Max(mx, float64(v)/t)
			}
			var z float64
			for _, v := range p.logits {
				z += math.Exp(float64(v)/t - mx)
			}
			lse = mx + math.Log(z)
		}
		mass := make([]float64, len(opts))
		var decided float64
		for k, id := range p.ids {
			pa := math.Exp(float64(p.logits[k])/t - lse)
			alt, ok := valueText(f.firstTokStart[j], m.tokenBytes(int(id)))
			if !ok {
				fc.Undecided += pa
				continue
			}
			switch hit := consistent(alt); len(hit) {
			case 0:
			case 1:
				mass[hit[0]] += pa
				decided += pa
			default:
				fc.Undecided += pa
			}
		}
		if decided <= 0 {
			return false
		}
		chosen := -1
		for k, e := range opts {
			if string(e) == string(fc.Value) {
				chosen = k
			}
		}
		fc.Distribution = map[string]float64{}
		for k, e := range opts {
			fc.Distribution[optionKey(e)] = mass[k] / decided
		}
		if chosen >= 0 {
			fc.Confidence = mass[chosen] / decided
		}
		fc.Calibrated = t != 1
		return true
	}
	return false
}

// fieldKind names the reported kinds; "" for a kind that is not reported.
func fieldKind(n *node) string {
	switch n.kind {
	case kEnum:
		if len(n.enum) == 2 {
			a, b := string(n.enum[0]), string(n.enum[1])
			if (a == "true" && b == "false") || (a == "false" && b == "true") {
				return "boolean"
			}
		}
		for _, e := range n.enum {
			if string(e) == "null" {
				return ""
			}
		}
		return "enum"
	case kNumber:
		if n.intOnly {
			return "integer"
		}
	}
	return "" // number, string: parked by C0; object / array: not scalars
}

// optionKey is an option's key in a Distribution: a string literal's decoded text, any other literal as written.
func optionKey(enc []byte) string {
	var s string
	if len(enc) > 0 && enc[0] == '"' && json.Unmarshal(enc, &s) == nil {
		return s
	}
	return string(enc)
}

// scalarTop returns the top frame when it is a scalar value (enum, number, string), else nil.
func scalarTop(g *schemaGrammar) *frame {
	if len(g.stack) == 0 {
		return nil
	}
	f := &g.stack[len(g.stack)-1]
	switch f.n.kind {
	case kEnum, kNumber, kString:
		return f
	}
	return nil
}

// stackPath is the JSON path of the top frame: the selected property of every enclosing object and the current
// index of every enclosing array.
func stackPath(g *schemaGrammar) string {
	var b strings.Builder
	for i := 0; i < len(g.stack)-1; i++ {
		f := &g.stack[i]
		switch f.n.kind {
		case kObject:
			if f.sel >= 0 && f.sel < len(f.n.props) {
				if b.Len() > 0 {
					b.WriteByte('.')
				}
				b.WriteString(f.n.props[f.sel].name)
			}
		case kArray:
			b.WriteString("[" + strconv.Itoa(f.count-1) + "]")
		}
	}
	return b.String()
}
