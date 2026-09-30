// Package decide is D1 of docs/tasks/task-constrained-confidence.md: decisions by label-token scoring ("Route A")
// on any loaded model. A decision is a state, a question and a closed option set; the answer is a probability
// distribution over the options, from ONE prefill and no decode:
//
//   - the prompt is rendered in one of two templates: bare-v1, byte-identical to the one autotrust's JEV models use
//     (docs/measurements/decisions-d0-prior-art-2026-09-27.md; no chat template, no BOS), or chat-v1, the same content
//     as one user turn in the model's own chat template with the assistant turn opened (and thinking switched off
//     where the template has it) — what an instruct model understands;
//   - each option has a bare verbalizer token (noul: false/true; score: 0..5; choice: A..P), which must be exactly
//     one token in the model's tokenizer;
//   - the model's logits at the last prompt position are read at those tokens, divided by a per-kind temperature
//     (1 unless a calibration.json says otherwise), and softmaxed over the active ones only.
//
// A distribution from here is the model's probability over the options it was shown. It is calibrated only when a
// fitted temperature for its kind was loaded (Result.Calibrated), and even then only in the sense the fit measured.
package decide

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/townsendmerino/goinfer/internal/confidence"
)

// Kinds, as autotrust's judge_config.json names them.
const (
	KindNoul   = "noul"   // yes/no; options must be ["false", "true"]
	KindScore  = "score"  // a 0..5 level; options must be ["0", "1", "2", "3", "4", "5"]
	KindChoice = "choice" // 2..16 free-text options, labelled A..P
)

// The templates. A calibration or a reference made under one is not interchangeable with the other.
const (
	TemplateBare = "bare-v1" // JEV's own; a trained decision head (Route B) reads this
	TemplateChat = "chat-v1" // the model's chat template, for instruct models under Route A
)

// MaxChoiceOptions is the choice kind's limit: the sixteen letter verbalizers A..P.
const MaxChoiceOptions = 16

// MaxScoreLevels is the score kind's limit: the ten digit verbalizers 0..9. JEV's own score is six levels (0..5);
// a Route A score may use any 2..10, which TypeSafe's API allows.
const MaxScoreLevels = 10

var (
	noulOptions  = []string{"false", "true"}
	scoreOptions = []string{"0", "1", "2", "3", "4", "5"} // JEV's six levels
	letters      = "ABCDEFGHIJKLMNOP"
	digits       = "0123456789"
)

// ScoreOptions returns the options of an n-level score: "0" .. "n-1".
func ScoreOptions(n int) []string {
	out := make([]string, n)
	for i := range out {
		out[i] = digits[i : i+1]
	}
	return out
}

// Request is one decision.
type Request struct {
	Kind     string
	State    string
	Question string
	Options  []string
	// Descriptions, when set, are aligned with Options and rendered beside them ("A) billing: payments, invoicing"):
	// what each option or level means. "" leaves an option bare. bare-v1 is byte-identical to JEV's template only
	// without descriptions.
	Descriptions []string
	// Permute > 1 (choice only) averages the distribution over that many option orders — cyclic rotations, so with
	// Permute == len(Options) every option sits at every letter once. It costs one prefill per order and exists
	// because label-order bias under shuffled options is a known failure of label scoring.
	Permute int
}

// Result is a decision's answer. Distribution is aligned with Request.Options.
type Result struct {
	Distribution  []float64
	Decision      string   // the most probable option
	Index         int      // its index in Options
	Confidence    float64  // the most probable option's probability (not TypeSafe's unpublished formula)
	ExpectedScore *float64 // score kind only: Σ level · p(level)
	Calibrated    bool     // a fitted temperature for this kind was applied
	Temperature   float64
	Template      string
	Route         string // RouteLabel or RouteHead
	PromptTokens  int
	Prefills      int
	Latency       time.Duration
}

// Validate checks a request against its kind's rules: noul's fixed options, a score's "0".."n-1" for 2..10 levels,
// 2..16 distinct non-empty options for choice, and descriptions (if any) aligned with the options.
func Validate(r Request) error {
	if r.Descriptions != nil && len(r.Descriptions) != len(r.Options) {
		return fmt.Errorf("decide: %d descriptions for %d options", len(r.Descriptions), len(r.Options))
	}
	switch r.Kind {
	case KindNoul:
		if !equal(r.Options, noulOptions) {
			return fmt.Errorf("decide: noul options must be %q, got %q", noulOptions, r.Options)
		}
	case KindScore:
		if n := len(r.Options); n < 2 || n > MaxScoreLevels || !equal(r.Options, ScoreOptions(n)) {
			return fmt.Errorf("decide: score options must be \"0\"..\"n-1\" for 2..%d levels, got %q", MaxScoreLevels, r.Options)
		}
	case KindChoice:
		if n := len(r.Options); n < 2 || n > MaxChoiceOptions {
			return fmt.Errorf("decide: choice takes 2..%d options, got %d", MaxChoiceOptions, n)
		}
		seen := map[string]bool{}
		for _, o := range r.Options {
			if strings.TrimSpace(o) == "" {
				return errors.New("decide: an empty choice option")
			}
			if seen[o] {
				return fmt.Errorf("decide: duplicate choice option %q", o)
			}
			seen[o] = true
		}
	default:
		return fmt.Errorf("decide: unknown kind %q (want noul, choice or score)", r.Kind)
	}
	if r.Permute > 1 && r.Kind != KindChoice {
		return fmt.Errorf("decide: permute applies to choice only (%s options are fixed)", r.Kind)
	}
	return nil
}

// optionLines renders the options as the templates list them: "A) option" for choice, the bare label otherwise, and
// ": description" after any option that has one.
func optionLines(kind string, options, descriptions []string) []string {
	lines := make([]string, len(options))
	for i, o := range options {
		lines[i] = o
		if kind == KindChoice {
			lines[i] = fmt.Sprintf("%c) %s", letters[i], o)
		}
		if i < len(descriptions) && descriptions[i] != "" {
			lines[i] += ": " + descriptions[i]
		}
	}
	return lines
}

// Render returns the bare-v1 prompt for a request whose options are listed in the given order. descriptions is
// optional (see Request.Descriptions).
func Render(kind, state, question string, options []string, descriptions ...string) string {
	return "[kind] " + kind + "\n[state] " + state + "\n[question] " + question + "\n[options]\n" +
		strings.Join(optionLines(kind, options, descriptions), "\n") + "\n[decision]:"
}

// RenderChat returns chat-v1's user message: the same state, question and options as bare-v1, and an instruction
// to answer with the label alone, so the assistant's first token is the verbalizer the readout scores.
func RenderChat(kind, state, question string, options []string, descriptions ...string) string {
	instr := "Answer with exactly one word: true or false."
	switch kind {
	case KindScore:
		instr = fmt.Sprintf("Answer with exactly one digit from 0 to %d.", len(options)-1)
	case KindChoice:
		instr = "Answer with exactly one letter: the letter of the best option."
	}
	return "State:\n" + state + "\n\nQuestion: " + question + "\n\nOptions:\n" +
		strings.Join(optionLines(kind, options, descriptions), "\n") + "\n\n" + instr
}

// verbalizers returns the label strings a kind reads, one per option position.
func verbalizers(kind string, n int) []string {
	switch kind {
	case KindNoul:
		return noulOptions
	case KindScore:
		return ScoreOptions(n)
	}
	out := make([]string, n)
	for i := range out {
		out[i] = letters[i : i+1]
	}
	return out
}

// Tokenizer is what the decider needs from a tokenizer. EncodePlain encodes text as one plain segment, so
// special-token text inside a state stays text (bare-v1 prompts and the labels). EncodeChat renders a user message in
// the model's chat template with the assistant turn opened and thinking off, keeping the template's markers special
// and the message plain (chat-v1).
type Tokenizer interface {
	EncodePlain(text string) ([]int, error)
	EncodeChat(user string) ([]int, error)
}

// Prefill runs one prompt and returns the model's logits at its last position.
type Prefill func(ctx context.Context, ids []int) ([]float32, error)

// Options configure a Decider.
type Options struct {
	Template    string       // TemplateBare (default) or TemplateChat
	Calibration *Calibration // per-kind temperatures; nil ⇒ T = 1, uncalibrated (with a Head: the head's own)
	// Head, when set, makes this a Route B decider (D4): the trained head over Hidden's final-norm hidden state
	// replaces the label logits, and the prefill is not used. It requires bare-v1 and a Hidden.
	Head   *Head
	Hidden Hidden
}

// Decider scores decisions on one model.
type Decider struct {
	tok      Tokenizer
	prefill  Prefill
	label    map[string]int // verbalizer → its single token id (Route A only)
	cal      *Calibration
	template string
	head     *Head
	hidden   Hidden
}

// New resolves every verbalizer to a single token and returns a Decider. It fails, naming the offenders, when any
// verbalizer is not exactly one token in this tokenizer: label scoring cannot read a label that spans tokens. It also
// refuses a calibration fitted under another template.
func New(tok Tokenizer, prefill Prefill, opts Options) (*Decider, error) {
	tmpl := opts.Template
	if tmpl == "" {
		tmpl = TemplateBare
	}
	if tmpl != TemplateBare && tmpl != TemplateChat {
		return nil, fmt.Errorf("decide: unknown template %q (want %s or %s)", tmpl, TemplateBare, TemplateChat)
	}
	cal := opts.Calibration
	if cal != nil && cal.Template != "" && cal.Template != tmpl {
		return nil, fmt.Errorf("decide: the calibration was fitted under template %q, not %q", cal.Template, tmpl)
	}
	if opts.Head != nil {
		// Route B reads slots, not label tokens, so the verbalizers need not be single tokens here.
		if opts.Hidden == nil {
			return nil, errors.New("decide: a decision head needs the model's hidden state (Options.Hidden)")
		}
		if tmpl != opts.Head.Template {
			return nil, fmt.Errorf("decide: the decision head was trained under template %q, not %q", opts.Head.Template, tmpl)
		}
		if cal == nil {
			cal = opts.Head.Calibration
		}
		if cal != nil && cal.Template != "" && cal.Template != tmpl {
			return nil, fmt.Errorf("decide: the calibration was fitted under template %q, not %q", cal.Template, tmpl)
		}
		return &Decider{tok: tok, cal: cal, template: tmpl, head: opts.Head, hidden: opts.Hidden}, nil
	}
	d := &Decider{tok: tok, prefill: prefill, label: map[string]int{}, cal: cal, template: tmpl}
	var bad []string
	for _, v := range append(append(append([]string(nil), noulOptions...), strings.Split(digits, "")...), strings.Split(letters, "")...) {
		ids, err := tok.EncodePlain(v)
		if err != nil {
			return nil, err
		}
		if len(ids) != 1 {
			bad = append(bad, fmt.Sprintf("%q → %v", v, ids))
			continue
		}
		d.label[v] = ids[0]
	}
	if len(bad) > 0 {
		return nil, fmt.Errorf("decide: these labels are not single tokens in this tokenizer, so it cannot score decisions: %s", strings.Join(bad, ", "))
	}
	return d, nil
}

// Template returns the template this decider renders.
func (d *Decider) Template() string { return d.template }

// Route returns RouteHead for a decider with a trained head, RouteLabel otherwise.
func (d *Decider) Route() string {
	if d.head != nil {
		return RouteHead
	}
	return RouteLabel
}

// validate is Validate plus, for Route B, the head's own limits.
func (d *Decider) validate(r Request) error {
	if err := Validate(r); err != nil {
		return err
	}
	if d.head != nil {
		return d.head.check(r)
	}
	return nil
}

// LabelID returns a verbalizer's token id (for tests and the reference comparison).
func (d *Decider) LabelID(v string) (int, bool) { id, ok := d.label[v]; return id, ok }

// Decide answers one request.
func (d *Decider) Decide(ctx context.Context, r Request) (Result, error) {
	if err := d.validate(r); err != nil {
		return Result{}, err
	}
	t0 := time.Now()
	temp, calibrated := d.cal.Temperature(r.Kind)
	orders := 1
	if r.Permute > 1 {
		orders = min(r.Permute, len(r.Options))
	}
	n := len(r.Options)
	acc := make([]float64, n)
	var res Result
	for k := 0; k < orders; k++ {
		// Order k is the rotation by k·n/orders: position i shows option perm[i].
		perm := make([]int, n)
		shift := k * n / orders
		for i := range perm {
			perm[i] = (i + shift) % n
		}
		shown := make([]string, n)
		var descs []string
		if r.Descriptions != nil {
			descs = make([]string, n)
		}
		for i, j := range perm {
			shown[i] = r.Options[j]
			if descs != nil {
				descs[i] = r.Descriptions[j]
			}
		}
		logp, promptTokens, err := d.score(ctx, r.Kind, r.State, r.Question, shown, descs, temp)
		if err != nil {
			return Result{}, err
		}
		res.PromptTokens = promptTokens
		for i, j := range perm {
			acc[j] += math.Exp(logp[i]) / float64(orders)
		}
	}
	res.Distribution = acc
	res.Prefills = orders
	res.Temperature, res.Calibrated, res.Template = temp, calibrated, d.template
	res.Route = d.Route()
	for i, p := range acc {
		if p > acc[res.Index] {
			res.Index = i
		}
	}
	res.Decision, res.Confidence = r.Options[res.Index], acc[res.Index]
	if r.Kind == KindScore {
		var e float64
		for i, p := range acc {
			e += float64(i) * p
		}
		res.ExpectedScore = &e
	}
	res.Latency = time.Since(t0)
	return res, nil
}

// Scores returns a request's restricted log-probabilities at temperature 1, in the request's option order, with no
// permutation — the raw material decisions-calibrate fits a temperature on.
func (d *Decider) Scores(ctx context.Context, r Request) ([]float64, error) {
	if err := d.validate(r); err != nil {
		return nil, err
	}
	lp, _, err := d.score(ctx, r.Kind, r.State, r.Question, r.Options, r.Descriptions, 1)
	return lp, err
}

func (d *Decider) score(ctx context.Context, kind, state, question string, shown, descs []string, temp float64) ([]float64, int, error) {
	var ids []int
	var err error
	if d.head != nil { // the head was trained on jev_core's prompts, which cut a long state
		if state, _, err = truncateState(d.tok, state); err != nil {
			return nil, 0, err
		}
	}
	if d.template == TemplateChat {
		ids, err = d.tok.EncodeChat(RenderChat(kind, state, question, shown, descs...))
	} else {
		ids, err = d.tok.EncodePlain(Render(kind, state, question, shown, descs...))
	}
	if err != nil {
		return nil, 0, err
	}
	if len(ids) == 0 {
		return nil, 0, errors.New("decide: empty prompt")
	}
	if d.head != nil {
		h, err := d.hidden(ctx, ids)
		if err != nil {
			return nil, 0, err
		}
		z, err := d.head.logits(h, kind, len(shown))
		if err != nil {
			return nil, 0, err
		}
		slots := make([]int, len(z))
		for i := range slots {
			slots[i] = i
		}
		return confidence.RestrictedLogSoftmax(z, slots, temp), len(ids), nil
	}
	logits, err := d.prefill(ctx, ids)
	if err != nil {
		return nil, 0, err
	}
	vs := verbalizers(kind, len(shown))
	lids := make([]int, len(vs))
	for i, v := range vs {
		lids[i] = d.label[v]
	}
	return confidence.RestrictedLogSoftmax(logits, lids, temp), len(ids), nil
}

func equal(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
