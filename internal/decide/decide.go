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
	"slices"
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
	// DescriptionsDropped: the request had option descriptions and a decision head answered it, which reads the
	// options' names only (JEV's template has no descriptions).
	DescriptionsDropped bool
	PromptTokens        int
	Prefills            int
	Latency             time.Duration
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
	// HiddenMany and PrefillMany, when set, are the many-prompt forms of Hidden and of the prefill: one call for all of a request's questions, which lets the
	// model prefill what the prompts share once (DecideMany). nil keeps one call per question. A hook may be given only for a model that can answer them
	// faster than one call each (decoder.Model.CanSharePrefix); the Route B form is always safe, since it falls back to Hidden itself.
	HiddenMany  HiddenMany
	PrefillMany PrefillMany
}

// HiddenMany is Hidden for several prompts at once, in order.
type HiddenMany func(ctx context.Context, prompts [][]int) ([][]float32, error)

// PrefillMany is the prefill for several prompts at once, in order.
type PrefillMany func(ctx context.Context, prompts [][]int) ([][]float32, error)

// Decider scores decisions on one model.
type Decider struct {
	tok      Tokenizer
	prefill  Prefill
	label    map[string]int // verbalizer → its single token id (Route A only)
	cal      *Calibration
	template string
	head     *Head
	hidden   Hidden

	hiddenMany  HiddenMany
	prefillMany PrefillMany
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
		return &Decider{tok: tok, cal: cal, template: tmpl, head: opts.Head, hidden: opts.Hidden, hiddenMany: opts.HiddenMany}, nil
	}
	d := &Decider{tok: tok, prefill: prefill, label: map[string]int{}, cal: cal, template: tmpl, prefillMany: opts.PrefillMany}
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

// Validate is the package-level Validate plus, for a Route B decider, the head's own limits (its kinds, its slot
// counts, JEV's six score levels, no option descriptions). A server calls it before any prefill, so a request the head
// cannot answer is refused as the client's error rather than failing mid-request.
func (d *Decider) Validate(r Request) error { return d.validate(r) }

// validate is Validate's body.
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
	return d.decideWith(ctx, r, nil)
}

// preScored is a request's prompt ids and the model's output for them, computed ahead by DecideMany.
type preScored struct {
	ids []int
	out []float32
}

func (d *Decider) decideWith(ctx context.Context, r Request, pre *preScored) (Result, error) {
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
		var logp []float64
		var promptTokens int
		var err error
		if pre != nil && orders == 1 {
			logp, promptTokens, err = d.scoreIDs(ctx, pre.ids, r.Kind, shown, temp, pre.out)
		} else {
			logp, promptTokens, err = d.score(ctx, r.Kind, r.State, r.Question, shown, descs, temp)
		}
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
	res.DescriptionsDropped = d.head != nil && slices.ContainsFunc(r.Descriptions, func(s string) bool { return s != "" })
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

// DecideMany answers several requests about (usually) one state. Where the model can share work across them it does: every request's prompt is built first and
// handed to the many-prompt hook (Options.HiddenMany for a Route B head, Options.PrefillMany for label scoring) in ONE call, which prefills the prefix the prompts
// share once and resumes each question from a copy of that cache (decoder.PromptHiddenMany; D8 of docs/tasks/task-constrained-confidence.md). The answers are the
// ones Decide gives, up to the numerical difference of splitting one prefill in two (bounded by the decoder's own gate). Without the hook, or for a request that asks
// for several answer orders (Permute > 1), each request is Decide'd alone, exactly as before. Results are in the order of reqs.
func (d *Decider) DecideMany(ctx context.Context, reqs []Request) ([]Result, error) {
	out := make([]Result, len(reqs))
	many := (d.head != nil && d.hiddenMany != nil) || (d.head == nil && d.prefillMany != nil)
	if len(reqs) < 2 || !many {
		for i, r := range reqs {
			res, err := d.Decide(ctx, r)
			if err != nil {
				return nil, err
			}
			out[i] = res
		}
		return out, nil
	}
	for _, r := range reqs {
		if err := d.validate(r); err != nil {
			return nil, err
		}
	}
	var idx []int
	var prompts [][]int
	pre := make([]*preScored, len(reqs))
	for i, r := range reqs {
		if r.Permute > 1 {
			continue
		}
		var descs []string
		if r.Descriptions != nil {
			descs = r.Descriptions
		}
		ids, err := d.promptIDs(r.Kind, r.State, r.Question, r.Options, descs)
		if err != nil {
			return nil, err
		}
		pre[i] = &preScored{ids: ids}
		idx, prompts = append(idx, i), append(prompts, ids)
	}
	if len(prompts) >= 2 {
		var outs [][]float32
		var err error
		if d.head != nil {
			outs, err = d.hiddenMany(ctx, prompts)
		} else {
			outs, err = d.prefillMany(ctx, prompts)
		}
		if err != nil {
			return nil, err
		}
		if len(outs) != len(prompts) {
			return nil, fmt.Errorf("decide: the many-prompt hook returned %d outputs for %d prompts", len(outs), len(prompts))
		}
		for k, i := range idx {
			pre[i].out = outs[k]
		}
	} else {
		for _, i := range idx {
			pre[i] = nil // a single prompt shares nothing: the ordinary path
		}
	}
	for i, r := range reqs {
		res, err := d.decideWith(ctx, r, pre[i])
		if err != nil {
			return nil, err
		}
		out[i] = res
	}
	return out, nil
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
	ids, err := d.promptIDs(kind, state, question, shown, descs)
	if err != nil {
		return nil, 0, err
	}
	return d.scoreIDs(ctx, ids, kind, shown, temp, nil)
}

// promptIDs renders and tokenizes one question's prompt.
func (d *Decider) promptIDs(kind, state, question string, shown, descs []string) ([]int, error) {
	var ids []int
	var err error
	if d.head != nil { // the head was trained on jev_core's prompts: a long state is cut, and no option is described
		if state, _, err = truncateState(d.tok, state); err != nil {
			return nil, err
		}
		descs = nil
	}
	if d.template == TemplateChat {
		ids, err = d.tok.EncodeChat(RenderChat(kind, state, question, shown, descs...))
	} else {
		ids, err = d.tok.EncodePlain(Render(kind, state, question, shown, descs...))
	}
	if err != nil {
		return nil, err
	}
	if len(ids) == 0 {
		return nil, errors.New("decide: empty prompt")
	}
	return ids, nil
}

// scoreIDs reads a rendered prompt's restricted log-probabilities. pre, when non-nil, is the model's output for ids computed in advance by a many-prompt call
// (a Route B hidden state or a Route A logit vector): the same value the one-prompt call would have produced, so nothing else changes.
func (d *Decider) scoreIDs(ctx context.Context, ids []int, kind string, shown []string, temp float64, pre []float32) ([]float64, int, error) {
	if d.head != nil {
		h := pre
		if h == nil {
			var err error
			if h, err = d.hidden(ctx, ids); err != nil {
				return nil, 0, err
			}
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
	logits := pre
	if logits == nil {
		var err error
		if logits, err = d.prefill(ctx, ids); err != nil {
			return nil, 0, err
		}
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
