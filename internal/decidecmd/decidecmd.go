// Package decidecmd is the goinfer-chat `decide` and `decisions-calibrate` subcommands (D1 of
// docs/tasks/task-constrained-confidence.md): decisions by label-token scoring on a loaded model, one JSONL line
// in and one out, and the per-kind temperature fit that makes their distributions calibrated in the fit's sense.
//
// A line in is a decision in the shape of SargeDev/jev-distill-corpus-v3's rows —
// {"id", "kind", "state", "question", "options", "target"?}; extra fields are ignored. A line out carries the
// distribution over the line's options and what produced it; a line that fails validation gets an "error" and the
// run continues.
package decidecmd

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/townsendmerino/goinfer/internal/decide"
	"github.com/townsendmerino/goinfer/internal/loadflags"
	"github.com/townsendmerino/goinfer/internal/modelload"
)

const decideUsage = `%[1]s decide --model <file.gguf|dir> [flags] <in.jsonl|->  — decisions by label-token scoring

Each input line is {"id","kind","state","question","options"} with kind noul (options ["false","true"]),
score (options "0".."5") or choice (2-16 options). Each output line carries the distribution over the
line's options, the decision, its probability, and what produced it. One prefill per line, no decode.

The number is the model's probability over the options it was shown. It is calibrated only for a kind
whose temperature a --calibration file (from %[1]s decisions-calibrate, or autotrust's own) supplies,
and then only in the sense that fit measured.

`

const calibrateUsage = `%[1]s decisions-calibrate --model <file.gguf|dir> [flags] -o calibration.json <labelled.jsonl>

Fits one temperature per kind on labelled lines ({"kind","state","question","options","target"}, target a
distribution over the options — one-hot for a gold label) by minimizing the mean KL(target || p_T), and writes
calibration.json in autotrust's format. Use the same --template for decide as for the fit.

`

// Row is one input line.
type Row struct {
	ID       json.RawMessage `json:"id,omitempty"`
	Kind     string          `json:"kind"`
	State    string          `json:"state"`
	Question string          `json:"question"`
	Options  []string        `json:"options"`
	Target   []float64       `json:"target,omitempty"`
}

// Out is one output line.
type Out struct {
	ID            json.RawMessage `json:"id,omitempty"`
	Kind          string          `json:"kind"`
	Options       []string        `json:"options,omitempty"`
	Distribution  []float64       `json:"distribution,omitempty"`
	Decision      string          `json:"decision,omitempty"`
	Confidence    float64         `json:"confidence,omitempty"`
	ExpectedScore *float64        `json:"expected_score,omitempty"`
	Calibrated    bool            `json:"calibrated"`
	Temperature   float64         `json:"temperature,omitempty"`
	Route         string          `json:"route"`
	Template      string          `json:"template,omitempty"`
	PromptTokens  int             `json:"prompt_tokens,omitempty"`
	Prefills      int             `json:"prefills,omitempty"`
	LatencyMs     float64         `json:"latency_ms,omitempty"`
	Target        []float64       `json:"target,omitempty"`
	Error         string          `json:"error,omitempty"`
}

type common struct {
	fs       *flag.FlagSet
	load     *loadflags.Flags
	model    *string
	template *string
	out      *string
}

func newCommon(name, usage string) *common {
	fs := flag.NewFlagSet(name, flag.ContinueOnError)
	c := &common{fs: fs, load: loadflags.Register(fs, loadflags.Chat)}
	c.model = fs.String("model", "", "model: a .gguf, an HF checkpoint dir, or an hf:/demo: reference")
	c.template = fs.String("template", decide.TemplateBare, "prompt template: bare-v1 (JEV's own; no chat template) or chat-v1 (the model's chat template, for instruct models)")
	c.out = fs.String("o", "", "output file (default stdout)")
	fs.Usage = func() {
		fmt.Fprintf(fs.Output(), usage, filepath.Base(os.Args[0]))
		fs.PrintDefaults()
	}
	return c
}

func (c *common) open(ctx context.Context, cal *decide.Calibration) (*decide.Decider, func(), error) {
	if *c.model == "" {
		return nil, nil, fmt.Errorf("--model is required")
	}
	if err := c.load.Validate(); err != nil {
		return nil, nil, err
	}
	res, err := modelload.Load(ctx, modelload.Request{Spec: *c.model, Opts: c.load.Options(), DirectLoad: c.load.DirectLoad,
		ExplicitQuant: c.load.ExplicitQuant()})
	if err != nil {
		return nil, nil, err
	}
	d, err := decide.New(decide.NewPlainTokenizer(res.Tokenizer), decide.ModelPrefill(res.Model),
		decide.Options{Template: *c.template, Calibration: cal})
	if err != nil {
		res.Model.Close()
		return nil, nil, err
	}
	return d, func() { res.Model.Close() }, nil
}

func (c *common) input() (io.ReadCloser, error) {
	args := c.fs.Args()
	if len(args) != 1 {
		return nil, fmt.Errorf("want exactly one input file (or - for stdin), got %d", len(args))
	}
	if args[0] == "-" {
		return io.NopCloser(os.Stdin), nil
	}
	return os.Open(args[0])
}

func (c *common) output() (io.WriteCloser, error) {
	if *c.out == "" {
		return nopWriteCloser{os.Stdout}, nil
	}
	return os.Create(*c.out)
}

type nopWriteCloser struct{ io.Writer }

func (nopWriteCloser) Close() error { return nil }

// Run is `decide`. It returns the process exit code.
func Run(args []string) int {
	c := newCommon("decide", decideUsage)
	calPath := c.fs.String("calibration", "", "calibration.json with per-kind temperatures (none: T = 1, and every line says calibrated: false)")
	permute := c.fs.Int("permute", 0, "choice only: average over this many option orders (cyclic rotations), one prefill each, to cancel label-position bias")
	if err := c.fs.Parse(args); err != nil {
		return 2
	}
	var cal *decide.Calibration
	if *calPath != "" {
		var err error
		if cal, err = decide.LoadCalibration(*calPath); err != nil {
			fmt.Fprintln(os.Stderr, "decide:", err)
			return 1
		}
	}
	in, err := c.input()
	if err != nil {
		fmt.Fprintln(os.Stderr, "decide:", err)
		return 2
	}
	defer in.Close()
	ctx := context.Background()
	d, closeModel, err := c.open(ctx, cal)
	if err != nil {
		fmt.Fprintln(os.Stderr, "decide:", err)
		return 1
	}
	defer closeModel()
	out, err := c.output()
	if err != nil {
		fmt.Fprintln(os.Stderr, "decide:", err)
		return 1
	}
	defer out.Close()
	enc := json.NewEncoder(out)
	failed, n := 0, 0
	t0 := time.Now()
	err = eachRow(in, func(r Row) error {
		n++
		o := Out{ID: r.ID, Kind: r.Kind, Options: r.Options, Route: "label", Template: d.Template(), Target: r.Target}
		req := decide.Request{Kind: r.Kind, State: r.State, Question: r.Question, Options: r.Options}
		if r.Kind == decide.KindChoice {
			req.Permute = *permute
		}
		res, err := d.Decide(ctx, req)
		if err != nil {
			o.Error, failed = err.Error(), failed+1
		} else {
			o.Distribution, o.Decision, o.Confidence, o.ExpectedScore = res.Distribution, res.Decision, res.Confidence, res.ExpectedScore
			o.Calibrated, o.Temperature, o.PromptTokens, o.Prefills = res.Calibrated, res.Temperature, res.PromptTokens, res.Prefills
			o.LatencyMs = float64(res.Latency.Microseconds()) / 1e3
		}
		if n%100 == 0 {
			fmt.Fprintf(os.Stderr, "decide: %d lines, %s\n", n, time.Since(t0).Round(time.Second))
		}
		return enc.Encode(o)
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "decide:", err)
		return 1
	}
	fmt.Fprintf(os.Stderr, "decide: %d lines in %s, %d failed validation\n", n, time.Since(t0).Round(time.Millisecond), failed)
	if failed > 0 {
		return 1
	}
	return 0
}

// RunCalibrate is `decisions-calibrate`. It returns the process exit code.
func RunCalibrate(args []string) int {
	c := newCommon("decisions-calibrate", calibrateUsage)
	if err := c.fs.Parse(args); err != nil {
		return 2
	}
	if *c.out == "" {
		fmt.Fprintln(os.Stderr, "decisions-calibrate: -o calibration.json is required")
		return 2
	}
	in, err := c.input()
	if err != nil {
		fmt.Fprintln(os.Stderr, "decisions-calibrate:", err)
		return 2
	}
	defer in.Close()
	ctx := context.Background()
	d, closeModel, err := c.open(ctx, nil)
	if err != nil {
		fmt.Fprintln(os.Stderr, "decisions-calibrate:", err)
		return 1
	}
	defer closeModel()
	var rows []decide.FitRow
	skipped, n := 0, 0
	t0 := time.Now()
	err = eachRow(in, func(r Row) error {
		n++
		if len(r.Target) != len(r.Options) {
			skipped++
			return nil
		}
		lp, err := d.Scores(ctx, decide.Request{Kind: r.Kind, State: r.State, Question: r.Question, Options: r.Options})
		if err != nil {
			skipped++
			return nil
		}
		rows = append(rows, decide.FitRow{Kind: r.Kind, LogP: lp, Target: r.Target})
		if n%100 == 0 {
			fmt.Fprintf(os.Stderr, "decisions-calibrate: %d lines, %s\n", n, time.Since(t0).Round(time.Second))
		}
		return nil
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "decisions-calibrate:", err)
		return 1
	}
	if len(rows) == 0 {
		fmt.Fprintln(os.Stderr, "decisions-calibrate: no usable labelled line (each needs a target aligned with its options)")
		return 1
	}
	cal := decide.FitTemperatures(rows, "goinfer decisions-calibrate "+filepath.Base(*c.model), d.Template())
	if err := cal.Save(*c.out); err != nil {
		fmt.Fprintln(os.Stderr, "decisions-calibrate:", err)
		return 1
	}
	for k, f := range cal.Fit {
		fmt.Fprintf(os.Stderr, "decisions-calibrate: %s  n=%d  T=%.4f  KL %.5f -> %.5f\n", k, f.N, f.T, f.KLBefore, f.KLAfter)
		if decide.AtSearchBound(f.T) {
			fmt.Fprintf(os.Stderr, "decisions-calibrate: WARNING %s: T hit the search bound — the model's ranking disagrees with the targets "+
				"more than any temperature can fix (flattening toward uniform, or sharpening without limit). Do not trust this kind's "+
				"calibration; check the template and the labels, and fit on more rows.\n", k)
		}
	}
	fmt.Fprintf(os.Stderr, "decisions-calibrate: %d rows fitted, %d skipped, wrote %s\n", len(rows), skipped, *c.out)
	return 0
}

// eachRow decodes JSONL rows, stopping at the first line that is not a JSON object.
func eachRow(r io.Reader, f func(Row) error) error {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 64<<20)
	line := 0
	for sc.Scan() {
		line++
		b := sc.Bytes()
		if len(b) == 0 {
			continue
		}
		var row Row
		if err := json.Unmarshal(b, &row); err != nil {
			return fmt.Errorf("line %d: %w", line, err)
		}
		if err := f(row); err != nil {
			return err
		}
	}
	return sc.Err()
}
