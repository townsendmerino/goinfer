package whisper

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// G-S14h1 of docs/tasks/task-multimodal-support-2026-10.md: the decode policy (conditioning on earlier text, the thresholds, silence skipping) against transformers' generate on the tiny random-weight Whisper
// (testdata/whisper-tiny-ts; scripts/pin_whisper_policy_tiny.py), temperature=(0.0,) so no sampled attempt is reachable.

type polGolden struct {
	Cases map[string]struct {
		Samples int
		A, B    int
		Results map[string]struct {
			Segments []struct {
				Start, End float64
				Tokens     []int
			}
			Sequence []int
		}
	}
}

func f64(v float64) *float64 { return &v }

var polConfigs = map[string]TranscribeOptions{
	"cond":  {ConditionOnPrev: true},
	"thr":   {LogprobThreshold: f64(-0.5), CompressionRatioThreshold: f64(1.2)},
	"skip":  {LogprobThreshold: f64(-0.5), NoSpeechThreshold: f64(1e-4)},
	"skip2": {LogprobThreshold: f64(-0.5), NoSpeechThreshold: f64(1e-5)},
	"all":   {ConditionOnPrev: true, LogprobThreshold: f64(-0.5), CompressionRatioThreshold: f64(1.2), NoSpeechThreshold: f64(1e-4)},
}

func loadPolicy(t *testing.T) polGolden {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(tsDir(), "policy_golden.json"))
	if err != nil {
		t.Skipf("no policy golden: %v", err)
	}
	var g polGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

// polMatches runs every case and configuration and reports which ones differ from transformers (an empty list: all equal) and how many windows each run skipped.
func polMatches(t *testing.T, tr *Transcriber, g polGolden) (diffs []string, skipped map[string]int) {
	t.Helper()
	skipped = map[string]int{}
	for cname, c := range g.Cases {
		x := tsClip(c.Samples, c.A, c.B)
		for oname, opts := range polConfigs {
			want := c.Results[oname]
			r, err := tr.TranscribeWith(x, opts)
			if err != nil {
				diffs = append(diffs, cname+"/"+oname+": "+err.Error())
				continue
			}
			for _, w := range r.Windows {
				if w.Skipped {
					skipped[cname+"/"+oname]++
				}
			}
			same := len(r.Segments) == len(want.Segments) && slices.Equal(r.Tokens, want.Sequence)
			for i := 0; same && i < len(r.Segments); i++ {
				w := want.Segments[i]
				same = r.Segments[i].Start-w.Start < 1e-9 && w.Start-r.Segments[i].Start < 1e-9 && r.Segments[i].End-w.End < 1e-9 && w.End-r.Segments[i].End < 1e-9 && slices.Equal(r.Segments[i].Tokens, w.Tokens)
			}
			if !same {
				diffs = append(diffs, cname+"/"+oname)
			}
		}
	}
	slices.Sort(diffs)
	return
}

func TestPolicy_tinyMatchesTransformers(t *testing.T) {
	tr, _ := tinyTS(t)
	g := loadPolicy(t)
	diffs, skipped := polMatches(t, tr, g)
	t.Logf("windows skipped as silence: %v", skipped)
	if len(diffs) != 0 {
		t.Errorf("differs from transformers in: %v", diffs)
	}
	if skipped["long68s/skip"] == 0 || skipped["long68s/skip2"] == 0 {
		t.Errorf("no window was skipped in the skip configurations (%v): the gate would not exercise the no-speech rule", skipped)
	}
}

// TestPolicy_plantedDefectsAreRed: each planted defect must change some result, or the gate could not see that mistake.
func TestPolicy_plantedDefectsAreRed(t *testing.T) {
	tr, _ := tinyTS(t)
	g := loadPolicy(t)
	for name, d := range map[string]int{"conditioning uncut": condNoCut, "double ending timestamp not trimmed": condNoDoubleTrim, "no startofprev": condNoStartOfPrev,
		"no-speech from processed scores": polNoSpeechProcessed, "silent window not skipped": polSkipIgnored} {
		tr.Dec.condDefect = d
		diffs, _ := polMatches(t, tr, g)
		t.Logf("%-38s red %v (%d configurations differ)", name, len(diffs) > 0, len(diffs))
		if len(diffs) == 0 {
			t.Errorf("defect %q left every result equal to transformers': the gate cannot see it", name)
		}
	}
	tr.Dec.condDefect = condNone
}

// TestPolicy_avgLogprobCountsTheStopToken: the average log-probability the fallback rule reads is over the generated tokens INCLUDING the stop token (generation_whisper.py, _retrieve_avg_logprobs: "don't
// remove the eos token logprob"). The tiny model never writes the stop token in the clips above, so the clip gate cannot see this; a run whose stop token is a token the model does write can.
func TestPolicy_avgLogprobCountsTheStopToken(t *testing.T) {
	tr, g := tinyTS(t)
	c := g.Cases["short8s"]
	x := tsClip(c.Samples, c.A, c.B)
	enc, err := tr.Encode(x)
	if err != nil {
		t.Fatal(err)
	}
	gen := tr.Gen
	prompt, _, err := tr.Dec.promptFor(enc, gen, "", "", true)
	if err != nil {
		t.Fatal(err)
	}
	free, err := tr.Dec.generate(enc, gen, prompt, true, 0, nil, -1)
	if err != nil {
		t.Fatal(err)
	}
	gen.EOS = free.ids[1] // the second token the model writes becomes the stop token: the run ends there
	for _, defect := range []int{condNone, polAvgNoStop} {
		tr.Dec.condDefect = defect
		a, err := tr.Dec.generate(enc, gen, prompt, true, 0, nil, -1)
		if err != nil {
			t.Fatal(err)
		}
		if !a.stopped || len(a.ids) != 2 || len(a.lps) != 2 {
			t.Fatalf("defect %d: ids %v stopped %v", defect, a.ids, a.stopped)
		}
		want := (a.lps[0] + a.lps[1]) / 2
		if got := a.avgLogprob(); defect == condNone && (got-want > 1e-12 || want-got > 1e-12) {
			t.Errorf("average log-probability %v, want the mean over both tokens %v", got, want)
		} else if defect == polAvgNoStop && got == want {
			t.Errorf("the planted defect (stop token left out) did not change the average")
		}
	}
	tr.Dec.condDefect = condNone
}
