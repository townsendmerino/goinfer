package decide

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/goinfer/decoder"
)

// Route names, as a Result and the CLI's output lines report them.
const (
	RouteLabel = "label" // Route A: the model's logits at the verbalizer tokens (D1)
	RouteHead  = "head"  // Route B: a trained decision head over the final-norm hidden state (D4)
)

// Head is a trained decision head (Route B, D4 of docs/tasks/task-constrained-confidence.md), in the shape autotrust's
// JEV models ship: judge_config.json (the slot layout, the template version, how the adapter is applied),
// head.safetensors (one fp32 linear map proj.weight [slots, hidden] and proj.bias [slots]) and calibration.json (a
// temperature per kind). The reference decide() runs the prompt through the backbone with the LoRA active, takes the
// post-final-norm hidden state at the last prompt token, applies the head, keeps only the slots of the request's kind
// (and, for choice, only as many as it has options), divides by the kind's temperature and softmaxes. That is what
// Decider does with a Head.
//
// The backbone is the caller's to load: the base weights with the adapter merged in (decoder.Options.LoRA =
// AdapterDir()) when WeightsMode is "unmerged", as-is when it is "merged". goinfer merges an "unmerged" adapter at load
// rather than applying it at compute time; the D3 gate measures that against PEFT's unmerged forward.
type Head struct {
	Dir         string
	Name        string // model_name, for the record
	Version     string // model_version
	Hidden      int
	WeightsMode string // "unmerged" or "merged"
	Template    string // the template the head was trained under; bare-v1 is the only one this loader knows
	// Calibration is calibration.json from Dir, or nil when there is none.
	Calibration *Calibration

	w, b    []float32 // [slots, hidden], [slots]
	ranges  map[string][2]int
	adapter string // the adapter dir, "" when merged
}

// judgeConfig is the part of judge_config.json the loader reads. softcap is kept as raw JSON so that any value other
// than null is refused rather than ignored.
type judgeConfig struct {
	HiddenSize int `json:"hidden_size"`
	Slots      struct {
		NumSlots        int               `json:"num_slots"`
		Ranges          map[string][2]int `json:"ranges"`
		Verbalizers     []string          `json:"verbalizers"`
		TemplateVersion string            `json:"template_version"`
	} `json:"slots"`
	Kinds            []string        `json:"kinds"`
	ModelName        string          `json:"model_name"`
	ModelVersion     string          `json:"model_version"`
	WeightsMode      string          `json:"weights_mode"`
	AdapterSubfolder string          `json:"adapter_subfolder"`
	Softcap          json.RawMessage `json:"softcap"`
}

// HeadQuant is the quant a model loads at when a decision head is attached: chosen, when the user chose one (set:
// a per-model override, or a --quant actually passed), else decoder.DecisionHeadQuant. set is separate from the
// value because an explicit "" is f32 (serve's quant= with no value), a real choice that must not be read as unset.
// goinfer-chat decide and goinfer-serve's head= both use it, so the CLI and the server cannot drift apart.
func HeadQuant(chosen string, set bool) string {
	if set {
		return chosen
	}
	return decoder.DecisionHeadQuant
}

// LoadHead reads a decision head from dir. It refuses what it cannot run exactly as the reference does:
//   - a weights_mode other than "unmerged" or "merged";
//   - a template version other than bare-v1, since the prompt must be byte-identical to the one the head was trained on;
//   - a softcap (its semantics are unpublished);
//   - a kind goinfer does not have, or a kind's slots whose verbalizers are not goinfer's labels for that kind in
//     order (false/true, 0..5, A..P), because slot i is read as option i (the kinds' labels are disjoint, so this
//     also rules out two kinds sharing a slot);
//   - slot ranges that leave the head's rows, and a head whose shape disagrees with the config.
func LoadHead(dir string) (*Head, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "judge_config.json"))
	if err != nil {
		return nil, fmt.Errorf("decide(head): %w", err)
	}
	var jc judgeConfig
	if err := json.Unmarshal(raw, &jc); err != nil {
		return nil, fmt.Errorf("decide(head): judge_config.json: %w", err)
	}
	h := &Head{Dir: dir, Name: jc.ModelName, Version: jc.ModelVersion, Hidden: jc.HiddenSize, WeightsMode: jc.WeightsMode,
		Template: jc.Slots.TemplateVersion, ranges: jc.Slots.Ranges}
	switch jc.WeightsMode {
	case "unmerged":
		sub := jc.AdapterSubfolder
		if sub == "" {
			sub = "adapter"
		}
		if filepath.IsAbs(sub) || strings.Contains(filepath.ToSlash(sub), "..") {
			return nil, fmt.Errorf("decide(head): adapter_subfolder %q must be a folder inside %s", sub, dir)
		}
		h.adapter = filepath.Join(dir, sub)
		if _, err := os.Stat(filepath.Join(h.adapter, "adapter_config.json")); err != nil {
			return nil, fmt.Errorf("decide(head): weights_mode is unmerged but the adapter is missing: %w", err)
		}
	case "merged":
	default:
		return nil, fmt.Errorf("decide(head): weights_mode %q; this loader knows \"unmerged\" and \"merged\"", jc.WeightsMode)
	}
	if jc.Slots.TemplateVersion != TemplateBare {
		return nil, fmt.Errorf("decide(head): template_version %q; this loader knows only %s, and the prompt must be the one the head was trained on",
			jc.Slots.TemplateVersion, TemplateBare)
	}
	if s := strings.TrimSpace(string(jc.Softcap)); s != "" && s != "null" {
		return nil, fmt.Errorf("decide(head): softcap %s is set; its semantics are unpublished, so this loader refuses it", s)
	}
	if h.Hidden <= 0 || jc.Slots.NumSlots <= 0 || len(jc.Slots.Verbalizers) != jc.Slots.NumSlots {
		return nil, fmt.Errorf("decide(head): hidden_size %d, num_slots %d, %d verbalizers", h.Hidden, jc.Slots.NumSlots, len(jc.Slots.Verbalizers))
	}
	if len(jc.Kinds) == 0 {
		return nil, errors.New("decide(head): judge_config.json lists no kinds")
	}
	for _, k := range jc.Kinds {
		r, ok := jc.Slots.Ranges[k]
		if !ok {
			return nil, fmt.Errorf("decide(head): kind %q has no slot range", k)
		}
		lo, hi := r[0], r[1]
		if lo < 0 || hi > jc.Slots.NumSlots || hi-lo < 2 {
			return nil, fmt.Errorf("decide(head): kind %q slots [%d,%d) outside the head's %d", k, lo, hi, jc.Slots.NumSlots)
		}
		var want []string
		switch k {
		case KindNoul:
			want = noulOptions
		case KindScore:
			want = ScoreOptions(hi - lo)
		case KindChoice:
			if hi-lo > MaxChoiceOptions {
				return nil, fmt.Errorf("decide(head): %d choice slots, more than the %d letters", hi-lo, MaxChoiceOptions)
			}
			want = verbalizers(KindChoice, hi-lo)
		default:
			return nil, fmt.Errorf("decide(head): kind %q is not one goinfer decides", k)
		}
		got := jc.Slots.Verbalizers[lo:hi]
		if !equal(got, want) {
			return nil, fmt.Errorf("decide(head): kind %q slots [%d,%d) are %q, want %q in that order", k, lo, hi, got, want)
		}
	}
	for k := range jc.Slots.Ranges {
		if !slices.Contains(jc.Kinds, k) {
			return nil, fmt.Errorf("decide(head): a slot range for %q, which kinds does not list", k)
		}
	}

	st, err := embed.OpenSafetensors(filepath.Join(dir, "head.safetensors"))
	if err != nil {
		return nil, fmt.Errorf("decide(head): %w", err)
	}
	defer st.Close()
	if h.w, err = st.TensorF32("proj.weight", jc.Slots.NumSlots, h.Hidden); err != nil {
		return nil, fmt.Errorf("decide(head): proj.weight: %w", err)
	}
	if h.b, err = st.TensorF32("proj.bias", jc.Slots.NumSlots); err != nil {
		return nil, fmt.Errorf("decide(head): proj.bias: %w", err)
	}
	h.w, h.b = append([]float32(nil), h.w...), append([]float32(nil), h.b...) // outlive the file

	if _, err := os.Stat(filepath.Join(dir, "calibration.json")); err == nil {
		if h.Calibration, err = LoadCalibration(filepath.Join(dir, "calibration.json")); err != nil {
			return nil, err
		}
	}
	return h, nil
}

// AdapterDir is the LoRA adapter the backbone needs merged in (decoder.Options.LoRA), or "" when the head's weights
// are already merged.
func (h *Head) AdapterDir() string { return h.adapter }

// Kinds returns the kinds this head has slots for, sorted.
func (h *Head) Kinds() []string {
	ks := make([]string, 0, len(h.ranges))
	for k := range h.ranges {
		ks = append(ks, k)
	}
	sort.Strings(ks)
	return ks
}

// check refuses a request this head was not trained to answer: a kind it has no slots for, more options than its
// slots, and a score whose levels are not exactly its own (JEV's prompt always shows 0..5). Option descriptions are not
// refused: the head route leaves them out of the prompt, as jev_core's build_decision_prompt does (it renders a score
// as bare 0..5 and a choice by option names), and Result.DescriptionsDropped says so.
func (h *Head) check(r Request) error {
	rg, ok := h.ranges[r.Kind]
	if !ok {
		return fmt.Errorf("decide: the decision head has no slots for kind %q (it has %s)", r.Kind, strings.Join(h.Kinds(), ", "))
	}
	n := rg[1] - rg[0]
	switch {
	case r.Kind == KindScore && len(r.Options) != n:
		return fmt.Errorf("decide: the decision head scores exactly %d levels (0..%d), not %d", n, n-1, len(r.Options))
	case len(r.Options) > n:
		return fmt.Errorf("decide: %d options, more than the decision head's %d %s slots", len(r.Options), n, r.Kind)
	}
	return nil
}

// logits applies the head to a hidden state and returns the first n slot logits of kind.
func (h *Head) logits(hidden []float32, kind string, n int) ([]float32, error) {
	if len(hidden) != h.Hidden {
		return nil, fmt.Errorf("decide: hidden state of size %d; the decision head reads %d (is this its backbone?)", len(hidden), h.Hidden)
	}
	lo := h.ranges[kind][0]
	z := make([]float32, n)
	for i := range z {
		row := h.w[(lo+i)*h.Hidden : (lo+i+1)*h.Hidden]
		var s float64
		for j, x := range hidden {
			s += float64(row[j]) * float64(x)
		}
		z[i] = float32(s + float64(h.b[lo+i]))
	}
	return z, nil
}

// Hidden runs one prompt and returns the final-norm hidden state at its last position (Route B's input).
type Hidden func(ctx context.Context, ids []int) ([]float32, error)

// ModelHidden returns a Hidden over a loaded model: decoder.Model.PromptHidden (D2).
func ModelHidden(m *decoder.Model) Hidden { return m.PromptHidden }

// ModelHiddenMany is the many-prompt form of ModelHidden: prompts that share a prefix are prefilled once where the model can (decoder.Model.PromptHiddenMany), and
// it is PromptHidden per prompt everywhere else, so it is safe to hand to any model.
func ModelHiddenMany(m *decoder.Model) HiddenMany { return m.PromptHiddenMany }
