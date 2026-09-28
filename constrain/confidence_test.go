package constrain

import (
	"math"
	"slices"
	"strings"
	"testing"
)

const confSchema = `{"type":"object","properties":{
"category":{"enum":["billing","technical","account"]},
"urgent":{"type":"boolean"},
"count":{"type":"integer"},
"amount":{"type":"number"},
"name":{"type":"string"},
"meta":{"type":"object","properties":{"level":{"enum":["low","high"]}},"required":["level"],"additionalProperties":false},
"tags":{"type":"array","items":{"enum":["x","y"]}}},
"required":["category","urgent","count","amount","name","meta","tags"],"additionalProperties":false}`

const confTarget = `{"category":"billing","urgent":true,"count":12,"amount":3.5,"name":"Al","meta":{"level":"high"},"tags":["x","y"]}`

// confVocab is every printable ASCII byte as a token (ids 0..127), a few multi-byte tokens, and an EOS with no bytes.
func confVocab() (tokens [][]byte, ids map[string]int, eos int) {
	ids = map[string]int{}
	for b := 0; b < 128; b++ {
		tokens = append(tokens, []byte{byte(b)})
		ids[string(rune(b))] = b
	}
	for _, s := range []string{"bill", "ing", "tech", "nical", "true", "false"} {
		ids[s] = len(tokens)
		tokens = append(tokens, []byte(s))
	}
	eos = len(tokens)
	tokens = append(tokens, nil)
	return tokens, ids, eos
}

// runConf drives a greedy generation of confTarget through a Masker. At each step every token gets logit −30 unless a
// design for the output so far says otherwise; with no design, the longest legal token that continues the target gets
// 0. Designs give exact probabilities (the −30 mass is < 1e-10).
func runConf(t *testing.T, m *Masker, tokens [][]byte, ids map[string]int, eos int, design map[string]map[string]float64) []int {
	t.Helper()
	var gen []int
	var out string
	// The driver's own copy of the grammar, advanced as tokens are picked: the Masker commits a token only at the
	// next Process call, so its own state lags one token behind the driver's.
	replica, err := JSONSchema([]byte(confSchema))
	if err != nil {
		t.Fatal(err)
	}
	for step := 0; step < 400; step++ {
		logits := make([]float32, len(tokens))
		for i := range logits {
			logits[i] = -30
		}
		if d, ok := design[out]; ok {
			for tok, p := range d {
				logits[ids[tok]] = float32(math.Log(p))
			}
		} else if out != confTarget {
			rem := confTarget[len(out):]
			best := -1
			for id, b := range tokens {
				if len(b) > 0 && strings.HasPrefix(rem, string(b)) && (best < 0 || len(b) > len(tokens[best])) && replica.TryBytes(b) {
					best = id
				}
			}
			if best < 0 {
				t.Fatalf("no legal token continues the target at %q", out)
			}
			logits[best] = 0
		}
		m.Process(gen, logits)
		pick := 0
		for i, v := range logits {
			if v > logits[pick] {
				pick = i
			}
		}
		gen = append(gen, pick)
		if pick == eos {
			return gen
		}
		out += string(tokens[pick])
		replica.Commit(tokens[pick])
		if !strings.HasPrefix(confTarget, out) {
			t.Fatalf("generation left the target: %q", out)
		}
	}
	t.Fatal("no EOS")
	return nil
}

// The designs: enum (a BPE split: "bill" then a forced "ing", and single letters beside the multi-byte tokens), a
// boolean, a two-digit integer, a nested enum, and two array items.
var confDesign = map[string]map[string]float64{
	`{"category":"`:                                 {"bill": 0.5, "b": 0.2, "tech": 0.2, "t": 0.05, "a": 0.05},
	`{"category":"billing","urgent":`:               {"true": 0.8, "t": 0.05, "false": 0.1, "f": 0.05},
	`{"category":"billing","urgent":true,"count":`:  {"1": 0.5, "2": 0.3, "9": 0.2},
	`{"category":"billing","urgent":true,"count":1`: {"2": 0.7, ",": 0.2, "0": 0.1},
	`{"category":"billing","urgent":true,"count":12,"amount":3.5,"name":"Al","meta":{"level":"`:                     {"h": 0.9, "l": 0.1},
	`{"category":"billing","urgent":true,"count":12,"amount":3.5,"name":"Al","meta":{"level":"high"},"tags":["`:     {"x": 0.6, "y": 0.4},
	`{"category":"billing","urgent":true,"count":12,"amount":3.5,"name":"Al","meta":{"level":"high"},"tags":["x","`: {"x": 0.3, "y": 0.7},
}

func TestFieldConfidence(t *testing.T) {
	tokens, ids, eos := confVocab()
	g, err := JSONSchema([]byte(confSchema))
	if err != nil {
		t.Fatal(err)
	}
	m := NewMasker(g, tokens, []int{eos}).StopWhenComplete().CaptureConfidence(ConfidenceOptions{})
	gen := runConf(t, m, tokens, ids, eos, confDesign)
	fcs, err := m.FieldConfidence(gen)
	if err != nil {
		t.Fatal(err)
	}
	type want struct {
		kind, value string
		conf        float64
		free        int
		dist        map[string]float64
	}
	wants := map[string]want{
		"category":   {"enum", `"billing"`, 0.7, 1, map[string]float64{"billing": 0.7, "technical": 0.25, "account": 0.05}},
		"urgent":     {"boolean", `true`, 0.85, 1, map[string]float64{"true": 0.85, "false": 0.15}},
		"count":      {"integer", `12`, 0.5, 2, nil},
		"meta.level": {"enum", `"high"`, 0.9, 1, map[string]float64{"high": 0.9, "low": 0.1}},
		// tags[0]'s opening quote is free too: it competes with ']' (close the empty array) — whether an item exists,
		// not what it is. The confidence still comes from the letter that decides the value.
		"tags[0]": {"enum", `"x"`, 0.6, 2, map[string]float64{"x": 0.6, "y": 0.4}},
		"tags[1]": {"enum", `"y"`, 0.7, 1, map[string]float64{"x": 0.3, "y": 0.7}},
	}
	var order []string
	for _, fc := range fcs {
		order = append(order, fc.Path)
		w, ok := wants[fc.Path]
		if !ok {
			t.Errorf("unexpected field %q (%s) — number and string fields must be omitted", fc.Path, fc.Kind)
			continue
		}
		if fc.Kind != w.kind || string(fc.Value) != w.value || fc.FreeTokens != w.free || fc.Calibrated {
			t.Errorf("%s: kind %s value %s free %d calibrated %v, want %s %s %d false", fc.Path, fc.Kind, fc.Value, fc.FreeTokens, fc.Calibrated, w.kind, w.value, w.free)
		}
		if math.Abs(fc.Confidence-w.conf) > 1e-6 {
			t.Errorf("%s: confidence %v, want %v", fc.Path, fc.Confidence, w.conf)
		}
		for k, p := range w.dist {
			if math.Abs(fc.Distribution[k]-p) > 1e-6 {
				t.Errorf("%s: p(%s) = %v, want %v (dist %v)", fc.Path, k, fc.Distribution[k], p, fc.Distribution)
			}
		}
	}
	if wantOrder := []string{"category", "urgent", "count", "meta.level", "tags[0]", "tags[1]"}; !slices.Equal(order, wantOrder) {
		t.Errorf("fields %v, want %v", order, wantOrder)
	}
}

// A temperature for a kind is applied at its deciding position and marks the field Calibrated; the others stay raw.
func TestFieldConfidence_temperature(t *testing.T) {
	tokens, ids, eos := confVocab()
	g, _ := JSONSchema([]byte(confSchema))
	m := NewMasker(g, tokens, []int{eos}).StopWhenComplete().CaptureConfidence(ConfidenceOptions{BooleanTemperature: 2})
	gen := runConf(t, m, tokens, ids, eos, confDesign)
	fcs, _ := m.FieldConfidence(gen)
	for _, fc := range fcs {
		switch fc.Path {
		case "urgent":
			// T = 2 takes each probability to its square root before renormalizing (the −30 tail stays negligible).
			tr, fa := math.Sqrt(0.8)+math.Sqrt(0.05), math.Sqrt(0.1)+math.Sqrt(0.05)
			if !fc.Calibrated || math.Abs(fc.Confidence-tr/(tr+fa)) > 1e-4 {
				t.Errorf("urgent at T=2: %v calibrated %v, want %v true", fc.Confidence, fc.Calibrated, tr/(tr+fa))
			}
		case "category":
			if fc.Calibrated || math.Abs(fc.Confidence-0.7) > 1e-6 {
				t.Errorf("category must stay raw without an enum temperature: %v calibrated %v", fc.Confidence, fc.Calibrated)
			}
		}
	}
}

// Off, Process is unchanged: identical masking, no capture state; FieldConfidence refuses. A non-schema grammar
// refuses too (it has no field kinds).
func TestFieldConfidence_offAndRefusals(t *testing.T) {
	tokens, _, eos := confVocab()
	g1, _ := JSONSchema([]byte(confSchema))
	g2, _ := JSONSchema([]byte(confSchema))
	on := NewMasker(g1, tokens, []int{eos}).CaptureConfidence(ConfidenceOptions{})
	off := NewMasker(g2, tokens, []int{eos})
	a, b := make([]float32, len(tokens)), make([]float32, len(tokens))
	for i := range a {
		a[i] = float32(i % 7)
		b[i] = a[i]
	}
	on.Process(nil, a)
	off.Process(nil, b)
	if !slices.Equal(a, b) {
		t.Error("capture changed the mask")
	}
	if off.conf != nil {
		t.Error("capture state exists on a masker that never enabled it")
	}
	if _, err := off.FieldConfidence(nil); err == nil {
		t.Error("FieldConfidence without CaptureConfidence did not refuse")
	}
	j := NewMasker(JSON(), tokens, []int{eos}).CaptureConfidence(ConfidenceOptions{})
	if _, err := j.FieldConfidence(nil); err == nil || !strings.Contains(err.Error(), "JSON Schema") {
		t.Errorf("a plain JSON grammar: %v, want a refusal naming JSON Schema", err)
	}
}

// Inside a free string nothing is read — the positions where the readout would cost the most, and no string field
// is reported.
func TestFieldConfidence_skipsFreeStrings(t *testing.T) {
	tokens, ids, eos := confVocab()
	g, _ := JSONSchema([]byte(confSchema))
	m := NewMasker(g, tokens, []int{eos}).StopWhenComplete().CaptureConfidence(ConfidenceOptions{})
	gen := runConf(t, m, tokens, ids, eos, confDesign)
	nameStart := strings.Index(confTarget, `"name":"`) + len(`"name":"`)
	var out string
	for i, id := range gen {
		if len(out) >= nameStart && len(out) < nameStart+2 { // the two free bytes "Al"
			if m.conf.pos[i].read {
				t.Errorf("position %d inside the free string %q was read", i, out[nameStart:])
			}
		}
		out += string(tokens[id])
	}
}
