package decoder

import (
	"fmt"
	"reflect"
	"slices"
	"strings"
	"testing"
)

// cpuScratchHolds is the CPU half's gate: a
// stream's decode scratch, read from the struct a cache holds, equals the plan's terms row by row, and holds no float row the terms do
// not name (a row allocated beside the list is red).
func cpuScratchHolds(scr *decodeScratch, terms []CPUScratchTerm) error {
	want := map[string]int64{}
	for _, t := range terms {
		want[t.Role] += t.Bytes
	}
	got := map[string]int64{}
	v := reflect.ValueOf(scr).Elem()
	for i := 0; i < v.NumField(); i++ {
		f := v.Field(i)
		if f.Kind() == reflect.Slice && f.Type().Elem().Kind() == reflect.Float32 && !f.IsNil() {
			got[v.Type().Field(i).Name] = 4 * int64(f.Len())
		}
	}
	var diffs []string
	for role, b := range want {
		if got[role] != b {
			diffs = append(diffs, fmt.Sprintf("%s: plan %d B, stream %d B", role, b, got[role]))
		}
	}
	for role, b := range got {
		if _, ok := want[role]; !ok {
			diffs = append(diffs, fmt.Sprintf("%s: the stream holds %d B the plan names no term for", role, b))
		}
	}
	if len(diffs) > 0 {
		slices.Sort(diffs)
		return fmt.Errorf("%s", strings.Join(diffs, "; "))
	}
	return nil
}

// TestCPUDecodeScratch_equalsTheBuild holds Model.CPUDecodeScratch to the scratch a made cache holds, on every census fixture that loads
// (the ones that do not are the vision towers and the like, which have no decode stream).
func TestCPUDecodeScratch_equalsTheBuild(t *testing.T) {
	held := 0
	for _, f := range censusFamilies(t) {
		m, err := Load(f.path, Options{})
		if err != nil {
			continue
		}
		c := m.NewCache(0)
		if c.scr == nil {
			m.Close()
			continue
		}
		if err := cpuScratchHolds(c.scr, m.CPUDecodeScratch()); err != nil {
			t.Errorf("%s: %v", f.name, err)
		}
		if n := m.CPUDecodeScratchBytes(); n <= 0 {
			t.Errorf("%s: a stream holds %d B of fixed scratch", f.name, n)
		}
		held++
		m.Close()
	}
	if held < 40 {
		t.Errorf("VOID: only %d census fixtures made a decode stream, the gate compared too little", held)
	}
	t.Logf("%d census fixtures held", held)
}

// TestCPUDecodeScratch_plantsGoRed plants the gate's two defects on llama-tiny: a float row allocated beside the list, and a term one element short.
func TestCPUDecodeScratch_plantsGoRed(t *testing.T) {
	m, err := Load("../testdata/llama-tiny", Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	scr := m.NewCache(0).scr
	terms := m.CPUDecodeScratch()
	if err := cpuScratchHolds(scr, terms); err != nil {
		t.Fatalf("unplanted: %v", err)
	}
	scr.scores = make([]float32, 8)
	if err := cpuScratchHolds(scr, terms); err == nil || !strings.Contains(err.Error(), "scores: the stream holds 32 B the plan names no term for") {
		t.Errorf("a row beside the list: err = %v, want one naming scores and 32 B", err)
	}
	scr.scores = nil
	short := slices.Clone(terms)
	for i := range short {
		if short[i].Role == "norm" {
			short[i].Bytes -= 4
		}
	}
	if err := cpuScratchHolds(scr, short); err == nil || !strings.Contains(err.Error(), "norm: plan 252 B, stream 256 B") {
		t.Errorf("a term one element short: err = %v, want one naming norm, plan 252 B, stream 256 B", err)
	}
}

// TestCPUDecodeScratch_handTable is the hand table: llama-tiny's fixed rows from its config (hidden 64, 4 heads of 16 over 2 KV heads, FFN 128,
// vocabulary 256) and qwen3moe-tiny's with its router and experts (hidden 64, 8 heads of 8 over 2 KV heads, FFN 128, 8 experts of width 32,
// vocabulary 512).
func TestCPUDecodeScratch_handTable(t *testing.T) {
	for _, c := range []struct {
		fx   string
		want map[string]int64
	}{
		{"llama-tiny", map[string]int64{"h": 64, "norm": 64, "sub": 64, "sub2": 64, "q": 64, "k": 32, "v": 32, "ctx": 64, "gate": 128, "up": 128, "logits": 256}},
		{"qwen3moe-tiny", map[string]int64{"h": 64, "norm": 64, "sub": 64, "sub2": 64, "q": 64, "k": 16, "v": 16, "ctx": 64, "gate": 128, "up": 128, "logits": 512,
			"moeLogits": 8, "moeOut": 64, "moeExpOut": 64, "moeGate": 32, "moeUp": 32}},
	} {
		m, err := Load("../testdata/"+c.fx, Options{})
		if err != nil {
			t.Fatal(err)
		}
		got := map[string]int64{}
		for _, tm := range m.CPUDecodeScratch() {
			got[tm.Role] = tm.Bytes
		}
		for k, n := range c.want {
			if got[k] != 4*n {
				t.Errorf("%s: %s = %d B, hand arithmetic says %d B", c.fx, k, got[k], 4*n)
			}
		}
		if len(got) != len(c.want) {
			t.Errorf("%s: %d rows, the hand table has %d", c.fx, len(got), len(c.want))
		}
		m.Close()
	}
}
