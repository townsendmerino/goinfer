package decoder

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
)

// rebuildProgramForTest rebuilds m's program from its Architecture, for a test that changes the Architecture after load to
// simulate another configuration: the program is what the forward reads.
func (m *Model) rebuildProgramForTest() { m.prog = buildProgram(m.w.arch, m.w.hasExperts) }

// Gates P and L of docs/tasks/task-forward-architecture-2026-10.md §4.8 (OP3), on the per-token walker. Both change a model's
// built program and run the per-token forward over a fixed prompt on every tiny fixture whose family walks a program; a change
// is detected when the last position's logits differ in any bit, or the forward errors or panics.

// programFixture is one tiny checkpoint whose family walks a program, loaded at f32 on the CPU.
type programFixture struct {
	name string
	m    *Model
}

// programFixtures loads every tiny checkpoint directory under the two testdata trees whose family has a layer program. A
// directory that does not load is reported, not hidden.
func programFixtures(t *testing.T) []programFixture {
	t.Helper()
	var dirs []string
	for _, root := range []string{"../testdata", "testdata"} {
		ents, err := os.ReadDir(root)
		if err != nil {
			continue
		}
		for _, e := range ents {
			if !strings.Contains(e.Name(), "tiny") {
				continue
			}
			d := filepath.Join(root, e.Name())
			if !e.IsDir() {
				if strings.HasSuffix(e.Name(), ".gguf") {
					dirs = append(dirs, d) // a tiny GGUF fixture (gpt-oss's is one)
				}
				continue
			}
			if _, err := os.Stat(filepath.Join(d, "config.json")); err == nil {
				dirs = append(dirs, d)
			}
		}
	}
	// GPT-2 (gitignored, so absent on CI) is the only fixture with learned positions and non-zero biases: the tiny fixtures'
	// biases are all zero, as HF initialises them.
	if _, err := os.Stat("../testdata/gpt2/config.json"); err == nil {
		dirs = append(dirs, "../testdata/gpt2")
	}
	var out []programFixture
	var notLoaded []string
	for _, d := range dirs {
		m, err := Load(d, Options{Backend: "cpu", Quant: "f32"})
		if err != nil {
			notLoaded = append(notLoaded, filepath.Base(d)+": "+firstLine(err.Error()))
			continue
		}
		t.Cleanup(func() { m.Close() })
		if m.prog.layers == nil {
			continue // its own loop: no layer program yet
		}
		out = append(out, programFixture{name: filepath.Base(d), m: m})
	}
	if len(notLoaded) > 0 {
		t.Logf("%d tiny directories did not load as text models (not counted): %s", len(notLoaded), strings.Join(notLoaded, "; "))
	}
	if len(out) == 0 {
		t.Skip("no tiny fixture with a layer program is present")
	}
	return out
}

func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}

// programPrompt is long enough to pass every tiny fixture's sliding window, and short enough for every context.
func programPrompt(m *Model) []int {
	n := 24
	if mp := m.w.Cfg.MaxPositions; mp > 0 && mp < n+2 {
		n = mp - 2
	}
	ids := make([]int, n)
	for i := range ids {
		ids[i] = (i*7 + 3) % m.w.arch.VocabSize
	}
	return ids
}

// walkerMode names the walker a run goes through: the per-token forward, the batched forward (forwardN, exact attention), or
// MC3c's multi-sequence step (two sequences in lockstep, the first one's logits read).
type walkerMode string

const (
	modeToken walkerMode = "token"
	modeBatch walkerMode = "batch"
	modeMulti walkerMode = "multi"
)

var walkerModes = []walkerMode{modeToken, modeBatch, modeMulti}

// eligible reports whether m's unchanged model runs mode at all, so a mode it declines is not read as a change.
func (mode walkerMode) eligible(m *Model, n int) bool {
	switch mode {
	case modeBatch:
		return m.canBatchN(n)
	case modeMulti:
		return m.cpuBatchModelEligible() == nil && cpuBatchCacheEligible(m.NewCache(n)) == nil
	}
	return true
}

// runProgram runs ids through mode with prog and returns the last logits, or the error or panic it ended on.
func runProgram(m *Model, prog *program, ids []int, mode walkerMode) (logits []float32, err error) {
	mm := *m
	mm.prog = prog
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("panic: %v", r)
		}
	}()
	newCache := func() *KVCache {
		c := mm.NewCache(len(ids))
		if len(m.w.arch.MRopeSection) > 0 {
			// A text prompt's m-RoPE positions are (p, p, p), where the section and the interleave cannot show: give each axis its own.
			c.mropePos = make([][3]int, len(ids))
			for i := range ids {
				c.mropePos[i] = [3]int{i, i / 2, i % 3}
			}
		}
		return c
	}
	switch mode {
	case modeBatch:
		rows, ferr := mm.forwardN(context.Background(), ids, newCache())
		if ferr != nil {
			return nil, ferr
		}
		return append([]float32(nil), rows[len(rows)-1]...), nil
	case modeMulti:
		c0, c1 := newCache(), newCache()
		for i, id := range ids {
			rows, ferr := mm.decodeMultiStep([]int{id, ids[len(ids)-1-i]}, []*KVCache{c0, c1})
			if ferr != nil {
				return nil, ferr
			}
			logits = rows[0]
		}
		return append([]float32(nil), logits...), nil
	}
	cache := newCache()
	for _, id := range ids {
		lg, ferr := mm.forward(id, cache)
		if ferr != nil {
			return nil, ferr
		}
		logits = lg
	}
	return append([]float32(nil), logits...), nil
}

// outcome compares a changed forward's result with the unchanged one.
func outcome(ref []float32, got []float32, err error) string {
	if err != nil {
		return "error"
	}
	for i := range ref {
		if math.Float32bits(ref[i]) != math.Float32bits(got[i]) {
			return "changed"
		}
	}
	return "same"
}

// cloneProgram copies p deeply enough that a test may change any operation's parameters.
func cloneProgram(p *program) *program {
	c := *p
	c.layers = make([][]op, len(p.layers))
	for l, ops := range p.layers {
		c.layers[l] = append([]op(nil), ops...)
		for i := range c.layers[l] {
			o := &c.layers[l][i]
			o.rope.InvFreq = append([]float64(nil), o.rope.InvFreq...)
			o.rope.MRopeSection = append([]int(nil), o.rope.MRopeSection...)
			o.mla.InvFreq = append([]float64(nil), o.mla.InvFreq...)
		}
	}
	return &c
}

// TestProgram_P_plants is gate P: a dropped operation, two swapped operations, and a parameter off by one each change the
// logits on at least one fixture, through each walker.
func TestProgram_P_plants(t *testing.T) {
	fx := programFixtures(t)
	plants := []struct {
		name string
		// apply changes layer 0 of a clone; false when the fixture has nothing it applies to.
		apply func(a *Architecture, ops []op) ([]op, bool)
	}{
		{"drop layer 0's second residual", func(_ *Architecture, ops []op) ([]op, bool) {
			idx := indexesOf(ops, opResidual)
			i := idx[len(idx)-1] // the parallel block has one
			if len(idx) > 1 {
				i = idx[1]
			}
			return append(ops[:i:i], ops[i+1:]...), true
		}},
		{"swap layer 0's qk-norm and rope, or its first norm and the operation after it", func(_ *Architecture, ops []op) ([]op, bool) {
			qn, rp := indexesOf(ops, opQKNorm), indexesOf(ops, opRope)
			i, j := 0, 1 // the input norm and the operation after it (proj-qkv, or a whole sublayer such as MLA)
			if len(qn) > 0 && len(rp) > 0 {
				i, j = qn[0], rp[0]
			}
			ops[i], ops[j] = ops[j], ops[i]
			return ops, true
		}},
		{"rope one pair short on layer 0", func(_ *Architecture, ops []op) ([]op, bool) {
			r := indexesOf(ops, opRope)
			if len(r) == 0 || len(ops[r[0]].rope.InvFreq) < 2 {
				return ops, false
			}
			p := &ops[r[0]].rope
			p.InvFreq = p.InvFreq[:len(p.InvFreq)-1]
			return ops, true
		}},
		{"attention window one key short on the first windowed layer", nil},
	}
	for _, mode := range walkerModes {
		for _, pl := range plants {
			changed, ran := 0, 0
			var cells []string
			for _, f := range fx {
				ids := programPrompt(f.m)
				if !mode.eligible(f.m, len(ids)) {
					continue
				}
				ref, err := runProgram(f.m, f.m.prog, ids, mode)
				if err != nil {
					t.Fatalf("%s (%s): the unchanged program fails: %v", f.name, mode, err)
				}
				c := cloneProgram(f.m.prog)
				ok := false
				if pl.apply != nil {
					c.layers[0], ok = pl.apply(f.m.w.arch, c.layers[0])
				} else {
				windowed:
					for l := range c.layers {
						for i := range c.layers[l] {
							if o := &c.layers[l][i]; o.kind == opAttention && o.attn.Window > 1 {
								o.attn.Window--
								ok = true
								break windowed
							}
						}
					}
				}
				if !ok {
					continue
				}
				ran++
				got, err := runProgram(f.m, c, ids, mode)
				res := outcome(ref, got, err)
				if res != "same" {
					changed++
				}
				cells = append(cells, f.name+"="+res)
			}
			if ran == 0 {
				t.Logf("P %s %q NOT RUN: no fixture this walker runs has what it changes", mode, pl.name)
				continue
			}
			t.Logf("P %s %q: changed on %d of %d fixtures it applies to (%s)", mode, pl.name, changed, ran, strings.Join(cells, " "))
			if changed == 0 {
				t.Errorf("P %s %q changed nothing on any fixture: the walker does not read what the plant changed", mode, pl.name)
			}
		}
	}
}

func indexesOf(ops []op, k opKind) []int {
	var idx []int
	for i := range ops {
		if ops[i].kind == k {
			idx = append(idx, i)
		}
	}
	return idx
}

// programParamUnreached names the (kind, parameter) pairs no tiny fixture's logits can show, each with the reason. Gate L
// passes a pair only when it changes some fixture's logits, or is listed here; a listed pair that does change is stale.
var programParamUnreached = map[string]string{
	"ffn.Limit":              "gpt-oss's clamp bound: the tiny fixture's gate and up values never reach 7, so a doubled bound clamps nothing either",
	"mla.HeadWise":           "the MLA output gate's granularity: DeepSeek has no gate; Bailing Hybrid's MLA layers do, and move in OP3f",
	"ffn.MoE.PerExpertScale": "Gemma 4's router extra, read only by its own loop",
	"ffn.MoE.RouterPreNorm":  "Gemma 4's router extra, read only by its own loop",
	"head.Norm.Slot":         "the head has one norm weight; the slot names it and is not read",
}

// programParamNeeds names the fixture that alone reaches a pair. Where that fixture is absent (it is gitignored), the pair is
// reported NOT RUN, never passed.
var programParamNeeds = map[string]string{
	"embed.LearnedPos": "gpt2",
	"norm.Bias":        "gpt2",
	"head.Norm.Bias":   "gpt2",
}

// TestProgram_L_everyParameterIsLive is gate L: every parameter the builder emits, changed alone in every operation that carries
// it, changes the logits of at least one fixture through the per-token walker. Through the batched and multi-sequence walkers it
// must change every fixture they run whose per-token logits it changes: a walker that read the parameter from elsewhere would
// keep that fixture's logits.
func TestProgram_L_everyParameterIsLive(t *testing.T) {
	fx := programFixtures(t)
	type tally struct {
		changed, errored, same []string
	}
	// res[mode][key][fixture]: what changing key did to that fixture's logits through that walker.
	res := map[walkerMode]map[string]map[string]string{}
	pairs := map[string]*tally{}
	for _, mode := range walkerModes {
		res[mode] = map[string]map[string]string{}
		for _, f := range fx {
			ids := programPrompt(f.m)
			if !mode.eligible(f.m, len(ids)) {
				continue
			}
			ref, err := runProgram(f.m, f.m.prog, ids, mode)
			if err != nil {
				t.Fatalf("%s (%s): the unchanged program fails: %v", f.name, mode, err)
			}
			for _, key := range programParamKeys(f.m.prog) {
				c := cloneProgram(f.m.prog)
				programPerturb(c, key)
				got, err := runProgram(f.m, c, ids, mode)
				r := outcome(ref, got, err)
				if res[mode][key] == nil {
					res[mode][key] = map[string]string{}
				}
				res[mode][key][f.name] = r
				if mode != modeToken {
					continue
				}
				tl := pairs[key]
				if tl == nil {
					tl = &tally{}
					pairs[key] = tl
				}
				switch r {
				case "changed":
					tl.changed = append(tl.changed, f.name)
				case "error":
					tl.errored = append(tl.errored, f.name+" ("+firstLine(err.Error())+")")
				default:
					tl.same = append(tl.same, f.name)
				}
			}
		}
	}
	keys := make([]string, 0, len(pairs))
	for k := range pairs {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		tl := pairs[k]
		isLive := len(tl.changed) > 0 || len(tl.errored) > 0
		t.Logf("L token %-28s changed %2d  error %2d  same %2d  %s", k, len(tl.changed), len(tl.errored), len(tl.same), firstOf(tl.changed, tl.errored))
		reason, listed := programParamUnreached[k]
		if need, ok := programParamNeeds[k]; ok && !isLive && !hasFixture(fx, need) {
			t.Logf("L %s NOT RUN: only %s reaches it, and it is not present (a skip is not a pass)", k, need)
			continue
		}
		switch {
		case isLive && listed:
			t.Errorf("L %s is listed as unreached (%s) but changes the logits: remove it from programParamUnreached", k, reason)
		case !isLive && !listed:
			t.Errorf("L %s changed no fixture's logits: the walker may read it from elsewhere, or no fixture reaches it (then list it, with the reason)", k)
		}
	}
	for k := range programParamUnreached {
		if pairs[k] == nil {
			t.Errorf("L %s is listed as unreached but no fixture's program carries it: remove it", k)
		}
	}
	// The other walkers, against the per-token one, fixture by fixture. Only a per-token CHANGE binds them: an error is the path's
	// own (a width one too large overruns the per-token scratch, and the batched path, allocating per call, computes an extra zero
	// column that the logits cannot show).
	for _, mode := range []walkerMode{modeBatch, modeMulti} {
		checked, notRun := 0, 0
		for _, k := range keys {
			n := 0
			for f, r := range res[mode][k] {
				if res[modeToken][k][f] != "changed" {
					continue
				}
				n++
				if r == "same" {
					t.Errorf("L %s %s: changes %s's per-token logits but not its %s logits: that walker does not read it from the program", mode, k, f, mode)
				}
			}
			if n == 0 {
				notRun++
				t.Logf("L %s %-28s NOT RUN: no fixture this walker runs shows it per token", mode, k)
				continue
			}
			checked++
		}
		t.Logf("L %s: %d parameters checked against the per-token walker, %d not run", mode, checked, notRun)
	}
}

func hasFixture(fx []programFixture, name string) bool {
	for _, f := range fx {
		if f.name == name {
			return true
		}
	}
	return false
}

func firstOf(a, b []string) string {
	if len(a) > 0 {
		return "e.g. " + a[0]
	}
	if len(b) > 0 {
		return "e.g. " + b[0]
	}
	return ""
}

// programParamKeys lists the "kind.Field" parameter paths p carries, once each. The embedding and head stages are "embed" and
// "head".
func programParamKeys(p *program) []string {
	seen := map[string]bool{}
	var keys []string
	add := func(prefix string, v reflect.Value) {
		walkParams(prefix, v, func(path string, _ reflect.Value) {
			if !seen[path] {
				seen[path] = true
				keys = append(keys, path)
			}
		})
	}
	add("embed", reflect.ValueOf(&p.embed).Elem())
	add("head", reflect.ValueOf(&p.head).Elem())
	for _, ops := range p.layers {
		for i := range ops {
			if v, name, ok := opParamValue(&ops[i]); ok {
				add(name, v)
			}
		}
	}
	return keys
}

// programPerturb changes the parameter at key in every place p carries it: an integer by one, a float doubled (or set to 1
// from 0), a flag toggled, a slice one element short.
func programPerturb(p *program, key string) {
	visit := func(prefix string, v reflect.Value) {
		walkParams(prefix, v, func(path string, f reflect.Value) {
			if path == key {
				perturbValue(f)
			}
		})
	}
	visit("embed", reflect.ValueOf(&p.embed).Elem())
	visit("head", reflect.ValueOf(&p.head).Elem())
	for _, ops := range p.layers {
		for i := range ops {
			if v, name, ok := opParamValue(&ops[i]); ok {
				visit(name, v)
			}
		}
	}
}

// opParamValue returns the parameter struct of o's kind, settable, and the kind's name.
func opParamValue(o *op) (reflect.Value, string, bool) {
	var p any
	switch o.kind {
	case opNorm:
		p = &o.norm
	case opProjQKV, opProjO:
		p = &o.proj
	case opQKNorm:
		p = &o.qkNorm
	case opRope:
		p = &o.rope
	case opAttnTemp:
		p = &o.temp
	case opAttention:
		p = &o.attn
	case opOutGate:
		p = &o.gate
	case opFFN:
		p = &o.ffn
	case opResidual:
		p = &o.res
	case opMLA:
		p = &o.mla
	default:
		return reflect.Value{}, "", false
	}
	return reflect.ValueOf(p).Elem(), o.kind.String(), true
}

// walkParams calls fn on every leaf field under v, recursing into nested structs (MoEConfig, the head's norm).
func walkParams(prefix string, v reflect.Value, fn func(path string, f reflect.Value)) {
	for i := range v.NumField() {
		sf, f := v.Type().Field(i), v.Field(i)
		if !sf.IsExported() {
			continue
		}
		path := prefix + "." + sf.Name
		if f.Kind() == reflect.Struct {
			walkParams(path, f, fn)
			continue
		}
		fn(path, f)
	}
}

func perturbValue(f reflect.Value) {
	switch f.Kind() {
	case reflect.Bool:
		f.SetBool(!f.Bool())
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		f.SetInt(f.Int() + 1)
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		f.SetUint(f.Uint() + 1)
	case reflect.Float32, reflect.Float64:
		if f.Float() == 0 {
			f.SetFloat(1)
		} else {
			f.SetFloat(f.Float() * 2)
		}
	case reflect.Slice:
		if f.Len() > 0 {
			f.Set(f.Slice(0, f.Len()-1))
		}
	default:
		panic(errors.New("perturbValue: unhandled kind " + f.Kind().String()))
	}
}
