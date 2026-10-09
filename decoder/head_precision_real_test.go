//go:build realckpt

// Option D of the --embed-int4 decision (docs/tasks/task-multimodal-support-2026-10.md, "option D", registered before this code): what the int4 embedding / LM-head table costs against the
// int8 pin. scripts/head_precision_hf.py wrote, per model, the Hugging Face float32 logits on a fixed greedy path (16 prompts x 32 positions); this teacher-forces that path through the CPU
// backend at --quant int4 TWICE, once with EmbedInt4 false (the int8 pin) and once true, the only difference, and records per position the KL(HF || arm) and whether the arm's argmax is HF's.
// The arms are loaded one at a time. HeadTable() must read int8 in the first and int4 in the second, or the run is void. GOINFER_HP_CONTROL=1 loads the int8 pin twice (the positive control:
// the two arms must then be bit-identical). A model that cannot run is recorded in its result file and the others still run. Grading is scripts/head_precision_grade.py.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_HP_ROOT=<dump root> [GOINFER_HP_ONLY=<name>] [GOINFER_HP_CONTROL=1] go test -tags realckpt ./decoder/ -run TestHeadPrecision_real -v -timeout 150m
package decoder

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

type hpArm struct {
	EmbedInt4  bool
	HeadTable  string
	KL         [][]float64
	Agree      [][]bool
	LoadSec    float64
	SecPerStep float64
}

type hpResult struct {
	Name, Error string
	Control     bool
	Int8Head    *hpArm
	Int4Head    *hpArm
}

func TestHeadPrecision_real(t *testing.T) {
	requireHeavyModel(t)
	root := os.Getenv("GOINFER_HP_ROOT")
	if root == "" {
		t.Skip("set GOINFER_HP_ROOT to scripts/head_precision_hf.py's output root")
	}
	only := os.Getenv("GOINFER_HP_ONLY")
	control := os.Getenv("GOINFER_HP_CONTROL") == "1"
	entries, err := os.ReadDir(root)
	if err != nil {
		t.Fatal(err)
	}
	T0 := time.Now()
	hb := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[hp %5.0fs] %s\n", time.Since(T0).Seconds(), fmt.Sprintf(format, a...))
	}
	ran := 0
	for _, e := range entries {
		if !e.IsDir() || (only != "" && e.Name() != only) {
			continue
		}
		name := e.Name()
		ran++
		t.Run(name, func(t *testing.T) {
			res := hpResult{Name: name, Control: control}
			defer func() {
				b, _ := json.MarshalIndent(res, "", " ")
				_ = os.WriteFile(filepath.Join(root, name, "result.json"), b, 0o644)
			}()
			var meta struct {
				Dir, Error string
				Vocab      int
				Steps      int
				Prompts    [][]int
				Conts      [][]int
			}
			raw, err := os.ReadFile(filepath.Join(root, name, "meta.json"))
			if err != nil {
				res.Error = "no meta.json: " + err.Error()
				t.Error(res.Error)
				return
			}
			if err := json.Unmarshal(raw, &meta); err != nil {
				res.Error = err.Error()
				t.Error(err)
				return
			}
			if meta.Error != "" {
				res.Error = "HF side failed: " + meta.Error
				t.Errorf("%s: %s", name, res.Error)
				return
			}
			if strings.HasPrefix(meta.Dir, "/srv/models") || strings.HasPrefix(meta.Dir, "/Volumes/") {
				t.Fatalf("%s is the archive (CLAUDE.md)", meta.Dir)
			}
			lraw, err := os.ReadFile(filepath.Join(root, name, "logits.f32"))
			if err != nil {
				res.Error = err.Error()
				t.Error(err)
				return
			}
			np, steps, V := len(meta.Prompts), meta.Steps, meta.Vocab
			if len(lraw) != np*steps*V*4 {
				res.Error = fmt.Sprintf("logits file is %d bytes, want %d x %d x %d x 4", len(lraw), np, steps, V)
				t.Error(res.Error)
				return
			}
			hf := make([]float32, np*steps*V)
			for i := range hf {
				hf[i] = math.Float32frombits(binary.LittleEndian.Uint32(lraw[4*i:]))
			}
			lraw = nil

			arm := func(embed4 bool, want string) (*hpArm, error) {
				t0 := time.Now()
				m, err := Load(meta.Dir, Options{Backend: "cpu", Quant: "int4", EmbedInt4: embed4})
				if err != nil {
					return nil, fmt.Errorf("load (EmbedInt4=%v): %w", embed4, err)
				}
				defer m.Close()
				a := &hpArm{EmbedInt4: embed4, HeadTable: m.HeadTable(), LoadSec: time.Since(t0).Seconds()}
				if a.HeadTable != want {
					return a, fmt.Errorf("VOID: EmbedInt4=%v loaded a %q head table, want %q (the flag did nothing)", embed4, a.HeadTable, want)
				}
				hb("%s: EmbedInt4=%v loaded in %.0fs, head table %s", name, embed4, a.LoadSec, a.HeadTable)
				t1 := time.Now()
				n := 0
				for p := range np {
					cache := m.NewCache(len(meta.Prompts[p]) + steps + 1)
					l, err := m.prefillLogits(context.Background(), meta.Prompts[p], cache)
					if err != nil {
						return a, err
					}
					kl, ag := make([]float64, steps), make([]bool, steps)
					for k := range steps {
						ref := hf[(p*steps+k)*V : (p*steps+k+1)*V]
						if len(l) != V {
							return a, fmt.Errorf("goinfer logits have %d entries, HF %d", len(l), V)
						}
						kl[k], ag[k] = klTo(ref, l), argmax(l) == argmax(ref)
						if k < steps-1 {
							if l, err = m.forward(meta.Conts[p][k], cache); err != nil {
								return a, err
							}
						}
						n++
					}
					a.KL, a.Agree = append(a.KL, kl), append(a.Agree, ag)
					hb("%s: EmbedInt4=%v prompt %d/%d (%.2fs per step so far)", name, embed4, p+1, np, time.Since(t1).Seconds()/float64(n))
				}
				a.SecPerStep = time.Since(t1).Seconds() / float64(n)
				return a, nil
			}
			wantB := "int4"
			if control {
				wantB = "int8"
			}
			if res.Int8Head, err = arm(false, "int8"); err != nil {
				res.Error = err.Error()
				t.Error(err)
				return
			}
			if res.Int4Head, err = arm(!control, wantB); err != nil {
				res.Error = err.Error()
				t.Error(err)
				return
			}
			hb("%s: done", name)
		})
	}
	if ran == 0 {
		t.Fatalf("no model directories under %s (GOINFER_HP_ONLY=%q)", root, only)
	}
}
