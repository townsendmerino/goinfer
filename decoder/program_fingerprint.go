package decoder

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"hash"
	"math"
	"os"
	"path/filepath"
	"reflect"
)

// programFingerprint is a SHA-256 over p's operations and parameters: VK4's program fingerprint
// (docs/tasks/task-subsystems-2026-10.md §5.7). Two programs with the same fingerprint run the same operations with the
// same parameters. The encoding is a reflection walk that writes each struct field's name and value (floats by their
// bits, slices with their length) and no type names, so renaming a type leaves it unchanged and renaming or adding a
// field changes it.
func programFingerprint(p *layerProgram) string {
	h := sha256.New()
	writeValue(h, reflect.ValueOf(*p))
	return "sha256:" + hex.EncodeToString(h.Sum(nil))
}

func writeValue(h hash.Hash, v reflect.Value) {
	var b [8]byte
	u64 := func(x uint64) {
		binary.LittleEndian.PutUint64(b[:], x)
		h.Write(b[:])
	}
	switch v.Kind() {
	case reflect.Struct:
		t := v.Type()
		h.Write([]byte{'{'})
		for i := range v.NumField() {
			h.Write([]byte(t.Field(i).Name))
			h.Write([]byte{'='})
			writeValue(h, v.Field(i))
		}
		h.Write([]byte{'}'})
	case reflect.Slice, reflect.Array:
		if v.Kind() == reflect.Slice && v.IsNil() {
			h.Write([]byte{'n'})
			return
		}
		h.Write([]byte{'['})
		u64(uint64(v.Len()))
		for i := range v.Len() {
			writeValue(h, v.Index(i))
		}
		h.Write([]byte{']'})
	case reflect.Bool:
		if v.Bool() {
			h.Write([]byte{'t'})
		} else {
			h.Write([]byte{'f'})
		}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		u64(uint64(v.Int()))
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		u64(v.Uint())
	case reflect.Float32:
		u64(uint64(math.Float32bits(float32(v.Float()))))
	case reflect.Float64:
		u64(math.Float64bits(v.Float()))
	case reflect.String:
		u64(uint64(v.Len()))
		h.Write([]byte(v.String()))
	default:
		// A program holds no pointers, maps, funcs or interfaces; one appearing is a change this encoding must learn.
		panic(fmt.Sprintf("programFingerprint: unsupported kind %s", v.Kind()))
	}
}

// configExperts is the experts predicate a config alone gives: Llama 4's per-layer MoE flags, else the layers from
// FirstKDense on of an architecture with MoE. It matches what the loaders build (Weights.hasExperts); TestProgramFingerprint_configMatchesWeights
// holds that on every fixture present.
func configExperts(a *Architecture) func(int) bool {
	if a.llama4 != nil {
		return func(l int) bool { return l < len(a.llama4.isMoE) && a.llama4.isMoE[l] }
	}
	return func(l int) bool { return a.MoE != nil && l >= a.FirstKDense }
}

// programForConfigDir builds the program of the config.json in dir, from the config alone. The loader decides the head's tie
// from whether the checkpoint has an lm_head tensor; from a config the best reading is tie_word_embeddings (top level,
// or under text_config), and where it is absent the adapter's default stands.
func programForConfigDir(dir string) (*layerProgram, error) {
	cfg, err := loadConfig(os.DirFS(dir), "config.json")
	if err != nil {
		return nil, err
	}
	arch, _, err := resolveArchitecture(cfg)
	if err != nil {
		return nil, err
	}
	if tied, ok := configTie(dir); ok {
		arch.TiedLMHead = tied
	}
	return buildProgram(arch, configExperts(arch)), nil
}

// configTie reads tie_word_embeddings from the config.json in dir.
func configTie(dir string) (tied, ok bool) {
	b, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return false, false
	}
	var c struct {
		Tie  *bool `json:"tie_word_embeddings"`
		Text struct {
			Tie *bool `json:"tie_word_embeddings"`
		} `json:"text_config"`
	}
	if json.Unmarshal(lenientConfigJSON(b), &c) != nil {
		return false, false
	}
	switch {
	case c.Tie != nil:
		return *c.Tie, true
	case c.Text.Tie != nil:
		return *c.Text.Tie, true
	}
	return false, false
}
