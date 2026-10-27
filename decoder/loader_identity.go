package decoder

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"hash"
	"math"
	"reflect"
	"sort"
	"unsafe"

	"github.com/townsendmerino/aikit/linalg"
)

// tensorIdentity is one tensor of a loaded *Weights as the model receives it: SB4's loader identity
// (docs/tasks/task-subsystems-2026-10.md §8 SB4). Path names the field ("Layers[3].QProj", "Layers[0].Experts[2].Up");
// Layout is the WeightMat's Kind, the element type of a plain slice, or "scalar"; Shape is "rows x cols" or the length;
// Hash covers every byte and field of the tensor, so two tensors with equal identities hand the model the same values in
// the same layout.
type tensorIdentity struct {
	Path, Layout, Shape, Hash string
}

func (t tensorIdentity) String() string {
	return t.Path + "\t" + t.Layout + "\t" + t.Shape + "\t" + t.Hash
}

// loaderSkip is the *Weights fields the identity leaves out: the config and architecture (VK4's program fingerprint covers
// what they build), the source handles, profile and lookup tables that reach the model only through the tensors the
// walk already hashes, and bakedQuant, the label a .giw load carries and a direct load does not.
var loaderSkip = map[string]bool{
	"prof": true, "Cfg": true, "arch": true, "st": true, "backing": true, "schema": true,
	"kquant": true, "ggufPath": true, "kquantFile": true, "bakedQuant": true,
}

var weightMatType = reflect.TypeOf(linalg.WeightMat{})

// loaderIdentity lists w's tensors, sorted by path. A nil slice, a zero WeightMat and a nil pointer are absent, not
// listed, so a field the loader never fills costs nothing and a newly filled one appears. The walk reads unexported
// fields, aikit's WeightMat's among them, by reflection; a pointer, map, func or interface it was not told about panics,
// so a new kind of field is a change this function must learn, not one it silently skips.
func loaderIdentity(w *Weights) []tensorIdentity {
	var out []tensorIdentity
	v := reflect.ValueOf(w).Elem()
	t := v.Type()
	for i := range v.NumField() {
		if loaderSkip[t.Field(i).Name] {
			continue
		}
		identWalk(&out, t.Field(i).Name, v.Field(i))
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Path < out[j].Path })
	return out
}

// loaderIdentityDigest is a SHA-256 over the identity lines, one per tensor in path order.
func loaderIdentityDigest(ids []tensorIdentity) string {
	h := sha256.New()
	for _, id := range ids {
		h.Write([]byte(id.String() + "\n"))
	}
	return "sha256:" + hex.EncodeToString(h.Sum(nil))
}

func identWalk(out *[]tensorIdentity, path string, v reflect.Value) {
	switch {
	case v.Type() == weightMatType:
		if v.IsZero() {
			return
		}
		h := sha256.New()
		identBytes(h, v)
		m := (*linalg.WeightMat)(unsafe.Pointer(v.UnsafeAddr()))
		*out = append(*out, tensorIdentity{path, m.Kind(), fmt.Sprintf("%dx%d", m.Rows(), m.Cols()), hashString(h)})
		return
	}
	switch v.Kind() {
	case reflect.Pointer:
		if v.IsNil() {
			return
		}
		if v.Elem().Kind() != reflect.Struct {
			panic(fmt.Sprintf("loaderIdentity: %s: pointer to %s", path, v.Elem().Kind()))
		}
		identWalk(out, path, v.Elem())
	case reflect.Struct:
		t := v.Type()
		for i := range v.NumField() {
			identWalk(out, path+"."+t.Field(i).Name, v.Field(i))
		}
	case reflect.Slice, reflect.Array:
		if v.Kind() == reflect.Slice && v.IsNil() {
			return
		}
		if isNumeric(v.Type().Elem().Kind()) {
			h := sha256.New()
			identBytes(h, v)
			*out = append(*out, tensorIdentity{path, v.Type().Elem().Kind().String(), fmt.Sprint(v.Len()), hashString(h)})
			return
		}
		for i := range v.Len() {
			identWalk(out, fmt.Sprintf("%s[%d]", path, i), v.Index(i))
		}
	case reflect.Bool, reflect.String, reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Float32, reflect.Float64:
		if v.IsZero() {
			return
		}
		h := sha256.New()
		identBytes(h, v)
		*out = append(*out, tensorIdentity{path, "scalar", v.Kind().String(), hashString(h)})
	default:
		panic(fmt.Sprintf("loaderIdentity: %s: unsupported kind %s", path, v.Kind()))
	}
}

func isNumeric(k reflect.Kind) bool {
	switch k {
	case reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64, reflect.Uint8, reflect.Uint16, reflect.Uint32,
		reflect.Uint64, reflect.Float32, reflect.Float64:
		return true
	}
	return false
}

// identBytes writes v to h: a numeric slice as its length and raw little-endian bytes, a struct as each field's name
// and value, a scalar by its bits. Only the identity of a WeightMat walks a struct here, and a WeightMat holds no
// pointer, map or func; one appearing panics.
func identBytes(h hash.Hash, v reflect.Value) {
	var b [8]byte
	u64 := func(x uint64) {
		binary.LittleEndian.PutUint64(b[:], x)
		h.Write(b[:])
	}
	switch v.Kind() {
	case reflect.Struct:
		t := v.Type()
		for i := range v.NumField() {
			h.Write([]byte(t.Field(i).Name + "="))
			identBytes(h, v.Field(i))
		}
	case reflect.Slice:
		if v.IsNil() {
			h.Write([]byte{'n'})
			return
		}
		if !isNumeric(v.Type().Elem().Kind()) {
			panic(fmt.Sprintf("loaderIdentity: slice of %s", v.Type().Elem().Kind()))
		}
		u64(uint64(v.Len()))
		if v.Len() > 0 {
			h.Write(unsafe.Slice((*byte)(v.UnsafePointer()), v.Len()*int(v.Type().Elem().Size())))
		}
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
		panic(fmt.Sprintf("loaderIdentity: unsupported kind %s inside a tensor", v.Kind()))
	}
}

func hashString(h hash.Hash) string { return hex.EncodeToString(h.Sum(nil)) }

// loaderIdentityDiff names the paths whose identity differs between a and b, each present in only one included.
func loaderIdentityDiff(a, b []tensorIdentity) []string {
	am := map[string]string{}
	for _, t := range a {
		am[t.Path] = t.String()
	}
	seen := map[string]bool{}
	var out []string
	for _, t := range b {
		seen[t.Path] = true
		if am[t.Path] != t.String() {
			out = append(out, t.Path)
		}
	}
	for p := range am {
		if !seen[p] {
			out = append(out, p)
		}
	}
	sort.Strings(out)
	return out
}
