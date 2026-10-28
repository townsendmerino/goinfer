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
// Layout is the WeightMat's Kind, the element type of a plain slice, "scalar", or "len" for the length of a slice whose
// elements are walked one by one; Shape is "rows x cols" or the length; Hash covers every byte and field of the tensor,
// so two tensors with equal identities hand the model the same values in the same layout.
type tensorIdentity struct {
	Path, Layout, Shape, Hash string
}

func (t tensorIdentity) String() string {
	return t.Path + "\t" + t.Layout + "\t" + t.Shape + "\t" + t.Hash
}

// loaderSkip is the *Weights fields the identity leaves out: the config and architecture (VK4's program fingerprint covers
// what they build), the source handles and profile, and bakedQuant, the label a .giw load carries and a direct load does
// not. kquant is not walked as a field: each entry is hashed into the identity of the WeightMat whose int8 copy it keys.
var loaderSkip = map[string]bool{
	"prof": true, "Cfg": true, "arch": true, "st": true, "backing": true, "schema": true,
	"kquant": true, "ggufPath": true, "kquantFile": true, "bakedQuant": true,
}

var weightMatType = reflect.TypeOf(linalg.WeightMat{})

// identWalker accumulates the identities of one walk. kquant is the walked Weights' k-quant source table, keyed by the
// int8 copy's data pointer; matched counts the entries a WeightMat claimed.
type identWalker struct {
	out     []tensorIdentity
	kquant  map[*int8]kquantSrc
	matched int
}

// loaderIdentity lists w's tensors, sorted by path. A nil slice, a zero WeightMat, a nil pointer and a zero scalar are
// absent, not listed, so a field the loader never fills costs nothing and a newly filled one appears; a slice whose
// elements are walked one by one also lists its length. The walk reads unexported fields, aikit's WeightMat's among them,
// by reflection. Before it reads a value it checks the static type of every field it will walk, so a map, func,
// interface or other kind it was not told about panics even where the fixtures leave it nil.
func loaderIdentity(w *Weights) []tensorIdentity {
	v := reflect.ValueOf(w).Elem()
	t := v.Type()
	iw := &identWalker{kquant: w.kquant}
	for i := range v.NumField() {
		if loaderSkip[t.Field(i).Name] {
			continue
		}
		checkIdentType(t.Field(i).Name, t.Field(i).Type, map[reflect.Type]bool{})
		iw.walk(t.Field(i).Name, v.Field(i))
	}
	if n := len(w.kquant) - iw.matched; n != 0 {
		iw.out = append(iw.out, tensorIdentity{"kquant", "unmatched", fmt.Sprint(n), ""})
	}
	sort.Slice(iw.out, func(i, j int) bool { return iw.out[i].Path < iw.out[j].Path })
	return iw.out
}

// checkIdentType panics on any kind the walk cannot hash, anywhere in t: a map, func, interface, channel or unsafe
// pointer, or a pointer to anything but a struct. A WeightMat is checked too, since identBytes hashes all its fields.
func checkIdentType(path string, t reflect.Type, seen map[reflect.Type]bool) {
	if seen[t] {
		return
	}
	seen[t] = true
	switch t.Kind() {
	case reflect.Pointer:
		if t.Elem().Kind() != reflect.Struct {
			panic(fmt.Sprintf("loaderIdentity: %s: pointer to %s", path, t.Elem().Kind()))
		}
		checkIdentType(path, t.Elem(), seen)
	case reflect.Struct:
		for i := range t.NumField() {
			checkIdentType(path+"."+t.Field(i).Name, t.Field(i).Type, seen)
		}
	case reflect.Slice, reflect.Array:
		checkIdentType(path+"[]", t.Elem(), seen)
	case reflect.Bool, reflect.String, reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Float32, reflect.Float64:
	default:
		panic(fmt.Sprintf("loaderIdentity: %s: unsupported kind %s", path, t.Kind()))
	}
}

// loaderIdentityDigest is a SHA-256 over the identity lines, one per tensor in path order.
func loaderIdentityDigest(ids []tensorIdentity) string {
	h := sha256.New()
	for _, id := range ids {
		h.Write([]byte(id.String() + "\n"))
	}
	return "sha256:" + hex.EncodeToString(h.Sum(nil))
}

func (iw *identWalker) walk(path string, v reflect.Value) {
	if v.Type() == weightMatType {
		if v.IsZero() {
			return
		}
		h := sha256.New()
		identBytes(h, v)
		if q8 := v.FieldByName("q8"); q8.Len() > 0 {
			if src, ok := iw.kquant[(*int8)(q8.UnsafePointer())]; ok {
				h.Write([]byte("kquant="))
				identBytes(h, reflect.ValueOf(src))
				iw.matched++
			}
		}
		m := (*linalg.WeightMat)(unsafe.Pointer(v.UnsafeAddr()))
		iw.add(path, m.Kind(), fmt.Sprintf("%dx%d", m.Rows(), m.Cols()), h)
		return
	}
	switch v.Kind() {
	case reflect.Pointer:
		if !v.IsNil() {
			iw.walk(path, v.Elem())
		}
	case reflect.Struct:
		t := v.Type()
		for i := range v.NumField() {
			iw.walk(path+"."+t.Field(i).Name, v.Field(i))
		}
	case reflect.Slice, reflect.Array:
		if v.Kind() == reflect.Slice && v.IsNil() {
			return
		}
		if isLeafElem(v.Type().Elem().Kind()) {
			h := sha256.New()
			identBytes(h, v)
			iw.add(path, v.Type().Elem().Kind().String(), fmt.Sprint(v.Len()), h)
			return
		}
		iw.out = append(iw.out, tensorIdentity{path, "len", fmt.Sprint(v.Len()), ""})
		for i := range v.Len() {
			iw.walk(fmt.Sprintf("%s[%d]", path, i), v.Index(i))
		}
	default: // a scalar: checkIdentType refused every other kind
		if v.IsZero() {
			return
		}
		h := sha256.New()
		identBytes(h, v)
		iw.add(path, "scalar", v.Kind().String(), h)
	}
}

func (iw *identWalker) add(path, layout, shape string, h hash.Hash) {
	iw.out = append(iw.out, tensorIdentity{path, layout, shape, hex.EncodeToString(h.Sum(nil))})
}

// isLeafElem reports whether a slice or array of k is hashed as one tensor: numbers by their raw bytes, bools and strings
// element by element.
func isLeafElem(k reflect.Kind) bool {
	return isNumeric(k) || k == reflect.Bool || k == reflect.String
}

func isNumeric(k reflect.Kind) bool {
	switch k {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64, reflect.Uint, reflect.Uint8,
		reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Float32, reflect.Float64:
		return true
	}
	return false
}

// identBytes writes v to h: a slice or array as its length then its elements (numbers as raw little-endian bytes in one
// write), a struct as each field's name and value, a scalar by its bits. A nil slice writes 'n', so it differs from an
// empty one. A kind checkIdentType refuses panics.
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
	case reflect.Slice, reflect.Array:
		if v.Kind() == reflect.Slice && v.IsNil() {
			h.Write([]byte{'n'})
			return
		}
		u64(uint64(v.Len()))
		if v.Len() == 0 {
			return
		}
		if isNumeric(v.Type().Elem().Kind()) && v.Kind() == reflect.Slice {
			h.Write(unsafe.Slice((*byte)(v.UnsafePointer()), v.Len()*int(v.Type().Elem().Size())))
			return
		}
		for i := range v.Len() {
			identBytes(h, v.Index(i))
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
