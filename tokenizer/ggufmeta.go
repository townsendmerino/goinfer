package tokenizer

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
)

// ggufMeta is the GGUF metadata the tokenizer reads, decoded to its own types while the key-value section is walked
// once: string arrays come out as []string, number arrays as []int or []float32, and every key the tokenizer does not
// read is skipped without being decoded. Its accessors answer as embed.GGUFFile's do over Metadata (Str, Uint, the
// array readers), which TestTK_typedReaderMatchesAikit checks key by key. The tensor directory is not read.
type ggufMeta struct {
	present map[string]bool // every wanted key seen, of any type
	strs    map[string]string
	ints    map[string]uint64 // any integer width, sign-extended as embed.GGUFFile.Uint does
	bools   map[string]bool
	strArrs map[string]ggufStrArr
	intArrs map[string][]int     // a non-integer element reads 0
	f32Arrs map[string][]float32 // a non-float element reads 0
}

// ggufStrArr is a wanted string array, or why it is not one.
type ggufStrArr struct {
	v   []string
	err string
}

// ggufMetaKeys are the keys the tokenizer reads.
var ggufMetaKeys = map[string]bool{
	ggufTokModel: true, ggufTokTokens: true, ggufTokMerges: true, ggufTokScores: true, ggufTokTypes: true,
	ggufTokPre: true, ggufTokChatTmpl: true, ggufTokBOS: true, ggufTokEOS: true, ggufTokUnk: true, ggufTokPad: true,
	ggufTokPrefix: true, "general.name": true,
}

// GGUF metadata value types.
const (
	gvU8 = iota
	gvI8
	gvU16
	gvI16
	gvU32
	gvI32
	gvF32
	gvBool
	gvString
	gvArray
	gvU64
	gvI64
	gvF64
)

const ggufMetaMaxDepth = 128 // aikit's nesting cap for arrays of arrays

// readGGUFMetaFile reads path's metadata from the head of the file.
func readGGUFMetaFile(path string) (*ggufMeta, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}
	return readGGUFMeta(f, fi.Size())
}

// readGGUFMeta walks a GGUF's header and key-value section from r, which holds size bytes.
func readGGUFMeta(r io.Reader, size int64) (*ggufMeta, error) {
	c := &gmCur{r: bufio.NewReaderSize(r, 64<<10), left: size}
	if c.u32() != 0x46554747 {
		if c.err != nil {
			return nil, c.err
		}
		return nil, errors.New("gguf: bad magic (not a GGUF file)")
	}
	if v := c.u32(); c.err == nil && v != 2 && v != 3 {
		return nil, fmt.Errorf("gguf: unsupported version %d (want 2 or 3)", v)
	}
	c.u64() // tensor count
	kvs := c.u64()
	m := &ggufMeta{present: map[string]bool{}, strs: map[string]string{}, ints: map[string]uint64{}, bools: map[string]bool{},
		strArrs: map[string]ggufStrArr{}, intArrs: map[string][]int{}, f32Arrs: map[string][]float32{}}
	for i := uint64(0); i < kvs && c.err == nil; i++ {
		key := c.str()
		vt := c.u32()
		if c.err != nil {
			break
		}
		if !ggufMetaKeys[key] {
			c.skip(vt, 0)
			continue
		}
		m.read(c, key, vt)
	}
	if c.err != nil {
		return nil, c.err
	}
	return m, nil
}

// read decodes a wanted key's value into the map its type answers from, clearing what an earlier duplicate of the key
// left (the last one wins, as in embed.GGUFFile.Metadata).
func (m *ggufMeta) read(c *gmCur, key string, vt uint32) {
	m.present[key] = true
	delete(m.strs, key)
	delete(m.ints, key)
	delete(m.bools, key)
	delete(m.strArrs, key)
	delete(m.intArrs, key)
	delete(m.f32Arrs, key)
	switch vt {
	case gvString:
		m.strs[key] = c.str()
	case gvBool:
		m.bools[key] = c.u8() != 0
	case gvU8, gvI8, gvU16, gvI16, gvU32, gvI32, gvU64, gvI64:
		m.ints[key] = c.int(vt)
	case gvF32:
		c.u32()
	case gvF64:
		c.u64()
	case gvArray:
		et, n := c.u32(), c.u64()
		if c.err == nil && n > uint64(c.left) {
			c.err = fmt.Errorf("gguf: array length %d exceeds %d remaining bytes", n, c.left)
		}
		if c.err != nil {
			return
		}
		m.readArray(c, key, et, n)
	default:
		c.err = fmt.Errorf("gguf: unknown metadata value type %d", vt)
	}
}

func (m *ggufMeta) readArray(c *gmCur, key string, et uint32, n uint64) {
	switch et {
	case gvString:
		out := make([]string, 0, min(n, uint64(c.left)/8))
		for i := uint64(0); i < n && c.err == nil; i++ {
			out = append(out, c.str())
		}
		m.strArrs[key] = ggufStrArr{v: out}
		return
	case gvU8, gvI8, gvU16, gvI16, gvU32, gvI32, gvU64, gvI64:
		out := make([]int, 0, min(n, uint64(c.left)))
		for i := uint64(0); i < n && c.err == nil; i++ {
			out = append(out, int(c.int(et)))
		}
		m.intArrs[key] = out
		m.f32Arrs[key] = make([]float32, len(out))
	case gvF32, gvF64:
		out := make([]float32, 0, min(n, uint64(c.left)/4))
		for i := uint64(0); i < n && c.err == nil; i++ {
			if et == gvF32 {
				out = append(out, math.Float32frombits(c.u32()))
			} else {
				out = append(out, float32(math.Float64frombits(c.u64())))
			}
		}
		m.f32Arrs[key] = out
		m.intArrs[key] = make([]int, len(out))
	default:
		for i := uint64(0); i < n && c.err == nil; i++ {
			c.skip(et, 1)
		}
		m.intArrs[key] = make([]int, n)
		m.f32Arrs[key] = make([]float32, n)
	}
	if n == 0 {
		m.strArrs[key] = ggufStrArr{v: []string{}}
		return
	}
	m.strArrs[key] = ggufStrArr{err: fmt.Sprintf("element type %d, want string", et)}
}

// Str is a string value.
func (m *ggufMeta) Str(key string) (string, bool) { v, ok := m.strs[key]; return v, ok }

// Uint is an integer value of any width.
func (m *ggufMeta) Uint(key string) (uint64, bool) { v, ok := m.ints[key]; return v, ok }

// stringArray is a string array, or an error naming why key is not one.
func (m *ggufMeta) stringArray(key string) ([]string, error) {
	a, ok := m.strArrs[key]
	switch {
	case !ok:
		return nil, fmt.Errorf("%s missing or not an array", key)
	case a.err != "":
		return nil, fmt.Errorf("%s: %s", key, a.err)
	}
	return a.v, nil
}

// intArray is an integer array (nil when key is not an array); an element that is not an integer reads 0.
func (m *ggufMeta) intArray(key string) []int {
	if v, ok := m.intArrs[key]; ok {
		return v
	}
	if a, ok := m.strArrs[key]; ok && a.err == "" {
		return make([]int, len(a.v))
	}
	return nil
}

// floatArray is a float array (nil when key is not an array); an element that is not a float reads 0.
func (m *ggufMeta) floatArray(key string) []float32 {
	if v, ok := m.f32Arrs[key]; ok {
		return v
	}
	if a, ok := m.strArrs[key]; ok && a.err == "" {
		return make([]float32, len(a.v))
	}
	return nil
}

// Bool is a bool value.
func (m *ggufMeta) Bool(key string) (bool, bool) { v, ok := m.bools[key]; return v, ok }

// gmCur reads GGUF values from a stream, tracking the bytes left so a length the input cannot hold is refused.
type gmCur struct {
	r    *bufio.Reader
	left int64
	err  error
	buf  [8]byte
}

func (c *gmCur) take(n int) []byte {
	if c.err != nil {
		return nil
	}
	if int64(n) > c.left {
		c.err = io.ErrUnexpectedEOF
		return nil
	}
	if _, err := io.ReadFull(c.r, c.buf[:n]); err != nil {
		c.err = io.ErrUnexpectedEOF
		return nil
	}
	c.left -= int64(n)
	return c.buf[:n]
}

func (c *gmCur) u8() uint8 {
	if b := c.take(1); b != nil {
		return b[0]
	}
	return 0
}

func (c *gmCur) u16() uint16 {
	if b := c.take(2); b != nil {
		return binary.LittleEndian.Uint16(b)
	}
	return 0
}

func (c *gmCur) u32() uint32 {
	if b := c.take(4); b != nil {
		return binary.LittleEndian.Uint32(b)
	}
	return 0
}

func (c *gmCur) u64() uint64 {
	if b := c.take(8); b != nil {
		return binary.LittleEndian.Uint64(b)
	}
	return 0
}

// int reads an integer of type vt, sign-extending the signed ones.
func (c *gmCur) int(vt uint32) uint64 {
	switch vt {
	case gvU8:
		return uint64(c.u8())
	case gvI8:
		return uint64(int8(c.u8()))
	case gvU16:
		return uint64(c.u16())
	case gvI16:
		return uint64(int16(c.u16()))
	case gvU32:
		return uint64(c.u32())
	case gvI32:
		return uint64(int32(c.u32()))
	case gvI64:
		return uint64(int64(c.u64()))
	}
	return c.u64()
}

// str reads a length-prefixed string with one allocation: the string itself.
func (c *gmCur) str() string {
	n := c.u64()
	if c.err != nil {
		return ""
	}
	if n > uint64(c.left) {
		c.err = fmt.Errorf("gguf: string length %d exceeds %d remaining bytes", n, c.left)
		return ""
	}
	if n <= uint64(c.r.Size()) {
		b, err := c.r.Peek(int(n))
		if err != nil {
			c.err = io.ErrUnexpectedEOF
			return ""
		}
		s := string(b)
		_, _ = c.r.Discard(int(n))
		c.left -= int64(n)
		return s
	}
	var sb bytes.Buffer
	if _, err := io.CopyN(&sb, c.r, int64(n)); err != nil {
		c.err = io.ErrUnexpectedEOF
		return ""
	}
	c.left -= int64(n)
	return sb.String()
}

// skip reads past a value of type vt without decoding it.
func (c *gmCur) skip(vt uint32, depth int) {
	if c.err != nil {
		return
	}
	var n int64
	switch vt {
	case gvU8, gvI8, gvBool:
		n = 1
	case gvU16, gvI16:
		n = 2
	case gvU32, gvI32, gvF32:
		n = 4
	case gvU64, gvI64, gvF64:
		n = 8
	case gvString:
		l := c.u64()
		if c.err == nil && l > uint64(c.left) {
			c.err = fmt.Errorf("gguf: string length %d exceeds %d remaining bytes", l, c.left)
			return
		}
		n = int64(l)
	case gvArray:
		if depth >= ggufMetaMaxDepth {
			c.err = fmt.Errorf("gguf: metadata array nesting exceeds %d levels", ggufMetaMaxDepth)
			return
		}
		et, cnt := c.u32(), c.u64()
		if c.err == nil && cnt > uint64(c.left) {
			c.err = fmt.Errorf("gguf: array length %d exceeds %d remaining bytes", cnt, c.left)
		}
		for i := uint64(0); i < cnt && c.err == nil; i++ {
			c.skip(et, depth+1)
		}
		return
	default:
		c.err = fmt.Errorf("gguf: unknown metadata value type %d", vt)
		return
	}
	if c.err != nil {
		return
	}
	if n > c.left {
		c.err = io.ErrUnexpectedEOF
		return
	}
	if _, err := c.r.Discard(int(n)); err != nil {
		c.err = io.ErrUnexpectedEOF
		return
	}
	c.left -= n
}
