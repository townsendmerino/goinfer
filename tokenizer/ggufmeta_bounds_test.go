package tokenizer

import (
	"bytes"
	"encoding/binary"
	"runtime"
	"strings"
	"testing"
)

func TestReadGGUFMeta_unknownArrayElemTypeDoesNotOOM(t *testing.T) {
	var buf bytes.Buffer
	w32 := func(v uint32) { _ = binary.Write(&buf, binary.LittleEndian, v) }
	w64 := func(v uint64) { _ = binary.Write(&buf, binary.LittleEndian, v) }
	w32(0x46554747) // GGUF magic
	w32(3)          // version
	w64(0)          // tensor count
	w64(1)          // 1 kv
	key := ggufTokTokens
	w64(uint64(len(key)))
	buf.WriteString(key)
	w32(gvArray)
	w32(999)     // unknown element type
	w64(1 << 20) // 1M elements would allocate ~12 MB of int/f32 slices if readArray didn't stop on c.err
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	_, err := readGGUFMeta(bytes.NewReader(buf.Bytes()), 1<<31)
	runtime.ReadMemStats(&after)
	if err == nil || !strings.Contains(err.Error(), "unknown metadata value type") {
		t.Fatalf("err = %v, want unknown metadata value type error", err)
	}
	if alloc := after.TotalAlloc - before.TotalAlloc; alloc > 1<<20 {
		t.Fatalf("unknown array element allocated %d bytes (should return before slice allocs)", alloc)
	}
}

func TestReadGGUFMeta_oversizedArrayAndStringRejected(t *testing.T) {
	t.Run("array_ceiling", func(t *testing.T) {
		var buf bytes.Buffer
		w32 := func(v uint32) { _ = binary.Write(&buf, binary.LittleEndian, v) }
		w64 := func(v uint64) { _ = binary.Write(&buf, binary.LittleEndian, v) }
		w32(0x46554747)
		w32(3)
		w64(0)
		w64(1)
		key := ggufTokTokens
		w64(uint64(len(key)))
		buf.WriteString(key)
		w32(gvArray)
		w32(gvU8)
		w64(ggufMetaMaxArrayLen + 1)
		_, err := readGGUFMeta(bytes.NewReader(buf.Bytes()), 1<<31)
		if err == nil || !strings.Contains(err.Error(), "ceiling") {
			t.Fatalf("err = %v, want ceiling error", err)
		}
	})
	t.Run("string_ceiling", func(t *testing.T) {
		var buf bytes.Buffer
		w32 := func(v uint32) { _ = binary.Write(&buf, binary.LittleEndian, v) }
		w64 := func(v uint64) { _ = binary.Write(&buf, binary.LittleEndian, v) }
		w32(0x46554747)
		w32(3)
		w64(0)
		w64(1)
		key := ggufTokModel
		w64(uint64(len(key)))
		buf.WriteString(key)
		w32(gvString)
		w64(ggufMetaMaxStringBytes + 1)
		_, err := readGGUFMeta(bytes.NewReader(buf.Bytes()), 1<<31)
		if err == nil || !strings.Contains(err.Error(), "ceiling") {
			t.Fatalf("err = %v, want ceiling error for oversized string", err)
		}
	})
}
