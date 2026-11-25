package tokenizer

import (
	"bytes"
	"encoding/binary"
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
	w64(1 << 20) // huge length, smaller than declared fake size
	// Pass a huge declared file size (like a multi-GB model) so n <= c.left passes,
	// forcing readArray's default branch to hit c.skip(999, 1) error on element 0.
	if _, err := readGGUFMeta(bytes.NewReader(buf.Bytes()), 1<<31); err == nil {
		t.Fatal("expected error for unknown GGUF array element type")
	}
}

func TestReadGGUFMeta_oversizedArrayAndStringRejected(t *testing.T) {
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
	if _, err := readGGUFMeta(bytes.NewReader(buf.Bytes()), 1<<31); err == nil {
		t.Fatal("expected error for oversized GGUF metadata array")
	}
}
