package kernelprint

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"sort"
)

// ggmlType is one GGUF tensor type the suite dequantizes: its id, the elements and bytes of one block, and the byte
// offsets of the block's binary16 scale fields (set to sane values so a block decodes to finite numbers).
type ggmlType struct {
	name        string
	id          uint32
	elems, size int
	f16At       []int
	e8m0At      int // MXFP4's one-byte exponent scale; -1 when none
}

// ggmlTypes are the quantized types embed's GGUF reader dequantizes (embed/gguf.go:ggmlBlockElems).
var ggmlTypes = []ggmlType{
	{"Q8_0", 8, 32, 34, []int{0}, -1},
	{"Q4_0", 2, 32, 18, []int{0}, -1},
	{"Q5_0", 6, 32, 22, []int{0}, -1},
	{"IQ4_NL", 20, 32, 18, []int{0}, -1},
	{"MXFP4", 39, 32, 17, nil, 0},
	{"Q2_K", 10, 256, 84, []int{80, 82}, -1},
	{"Q3_K", 11, 256, 110, []int{108}, -1},
	{"Q4_K", 12, 256, 144, []int{0, 2}, -1},
	{"Q5_K", 13, 256, 176, []int{0, 2}, -1},
	{"Q6_K", 14, 256, 210, []int{208}, -1},
	{"IQ4_XS", 23, 256, 136, []int{0}, -1},
	{"IQ2_S", 22, 256, 82, []int{0}, -1},
	{"IQ3_S", 21, 256, 110, []int{0}, -1},
}

// blocks returns nBlocks random blocks of t with their scales set to binary16 values in [2^-10, 2^-3).
func (g *gen) blocks(t ggmlType, nBlocks int) []byte {
	raw := g.bytes(nBlocks * t.size)
	for b := range nBlocks {
		base := b * t.size
		for _, off := range t.f16At {
			binary.LittleEndian.PutUint16(raw[base+off:], g.half16(1, 5, 11)[0]&0x7fff)
		}
		if t.e8m0At >= 0 {
			raw[base+t.e8m0At] = byte(118 + g.next()%12)
		}
	}
	return raw
}

// ggufTensor is one tensor of a GGUF file: dims in GGUF order (dims[0] innermost).
type ggufTensor struct {
	name string
	dims []uint64
	typ  uint32
	data []byte
}

// buildGGUF writes a GGUF v3 file with no metadata and the given tensors, each at a 32-byte aligned offset.
func buildGGUF(ts []ggufTensor) []byte {
	var b []byte
	u32 := func(v uint32) { b = binary.LittleEndian.AppendUint32(b, v) }
	u64 := func(v uint64) { b = binary.LittleEndian.AppendUint64(b, v) }
	u32(0x46554747) // "GGUF"
	u32(3)
	u64(uint64(len(ts)))
	u64(0)
	var off uint64
	offs := make([]uint64, len(ts))
	for i, t := range ts {
		u64(uint64(len(t.name)))
		b = append(b, t.name...)
		u32(uint32(len(t.dims)))
		for _, d := range t.dims {
			u64(d)
		}
		u32(t.typ)
		offs[i] = off
		u64(off)
		off += uint64(len(t.data))
		off = (off + 31) &^ 31
	}
	for len(b)%32 != 0 {
		b = append(b, 0)
	}
	start := len(b)
	for i, t := range ts {
		for uint64(len(b)-start) < offs[i] {
			b = append(b, 0)
		}
		b = append(b, t.data...)
	}
	return b
}

// stTensor is one tensor of a safetensors file.
type stTensor struct {
	name, dtype string
	shape       []int
	data        []byte
}

// buildSafetensors writes a safetensors file holding ts, in name order.
func buildSafetensors(ts []stTensor) []byte {
	sort.Slice(ts, func(i, j int) bool { return ts[i].name < ts[j].name })
	type entry struct {
		DType   string `json:"dtype"`
		Shape   []int  `json:"shape"`
		Offsets [2]int `json:"data_offsets"`
	}
	hdr := map[string]entry{}
	var data []byte
	for _, t := range ts {
		hdr[t.name] = entry{t.dtype, t.shape, [2]int{len(data), len(data) + len(t.data)}}
		data = append(data, t.data...)
	}
	h, err := json.Marshal(hdr)
	if err != nil {
		panic(fmt.Sprintf("kernelprint: safetensors header: %v", err))
	}
	for len(h)%8 != 0 {
		h = append(h, ' ')
	}
	out := binary.LittleEndian.AppendUint64(nil, uint64(len(h)))
	out = append(out, h...)
	return append(out, data...)
}

func u16bytes(u []uint16) []byte {
	b := make([]byte, 2*len(u))
	for i, v := range u {
		binary.LittleEndian.PutUint16(b[2*i:], v)
	}
	return b
}
