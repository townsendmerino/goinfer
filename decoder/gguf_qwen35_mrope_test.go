package decoder

import (
	"testing"

	"github.com/townsendmerino/aikit/embed"
)

func kvI32Arr(k string, vs []int32) ggufKV {
	b := append(gU32(5), gU64(uint64(len(vs)))...) // 5 = int32
	for _, v := range vs {
		b = append(b, gU32(uint32(v))...)
	}
	return ggufKV{k, gtArray, b}
}

// TestGGUFMRopeJSON: a Qwen3.5 GGUF's rope.dimension_sections ([11,11,10,0], llama.cpp's four) becomes the
// safetensors config's mrope_section [11,11,10] with mrope_interleaved, so a GGUF text model can take an image turn
// (docs/multimodal.md F5); and a missing key, a nonzero fourth section, a negative one, or all zero leaves it out, so
// such a model stays text-only and an image turn is refused by name (parseMRopeFlat then returns no section).
func TestGGUFMRopeJSON(t *testing.T) {
	cases := []struct {
		kvs  []ggufKV
		want string
	}{
		{[]ggufKV{kvI32Arr("qwen35.rope.dimension_sections", []int32{11, 11, 10, 0})}, `,"mrope_section":[11,11,10],"mrope_interleaved":true`},
		{nil, ""},
		{[]ggufKV{kvI32Arr("qwen35.rope.dimension_sections", []int32{11, 11, 10, 2})}, ""},
		{[]ggufKV{kvI32Arr("qwen35.rope.dimension_sections", []int32{11, -1, 10, 0})}, ""},
		{[]ggufKV{kvI32Arr("qwen35.rope.dimension_sections", []int32{0, 0, 0, 0})}, ""},
		{[]ggufKV{kvI32Arr("qwen35.rope.dimension_sections", []int32{11, 11, 10})}, ""},
	}
	for i, c := range cases {
		g, err := embed.OpenGGUFBytes(buildGGUF(c.kvs, nil))
		if err != nil {
			t.Fatal(err)
		}
		if got := ggufMRopeJSON(g, "qwen35"); got != c.want {
			t.Errorf("case %d: %q, want %q", i, got, c.want)
		}
	}
	sec, inter := parseMRopeFlat([]byte(`{"rope_type":"default","rope_theta":1e7,"partial_rotary_factor":0.25,"mrope_section":[11,11,10],"mrope_interleaved":true}`))
	if len(sec) != 3 || sec[0] != 11 || sec[2] != 10 || !inter {
		t.Errorf("parseMRopeFlat of the synthesized form: %v, %v", sec, inter)
	}
}
