package vision

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/townsendmerino/aikit/embed"
)

// A Qwen3.5+ vision tower from a llama.cpp GGUF mmproj (clip architecture, projector type qwen3vl_merger): the layout
// unsloth's mmproj-*.gguf files and Ollama's qwen3.5 / qwen3.6 projector blobs both ship (read 2026-10-06, goinfer's
// docs/multimodal.md P8b). The tensors are HF's under llama.cpp names, with ggml's innermost-first dimension order,
// except the patch embedding: HF's Conv3d kernel [hidden, 3, 2, 16, 16] is split along its temporal axis into two
// tensors (v.patch_embd.weight for t=0, .weight.1 for t=1) and recombined here. The config comes from the clip.*
// metadata. The merger runs HF's erf GELU as the safetensors tower does (llama.cpp's own graph runs tanh there; the
// weights were trained against erf).

// LoadQwen3VisionEncoderMMProj loads a Qwen3.5+ tower from a GGUF mmproj file. It refuses another projector type, a
// DeepStack layer, a layer-norm epsilon the tower does not run, and a patch embedding that is not split in two.
func LoadQwen3VisionEncoderMMProj(path string, quant bool) (*Qwen3VisionEncoder, error) {
	g, err := embed.OpenGGUFMmap(path)
	if err != nil {
		return nil, fmt.Errorf("vision: open mmproj: %w", err)
	}
	defer g.Close()
	cfg, err := qwen3MMProjConfig(g)
	if err != nil {
		return nil, fmt.Errorf("vision: %s: %w", path, err)
	}
	enc, err := LoadQwen3VisionEncoderFrom(cfg, mmprojSource{g}, quant)
	if err != nil {
		return nil, fmt.Errorf("vision: %s: %w", path, err)
	}
	enc.rel = newBlockRelease(func() (*Qwen3VisionEncoder, error) { return LoadQwen3VisionEncoderMMProj(path, quant) })
	return enc, nil
}

// ReadQwen3MMProjConfig reads a GGUF mmproj's tower config without loading the weights, with
// LoadQwen3VisionEncoderMMProj's refusals (another projector type, DeepStack, an unsupported epsilon, an unsplit patch
// embedding), so a server can check it against its text model at startup.
func ReadQwen3MMProjConfig(path string) (Qwen3EncoderConfig, error) {
	g, err := embed.OpenGGUFMmap(path)
	if err != nil {
		return Qwen3EncoderConfig{}, fmt.Errorf("vision: open mmproj: %w", err)
	}
	defer g.Close()
	cfg, err := qwen3MMProjConfig(g)
	if err != nil {
		return cfg, fmt.Errorf("vision: %s: %w", path, err)
	}
	return cfg, nil
}

// qwen3MMProjConfig reads the tower's config from an mmproj's metadata.
func qwen3MMProjConfig(g *embed.GGUFFile) (Qwen3EncoderConfig, error) {
	var c Qwen3EncoderConfig
	if a, _ := g.Str("general.architecture"); a != "clip" {
		return c, fmt.Errorf("not a clip mmproj (general.architecture %q)", a)
	}
	if p, _ := g.Str("clip.projector_type"); p != "qwen3vl_merger" {
		return c, fmt.Errorf("projector type %q: only qwen3vl_merger (Qwen3.5+) is supported", p)
	}
	var missing error
	u := func(k string) int {
		v, ok := g.Uint(k)
		if !ok && missing == nil {
			missing = fmt.Errorf("mmproj has no %s", k)
		}
		return int(v)
	}
	c.Depth = u("clip.vision.block_count")
	c.HiddenSize = u("clip.vision.embedding_length")
	c.IntermediateSize = u("clip.vision.feed_forward_length")
	c.NumHeads = u("clip.vision.attention.head_count")
	c.PatchSize = u("clip.vision.patch_size")
	c.SpatialMergeSize = u("clip.vision.spatial_merge_size")
	c.OutHiddenSize = u("clip.vision.projection_dim")
	if missing != nil {
		return c, missing
	}
	if eps, ok := g.Float("clip.vision.attention.layer_norm_epsilon"); !ok || math.Abs(eps-qwen3LNEps) > 1e-9 {
		return c, fmt.Errorf("layer-norm epsilon %g (present %v): the tower runs %g", eps, ok, qwen3LNEps)
	}
	if ds, ok := g.Metadata["clip.vision.is_deepstack_layers"].([]any); ok {
		for i, v := range ds {
			if b, _ := v.(bool); b {
				return c, fmt.Errorf("block %d is a DeepStack layer (Qwen3-VL); this tower has none", i)
			}
		}
	}
	if !g.Has("v.patch_embd.weight") || !g.Has("v.patch_embd.weight.1") {
		return c, fmt.Errorf("the patch embedding is not split in two (v.patch_embd.weight and .weight.1): not the qwen3vl_merger layout")
	}
	dims, ok := g.Dims("v.position_embd.weight")
	if !ok || len(dims) != 2 {
		return c, fmt.Errorf("no 2-D v.position_embd.weight")
	}
	c.NumPositionEmbeddings = dims[1]
	c.TemporalPatchSize = 2
	c.InChannels = 3
	c.HiddenAct = qwen3ActBlockMLP
	c.DeepstackVisualIndexes = nil
	return c, nil
}

// mmprojSource serves HF tower names from an mmproj's llama.cpp names.
type mmprojSource struct{ g *embed.GGUFFile }

// qwen3MMProjName maps an HF tower tensor name to its llama.cpp mmproj name.
func qwen3MMProjName(name string) (string, bool) {
	switch name {
	case "patch_embed.proj.bias":
		return "v.patch_embd.bias", true
	case "pos_embed.weight":
		return "v.position_embd.weight", true
	case "merger.norm.weight":
		return "v.post_ln.weight", true
	case "merger.norm.bias":
		return "v.post_ln.bias", true
	case "merger.linear_fc1.weight":
		return "mm.0.weight", true
	case "merger.linear_fc1.bias":
		return "mm.0.bias", true
	case "merger.linear_fc2.weight":
		return "mm.2.weight", true
	case "merger.linear_fc2.bias":
		return "mm.2.bias", true
	}
	rest, ok := strings.CutPrefix(name, "blocks.")
	if !ok {
		return "", false
	}
	idx, sub, ok := strings.Cut(rest, ".")
	if _, err := strconv.Atoi(idx); !ok || err != nil {
		return "", false
	}
	m := map[string]string{
		"norm1.weight": "ln1.weight", "norm1.bias": "ln1.bias", "norm2.weight": "ln2.weight", "norm2.bias": "ln2.bias",
		"attn.qkv.weight": "attn_qkv.weight", "attn.qkv.bias": "attn_qkv.bias",
		"attn.proj.weight": "attn_out.weight", "attn.proj.bias": "attn_out.bias",
		"mlp.linear_fc1.weight": "ffn_up.weight", "mlp.linear_fc1.bias": "ffn_up.bias",
		"mlp.linear_fc2.weight": "ffn_down.weight", "mlp.linear_fc2.bias": "ffn_down.bias",
	}
	g, ok := m[sub]
	if !ok {
		return "", false
	}
	return "v.blk." + idx + "." + g, true
}

// tensor reads a GGUF tensor whose HF shape (row-major, outermost first) is want: ggml's dims are the same reversed.
func (s mmprojSource) tensor(gname string, want []int) ([]float32, error) {
	dims, data, err := s.g.Tensor(gname)
	if err != nil {
		return nil, err
	}
	if len(want) > 0 {
		if len(dims) != len(want) {
			return nil, fmt.Errorf("%s: dims %v, want the reverse of %v", gname, dims, want)
		}
		for i := range want {
			if dims[len(dims)-1-i] != want[i] {
				return nil, fmt.Errorf("%s: dims %v, want the reverse of %v", gname, dims, want)
			}
		}
	}
	return data, nil
}

func (s mmprojSource) TensorF32(name string, want ...int) ([]float32, error) {
	if name == "patch_embed.proj.weight" {
		// HF [hidden, C, T=2, P, P]; each mmproj half is [hidden, C, P, P] (ggml [P, P, C, hidden]).
		if len(want) != 5 || want[2] != 2 {
			return nil, fmt.Errorf("patch_embed.proj.weight: want %v, need [hidden, C, 2, P, P]", want)
		}
		H, C, P := want[0], want[1], want[3]
		half := []int{H, C, P, P}
		w0, err := s.tensor("v.patch_embd.weight", half)
		if err != nil {
			return nil, err
		}
		w1, err := s.tensor("v.patch_embd.weight.1", half)
		if err != nil {
			return nil, err
		}
		out := make([]float32, H*C*2*P*P)
		pp := P * P
		for o := range H {
			for c := range C {
				src := (o*C + c) * pp
				dst := ((o*C+c)*2 + 0) * pp
				copy(out[dst:dst+pp], w0[src:src+pp])
				copy(out[dst+pp:dst+2*pp], w1[src:src+pp])
			}
		}
		return out, nil
	}
	gname, ok := qwen3MMProjName(name)
	if !ok {
		return nil, fmt.Errorf("no mmproj tensor for %s", name)
	}
	return s.tensor(gname, want)
}
