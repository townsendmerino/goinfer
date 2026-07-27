package multimodal

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/townsendmerino/aikit/linalg"
)

// Projector is the Gemma 3 multimodal connector (Gemma3MultiModalProjector): it
// turns the vision encoder's last_hidden_state into the fixed block of image-token
// embeddings the text decoder interleaves at the <image> placeholders. Op order
// matches HF exactly: reshape patches → grid, 2D average-pool to mm_tokens_per_image,
// Gemma3RMSNorm (the 1+weight convention, at the vision layer-norm eps), then the
// [vision_hidden → text_hidden] projection. f32; parity is cosine vs the HF golden.
type Projector struct {
	visionHidden, textHidden int
	patchesPerImage          int // vision grid side = image_size / patch_size
	tokensPerSide            int // sqrt(mm_tokens_per_image)
	eps                      float64
	normW                    []float32 // mm_soft_emb_norm.weight [visionHidden]
	projWt                   []float32 // mm_input_projection_weight TRANSPOSED [textHidden, visionHidden]
}

// vlConfig is the slice of a Gemma3Config (VL) the projector needs.
type vlConfig struct {
	MMTokensPerImage int `json:"mm_tokens_per_image"`
	TextConfig       struct {
		HiddenSize int `json:"hidden_size"`
	} `json:"text_config"`
	VisionConfig struct {
		HiddenSize   int     `json:"hidden_size"`
		ImageSize    int     `json:"image_size"`
		PatchSize    int     `json:"patch_size"`
		LayerNormEps float64 `json:"layer_norm_eps"`
	} `json:"vision_config"`
}

// LoadProjector reads the projector from a Gemma 3 VL checkpoint (config.json +
// model.safetensors). Weights are copied out so the file is closed before return.
func LoadProjector(dir string) (*Projector, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("multimodal: read VL config: %w", err)
	}
	var c vlConfig
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, fmt.Errorf("multimodal: parse VL config: %w", err)
	}
	eps := c.VisionConfig.LayerNormEps
	if eps == 0 {
		eps = 1e-6
	}
	// A hostile/truncated VL config would otherwise divide by zero (patch_size) or
	// build a nonsense grid; validate before use (the tokenizer loaders do the same).
	if c.VisionConfig.PatchSize <= 0 || c.VisionConfig.ImageSize <= 0 || c.MMTokensPerImage <= 0 {
		return nil, fmt.Errorf("multimodal: invalid VL config (image_size=%d patch_size=%d mm_tokens_per_image=%d)",
			c.VisionConfig.ImageSize, c.VisionConfig.PatchSize, c.MMTokensPerImage)
	}
	p := &Projector{
		visionHidden:    c.VisionConfig.HiddenSize,
		textHidden:      c.TextConfig.HiddenSize,
		patchesPerImage: c.VisionConfig.ImageSize / c.VisionConfig.PatchSize,
		tokensPerSide:   int(math.Round(math.Sqrt(float64(c.MMTokensPerImage)))),
		eps:             eps,
	}
	st, err := openWeights(dir) // single model.safetensors (tiny) or the sharded set (real VL)
	if err != nil {
		return nil, fmt.Errorf("multimodal: open VL safetensors: %w", err)
	}
	defer st.Close()
	normW, e1 := st.TensorF32("multi_modal_projector.mm_soft_emb_norm.weight")
	projW, e2 := st.TensorF32("multi_modal_projector.mm_input_projection_weight") // [visionHidden, textHidden]
	if e1 != nil || e2 != nil {
		return nil, fmt.Errorf("multimodal: projector weights: %v %v", e1, e2)
	}
	p.normW = append([]float32(nil), normW...)
	// Transpose [visionHidden, textHidden] → [textHidden, visionHidden] for MatmulBT.
	vh, th := p.visionHidden, p.textHidden
	if vh <= 0 || th <= 0 || len(projW) != vh*th { // guard the transpose indexing against a truncated tensor
		return nil, fmt.Errorf("multimodal: projection weight length %d, want %d (vision_hidden %d × text_hidden %d)", len(projW), vh*th, vh, th)
	}
	p.projWt = make([]float32, th*vh)
	for i := range vh {
		for o := range th {
			p.projWt[o*vh+i] = projW[i*th+o]
		}
	}
	return p, nil
}

// MMTokens is the number of image-token embeddings the projector emits per image.
func (p *Projector) MMTokens() int { return p.tokensPerSide * p.tokensPerSide }

// Forward maps vision last_hidden_state [numPatches*visionHidden] to the image
// embedding block [MMTokens*textHidden].
func (p *Projector) Forward(visionHidden []float32) ([]float32, error) {
	vh, grid, tps := p.visionHidden, p.patchesPerImage, p.tokensPerSide
	np := grid * grid
	if len(visionHidden) != np*vh {
		return nil, fmt.Errorf("multimodal: projector input len %d, want %d (%d patches × %d)", len(visionHidden), np*vh, np, vh)
	}
	kernel := grid / tps
	mm := tps * tps

	// 2D average pool the grid → tps×tps (patch (ph,pw) at seq ph*grid+pw).
	pooled := make([]float32, mm*vh)
	for oh := range tps {
		for ow := range tps {
			row := pooled[(oh*tps+ow)*vh:]
			for i := range vh {
				var sum float64
				for ph := oh * kernel; ph < (oh+1)*kernel; ph++ {
					for pw := ow * kernel; pw < (ow+1)*kernel; pw++ {
						sum += float64(visionHidden[(ph*grid+pw)*vh+i])
					}
				}
				row[i] = float32(sum / float64(kernel*kernel))
			}
		}
	}
	// Gemma3RMSNorm: x · rsqrt(mean(x²)+eps) · (1 + weight).
	for t := range mm {
		row := pooled[t*vh : t*vh+vh]
		var ms float64
		for _, v := range row {
			ms += float64(v) * float64(v)
		}
		inv := 1.0 / math.Sqrt(ms/float64(vh)+p.eps)
		for i := range vh {
			row[i] = float32(float64(row[i])*inv) * (1.0 + p.normW[i])
		}
	}
	// Project to text hidden: out[mm,textHidden] = pooled[mm,vh] · projWt[textHidden,vh]ᵀ.
	out := make([]float32, mm*p.textHidden)
	linalg.MatmulBT(pooled, p.projWt, out, mm, vh, p.textHidden)
	return out, nil
}
