package audio

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/townsendmerino/aikit/linalg"
)

// Voxtral's audio path, float32 on the CPU (transformers 5.15.0 VoxtralModel.get_audio_features): the Whisper encoder (WhisperEncoder: Voxtral's tower is the same module), whose
// [1500][d] output is reshaped to rows of audio_config.intermediate_size = stack x d (four consecutive frames, so 375 rows per 30 s window), then
// multi_modal_projector.linear_1 (no bias), exact GELU, linear_2 (no bias) into the text hidden size. The rows replace the embeddings at the [AUDIO] placeholders.

// Planted defects for G-S14e2's red runs. Zero is the shipped computation.
const (
	voxtralDefectNone = iota
	voxtralDefectNoStack
	voxtralDefectChunksReversed
	voxtralDefectNoGELU
	voxtralDefectTanhGELU
	voxtralDefectBias
	voxtralDefectNoSecondLinear
)

// VoxtralDefectCountForTest is the number of planted defects SetDefectForTest accepts (1..N).
const VoxtralDefectCountForTest = voxtralDefectNoSecondLinear

// VoxtralAudio is the loaded audio tower and projector.
type VoxtralAudio struct {
	Enc        *WhisperEncoder
	TextHidden int // the projector's output width (the text decoder's hidden size)
	Stack      int // encoder frames per projector row
	w1, w2     []float32
	defect     int
}

// SetDefectForTest plants one defect (0 clears it); it also reaches the encoder's own defects through Enc. Never call it in production.
func (v *VoxtralAudio) SetDefectForTest(d int) { v.defect = d }

// LoadVoxtralAudio reads the audio tower and projector of a Voxtral checkpoint directory (config.json with audio_config and text_config; tensors under audio_tower.* and
// multi_modal_projector.*, in model.safetensors or a shard index).
func LoadVoxtralAudio(dir string) (*VoxtralAudio, error) {
	enc, err := LoadWhisperEncoder(dir)
	if err != nil {
		return nil, err
	}
	loaded := false
	defer func() {
		if !loaded {
			_ = enc.Close() //nolint:errcheck // a failed load: the load error is the one to report
		}
	}()
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var cfg struct {
		Text struct {
			Hidden int `json:"hidden_size"`
		} `json:"text_config"`
		Audio struct {
			Intermediate int `json:"intermediate_size"`
		} `json:"audio_config"`
		Act string `json:"projector_hidden_act"`
	}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("audio: %s/config.json: %w", dir, err)
	}
	d, inter, H := enc.Cfg.DModel, cfg.Audio.Intermediate, cfg.Text.Hidden
	switch {
	case cfg.Act != "gelu":
		return nil, fmt.Errorf("audio: Voxtral projector_hidden_act %q (only exact gelu is built)", cfg.Act)
	case inter <= 0 || H <= 0 || inter%d != 0 || enc.Cfg.MaxPositions%(inter/d) != 0:
		return nil, fmt.Errorf("audio: Voxtral audio intermediate_size %d (d_model %d, %d frames) / text hidden %d do not stack into whole rows", inter, d, enc.Cfg.MaxPositions, H)
	}
	// The projector is in the encoder's checkpoint: read it through the file the encoder holds, which keeps an F32 projector's views mapped too.
	v := &VoxtralAudio{Enc: enc, TextHidden: H, Stack: inter / d}
	if v.w1, err = enc.st.TensorF32("multi_modal_projector.linear_1.weight", H, inter); err == nil {
		v.w2, err = enc.st.TensorF32("multi_modal_projector.linear_2.weight", H, H)
	}
	if err != nil {
		return nil, fmt.Errorf("audio: loading %s: %w", dir, err)
	}
	loaded = true
	return v, nil
}

// Close releases the checkpoint mapping the tower and projector weights may alias (it closes Enc); Embed returns an error afterwards. Calling Close again does nothing.
func (v *VoxtralAudio) Close() error {
	v.w1, v.w2 = nil, nil
	return v.Enc.Close()
}

// Rows is the number of audio tokens one 30 s window yields.
func (v *VoxtralAudio) Rows() int { return v.Enc.Cfg.MaxPositions / v.Stack }

func (v *VoxtralAudio) gelu(x float32) float32 {
	switch v.defect {
	case voxtralDefectNoGELU:
		return x
	case voxtralDefectTanhGELU:
		f := float64(x)
		return float32(0.5 * f * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(f+0.044715*f*f*f))))
	}
	return float32(0.5 * float64(x) * (1 + math.Erf(float64(x)/math.Sqrt2)))
}

// Embed runs each 30 s window's features ([mels][3000], WhisperFeatures' layout, one entry per window, in time order) through the tower and projector and returns the audio tokens'
// embeddings, window after window: [len(chunks) x Rows()][TextHidden], flattened.
func (v *VoxtralAudio) Embed(chunks [][]float32) ([]float32, error) {
	rows, _, err := v.embed(chunks, false)
	return rows, err
}

// EmbedWithTower is Embed that also returns the tower's own output, [len(chunks) x 1500][d_model] flattened, for the gate that compares the encoder and the projector separately.
func (v *VoxtralAudio) EmbedWithTower(chunks [][]float32) (rows, tower []float32, err error) {
	return v.embed(chunks, true)
}

func (v *VoxtralAudio) embed(chunks [][]float32, keepTower bool) ([]float32, []float32, error) {
	if len(chunks) == 0 {
		return nil, nil, fmt.Errorf("audio: no audio windows")
	}
	var tower []float32
	d, inter, H, rows := v.Enc.Cfg.DModel, v.Enc.Cfg.DModel*v.Stack, v.TextHidden, v.Rows()
	out := make([]float32, 0, len(chunks)*rows*H)
	for ci, f := range chunks {
		hid, err := v.Enc.Forward(f) // [1500][d]
		if err != nil {
			return nil, nil, fmt.Errorf("audio: window %d: %w", ci, err)
		}
		if keepTower {
			tower = append(tower, hid...)
		}
		x := hid // [1500][d] is [rows][stack*d] row-major: four consecutive frames per row, which is the reshape
		if v.defect == voxtralDefectNoStack {
			x = make([]float32, rows*inter) // only the first frame of each group, the rest zero
			for r := range rows {
				copy(x[r*inter:r*inter+d], hid[r*v.Stack*d:r*v.Stack*d+d])
			}
		}
		y := make([]float32, rows*H)
		linalg.MatmulBT(x, v.w1, y, rows, inter, H)
		for i := range y {
			if v.defect == voxtralDefectBias {
				y[i] += 0.1
			}
			y[i] = v.gelu(y[i])
		}
		if v.defect == voxtralDefectNoSecondLinear {
			out = append(out, y...)
			continue
		}
		z := make([]float32, rows*H)
		linalg.MatmulBT(y, v.w2, z, rows, H, H)
		out = append(out, z...)
	}
	if v.defect == voxtralDefectChunksReversed && len(chunks) > 1 {
		rev := make([]float32, 0, len(out))
		for c := len(chunks) - 1; c >= 0; c-- {
			rev = append(rev, out[c*rows*H:(c+1)*rows*H]...)
		}
		out = rev
	}
	return out, tower, nil
}
