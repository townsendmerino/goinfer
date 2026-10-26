package audio

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"testing"
	"weak"

	"github.com/townsendmerino/aikit/embed"
)

// stTensor is one F32 tensor of a test checkpoint.
type stTensor struct {
	shape []int
	data  []float32
}

// writeF32Checkpoint writes config and tensors as dir/config.json and dir/model.safetensors. The header is padded to 8 bytes so every F32 payload is aligned, which makes
// TensorF32 hand out views into the mapping (the case the lifetime test is about) rather than copies.
func writeF32Checkpoint(t *testing.T, dir string, config any, tensors map[string]stTensor) {
	t.Helper()
	cfg, err := json.Marshal(config)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), cfg, 0o644); err != nil {
		t.Fatal(err)
	}
	names := make([]string, 0, len(tensors))
	for n := range tensors {
		names = append(names, n)
	}
	sort.Strings(names)
	header := map[string]any{}
	var payload []byte
	for _, n := range names {
		tt := tensors[n]
		raw := make([]byte, 4*len(tt.data))
		for i, v := range tt.data {
			binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(v))
		}
		header[n] = map[string]any{"dtype": "F32", "shape": tt.shape, "data_offsets": []int{len(payload), len(payload) + len(raw)}}
		payload = append(payload, raw...)
	}
	hb, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	for len(hb)%8 != 0 {
		hb = append(hb, ' ')
	}
	out := binary.LittleEndian.AppendUint64(nil, uint64(len(hb)))
	out = append(append(out, hb...), payload...)
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), out, 0o644); err != nil {
		t.Fatal(err)
	}
}

// tensorSet fills tensors with small seeded values.
type tensorSet struct {
	m   map[string]stTensor
	rng *rand.Rand
}

func newTensorSet() *tensorSet {
	return &tensorSet{m: map[string]stTensor{}, rng: rand.New(rand.NewSource(3))}
}

func (s *tensorSet) add(name string, shape ...int) {
	n := 1
	for _, d := range shape {
		n *= d
	}
	data := make([]float32, n)
	for i := range data {
		data[i] = float32(s.rng.NormFloat64()) * 0.2
	}
	s.m[name] = stTensor{shape: shape, data: data}
}

// whisperTensors adds a one-layer Whisper encoder under prefix.
func whisperTensors(s *tensorSet, prefix string, D, mels, ffn int) {
	s.add(prefix+"conv1.weight", D, mels, 3)
	s.add(prefix+"conv1.bias", D)
	s.add(prefix+"conv2.weight", D, D, 3)
	s.add(prefix+"conv2.bias", D)
	s.add(prefix+"embed_positions.weight", WhisperFrames/2, D)
	p := prefix + "layers.0."
	for _, n := range []string{"self_attn_layer_norm.weight", "self_attn_layer_norm.bias", "final_layer_norm.weight", "final_layer_norm.bias",
		"self_attn.q_proj.bias", "self_attn.v_proj.bias", "self_attn.out_proj.bias", "fc2.bias"} {
		s.add(p+n, D)
	}
	for _, n := range []string{"self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.out_proj.weight"} {
		s.add(p+n, D, D)
	}
	s.add(p+"fc1.weight", ffn, D)
	s.add(p+"fc1.bias", ffn)
	s.add(p+"fc2.weight", D, ffn)
	s.add(prefix+"layer_norm.weight", D)
	s.add(prefix+"layer_norm.bias", D)
}

// featsOf is n seeded feature values.
func featsOf(n int) []float32 {
	rng := rand.New(rand.NewSource(5))
	f := make([]float32, n)
	for i := range f {
		f[i] = float32(rng.NormFloat64())
	}
	return f
}

// lifetimeCase is one audio loader under the lifetime test: load returns a forward over fixed input and the loader's Close.
type lifetimeCase struct {
	name  string
	write func(t *testing.T, dir string)
	load  func(dir string) (run func() ([]float32, error), closeFn func() error, err error)
}

func audioLifetimeCases() []lifetimeCase {
	const D, ffn, mels = 4, 8, 80
	whisperCfg := map[string]any{"d_model": D, "encoder_layers": 1, "encoder_attention_heads": 2, "encoder_ffn_dim": ffn, "num_mel_bins": mels, "max_source_positions": WhisperFrames / 2}
	return []lifetimeCase{
		{
			name: "whisper",
			write: func(t *testing.T, dir string) {
				s := newTensorSet()
				whisperTensors(s, "", D, mels, ffn)
				writeF32Checkpoint(t, dir, whisperCfg, s.m)
			},
			load: func(dir string) (func() ([]float32, error), func() error, error) {
				e, err := LoadWhisperEncoder(dir)
				if err != nil {
					return nil, nil, err
				}
				feats := featsOf(mels * WhisperFrames)
				return func() ([]float32, error) { return e.Forward(feats) }, e.Close, nil
			},
		},
		{
			name: "voxtral",
			write: func(t *testing.T, dir string) {
				const inter, H = 4 * D, 3
				audio := map[string]any{"intermediate_size": inter}
				for k, v := range whisperCfg {
					audio[k] = v
				}
				s := newTensorSet()
				whisperTensors(s, "audio_tower.", D, mels, ffn)
				s.add("multi_modal_projector.linear_1.weight", H, inter)
				s.add("multi_modal_projector.linear_2.weight", H, H)
				writeF32Checkpoint(t, dir, map[string]any{"audio_config": audio, "text_config": map[string]any{"hidden_size": H}, "projector_hidden_act": "gelu"}, s.m)
			},
			load: func(dir string) (func() ([]float32, error), func() error, error) {
				v, err := LoadVoxtralAudio(dir)
				if err != nil {
					return nil, nil, err
				}
				feats := featsOf(mels * WhisperFrames)
				return func() ([]float32, error) { return v.Embed([][]float32{feats}) }, v.Close, nil
			},
		},
		{
			name: "qwen3-asr",
			write: func(t *testing.T, dir string) {
				const H, out = 2, 3
				hf := (((QwenASRMels+1)/2+1)/2 + 1) / 2
				s := newTensorSet()
				for i, cin := range []int{1, H, H} {
					s.add(fmt.Sprintf("conv2d%d.weight", i+1), H, cin, 3, 3)
					s.add(fmt.Sprintf("conv2d%d.bias", i+1), H)
				}
				s.add("conv_out.weight", D, H*hf)
				p := "layers.0."
				for _, n := range []string{"self_attn_layer_norm.weight", "self_attn_layer_norm.bias", "final_layer_norm.weight", "final_layer_norm.bias",
					"self_attn.q_proj.bias", "self_attn.k_proj.bias", "self_attn.v_proj.bias", "self_attn.out_proj.bias", "fc2.bias"} {
					s.add(p+n, D)
				}
				for _, n := range []string{"self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.out_proj.weight"} {
					s.add(p+n, D, D)
				}
				s.add(p+"fc1.weight", ffn, D)
				s.add(p+"fc1.bias", ffn)
				s.add(p+"fc2.weight", D, ffn)
				s.add("ln_post.weight", D)
				s.add("ln_post.bias", D)
				s.add("proj1.weight", D, D)
				s.add("proj1.bias", D)
				s.add("proj2.weight", out, D)
				s.add("proj2.bias", out)
				cfg := map[string]any{"d_model": D, "encoder_layers": 1, "encoder_attention_heads": 2, "encoder_ffn_dim": ffn, "downsample_hidden_size": H,
					"num_mel_bins": QwenASRMels, "n_window": 50, "n_window_infer": 100, "output_dim": out}
				writeF32Checkpoint(t, dir, map[string]any{"audio_config": cfg}, s.m)
			},
			load: func(dir string) (func() ([]float32, error), func() error, error) {
				e, err := LoadQwenASREncoder(dir)
				if err != nil {
					return nil, nil, err
				}
				feats := featsOf(QwenASRMels * QwenASRChunkFrames)
				return func() ([]float32, error) {
					out, _, err := e.Forward(feats, QwenASRChunkFrames, QwenASRChunkFrames)
					return out, err
				}, e.Close, nil
			},
		},
	}
}

// MEM3 (goinfer task-memory-campaign-2026-10): an F32 checkpoint's weights are views into the file's mapping, and the file's finalizer unmaps it once the file is unreachable. Each loader
// must keep the file it opened reachable for as long as the encoder lives. The file is watched through a weak pointer, so a loader that drops it fails here, after a few GC cycles,
// instead of reading unmapped memory later. Then Close must release it: the forward errors, and a second Close is a no-op.
//
// Since v1.71.0 nothing unmaps a dropped file at all (embed.SafetensorsFile.Close is the only release), so a loader that dropped its file would leak its mapping rather than
// crash, and this test cannot go red on its own: it pins the ownership the encoders' Close relies on, and guards against an automatic release ever coming back. Measured
// 2026-10-17, with the finalizer the tree had then made to run: a loader that drops its file fails here (all three loaders, 3/3 rounds), and without the pre-forward
// check the forward over the unmapped weights is a SIGSEGV.
func TestAudioLoaders_holdTheirCheckpointUntilClose(t *testing.T) {
	for _, c := range audioLifetimeCases() {
		t.Run(c.name, func(t *testing.T) {
			dir := t.TempDir()
			c.write(t, dir)
			var opened weak.Pointer[embed.SafetensorsFile]
			testHookOpenedCheckpoint = func(st *embed.SafetensorsFile) { opened = weak.Make(st) }
			t.Cleanup(func() { testHookOpenedCheckpoint = nil })

			run, closeFn, err := c.load(dir)
			if err != nil {
				t.Fatal(err)
			}
			if opened.Value() == nil {
				t.Fatal("the loader did not open its checkpoint through openSafetensorsDir; the test watches nothing")
			}
			// Check before any forward: a forward over unmapped weights is a SIGSEGV that takes the whole test binary down, not a failure.
			collectAndCheck := func() {
				for range 4 {
					runtime.GC()
				}
				if opened.Value() == nil {
					t.Fatal("the checkpoint file became unreachable while the encoder is alive: its finalizer unmaps the F32 weights the encoder still reads")
				}
			}
			collectAndCheck()
			want, err := run()
			if err != nil {
				t.Fatal(err)
			}
			collectAndCheck()
			got, err := run()
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != len(want) {
				t.Fatalf("forward after GC: %d values, before %d", len(got), len(want))
			}
			for i := range got {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("forward after GC differs at %d: %v, before %v", i, got[i], want[i])
				}
			}

			if err := closeFn(); err != nil {
				t.Fatal(err)
			}
			if _, err := run(); !errors.Is(err, errEncoderClosed) {
				t.Fatalf("forward after Close: err %v, want errEncoderClosed", err)
			}
			if err := closeFn(); err != nil {
				t.Fatalf("second Close: %v", err)
			}
		})
	}
}

// The fixture must exercise the aliasing case: two F32 reads of one tensor share their memory (a copy would not), so the weights really are views into the mapping.
func TestWriteF32Checkpoint_tensorsAliasTheMapping(t *testing.T) {
	dir := t.TempDir()
	s := newTensorSet()
	s.add("a.weight", 3, 5)
	s.add("b.weight", 7)
	writeF32Checkpoint(t, dir, map[string]any{}, s.m)
	st, err := openSafetensorsDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	for _, n := range []string{"a.weight", "b.weight"} {
		x, err := st.TensorF32(n)
		if err != nil {
			t.Fatal(err)
		}
		y, err := st.TensorF32(n)
		if err != nil {
			t.Fatal(err)
		}
		if &x[0] != &y[0] {
			t.Fatalf("%s: two reads returned different memory, so the fixture's F32 tensors are copies, not views into the mapping", n)
		}
	}
}
