//go:build cuda && realckpt

package cuda

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/audio"
	"github.com/townsendmerino/goinfer/internal/whisper"
	"github.com/townsendmerino/goinfer/multimodal"
)

// G-S14t4 on the real checkpoints (docs/tasks/task-multimodal-support-2026-10.md, S14.6 lever B): the device encoder against the CPU encoder, whole-window, every position, on Whisper-small
// (80 mels, 12 layers, d 768) and Voxtral Mini's tower (128 mels, 32 layers, d 1280, head dim 64: the fused attention kernel's shape). The features are a clip's own log-mel when
// GOINFER_SPEED_WAV is set, else seeded noise.

func realAudioEncoderDirs(t *testing.T) map[string]string {
	home, _ := os.UserHomeDir()
	dirs := map[string]string{"whisper-small": filepath.Join(home, "models", "whisper-small"), "voxtral-tower": filepath.Join(home, "models", "voxtral-mini-3b-2507")}
	for k, d := range dirs {
		if _, err := os.Stat(filepath.Join(d, "config.json")); err != nil {
			t.Logf("%s: no checkpoint at %s", k, d)
			delete(dirs, k)
		}
	}
	return dirs
}

func realAudioFeats(t *testing.T, mels int) []float32 {
	if wav := os.Getenv("GOINFER_SPEED_WAV"); wav != "" {
		raw, err := os.ReadFile(wav)
		if err != nil {
			t.Fatal(err)
		}
		samples, err := multimodal.DecodeWAVAnyRate(raw)
		if err != nil {
			t.Fatal(err)
		}
		f, _, err := audio.WhisperFeatures(samples[:min(len(samples), 30*audio.WhisperSampleRate)], mels)
		if err != nil {
			t.Fatal(err)
		}
		return f
	}
	return audioRandFeats(mels, 21)
}

func gpuMemUsedMiB() int {
	out, err := exec.Command("nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return -1
	}
	n, _ := strconv.Atoi(strings.TrimSpace(strings.Split(string(out), "\n")[0]))
	return n
}

func TestAudioEncoder_realMatchesCPU(t *testing.T) {
	for name, dir := range realAudioEncoderDirs(t) {
		t.Run(name, func(t *testing.T) {
			enc, err := audio.LoadWhisperEncoder(dir)
			if err != nil {
				t.Fatal(err)
			}
			feats := realAudioFeats(t, enc.Cfg.MelBins)
			t0 := time.Now()
			want, err := enc.Forward(feats)
			if err != nil {
				t.Fatal(err)
			}
			cpu := time.Since(t0)
			before := gpuMemUsedMiB()
			tw, err := newAudioTower(enc)
			if err != nil {
				t.Skipf("no CUDA device: %v", err)
			}
			defer tw.Close()
			t0 = time.Now()
			got, err := tw.Forward(feats)
			if err != nil {
				t.Fatal(err)
			}
			first := time.Since(t0)
			t0 = time.Now()
			if _, err = tw.Forward(feats); err != nil {
				t.Fatal(err)
			}
			second := time.Since(t0)
			c := audioMinCosine(got, want, enc.Cfg.DModel)
			t.Logf("%s: min per-position cosine %.9f; CPU %.2fs, CUDA first %.3fs then %.3fs; VRAM held %d MiB", name, c, cpu.Seconds(), first.Seconds(), second.Seconds(), gpuMemUsedMiB()-before)
			if c < 0.99999 {
				t.Fatalf("%s: min per-position cosine %.8f against the CPU encoder, want >= 0.99999", name, c)
			}
		})
	}
}

// TestAudioEncoderSpeed_record is G-S14t6's record: the encoder window, CPU against CUDA, GOINFER_SPEED_ROUNDS interleaved rounds (default 5) after one warm-up of each. A record, not a gate:
// it asserts nothing. Run it under the timing lock.
func TestAudioEncoderSpeed_record(t *testing.T) {
	if os.Getenv("GOINFER_SPEED_ROUNDS") == "" {
		t.Skip("set GOINFER_SPEED_ROUNDS (5 for the record)")
	}
	rounds, _ := strconv.Atoi(os.Getenv("GOINFER_SPEED_ROUNDS"))
	for name, dir := range realAudioEncoderDirs(t) {
		enc, err := audio.LoadWhisperEncoder(dir)
		if err != nil {
			t.Fatal(err)
		}
		feats := realAudioFeats(t, enc.Cfg.MelBins)
		before := gpuMemUsedMiB()
		tw, err := newAudioTower(enc)
		if err != nil {
			t.Skipf("no CUDA device: %v", err)
		}
		held := gpuMemUsedMiB() - before
		cpuOnce := func() float64 {
			enc.SetDevice(nil)
			t0 := time.Now()
			if _, err := enc.Forward(feats); err != nil {
				t.Fatal(err)
			}
			return time.Since(t0).Seconds()
		}
		gpuOnce := func() float64 {
			t0 := time.Now()
			if _, err := tw.Forward(feats); err != nil {
				t.Fatal(err)
			}
			return time.Since(t0).Seconds()
		}
		cpuOnce()
		gpuOnce()
		var cs, gs, ratio []float64
		for r := 0; r < rounds; r++ {
			c, g := cpuOnce(), gpuOnce()
			cs, gs, ratio = append(cs, c), append(gs, g), append(ratio, c/g)
			t.Logf("SPEED %s round %d cpu_s=%.3f cuda_s=%.3f ratio=%.2f", name, r+1, c, g, c/g)
		}
		med := func(v []float64) float64 { s := append([]float64(nil), v...); sort.Float64s(s); return s[len(s)/2] }
		t.Logf("SPEED %s median cpu_s=%.3f cuda_s=%.3f ratio_of_medians=%.2f median_pair_ratio=%.2f vram_mib=%d", name, med(cs), med(gs), med(cs)/med(gs), med(ratio), held)
		_ = tw.Close()
	}
}

// TestWhisperSmall_clipsOnDevice is G-S14t5's goinfer arm: internal/whisper's TestWhisperSmall_clips with the encoder on the CUDA device, writing {id: {text, lang}} for scripts/s14f_grade.py.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_S14F_DIR=<wav dir> GOINFER_S14F_OUT=<json> go test -tags 'cuda realckpt' . -run TestWhisperSmall_clipsOnDevice -v -timeout 30m
func TestWhisperSmall_clipsOnDevice(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1")
	}
	data, out := os.Getenv("GOINFER_S14F_DIR"), os.Getenv("GOINFER_S14F_OUT")
	if data == "" || out == "" {
		t.Skip("set GOINFER_S14F_DIR and GOINFER_S14F_OUT")
	}
	home, _ := os.UserHomeDir()
	tr, err := whisper.LoadTranscriber(filepath.Join(home, "models", "whisper-small"))
	if err != nil {
		t.Fatal(err)
	}
	dev, err := multimodal.UseWhisperDevice("cuda", tr.Enc)
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	defer dev.Close()
	raw, err := os.ReadFile(filepath.Join(data, "refs.json"))
	if err != nil {
		t.Fatal(err)
	}
	var refs map[string]string
	if err := json.Unmarshal(raw, &refs); err != nil {
		t.Fatal(err)
	}
	ids := make([]string, 0, len(refs))
	for id := range refs {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	type row struct {
		Text string `json:"text"`
		Lang string `json:"lang"`
	}
	res := map[string]row{}
	t0 := time.Now()
	for i, id := range ids {
		wav, err := os.ReadFile(filepath.Join(data, id+".wav"))
		if err != nil {
			t.Fatal(err)
		}
		samples, err := multimodal.DecodeWAVAnyRate(wav)
		if err != nil {
			t.Fatal(err)
		}
		text, r, err := tr.Transcribe(samples, "", "transcribe")
		if err != nil {
			t.Fatalf("%s: %v", id, err)
		}
		res[id] = row{text, r.Language}
		fmt.Fprintf(os.Stderr, "[s14t5 %5.0fs] %d/%d %s: %s %q\n", time.Since(t0).Seconds(), i+1, len(ids), id, r.Language, text)
	}
	b, _ := json.MarshalIndent(res, "", " ")
	if err := os.WriteFile(out, b, 0o644); err != nil {
		t.Fatal(err)
	}
	t.Logf("%d clips in %.0f s, wrote %s", len(ids), time.Since(t0).Seconds(), out)
}
