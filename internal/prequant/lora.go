package prequant

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/townsendmerino/goinfer/decoder"
)

// A .giw built with a PEFT adapter merged in (TranscodeLoRA, D3's "transcode --lora" in
// docs/tasks/task-constrained-confidence.md) carries a sidecar, <bundle>.lora.json, naming the adapter and the sha256
// of its weights. The whole-model f32 merge is then paid once, at transcode, instead of on every load, which on a 16 GB
// Mac is the anonymous-memory spike a 27B merge-at-load would be.

type loraRecord struct {
	Adapter string `json:"adapter"`        // the adapter dir, as given to the transcode
	SHA256  string `json:"adapter_sha256"` // of its adapter_model.safetensors
}

func loraSidecar(giwPath string) string { return giwPath + ".lora.json" }

func adapterSHA(adapterDir string) (string, error) {
	f, err := os.Open(filepath.Join(adapterDir, "adapter_model.safetensors"))
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func loraSidecarBytes(adapterDir string) ([]byte, error) {
	sum, err := adapterSHA(adapterDir)
	if err != nil {
		return nil, fmt.Errorf("lora %s: %w", adapterDir, err)
	}
	b, _ := json.MarshalIndent(loraRecord{Adapter: adapterDir, SHA256: sum}, "", "  ")
	return append(b, '\n'), nil
}

// TranscodeLoRA builds a .giw from a safetensors model directory with a PEFT adapter merged into it at load (D3), and
// records the adapter beside the bundle. A GGUF base is refused: an adapter names HF tensors, which a GGUF does not use.
func TranscodeLoRA(ctx context.Context, dir, adapter, out, quant string, embedInt4 bool, target decoder.GIWTarget) error {
	if fi, err := os.Stat(dir); err != nil || !fi.IsDir() {
		return fmt.Errorf("-lora needs a safetensors model directory as the input, not %s: a PEFT adapter names HF tensors", dir)
	}
	return transcodeDir(ctx, dir, adapter, out, quant, embedInt4, target)
}

// AdapterLoRA is the LoRA a load of modelPath must merge for an adapter a decision head brings: adapterDir itself for
// a safetensors or GGUF model (Load merges it, or refuses a GGUF), and "" for a .giw whose sidecar records this exact
// adapter (it is merged in already). A .giw without the sidecar, or with another adapter's, is refused: loading it
// would answer with a backbone the head was not trained on, and nothing downstream could tell.
func AdapterLoRA(adapterDir, modelPath string) (string, error) {
	if adapterDir == "" || !strings.HasSuffix(modelPath, ".giw") {
		return adapterDir, nil
	}
	raw, err := os.ReadFile(loraSidecar(modelPath))
	if err != nil {
		return "", fmt.Errorf("%s has no %s, so it does not carry the head's adapter (%s): transcode it with -lora, or serve the safetensors directory",
			modelPath, filepath.Base(loraSidecar(modelPath)), adapterDir)
	}
	var rec loraRecord
	if err := json.Unmarshal(raw, &rec); err != nil {
		return "", fmt.Errorf("%s: %w", loraSidecar(modelPath), err)
	}
	sum, err := adapterSHA(adapterDir)
	if err != nil {
		return "", fmt.Errorf("the head's adapter %s: %w", adapterDir, err)
	}
	if sum != rec.SHA256 {
		return "", fmt.Errorf("%s was built with another adapter (%s, sha256 %.12s…), not the head's %s (sha256 %.12s…)",
			modelPath, rec.Adapter, rec.SHA256, adapterDir, sum)
	}
	return "", nil
}
