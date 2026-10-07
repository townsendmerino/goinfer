//go:build darwin

package metal

// aikit's native-Metal vision towers (S3 of docs/tasks/task-multimodal-support-2026-10.md): importing them is the opt-in.
// visionmetal plugs a SigLIP tower (Gemma 3's) into vision.RegisterResident and qwenmetal a Qwen2.5-VL one into
// vision.RegisterQwenResident, so serve's EnableResident routes those towers' Forward to the GPU under --backend metal.
import (
	_ "github.com/townsendmerino/aikit/gpu/qwenmetal"
	_ "github.com/townsendmerino/aikit/gpu/visionmetal"
)
